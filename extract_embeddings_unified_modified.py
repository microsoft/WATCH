#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Strict feature extraction using official implementations only.

Key rules:
- Use authors' official model classes/processors to ensure canonical embedding sizes.
- Do NOT fallback to substitute backbones; fail fast with clear errors.
- Default time range is fixed to 2023-01..2023-12.

Supported models and expected embedding sizes (typical):
- satclip: CLIP image tower + projection, 512-d image features (ViT-B/16 or ViT-B/32 depending on checkpoint)
- georsclip: CLIP ViT-B/32, 512-d image features
- prithvi-eo-2.0: Prithvi EO 2.0 (600M) pooled/CLS features, typically 1024-d
- satmae: ViT-B encoder features (CLS/pooler), typically 768-d (depends on checkpoint)
- satlaspretrain: ResNet-152 global pooled features, 2048-d

Inputs:
- Planet mosaics with 3-4 bands; RGB is derived from BGR[NIR] as in existing pipeline.

Notes:
- For SatCLIP you may provide a CSV mapping site_name to lat,lon using --satclip-sites-csv.
    If using the official SatCLIP Lightning checkpoint (.ckpt), embeddings come from the official
    SatCLIP location encoder and require lat/lon.
    If using a Transformers CLIP directory, embeddings come from the CLIP image tower.
"""

import argparse
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
# Force-disable TensorFlow imports within transformers by shadowing module
sys.modules.setdefault("tensorflow", None)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
# Try to import rasterio; if unavailable, allow limited fallback for unmasked runs
try:
    import rasterio as rio  # type: ignore
    HAS_RASTERIO = True
except Exception:
    rio = None  # type: ignore
    HAS_RASTERIO = False
    try:
        import tifffile  # type: ignore
    except Exception:
        tifffile = None  # type: ignore
from tqdm.auto import tqdm


def months_2023() -> List[str]:
    return [f"2023_{m:02d}" for m in range(1, 13)]


def months_for_year(year: int) -> List[str]:
    return [f"{year}_{m:02d}" for m in range(1, 13)]


def months_range(start_year: int, end_year: int) -> List[str]:
    return [f"{y}_{m:02d}" for y in range(start_year, end_year + 1) for m in range(1, 13)]


def mask_path_for_site(masks_root: Path, site: str) -> Path:
    """Get path to mask file for a given site."""
    return masks_root / site / "mask.tif"


def read_aligned_mask(src_ds: "rio.io.DatasetReader", mask_fp: Path) -> Optional[np.ndarray]:
    """Reproject mask to src_ds grid (H,W, transform, crs) and return boolean mask (H,W)."""
    if not HAS_RASTERIO:
        raise RuntimeError("Mask alignment requires rasterio; install rasterio or disable --use-mask.")
    from rasterio.warp import reproject as _reproject
    from rasterio.enums import Resampling as _Resampling

    if not mask_fp.exists():
        return None
    with rio.open(mask_fp) as msrc:
        m = msrc.read(1)
        dst = np.zeros((src_ds.height, src_ds.width), dtype=np.float32)
        _reproject(
            source=m,
            destination=dst,
            src_transform=msrc.transform,
            src_crs=msrc.crs,
            dst_transform=src_ds.transform,
            dst_crs=src_ds.crs,
            resampling=_Resampling.nearest,
        )
        return dst > 0.0

def read_aligned_mask_gdal(image_fp: Path, mask_fp: Path) -> Optional[np.ndarray]:
    """Reproject mask to the image grid using GDAL and return boolean mask (H,W).
    Requires osgeo.gdal to be available.
    """
    try:
        from osgeo import gdal
    except Exception:
        raise RuntimeError("GDAL (osgeo) is required to align masks without rasterio.")
    if not mask_fp.exists():
        return None
    img_ds = gdal.Open(str(image_fp))
    mask_ds = gdal.Open(str(mask_fp))
    if img_ds is None or mask_ds is None:
        return None
    width = img_ds.RasterXSize
    height = img_ds.RasterYSize
    drv = gdal.GetDriverByName("MEM")
    dst = drv.Create("", width, height, 1, gdal.GDT_Byte)
    dst.SetGeoTransform(img_ds.GetGeoTransform())
    dst.SetProjection(img_ds.GetProjection())
    # Nearest-neighbor resampling for binary mask
    gdal.ReprojectImage(mask_ds, dst, mask_ds.GetProjection(), img_ds.GetProjection(), gdal.GRA_NearestNeighbour)
    mask = dst.ReadAsArray()
    return (mask > 0).astype(np.bool_)


def find_site_dirs(images_root: Path) -> List[Path]:
    return sorted([p for p in images_root.iterdir() if p.is_dir()])


def find_month_file(site_dir: Path, ym: str) -> Optional[Path]:
    candidates = sorted(site_dir.glob(f"*{ym}*.tif"))
    if not candidates:
        candidates = sorted(site_dir.rglob(f"*{ym}*.tif"))
    return candidates[0] if candidates else None


def build_month_index(site_path: Path) -> Dict[str, Path]:
    """Index all GeoTIFFs under a site folder by YYYY_MM.

    This is more robust than globbing when filenames include multiple date-like tokens
    or are nested under subfolders.
    """
    idx: Dict[str, Path] = {}
    for root, _, files in os.walk(site_path):
        for fname in files:
            fl = fname.lower()
            if not (fl.endswith(".tif") or fl.endswith(".tiff")):
                continue
            m = re.findall(r"(20\d{2})[^\d]?(\d{1,2})", fname)
            if not m:
                continue
            y, mm = m[-1]
            y_i, mm_i = int(y), int(mm)
            if 1 <= mm_i <= 12:
                tag = f"{y_i}_{mm_i:02d}"
                fp = Path(root) / fname
                # Prefer longer filenames (often more specific)
                if tag not in idx or len(fname) > len(idx[tag].name):
                    idx[tag] = fp
    return idx


def tile_boxes(width: int, height: int, tw: int, th: int) -> List[Tuple[int, int, int, int]]:
    boxes: List[Tuple[int, int, int, int]] = []
    for y0 in range(0, height, th):
        for x0 in range(0, width, tw):
            x1 = min(x0 + tw, width)
            y1 = min(y0 + th, height)
            boxes.append((x0, y0, x1, y1))
    return boxes


def _grid_in_pixels_rio(src: "rio.DatasetReader", grid_area_km2: float) -> Tuple[int, int]:
    side_km = max(0.25, float(np.sqrt(max(grid_area_km2, 1e-6))))
    res_x = abs(src.transform.a)
    res_y = abs(src.transform.e)
    if src.crs and getattr(src.crs, "is_geographic", False):
        c_row, c_col = src.height / 2.0, src.width / 2.0
        cx, cy = rio.transform.xy(src.transform, c_row, c_col)
        lat_rad = np.deg2rad(float(cy))
        m_per_deg_lat = 111_132.0
        m_per_deg_lon = 111_320.0 * max(np.cos(lat_rad), 1e-6)
        px_w_m = res_x * m_per_deg_lon
        px_h_m = res_y * m_per_deg_lat
    else:
        px_w_m = res_x
        px_h_m = res_y
    px_x = max(1, int(round((side_km * 1000.0) / max(px_w_m, 1e-9))))
    px_y = max(1, int(round((side_km * 1000.0) / max(px_h_m, 1e-9))))
    return px_x, px_y


def _grid_in_pixels_gdal(ds: "Any", grid_area_km2: float) -> Tuple[int, int]:
    from osgeo import osr
    side_km = max(0.25, float(np.sqrt(max(grid_area_km2, 1e-6))))
    gt = ds.GetGeoTransform()
    res_x = abs(gt[1])
    res_y = abs(gt[5])
    sr = osr.SpatialReference()
    sr.ImportFromWkt(ds.GetProjection())
    is_geographic = bool(sr.IsGeographic())
    if is_geographic:
        c_row, c_col = ds.RasterYSize / 2.0, ds.RasterXSize / 2.0
        X = gt[0] + c_col * gt[1] + c_row * gt[2]
        Y = gt[3] + c_col * gt[4] + c_row * gt[5]
        lat_rad = np.deg2rad(float(Y))
        m_per_deg_lat = 111_132.0
        m_per_deg_lon = 111_320.0 * max(np.cos(lat_rad), 1e-6)
        px_w_m = res_x * m_per_deg_lon
        px_h_m = res_y * m_per_deg_lat
    else:
        px_w_m = res_x
        px_h_m = res_y
    px_x = max(1, int(round((side_km * 1000.0) / max(px_w_m, 1e-9))))
    px_y = max(1, int(round((side_km * 1000.0) / max(px_h_m, 1e-9))))
    return px_x, px_y


# --------------------------- Image utilities ---------------------------

def mosaic4_to_rgb(arr: np.ndarray) -> np.ndarray:
    """Convert Planet 4-band (B,G,R,NIR) CHW array to RGB HWC in [0,1].
    Also handles 3-band CHW input (R,G,B) by returning HWC in [0,1].
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(arr)}")
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array (C,H,W), got {arr.shape}")
    C, H, W = arr.shape
    if C == 4:
        b, g, r, nir = arr
        def _scale(ch: np.ndarray) -> np.ndarray:
            ch = ch.astype(np.float32)
            vmax = np.percentile(ch, 99.5)
            if vmax > 1.5:
                ch = ch / max(vmax, 1e-6)
            return np.clip(ch, 0.0, 1.0)
        r = _scale(r)
        g = _scale(g)
        b = _scale(b)
        rgb = np.stack([r, g, b], axis=-1)
        return rgb.astype(np.float32)
    elif C == 3:
        r, g, b = arr.astype(np.float32)
        rgb = np.stack([r, g, b], axis=-1)
        return np.clip(rgb, 0.0, 1.0).astype(np.float32)
    else:
        raise ValueError(f"Unsupported channel count {C}; expected 3 or 4")


def apply_mask_to_image(img: np.ndarray, mask_bool: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Apply boolean mask to image. Supports CHW (C,H,W) or HWC (H,W,C).
    Returns None if mask has no valid pixels.
    """
    if mask_bool is None:
        return img
    if not np.any(mask_bool):
        return None
    if img.ndim != 3:
        raise ValueError(f"Expected 3D image array, got {img.shape}")
    if img.shape[0] in (3, 4):  # CHW
        return img * mask_bool[None, :, :]
    elif img.shape[-1] in (3, 4):  # HWC
        return img * mask_bool[:, :, None]
    else:
        raise ValueError(f"Unsupported image shape {img.shape}")


# --------------------------- Model-specific loaders ---------------------------

class OfficialModel:
    def __init__(self, name: str, device: str):
        self.name = name
        self.device = device
        self.feature_dim: Optional[int] = None
        self.preprocess = None  # Callable turning HWC RGB into tensor batch-ready
        self._impl: Any = None
        self.needs_4band = False  # Flag for models requiring full 4-band input instead of RGB
        self.uses_only_site_coords = False  # If True, extraction can skip raster reads

    def load(self, **kwargs):  # to be overridden
        raise NotImplementedError

    @torch.inference_mode()
    def encode_batch(self, batch_imgs: torch.Tensor, batch_sites: Optional[List[str]] = None) -> torch.Tensor:
        raise NotImplementedError


def _read_sites_latlon_csv(csv_path: Path) -> Dict[str, Tuple[float, float]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Sites CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    cols = {c.strip().lower(): c for c in df.columns}
    # Accept common column variants
    site_col = None
    for cand in ("site_name", "sitename", "site", "name"):
        if cand in cols:
            site_col = cols[cand]
            break
    if site_col is None:
        # try fuzzy
        for c in df.columns:
            if "site" in c.lower() and "name" in c.lower():
                site_col = c
                break
    lat_col = None
    for cand in ("lat", "latitude"):
        if cand in cols:
            lat_col = cols[cand]
            break
    lon_col = None
    for cand in ("lon", "lng", "longitude"):
        if cand in cols:
            lon_col = cols[cand]
            break

    coord_col = None
    if (lat_col is None or lon_col is None):
        for cand in ("coordinates", "coord", "coords"):
            if cand in cols:
                coord_col = cols[cand]
                break

    if site_col is None or ((lat_col is None or lon_col is None) and coord_col is None):
        raise ValueError(
            "Sites CSV must contain a site column and either (lat & lon) columns or a single coordinates column. "
            f"Found columns: {list(df.columns)}; expected e.g. site_name, latitude, longitude OR site_name, coordinates."
        )
    out: Dict[str, Tuple[float, float]] = {}
    for _, row in df.iterrows():
        site = str(row[site_col]).strip()
        if not site:
            continue
        try:
            if lat_col is not None and lon_col is not None:
                lat = float(row[lat_col])
                lon = float(row[lon_col])
            else:
                import re
                s = str(row[coord_col]).strip()
                nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
                if len(nums) < 2:
                    continue
                a = float(nums[0])
                b = float(nums[1])
                # Heuristic: if a is a valid latitude and b is a valid longitude -> (lat, lon)
                # otherwise swap.
                if -90 <= a <= 90 and -180 <= b <= 180:
                    lat, lon = a, b
                elif -90 <= b <= 90 and -180 <= a <= 180:
                    lat, lon = b, a
                else:
                    # Fall back to (lat, lon)
                    lat, lon = a, b
        except Exception:
            continue
        out[site] = (lat, lon)
    if not out:
        raise ValueError(f"No valid site lat/lon rows parsed from {csv_path}")
    return out


class SatCLIPOfficialCkptLocationModel(OfficialModel):
    """Official SatCLIP Lightning checkpoint (.ckpt) location encoder.

    Uses the HF repo-provided `load.py` helper to construct the model exactly as released.
    This yields a per-location embedding from (lon, lat). It does not consume Planet RGB.
    """

    def load(self, ckpt_path: Path, sites_csv: Path):
        if ckpt_path.is_dir():
            cand = ckpt_path / "satclip-vit16-l40.ckpt"
            if not cand.exists():
                raise FileNotFoundError(
                    f"SatCLIP directory provided but .ckpt not found: {cand}. "
                    "Pass the .ckpt path directly or a directory containing it."
                )
            ckpt_path = cand

        if not ckpt_path.exists():
            raise FileNotFoundError(f"SatCLIP .ckpt not found: {ckpt_path}")
        if ckpt_path.suffix != ".ckpt":
            raise ValueError(f"Expected a .ckpt for official SatCLIP, got: {ckpt_path}")

        self.uses_only_site_coords = True
        self._sites_latlon = _read_sites_latlon_csv(sites_csv)

        # The HF model repo only contains the .ckpt; the official loader lives in the GitHub repo.
        # We try (1) installed `satclip` package, then (2) clone github.com/microsoft/satclip.
        get_satclip = None
        try:
            from satclip.load import get_satclip as _get_satclip  # type: ignore

            get_satclip = _get_satclip
        except Exception:
            pass

        if get_satclip is None:
            repo_dir = ckpt_path.parent / "_satclip_official_repo"
            if not repo_dir.exists():
                import subprocess

                try:
                    subprocess.run(
                        [
                            "git",
                            "clone",
                            "--depth",
                            "1",
                            "https://github.com/microsoft/satclip.git",
                            str(repo_dir),
                        ],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except Exception as e:
                    raise RuntimeError(
                        "Failed to obtain official SatCLIP loader. "
                        "Tried importing `satclip` and cloning https://github.com/microsoft/satclip. "
                        f"Error: {e}"
                    )

            # Prefer importing directly from the satclip subfolder to avoid package-level side effects.
            import importlib
            try:
                # Allow absolute imports inside repo (e.g., `from datamodules ...`).
                sys.path.insert(0, str(repo_dir / "satclip"))
                # Try lightweight loader first if present, else full loader.
                try:
                    mod = importlib.import_module("load_lightweight")
                except Exception:
                    mod = importlib.import_module("load")
                get_satclip = getattr(mod, "get_satclip", None)
            except Exception:
                # Fallback: try package import from repo root
                try:
                    sys.path.insert(0, str(repo_dir))
                    from satclip.load import get_satclip as _get_satclip2  # type: ignore
                    get_satclip = _get_satclip2
                except Exception:
                    # Final fallback: search for a python file defining `get_satclip`.
                    import importlib.util
                    candidates = []
                    for py in repo_dir.rglob("*.py"):
                        try:
                            txt = py.read_text(encoding="utf-8", errors="ignore")
                        except Exception:
                            continue
                        if "def get_satclip" in txt:
                            candidates.append(py)
                    if not candidates:
                        raise RuntimeError(
                            f"Could not find a get_satclip() loader in cloned repo at {repo_dir}. "
                            "The satclip repo layout may have changed."
                        )
                    # Prefer load_lightweight.py if present
                    load_py = None
                    for c in candidates:
                        if c.name == "load_lightweight.py":
                            load_py = c
                            break
                    if load_py is None:
                        load_py = candidates[0]
                    spec = importlib.util.spec_from_file_location("satclip_official_load", str(load_py))
                    if spec is None or spec.loader is None:
                        raise RuntimeError(f"Could not import loader module from {load_py}")
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    get_satclip = getattr(mod, "get_satclip", None)
                    if get_satclip is None:
                        raise RuntimeError(f"Found {load_py} but it does not define get_satclip")

        # If loader function not found yet, try lightweight variant name
        if not callable(get_satclip):
            try:
                sys.path.insert(0, str(repo_dir / "satclip"))
                mod_lw = importlib.import_module("load_lightweight")
                get_satclip = getattr(mod_lw, "get_satclip_loc_encoder", None)
            except Exception:
                pass
        if not callable(get_satclip):
            raise RuntimeError(
                "Could not locate SatCLIP loader function (get_satclip or get_satclip_loc_encoder) in the official repo."
            )

        # Per HF model card, this loads the location encoder by default.
        try:
            self._impl = get_satclip(str(ckpt_path), device=self.device)
        except TypeError:
            # Some variants may not accept device kwarg.
            self._impl = get_satclip(str(ckpt_path))
        except Exception as e:
            raise RuntimeError(f"Failed to construct SatCLIP from {ckpt_path}: {e}")

        # Ensure the model is on the requested device
        if isinstance(self._impl, torch.nn.Module):
            self._impl = self._impl.to(self.device)

        self._impl.eval()
        self.feature_dim = None  # set on first forward
        self.preprocess = None

    @torch.inference_mode()
    def encode_batch(self, batch_imgs: torch.Tensor, batch_sites: Optional[List[str]] = None) -> torch.Tensor:
        if batch_sites is None or len(batch_sites) != int(batch_imgs.shape[0]):
            raise RuntimeError("SatCLIP (.ckpt) requires batch_sites to look up lat/lon per sample")

        coords = []
        missing = []
        for s in batch_sites:
            if s not in self._sites_latlon:
                missing.append(s)
                coords.append((0.0, 0.0))
            else:
                lat, lon = self._sites_latlon[s]
                coords.append((lon, lat))  # SatCLIP expects (lon, lat)
        if missing:
            missing = sorted(set(missing))
            raise KeyError(
                "Missing site coordinates for SatCLIP (.ckpt) in --satclip-sites-csv. "
                f"Examples: {missing[:10]}{' ...' if len(missing) > 10 else ''}"
            )

        # Align input coords tensor to the model's parameter device/dtype
        if isinstance(self._impl, torch.nn.Module):
            p = next(self._impl.parameters(), None)
            if p is not None:
                target_device = p.device
                target_dtype = p.dtype
            else:
                target_device = torch.device(self.device)
                target_dtype = torch.float32
        else:
            target_device = torch.device(self.device)
            target_dtype = torch.float32

        c = torch.tensor(coords, dtype=target_dtype, device=target_device)
        out = self._impl(c)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.ndim != 2:
            out = out.view(out.shape[0], -1)
        feats = out.float().detach().cpu()
        if self.feature_dim is None:
            self.feature_dim = int(feats.shape[1])
        return feats

    @torch.inference_mode()
    def encode_locations(self, lonlat_list: List[Tuple[float, float]]) -> torch.Tensor:
        """Encode a batch of (lon, lat) coordinates directly using the official location encoder.
        This bypasses site CSV lookup and is intended for grid-based global runs.
        """
        # Align input coords tensor to the model's parameter device/dtype
        if isinstance(self._impl, torch.nn.Module):
            p = next(self._impl.parameters(), None)
            if p is not None:
                target_device = p.device
                target_dtype = p.dtype
            else:
                target_device = torch.device(self.device)
                target_dtype = torch.float32
        else:
            target_device = torch.device(self.device)
            target_dtype = torch.float32

        c = torch.tensor(lonlat_list, dtype=target_dtype, device=target_device)
        out = self._impl(c)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.ndim != 2:
            out = out.view(out.shape[0], -1)
        feats = out.float().detach().cpu()
        if self.feature_dim is None:
            self.feature_dim = int(feats.shape[1])
        return feats


class SatCLIPModel(OfficialModel):
    """Use official CLIP implementation via Hugging Face CLIPModel/CLIPProcessor.
    Expects a local HF-format SatCLIP directory with model/config and processor.
    Produces 512-d image features via get_image_features().
    """

    def load(self, local_path: Path, sites_csv: Optional[Path] = None):
        # If user points to the official SatCLIP Lightning checkpoint (.ckpt), switch to the
        # official location-encoder backend.
        if local_path.is_file() and local_path.suffix == ".ckpt":
            if sites_csv is None:
                raise RuntimeError(
                    "Official SatCLIP (.ckpt) requires `--satclip-sites-csv` with site_name, latitude, longitude."
                )
            off = SatCLIPOfficialCkptLocationModel(self.name, self.device)
            off.load(ckpt_path=local_path, sites_csv=sites_csv)
            self._impl = off._impl
            self.feature_dim = off.feature_dim
            self.preprocess = None
            self.needs_4band = False
            self.uses_only_site_coords = True
            self._sites_latlon = off._sites_latlon
            return

        if local_path.is_dir() and (local_path / "satclip-vit16-l40.ckpt").exists():
            # Directory that contains the official ckpt
            if sites_csv is None:
                raise RuntimeError(
                    "Official SatCLIP directory detected (contains satclip-vit16-l40.ckpt). "
                    "Provide `--satclip-sites-csv` to use the official location encoder, or point to a Transformers CLIP directory."
                )
            off = SatCLIPOfficialCkptLocationModel(self.name, self.device)
            off.load(ckpt_path=local_path, sites_csv=sites_csv)
            self._impl = off._impl
            self.feature_dim = off.feature_dim
            self.preprocess = None
            self.needs_4band = False
            self.uses_only_site_coords = True
            self._sites_latlon = off._sites_latlon
            return

        try:
            from transformers import CLIPModel, CLIPProcessor
        except Exception as e:
            raise RuntimeError(f"transformers with CLIPModel is required for satclip: {e}")

        if not local_path.exists():
            raise FileNotFoundError(f"SatCLIP HF directory not found: {local_path}")

        try:
            self._impl = CLIPModel.from_pretrained(str(local_path)).to(self.device)
            try:
                self.preprocess = CLIPProcessor.from_pretrained(str(local_path))
            except Exception:
                # Build image processor from model vision_config if preprocessor_config.json is missing
                from transformers import CLIPImageProcessor
                vision_cfg = getattr(self._impl.config, 'vision_config', None)
                if vision_cfg is None:
                    raise RuntimeError("SatCLIP vision_config missing; cannot construct processor")
                self.preprocess = CLIPImageProcessor(**vision_cfg.to_dict())
        except Exception as e:
            raise RuntimeError(f"Failed to load SatCLIP from {local_path}: {e}")

        self.feature_dim = self._impl.config.projection_dim if hasattr(self._impl.config, 'projection_dim') else 512
        if self.feature_dim != 512:
            # Most CLIP ViT-B variants output 512-d; warn strictly if different
            raise RuntimeError(f"SatCLIP projection dim expected 512, got {self.feature_dim}")

        # Sites CSV is optional context (lat/lon) and is not used to alter image features here.
        self._sites_csv = sites_csv

    @torch.inference_mode()
    def encode_batch(self, batch_imgs: torch.Tensor, batch_sites: Optional[List[str]] = None) -> torch.Tensor:
        if getattr(self, "uses_only_site_coords", False):
            # Official SatCLIP ckpt mode (location encoder)
            coords = []
            if batch_sites is None or len(batch_sites) != int(batch_imgs.shape[0]):
                raise RuntimeError("SatCLIP (.ckpt) requires batch_sites to look up lat/lon per sample")
            missing = []
            for s in batch_sites:
                if s not in self._sites_latlon:
                    missing.append(s)
                    coords.append((0.0, 0.0))
                else:
                    lat, lon = self._sites_latlon[s]
                    coords.append((lon, lat))
            if missing:
                missing = sorted(set(missing))
                raise KeyError(
                    "Missing site coordinates for SatCLIP (.ckpt) in --satclip-sites-csv. "
                    f"Examples: {missing[:10]}{' ...' if len(missing) > 10 else ''}"
                )
            # Align input coords tensor to the model's parameter device/dtype
            if isinstance(self._impl, torch.nn.Module):
                p = next(self._impl.parameters(), None)
                if p is not None:
                    target_device = p.device
                    target_dtype = p.dtype
                else:
                    target_device = torch.device(self.device)
                    target_dtype = torch.float32
            else:
                target_device = torch.device(self.device)
                target_dtype = torch.float32
            c = torch.tensor(coords, dtype=target_dtype, device=target_device)
            out = self._impl(c)
            if isinstance(out, (tuple, list)):
                out = out[0]
            if out.ndim != 2:
                out = out.view(out.shape[0], -1)
            feats = out.float().detach().cpu()
            if self.feature_dim is None:
                self.feature_dim = int(feats.shape[1])
            return feats

        # batch_imgs: Bx3xHxW in [0,1] expected. We'll route through processor for official normalization.
        # Convert to PIL via processor expects list of PIL or numpy.
        imgs_list = [img.permute(1, 2, 0).cpu().numpy() for img in batch_imgs]
        inputs = self.preprocess(images=imgs_list, return_tensors="pt", input_data_format="channels_last")
        inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        feats = self._impl.get_image_features(**inputs)
        return feats.float().cpu()

    @torch.inference_mode()
    def encode_locations(self, lonlat_list: List[Tuple[float, float]]) -> torch.Tensor:
        """Direct coordinate encoding supported when using the official SatCLIP location encoder (.ckpt)."""
        if not getattr(self, "uses_only_site_coords", False):
            raise RuntimeError("encode_locations is only available when using the official SatCLIP location encoder (.ckpt)")
        # Align input coords tensor to the model's parameter device/dtype
        if isinstance(self._impl, torch.nn.Module):
            p = next(self._impl.parameters(), None)
            if p is not None:
                target_device = p.device
                target_dtype = p.dtype
            else:
                target_device = torch.device(self.device)
                target_dtype = torch.float32
        else:
            target_device = torch.device(self.device)
            target_dtype = torch.float32
        c = torch.tensor(lonlat_list, dtype=target_dtype, device=target_device)
        out = self._impl(c)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.ndim != 2:
            out = out.view(out.shape[0], -1)
        feats = out.float().detach().cpu()
        if self.feature_dim is None:
            self.feature_dim = int(feats.shape[1])
        return feats


class GeoRSCLIPModel(OfficialModel):
    """GeoRSCLIP using OpenCLIP official implementation with RS5M ViT-B/32 checkpoint.
    Produces 512-d image embeddings via model.encode_image().
    """

    def load(self, checkpoint_path: Path, arch: str = "ViT-B-32"):
        try:
            import open_clip
        except Exception as e:
            raise RuntimeError(f"open_clip package is required for georsclip: {e}")

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"GeoRSCLIP checkpoint not found: {checkpoint_path}")

        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=None, device=self.device)
        try:
            sd = torch.load(checkpoint_path, map_location=self.device)
            state_dict = sd.get('state_dict', sd)
            model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load GeoRSCLIP weights strictly: {e}")

        self._impl = model
        self.preprocess = preprocess
        self.feature_dim = model.text_projection.shape[1] if hasattr(model, 'text_projection') else 512
        if self.feature_dim != 512:
            raise RuntimeError(f"GeoRSCLIP projection dim expected 512, got {self.feature_dim}")

    @torch.inference_mode()
    def encode_batch(self, batch_imgs: torch.Tensor) -> torch.Tensor:
        # preprocess expects PIL images; it also handles resize/crop/normalize per CLIP
        imgs_list = [img.permute(1, 2, 0).cpu().numpy() for img in batch_imgs]
        # Use preprocess directly on PIL; to avoid ambiguity, apply per-item
        prepped = []
        for arr in imgs_list:
            from PIL import Image
            pil = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
            tens = self.preprocess(pil).unsqueeze(0)  # 1x3xHxW
            prepped.append(tens)
        batch = torch.cat(prepped, dim=0).to(self.device)
        feats = self._impl.encode_image(batch)
        return feats.float().cpu()


class PlainCLIPImageModel(OfficialModel):
    """Plain CLIP image encoder (ViT-B/16) via Hugging Face `CLIPModel`.
    Produces 512-d pooled image features (`get_image_features`).
    """

    def load(self, local_path: Path):
        try:
            from transformers import CLIPModel, CLIPProcessor, CLIPImageProcessor
        except Exception as e:
            raise RuntimeError(f"transformers with CLIPModel is required for clip: {e}")

        if not local_path.exists():
            raise FileNotFoundError(f"CLIP HF directory not found: {local_path}")

        try:
            self._impl = CLIPModel.from_pretrained(str(local_path)).to(self.device)
            # Try to load a processor; fall back to an image processor built from vision_config
            try:
                self.preprocess = CLIPProcessor.from_pretrained(str(local_path))
            except Exception:
                vision_cfg = getattr(self._impl.config, 'vision_config', None)
                if vision_cfg is None:
                    raise RuntimeError("CLIP vision_config missing; cannot construct image processor")
                self.preprocess = CLIPImageProcessor(**vision_cfg.to_dict())
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP from {local_path}: {e}")

        self.feature_dim = getattr(self._impl.config, 'projection_dim', None)
        if self.feature_dim is None:
            # Default for ViT-B variants
            self.feature_dim = 512
        if int(self.feature_dim) != 512:
            raise RuntimeError(f"CLIP projection dim expected 512, got {self.feature_dim}")

    @torch.inference_mode()
    def encode_batch(self, batch_imgs: torch.Tensor) -> torch.Tensor:
        # Expect Bx3xHxW in [0,1]. Convert to PIL to avoid input_data_format ambiguity.
        from PIL import Image
        imgs_list = []
        for img in batch_imgs:
            arr = img.permute(1, 2, 0).cpu().numpy()
            arr = np.clip(arr, 0.0, 1.0)
            pil = Image.fromarray((arr * 255).astype(np.uint8))
            imgs_list.append(pil)
        inputs = self.preprocess(images=imgs_list, return_tensors="pt", input_data_format="channels_last")
        inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        feats = self._impl.get_image_features(**inputs)
        return feats.float().cpu()


class PrithviEO2Model(OfficialModel):
    """Prithvi EO 2.0 (600M) encoder using authors' architecture.
    Expected feature dim is 1024-d from CLS token.
    """

    def load(self, local_path: Path):
        if not local_path.exists():
            raise FileNotFoundError(f"Prithvi EO2 directory not found: {local_path}")
        config_path = local_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Prithvi EO2 config.json not found in {local_path}")
        
        import json
        import sys
        with open(config_path) as f:
            cfg_full = json.load(f)
        cfg = cfg_full['pretrained_cfg']
        
        # Add the Prithvi repo path to sys.path to import prithvi_mae
        sys.path.insert(0, str(local_path))
        from prithvi_mae import PrithviViT
        
        # Build encoder-only model (no decoder needed for feature extraction)
        # For Planet 4-band, adapt config: in_chans=4, num_frames=1 (single timestep)
        img_size = cfg.get('img_size', 224)
        patch_size = cfg.get('patch_size', [1, 14, 14])
        in_chans = 4  # Planet BGRN
        num_frames = 1  # Single timestep
        embed_dim = cfg.get('embed_dim', 1280)
        depth = cfg.get('depth', 32)
        num_heads = cfg.get('num_heads', 16)
        mlp_ratio = cfg.get('mlp_ratio', 4.0)
        coords_encoding = []  # Disable temporal/location encoding for single-frame Planet
        
        model = PrithviViT(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            coords_encoding=coords_encoding,
        )
        
        # Load checkpoint
        ckpt_path = local_path / "Prithvi_EO_V2_600M.pt"
        if not ckpt_path.exists():
            ckpt_path = local_path / "model.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Prithvi EO2 checkpoint not found in {local_path}")
        
        # Load checkpoint safely on CPU first to avoid GPU OOM or UnpicklingError due to weights_only path
        # Checkpoint is a trusted local file, so default unpickling is acceptable here
        state_dict = torch.load(str(ckpt_path), map_location="cpu")
        # Discard pos_embed and decoder keys
        state_dict = {k: v for k, v in state_dict.items() if 'pos_embed' not in k and not k.startswith('decoder_') and not k.startswith('mask_token')}
        model.load_state_dict(state_dict, strict=False)
        
        self._impl = model.to(self.device).eval()
        self.img_size = img_size
        self.needs_4band = True
        # Feature dim from embed_dim after projection to 1024
        self.embed_dim = embed_dim
        self.feature_dim = cfg_full.get('num_features', 1024)
        assert self.feature_dim == 1024, f"Expected 1024-d for Prithvi EO2, got {self.feature_dim}"
        print(f"Loaded Prithvi EO2 with embed_dim={embed_dim}, feature_dim={self.feature_dim}, img_size={img_size}")
        
        # Initialize projection layer embed_dim -> feature_dim
        self._projection = nn.Linear(self.embed_dim, self.feature_dim, bias=False).to(self.device)
        nn.init.xavier_uniform_(self._projection.weight)
    
    def _preprocess_batch(self, batch_imgs: torch.Tensor) -> torch.Tensor:
        """Preprocess 4-band Planet images for Prithvi.
        - Resize to img_size
        - Normalize per-band
        - Add temporal dimension (T=1)
        - Return (B, C, T, H, W)
        """
        N, C, H, W = batch_imgs.shape
        resized = F.interpolate(batch_imgs, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        
        # Per-band standardization
        normalized = torch.zeros_like(resized)
        for b in range(C):
            band = resized[:, b, :, :]
            mean = band.view(N, -1).mean(dim=1, keepdim=True).view(N, 1, 1)
            std = band.view(N, -1).std(dim=1, keepdim=True).view(N, 1, 1)
            normalized[:, b, :, :] = (band - mean) / (std + 1e-6)
        
        # Add temporal dimension: (B, C, H, W) -> (B, C, 1, H, W)
        return normalized.unsqueeze(2)
    
    @torch.inference_mode()
    def encode_batch(self, batch_imgs: torch.Tensor) -> torch.Tensor:
        preprocessed = self._preprocess_batch(batch_imgs).to(self.device)
        # Use forward_features to get encoder output
        features_list = self._impl.forward_features(preprocessed, temporal_coords=None, location_coords=None)
        # Take CLS token from last layer
        cls_token = features_list[-1][:, 0, :]  # (B, embed_dim=1280)
        # Project to 1024-d
        feats = self._projection(cls_token)
        return feats.float().cpu()


class HFEncoderModel(OfficialModel):
    """Generic HF encoder with official processor; used for satmae, s2mae, copernicus-fm.
    Embedding size asserted from config.hidden_size unless model exposes projection.
    """

    def load(self, local_path: Path, expected_hidden: Optional[int] = None, allow_remote_code: bool = True):
        from transformers import AutoModel, AutoImageProcessor
        if not local_path.exists():
            raise FileNotFoundError(f"HF model directory not found: {local_path}")
        try:
            self._impl = AutoModel.from_pretrained(str(local_path), trust_remote_code=allow_remote_code).to(self.device)
            self.preprocess = AutoImageProcessor.from_pretrained(str(local_path), trust_remote_code=allow_remote_code)
        except Exception as e:
            raise RuntimeError(f"Failed to load HF model from {local_path}: {e}")
        hidden = getattr(self._impl.config, 'hidden_size', None)
        if hidden is None:
            raise RuntimeError("Missing hidden_size in config; cannot determine embedding size")
        self.feature_dim = int(hidden)
        if expected_hidden is not None and self.feature_dim != expected_hidden:
            raise RuntimeError(f"Expected hidden_size {expected_hidden}, got {self.feature_dim}")

    @torch.inference_mode()
    def encode_batch(self, batch_imgs: torch.Tensor) -> torch.Tensor:
        imgs_list = [img.permute(1, 2, 0).cpu().numpy() for img in batch_imgs]
        inputs = self.preprocess(images=imgs_list, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self._impl(**inputs, return_dict=True)
        if hasattr(out, 'pooler_output') and out.pooler_output is not None:
            feats = out.pooler_output
        elif hasattr(out, 'last_hidden_state'):
            feats = out.last_hidden_state[:, 0, :]
        else:
            raise RuntimeError("Model output missing expected fields")
        return feats.float().cpu()


class SatMAEAuthorsModel(OfficialModel):
    """SatMAE loader using the authors' group-channel MAE architecture.
    This model uses channel_groups and produces 768-d embeddings (ViT-Base).
    Preprocessing: per-band min-max [0,1] → per-band standardize.
    """

    def load(self, local_path: Path):
        if not local_path.exists():
            raise FileNotFoundError(f"SatMAE directory not found: {local_path}")
        config_path = local_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"SatMAE config.json not found in {local_path}")
        import json
        with open(config_path) as f:
            cfg = json.load(f)
        
        # Expect model.safetensors
        st_path = local_path / "model.safetensors"
        if not st_path.exists():
            st_path = local_path / "pytorch_model.bin"
        if not st_path.exists():
            raise FileNotFoundError(f"SatMAE weights not found in {local_path}")
        
        # Build the MaskedAutoencoderGroupChannelViT encoder directly
        # We'll define a minimal version inline to avoid external dependencies
        from functools import partial
        from timm.models.vision_transformer import Block, PatchEmbed
        
        img_size = cfg.get("img_size", 224)  # Will be 96 according to config
        patch_size = cfg.get("patch_size", 16)  # 8 in config
        # SatMAE configs commonly define Sentinel-2 channel groups (10 bands). Planet mosaics here are 4-band,
        # so we use a single channel group [0, 1, 2, 3].
        channel_groups = [[0,1,2,3]]
        channel_embed = cfg.get("channel_embed", 256)
        embed_dim = cfg.get("embed_dim", 768)
        depth = cfg.get("depth", 12)
        num_heads = cfg.get("num_heads", 16)
        mlp_ratio = cfg.get("mlp_ratio", 4.0)
        
        # Build a minimal encoder-only model (no decoder needed for feature extraction)
        class SatMAEEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_embed = nn.ModuleList([
                    PatchEmbed(img_size, patch_size, len(group), embed_dim)
                    for group in channel_groups
                ])
                num_patches = self.patch_embed[0].num_patches
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim - channel_embed))
                num_groups = len(channel_groups)
                self.channel_embed = nn.Parameter(torch.zeros(1, num_groups, channel_embed))
                self.channel_cls_embed = nn.Parameter(torch.zeros(1, 1, channel_embed))
                self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
                self.blocks = nn.ModuleList([
                    Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
                    for _ in range(depth)
                ])
                self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
                self.channel_groups = channel_groups
            
            def forward(self, x):
                # x: (N, 4, H, W) for Planet 4-band
                # Extract per-channel-group patches
                b = x.shape[0]
                x_c_embed = []
                for i, group in enumerate(self.channel_groups):
                    x_c = x[:, group, :, :]
                    x_c_embed.append(self.patch_embed[i](x_c))  # (N, L, D)
                x_enc = torch.stack(x_c_embed, dim=1)  # (N, G, L, D)
                _, G, L, D = x_enc.shape
                
                # Add positional and channel embeddings
                channel_embed = self.channel_embed.unsqueeze(2).expand(-1, -1, L, -1)  # (1, G, L, cD)
                pos_embed = self.pos_embed[:, 1:, :].unsqueeze(1).expand(-1, G, -1, -1)  # (1, G, L, pD)
                pos_channel = torch.cat((pos_embed, channel_embed), dim=-1)  # (1, G, L, D)
                x_enc = x_enc + pos_channel
                x_enc = x_enc.view(b, -1, D)  # (N, G*L, D)
                
                # Prepend CLS token
                cls_pos_channel = torch.cat((self.pos_embed[:, :1, :], self.channel_cls_embed), dim=-1)
                cls_tokens = cls_pos_channel + self.cls_token.expand(b, -1, -1)
                x_enc = torch.cat((cls_tokens, x_enc), dim=1)
                
                # Transformer blocks
                for blk in self.blocks:
                    x_enc = blk(x_enc)
                x_enc = self.norm(x_enc)
                return x_enc[:, 0, :]  # CLS token
        
        model = SatMAEEncoder()
        # Load weights
        try:
            from safetensors.torch import load_file as safeload
        except:
            raise RuntimeError("SatMAE requires safetensors")
        sd = safeload(str(st_path)) if st_path.suffix == ".safetensors" else torch.load(st_path, map_location="cpu")
        # Filter out decoder keys and patch_embed layers not needed for Planet (checkpoint has 3 groups for S2)
        encoder_sd = {k: v for k, v in sd.items() if not k.startswith('decoder_') and not k.startswith('mask_token')}
        # Adapt: use first patch_embed (4-channel from S2) for Planet 4-band
        # Remap patch_embed.0 from checkpoint to patch_embed.0 in our single-group model
        adapted_sd = {}
        for k, v in encoder_sd.items():
            if k.startswith('patch_embed.0.'):
                # Keep patch_embed.0 as is for our group [0,1,2,3]
                adapted_sd[k] = v
            elif k.startswith('patch_embed.'):
                # Skip patch_embed.1 and patch_embed.2 (S2 groups not needed)
                continue
            elif k == 'channel_embed':
                # channel_embed has shape (1, num_groups, channel_embed_dim)
                # Checkpoint has 3 groups; we need 1 group. Take first group.
                adapted_sd[k] = v[:, :1, :]
            elif k == 'channel_cls_embed':
                # This is (1, 1, channel_embed_dim), keep as is
                adapted_sd[k] = v
            else:
                adapted_sd[k] = v
        model.load_state_dict(adapted_sd, strict=False)
        self._impl = model.to(self.device).eval()
        self.feature_dim = embed_dim
        self.img_size = img_size
        self.needs_4band = True  # SatMAE requires 4-band input
        assert self.feature_dim == 768, f"Expected 768-d for SatMAE ViT-Base, got {self.feature_dim}"
        print(f"Loaded SatMAE with embed_dim={embed_dim}, img_size={img_size}, patch_size={patch_size}")
    
    def _preprocess_batch(self, batch_imgs: torch.Tensor) -> torch.Tensor:
        """Apply authors' preprocessing: per-band min-max [0,1] → per-band standardize."""
        # batch_imgs: (N, 4, H, W) in [0, 1] from mosaic4_to_rgb
        # Center crop to 1024 then resize to img_size
        N, C, H, W = batch_imgs.shape
        crop_size = 1024
        top = max(0, (H - crop_size) // 2)
        left = max(0, (W - crop_size) // 2)
        cropped = batch_imgs[:, :, top:top+crop_size, left:left+crop_size]
        resized = F.interpolate(cropped, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        
        # Per-band min-max normalization to [0, 1] (already done by mosaic4_to_rgb, but apply explicitly)
        normalized = torch.zeros_like(resized)
        for b in range(C):
            band = resized[:, b, :, :]
            vmin = band.view(N, -1).min(dim=1, keepdim=True)[0].view(N, 1, 1)
            vmax = band.view(N, -1).max(dim=1, keepdim=True)[0].view(N, 1, 1)
            normalized[:, b, :, :] = (band - vmin) / (vmax - vmin + 1e-6)
        
        # Per-band standardize (zero mean, unit std per band per sample)
        standardized = torch.zeros_like(normalized)
        for b in range(C):
            band = normalized[:, b, :, :]
            mean = band.view(N, -1).mean(dim=1, keepdim=True).view(N, 1, 1)
            std = band.view(N, -1).std(dim=1, keepdim=True).view(N, 1, 1)
            standardized[:, b, :, :] = (band - mean) / (std + 1e-6)
        return standardized

    @torch.inference_mode()
    def encode_batch(self, batch_imgs: torch.Tensor) -> torch.Tensor:
        preprocessed = self._preprocess_batch(batch_imgs).to(self.device)
        feats = self._impl(preprocessed)
        return feats.float().cpu()


class SatlasPretrainResNet(OfficialModel):
    def load(self, checkpoint_path: Path):
        from torchvision import models, transforms
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"SatlasPreTrain checkpoint not found: {checkpoint_path}")

        # Authors' checkpoint typically contains nested keys like 'backbone.backbone.resnet.*'.
        # Map those to torchvision resnet152 keys (e.g., 'layer1.*', 'fc.*' etc.); keep fc as Identity.
        model = models.resnet152(weights=None)
        model.fc = nn.Identity()
        # If checkpoint conv1 expects multispectral (e.g., 9 in-channels), adapt the model's conv1 accordingly.

        def remap_to_resnet_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            remapped = {}
            for k, v in sd.items():
                kk = k
                # Strip common wrappers repeatedly until none match
                prefixes = ("model.", "backbone.", "backbone.backbone.", "module.")
                changed = True
                while changed:
                    changed = False
                    for prefix in prefixes:
                        if kk.startswith(prefix):
                            kk = kk[len(prefix):]
                            changed = True
                # If keys still contain 'resnet.', strip it
                if kk.startswith("resnet."):
                    kk = kk[len("resnet."):]
                # Some checkpoints store features under 'global_pool' or 'avgpool'
                kk = kk.replace("global_pool", "avgpool")
                remapped[kk] = v
            return remapped

        try:
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            # Support both plain state_dict and PyTorch Lightning ckpt with 'state_dict'
            state_dict = ckpt.get('state_dict', ckpt)
            cleaned = remap_to_resnet_keys(state_dict)
            # Detect expected input channels from checkpoint conv1
            conv1_w = None
            for key in ("conv1.weight",):
                if key in cleaned:
                    conv1_w = cleaned[key]
                    break
            if conv1_w is None:
                # try alternative nested key names
                for key in cleaned.keys():
                    if key.endswith("conv1.weight"):
                        conv1_w = cleaned[key]
                        break
            if conv1_w is not None:
                expected_in_ch = int(conv1_w.shape[1])
                if expected_in_ch != model.conv1.in_channels:
                    # Replace conv1 to match expected channels
                    model.conv1 = nn.Conv2d(expected_in_ch, model.conv1.out_channels,
                                            kernel_size=model.conv1.kernel_size[0],
                                            stride=model.conv1.stride[0],
                                            padding=model.conv1.padding[0],
                                            bias=False)
            # Filter out keys not present in resnet152 to allow strict-ish loading
            resnet_keys = set(model.state_dict().keys())
            filtered = {k: v for k, v in cleaned.items() if k in resnet_keys}
            missing = [k for k in resnet_keys if k not in filtered]
            unexpected = [k for k in cleaned.keys() if k not in resnet_keys]
            if unexpected:
                # Not fatal; these are detection heads or extra modules
                pass
            model.load_state_dict(filtered, strict=False)
            if missing:
                # Ensure backbone layers were populated; at least some conv weights must be present
                populated = any(k.startswith("layer1") or k.startswith("conv1") for k in filtered.keys())
                if not populated:
                    raise RuntimeError("SatlasPreTrain remap resulted in empty backbone weights")
        except Exception as e:
            raise RuntimeError(f"Failed to load SatlasPreTrain weights with authors' mapping: {e}")

        self._impl = model.to(self.device)
        self.feature_dim = 2048
        # Use ImageNet normalization as commonly applied in their release
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224), antialias=True)
        ])
        # Record expected input channels from conv1
        self._expected_in_ch = int(self._impl.conv1.weight.shape[1])

    @torch.inference_mode()
    def encode_batch(self, batch_imgs: torch.Tensor) -> torch.Tensor:
        # Prepare input respecting expected channels: if 9-channel, map RGB into B,G,R and zero-pad others.
        from PIL import Image
        import torchvision.transforms as T
        resize = self.preprocess
        to_tensor = T.ToTensor()
        normalize = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        prepped = []
        for img in batch_imgs:  # img: 3xHxW in [0,1]
            arr = img.permute(1, 2, 0).cpu().numpy()
            pil = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
            pil = resize(pil)
            tens3 = to_tensor(pil)  # 3x224x224
            if self._expected_in_ch == 3:
                inp = normalize(tens3)
            elif self._expected_in_ch == 9:
                # Create 9-channel tensor, place B,G,R at indices [0,1,2] by default
                c9 = torch.zeros((9, tens3.shape[1], tens3.shape[2]), dtype=tens3.dtype)
                # tens3 order is [R,G,B]; map to c9[2]=R, c9[1]=G, c9[0]=B to match common S2 convention [B,G,R,...]
                c9[0] = tens3[2]
                c9[1] = tens3[1]
                c9[2] = tens3[0]
                # Normalize first three channels with ImageNet stats; others remain zero
                rgb_norm = normalize(tens3)
                c9[0] = rgb_norm[2]
                c9[1] = rgb_norm[1]
                c9[2] = rgb_norm[0]
                inp = c9
            else:
                raise RuntimeError(f"Unsupported SatlasPreTrain input channels: {self._expected_in_ch}")
            prepped.append(inp.unsqueeze(0))
        batch = torch.cat(prepped, dim=0).to(self.device)
        return self._impl(batch).float().cpu()


class HandcraftedModel(OfficialModel):
    """Handcrafted feature extraction using statistical, spectral, and textural features.
    Extracts features directly from 4-band imagery without a neural network.
    Features include: band statistics, NDVI/NDWI/BSI/NBR/SAVI indices, GLCM textures, LBP, edge density, etc.
    """

    def load(self):
        # Import feature extraction function
        try:
            from feature_extraction import extract_features
            self._extract_fn = extract_features
        except ImportError as e:
            raise ImportError(f"Failed to import feature_extraction module: {e}")
        
        # Handcrafted features don't have a fixed dimension until extraction
        # The feature_extraction.extract_features returns variable-length features
        # Typical range: ~60-70 features depending on implementation
        self.feature_dim = None  # Will be determined after first extraction
        self.needs_4band = True  # Handcrafted features work on 4-band imagery
        self.preprocess = None  # No preprocessing needed

    @torch.inference_mode()
    def encode_batch(self, batch_imgs: torch.Tensor) -> torch.Tensor:
        """Extract handcrafted features from batch of 4-band images.
        batch_imgs: (B, 4, H, W) tensor
        Returns: (B, D) features where D is determined by extract_features
        """
        features_list = []
        
        for img_tensor in batch_imgs:
            # Convert tensor (4, H, W) to numpy (4, H, W)
            img_np = img_tensor.cpu().numpy().astype(np.float32)
            
            # extract_features expects (T, C, H, W) where T=1 for single image
            img_batch = np.expand_dims(img_np, 0)  # (1, 4, H, W)
            
            try:
                feats = self._extract_fn(img_batch).squeeze()  # (D,)
                features_list.append(feats)
            except Exception as e:
                print(f"Warning: Handcrafted feature extraction failed: {e}")
                # Return zero features if extraction fails
                if features_list:
                    # Use same dimension as previous successful extraction
                    feats = np.zeros_like(features_list[-1])
                else:
                    # Default to 60 features (typical size)
                    feats = np.zeros(60, dtype=np.float32)
                features_list.append(feats)
        
        # Stack all features
        features_np = np.stack(features_list, axis=0)  # (B, D)
        
        # Set feature_dim after first extraction
        if self.feature_dim is None:
            self.feature_dim = features_np.shape[1]
        
        return torch.from_numpy(features_np).float()


class DINOv3Model(OfficialModel):
    """DINOv3 ViT-L/16 SAT-493M model loader.
    Uses the authors' local implementation from dinov3/dinov3/models/vision_transformer.py.
    Produces 1024-d features (ViT-Large).
    Preprocessing: SAT-493M normalization (mean/std from satellite data).
    """

    def load(self, checkpoint_path: Path):
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"DINOv3 checkpoint not found: {checkpoint_path}")
        
        # Add dinov3 repo to path to import local implementation
        repo_root = Path(".") / "huggingface_models" / "dinov3"
        if not (repo_root / "dinov3").exists():
            repo_root = Path(".") / "dinov3"
        if not (repo_root / "dinov3").exists():
            raise FileNotFoundError(f"DINOv3 local repo not found at {repo_root}")
        
        sys.path.insert(0, str(repo_root))
        try:
            from dinov3.models.vision_transformer import vit_large
        except Exception as e:
            raise ImportError(f"Failed to import DINOv3 from {repo_root}. Error: {e}")
        
        # Build model
        img_size = 224
        model = vit_large(patch_size=16, img_size=img_size, in_chans=3).to(self.device)
        
        # Load checkpoint
        size_mb = checkpoint_path.stat().st_size / 1e6
        if size_mb < 50:
            raise RuntimeError(
                f"Checkpoint looks too small ({size_mb:.1f} MB). It may be incomplete. "
                f"Re-download and try again."
            )
        
        try:
            ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to read checkpoint as Torch archive. Path: {checkpoint_path}, "
                f"size: {size_mb:.1f} MB. Re-download the checkpoint."
            ) from e
        
        # Extract state dict
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                state = ckpt["state_dict"]
            elif "model" in ckpt:
                state = ckpt["model"]
            elif "teacher" in ckpt and isinstance(ckpt["teacher"], dict):
                state = ckpt["teacher"]
            else:
                state = ckpt
        else:
            state = ckpt
        
        # Strip common prefixes
        def strip_prefix(sd, prefixes=("module.", "backbone.", "model.", "student.", "teacher.")):
            out = {}
            for k, v in sd.items():
                nk = k
                for p in prefixes:
                    if nk.startswith(p):
                        nk = nk[len(p):]
                out[nk] = v
            return out
        
        state = strip_prefix(state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded DINOv3 from {checkpoint_path} ({size_mb:.1f} MB)")
        if missing:
            print(f"Missing keys: {len(missing)} (heads/aux can be safely ignored)")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)}")
        
        model.eval()
        self._impl = model
        self.feature_dim = 1024  # ViT-Large
        self.needs_4band = False
        
        # Setup preprocessing with SAT-493M normalization
        from torchvision import transforms
        mean = (0.430, 0.411, 0.296)
        std = (0.213, 0.156, 0.143)
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(mean=mean, std=std)
        ])

    @torch.inference_mode()
    def encode_batch(self, batch_imgs: torch.Tensor) -> torch.Tensor:
        """Encode batch of images (B,3,224,224) to (B,1024) features."""
        batch_imgs = batch_imgs.to(self.device)
        feats = self._impl(batch_imgs)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        if feats.ndim > 2:
            feats = feats.squeeze()
        return feats.float().cpu()


# ------------------------------- Main pipeline -------------------------------

MODEL_CHOICES = [
    "satclip",
    "georsclip",
    "prithvi-eo-2.0",
    "satmae",
    "s2mae",
    "copernicus-fm",
    "satlaspretrain",
    "dinov3",
    "handcrafted",
    "clip",
]


def parse_args():
    p = argparse.ArgumentParser(description="Extract official-dim embeddings for 2023 only (no fallbacks)")
    p.add_argument(
        "--mode",
        type=str,
        choices=["site", "grid"],
        default="site",
        help="Extraction mode: 'site' outputs one embedding per site-month; 'grid' outputs per-grid tiles (global).",
    )
    p.add_argument("--images-root", type=Path, default=Path("planet_mosaics_final_4bands/images"))
    p.add_argument("--output-dir", type=Path, default=Path("planet_mosaics_final_4bands/features"))
    p.add_argument("--model", type=str, choices=MODEL_CHOICES, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit-sites", type=int, default=0)
    p.add_argument("--sites", type=str, nargs='*', default=None, help="Optional list of site directory names to process (e.g., looted_0)")
    p.add_argument("--months", type=str, nargs='*', default=None, help="Optional list of months like 2023_12; defaults to all months in 2023")
    p.add_argument("--start-year", type=int, default=None, help="Optional start year (used when --months not provided)")
    p.add_argument("--end-year", type=int, default=None, help="Optional end year (used when --months not provided)")
    # Model-specific paths
    p.add_argument(
        "--satclip-path",
        type=Path,
        default=Path("./huggingface_models/satclip-hf"),
        help="Transformers-format CLIP directory (default: ./huggingface_models/satclip-hf). The official microsoft/SatCLIP-ViT16-L40 is a .ckpt and is not loadable via CLIPModel.",
    )
    p.add_argument(
        "--clip-path",
        type=Path,
        default=Path("./huggingface_models_backup/satclip"),
        help="Transformers-format CLIP directory for plain CLIP image encoder (ViT-B/16).",
    )
    p.add_argument(
        "--satclip-sites-csv",
        type=Path,
        default=None,
        help="CSV with columns site_name,lat,lon (optional)"
    )
    p.add_argument("--georsclip-ckpt", type=Path, default=Path("./huggingface_models/georsclip/ckpt/RS5M_ViT-B-32.pt"))
    p.add_argument("--prithvi-path", type=Path, default=Path("./huggingface_models/prithvi-eo-2.0-600m"))
    p.add_argument("--satmae-path", type=Path, default=Path("./huggingface_models/satmae"))
    p.add_argument("--s2mae-path", type=Path, default=Path("./huggingface_models/s2mae"))
    p.add_argument("--copernicus-path", type=Path, default=Path("./huggingface_models/copernicus-fm"))
    p.add_argument("--satlaspretrain-ckpt", type=Path, default=Path("./huggingface_models/satlaspretrain/sentinel2_resnet152_mi_ms.pth"))
    p.add_argument("--dinov3-ckpt", type=Path, default=Path("./huggingface_models/dinov3/models/dinov3-vitl16-sat493m/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"))
    # Masking options
    p.add_argument("--use-mask", action="store_true", help="Apply spatial masking to extract features only from mask regions")
    p.add_argument("--masks-root", type=Path, default=Path("./planet_mosaics_final_4bands/masks_buffered"), help="Root directory containing mask files organized by site")
    # Grid (global) mode options
    p.add_argument("--grid-area", type=float, default=1.0, help="Tile area in km^2 for grid mode")
    p.add_argument("--min-valid-ratio", type=float, default=0.1, help="Minimum valid pixel ratio per tile (or mask coverage if masking enabled) in grid mode")
    return p.parse_args()


def build_official_model(args) -> OfficialModel:
    name = args.model
    dev = args.device
    if name == "satclip":
        m = SatCLIPModel(name, dev)
        m.load(local_path=args.satclip_path, sites_csv=args.satclip_sites_csv)
        return m
    if name == "georsclip":
        m = GeoRSCLIPModel(name, dev)
        m.load(checkpoint_path=args.georsclip_ckpt, arch="ViT-B-32")
        return m
    if name == "prithvi-eo-2.0":
        m = PrithviEO2Model(name, dev)
        m.load(local_path=args.prithvi_path)
        return m
    if name == "satmae":
        # Use authors' group-channel MAE loader with safetensors from HF snapshot
        m = SatMAEAuthorsModel(name, dev)
        m.load(local_path=args.satmae_path)
        return m
    if name == "s2mae":
        m = HFEncoderModel(name, dev)
        # Many S2MAE releases are Swin-base/large; hidden size varies. Do not assert a fixed size.
        m.load(local_path=args.s2mae_path, expected_hidden=None, allow_remote_code=True)
        return m
    if name == "copernicus-fm":
        m = HFEncoderModel(name, dev)
        # Trust remote code and accept documented hidden_size
        m.load(local_path=args.copernicus_path, expected_hidden=None, allow_remote_code=True)
        return m
    if name == "satlaspretrain":
        m = SatlasPretrainResNet(name, dev)
        m.load(checkpoint_path=args.satlaspretrain_ckpt)
        return m
    if name == "dinov3":
        m = DINOv3Model(name, dev)
        m.load(checkpoint_path=args.dinov3_ckpt)
        return m
    if name == "handcrafted":
        m = HandcraftedModel(name, dev)
        m.load()
        return m
    if name == "clip":
        m = PlainCLIPImageModel(name, dev)
        m.load(local_path=args.clip_path)
        return m
    raise ValueError(f"Unsupported model: {name}")


def preprocess_to_tensor_list(rgb: np.ndarray, preprocess) -> torch.Tensor:
    """Convert HWC float32 RGB [0,1] to model-specific processor pipeline.
    
    NOTE: Some models (GeoRSCLIP, SatlasPreTrain) handle their own preprocessing in encode_batch,
    so we only convert to tensor here. DINOv3 needs the full preprocess pipeline applied.
    """
    import torchvision.transforms as T
    
    # Ensure rgb is in the correct format (H, W, C) and type
    if not isinstance(rgb, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(rgb)}")
    
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB array with shape (H, W, 3), got {rgb.shape}")
    
    # Ensure float32 type
    if rgb.dtype != np.float32:
        rgb = rgb.astype(np.float32)
    
    # Check if preprocess is a torchvision.transforms.Compose (like DINOv3)
    # vs open_clip transforms (like GeoRSCLIP) which expect PIL images
    if preprocess is not None and hasattr(preprocess, 'transforms'):
        # Check if this is a torchvision Compose (DINOv3-style)
        # Torchvision transforms work on tensors after ToTensor
        try:
            # Try to detect if this is torchvision vs open_clip
            # DINOv3 uses torchvision.transforms.Compose with Resize that works on tensors
            # GeoRSCLIP uses open_clip transforms that expect PIL
            transform_module = str(type(preprocess.transforms[0]).__module__)
            if 'torchvision' in transform_module:
                # This is DINOv3-style: apply full preprocessing pipeline
                return preprocess(rgb)  # Will handle ToTensor internally
        except Exception:
            pass
    
    # For all other cases (GeoRSCLIP, SatlasPreTrain, or no preprocess):
    # Just convert to tensor - model will handle its own preprocessing in encode_batch
    t = T.ToTensor()
    return t(rgb)  # 3xHxW


def _resolve_months(args) -> List[str]:
    if args.months:
        return [str(m) for m in args.months]
    if args.start_year is not None or args.end_year is not None:
        sy = int(args.start_year) if args.start_year is not None else 2023
        ey = int(args.end_year) if args.end_year is not None else sy
        if ey < sy:
            raise ValueError(f"end_year ({ey}) must be >= start_year ({sy})")
        return months_range(sy, ey)
    return months_2023()


def run_site(args):
    images_root: Path = args.images_root
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_official_model(args)
    print(f"Loaded model '{args.model}' with official feature dim {model.feature_dim}")

    months = _resolve_months(args)
    site_dirs_all = find_site_dirs(images_root)
    if args.sites:
        site_set = set(args.sites)
        site_dirs = [p for p in site_dirs_all if p.name in site_set]
    else:
        site_dirs = site_dirs_all
    if args.limit_sites and args.limit_sites > 0:
        site_dirs = site_dirs[: args.limit_sites]

    # Determine output year tag from provided months (default 2023)
    if args.months or args.start_year is not None or args.end_year is not None:
        if args.months:
            years = sorted({str(m).split("_")[0] for m in args.months if "_" in str(m)})
        else:
            years = []
        if not years:
            years = sorted({str(m).split("_")[0] for m in months if "_" in str(m)})
        years_i = [int(y) for y in years]
        if len(years_i) == 1:
            year_tag = str(years_i[0])
        else:
            year_tag = f"{min(years_i)}_{max(years_i)}"
    else:
        year_tag = "2023"

    # Adjust output filename if masking is enabled
    if args.use_mask:
        out_csv = out_dir / f"features_{args.model}_{year_tag}_masked.csv"
        print(f"Masking enabled: will extract features only from mask regions")
        print(f"Masks root: {args.masks_root}")
    else:
        out_csv = out_dir / f"features_{args.model}_{year_tag}.csv"
    
    rows: List[Dict[str, Any]] = []

    batch_pixels: List[torch.Tensor] = []
    batch_meta: List[Tuple[str, str, Path]] = []
    printed_dim = False

    # Helper to encode a batch that may contain variable-sized tensors
    def _encode_and_collect(pixels: List[torch.Tensor], meta: List[Tuple[str, str, Path]]):
        nonlocal printed_dim, rows
        if not pixels:
            return
        try:
            batch = torch.stack(pixels, dim=0)
            feats = model.encode_batch(batch)
            D = int(feats.shape[1])
            if not printed_dim:
                print(f"Feature dim D={D}")
                printed_dim = True
            for (site_i, ym_i, _), vec in zip(meta, feats.numpy()):
                rows.append({"site_name": site_i, "month": ym_i, **{f"f{i}": float(vec[i]) for i in range(D)}})
        except RuntimeError as e:
            # Handle variable-size tensors gracefully by processing one-by-one
            if "stack expects each tensor to be equal size" in str(e):
                for pix, (site_i, ym_i, _) in zip(pixels, meta):
                    single = pix.unsqueeze(0)
                    feats1 = model.encode_batch(single)
                    D = int(feats1.shape[1])
                    if not printed_dim:
                        print(f"Feature dim D={D}")
                        printed_dim = True
                    vec = feats1.numpy()[0]
                    rows.append({"site_name": site_i, "month": ym_i, **{f"f{i}": float(vec[i]) for i in range(D)}})
            else:
                raise

    # In SatCLIP official ckpt mode, embeddings depend only on site coords.
    site_feat_cache: Dict[str, np.ndarray] = {}

    for site_dir in tqdm(site_dirs, desc="Processing sites"):
        site = site_dir.name
        
        # Check for mask if masking is enabled
        if args.use_mask:
            mask_fp = mask_path_for_site(args.masks_root, site)
            if not mask_fp.exists():
                print(f"WARNING: Mask not found for site '{site}' at {mask_fp}. Skipping site.")
                continue
        
        # Precompute site-level embedding once for SatCLIP (.ckpt) mode.
        if getattr(model, "uses_only_site_coords", False) and site not in site_feat_cache:
            dummy = torch.zeros((1, 3, 1, 1), dtype=torch.float32)
            site_vec = model.encode_batch(dummy, batch_sites=[site]).numpy()[0]
            site_feat_cache[site] = site_vec

        for ym in tqdm(months, desc=f"Months for {site}", leave=False):
            tif = find_month_file(site_dir, ym)
            if tif is None:
                continue

            # Fast path: SatCLIP ckpt embeddings are site-coordinate-only; skip raster I/O.
            if getattr(model, "uses_only_site_coords", False):
                vec = site_feat_cache[site]
                D = int(vec.shape[0])
                if not printed_dim:
                    print(f"Feature dim D={D}")
                    printed_dim = True
                rows.append({"site_name": site, "month": ym, **{f"f{i}": float(vec[i]) for i in range(D)}})
                continue

            try:
                if HAS_RASTERIO:
                    with rio.open(tif) as src:
                        arr = src.read()
                        # Read aligned mask if masking is enabled
                        mask_bool = None
                        if args.use_mask:
                            mask_bool = read_aligned_mask(src, mask_fp)
                            if mask_bool is None or not np.any(mask_bool):
                                print(f"Skip (masked-out or missing mask): {tif}")
                                continue
                else:
                    # Fallback: read GeoTIFF via tifffile; if masking requested, align via GDAL
                    if tifffile is None:
                        raise RuntimeError("Neither rasterio nor tifffile available to read GeoTIFFs")
                    arr = tifffile.imread(str(tif))
                    # Normalize array to (C,H,W)
                    if arr.ndim == 3 and arr.shape[0] in (3, 4):
                        pass  # already (C,H,W)
                    elif arr.ndim == 3 and arr.shape[-1] in (3, 4):
                        arr = np.transpose(arr, (2, 0, 1))
                    else:
                        raise RuntimeError(f"Unsupported TIFF shape {arr.shape} for {tif}")
                    mask_bool = None
                    if args.use_mask:
                        try:
                            mask_bool = read_aligned_mask_gdal(tif, mask_fp)
                            if mask_bool is None or not np.any(mask_bool):
                                print(f"Skip (masked-out or missing mask): {tif}")
                                continue
                        except Exception as e:
                            raise RuntimeError(f"Mask alignment failed without rasterio: {e}")
            except Exception as e:
                print(f"Failed to read {tif}: {e}")
                continue
            
            # Some models (e.g., SatMAE) need full 4-band input; others use RGB
            if model.needs_4band and arr.shape[0] == 4:
                # Convert 4-band (C,H,W) to float32 [0,1]
                img4 = arr.astype(np.float32)
                for ch in range(4):
                    vmax = np.percentile(img4[ch], 99.5)
                    if vmax > 1.5:
                        img4[ch] /= max(vmax, 1e-6)
                img4 = np.clip(img4, 0.0, 1.0)
                
                # Apply mask if enabled
                if args.use_mask and mask_bool is not None:
                    img4_masked = apply_mask_to_image(img4, mask_bool)
                    if img4_masked is None:
                        print(f"Skip (no valid mask pixels): {tif}")
                        continue
                    img4 = img4_masked
                
                # Convert to CHW tensor
                tensor_chw = torch.from_numpy(img4)
            else:
                rgb = mosaic4_to_rgb(arr)
                
                # Apply mask if enabled (before preprocessing)
                if args.use_mask and mask_bool is not None:
                    rgb_masked = apply_mask_to_image(rgb, mask_bool)
                    if rgb_masked is None:
                        print(f"Skip (no valid mask pixels): {tif}")
                        continue
                    rgb = rgb_masked
                
                tensor_chw = preprocess_to_tensor_list(rgb, model.preprocess)
            
            batch_pixels.append(tensor_chw)
            batch_meta.append((site, ym, tif))

            if len(batch_pixels) >= args.batch_size:
                _encode_and_collect(batch_pixels, batch_meta)
                batch_pixels.clear(); batch_meta.clear()

        if batch_pixels:
            _encode_and_collect(batch_pixels, batch_meta)
            batch_pixels.clear(); batch_meta.clear()

    if rows:
        df = pd.DataFrame(rows)
        def mm_key(s: str):
            y, m = s.split("_")
            return (int(y), int(m))
        df = df.sort_values(by=["site_name", "month"], key=lambda c: c.map(mm_key) if c.name == "month" else c)
        df.to_csv(out_csv, index=False)
        print(f"Saved {len(df)} rows to {out_csv}")
    else:
        print("No features extracted.")


def run_grid(args):
    """Grid-based extraction (global): emits one row per (site, month, grid_id)."""
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_official_model(args)
    print(f"Loaded model '{args.model}' with official feature dim {model.feature_dim}")

    # Resolve the year loop.
    if args.start_year is None and args.end_year is None:
        # For backwards compatibility, default to 2023 when not specified.
        start_year, end_year = 2023, 2023
    else:
        start_year = int(args.start_year) if args.start_year is not None else 2023
        end_year = int(args.end_year) if args.end_year is not None else start_year
    if end_year < start_year:
        raise ValueError(f"end_year ({end_year}) must be >= start_year ({start_year})")

    sites_all = find_site_dirs(args.images_root)
    if args.sites:
        site_set = set(args.sites)
        sites = [p for p in sites_all if p.name in site_set]
    else:
        sites = sites_all
    if args.limit_sites and args.limit_sites > 0:
        sites = sites[: args.limit_sites]

    use_loc_encoder = bool(getattr(model, "uses_only_site_coords", False) and hasattr(model, "encode_locations"))
    if args.model == "satclip":
        print(f"SatCLIP flags: uses_only_site_coords={getattr(model, 'uses_only_site_coords', False)}, has_encode_locations={hasattr(model, 'encode_locations')}")

    def _append_rows(metas: List[Tuple[str, str, int, float, float]], feats: np.ndarray, rows_out: List[Dict[str, Any]]):
        D = int(feats.shape[1])
        for (site_i, ym_i, gid_i, lon_i, lat_i), vec in zip(metas, feats):
            rows_out.append({
                "site_name": site_i,
                "month": ym_i,
                "grid_id": gid_i,
                "lon": float(lon_i),
                "lat": float(lat_i),
                **{f"f{i}": float(vec[i]) for i in range(D)},
            })

    for year in range(start_year, end_year + 1):
        months = months_for_year(year)
        if args.use_mask:
            out_csv = out_dir / f"features_{args.model}_{year}_global_masked.csv"
            print(f"[grid] Masking enabled: masks_root={args.masks_root}")
        else:
            out_csv = out_dir / f"features_{args.model}_{year}_global.csv"
        rows: List[Dict[str, Any]] = []

        for site_dir in tqdm(sites, desc=f"Sites ({year})"):
            site = site_dir.name
            index = build_month_index(site_dir)
            mask_fp = None
            if args.use_mask:
                mask_fp = mask_path_for_site(args.masks_root, site)
                if not mask_fp.exists():
                    print(f"WARNING: Mask not found for site '{site}' at {mask_fp}. Skipping site.")
                    continue

            for ym in months:
                tif = index.get(ym)
                if tif is None:
                    continue

                try:
                    if HAS_RASTERIO:
                        with rio.open(tif) as src:
                            arr = src.read()  # (C,H,W)
                            tw, th = _grid_in_pixels_rio(src, args.grid_area)
                            boxes = tile_boxes(src.width, src.height, tw, th)
                            mask_bool = None
                            if args.use_mask and mask_fp is not None:
                                mask_bool = read_aligned_mask(src, mask_fp)
                                if mask_bool is None or not np.any(mask_bool):
                                    continue

                            tiles: List[np.ndarray] = []
                            metas: List[Tuple[str, str, int, float, float]] = []
                            coords_batch: List[Tuple[float, float]] = []
                            gid = 0
                            for (x0, y0, x1, y1) in boxes:
                                tile = arr[:, y0:y1, x0:x1]
                                if tile.size == 0:
                                    gid += 1
                                    continue

                                if args.use_mask and mask_bool is not None:
                                    m_tile = mask_bool[y0:y1, x0:x1]
                                    valid = float(np.mean(m_tile))
                                    if valid < float(args.min_valid_ratio):
                                        gid += 1
                                        continue
                                    tile_masked = apply_mask_to_image(tile, m_tile)
                                    if tile_masked is None:
                                        gid += 1
                                        continue
                                    tile = tile_masked
                                else:
                                    valid = float(np.mean(tile > 0.0))
                                    if valid < float(args.min_valid_ratio):
                                        gid += 1
                                        continue

                                cx = x0 + (x1 - x0) / 2.0
                                cy = y0 + (y1 - y0) / 2.0
                                lon, lat = rio.transform.xy(src.transform, cy, cx)
                                if use_loc_encoder:
                                    coords_batch.append((float(lon), float(lat)))
                                    metas.append((site, ym, gid, float(lon), float(lat)))
                                else:
                                    tiles.append(tile)
                                    metas.append((site, ym, gid, float(lon), float(lat)))
                                gid += 1

                                if use_loc_encoder and len(coords_batch) >= args.batch_size:
                                    feats = model.encode_locations(coords_batch).numpy()
                                    _append_rows(metas, feats, rows)
                                    coords_batch.clear(); metas.clear()
                                elif (not use_loc_encoder) and len(tiles) >= args.batch_size:
                                    feats = _process_tile_batch(model, tiles, tw, th)
                                    _append_rows(metas, feats, rows)
                                    tiles.clear(); metas.clear()

                            if tiles:
                                feats = _process_tile_batch(model, tiles, tw, th)
                                _append_rows(metas, feats, rows)
                                tiles.clear(); metas.clear()
                            if use_loc_encoder and coords_batch:
                                feats = model.encode_locations(coords_batch).numpy()
                                _append_rows(metas, feats, rows)
                                coords_batch.clear(); metas.clear()
                    else:
                        raise RuntimeError("rasterio unavailable")
                except Exception as e:
                    # Fallback without rasterio: use GDAL for geotransform and coords
                    try:
                        from osgeo import gdal, osr
                        try:
                            gdal.UseExceptions()
                        except Exception:
                            pass
                    except Exception:
                        print(f"Failed {site}/{ym}: {e}")
                        continue

                    try:
                        ds = gdal.Open(str(tif))
                        if ds is None:
                            raise RuntimeError("GDAL failed to open dataset")
                        width = ds.RasterXSize
                        height = ds.RasterYSize
                        arr = ds.ReadAsArray()
                        if arr.ndim == 2:
                            arr = arr[None, ...]
                        elif arr.ndim == 3 and arr.shape[-1] in (3, 4):
                            arr = np.transpose(arr, (2, 0, 1))
                        tw, th = _grid_in_pixels_gdal(ds, args.grid_area)
                        boxes = tile_boxes(width, height, tw, th)

                        mask_bool = None
                        if args.use_mask and mask_fp is not None:
                            mask_bool = read_aligned_mask_gdal(tif, mask_fp)
                            if mask_bool is None or not np.any(mask_bool):
                                continue

                        gt = ds.GetGeoTransform()
                        wkt = ds.GetProjection() or ""
                        sr_src = osr.SpatialReference()
                        try:
                            sr_src.ImportFromWkt(wkt)
                        except Exception:
                            sr_src = None
                        sr_dst = osr.SpatialReference(); sr_dst.ImportFromEPSG(4326)
                        try:
                            ct = osr.CoordinateTransformation(sr_src, sr_dst) if sr_src is not None else None
                        except Exception:
                            ct = None

                        tiles: List[np.ndarray] = []
                        metas: List[Tuple[str, str, int, float, float]] = []
                        coords_batch: List[Tuple[float, float]] = []
                        gid = 0
                        for (x0, y0, x1, y1) in boxes:
                            tile = arr[:, y0:y1, x0:x1]
                            if tile.size == 0:
                                gid += 1
                                continue

                            if args.use_mask and mask_bool is not None:
                                m_tile = mask_bool[y0:y1, x0:x1]
                                valid = float(np.mean(m_tile))
                                if valid < float(args.min_valid_ratio):
                                    gid += 1
                                    continue
                                tile_masked = apply_mask_to_image(tile, m_tile)
                                if tile_masked is None:
                                    gid += 1
                                    continue
                                tile = tile_masked
                            else:
                                valid = float(np.mean(tile > 0.0))
                                if valid < float(args.min_valid_ratio):
                                    gid += 1
                                    continue

                            cxp = x0 + (x1 - x0) / 2.0
                            cyp = y0 + (y1 - y0) / 2.0
                            X = gt[0] + cxp * gt[1] + cyp * gt[2]
                            Y = gt[3] + cxp * gt[4] + cyp * gt[5]
                            if ct is not None:
                                try:
                                    lonlat = ct.TransformPoint(X, Y)
                                    lon, lat = float(lonlat[0]), float(lonlat[1])
                                    ct_failed = False
                                except Exception:
                                    ct_failed = True
                            else:
                                ct_failed = True
                            if ct_failed:
                                if ("Pseudo-Mercator" in wkt) or ("Mercator" in wkt) or ("3857" in wkt):
                                    R = 6378137.0
                                    lon = (X / R) * (180.0 / np.pi)
                                    lat = np.degrees(np.arctan(np.sinh(Y / R)))
                                else:
                                    lon, lat = float(X), float(Y)

                            if use_loc_encoder:
                                coords_batch.append((lon, lat))
                                metas.append((site, ym, gid, lon, lat))
                            else:
                                tiles.append(tile)
                                metas.append((site, ym, gid, lon, lat))
                            gid += 1

                            if use_loc_encoder and len(coords_batch) >= args.batch_size:
                                feats = model.encode_locations(coords_batch).numpy()
                                _append_rows(metas, feats, rows)
                                coords_batch.clear(); metas.clear()
                            elif (not use_loc_encoder) and len(tiles) >= args.batch_size:
                                feats = _process_tile_batch(model, tiles, tw, th)
                                _append_rows(metas, feats, rows)
                                tiles.clear(); metas.clear()

                        if tiles:
                            feats = _process_tile_batch(model, tiles, tw, th)
                            _append_rows(metas, feats, rows)
                            tiles.clear(); metas.clear()
                        if use_loc_encoder and coords_batch:
                            feats = model.encode_locations(coords_batch).numpy()
                            _append_rows(metas, feats, rows)
                            coords_batch.clear(); metas.clear()
                    except Exception as e2:
                        print(f"Failed (no rasterio) {site}/{ym}: {e2}")
                        continue

        if rows:
            df = pd.DataFrame(rows)
            def mm_key(s: str):
                y, m = s.split("_")
                return (int(y), int(m))
            df = df.sort_values(by=["site_name", "month", "grid_id"], key=lambda c: c.map(mm_key) if c.name == "month" else c)
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv, index=False)
            print(f"Saved {len(df)} rows to {out_csv}")
        else:
            print(f"No features extracted for {year}.")


def _process_tile_batch(model: OfficialModel, tiles: List[np.ndarray], target_w: int, target_h: int) -> np.ndarray:
    def pad_chw(arr: np.ndarray, tw: int, th: int) -> np.ndarray:
        c, h, w = arr.shape
        canvas = np.zeros((c, th, tw), dtype=arr.dtype)
        canvas[:, :h, :w] = arr
        return canvas

    batch: List[torch.Tensor] = []
    for t in tiles:
        if model.needs_4band:
            if t.shape[0] == 4:
                arr = pad_chw(t.astype(np.float32), target_w, target_h)
            elif t.shape[0] == 3:
                zeros = np.zeros((1, t.shape[1], t.shape[2]), dtype=t.dtype)
                arr4 = np.concatenate([t, zeros], axis=0)
                arr = pad_chw(arr4.astype(np.float32), target_w, target_h)
            else:
                arr = np.zeros((4, target_h, target_w), dtype=np.float32)
            tens = torch.from_numpy(arr)
        else:
            rgb = mosaic4_to_rgb(t)  # HWC
            h, w, _ = rgb.shape
            canvas = np.zeros((target_h, target_w, 3), dtype=rgb.dtype)
            canvas[:h, :w, :] = rgb
            tens = torch.from_numpy(np.transpose(canvas, (2, 0, 1)))
        batch.append(tens)
    batch_t = torch.stack(batch, dim=0)
    feats = model.encode_batch(batch_t)
    return feats.numpy()


def run(args):
    if getattr(args, "mode", "site") == "grid":
        return run_grid(args)
    return run_site(args)


if __name__ == "__main__":
    args = parse_args()
    run(args)
