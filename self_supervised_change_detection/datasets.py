# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Dataset and month indexing utilities for monthly pipelines."""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
# Extend years to include 2024 (inclusive) -> 2017-2024 = 8 * 12 = 96 months
MONTHS = [f"{y}_{m:02d}" for y in range(2017, 2025) for m in range(1, 13)]  # 96 months (T)
T = len(MONTHS)

def load_scaler_npz(scaler_path: str | None, F: int):
    """Load feature normalization statistics from an .npz file.

    Args:
        scaler_path: Path to the .npz file containing scaler statistics.
            If None or the file does not exist, identity statistics are returned.
        F: Number of features (used to initialise default mean/std arrays).

    Returns:
        Dictionary with keys 'mean', 'std', and optionally 'month_mean',
        'month_std', 'feat_median', 'feat_mad', etc.
    """
    out = {}
    out["mean"] = np.zeros(F, dtype=np.float32)
    out["std"] = np.ones(F, dtype=np.float32)
    out["month_mean"] = None
    out["month_std"] = None
    if scaler_path and os.path.exists(scaler_path):
        try:
            s = np.load(scaler_path)
        except Exception as e:
            print(f"[Scaler] Failed to load scaler npz ({e}); using identity stats.")
            return out
        def _maybe(key, dest):
            if key in s:
                out[dest] = s[key].astype(np.float32)
        _maybe("mean", "mean"); _maybe("std", "std")
        _maybe("feat_median", "feat_median"); _maybe("feat_mad", "feat_mad")
        _maybe("month_mean", "month_mean"); _maybe("month_std", "month_std")
        _maybe("month_median", "month_median"); _maybe("month_mad", "month_mad")
    out["std"][out["std"] == 0] = 1.0
    if out.get("month_std") is not None:
        out["month_std"][out["month_std"] == 0] = 1.0
    return out

class UnsupervisedSiteDataset(Dataset):
    """Dataset of per-site monthly feature time series.

    Loads feature vectors from a CSV, normalises them using precomputed
    statistics, interpolates missing months, and exposes each site as a
    (T, F) tensor.
    """

    def __init__(self, features_csv: str, gt_csv: str | None, split: str, scaler_path: str | None,
                 split_col: str = "split", per_month_feature_norm: bool = True, robust: bool = False):
        """Initialize UnsupervisedSiteDataset.

        Args:
            features_csv: Path to the CSV containing per-site monthly features.
            gt_csv: Path to a ground-truth CSV with site labels and splits.
                If None, all sites default to the 'train' split.
            split: One of 'train', 'val', 'test', 'all', or 'all_looted'.
            scaler_path: Path to an .npz file with normalization statistics.
            split_col: Column name in gt_csv that indicates the split.
            per_month_feature_norm: Whether to apply per-month normalisation
                before global normalisation.
            robust: If True, use median/MAD statistics instead of mean/std.
        """
        super().__init__()
        self.fdf = pd.read_csv(features_csv)
        self.feature_cols = sorted([c for c in self.fdf.columns if c.startswith("f") and c[1:].isdigit()], key=lambda x: int(x[1:]))
        self.F = len(self.feature_cols)
        stats = load_scaler_npz(scaler_path, self.F)
        if robust and ("feat_median" in stats and "feat_mad" in stats):
            self.mean = stats["feat_median"].astype(np.float32)
            mad = stats["feat_mad"].astype(np.float32); mad[mad == 0] = 1.0
            self.std = (mad * 1.4826).astype(np.float32)
        else:
            self.mean = stats["mean"]; self.std = stats["std"]
        self.month_mean = None; self.month_std = None
        if per_month_feature_norm and stats.get("month_mean") is not None and stats.get("month_std") is not None:
            if robust and ("month_median" in stats and "month_mad" in stats):
                m_median = stats.get("month_median"); m_mad = stats.get("month_mad")
                if m_median is not None and m_mad is not None:
                    self.month_mean = m_median.astype(np.float32)
                    mm = m_mad.astype(np.float32); mm[mm == 0] = 1.0
                    self.month_std = (mm * 1.4826).astype(np.float32)
            if self.month_mean is None or self.month_std is None:
                self.month_mean = stats.get("month_mean"); self.month_std = stats.get("month_std")
        if self.month_mean is not None and self.month_mean.shape != (T, self.F):
            self.month_mean = None; self.month_std = None

        if gt_csv and os.path.exists(gt_csv):
            gdf = pd.read_csv(gt_csv)
            scol = split_col if (split_col in gdf.columns) else "split"
            gdf = gdf[["site_name", scol, "looted"]].copy() if "looted" in gdf.columns else gdf[["site_name", scol]].copy()
        else:
            sites = sorted(self.fdf["site_name"].unique().tolist())
            gdf = pd.DataFrame({"site_name": sites, split_col: ["train"] * len(sites)})

        if split in ("train", "val", "test"):
            self.sites = gdf[gdf.get(split_col) == split]["site_name"].unique().tolist()
        elif split == "all":
            self.sites = gdf["site_name"].unique().tolist()
        elif split == "all_looted":
            if "looted" in gdf.columns:
                try:
                    mask = gdf["looted"].astype(int) == 1
                except Exception:
                    mask = gdf["looted"] == 1
                self.sites = gdf[mask]["site_name"].unique().tolist()
            else:
                raise ValueError("'all_looted' split requested but 'looted' column not found in ground truth CSV")
        else:
            raise ValueError("split must be one of train/val/test/all/all_looted")
        self.sites = sorted([s for s in self.sites if s in set(self.fdf["site_name"].unique())])
        if split == "all" and len(self.sites) == 0:
            print("[WARN] Ground truth provided no sites for split 'all'; falling back to all feature sites (ignoring GT).")
            self.sites = sorted(self.fdf["site_name"].unique().tolist())

        self.preserved = {}
        if "looted" in gdf.columns:
            for _, r in gdf.iterrows():
                self.preserved[r["site_name"]] = 0 if int(r.get("looted", 1)) == 1 else 1

        self.tensors = []
        for s in self.sites:
            mat = np.full((T, self.F), np.nan, dtype=np.float32)
            sdf = self.fdf[self.fdf["site_name"] == s]
            for _, r in sdf.iterrows():
                try:
                    y_str, m_str = str(r["month"]).split("_")
                    y, m = int(y_str), int(m_str)
                    t = (y - 2017) * 12 + (m - 1)
                    if 0 <= t < T:
                        mat[t, :] = r[self.feature_cols].to_numpy(dtype=np.float32, copy=False)
                except Exception:
                    continue
            idx = np.arange(T)
            for j in range(self.F):
                col = mat[:, j]
                mask = np.isfinite(col)
                if mask.any():
                    mat[:, j] = np.interp(idx, idx[mask], col[mask]).astype(np.float32)
                else:
                    mat[:, j] = self.mean[j]
            if self.month_mean is not None and self.month_std is not None:
                mat = (mat - self.month_mean) / self.month_std
            mat = (mat - self.mean) / self.std
            self.tensors.append(torch.tensor(mat, dtype=torch.float32))

    def __len__(self):
        """Return the number of sites in the dataset."""
        return len(self.sites)

    def __getitem__(self, i):
        """Return the feature tensor, site name, and preservation label for a site.

        Args:
            i: Index of the site.

        Returns:
            Tuple of (features, site_name, preserved) where features is a
            (T, F) tensor, site_name is a string, and preserved is 0
            (looted), 1 (preserved), or -1 (unknown).
        """
        x = self.tensors[i]
        s = self.sites[i]
        pres = self.preserved.get(s, -1)
        return x, s, pres
