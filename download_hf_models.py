#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""download_hf_models.py

Downloads the *exact* model artifacts expected by the embedding extraction scripts.

This repo's extractors are a mix of:
- HF snapshot repos that include custom code / non-transformers checkpoints (Prithvi, SatMAE)
- HF checkpoints consumed directly by our loaders (SatCLIP, GeoRSCLIP, Satlas)
- DINOv3: local code repo + a specific SAT-493M .pth checkpoint (downloaded from HF)

All downloads land under `./huggingface_models/<model_name>/`.
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path
import requests
from tqdm import tqdm
import logging

# Set environment variables to avoid TensorFlow imports
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_download.log')
    ]
)
logger = logging.getLogger(__name__)

# Model registry used by the model downloader and embedding extraction scripts.
MODEL_REGISTRY = {
    # --- Foundation models used in embedding extraction ---
    "satclip": {
        # Official Microsoft SatCLIP checkpoint (PyTorch Lightning .ckpt, not a Transformers CLIPModel directory)
        "method": "hf_file",
        "repo_id": "microsoft/SatCLIP-ViT16-L40",
        "revision": "main",
        "local_path": "huggingface_models/satclip",
        "hf_filename": "satclip-vit16-l40.ckpt",
        "feature_dim": 512,
        "architecture": "SatCLIP ViT16-L40 (.ckpt)",
    },
    # Aliases
    "satclip-vit16-l40": {
        "method": "alias",
        "alias_of": "satclip",
    },
    "satclip-official": {
        "method": "alias",
        "alias_of": "satclip",
    },
    # Transformers-format CLIP variant previously used by some scripts (RGB-style).
    # Kept for backwards compatibility.
    "satclip-hf": {
        "method": "snapshot",
        "repo_id": "NemesisAlm/clip-fine-tuned-satellite",
        "local_path": "huggingface_models/satclip-hf",
        "feature_dim": 512,
        "architecture": "CLIP ViT-B (projection=512)",
    },
    "georsclip": {
        "method": "snapshot_allow_patterns",
        "repo_id": "Zilun/GeoRSCLIP",
        "local_path": "huggingface_models/georsclip",
        "allow_patterns": ["ckpt/*", "README.md"],
        "primary_file": "ckpt/RS5M_ViT-B-32.pt",
        "feature_dim": 512,
        "architecture": "OpenCLIP ViT-B/32",
    },
    "satmae": {
        "method": "snapshot",
        "repo_id": "MVRL/satmae-vitbase-multispec-pretrain",
        "local_path": "huggingface_models/satmae",
        "feature_dim": 768,
        "architecture": "SatMAE ViT-Base",
    },
    "prithvi-eo-2.0-600m": {
        "method": "snapshot",
        "repo_id": "ibm-nasa-geospatial/Prithvi-EO-2.0-600M",
        "local_path": "huggingface_models/prithvi-eo-2.0-600m",
        "feature_dim": 1024,
        "architecture": "Prithvi EO 2.0 600M",
    },
    # # Backward-compatible alias (older docs / scripts)
    # "prithvi-eo-2.0": {
    #     "method": "alias",
    #     "alias_of": "prithvi-eo-2.0-600m",
    # },
    "satlaspretrain": {
        "method": "hf_file",
        "repo_id": "maxhuber15/SatlasPretrain",
        "revision": "main",
        "local_path": "huggingface_models/satlaspretrain",
        "hf_filename": "sentinel2_resnet152_mi_ms.pth",
        "fallback_local_paths": [
            "huggingface_models_backup/satlaspretrain/sentinel2_resnet152_mi_ms.pth"
        ],
        "feature_dim": 2048,
        "architecture": "ResNet152",
    },
    "dinov3": {
        # Our extractor expects:
        # - local code at huggingface_models/dinov3/dinov3/...
        # - checkpoint at huggingface_models/dinov3/models/dinov3-vitl16-sat493m/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
        "method": "git_and_hf_file",
        "local_path": "huggingface_models/dinov3",
        "git_url": "https://github.com/facebookresearch/dinov3.git",
        "git_ref": "main",
        # Public HF repo that hosts the exact .pth expected by this codebase
        "weights_repo_id": "MVRL/dinov3_vitl16_sat",
        "weights_revision": "6001d1230be0410921db4e6c8eabf34ace1f1621",
        "weights_filename": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        "weights_dest_relpath": "models/dinov3-vitl16-sat493m/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        "feature_dim": 1024,
        "architecture": "ViT-L/16 (SAT-493M)",
    },

    # --- Other optional models (kept for convenience) ---
    "s2mae": {
        "method": "files",
        "repo_id": "ibm-nasa-geospatial/s2mae",
        "local_path": "huggingface_models/s2mae",
        "files": ["config.json", "pytorch_model.bin"],
        "feature_dim": 768,
        "architecture": "ViT/Swin (varies)",
    },
    "copernicus-fm": {
        "method": "files",
        "repo_id": "amazon-science/copernicus-foundation-model",
        "local_path": "huggingface_models/copernicus-fm",
        "files": ["config.json", "pytorch_model.bin"],
        "feature_dim": 768,
        "architecture": "ViT (varies)",
    },
}


def _resolve_alias(model_name: str) -> str:
    info = MODEL_REGISTRY.get(model_name)
    if not info:
        return model_name
    if info.get("method") == "alias":
        return str(info["alias_of"])
    return model_name


def _ensure_git_checkout(dest_dir: Path, git_url: str, git_ref: str = "main") -> None:
    """Ensure a working tree exists at dest_dir.

    If dest_dir exists and looks like the dinov3 repo (has dinov3/), do nothing.
    Otherwise, clone a shallow checkout.
    """
    if (dest_dir / "dinov3").exists():
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    # If directory is non-empty but not the expected repo, fail fast to avoid destructive behavior.
    if any(dest_dir.iterdir()):
        raise RuntimeError(
            f"Target dir {dest_dir} exists but does not look like a dinov3 checkout (missing 'dinov3/' folder). "
            f"Please clear it or choose a different destination."
        )

    logger.info(f"Cloning {git_url}@{git_ref} into {dest_dir}")
    subprocess.run(["git", "clone", "--depth", "1", "--branch", git_ref, git_url, str(dest_dir)], check=True)

def download_file(url, dest_path, desc=None, hf_token: str | None = None):
    """
    Download a file with progress bar
    """
    try:
        headers = {}
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
        response = requests.get(url, stream=True, headers=headers)
        if response.status_code != 200:
            logger.error(f"Failed to download {url}, status code: {response.status_code}")
            return None
            
        total_size = int(response.headers.get('content-length', 0))
        desc = desc or f"Downloading {dest_path.name}"
        
        with open(dest_path, 'wb') as f, tqdm(
            desc=desc,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        file_size = dest_path.stat().st_size
        if file_size < 1024:  # Less than 1 KB is suspicious for model files
            logger.warning(f"Downloaded file is suspiciously small: {file_size} bytes")
            
        return file_size
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return None

def download_model_direct(model_name, hf_token: str | None = None, force: bool = False):
    """
    Download a model using direct file URLs
    """
    model_name = _resolve_alias(model_name)
    model_info = MODEL_REGISTRY[model_name]
    local_path = Path(model_info["local_path"])

    # Always keep downloads under ./huggingface_models/<model_name>/ (project convention).
    base_dir = Path("huggingface_models")
    base_dir.mkdir(parents=True, exist_ok=True)
    if base_dir not in local_path.parents and local_path != base_dir:
        logger.warning(
            f"Model '{model_name}' local_path is '{local_path}', which is outside '{base_dir}/'. "
            "This repo expects all artifacts under ./huggingface_models/<model_name>/."
        )

    local_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading {model_name} using direct file method")
    
    method = model_info.get("method")

    # HF snapshot (entire repo)
    if method == "snapshot":
        try:
            from huggingface_hub import snapshot_download
            logger.info(f"Snapshot downloading {model_name} repo {model_info['repo_id']}")
            snapshot_download(
                repo_id=model_info["repo_id"],
                revision=model_info.get("revision") or "main",
                local_dir=model_info["local_path"],
                local_dir_use_symlinks=False,
                token=hf_token,
                force_download=force,
            )
            logger.info(f"Snapshot complete: {model_name} -> {model_info['local_path']}")
            return True
        except Exception as e:
            logger.error(f"Snapshot download failed for {model_name}: {e}")
            return False

    # HF snapshot with allow_patterns + primary file check
    if method == "snapshot_allow_patterns":
        try:
            from huggingface_hub import snapshot_download

            primary_file = model_info["primary_file"]
            primary_dir = Path(model_info["local_path"]) / Path(primary_file).parent
            primary_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Downloading {model_name} using snapshot_download (allow_patterns)")
            snapshot_download(
                repo_id=model_info["repo_id"],
                revision=model_info.get("revision") or "main",
                allow_patterns=model_info["allow_patterns"],
                local_dir=model_info["local_path"],
                local_dir_use_symlinks=False,
                token=hf_token,
                force_download=force,
            )

            primary_path = Path(model_info["local_path"]) / primary_file
            if primary_path.exists():
                logger.info(f"Successfully downloaded {model_name}, primary file: {primary_file}")
                return True
            logger.error(f"Primary file {primary_file} not found after download")
            return False
        except Exception as e:
            logger.error(f"Error using snapshot_download for {model_name}: {e}")
            return False

    # Download a specific file from a HF repo into local_path/<hf_filename>
    if method == "hf_file":
        try:
            from huggingface_hub import hf_hub_download

            filename = model_info["hf_filename"]
            target_path = local_path / Path(filename).name
            if target_path.exists() and target_path.stat().st_size > 10 * 1024 * 1024 and not force:
                logger.info(
                    f"File already exists: {target_path} ({target_path.stat().st_size / 1024 / 1024:.2f} MB)"
                )
                return True

            logger.info(f"Downloading {filename} from {model_info['repo_id']}")
            downloaded = hf_hub_download(
                repo_id=model_info["repo_id"],
                filename=filename,
                revision=model_info.get("revision") or "main",
                local_dir=str(local_path),
                local_dir_use_symlinks=False,
                token=hf_token,
                force_download=force,
            )
            # hf_hub_download writes to local_dir; ensure expected basename exists
            if not target_path.exists():
                shutil.copy2(downloaded, target_path)
            logger.info(f"Downloaded {model_name}: {target_path}")
            return True
        except Exception as e:
            logger.error(f"HF file download failed for {model_name}: {e}")

            # Optional local fallback (useful when a repo is renamed/removed but a known-good
            # checkpoint exists in the workspace).
            fallback_paths = model_info.get("fallback_local_paths") or []
            for fallback in fallback_paths:
                fallback_path = Path(fallback)
                if fallback_path.exists() and fallback_path.is_file():
                    try:
                        shutil.copy2(fallback_path, target_path)
                        logger.info(f"Recovered {model_name} from local fallback: {fallback_path} -> {target_path}")
                        return True
                    except Exception as copy_e:
                        logger.error(f"Failed copying fallback {fallback_path} for {model_name}: {copy_e}")

            return False

    # DINOv3: git clone code + download .pth weights from HF
    if method == "git_and_hf_file":
        try:
            from huggingface_hub import hf_hub_download

            repo_root = Path(model_info["local_path"])
            _ensure_git_checkout(repo_root, model_info["git_url"], model_info.get("git_ref") or "main")

            dest_rel = Path(model_info["weights_dest_relpath"])
            dest_path = repo_root / dest_rel
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            if dest_path.exists() and dest_path.stat().st_size > 50 * 1024 * 1024 and not force:
                logger.info(
                    f"DINOv3 checkpoint already exists: {dest_path} ({dest_path.stat().st_size / 1024 / 1024:.1f} MB)"
                )
                return True

            logger.info(
                f"Downloading DINOv3 weights {model_info['weights_filename']} from {model_info['weights_repo_id']}"
            )
            downloaded = hf_hub_download(
                repo_id=model_info["weights_repo_id"],
                filename=model_info["weights_filename"],
                revision=model_info.get("weights_revision") or "main",
                token=hf_token,
                force_download=force,
            )
            shutil.copy2(downloaded, dest_path)

            size_mb = dest_path.stat().st_size / 1024 / 1024
            if size_mb < 50:
                raise RuntimeError(f"Downloaded checkpoint looks too small ({size_mb:.1f} MB): {dest_path}")

            logger.info(f"DINOv3 ready: code={repo_root}, weights={dest_path} ({size_mb:.1f} MB)")
            return True
        except Exception as e:
            logger.error(f"DINOv3 download failed for {model_name}: {e}")
            return False
    # Download files list from the repository (simple, non-LFS-safe fallback)
    if "files" in model_info:
        repo_id = model_info["repo_id"]
        success = True
        
        for file in model_info["files"]:
            file_url = f"https://huggingface.co/{repo_id}/resolve/main/{file}"
            file_path = local_path / file
            
            if file_path.exists():
                logger.info(f"File {file} already exists, skipping")
                continue
                
            logger.info(f"Downloading {file} from {repo_id}")
            file_size = download_file(file_url, file_path, hf_token=hf_token)
            
            if not file_size:
                logger.warning(f"Failed to download {file}")
                success = False
                
        return success
    
    logger.error(f"No download method found for {model_name}")
    return False

def download_with_transformers(model_name, hf_token: str | None = None):
    """Legacy helper (kept for compatibility).

    NOTE: Many models in this repo are NOT loaded via `AutoModel` and require
    their original repo artifacts (e.g., Prithvi .pt + code). Prefer hub-based
    downloads via `download_model_direct`.
    """
    logger.error("Transformers-based downloader is deprecated for this repo.")
    return False

def download_model(model_name, hf_token: str | None = None, force: bool = False):
    """
    Download a specific model using the best available method
    """
    if model_name not in MODEL_REGISTRY:
        logger.error(f"Unknown model: {model_name}")
        return False

    resolved = _resolve_alias(model_name)
    if resolved != model_name:
        logger.info(f"Resolved alias: {model_name} -> {resolved}")
        model_name = resolved
        
    logger.info(f"=== Downloading {model_name} ===")
    
    # Use hub-based method aligned with our extractors
    try:
        if download_model_direct(model_name, hf_token=hf_token, force=force):
            logger.info(f"Successfully downloaded {model_name}")
            return True
    except Exception as e:
        logger.error(f"Download failed for {model_name}: {e}")
    
    logger.error(f"All download methods failed for {model_name}")
    return False

def download_all_models(models=None, hf_token: str | None = None, force: bool = False):
    """
    Download all models or specified subset
    """
    models = models or list(MODEL_REGISTRY.keys())
    
    logger.info(f"Downloading {len(models)} models: {', '.join(models)}")
    
    results = {}
    for model_name in models:
        results[model_name] = download_model(model_name, hf_token=hf_token, force=force)
    
    # Print summary
    logger.info("\n=== Download Summary ===")
    success_count = sum(results.values())
    for model_name, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"{model_name}: {status}")
    
    logger.info(f"\nSuccessfully downloaded {success_count}/{len(models)} models")
    return success_count == len(models)

def list_available_models():
    """
    List all available models with their information
    """
    print("\nAvailable Models for Download:")
    print("="*80)
    print(f"{'Model Name':<20} {'Architecture':<15} {'Feature Dim':<12} {'Repository'}")
    print("-"*80)
    
    for name, info in MODEL_REGISTRY.items():
        if info.get("method") == "alias":
            repo = f"alias -> {info.get('alias_of')}"
        else:
            repo = info.get('repo_id', info.get('weights_repo_id', 'N/A'))
        print(f"{name:<20} {info.get('architecture', 'Unknown'):<15} {info.get('feature_dim', 'Unknown'):<12} {repo}")
    
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Download Hugging Face models for feature extraction")
    parser.add_argument("--models", nargs="+", choices=list(MODEL_REGISTRY.keys()) + ["all"],
                      help="Models to download (space-separated list or 'all')")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN"), help="Hugging Face access token for private models")
    parser.add_argument("--force", action="store_true", help="Force re-download even if local files exist")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
        return
    
    if not args.models:
        parser.print_help()
        list_available_models()
        return
    
    if "all" in args.models:
        models = list(MODEL_REGISTRY.keys())
    else:
        models = args.models
    
    download_all_models(models, hf_token=args.token, force=args.force)

if __name__ == "__main__":
    main()