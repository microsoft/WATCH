# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Dataset for weakly-supervised monthly change detection."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from .month_utils import MONTHS, normalize_month_str

T = len(MONTHS)


def _ensure_month_str(s: pd.Series) -> pd.Series:
    """Normalize values to 'YYYY_MM' strings; invalid values -> NaN."""
    s = s.astype(str).str.replace("-", "_", regex=False)
    s = s.str.extract(r"(\d{4}_\d{2})", expand=False)
    return s


def month_to_idx(mm: str) -> int:
    mm = normalize_month_str(mm)
    if mm not in MONTHS:
        return -1
    y, m = map(int, mm.split("_"))
    return (y - 2017) * 12 + (m - 1)


@dataclass(frozen=True)
class LabelWindow:
    window: int = 1
    smooth_type: str = "gauss"  # "gauss" or "box"


class SiteTimeSeriesDataset(Dataset):
    """Site-level time series dataset for weakly supervised month training.

    Key differences:
    - Month labels are only considered "known" up to `label_end_month` (default 2020_12).
    - By default, we exclude looted sites whose month label is unknown (after cutoff).

    Each item returns:
      feats: (T, F) float32
      looted: int64 {0,1}
      known_idx: int64 in [0..T-1] for known labels else -1
      site_name: str
    """

    def __init__(
        self,
        features_csv: str,
        gt_csv: str,
        split: str = "train",
        split_col: str = "split",
        scaler_path: str | None = None,
        save_fitted_scaler: bool = False,
        include_preserved: bool = True,
        include_unknown_looted: bool = False,
        label_end_month: str = "2020_12",
    ):
        self.split = split
        self.split_col = split_col or "split"
        self.include_preserved = bool(include_preserved)
        self.include_unknown_looted = bool(include_unknown_looted)

        # Load
        self.features = pd.read_csv(features_csv)
        self.gt = pd.read_csv(gt_csv)

        if not {"site_name", "month"}.issubset(self.features.columns):
            raise ValueError("features_csv must contain site_name and month columns")
        if "site_name" not in self.gt.columns:
            raise ValueError("groundtruth_csv must contain site_name")

        # Normalize month strings & keep only our fixed window
        self.features["month"] = _ensure_month_str(self.features["month"])
        self.features = self.features[self.features["month"].isin(MONTHS)].copy()

        if "looted_month" in self.gt.columns:
            self.gt["looted_month"] = _ensure_month_str(self.gt["looted_month"])

        split_col_use = self.split_col
        if split_col_use not in self.gt.columns:
            split_col_use = "split"
            if split_col_use not in self.gt.columns:
                raise ValueError("groundtruth_csv must contain a split column")

        if split in ("train", "val", "test"):
            self.gt = self.gt[self.gt[split_col_use] == split].copy()
        elif split == "all":
            pass
        else:
            raise ValueError(f"Unknown split: {split}")

        if "looted" not in self.gt.columns:
            raise ValueError("groundtruth_csv must contain 'looted' column (0/1)")

        if not self.include_preserved:
            self.gt = self.gt[self.gt["looted"] == 1].copy()

        # Apply label cutoff logic
        end_idx = month_to_idx(label_end_month)
        if end_idx < 0:
            raise ValueError(f"Invalid label_end_month={label_end_month}; expected YYYY_MM")

        def _known_idx_row(r) -> int:
            if int(r.get("looted", 0)) != 1:
                return -1
            lm = r.get("looted_month", "")
            idx = month_to_idx(str(lm))
            if 0 <= idx <= end_idx:
                return idx
            return -1

        self.gt = self.gt.copy()
        self.gt["known_idx"] = self.gt.apply(_known_idx_row, axis=1)

        if not self.include_unknown_looted:
            keep = (self.gt["looted"] == 0) | ((self.gt["looted"] == 1) & (self.gt["known_idx"] >= 0))
            self.gt = self.gt[keep].copy()

        valid_sites = set(self.gt["site_name"].unique())
        self.features = self.features[self.features["site_name"].isin(valid_sites)].copy()

        self.feature_cols = [c for c in self.features.columns if c.startswith("f")]
        if not self.feature_cols:
            raise ValueError("No feature columns (f*) found in features CSV")

        # Fit or load scaler
        self.scaler = StandardScaler()
        self.monthly_mean = None
        self.monthly_scale = None

        if scaler_path and os.path.exists(scaler_path):
            arr = np.load(scaler_path)
            mean_, scale_ = arr.get("mean"), arr.get("scale")
            if mean_ is not None and scale_ is not None and mean_.shape[0] == len(self.feature_cols):
                scale_safe = scale_.astype(np.float64).copy()
                scale_safe[scale_safe == 0.0] = 1.0
                self.scaler.mean_ = mean_.astype(np.float64)
                self.scaler.scale_ = scale_safe
                self.scaler.var_ = (self.scaler.scale_ ** 2).astype(np.float64)
                self.scaler.n_features_in_ = len(self.feature_cols)
            else:
                X = self.features[self.feature_cols].to_numpy(dtype=np.float64)
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                self.scaler.fit(X)

            mm = arr.get("monthly_mean")
            ms = arr.get("monthly_scale")
            if mm is not None and ms is not None and mm.shape == (T, len(self.feature_cols)):
                ms = ms.astype(np.float64)
                ms[ms == 0.0] = 1.0
                self.monthly_mean = mm.astype(np.float64)
                self.monthly_scale = ms
        else:
            X = self.features[self.feature_cols].to_numpy(dtype=np.float64)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            self.scaler.fit(X)

        # Save fitted scaler (train only)
        if save_fitted_scaler and scaler_path and split == "train":
            F = len(self.feature_cols)
            monthly_mean = np.zeros((T, F), dtype=np.float64)
            monthly_scale = np.ones((T, F), dtype=np.float64)
            for mi, mstr in enumerate(MONTHS):
                sub = self.features[self.features["month"] == mstr]
                if len(sub) == 0:
                    monthly_mean[mi, :] = self.scaler.mean_
                    monthly_scale[mi, :] = self.scaler.scale_
                else:
                    X = sub[self.feature_cols].to_numpy(dtype=np.float64)
                    X = np.nan_to_num(X, nan=np.nan, posinf=np.nan, neginf=np.nan)
                    mu = np.nanmean(X, axis=0)
                    sd = np.nanstd(X, axis=0)
                    mu[~np.isfinite(mu)] = self.scaler.mean_[~np.isfinite(mu)]
                    sd[~np.isfinite(sd)] = 1.0
                    sd[sd == 0.0] = 1.0
                    monthly_mean[mi, :] = mu
                    monthly_scale[mi, :] = sd

            np.savez(
                scaler_path,
                mean=self.scaler.mean_.astype(np.float32),
                scale=self.scaler.scale_.astype(np.float32),
                monthly_mean=monthly_mean.astype(np.float32),
                monthly_scale=monthly_scale.astype(np.float32),
            )
            self.monthly_mean = monthly_mean
            self.monthly_scale = monthly_scale

        self.sites = sorted(valid_sites)
        self.gt = self.gt.set_index("site_name")

    def __len__(self) -> int:
        return len(self.sites)

    def __getitem__(self, idx: int):
        site_name = self.sites[idx]
        gtr = self.gt.loc[site_name]
        looted_flag = int(gtr["looted"])
        known_idx = int(gtr.get("known_idx", -1))

        sdf = self.features[self.features["site_name"] == site_name].copy()
        sdf = sdf.set_index("month").reindex(MONTHS)
        feats = sdf[self.feature_cols].to_numpy(dtype=np.float64)

        # Fill missing
        if self.monthly_mean is not None:
            mask = ~np.isfinite(feats)
            if mask.any():
                feats = np.where(mask, self.monthly_mean, feats)
        else:
            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        # Standardize
        if self.monthly_mean is not None and self.monthly_scale is not None:
            scale = self.monthly_scale.copy()
            scale[scale == 0.0] = 1.0
            feats = (feats - self.monthly_mean) / scale
            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        else:
            feats = self.scaler.transform(feats).astype(np.float32)

        feats_t = torch.tensor(feats, dtype=torch.float32)
        looted_t = torch.tensor(looted_flag, dtype=torch.long)
        known_idx_t = torch.tensor(known_idx, dtype=torch.long)
        return feats_t, looted_t, known_idx_t, site_name
