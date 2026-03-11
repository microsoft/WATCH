# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Numerical utilities for the unsupervised pipeline."""

import numpy as np
import torch

def topk_months(scores: np.ndarray, k: int = 3):
    idx = np.argsort(-scores)[:k]
    return idx.tolist()

def softmax_np(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-12)

@torch.no_grad()
def knn_density(z: torch.Tensor, k: int = 5):
    N = z.size(0)
    if N <= k:
        return torch.zeros(N, device=z.device)
    dists = torch.cdist(z, z)
    d_sorted, _ = torch.sort(dists, dim=1)
    knn = d_sorted[:, 1:k+1].mean(dim=1)
    return knn

def robust_zscore(x: np.ndarray, window: int = 12, positive_only: bool = True, eps: float = 1e-8) -> np.ndarray:
    """
    Robust per-time z over a local window using median/MAD, with safe fallbacks:
    - If MAD is ~0 or non-finite, fall back to mean/std over the window.
    - If the window has insufficient finite values, return 0 at that index.
    - Optionally allow negative values (disable positive-part clipping).
    """
    x = np.asarray(x, dtype=np.float32)
    T = x.shape[0]
    z = np.zeros_like(x, dtype=np.float32)
    for t in range(T):
        a = max(0, t - window)
        b = min(T, t + window + 1)
        win = x[a:b]
        # Use only finite values to compute statistics
        wfin = np.isfinite(win)
        if not wfin.any():
            z[t] = 0.0
            continue
        w = win[wfin]
        med = np.median(w)
        mad = np.median(np.abs(w - med))
        use_std = (not np.isfinite(mad)) or (mad < eps)
        if use_std:
            mu = np.mean(w)
            sd = np.std(w)
            if (not np.isfinite(sd)) or (sd < eps):
                z[t] = 0.0
                continue
            val = (x[t] - mu) / (sd + eps)
        else:
            val = (x[t] - med) / (mad + eps)
        z[t] = max(0.0, float(val)) if positive_only else float(val)
    return z

def normalize_scores(scores: np.ndarray, method: str = "none", temperature: float = 1.0, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(scores, dtype=np.float32)
    if x.ndim != 1:
        x = x.reshape(-1)
    if method == "none":
        return x
    if method == "minmax":
        mn, mx = np.nanmin(x), np.nanmax(x)
        denom = (mx - mn)
        if not np.isfinite(denom) or denom < eps:
            return np.zeros_like(x)
        return (x - mn) / (denom + eps)
    if method == "sigmoid":
        mu = np.nanmean(x)
        sd = np.nanstd(x)
        sd = sd if (np.isfinite(sd) and sd > 0) else 1.0
        z = (x - mu) / (sd * max(temperature, eps))
        return 1.0 / (1.0 + np.exp(-z))
    if method == "softmax":
        t = max(temperature, eps)
        y = x / t
        y = y - np.nanmax(y)
        e = np.exp(y)
        den = np.nansum(e)
        if not np.isfinite(den) or den < eps:
            n = x.size
            return np.full_like(x, 1.0 / max(n, 1))
        return e / (den + eps)
    return x
