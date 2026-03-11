# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Numerical utilities for the unsupervised pipeline."""

import numpy as np
import torch
from scipy.spatial.distance import cdist as _scipy_cdist

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


def causal_knn_density(z: np.ndarray, k: int = 5, k_prior: int = None) -> np.ndarray:
    """KNN density using only previous k_prior months as reference (causal).

    For each time step t, computes average distance to the k nearest neighbors
    among the previous k_prior months only. Mirrors the distance baseline trick:
    novelty = how different is the current month from its recent history.
    """
    z = np.asarray(z, dtype=np.float32)
    T = z.shape[0]
    scores = np.zeros(T, dtype=np.float32)
    for t in range(T):
        start = max(0, t - k_prior) if k_prior is not None else 0
        ref = z[start:t]
        if len(ref) < 1:
            continue
        dists = _scipy_cdist(z[t:t+1], ref)[0]
        n_neighbors = min(k, len(ref))
        scores[t] = float(np.mean(np.sort(dists)[:n_neighbors]))
    return scores


def delta_signal(errors: np.ndarray, k_prior: int = 3) -> np.ndarray:
    """Compute change in error relative to rolling median of previous k_prior months.

    Returns max(0, error_t - median(errors[t-k_prior:t])).
    Detects sudden spikes in reconstruction or forecast error, analogous to
    how the distance baseline measures change vs. a causal rolling median.
    """
    errors = np.asarray(errors, dtype=np.float32)
    T = errors.shape[0]
    delta = np.zeros(T, dtype=np.float32)
    for t in range(1, T):
        start = max(0, t - k_prior)
        baseline = float(np.median(errors[start:t]))
        delta[t] = max(0.0, float(errors[t]) - baseline)
    return delta

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

def population_knn_novelty(all_z: np.ndarray, k: int = 10) -> np.ndarray:
    """Cross-site population KNN novelty score.

    For each month t and site s, computes the mean distance to the k nearest
    *other* sites' latent embeddings at the same calendar month.  A site that
    is an outlier relative to the population in month t (but was typical before)
    is likely affected by looting or another structural change.

    Args:
        all_z: Latent embeddings, shape (N_sites, T, d_model), float32.
        k:     Number of neighbours (excluding self) to average.

    Returns:
        Novelty scores of shape (N_sites, T), non-negative.
    """
    N, T_len, d = all_z.shape
    scores = np.zeros((N, T_len), dtype=np.float32)
    n_neighbors = min(k, N - 1)
    for t in range(T_len):
        Z_t = all_z[:, t, :]                    # (N, d)
        dists = _scipy_cdist(Z_t, Z_t, metric="euclidean")  # (N, N)
        np.fill_diagonal(dists, np.inf)
        sorted_d = np.sort(dists, axis=1)       # (N, N) sorted ascending
        scores[:, t] = sorted_d[:, :n_neighbors].mean(axis=1)
    return scores


def cusum_change_score(scores: np.ndarray, drift: float = 0.5) -> np.ndarray:
    """One-sided CUSUM persistent change score.

    Accumulates evidence of sustained anomaly: C[t] = max(0, C[t-1] + scores[t] - drift).
    A single anomalous month barely budges the CUSUM; repeated anomalous months (as
    looting damage persists visually) cause it to grow quickly.  Resets to 0 whenever
    the signal falls back below the drift level, so isolated noise does not accumulate.

    Args:
        scores: Per-month anomaly signal (T,), typically after robust_zscore fusion.
        drift:  Slack parameter — penalises each step to suppress noise.
                Set to ~0.5 * expected_shift for unit-variance inputs (default 0.5).

    Returns:
        Cumulative change score of shape (T,), non-negative.
    """
    scores = np.asarray(scores, dtype=np.float32)
    T = scores.shape[0]
    C = np.zeros(T, dtype=np.float32)
    for t in range(1, T):
        C[t] = max(0.0, C[t - 1] + scores[t] - drift)
    return C


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
