# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Spectral indices and texture feature extraction from satellite imagery."""

import numpy as np
from skimage.filters import sobel
from skimage.measure import shannon_entropy
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy.stats import skew, kurtosis, entropy

# alias to match function usage below
from skimage.feature import graycomatrix as greycomatrix, graycoprops as greycoprops

# ========= Basic band math (assumes [R, G, B, NIR]) =========
def compute_ndvi(img):
    nir = img[3]; red = img[0]
    return (nir - red) / (nir + red + 1e-6)

def compute_ndwi(img):
    nir = img[3]; green = img[1]
    return (green - nir) / (green + nir + 1e-6)

def compute_bsi(img):
    # Bare Soil Index (BSI) using RGB+NIR proxy
    # BSI = ((R + NIR) - (G + B)) / ((R + NIR) + (G + B))
    r, g, b, n = img[0], img[1], img[2], img[3]
    num = (r + n) - (g + b)
    den = (r + n) + (g + b) + 1e-6
    return num / den

def compute_nbr(img):
    # NBR (commonly SWIR/NIR), approximate here with NIR vs RED
    # NBR ~ (NIR - RED) / (NIR + RED) -> effectively NDVI proxy
    nir = img[3]; red = img[0]
    return (nir - red) / (nir + red + 1e-6)

def compute_savi(img, L=0.5):
    nir = img[3]; red = img[0]
    return (1 + L) * (nir - red) / (nir + red + L + 1e-6)


# ========= Safe helpers =========
def compute_sobel_strength(band):
    return float(np.nanmean(sobel(band)))

def compute_entropy_img(band):
    try:
        return float(shannon_entropy(band))
    except Exception:
        return 0.0

def _percentile(a, p):
    try:
        return float(np.percentile(a, p))
    except Exception:
        return 0.0

def _safe_skew(a):
    try:
        return float(skew(a, axis=None, nan_policy='omit'))
    except Exception:
        return 0.0

def _safe_kurt(a):
    try:
        return float(kurtosis(a, axis=None, nan_policy='omit'))
    except Exception:
        return 0.0

def _safe_ratio(curr, prev_mean):
    return float(curr / (prev_mean + 1e-6)) if np.isfinite(prev_mean) else 0.0


# ========= Textures =========
def _glcm_feats_from_band(band, patch_size=128):
    """
    Compute GLCM (contrast, homogeneity, energy, entropy) on a center patch.
    """
    try:
        h, w = band.shape
        if h < 8 or w < 8:
            return 0.0, 0.0, 0.0, 0.0
        y0 = max(0, h//2 - patch_size//2)
        x0 = max(0, w//2 - patch_size//2)
        patch = band[y0:y0+patch_size, x0:x0+patch_size]
        if patch.size == 0:
            patch = band

        vmin, vmax = np.nanmin(patch), np.nanmax(patch)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            return 0.0, 0.0, 0.0, 0.0

        patch_u8 = np.clip(((patch - vmin) / (vmax - vmin + 1e-6) * 255).astype(np.uint8), 0, 255)
        glcm = greycomatrix(patch_u8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = float(greycoprops(glcm, 'contrast')[0, 0])
        homog    = float(greycoprops(glcm, 'homogeneity')[0, 0])
        energy   = float(greycoprops(glcm, 'energy')[0, 0])
        g = glcm.squeeze().astype(np.float64)
        glcm_ent = float(-np.sum(g * np.log(g + 1e-12)))
        return contrast, homog, energy, glcm_ent
    except Exception:
        return 0.0, 0.0, 0.0, 0.0

def _lbp_feats(band, P=8, R=1):
    """
    Local Binary Pattern (uniform). Returns: mean LBP value, entropy of LBP hist, uniform proportion.
    """
    try:
        vmin, vmax = np.nanmin(band), np.nanmax(band)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            return 0.0, 0.0, 0.0
        band_norm = (band - vmin) / (vmax - vmin + 1e-6)
        lbp = local_binary_pattern(band_norm, P=P, R=R, method='uniform')
        # bins: P+2 for uniform
        bins = int(P + 2)
        hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins), density=True)
        hist = np.clip(hist, 1e-12, None)
        lbp_entropy = float(-(hist * np.log(hist)).sum())
        uniform_prop = float(hist[:-1].sum())  # last bin often for non-uniform
        lbp_mean = float(np.mean(lbp))
        return lbp_mean, lbp_entropy, uniform_prop
    except Exception:
        return 0.0, 0.0, 0.0


# ========= Main extractor =========
def extract_features(time_series):
    """
    time_series: np.ndarray (T, 4, H, W), band order [R, G, B, NIR]
    Returns: np.ndarray (T, F) float32 with original 12 + extended features.
    """
    features = []

    prev_nir_mean = None
    prev_ndvi_mean = None
    prev_nir_std = None
    prev_ndvi_std = None
    prev_band_means = None  # for CVA: [R,G,B,NIR] means previous month
    persistence_counter = 0

    # rolling buffers for 3-month stats
    last3_nir_mean = []
    last3_ndvi_mean = []
    last3_ndvi_means_for_ratio = []
    last3_nir_means_for_ratio = []

    for t in range(time_series.shape[0]):
        img = time_series[t]  # (4, H, W)
        R, G, B, N = img[0], img[1], img[2], img[3]

        # Spectral indices
        ndvi = compute_ndvi(img)
        ndwi = compute_ndwi(img)
        bsi  = compute_bsi(img)
        nbr  = compute_nbr(img)
        savi = compute_savi(img)

        # Means/stats
        nir_mean = float(np.nanmean(N))
        nir_std  = float(np.nanstd(N))
        ndvi_mean = float(np.nanmean(ndvi))
        ndvi_std  = float(np.nanstd(ndvi))
        ndwi_mean = float(np.nanmean(ndwi))
        ndwi_std  = float(np.nanstd(ndwi))
        all_mean  = float(np.nanmean(img))
        all_std   = float(np.nanstd(img))

        # Temporal deltas (existing)
        nir_diff  = 0.0 if prev_nir_mean is None else float(nir_mean - prev_nir_mean)
        ndvi_diff = 0.0 if prev_ndvi_mean is None else float(ndvi_mean - prev_ndvi_mean)

        # Distributional stats
        nir_median = float(np.nanmedian(N))
        nir_p10 = _percentile(N, 10); nir_p90 = _percentile(N, 90)
        nir_iqr = float(nir_p90 - _percentile(N, 25))
        nir_sk = _safe_skew(N); nir_ku = _safe_kurt(N)

        ndvi_median = float(np.nanmedian(ndvi))
        ndvi_p10 = _percentile(ndvi, 10); ndvi_p90 = _percentile(ndvi, 90)
        ndvi_iqr = float(ndvi_p90 - _percentile(ndvi, 25))

        # Textures (GLCM on R,G,NIR)
        glcm_R = _glcm_feats_from_band(R)
        glcm_G = _glcm_feats_from_band(G)
        glcm_N = _glcm_feats_from_band(N)  # focus on NIR

        # LBP on NIR
        lbp_mean, lbp_ent, lbp_uniform = _lbp_feats(N)

        # Edge density (NIR)
        sob = sobel(N)
        thr = float(np.nanmean(sob) + 1.0 * np.nanstd(sob))
        edge_density = float(np.nanmean(sob > thr))

        # Δ std (temporal)
        d_nir_std  = 0.0 if prev_nir_std  is None else float(nir_std  - prev_nir_std)
        d_ndvi_std = 0.0 if prev_ndvi_std is None else float(ndvi_std - prev_ndvi_std)

        # Rolling slopes (3-mo)
        last3_nir_mean.append(nir_mean);   last3_ndvi_mean.append(ndvi_mean)
        if len(last3_nir_mean)  > 3: last3_nir_mean.pop(0)
        if len(last3_ndvi_mean) > 3: last3_ndvi_mean.pop(0)

        def _slope(vals):
            if len(vals) < 3:
                return 0.0
            x = np.arange(len(vals), dtype=np.float32)
            try:
                return float(np.polyfit(x, np.array(vals, dtype=np.float32), 1)[0])
            except Exception:
                return 0.0

        nir_mean_slope_3  = _slope(last3_nir_mean)
        ndvi_mean_slope_3 = _slope(last3_ndvi_mean)

        # Pre/current ratios using previous 3-month means
        last3_ndvi_means_for_ratio.append(ndvi_mean)
        last3_nir_means_for_ratio.append(nir_mean)
        if len(last3_ndvi_means_for_ratio) > 3: last3_ndvi_means_for_ratio.pop(0)
        if len(last3_nir_means_for_ratio) > 3:  last3_nir_means_for_ratio.pop(0)
        prev3_ndvi_mean = np.mean(last3_ndvi_means_for_ratio[:-1]) if len(last3_ndvi_means_for_ratio) >= 2 else np.nan
        prev3_nir_mean  = np.mean(last3_nir_means_for_ratio[:-1])  if len(last3_nir_means_for_ratio)  >= 2 else np.nan
        ndvi_precur_ratio = _safe_ratio(ndvi_mean, prev3_ndvi_mean)
        nir_precur_ratio  = _safe_ratio(nir_mean,  prev3_nir_mean)

        # CVA magnitude using band means (R,G,B,N)
        curr_means = np.array([
            float(np.nanmean(R)), float(np.nanmean(G)), float(np.nanmean(B)), nir_mean
        ], dtype=np.float32)
        if prev_band_means is None:
            cva_mag = 0.0
        else:
            delta = curr_means - prev_band_means
            cva_mag = float(np.sqrt(np.sum(delta**2)))
        prev_band_means = curr_means

        # Adaptive change flag & persistence (using 3-mo window on CVA)
        # Simple rule: change if CVA > mean(prev3 CVA) + 2*std(prev3 CVA)
        if t == 0:
            cva_history = []
        if t == 0:
            cva_thr = np.inf
            change_flag = 0.0
        else:
            hist = np.array(cva_history[-3:], dtype=np.float32)
            mu = float(np.nanmean(hist)) if hist.size else 0.0
            sd = float(np.nanstd(hist)) if hist.size else 0.0
            cva_thr = mu + 2.0 * sd
            change_flag = 1.0 if cva_mag > cva_thr and np.isfinite(cva_thr) else 0.0

        cva_history.append(cva_mag)
        if change_flag > 0.5:
            persistence_counter += 1
        else:
            persistence_counter = 0

        # ===== Original 12 features (keep order for compatibility) =====
        base12 = [
            nir_mean, nir_std,
            compute_sobel_strength(N),
            compute_entropy_img(N),
            ndvi_mean, ndvi_std,
            ndwi_mean, ndwi_std,
            all_mean, all_std,
            nir_diff, ndvi_diff,
        ]

        # ===== Extended features =====
        extended = [
            # indices beyond NDVI/NDWI
            float(np.nanmean(bsi)),  float(np.nanstd(bsi)),
            float(np.nanmean(nbr)),  float(np.nanstd(nbr)),
            float(np.nanmean(savi)), float(np.nanstd(savi)),

            # distributions (NIR)
            nir_median, nir_p10, nir_p90, nir_iqr, nir_sk, nir_ku,
            # distributions (NDVI)
            ndvi_median, ndvi_p10, ndvi_p90, ndvi_iqr,

            # textures GLCM for R, G, NIR (contrast, homog, energy, entropy) -> 12 feats
            *glcm_R, *glcm_G, *glcm_N,

            # LBP on NIR
            lbp_mean, lbp_ent, lbp_uniform,

            # edges
            edge_density,

            # temporal std deltas & slopes
            d_nir_std, d_ndvi_std, nir_mean_slope_3, ndvi_mean_slope_3,

            # pre/current ratios
            ndvi_precur_ratio, nir_precur_ratio,

            # CVA + persistence
            cva_mag, float(cva_thr if np.isfinite(cva_thr) else 0.0),
            change_flag, float(persistence_counter),
        ]

        vec = base12 + extended
        features.append(vec)

        # update prevs
        prev_nir_mean = nir_mean
        prev_ndvi_mean = ndvi_mean
        prev_nir_std = nir_std
        prev_ndvi_std = ndvi_std

    return np.asarray(features, dtype=np.float32)
