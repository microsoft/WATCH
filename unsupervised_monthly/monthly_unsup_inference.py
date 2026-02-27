#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Monthly unsupervised inference using the legacy ensemble logic.
Loads `unsup_models.pt` and a `scaler_stats.npz`, computes calibrated temporal scores
(reconstruction, forecast, latent novelty) per site, then applies optional month-of-year
seasonal normalization, intra-site grid calibration, and cross-grid softmax. Exports
per-target-month outputs and updates the aggregated `unsup_month_scores_<split>.csv`.

This script is compatible with older naming and export conventions and can restrict
outputs to a single target month `YYYY_MM`.
"""
from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Reuse modules from the unsupervised package
from .datasets import MONTHS, T as TIME_LEN, UnsupervisedSiteDataset, load_scaler_npz
from .utils import normalize_scores, knn_density, robust_zscore

from .monthly_inference import update_aggregated_scores  # reuse aggregator


def _resolve_output_dir(output_dir: str) -> str:
    if not output_dir:
        return str(Path(__file__).resolve().parent)
    if os.path.isabs(output_dir):
        return output_dir
    # Resolve relative paths against the current working directory to avoid nested package paths
    import os as _os
    return _os.path.abspath(output_dir)


def parse_args():
    ap = argparse.ArgumentParser("Monthly unsupervised ensemble inference")
    ap.add_argument("--features_csv", type=str, required=True, help="Features CSV with rows (site_name, month, f0..fF)")
    ap.add_argument("--groundtruth_csv", type=str, default=None)
    ap.add_argument("--split_col", type=str, default="split")
    ap.add_argument("--split", type=str, default="all", choices=["train","val","test","all","all_looted"])  # for deployment
    ap.add_argument("--scaler_path", type=str, required=True, help="Path to scaler stats npz (reused from training)")
    ap.add_argument("--trained_model", type=str, required=True, help="Path to checkpoint unsup_models.pt")
    ap.add_argument("--output_dir", type=str, default=None, help="Defaults to directory of the trained model if not provided")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--merge", action="store_true", help="After updating aggregated matrix, remove the per-month CSV")

    # Target month
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--month", type=int, required=True)

    # Scoring ensemble & calibrations (match unsupervised)
    ap.add_argument("--k_density", type=int, default=5)
    ap.add_argument("--alpha_rec", type=float, default=0.6)
    ap.add_argument("--alpha_fore", type=float, default=0.3)
    ap.add_argument("--alpha_novel", type=float, default=0.4)
    ap.add_argument("--rw_window", type=int, default=12)
    ap.add_argument("--global_spike_q", type=float, default=0.98)
    ap.add_argument("--calendar_population", type=str, default="preserved", choices=["train","preserved","looted"])
    ap.add_argument("--no_calendar_calibration", action="store_true")
    # Robust z control: allow negative values instead of positive-part clipping
    ap.add_argument("--negative_z_ok", action="store_true", help="Allow negative robust z (disable positive-only clipping)")

    # Adaptation flags
    ap.add_argument("--intra_site_grid_calibration", action="store_true")
    ap.add_argument("--month_of_year_seasonal", action="store_true")
    ap.add_argument("--enable_cross_grid", action="store_true")
    ap.add_argument("--w_fuse_temporal", type=float, default=1.0)
    ap.add_argument("--w_fuse_cgz", type=float, default=1.0)
    ap.add_argument("--w_fuse_cgs", type=float, default=0.5)
    ap.add_argument("--output_components", action="store_true")

    # Normalization flags (feature scaler control)
    ap.add_argument("--robust_scaler", action="store_true")
    ap.add_argument("--disable_per_month_feature_norm", action="store_true")

    # Probability normalization of fused scores
    ap.add_argument("--score_norm_method", type=str, default="sigmoid", choices=["none","minmax","sigmoid","softmax"])
    ap.add_argument("--score_norm_temperature", type=float, default=1.0)
    ap.add_argument("--export_probabilities", action="store_true", help="Write probability for the target month (from per-site normalized vector)")

    return ap.parse_args()


def export_single_month_unsup(args):
    out_dir = _resolve_output_dir(args.output_dir) if args.output_dir else os.path.dirname(os.path.abspath(args.trained_model))
    os.makedirs(out_dir, exist_ok=True)
    month_str = f"{args.year}_{args.month:02d}"
    t_index = MONTHS.index(month_str)

    # Preload models and calendar stats by instantiating compute_calendar_stats path
    import torch
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # Dummy args for calendar stats (reuse fields needed)
    class _A:
        pass
    _a = _A()
    # Map required fields from args
    for k in ["features_csv","groundtruth_csv","scaler_path","split_col","disable_per_month_feature_norm","robust_scaler","k_density","alpha_rec","alpha_fore","alpha_novel","rw_window","global_spike_q","calendar_population"]:
        setattr(_a, k, getattr(args, k))
    # Build minimal modules by loading checkpoint via main_train_unsupervised logic
    from .models import MaskedAutoencoder, NextMonthForecaster, TemporalTransformer
    ckpt = torch.load(args.trained_model, map_location=device)
    input_dim = ckpt["input_dim"]; d_model = ckpt.get("d_model", 256)
    depth = ckpt.get("depth", 4)
    nhead = ckpt.get("nhead", 8)
    ff = ckpt.get("ff", 512)
    dropout = ckpt.get("dropout", 0.1)
    mask_ratio = ckpt.get("mask_ratio", 0.3)
    mae = MaskedAutoencoder(input_dim, d_model=d_model, depth=depth, nhead=nhead, ff=ff, dropout=dropout, mask_ratio=mask_ratio).to(device); mae.load_state_dict(ckpt["mae"]); mae.eval()
    fore = NextMonthForecaster(input_dim, d_model=d_model, depth=max(1, depth-1), nhead=nhead, ff=ff, dropout=dropout).to(device); fore.load_state_dict(ckpt["fore"]); fore.eval()
    trunk = TemporalTransformer(input_dim, d_model=d_model, nhead=nhead, num_layers=depth, dim_feedforward=ff, dropout=dropout).to(device); trunk.load_state_dict(ckpt["trunk"]); trunk.eval()
    proj  = torch.nn.Sequential(torch.nn.Linear(d_model, 128), torch.nn.ReLU(), torch.nn.Linear(128, 128)).to(device)
    proj.load_state_dict(ckpt["proj"]); proj.eval()

    # Compute calendar stats over selected population if requested
    def compute_calendar_stats(population: str):
        ds = UnsupervisedSiteDataset(args.features_csv, args.groundtruth_csv, split="train", scaler_path=args.scaler_path,
                                     split_col=args.split_col, per_month_feature_norm=not args.disable_per_month_feature_norm,
                                     robust=args.robust_scaler)
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
        rec_all, fore_all, nov_all = [], [], []
        for x, _, pres in tqdm(loader, desc=f"calendar stats [{population}]"):
            pval = pres if not hasattr(pres, 'item') else pres.item()
            if population == "preserved" and pval != 1:
                continue
            if population == "looted" and pval != 0:
                continue
            xx = x[0].to(device)
            with __import__('torch').no_grad():
                _, recon, _ = mae(xx.unsqueeze(0))
                rec_err = ((recon - xx.unsqueeze(0))**2).mean(dim=-1).squeeze(0).cpu().numpy()
                _, pred = fore(xx.unsqueeze(0))
                TT = rec_err.shape[0]
                err_t = ((pred - xx.unsqueeze(0)[:, 1:, :])**2).mean(dim=-1).squeeze(0)
                fore_err = __import__('torch').zeros(TT, device=xx.device)
                if err_t.numel() > 0:
                    fore_err[:-1] = err_t
                    fore_err[-1] = err_t[-1]
                fore_err = fore_err.cpu().numpy()
                z = trunk(xx.unsqueeze(0)).squeeze(0)
                dens = knn_density(z, k=args.k_density).cpu().numpy()
                nov = (dens - dens.min()) / (dens.max() - dens.min() + 1e-8)
                nov = 1.0 - nov
            rec_all.append(rec_err); fore_all.append(fore_err); nov_all.append(nov)
        if len(rec_all) == 0:
            return None
        rec_all = np.stack(rec_all, axis=0)
        fore_all = np.stack(fore_all, axis=0)
        nov_all = np.stack(nov_all, axis=0)
        return {
            "rec_mean": rec_all.mean(axis=0),
            "rec_std":  rec_all.std(axis=0) + 1e-8,
            "fore_mean": fore_all.mean(axis=0),
            "fore_std":  fore_all.std(axis=0) + 1e-8,
            "nov_mean":  nov_all.mean(axis=0),
            "nov_std":   nov_all.std(axis=0) + 1e-8,
        }

    calendar_stats = None
    if not args.no_calendar_calibration:
        stats = compute_calendar_stats(args.calendar_population)
        if stats is None and args.calendar_population != "train":
            stats = compute_calendar_stats("train")
        calendar_stats = stats

    # Build dataset for selected split
    ds = UnsupervisedSiteDataset(args.features_csv, args.groundtruth_csv, split=args.split, scaler_path=args.scaler_path,
                                 split_col=args.split_col, per_month_feature_norm=not args.disable_per_month_feature_norm, robust=args.robust_scaler)

    # First pass: compute temporal base scores per site for all months, then select t_index
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    site_names: list[str] = []
    base_temporal = []  # (N_sites, T)
    for x, s, _ in tqdm(loader, desc=f"scores ({args.split})"):
        xx = x[0].to(device)
        with __import__('torch').no_grad():
            # Reconstruction error per month
            _, recon, _ = mae(xx.unsqueeze(0))
            rec_err = ((recon - xx.unsqueeze(0))**2).mean(dim=-1).squeeze(0).cpu().numpy()
            TT = rec_err.shape[0]
            # Forecast error per month (align T-1 -> first T-1 months, copy last)
            _, pred = fore(xx.unsqueeze(0))
            err_t = ((pred - xx.unsqueeze(0)[:, 1:, :])**2).mean(dim=-1).squeeze(0)
            import torch as _t
            fore_err = _t.zeros(TT, device=xx.device)
            if err_t.numel() > 0:
                fore_err[:-1] = err_t
                fore_err[-1] = err_t[-1]
            fore_err = fore_err.cpu().numpy()
            # Latent novelty via kNN density (lower density -> anomaly)
            z = trunk(xx.unsqueeze(0)).squeeze(0)
            dens = knn_density(z, k=args.k_density)
            dens = dens.cpu().numpy()
            novelty = (dens - dens.min()) / (dens.max() - dens.min() + 1e-8)
            novelty = 1.0 - novelty

        # Calendar/month calibration: per-month z-norm using training stats if available
        if calendar_stats is not None:
            rec_err = (rec_err - calendar_stats["rec_mean"]) / calendar_stats["rec_std"]
            fore_err = (fore_err - calendar_stats["fore_mean"]) / calendar_stats["fore_std"]
            novelty = (novelty - calendar_stats["nov_mean"]) / calendar_stats["nov_std"]
            rec_err = np.maximum(rec_err, 0.0)
            fore_err = np.maximum(fore_err, 0.0)
            novelty = np.maximum(novelty, 0.0)

        # Rolling-window robust normalization (site-local)
        rec_rw = robust_zscore(rec_err, window=args.rw_window, positive_only=(not args.negative_z_ok))
        fore_rw = robust_zscore(fore_err, window=args.rw_window, positive_only=(not args.negative_z_ok))
        nov_rw = robust_zscore(novelty, window=args.rw_window, positive_only=(not args.negative_z_ok))

        # Global spike downweighting
        atten = 1.0
        if calendar_stats is not None:
            glob = calendar_stats["rec_mean"] + calendar_stats["fore_mean"] + calendar_stats["nov_mean"]
            thr = np.quantile(glob, args.global_spike_q)
            spike_mask = (glob >= thr).astype(np.float32)
            atten_arr = 1.0 - 0.5 * spike_mask
            # apply elementwise attenuation
            scores = atten_arr * (args.alpha_rec * rec_rw + args.alpha_fore * fore_rw + args.alpha_novel * nov_rw)
        else:
            scores = (args.alpha_rec * rec_rw + args.alpha_fore * fore_rw + args.alpha_novel * nov_rw)
        site_names.append(s[0])
        base_temporal.append(scores)
    base_temporal = np.stack(base_temporal, axis=0) if len(base_temporal) > 0 else np.zeros((0, TIME_LEN), dtype=np.float32)

    # Optional month-of-year seasonal normalization across sites for the target calendar month
    if args.month_of_year_seasonal and base_temporal.shape[0] > 0:
        moy = t_index % 12
        idxs = [i for i in range(TIME_LEN) if i % 12 == moy]
        global_vals = base_temporal[:, idxs].reshape(-1)
        g_med = np.median(global_vals)
        g_mad = np.median(np.abs(global_vals - g_med)) + 1e-8
        base_temporal[:, t_index] = np.maximum(0.0, (base_temporal[:, t_index] - g_med) / g_mad)

    # Intra-site grid calibration at month t
    cgz_t = np.zeros((len(site_names),), dtype=np.float32)
    if args.intra_site_grid_calibration and base_temporal.shape[0] > 0:
        base_map: dict[str, list[int]] = {}
        for idx, name in enumerate(site_names):
            base = name.split("_grid_")[0] if "_grid_" in name else name
            base_map.setdefault(base, []).append(idx)
        for base, idxs in base_map.items():
            sub = base_temporal[idxs, t_index]
            if sub.size < 2:
                continue
            med = np.median(sub)
            mad = np.median(np.abs(sub - med)) + 1e-8
            z = (sub - med) / mad
            z[z < 0] = 0.0
            cgz_t[idxs] = z.astype(np.float32)

    # Cross-grid softmax at month t
    cgs_t = np.zeros((len(site_names),), dtype=np.float32)
    if args.enable_cross_grid and base_temporal.shape[0] > 0:
        base_map: dict[str, list[int]] = {}
        for idx, name in enumerate(site_names):
            base = name.split("_grid_")[0] if "_grid_" in name else name
            base_map.setdefault(base, []).append(idx)
        for base, idxs in base_map.items():
            sub = base_temporal[idxs, t_index]
            if sub.size == 1:
                cgs_t[idxs] = 1.0
                continue
            mmax = np.max(sub)
            ex = np.exp(sub - mmax)
            den = np.sum(ex) + 1e-8
            cgs_t[idxs] = (ex / den).astype(np.float32)

    # Fused score at month t
    fused_t = (
        args.w_fuse_temporal * base_temporal[:, t_index].astype(np.float32) +
        args.w_fuse_cgz * cgz_t.astype(np.float32) +
        args.w_fuse_cgs * cgs_t.astype(np.float32)
    ).astype(np.float32)

    # Optional probability normalization based on the full per-site vector
    prob_t = None
    if args.export_probabilities:
        probs = []
        for i in range(base_temporal.shape[0]):
            p = normalize_scores(base_temporal[i], method=args.score_norm_method, temperature=args.score_norm_temperature)
            probs.append(float(p[t_index]))
        prob_t = np.array(probs, dtype=np.float32)

    # Build output dataframe
    rows = []
    for i, name in enumerate(site_names):
        if args.output_components:
            rows.append([name, month_str, float(base_temporal[i, t_index]), float(cgz_t[i]), float(cgs_t[i]), float(fused_t[i])])
        else:
            val = float(prob_t[i]) if prob_t is not None else float(fused_t[i])
            rows.append([name, month_str, val])
    cols = ["site_name","month","t_score","cgz_score","cgs_score","fused_score"] if args.output_components else ["site_name","month","score"]
    df_out = pd.DataFrame(rows, columns=cols)

    # Save per-month CSV
    # Add mode column and use explicit canonical mode name in filename
    if 'mode' not in df_out.columns:
        df_out = df_out.copy()
        df_out.insert(1, 'mode', 'learned_unsupervised')
    pm_path = Path(out_dir) / f"monthly_learned_unsupervised_{args.year}_{args.month:02d}_{args.split}.csv"
    df_out.to_csv(pm_path, index=False)

    # Update aggregated matrix (stores the selected metric per site under month name)
    df_for_agg = df_out[["site_name","month"]].copy()
    if args.output_components:
        # prefer fused for aggregation
        df_for_agg["score"] = df_out["fused_score"].astype(np.float32)
    else:
        df_for_agg["score"] = df_out["score"].astype(np.float32)
    df_for_agg['mode'] = 'learned_unsupervised'
    update_aggregated_scores(out_dir, args.split, month_str, df_for_agg)
    # Optionally remove per-month file after aggregation
    if getattr(args, 'merge', False):
        try:
            Path(pm_path).unlink(missing_ok=True)
        except Exception as _e:
            print(f"[warn] could not remove per-month file {pm_path}: {_e}")
    print(f"[done] Exported {len(df_out)} rows for {month_str} to {out_dir} (aggregated)")


def main():
    args = parse_args()
    export_single_month_unsup(args)


if __name__ == "__main__":
    main()
