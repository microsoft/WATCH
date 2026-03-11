# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Training and scoring entry point for unsupervised monthly change detection."""

import os
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import UnsupervisedSiteDataset, MONTHS, T as TIME_LEN
from .models import MaskedAutoencoder, NextMonthForecaster, TemporalTransformer
from .criteria import BarlowTwinsLoss, TemporalOrderLoss
from .utils import knn_density, robust_zscore, normalize_scores


def _resolve_output_dir(output_dir: str) -> str:
    """Resolve a potentially relative output directory against the package directory.

    Args:
        output_dir: Absolute or relative path for model outputs.

    Returns:
        Absolute path string.
    """
    # Resolve relative paths against the package directory to avoid nested CWD resolutions
    if os.path.isabs(output_dir):
        return output_dir
    pkg_dir = Path(__file__).resolve().parent
    return str((pkg_dir / output_dir).resolve())


def make_loader(ds, batch, workers, pin_memory=True, persistent_workers=False, prefetch_factor=2, drop_last=True):
    """Create a DataLoader with sensible defaults.

    Args:
        ds: PyTorch Dataset to wrap.
        batch: Batch size.
        workers: Number of data-loading worker processes.
        pin_memory: Whether to pin memory for faster GPU transfers.
        persistent_workers: Whether to keep workers alive between epochs.
        prefetch_factor: Number of batches to prefetch per worker.
        drop_last: Whether to drop the last incomplete batch.

    Returns:
        Configured DataLoader instance.
    """
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if workers > 0 else False,
        prefetch_factor=prefetch_factor if workers > 0 else None,
        drop_last=drop_last,
    )


def compute_and_save_scaler(args):
    """Compute and save feature normalization statistics to an .npz file.

    Computes global and per-month mean/std and median/MAD from the training
    split of the features CSV.  Skips recomputation if the file already exists
    unless ``--overwrite_scaler`` is set.

    Args:
        args: Parsed command-line arguments (must include ``scaler_path``,
            ``features_csv``, ``groundtruth_csv``, ``split_col``,
            ``robust_scaler``, ``overwrite_scaler``, and
            ``disable_per_month_feature_norm``).
    """
    if args.scaler_path is None:
        return
    if os.path.exists(args.scaler_path) and not args.overwrite_scaler:
        print(f"[Scaler] Existing scaler found at {args.scaler_path}; skipping recompute (use --overwrite_scaler to force).")
        return
    import pandas as pd
    df = pd.read_csv(args.features_csv)
    # filter to train rows using ground truth split if available
    if args.groundtruth_csv and os.path.exists(args.groundtruth_csv):
        try:
            gdf = pd.read_csv(args.groundtruth_csv)
            scol = args.split_col if args.split_col in gdf.columns else 'split'
            if scol in gdf.columns:
                train_sites = set(gdf[gdf[scol] == 'train']["site_name"].unique().tolist())
                df = df[df["site_name"].isin(train_sites)]
        except Exception as e:
            print(f"[Scaler] Warning: could not filter by train split due to: {e}")
    feat_cols = sorted([c for c in df.columns if c.startswith('f') and c[1:].isdigit()], key=lambda x: int(x[1:]))
    if len(feat_cols) == 0:
        print("[Scaler] No feature columns found; aborting scaler computation.")
        return
    F = len(feat_cols)
    arr = df[feat_cols].to_numpy(dtype=np.float32, copy=True)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    std[~np.isfinite(std) | (std == 0)] = 1.0
    feat_median = np.nanmedian(arr, axis=0)
    feat_mad = np.nanmedian(np.abs(arr - feat_median), axis=0)
    feat_mad[~np.isfinite(feat_mad) | (feat_mad == 0)] = 1.0

    # Per-calendar-month stats
    months_all = []
    values_all = []
    for _, r in df.iterrows():
        try:
            y_str, m_str = str(r['month']).split('_')
            y, m = int(y_str), int(m_str)
            t = (y - 2017) * 12 + (m - 1)
            if 0 <= t < TIME_LEN:
                months_all.append(t)
                values_all.append(r[feat_cols].to_numpy(dtype=np.float32, copy=False))
        except Exception:
            continue
    month_mean = month_std = month_median = month_mad = None
    if len(values_all) == 0:
        print("[Scaler] Warning: no month-aligned values collected for month stats; skipping month-wise stats.")
    else:
        months_all = np.array(months_all, dtype=np.int32)
        values_all = np.stack(values_all, axis=0)
        month_mean = np.zeros((TIME_LEN, F), dtype=np.float32)
        month_std  = np.ones((TIME_LEN, F), dtype=np.float32)
        month_median = np.zeros((TIME_LEN, F), dtype=np.float32)
        month_mad    = np.ones((TIME_LEN, F), dtype=np.float32)
        for t in range(TIME_LEN):
            mask = months_all == t
            if mask.any():
                vals = values_all[mask]
                month_mean[t] = np.nanmean(vals, axis=0)
                ms = np.nanstd(vals, axis=0)
                ms[~np.isfinite(ms) | (ms == 0)] = 1.0
                month_std[t] = ms
                md = np.nanmedian(vals, axis=0)
                mmad = np.nanmedian(np.abs(vals - md), axis=0)
                mmad[~np.isfinite(mmad) | (mmad == 0)] = 1.0
                month_median[t] = md
                month_mad[t] = mmad
            else:
                month_mean[t] = mean
                month_std[t] = std
                month_median[t] = feat_median
                month_mad[t] = feat_mad

    os.makedirs(os.path.dirname(args.scaler_path), exist_ok=True) if os.path.dirname(args.scaler_path) else None
    save_dict = dict(
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        feat_median=feat_median.astype(np.float32),
        feat_mad=feat_mad.astype(np.float32),
        feature_dim=np.int32(F),
        robust=np.int32(1 if args.robust_scaler else 0),
        per_month_enabled=np.int32(0 if args.disable_per_month_feature_norm else 1),
    )
    if month_mean is not None:
        save_dict.update(
            month_mean=month_mean.astype(np.float32),
            month_std=month_std.astype(np.float32),
            month_median=month_median.astype(np.float32),
            month_mad=month_mad.astype(np.float32),
        )
    np.savez(args.scaler_path, **save_dict)
    print(f"[Scaler] Saved scaler statistics to {args.scaler_path}")


def train(args):
    """Run the unsupervised training loop.

    Trains a masked autoencoder, next-month forecaster, temporal transformer,
    and Barlow Twins projection head jointly.  The best checkpoint (by training
    loss) is saved to ``args.output_dir``.

    Args:
        args: Parsed command-line arguments.
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    compute_and_save_scaler(args)

    train_ds = UnsupervisedSiteDataset(args.features_csv, args.groundtruth_csv, split="train", scaler_path=args.scaler_path, split_col=args.split_col,
                                       per_month_feature_norm=not args.disable_per_month_feature_norm, robust=args.robust_scaler)
    val_ds   = UnsupervisedSiteDataset(args.features_csv, args.groundtruth_csv, split="val",   scaler_path=args.scaler_path, split_col=args.split_col,
                                       per_month_feature_norm=not args.disable_per_month_feature_norm, robust=args.robust_scaler)

    input_dim = train_ds.F

    mae = MaskedAutoencoder(input_dim, d_model=args.d_model, depth=args.depth, nhead=args.nhead, ff=args.ff, dropout=args.dropout, mask_ratio=args.mask_ratio).to(device)
    fore = NextMonthForecaster(input_dim, d_model=args.d_model, depth=max(1, args.depth-1), nhead=args.nhead, ff=args.ff, dropout=args.dropout).to(device)
    trunk = TemporalTransformer(input_dim, d_model=args.d_model, nhead=args.nhead, num_layers=args.depth, dim_feedforward=args.ff, dropout=args.dropout).to(device)
    proj_head = torch.nn.Sequential(torch.nn.Linear(args.d_model, args.proj_dim), torch.nn.ReLU(), torch.nn.Linear(args.proj_dim, args.proj_dim)).to(device)

    opt = optim.AdamW(list(mae.parameters()) + list(fore.parameters()) + list(trunk.parameters()) + list(proj_head.parameters()), lr=args.lr, weight_decay=args.wd)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.eta_min)

    bt = BarlowTwinsLoss(lambda_offdiag=args.bt_lambda)
    tol = TemporalOrderLoss(temperature=args.temp_tau)

    train_loader = make_loader(train_ds, args.batch_size, args.num_workers, pin_memory=not args.no_pin_memory, persistent_workers=not args.no_persistent_workers, drop_last=True)
    val_loader = None
    if len(val_ds) > 0:
        val_loader = make_loader(val_ds, args.batch_size, args.num_workers, pin_memory=not args.no_pin_memory, persistent_workers=False, drop_last=False)

    out_dir = _resolve_output_dir(args.output_dir)

    def compute_val_objective():
        if val_loader is None:
            return float("inf")
        mae.eval(); fore.eval(); trunk.eval(); proj_head.eval()
        total = 0.0; count = 0
        with torch.no_grad():
            for x, _, _ in val_loader:
                x = x.to(device)
                l_mae, _, _ = mae(x)
                l_fore, _ = fore(x)
                z = trunk(x)
                l_tol = tol(z)
                noise_std = getattr(args, "bt_noise_std", 0.01)
                zp = z.mean(dim=1)
                z1 = proj_head(zp + noise_std * torch.randn_like(zp))
                z2 = proj_head(zp + noise_std * torch.randn_like(zp))
                l_bt = bt(z1, z2)
                def safe(v):
                    try:
                        finite = torch.isfinite(v).all().item()
                    except Exception:
                        finite = True
                    return v if finite else torch.tensor(0.0, device=device, requires_grad=True)
                l_mae = safe(l_mae); l_fore = safe(l_fore); l_tol = safe(l_tol); l_bt = safe(l_bt)
                loss = args.w_mae * l_mae + args.w_fore * l_fore + args.w_tol * l_tol + args.w_bt * l_bt
                if torch.isfinite(loss):
                    total += float(loss.detach().cpu()); count += 1
        return (total / count) if count > 0 else float("inf")

    best_metric = float("inf"); patience = args.patience
    use_val = (val_loader is not None)
    for epoch in range(args.epochs):
        mae.train(); fore.train(); trunk.train(); proj_head.train();
        running = 0.0
        finite_batches = 0
        for x, _, _ in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x = x.to(device)
            l_mae, recon, mask = mae(x)
            l_fore, pred = fore(x)
            z = trunk(x)
            zp = z.mean(dim=1)
            l_tol = tol(z)
            noise_std = getattr(args, "bt_noise_std", 0.01)
            z1 = proj_head(zp + noise_std * torch.randn_like(zp))
            z2 = proj_head(zp + noise_std * torch.randn_like(zp))
            l_bt = bt(z1, z2)
            def safe(v):
                try:
                    finite = torch.isfinite(v).all().item()
                except Exception:
                    finite = True
                return v if finite else torch.tensor(0.0, device=device, requires_grad=True)
            l_mae = safe(l_mae); l_fore = safe(l_fore); l_tol = safe(l_tol); l_bt = safe(l_bt)
            loss = args.w_mae * l_mae + args.w_fore * l_fore + args.w_tol * l_tol + args.w_bt * l_bt
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(mae.parameters()) + list(fore.parameters()) + list(trunk.parameters()) + list(proj_head.parameters()), args.grad_clip)
            opt.step()
            running += float(loss.detach().cpu())
            finite_batches += 1
        avg = (running / finite_batches) if finite_batches > 0 else float("inf")
        val_obj = compute_val_objective() if use_val else float("inf")
        if np.isfinite(val_obj):
            print(f"train_loss={avg:.4f} | val_loss={val_obj:.4f}")
        else:
            print(f"train_loss={avg:.4f}")
        if scheduler is not None:
            scheduler.step()
        metric = avg
        if np.isfinite(metric) and (metric + 1e-6 < best_metric):
            best_metric = metric; patience = args.patience
            os.makedirs(out_dir, exist_ok=True)
            torch.save({
                "mae": mae.state_dict(),
                "fore": fore.state_dict(),
                "trunk": trunk.state_dict(),
                "proj": proj_head.state_dict(),
                "input_dim": input_dim,
                "d_model": args.d_model,
                "depth": args.depth,
                "nhead": args.nhead,
                "ff": args.ff,
                "dropout": args.dropout,
                "mask_ratio": args.mask_ratio,
            }, os.path.join(out_dir, "unsup_models.pt"))
            print(f"✓ saved (best on train) -> {os.path.join(out_dir, 'unsup_models.pt')}")
        else:
            if np.isfinite(metric):
                patience -= 1
            if patience <= 0:
                print("early stopping")
                break


def export_month_scores(args):
    """Load a trained checkpoint and export per-site monthly anomaly scores.

    For each site the reconstruction error, forecast error, and novelty
    (KNN density) are fused into a single score per month and written to a
    CSV file.

    Args:
        args: Parsed command-line arguments.
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = _resolve_output_dir(args.output_dir)
    ckpt_path = args.trained_model if getattr(args, "trained_model", None) else os.path.join(out_dir, "unsup_models.pt")
    if not os.path.exists(ckpt_path):
        print(f"[WARN] Checkpoint not found: {ckpt_path}. Skipping export.")
        return
    ckpt = torch.load(ckpt_path, map_location=device)
    input_dim = ckpt.get("input_dim", None)
    d_model = ckpt.get("d_model", 256)
    depth   = ckpt.get("depth", 4)
    nhead   = ckpt.get("nhead", 8)
    ff      = ckpt.get("ff", 512)
    dropout = ckpt.get("dropout", 0.1)
    mask_ratio = ckpt.get("mask_ratio", 0.3)

    mae = MaskedAutoencoder(input_dim, d_model=d_model, depth=depth, nhead=nhead, ff=ff, dropout=dropout, mask_ratio=mask_ratio).to(device)
    fore = NextMonthForecaster(input_dim, d_model=d_model, depth=max(1, depth-1), nhead=nhead, ff=ff, dropout=dropout).to(device)
    trunk = TemporalTransformer(input_dim, d_model=d_model, nhead=nhead, num_layers=depth, dim_feedforward=ff, dropout=dropout).to(device)
    mae.load_state_dict(ckpt["mae"]); fore.load_state_dict(ckpt["fore"]); trunk.load_state_dict(ckpt["trunk"])
    mae.eval(); fore.eval(); trunk.eval()

    def score_one_site(x: torch.Tensor):
        """Compute reconstruction, forecast, and novelty scores for one site.

        Args:
            x: Feature tensor of shape (T, F) for a single site.

        Returns:
            Tuple of (rec_err, fore_err, novelty) arrays, each of shape (T,).
        """
        with torch.no_grad():
            l_mae, recon, _ = mae(x.unsqueeze(0))
            rec_err = ((recon - x.unsqueeze(0))**2).mean(dim=-1).squeeze(0).cpu().numpy()
            _, pred = fore(x.unsqueeze(0))
            TT = rec_err.shape[0]
            err_t = ((pred - x.unsqueeze(0)[:, 1:, :])**2).mean(dim=-1).squeeze(0)
            fore_err = torch.zeros(TT, device=x.device)
            if err_t.numel() > 0:
                fore_err[:-1] = err_t
                fore_err[-1] = err_t[-1]
            fore_err = fore_err.cpu().numpy()
            z = trunk(x.unsqueeze(0)).squeeze(0)
            dens = knn_density(z, k=args.k_density).cpu().numpy()
            nov = (dens - dens.min()) / (dens.max() - dens.min() + 1e-8)
            nov = 1.0 - nov
        return rec_err, fore_err, nov

    ds_all = UnsupervisedSiteDataset(args.features_csv, args.groundtruth_csv, split=args.split, scaler_path=args.scaler_path, split_col=args.split_col,
                                     per_month_feature_norm=not args.disable_per_month_feature_norm, robust=args.robust_scaler)
    loader = DataLoader(ds_all, batch_size=1, shuffle=False, num_workers=0)
    out_path = os.path.join(out_dir, f"unsup_month_scores_{args.split}.csv")
    import pandas as pd
    sites = []; scores = []
    for x, s, _ in tqdm(loader, total=len(ds_all)):
        x = x[0].to(device)
        rec, fore_e, nov = score_one_site(x)
        alpha_rec = getattr(args, "alpha_rec", 0.6)
        alpha_fore = getattr(args, "alpha_fore", 0.3)
        alpha_novel = getattr(args, "alpha_novel", 0.4)
        fused = alpha_rec * rec + alpha_fore * fore_e + alpha_novel * nov
        fused = normalize_scores(fused, method=args.score_norm_method, temperature=args.score_norm_temperature)
        sites.append(s[0]); scores.append(fused)
    if len(scores) > 0:
        S = np.stack(scores, axis=0)  # (N,T)
        df = pd.DataFrame(S, columns=MONTHS)
        df.insert(0, 'site_name', sites)
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", type=str)
    ap.add_argument("--groundtruth_csv", type=str, default=None)
    ap.add_argument("--split_col", type=str, default="split")
    ap.add_argument("--scaler_path", type=str, default=None)
    ap.add_argument("--trained_model", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--ff", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--no_pin_memory", action="store_true")
    ap.add_argument("--no_persistent_workers", action="store_true")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine"])
    ap.add_argument("--eta_min", type=float, default=1e-5)

    ap.add_argument("--mask_ratio", type=float, default=0.3)
    ap.add_argument("--bt_lambda", type=float, default=5e-3)
    ap.add_argument("--temp_tau", type=float, default=0.2)
    ap.add_argument("--w_mae", type=float, default=1.0)
    ap.add_argument("--w_fore", type=float, default=0.5)
    ap.add_argument("--w_tol", type=float, default=0.2)
    ap.add_argument("--w_bt", type=float, default=0.5)
    ap.add_argument("--bt_noise_std", type=float, default=0.005)

    ap.add_argument("--k_density", type=int, default=5)
    ap.add_argument("--proj_dim", type=int, default=128)
    ap.add_argument("--alpha_rec", type=float, default=0.6)
    ap.add_argument("--alpha_fore", type=float, default=0.3)
    ap.add_argument("--alpha_novel", type=float, default=0.4)
    ap.add_argument("--no_calendar_calibration", action="store_true")
    ap.add_argument("--rw_window", type=int, default=12)
    ap.add_argument("--global_spike_q", type=float, default=0.98)
    ap.add_argument("--calendar_population", type=str, default="preserved", choices=["train", "preserved", "looted"])

    ap.add_argument("--score_norm_method", type=str, default="sigmoid", choices=["none", "minmax", "sigmoid", "softmax"])
    ap.add_argument("--score_norm_temperature", type=float, default=1.0)

    ap.add_argument("--export_probs_only_month_names", action="store_true")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all", "all_looted"])  

    ap.add_argument("--export_only", action="store_true")
    ap.add_argument("--scaler_only", action="store_true")
    ap.add_argument("--intra_site_grid_calibration", action="store_true")
    ap.add_argument("--enable_cross_grid", action="store_true")
    ap.add_argument("--w_fuse_temporal", type=float, default=1.0)
    ap.add_argument("--w_fuse_cgz", type=float, default=1.0)
    ap.add_argument("--w_fuse_cgs", type=float, default=0.5)
    ap.add_argument("--output_components", action="store_true")
    ap.add_argument("--month_of_year_seasonal", action="store_true")
    ap.add_argument("--robust_scaler", action="store_true")
    ap.add_argument("--overwrite_scaler", action="store_true")
    ap.add_argument("--disable_per_month_feature_norm", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    try:
        import random as _rnd
        _rnd.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        print(f"[seed] Set global seed to {args.seed}")
    except Exception as _e_seed:
        print(f"[seed] WARNING unable to fully set seed: {_e_seed}")

    if args.output_dir is None:
        # Default under package dir: unsupervised_monthly/model_runs/default
        pkg_dir = Path(__file__).resolve().parent
        args.output_dir = str((pkg_dir / "model_runs" / "default").resolve())

    if args.scaler_only:
        compute_and_save_scaler(args)
    elif args.export_only:
        export_month_scores(args)
    else:
        train(args)
        export_month_scores(args)
