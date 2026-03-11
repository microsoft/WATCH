#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Training script for the weakly-supervised change detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

from .dataset import SiteTimeSeriesDataset, T, LabelWindow
from .model import LSTMChangeDetector


def _make_targets(looted: torch.Tensor, known_idx: torch.Tensor, label_window: LabelWindow) -> torch.Tensor:
    """Build (B,T) targets in [0,1] for BCE training on the time head."""
    B = looted.shape[0]
    targets = torch.zeros((B, T), dtype=torch.float32, device=looted.device)

    kmask = (looted == 1) & (known_idx >= 0)
    if not kmask.any():
        return targets

    rows = torch.nonzero(kmask, as_tuple=False).squeeze(1)
    w = int(max(0, label_window.window))
    for r in rows.tolist():
        c = int(known_idx[r].item())
        left = max(0, c - w)
        right = min(T - 1, c + w)
        pos = torch.arange(left, right + 1, device=looted.device)
        if label_window.smooth_type == "gauss" and w > 0:
            sigma = max(1.0, w / 2.0)
            d = (pos - c).float()
            vals = torch.exp(-(d * d) / (2.0 * sigma * sigma))
            vals = vals / vals.max().clamp_min(1e-8)
        else:
            vals = torch.ones_like(pos, dtype=torch.float32)
        targets[r, pos] = vals
    return targets


def parse_args():
    ap = argparse.ArgumentParser("Train weakly-supervised monthly model (labels up to a cutoff)")
    ap.add_argument("--features_csv", type=str, required=True)
    ap.add_argument("--groundtruth_csv", type=str, required=True)
    ap.add_argument("--split_col", type=str, default="split")
    ap.add_argument("--label_end_month", type=str, default="2020_12")

    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--enc_hidden", type=int, default=128)
    ap.add_argument("--lstm_hidden", type=int, default=128)
    ap.add_argument("--lstm_layers", type=int, default=2)

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--label_window", type=int, default=1, help="Label smoothing radius in months")
    ap.add_argument("--label_smooth_type", type=str, default="gauss", choices=["gauss", "box"])

    ap.add_argument(
        "--pos_weight",
        type=float,
        default=0.0,
        help=(
            "BCE positive-class weight. If <= 0, compute an automatic value based on the "
            "effective positive mass induced by --label_window and the number of known-month looted sites."
        ),
    )
    ap.add_argument(
        "--max_pos_weight",
        type=float,
        default=200.0,
        help="Cap for auto-computed pos_weight to avoid unstable training.",
    )

    ap.add_argument(
        "--oversample_pos_sites",
        type=float,
        default=0.0,
        help=(
            "If > 1, use a WeightedRandomSampler to oversample looted sites with known month labels "
            "by this factor (preserved sites keep weight 1.0)."
        ),
    )

    ap.add_argument(
        "--early_stop_patience",
        type=int,
        default=15,
        help="Stop after this many epochs without improving best train loss (0 disables early stopping).",
    )

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=0)
    return ap.parse_args()


def _effective_positive_mass(label_window: LabelWindow) -> float:
    """Approximate the sum of target values across months for one positive site."""
    w = int(max(0, label_window.window))
    if w == 0:
        return 1.0
    pos = np.arange(-w, w + 1, dtype=np.float64)
    if label_window.smooth_type == "gauss":
        sigma = max(1.0, w / 2.0)
        vals = np.exp(-(pos * pos) / (2.0 * sigma * sigma))
        vals = vals / max(vals.max(), 1e-8)
        return float(vals.sum())
    return float((2 * w + 1))


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(f"cuda:{args.gpu_id}")

    scaler_path = out_dir / "scaler_stats.npz"
    ds = SiteTimeSeriesDataset(
        features_csv=args.features_csv,
        gt_csv=args.groundtruth_csv,
        split="train",
        split_col=args.split_col,
        scaler_path=str(scaler_path),
        save_fitted_scaler=True,
        include_preserved=True,
        include_unknown_looted=False,
        label_end_month=args.label_end_month,
    )

    # Optional oversampling of the rare positive (known-month looted) sites.
    sampler = None
    if float(args.oversample_pos_sites) and float(args.oversample_pos_sites) > 1.0:
        gt = ds.gt.reset_index()
        is_pos_site = (gt["looted"].astype(int) == 1) & (gt["known_idx"].astype(int) >= 0)
        weight_pos = float(args.oversample_pos_sites)
        weights_by_site = {row["site_name"]: (weight_pos if bool(p) else 1.0) for row, p in zip(gt.to_dict("records"), is_pos_site)}
        weights = [weights_by_site.get(s, 1.0) for s in ds.sites]
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    input_dim = len(ds.feature_cols)
    model = LSTMChangeDetector(
        input_dim=input_dim,
        enc_hidden=args.enc_hidden,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lw = LabelWindow(window=args.label_window, smooth_type=args.label_smooth_type)

    # Automatic pos_weight tuned to the extreme imbalance in month-level targets.
    # total entries = N_sites * T; effective positives ~ N_pos_sites * sum(target window)
    pos_weight_val = float(args.pos_weight)
    if pos_weight_val <= 0.0:
        gt = ds.gt.reset_index()
        n_pos_sites = int(((gt["looted"].astype(int) == 1) & (gt["known_idx"].astype(int) >= 0)).sum())
        total_entries = float(len(ds) * T)
        eff_pos = float(n_pos_sites) * _effective_positive_mass(lw)
        eff_pos = max(eff_pos, 1.0)
        auto_pw = (total_entries - eff_pos) / eff_pos
        pos_weight_val = float(max(1.0, min(auto_pw, float(args.max_pos_weight))))
        print(
            f"[info] auto pos_weight={pos_weight_val:.3f} "
            f"(n_pos_sites={n_pos_sites}, total_entries={int(total_entries)}, eff_pos={eff_pos:.2f})"
        )

    pos_w = torch.tensor(max(pos_weight_val, 1.0), device=device)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)

    best = float("inf")
    epochs_since_best = 0
    for epoch in range(int(args.epochs)):
        model.train()
        running = 0.0
        it = tqdm(dl, desc=f"epoch {epoch}", dynamic_ncols=True)
        for feats, looted, known_idx, _site in it:
            feats = feats.to(device, non_blocking=True)
            looted = looted.to(device, non_blocking=True)
            known_idx = known_idx.to(device, non_blocking=True)

            t_logits = model.forward_time_logits(feats)
            targets = _make_targets(looted, known_idx, lw)

            loss = bce(t_logits, targets)
            if not torch.isfinite(loss):
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
            opt.step()

            running += float(loss.detach())
            it.set_postfix(loss=float(loss.detach()))

        avg = running / max(1, len(dl))
        if avg < best:
            best = avg
            epochs_since_best = 0
            torch.save(model.state_dict(), out_dir / "model.pt")
        else:
            epochs_since_best += 1

        print(f"[train] epoch={epoch} avg_loss={avg:.6f} best={best:.6f}")

        if int(args.early_stop_patience) > 0 and epochs_since_best >= int(args.early_stop_patience):
            print(f"[train] early stopping: no improvement for {epochs_since_best} epochs")
            break

    meta = {
        "features_csv": args.features_csv,
        "groundtruth_csv": args.groundtruth_csv,
        "split_col": args.split_col,
        "label_end_month": args.label_end_month,
        "input_dim": input_dim,
        "enc_hidden": args.enc_hidden,
        "lstm_hidden": args.lstm_hidden,
        "lstm_layers": args.lstm_layers,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "pos_weight": args.pos_weight,
        "pos_weight_effective": pos_weight_val,
        "max_pos_weight": args.max_pos_weight,
        "oversample_pos_sites": args.oversample_pos_sites,
        "early_stop_patience": args.early_stop_patience,
        "label_window": args.label_window,
        "label_smooth_type": args.label_smooth_type,
    }
    (out_dir / "train_config.json").write_text(json.dumps(meta, indent=2))
    print(f"[ok] saved model -> {out_dir / 'model.pt'}")
    print(f"[ok] saved scaler -> {scaler_path}")


if __name__ == "__main__":
    main()
