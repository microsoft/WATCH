#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Generate paper-ready figures and LaTeX tables from evaluation recall CSVs.

Reads recall matrices from:
    temporal_embedding_distance/results/recall_{split}_temporal_embedding_distance[_{positive,negative}].csv
    self_supervised_change_detection/results/recall_{split}_self_supervised_change_detection[_{positive,negative}].csv
    weakly_supervised/results/recall_{split}_weakly_supervised[_{positive,negative}].csv

Writes to results/paper_results/:
    - fig_recall_{split}_m{M}_by_embedding.png
    - fig_foundation_vs_handcrafted_{split}_m{M}.png
    - fig_directional_{split}_m{M}.png
    - table_recall_{split}_m{M}.tex
    - table_best_{split}_m{M}.tex
    - table_directional_gap_{split}_m{M}.tex
    - README.txt, README_{split}.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────

EXCLUDE_EMBEDDINGS = {"satclip"}

DISPLAY_NAMES = {
    "clip": "CLIP",
    "dinov3": "DINOv3",
    "georsclip": "GeoRSCLIP",
    "handcrafted": "Handcrafted",
    "prithvi-eo-2.0": "Prithvi-EO-2.0",
    "satlaspretrain": "Satlas-Pretrain",
    "satmae": "SatMAE",
}

# Row ordering for tables (foundation models first, handcrafted last)
EMBEDDING_ORDER = [
    "clip", "dinov3", "georsclip", "prithvi-eo-2.0",
    "satlaspretrain", "satmae", "handcrafted",
]

# Methods in display order
METHODS = [
    ("temporal_embedding_distance", "Temporal Embedding Distance (TED)", "temporal_embedding_distance"),
    ("self_supervised_change_detection", "Self-Supervised Change Detection (SSCD)", "self_supervised_change_detection"),
    ("weakly_supervised", "Weakly-supervised", "weakly_supervised"),
]

FOUNDATION_EMBEDDINGS = [
    "clip", "dinov3", "georsclip", "prithvi-eo-2.0", "satlaspretrain", "satmae",
]

REPRESENTATIVE_MARGINS = [0, 3, 6]


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_recall_csv(path: Path) -> pd.DataFrame:
    """Load a recall CSV with embedding as index."""
    df = pd.read_csv(path, index_col="embedding")
    df = df.loc[~df.index.isin(EXCLUDE_EMBEDDINGS)]
    return df


def _display(emb: str) -> str:
    return DISPLAY_NAMES.get(emb, emb)


def _pct(v: float) -> str:
    """Format a 0–1 recall value as a percentage string with 1 decimal."""
    return f"{v * 100:.1f}"


def _signed_pct(v: float) -> str:
    """Format a signed percentage-point gap."""
    pct = v * 100
    if abs(pct) < 0.05:
        return "+0.0"
    return f"+{pct:.1f}" if pct > 0 else f"{pct:.1f}"


# ── Table generators ──────────────────────────────────────────────────────────

def generate_recall_table(
    recalls: dict[str, pd.DataFrame],
    split: str,
    margin: int,
    out_dir: Path,
):
    """Recall@K table with columns for each (method, margin) combination."""
    margins = REPRESENTATIVE_MARGINS
    header_parts = []
    for _, method_disp, _ in METHODS:
        for m in margins:
            header_parts.append(f"{method_disp} m{m}")

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        rf"\caption{{Recall@K (\%) on the {split} split at representative "
        rf"temporal margins. Columns show symmetric margins "
        rf"$m\in\{{{','.join(str(m) for m in margins)}\}}$.}}"
        rf"\label{{tab:recall_{split}_m{margin}}}"
    )
    ncols = len(header_parts)
    lines.append(r"\begin{tabular}{l" + "r" * ncols + "}")
    lines.append(r"\toprule")
    lines.append("Embedding & " + " & ".join(header_parts) + r" \\")
    lines.append(r"\midrule")

    for emb in EMBEDDING_ORDER:
        row_vals = []
        for method_key, _, _ in METHODS:
            df = recalls[method_key]
            for m in margins:
                col = f"margin_{m}"
                if emb in df.index and col in df.columns:
                    row_vals.append(_pct(df.loc[emb, col]))
                else:
                    row_vals.append("--")
        lines.append(f"{_display(emb)} & " + " & ".join(row_vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    out_path = out_dir / f"table_recall_{split}_m{margin}.tex"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[write] {out_path}")


def generate_best_table(
    recalls: dict[str, pd.DataFrame],
    split: str,
    margin: int,
    out_dir: Path,
):
    """Best-performing embedding per method at a given margin."""
    col = f"margin_{margin}"
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        rf"\caption{{Best-performing embedding per method on the {split} "
        rf"split at $m={margin}$ (Recall@K).}}"
        rf"\label{{tab:best_{split}_m{margin}}}"
    )
    lines.append(r"\begin{tabular}{l l r}")
    lines.append(r"\toprule")
    lines.append(r"Method & Best embedding at m" + str(margin) + r" & Recall (\%) \\")
    lines.append(r"\midrule")

    for method_key, method_disp, _ in METHODS:
        df = recalls[method_key]
        if col not in df.columns:
            continue
        best_emb = df[col].idxmax()
        best_val = df.loc[best_emb, col]
        lines.append(
            f"{method_disp} & {_display(best_emb)} & {_pct(best_val)} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out_path = out_dir / f"table_best_{split}_m{margin}.tex"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[write] {out_path}")


def generate_directional_table(
    pos_recalls: dict[str, pd.DataFrame],
    neg_recalls: dict[str, pd.DataFrame],
    split: str,
    margin: int,
    out_dir: Path,
):
    """Directional asymmetry table: positive - negative recall at a margin."""
    col = f"margin_{margin}"
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        rf"\caption{{Directional margin asymmetry on the {split} split at "
        rf"$m={margin}$. Values are (positive-margin recall) minus "
        rf"(negative-margin recall), in percentage points.}}"
        rf"\label{{tab:directional_gap_{split}_m{margin}}}"
    )
    header_parts = []
    for _, method_disp, _ in METHODS:
        header_parts.append(rf"{method_disp} ($+/-$) at m{margin}")
    lines.append(r"\begin{tabular}{l" + "r" * len(header_parts) + "}")
    lines.append(r"\toprule")
    lines.append("Embedding & " + " & ".join(header_parts) + r" \\")
    lines.append(r"\midrule")

    for emb in EMBEDDING_ORDER:
        vals = []
        for method_key, _, _ in METHODS:
            pos_df = pos_recalls.get(method_key)
            neg_df = neg_recalls.get(method_key)
            if (
                pos_df is not None
                and neg_df is not None
                and emb in pos_df.index
                and emb in neg_df.index
                and col in pos_df.columns
                and col in neg_df.columns
            ):
                gap = pos_df.loc[emb, col] - neg_df.loc[emb, col]
                vals.append(_signed_pct(gap))
            else:
                vals.append("--")
        lines.append(f"{_display(emb)} & " + " & ".join(vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out_path = out_dir / f"table_directional_gap_{split}_m{margin}.tex"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[write] {out_path}")


# ── Figure generators ─────────────────────────────────────────────────────────

# Publication-quality settings
_DPI = 300
_TICK_SIZE = 17
_LABEL_SIZE = 19
_TITLE_SIZE = 21
_LEGEND_SIZE = 17
_XTICK_ROTATION_SIZE = 17


def _style_ax(ax):
    """Remove top/right spines, use large fonts."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=_TICK_SIZE)


def generate_recall_figure(
    recalls: dict[str, pd.DataFrame],
    split: str,
    margin: int,
    out_dir: Path,
):
    """Grouped bar chart of Recall@K at different margins, grouped by embedding."""
    margins = REPRESENTATIVE_MARGINS
    n_emb = len(EMBEDDING_ORDER)
    n_methods = len(METHODS)
    bar_width = 0.22
    x = np.arange(n_emb)

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = plt.cm.tab10.colors

    for i, (method_key, method_disp, _) in enumerate(METHODS):
        df = recalls[method_key]
        col = f"margin_{margin}"
        vals = []
        for emb in EMBEDDING_ORDER:
            if emb in df.index and col in df.columns:
                vals.append(df.loc[emb, col] * 100)
            else:
                vals.append(0)
        ax.bar(
            x + i * bar_width, vals, bar_width,
            label=method_disp, color=colors[i], edgecolor="white",
        )

    ax.set_xlabel("Embedding", fontsize=_LABEL_SIZE)
    ax.set_ylabel(f"Recall@K (%) at m={margin}", fontsize=_LABEL_SIZE)
    ax.set_xticks(x + bar_width * (n_methods - 1) / 2)
    ax.set_xticklabels([_display(e) for e in EMBEDDING_ORDER], fontsize=_XTICK_ROTATION_SIZE, rotation=25, ha="right")
    ax.legend(fontsize=_LEGEND_SIZE, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=len(METHODS))
    ax.set_ylim(0, 105)
    _style_ax(ax)
    fig.tight_layout()

    out_path = out_dir / f"fig_recall_{split}_m{margin}_by_embedding.png"
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {out_path}")


def generate_foundation_vs_handcrafted_figure(
    recalls: dict[str, pd.DataFrame],
    split: str,
    margin: int,
    out_dir: Path,
):
    """Compare foundation model average vs handcrafted across methods."""
    col = f"margin_{margin}"

    method_labels = []
    foundation_vals = []
    handcrafted_vals = []

    for method_key, method_disp, _ in METHODS:
        df = recalls[method_key]
        if col not in df.columns:
            continue
        # Foundation average
        fm_vals = [
            df.loc[emb, col] * 100
            for emb in FOUNDATION_EMBEDDINGS
            if emb in df.index
        ]
        foundation_vals.append(np.mean(fm_vals) if fm_vals else 0)
        # Handcrafted
        hc = df.loc["handcrafted", col] * 100 if "handcrafted" in df.index else 0
        handcrafted_vals.append(hc)
        method_labels.append(method_disp)

    x = np.arange(len(method_labels))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - bar_width / 2, foundation_vals, bar_width, label="Foundation models (avg)", color="steelblue")
    ax.bar(x + bar_width / 2, handcrafted_vals, bar_width, label="Handcrafted", color="darkorange")

    ax.set_xlabel("Method", fontsize=_LABEL_SIZE)
    ax.set_ylabel(f"Recall@K (%) at m={margin}", fontsize=_LABEL_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, fontsize=_XTICK_ROTATION_SIZE)
    ax.legend(fontsize=_LEGEND_SIZE, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2)
    ax.set_ylim(0, 105)
    _style_ax(ax)
    fig.tight_layout()

    out_path = out_dir / f"fig_foundation_vs_handcrafted_{split}_m{margin}.png"
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {out_path}")


def generate_directional_figure(
    pos_recalls: dict[str, pd.DataFrame],
    neg_recalls: dict[str, pd.DataFrame],
    split: str,
    margin: int,
    out_dir: Path,
):
    """Grouped bar chart of directional gap (positive - negative) per embedding and method."""
    col = f"margin_{margin}"
    n_emb = len(EMBEDDING_ORDER)
    n_methods = len(METHODS)
    bar_width = 0.22
    x = np.arange(n_emb)
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(16, 4.5))

    for i, (method_key, method_disp, _) in enumerate(METHODS):
        pos_df = pos_recalls.get(method_key)
        neg_df = neg_recalls.get(method_key)
        vals = []
        for emb in EMBEDDING_ORDER:
            if (
                pos_df is not None
                and neg_df is not None
                and emb in pos_df.index
                and emb in neg_df.index
                and col in pos_df.columns
                and col in neg_df.columns
            ):
                gap = (pos_df.loc[emb, col] - neg_df.loc[emb, col]) * 100
            else:
                gap = 0
            vals.append(gap)
        ax.bar(
            x + i * bar_width, vals, bar_width,
            label=method_disp, color=colors[i], edgecolor="white",
        )

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Embedding", fontsize=_LABEL_SIZE)
    ax.set_ylabel(f"Directional gap (pp) at m={margin}", fontsize=_LABEL_SIZE)
    ax.set_xticks(x + bar_width * (n_methods - 1) / 2)
    ax.set_xticklabels([_display(e) for e in EMBEDDING_ORDER], fontsize=_XTICK_ROTATION_SIZE, rotation=25, ha="right")
    ax.legend(fontsize=_LEGEND_SIZE, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=len(METHODS))
    _style_ax(ax)
    fig.tight_layout()

    out_path = out_dir / f"fig_directional_{split}_m{margin}.png"
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {out_path}")


# ── README generator ──────────────────────────────────────────────────────────

def write_readme(split: str, margin: int, out_dir: Path):
    txt = f"""Paper artifacts generated from existing evaluation CSVs.

- Split: {split}
- Key margin used in summary figures/tables: m={margin}

Figures (PNG only):
- fig_recall_{split}_m{margin}_by_embedding.png
- fig_foundation_vs_handcrafted_{split}_m{margin}.png
- fig_directional_{split}_m{margin}.png

Tables (LaTeX):
- table_recall_{split}_m{margin}.tex
- table_best_{split}_m{margin}.tex
- table_directional_gap_{split}_m{margin}.tex

Notes:
- Figures use large fonts and remove top/right spines.
- Tables report Recall@K as percentages.
- The embedding 'satclip' is excluded from all figures and tables.
"""
    out_path = out_dir / f"README_{split}.txt"
    out_path.write_text(txt)
    print(f"[write] {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate paper results (tables + figures)")
    parser.add_argument("--root", type=str, default=".", help="Repository root directory")
    parser.add_argument("--margin", type=int, default=3, help="Key margin for summary outputs")
    parser.add_argument(
        "--splits", type=str, default="test,all",
        help="Comma-separated list of splits to process",
    )
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="Output directory (default: <root>/results/paper_results)",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else root / "results" / "paper_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = [s.strip() for s in args.splits.split(",")]

    ted_results = root / "temporal_embedding_distance" / "results"
    sscd_results = root / "self_supervised_change_detection" / "results"
    weak_results = root / "weakly_supervised" / "results"

    # Map from method_key to (list of candidate file mode names, results_dir)
    # Try canonical names first, then legacy
    METHOD_FILE_MAP = {
        "temporal_embedding_distance": (
            ["temporal_embedding_distance", "distance_baseline", "baseline"],
            ted_results,
        ),
        "self_supervised_change_detection": (
            ["self_supervised_change_detection", "learned_unsupervised", "unsupervised"],
            sscd_results,
        ),
        "weakly_supervised": (["weakly_supervised"], weak_results),
    }

    def _find_recall_csv(results_dir, split, file_modes, suffix=""):
        """Try multiple mode names to find the recall CSV."""
        for fm in file_modes:
            name = f"recall_{split}_{fm}{suffix}.csv"
            path = results_dir / name
            if path.exists():
                return path
        return None

    for split in splits:
        print(f"\n{'='*60}")
        print(f"  Generating paper results for split={split}, margin={args.margin}")
        print(f"{'='*60}")

        recalls: dict[str, pd.DataFrame] = {}
        pos_recalls: dict[str, pd.DataFrame] = {}
        neg_recalls: dict[str, pd.DataFrame] = {}

        for method_key, (file_modes, results_dir) in METHOD_FILE_MAP.items():
            # Symmetric recall
            path = _find_recall_csv(results_dir, split, file_modes)
            if path:
                recalls[method_key] = load_recall_csv(path)
                print(f"  [load] {path.relative_to(root)}")
            else:
                print(f"  [warn] Missing: recall_{split}_{'|'.join(file_modes)}.csv in {results_dir}")

            # Positive
            path_pos = _find_recall_csv(results_dir, split, file_modes, "_positive")
            if path_pos:
                pos_recalls[method_key] = load_recall_csv(path_pos)
            else:
                print(f"  [warn] Missing directional positive for {method_key}")

            # Negative
            path_neg = _find_recall_csv(results_dir, split, file_modes, "_negative")
            if path_neg:
                neg_recalls[method_key] = load_recall_csv(path_neg)
            else:
                print(f"  [warn] Missing directional negative for {method_key}")

        if not recalls:
            print(f"  [skip] No recall data found for split={split}")
            continue

        # Tables
        generate_recall_table(recalls, split, args.margin, out_dir)
        generate_best_table(recalls, split, args.margin, out_dir)
        if pos_recalls and neg_recalls:
            generate_directional_table(pos_recalls, neg_recalls, split, args.margin, out_dir)

        # Figures
        generate_recall_figure(recalls, split, args.margin, out_dir)
        generate_foundation_vs_handcrafted_figure(recalls, split, args.margin, out_dir)
        if pos_recalls and neg_recalls:
            generate_directional_figure(pos_recalls, neg_recalls, split, args.margin, out_dir)

        # README
        write_readme(split, args.margin, out_dir)

    # Write top-level README with pointers
    readme_lines = [
        "Paper artifacts generated from existing evaluation CSVs.\n",
        f"- Split: {splits[-1]}",
        f"- Key margin used in summary figures/tables: m={args.margin}\n",
        f"Figures (PNG only):",
        f"- fig_recall_{splits[-1]}_m{args.margin}_by_embedding.png",
        f"- fig_foundation_vs_handcrafted_{splits[-1]}_m{args.margin}.png",
        f"- fig_directional_{splits[-1]}_m{args.margin}.png\n",
        f"Tables (LaTeX):",
        f"- table_recall_{splits[-1]}_m{args.margin}.tex",
        f"- table_best_{splits[-1]}_m{args.margin}.tex",
        f"- table_directional_gap_{splits[-1]}_m{args.margin}.tex\n",
        "Notes:",
        "- Figures use large fonts and remove top/right spines.",
        "- Tables report Recall@K as percentages.",
        "- The embedding 'satclip' is excluded from all figures and tables.",
    ]
    for s in splits:
        readme_lines.append(f"- README_{s}.txt")
    readme_path = out_dir / "README.txt"
    readme_path.write_text("\n".join(readme_lines) + "\n")
    print(f"\n[write] {readme_path}")
    print("[done] Paper results generated.")


if __name__ == "__main__":
    main()
