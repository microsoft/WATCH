# Archaeological Change Detection (Monthly Pipelines)

This repository provides a standardized, end-to-end month-level change detection workflow based on:

1) extracting per-site monthly embeddings (foundation models or handcrafted), then
2) running one (or more) maintained monthly pipelines:
   - **Distance baseline** (no training): `unsupervised_monthly` `distance_baseline` mode
   - **Learned unsupervised** (train once, then export): `unsupervised_monthly` `learned_unsupervised` mode
   - **Weakly-supervised** (train on month labels up to a cutoff): `weakly_supervised_monthly`

All three pipelines output a per-site probability distribution over months (columns `2017_01..2024_12`) and share the same evaluation tooling.

---

## 0) Data Layout

The maintained monthly pipelines assume:

```
planet_mosaics_final_4bands/
├── images/
│   ├── <site_name>/
│   │   ├── 2017_01.tif
│   │   ├── 2017_02.tif
│   │   └── ...
│   └── ...
├── masks_buffered/                  # optional but recommended
│   ├── <site_name>/mask.tif
│   └── ...
└── ground_truth_split_balanced_aux.csv
```

Ground-truth CSV requirements (minimum):

- `site_name`
- `split` (e.g., `train|val|test|all`)
- `looted` in `{0,1}`
- `looted_month` as `YYYY_MM` for looted sites (or empty/None)

---

## 1) Installation

The runners assume an environment with the repo dependencies installed. The provided shell runners default to a local venv named `change_detect`.

```bash
python -m venv change_detect
source change_detect/bin/activate
pip install -r requirements.txt
```

---

## License

This project is licensed under the MIT License. See `LICENSE`.

## Third-party

Third-party dependency attributions are listed in `THIRD_PARTY_NOTICES.md`.

## Privacy / security notes

- This repository does not ship datasets or model checkpoints.
- Do not hardcode API keys/tokens in code or config. Use environment variables.
- Avoid absolute machine-specific paths; the runners and extractors default to repo-relative paths.

---

## 2) Feature Extraction (One Standard Entry Point)

Use the unified extractor wrapper:

```bash
./extract_embeddings.sh --help
```

### 2.1 Afghanistan / per-site monthly embeddings (recommended)

This produces one embedding per `(site_name, month)` and writes a CSV with columns:
`site_name, month, f0..fN`.

Recommended output directory for the monthly pipelines:

- `planet_mosaics_final_4bands/features_unified_new_with_mask/`

Example (masked) for one embedding:

```bash
./extract_embeddings.sh --mode site \
   --model handcrafted \
   --images-root planet_mosaics_final_4bands/images \
   --use-mask --masks-root planet_mosaics_final_4bands/masks_buffered \
   --start-year 2017 --end-year 2024 \
   --output-dir planet_mosaics_final_4bands/features_unified_new_with_mask
```

This produces (example):

- `planet_mosaics_final_4bands/features_unified_new_with_mask/features_handcrafted_2017_2024_masked.csv`

Foundation models are supported too. Download models with:

```bash
# List available model IDs
python download_hf_models.py --list

# Download specific models (examples)
python download_hf_models.py --models dinov3 prithvi-eo-2.0 satmae satclip

# Download everything
python download_hf_models.py --models all
```

Then extract features, e.g.:

```bash
./extract_embeddings.sh --mode site \
   --model dinov3 \
   --images-root planet_mosaics_final_4bands/images \
   --use-mask --masks-root planet_mosaics_final_4bands/masks_buffered \
   --start-year 2017 --end-year 2024 \
   --device cuda:0 \
   --output-dir planet_mosaics_final_4bands/features_unified_new_with_mask
```

### 2.2 Global / grid embeddings (optional)

If you are running global site collections where each site is large and should be tiled into km² grids, use:

```bash
./extract_embeddings.sh --mode grid --model dinov3 --grid-area 1.0 --min-valid-ratio 0.1 \
   --images-root <global_images_root> \
   --start-year 2017 --end-year 2024 \
   --output-dir <output_dir>
```

Grid outputs include `grid_id, lon, lat` columns and are separate from the monthly Afghanistan-style pipelines.

---

## 3) Run Monthly Pipelines

All runners assume you activated the environment:

```bash
source change_detect/bin/activate
```

### 3.1 Unsupervised monthly: Distance baseline mode

Start runs (one tmux session per embedding), then merge/evaluate/aggregate:

```bash
# Start tmux sessions (all default embeddings)
bash unsupervised_monthly/run_baseline_pipeline.sh start

# After sessions finish, merge + evaluate + aggregate
bash unsupervised_monthly/run_baseline_pipeline.sh all
```

Outputs:

- Unified score matrix (per embedding):
   - `unsupervised_monthly/model_runs/<embedding>/unsup_month_scores_all_distance_baseline_new.csv`
- Per-embedding metrics:
   - `unsupervised_monthly/results/<embedding>/<embedding>_distance_baseline_{test|all}_metrics{,_positive,_negative}.csv`

### 3.2 Unsupervised monthly: Learned unsupervised mode

```bash
# Start tmux sessions (all default embeddings)
bash unsupervised_monthly/run_unsupervised_pipeline.sh start

# After sessions finish, merge + evaluate + aggregate
bash unsupervised_monthly/run_unsupervised_pipeline.sh all
```

Outputs:

- Unified score matrix (per embedding):
   - `unsupervised_monthly/model_runs/<embedding>/unsup_month_scores_all_learned_unsupervised_new.csv`

### 3.3 Weakly-supervised monthly

This trains a lightweight sequence model using month labels only up to a cutoff (default `LABEL_END_MONTH=2020_12`), then exports a full `2017_01..2024_12` probability table.

```bash
# Start runs (one tmux session per embedding)
bash weakly_supervised_monthly/scripts/run_weakly_supervised_monthly.sh start

# Aggregate recall tables across embeddings
bash weakly_supervised_monthly/scripts/run_weakly_supervised_monthly.sh aggregate
```

Outputs:

- Unified score matrix (per embedding):
   - `weakly_supervised_monthly/model_runs/<embedding>/unsup_month_scores_all_weakly_supervised.csv`
- Per-embedding metrics:
   - `weakly_supervised_monthly/results/<embedding>/<embedding>_weakly_supervised_{test|all}_metrics{,_positive,_negative}.csv`

The default knobs for each pipeline are documented in the runner scripts themselves:

- `unsupervised_monthly/run_baseline_pipeline.sh`
- `unsupervised_monthly/run_unsupervised_pipeline.sh`
- `weakly_supervised_monthly/scripts/run_weakly_supervised_monthly.sh`

---

## 4) Evaluation + Inference Tables

### 4.1 Export a single “all sites × all months” inference table

This materializes stable inference tables under each pipeline’s `results/<embedding>/` folder:

```bash
python export_merged_monthly_inference_tables.py --pipelines all --year_start 2017 --year_end 2024
```

Examples of exported artifacts:

- `unsupervised_monthly/results/<embedding>/inference_all_months_distance_baseline.csv`
- `unsupervised_monthly/results/<embedding>/inference_all_months_learned_unsupervised.csv`
- `weakly_supervised_monthly/results/<embedding>/inference_all_months_weakly_supervised.csv`

Each has columns: `site_name, known_month_of_change, 2017_01..2024_12`.

### 4.2 Aggregated recall tables (monthly / top-k / margins)

The pipeline runners call the evaluator automatically, but you can run it manually:

```bash
python -m unsupervised_monthly.evaluate_unified_monthlies \
   --scores_csv <path_to_unsup_month_scores_all_*.csv> \
   --mode distance_baseline \
   --split test \
   --top_k 12 \
   --max_margin 6 \
   --directional \
   --groundtruth_csv planet_mosaics_final_4bands/ground_truth_split_balanced_aux.csv \
   --results_dir <output_results_dir> \
   --embedding <embedding_id>
```

To aggregate recall summaries across embeddings:

```bash
python unsupervised_monthly/aggregate_recalls.py
```

---
## Model Architecture

The system uses a hybrid architecture combining:

1. **Spatial Feature Extractor**: U-Net with ResNet20 encoder
   - Processes 4-band satellite images
   - Uses ImageNet pre-trained weights
   - Extracts spatial features for each time step

2. **Temporal Processing**: 
   - Temporal convolutional networks with dilated convolutions
   - Multi-head attention mechanism
   - Global feature aggregation

3. **Multi-task Heads**:
   - Change classification (binary)
   - Temporal localization (month prediction)
   - Uncertainty estimation

## Key Features

### Mask Processing
- Each site has a corresponding mask file (`mask.tif`)
- Non-zero mask values (1 or 2) are treated as areas of interest
- Images are masked before processing to focus on relevant areas

### Balanced Data Split
- 60% training, 20% validation, 20% test
- Stratified split ensures balanced representation of looted and preserved sites
- Split information is saved for reproducibility

### Uncertainty Quantification
- Monte Carlo dropout for uncertainty estimation
- Confidence-based filtering for reliable predictions
- Uncertainty correlation analysis

## Outputs

### Training Outputs
- Model checkpoints (`best_model.pth`, `latest_checkpoint.pth`)
- Training logs and TensorBoard summaries
- Split information (`split_info.json`)
- Final evaluation results (`final_results.json`)

### Inference Outputs
- Prediction summaries (`summary_report.json`)
- Attention map visualizations (`attention_maps.png`)
- Distribution plots (`summary_plots.png`)
- Detailed predictions CSV

### Unsupervised Outputs
- `unsup_models.pt` (ensemble checkpoint)
- `scaler_stats.npz` (feature & calendar normalization stats)
- `unsup_month_scores_<split>.csv` (raw scores / probabilities)

## Evaluation Metrics

The system evaluates:
- **Classification**: Accuracy, Precision, Recall, F1-score, ROC-AUC
- **Temporal Localization**: Month prediction accuracy within ±3 months
- **Uncertainty**: Correlation with prediction confidence
- **Interpretability**: Attention map analysis

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Rasterio for GeoTIFF processing
- Segmentation Models PyTorch for U-Net
- Other dependencies listed in `requirements.txt`

## Citation

If you use this code, please cite:

```bibtex
@article{archaeological_change_detection,
  title={Deep Learning for Archaeological Site Change Detection},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Quick Start Cheat Sheet (Feature Extraction & Unsupervised Global Inference)

This section summarizes (a) canonical feature CSV filename patterns, (b) minimal extraction command templates, and (c) how to trigger unified unsupervised global inference using the standardized pipeline. Adjust GPU device (`--device cuda:0`) as needed.

### 1. Feature CSV Naming Patterns

Feature CSVs are expected to include the model key and the collection type in the filename (e.g., monthly per-site vs global-sites grid), and many scripts use a `global_sites_grid` substring to locate the right files.

Example (global grid, 1.0 km$^2$ tiles, DINOv3): `planet_mosaics_final_4bands/features/features_dinov3_global_sites_grid_1.0km2.csv`.

### 2. Minimal Feature Extraction Command Templates

Assumes imagery root directory `planet_mosaics_final_4bands` with per-site folders and month TIFFs.

```bash
# DINOv3 (example variant convnext-base)
python extract_dinov3_features.py \
   --images-root planet_mosaics_final_4bands \
   --output-csv planet_mosaics_final_4bands/features/features_dinov3_global_sites_grid_1.0km2.csv \
   --model-type convnext-base \
   --device cuda:0

# Unified extractor (Prithvi, SatMAE, SatCLIP, GeoRSCLIP, Satlaspretrain)
python extract_embeddings_unified_modified.py --mode grid \
   --model prithvi-eo-2.0 \
   --images-root planet_mosaics_final_4bands \
   --output-csv planet_mosaics_final_4bands/features/features_prithvi-2.0_global_sites_grid_1.0km2.csv \
   --start-year 2017 --end-year 2024 --device cuda:0

python extract_embeddings_unified_modified.py --mode grid \
   --model satmae \
   --images-root planet_mosaics_final_4bands \
   --output-csv planet_mosaics_final_4bands/features/features_satmae_global_sites_grid_1.0km2.csv \
   --start-year 2017 --end-year 2024 --device cuda:0

# Repeat for satclip, georsclip, satlaspretrain changing --model and output filename accordingly
```

### 3. Unified Unsupervised Global Inference (Per Model)

After all feature CSVs are present inside `planet_mosaics_final_4bands/features/`, run (example for `satmae`):

```bash
cd old_model_scripts_backup/unsupervised
bash run_unsupervised_global_inference.sh --model satmae
```

Key behaviors:
1. EXPORT stage: Produces `unsup_month_scores_all.csv` inside `old_model_scripts_backup/unsupervised/model_runs/<model>/global/` with month-labeled probability columns (2017_01 .. 2024_12) using configured normalization (e.g., sigmoid temperature, seasonal, cross-grid).
2. Evaluation stage: Consumes the exported probabilities directly (`--score_norm_method none`) to compute month probability matrix, top-k, histograms, writing all outputs into the same `global/` subdirectory.
3. Consistency: The suffixed copy (e.g., `unsup_month_scores_all_global_all.csv`) is synchronized with the canonical export when month-name-only mode is used so they match byte-for-byte.

### 4. Force Re-Export When Needed

Use `--force-export` (script flag) if:
- You modified normalization hyperparameters (temperature, seasonal toggles, fusion weights).
- You changed component alpha weights (`alpha_rec`, `alpha_fore`, `alpha_novel`).
- You updated the feature CSVs (new extraction run or bug fix).

Without `--force-export`, the script will reuse an existing probability export if found, ensuring reproducibility.

### 5. Quick Diagnostic Checks

```bash
# List produced probability files for a model
ls -1 old_model_scripts_backup/unsupervised/model_runs/<model>/global/unsup_month_scores_all*.csv

# Head probabilities (verify month headers present)
head -n 2 old_model_scripts_backup/unsupervised/model_runs/<model>/global/unsup_month_scores_all.csv

# Confirm top-k file exists
ls old_model_scripts_backup/unsupervised/model_runs/<model>/global/topk_all_global_all.csv
```

### 6. Aggregating Metrics Across Models

Example (already prototyped in `test.ipynb`) building a unified margin vs accuracy table:

```python
import os, pandas as pd
csv_dir = "old_model_scripts_backup/unsupervised/model_runs"  # adjust if different
models = ["prithvi-2.0", "satmae", "satclip", "georsclip", "satlaspretrain", "dinov3"]
results = {}
for model in models:
      mfile = os.path.join(csv_dir, model, "metrics_all.csv")
      if not os.path.exists(mfile):
            continue
      df = pd.read_csv(mfile)
      if not {"margin","acc"}.issubset(df.columns):
            continue
      for margin, acc in zip(df.margin, df.acc):
            results.setdefault(margin, {})[model] = acc
unified = (pd.DataFrame.from_dict(results, orient='index')
                .rename_axis('margin').reset_index().sort_values('margin'))
unified.to_csv(os.path.join(csv_dir, "all_models_metrics_all.csv"), index=False)
print(unified.head())
```

### 7. Troubleshooting Quick Tips

| Symptom | Likely Cause | Action |
|---------|-------------|--------|
| Missing probability columns | Export skipped / reused stale file | Add `--force-export` and rerun script |
| Different top-k vs manual softmax | Evaluation re-normalized | Ensure script passes `--score_norm_method none` in evaluation stage |
| Model directory empty | Feature CSV pattern not matched | Verify filename contains `<model>_global_sites_grid` substring |
| GPU OOM during extraction | Batch too large | Reduce batch size flag (if available) or use CPU fallback for small subset |
| Misaligned month range | Start/End year mismatch | Re-extract features with consistent `--start-year 2017 --end-year 2024` |

### 8. Minimal End-to-End (Single Model Example)

```bash
# 1. Extract features
./extract_embeddings.sh --mode grid \
   --model satmae \
   --images-root planet_mosaics_final_4bands \
   --output-csv planet_mosaics_final_4bands/features/features_satmae_global_sites_grid_1.0km2.csv \
   --start-year 2017 --end-year 2024

# 2. Run global unsupervised inference
cd old_model_scripts_backup/unsupervised
bash run_unsupervised_global_inference.sh --model satmae --force-export

# 3. Inspect outputs
head -n 2 satmae/unsup_month_scores_all.csv
head -n 2 satmae/topk_all_global_all.csv
```

---