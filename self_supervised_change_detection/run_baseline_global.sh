#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
set -euo pipefail

# Run distance-baseline global inference for each embedding using:
# - global features CSVs: planet_mosaics_final_4bands/features_unified_global_without_mask
# - trained scaler stats (Afghanistan-trained): self_supervised_change_detection/model_runs/<embedding>/scaler_stats.npz
# - output: self_supervised_change_detection/global_results/<embedding>/inference_all_months_temporal_embedding_distance.csv

FEATURES_DIR=${FEATURES_DIR:-planet_mosaics_final_4bands/features_unified_global_without_mask}
MODEL_RUNS_DIR=${MODEL_RUNS_DIR:-self_supervised_change_detection/model_runs}
OUT_BASE=${OUT_BASE:-self_supervised_change_detection/global_results}
PY=${PY:-./change_detect/bin/python}

MODELS=${1:-all}
DISTANCE=${DISTANCE:-l2}
BASELINE_NORM_METHOD=${BASELINE_NORM_METHOD:-sigmoid}
BASELINE_NORM_TEMPERATURE=${BASELINE_NORM_TEMPERATURE:-1.0}

if [[ ! -d "$FEATURES_DIR" ]]; then
  echo "[err] FEATURES_DIR not found: $FEATURES_DIR" >&2
  exit 1
fi

embeddings=()
if [[ "$MODELS" == "all" ]]; then
  while IFS= read -r -d '' f; do
    bn=$(basename "$f")
    emb=${bn#features_}
    emb=${emb%_2017_2024_global.csv}
    embeddings+=("$emb")
  done < <(find "$FEATURES_DIR" -maxdepth 1 -type f -name 'features_*_2017_2024_global.csv' -print0)
else
  embeddings=("$MODELS")
fi

if [[ ${#embeddings[@]} -eq 0 ]]; then
  echo "[err] No global features CSVs found under $FEATURES_DIR" >&2
  exit 1
fi

for emb in "${embeddings[@]}"; do
  feats="$FEATURES_DIR/features_${emb}_2017_2024_global.csv"
  if [[ ! -f "$feats" ]]; then
    echo "[warn] missing features CSV for $emb: $feats" >&2
    continue
  fi

  model_dir="$MODEL_RUNS_DIR/$emb"
  scaler_npz="$model_dir/scaler_stats.npz"
  if [[ ! -f "$scaler_npz" ]]; then
    echo "[warn] missing scaler for $emb under $model_dir; skipping" >&2
    continue
  fi

  out_dir="$OUT_BASE/$emb"
  mkdir -p "$out_dir"
  out_csv="$out_dir/inference_all_months_temporal_embedding_distance.csv"

  echo "[run] distance-baseline global: $emb"
  "$PY" -m temporal_embedding_distance.infer_all_months \
    --features_csv "$feats" \
    --scaler_path "$scaler_npz" \
    --output_csv "$out_csv" \
    --group_cols "site_name,grid_id" \
    --meta_cols "lon,lat" \
    --distance "$DISTANCE" \
    --score_norm_method "$BASELINE_NORM_METHOD" \
    --score_norm_temperature "$BASELINE_NORM_TEMPERATURE"

done

echo "[ok] wrote outputs under $OUT_BASE"