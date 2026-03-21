#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
set -euo pipefail

# Unified embedding extraction entrypoint.
#
# Thin wrapper around `extract_embeddings_unified_modified.py`.
# Use it for both Afghanistan/site extraction (no grids) and global/grid extraction,
# with or without masking.
#
# Examples:
#   # Afghanistan / per-site mode (default):
#   ./extract_embeddings.sh --mode site --model prithvi-eo-2.0 --images-root planet_mosaics_final_4bands/images --output-dir planet_mosaics_final_4bands/features --start-year 2017 --end-year 2024
#
#   # Same, with masks:
#   ./extract_embeddings.sh --mode site --use-mask --masks-root planet_mosaics_final_4bands/masks_buffered --model prithvi-eo-2.0 --images-root planet_mosaics_final_4bands/images --output-dir planet_mosaics_final_4bands/features_with_mask --start-year 2017 --end-year 2024
#
#   # Global grid mode (RGB mosaics):
#   ./extract_embeddings.sh --mode grid --model satmae --images-root <global_images_root> --output-dir <output_dir> --start-year 2017 --end-year 2024 --grid-area 1.0 --min-valid-ratio 0.1
#

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON:-python}"

exec "$PYTHON_BIN" "$REPO_ROOT/extract_embeddings_unified_modified.py" "$@"
