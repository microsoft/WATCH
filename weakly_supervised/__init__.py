# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Weakly-supervised monthly change detection.

This package trains a lightweight weakly-supervised model using month-level labels
available up to a cutoff (default: 2020_12), then runs inference for the full
2017_01..2024_12 window.

Design goals:
- Reuse the same unified features CSV format as self_supervised_change_detection.
- Export per-site, per-month probabilities in a matrix compatible with
  self_supervised_change_detection.evaluate_unified_monthlies.
"""

from .dataset import MONTHS, T
