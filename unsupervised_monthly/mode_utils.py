# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Canonical mode names and alias resolution."""

from __future__ import annotations

from typing import Optional


# Canonical, user-facing mode names.
CANONICAL_MODES = (
    "distance_baseline",
    "learned_unsupervised",
    "weakly_supervised",
)

# Backwards-compatible aliases.
MODE_ALIASES = {
    # Legacy names
    "baseline": "distance_baseline",
    "unsupervised": "learned_unsupervised",
    # Canonical names
    "distance_baseline": "distance_baseline",
    "learned_unsupervised": "learned_unsupervised",
    "weakly_supervised": "weakly_supervised",
}


def normalize_mode(mode: Optional[str]) -> Optional[str]:
    if mode is None:
        return None
    m = str(mode).strip()
    return MODE_ALIASES.get(m, m)


def is_legacy_mode(mode: str) -> bool:
    m = str(mode).strip()
    return m in ("baseline", "unsupervised")


def infer_mode_from_filename(filename: str) -> Optional[str]:
    """Infer canonical mode from common filename tokens."""
    name = filename

    # Prefer canonical tokens first.
    if "_distance_baseline" in name:
        return "distance_baseline"
    if "_learned_unsupervised" in name:
        return "learned_unsupervised"
    if "_weakly_supervised" in name:
        return "weakly_supervised"

    # Legacy tokens.
    if "_baseline" in name:
        return "distance_baseline"
    if "_unsupervised" in name:
        return "learned_unsupervised"

    return None
