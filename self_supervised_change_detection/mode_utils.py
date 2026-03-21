# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Canonical mode names and alias resolution."""

from __future__ import annotations

from typing import Optional


# Canonical, user-facing mode names.
CANONICAL_MODES = (
    "temporal_embedding_distance",
    "self_supervised_change_detection",
    "weakly_supervised",
)

# Backwards-compatible aliases (old canonical names and legacy shorthand).
MODE_ALIASES = {
    # New canonical names
    "temporal_embedding_distance": "temporal_embedding_distance",
    "self_supervised_change_detection": "self_supervised_change_detection",
    "weakly_supervised": "weakly_supervised",
    # Old canonical names (backward compatibility)
    "distance_baseline": "temporal_embedding_distance",
    "learned_unsupervised": "self_supervised_change_detection",
    # Legacy shorthand
    "baseline": "temporal_embedding_distance",
    "unsupervised": "self_supervised_change_detection",
}


def normalize_mode(mode: Optional[str]) -> Optional[str]:
    if mode is None:
        return None
    m = str(mode).strip()
    return MODE_ALIASES.get(m, m)


def is_legacy_mode(mode: str) -> bool:
    m = str(mode).strip()
    return m in ("baseline", "unsupervised", "distance_baseline", "learned_unsupervised")


def infer_mode_from_filename(filename: str) -> Optional[str]:
    """Infer canonical mode from common filename tokens."""
    name = filename

    # Prefer new canonical tokens first.
    if "_temporal_embedding_distance" in name:
        return "temporal_embedding_distance"
    if "_self_supervised_change_detection" in name:
        return "self_supervised_change_detection"
    if "_weakly_supervised" in name:
        return "weakly_supervised"

    # Old canonical tokens (backward compatibility).
    if "_distance_baseline" in name:
        return "temporal_embedding_distance"
    if "_learned_unsupervised" in name:
        return "self_supervised_change_detection"

    # Legacy tokens.
    if "_baseline" in name:
        return "temporal_embedding_distance"
    if "_unsupervised" in name:
        return "self_supervised_change_detection"

    return None
