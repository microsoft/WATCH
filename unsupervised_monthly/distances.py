"""Distance utilities."""

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np

def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.sqrt(np.sum(diff * diff)))

def cosine_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    na = np.linalg.norm(a) + eps
    nb = np.linalg.norm(b) + eps
    sim = float(np.dot(a, b) / (na * nb))
    return 1.0 - sim
