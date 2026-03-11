# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Month utilities (shared conventions).

We re-export MONTHS and normalizers from self_supervised_change_detection to guarantee the
same month ordering and parsing across pipelines.
"""

from __future__ import annotations

from self_supervised_change_detection.month_utils import MONTHS, normalize_month_str

T = len(MONTHS)
