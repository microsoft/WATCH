# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations
import os
from typing import List, Tuple

MONTHS: List[str] = [f"{y}_{m:02d}" for y in range(2017, 2025) for m in range(1, 13)]
MONTH_TO_INDEX = {m: i for i, m in enumerate(MONTHS)}

def prev_months(year: int, month: int, k: int = 3) -> List[Tuple[int,int]]:
    out = []
    y, m = year, month
    for _ in range(k):
        m -= 1
        if m == 0:
            y -= 1
            m = 12
        if y < 2017:
            break
        out.append((y, m))
    return out

def normalize_month_str(val) -> str:
    if val is None:
        return 'NA'
    try:
        s = str(val).strip()
    except Exception:
        return 'NA'
    if s.upper() in {'NA','NONE','', 'NAN','-1'}:
        return 'NA'
    import re
    m = re.search(r'(20\d{2})[^0-9]?([0-1]?\d)', s)
    if not m:
        return 'NA'
    year = int(m.group(1))
    month = int(m.group(2))
    if not (1 <= month <= 12):
        return 'NA'
    return f"{year}_{month:02d}"

