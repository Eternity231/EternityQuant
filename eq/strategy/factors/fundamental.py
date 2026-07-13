"""基本面因子占位（第一版无跨表数据源接入）。

未来接入：调用 eq.data.fundamental 拉财报数据，PE/PB/ROE/北向等。
"""

from __future__ import annotations

import pandas as pd


def pe_dummy(df: pd.DataFrame) -> pd.Series:
    """占位：返回常数 0 的 PE 因子，后续替换。"""
    return pd.Series(0, index=df.index, name="pe")
