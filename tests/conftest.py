"""EternityQuant 单元测试套（v0.12 固化）。

测纯逻辑无网络的模块，慢路径（拉行情/qlib 训练）标 `slow` marker，可 `pytest -m "not slow"` 跳过。
"""

from __future__ import annotations

import datetime as dt
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------- fixtures ----------

@pytest.fixture
def tmp_db(tmp_path):
    """用 pytest tmp_path fixture 的临时目录，pytest 自己管清理（不触发 WinError 32）。"""
    import eq.db as db_mod
    original_default = db_mod.DEFAULT_HOME
    tmp_home = tmp_path / ".eternityquant"
    tmp_home.mkdir(parents=True, exist_ok=True)
    db_mod.DEFAULT_HOME = tmp_home
    # 触发建表
    from eq.db import get_state_conn, get_cache_conn
    with get_state_conn() as _:
        pass
    with get_cache_conn() as _:
        pass
    yield tmp_home
    db_mod.DEFAULT_HOME = original_default


@pytest.fixture
def sample_bars():
    """造 30 日模拟 OHLCV 数据，测因子用。"""
    dates = pd.bdate_range("2025-01-01", periods=30)
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(30) * 2)
    high = close + np.abs(np.random.randn(30)) * 1.5
    low = close - np.abs(np.random.randn(30)) * 1.5
    open_ = close + np.random.randn(30) * 0.5
    volume = np.random.randint(1_000_000, 10_000_000, 30).astype(float)
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    }, index=dates)
