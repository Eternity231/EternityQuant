"""db 层单元测试（用 tmp_db fixture，不污染真实库）。"""

from __future__ import annotations

import pytest


def test_execute_empty(tmp_db):
    """空库 execute 应返回空列表。"""
    from eq.db import execute
    rows = execute("SELECT * FROM watchlist")
    assert rows == []


def test_execute_write_and_read(tmp_db):
    """写入一条记录再读回，应一致。"""
    from eq.db import execute, execute_write
    execute_write("INSERT INTO watchlist (symbol, reason, tags) VALUES (?, ?, ?)", ("600519.SH", "测试", "unit"))
    rows = execute("SELECT symbol, reason FROM watchlist")
    assert len(rows) == 1
    assert rows[0]["symbol"] == "600519.SH"
    assert rows[0]["reason"] == "测试"


def test_state_conn(tmp_db):
    """get_state_conn 应返回可用的 sqlite 连接。"""
    from eq.db import get_state_conn
    with get_state_conn() as conn:
        # 简单查询
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        tables = [r[0] for r in rows]
        # 10 张业务表 + backtest_runs 都应在
        assert "watchlist" in tables
        assert "portfolio" in tables
        assert "rules" in tables
        assert "ml_models" in tables
        assert "ml_predictions" in tables
        assert "backtest_runs" in tables


def test_cache_conn(tmp_db):
    """get_cache_conn 应返回缓存库连接。"""
    from eq.db import get_cache_conn
    with get_cache_conn() as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        tables = [r[0] for r in rows]
        assert "bar_cache" in tables
