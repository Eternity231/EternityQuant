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
        assert "cache_meta" in tables


def test_with_block_closes_connection(tmp_db):
    """`with get_state_conn()` 退出必须关连接。

    sqlite3.Connection 原生的上下文管理器只 commit/rollback、**不关连接**，
    而项目里每次 CRUD 都走 `with get_state_conn()`——此前每次调用泄漏一个连接
    和一个文件句柄，长跑的 scheduler daemon / Streamlit 会话最终会耗尽句柄。
    """
    import sqlite3

    from eq.db import get_state_conn

    with get_state_conn() as conn:
        conn.execute("SELECT 1")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        conn.execute("SELECT 1")


def test_with_block_closes_connection_on_exception(tmp_db):
    """块内抛异常也要关，否则出错路径照样泄漏。"""
    import sqlite3

    from eq.db import get_state_conn

    with pytest.raises(RuntimeError):
        with get_state_conn() as conn:
            raise RuntimeError("boom")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        conn.execute("SELECT 1")


def test_repeated_writes_do_not_leak_connections(tmp_db):
    """连续 200 次写不该累积未关闭的连接对象。"""
    import gc
    import sqlite3

    from eq.db import execute_write

    for i in range(200):
        execute_write("INSERT INTO watchlist (symbol) VALUES (?)", (f"T{i:06d}.SH",))
    gc.collect()
    alive = [o for o in gc.get_objects()
             if isinstance(o, sqlite3.Connection) and _is_open(o)]
    assert len(alive) < 10, f"疑似连接泄漏，仍有 {len(alive)} 个打开的连接"


def _is_open(conn) -> bool:
    import sqlite3

    try:
        conn.execute("SELECT 1")
        return True
    except sqlite3.ProgrammingError:
        return False
    except sqlite3.Error:
        return True


def test_execute_many(tmp_db):
    """批量写接口。"""
    from eq.db import execute, execute_many

    n = execute_many(
        "INSERT INTO watchlist (symbol, tags) VALUES (?, ?)",
        [("600519.SH", "a"), ("000001.SZ", "b"), ("00700.HK", "c")],
    )
    assert n == 3
    assert len(execute("SELECT symbol FROM watchlist")) == 3
    assert execute_many("INSERT INTO watchlist (symbol) VALUES (?)", []) == 0
