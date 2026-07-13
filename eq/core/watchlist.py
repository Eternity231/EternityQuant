"""自选股管理（CRUD）：watchlist 表。

watchlist 表结构（problem 13 决议，A 方案独立于 portfolio）：
    id, symbol(UNIQUE), name, market, added_at, reason, tags
"""

from __future__ import annotations

import sqlite3
from typing import Any

from eq.data.market import detect_market
from eq.db import execute, execute_write, get_state_conn


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {k: row[k] for k in row.keys()}


def add(symbol: str, reason: str = "", tags: str = "") -> int:
    """加入自选。重复符号静默忽略（INSERT OR IGNORE）。返回 rowid（0 表示已存在）。"""
    try:
        market = detect_market(symbol)
    except ValueError:
        market = None  # 不识别的市场也允许加，只是 market 列为 NULL
    return execute_write(
        "INSERT OR IGNORE INTO watchlist (symbol, market, reason, tags) VALUES (?, ?, ?, ?)",
        (symbol, market, reason or None, tags or None),
    )


def remove(symbol: str) -> bool:
    """移出自选。返回是否真的删了一行。"""
    with get_state_conn() as conn:
        cur = conn.execute("DELETE FROM watchlist WHERE symbol = ?", (symbol,))
        conn.commit()
        return cur.rowcount > 0


def list_all() -> list[dict[str, Any]]:
    """列出全部自选，按加入时间倒序。"""
    rows = execute("SELECT id, symbol, name, market, added_at, reason, tags FROM watchlist ORDER BY added_at DESC")
    return [_row_to_dict(r) for r in rows]


def find(symbol: str) -> dict[str, Any] | None:
    """查单只自选。返回 None 表示不在自选。"""
    rows = execute(
        "SELECT id, symbol, name, market, added_at, reason, tags FROM watchlist WHERE symbol = ?",
        (symbol,),
    )
    return _row_to_dict(rows[0]) if rows else None


def update_name(symbol: str, name: str) -> None:
    """缓存股票名称（数据层拉到名称后调）。"""
    execute_write("UPDATE watchlist SET name = ? WHERE symbol = ?", (name, symbol))
