"""自选股管理（CRUD）：watchlist 表。

watchlist 表结构（problem 13 决议，A 方案独立于 portfolio）：
    id, symbol(UNIQUE), name, market, added_at, reason, tags
"""

from __future__ import annotations

import sqlite3
from typing import Any

from eq.data.market import detect_market, normalize_symbol
from eq.db import execute, execute_write, get_state_conn


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return dict(row)


def _norm(symbol: str) -> str:
    """符号规整。同一只票不同写法（600519 / 600519.sh / SH600519）落到同一行。"""
    try:
        return normalize_symbol(symbol)
    except ValueError:
        return str(symbol).strip().upper()


def add(symbol: str, reason: str = "", tags: str = "") -> int:
    """加入自选。重复符号静默忽略（INSERT OR IGNORE）。返回 rowid（0 表示已存在）。"""
    symbol = _norm(symbol)
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
    symbol = _norm(symbol)
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
    symbol = _norm(symbol)
    rows = execute(
        "SELECT id, symbol, name, market, added_at, reason, tags FROM watchlist WHERE symbol = ?",
        (symbol,),
    )
    return _row_to_dict(rows[0]) if rows else None


def update_name(symbol: str, name: str) -> None:
    """缓存股票名称（数据层拉到名称后调）。"""
    execute_write("UPDATE watchlist SET name = ? WHERE symbol = ?", (name, _norm(symbol)))


def list_by_tag(tag: str) -> list[dict[str, Any]]:
    """按标签过滤自选（tags 列是逗号分隔的字符串）。"""
    tag = tag.strip()
    if not tag:
        return list_all()
    return [
        r for r in list_all()
        if tag in [t.strip() for t in (r["tags"] or "").split(",") if t.strip()]
    ]


def quotes(symbols: list[str] | None = None) -> list[dict[str, Any]]:
    """自选股批量行情：一次并发拉全部自选的最新快照。

    Returns:
        每只：symbol / name / market / tags / close / change_pct / volume / date /
        ``ok``（行情是否拉到）
    """
    from eq.data.market import get_snapshots

    rows = list_all()
    if symbols:
        wanted = {_norm(s) for s in symbols}
        rows = [r for r in rows if r["symbol"] in wanted]
    if not rows:
        return []
    snaps = get_snapshots([r["symbol"] for r in rows])
    out = []
    for r in rows:
        snap = snaps.get(r["symbol"])
        out.append({
            "symbol": r["symbol"],
            "name": r["name"] or "",
            "market": r["market"] or "",
            "tags": r["tags"] or "",
            "close": float(snap["close"]) if snap else None,
            "change_pct": float(snap["change_pct"]) if snap else None,
            "volume": float(snap["volume"]) if snap else None,
            "date": snap["date"] if snap else "",
            "ok": snap is not None,
        })
    out.sort(key=lambda d: (d["change_pct"] is None, -(d["change_pct"] or 0)))
    return out
