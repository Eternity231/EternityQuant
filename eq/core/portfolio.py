"""持仓管理：建仓 / 加仓 / 减仓 / 清仓 + 历史回放。

portfolio 与 trade_history 双表联动（problem 13 决议）：
- portfolio: 当前持仓状态（一行一只标的，status=open/closed）
- trade_history: 每次买卖的明细，支持回放和审计

成本价规则：
- 建仓 (buy)：cost_price = 成交价
- 加仓 (add)：cost_price = 加权平均 (旧成本 * 旧仓 + 新价 * 新仓) / 总仓
- 减仓 (trim)：cost_price 不变，realized_pnl += (成交价 - cost_price) * 减仓股数
- 清仓 (sell)：cost_price 保持，realized_pnl += (成交价 - cost_price) * 剩余仓，status=closed
"""

from __future__ import annotations

import sqlite3
from typing import Any

from eq.data.market import detect_market
from eq.db import execute, execute_write, get_state_conn


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {k: row[k] for k in row.keys()}


def open_position(
    symbol: str,
    shares: float,
    price: float,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    note: str = "",
) -> int:
    """建仓。若已存在 open 持仓则改用 add 加仓，避免 UNIQUE 冲突。"""
    existing = get_open(symbol)
    if existing is not None:
        return add(symbol, shares, price, note=note)
    try:
        market = detect_market(symbol)
    except ValueError:
        market = None
    pos_id = execute_write(
        """INSERT INTO portfolio (symbol, market, shares, cost_price, stop_loss, take_profit, status)
           VALUES (?, ?, ?, ?, ?, ?, 'open')""",
        (symbol, market, shares, price, stop_loss, take_profit),
    )
    _record_trade(symbol, "buy", shares, price, note)
    return pos_id


def add(symbol: str, shares: float, price: float, note: str = "") -> int:
    """加仓。加权平均更新成本价。持仓不存在则抛错。"""
    pos = get_open(symbol)
    if pos is None:
        raise ValueError(f"无 open 持仓：{symbol}")
    old_shares = pos["shares"]
    old_cost = pos["cost_price"]
    new_shares = old_shares + shares
    new_cost = (old_cost * old_shares + price * shares) / new_shares if new_shares else price
    execute_write(
        "UPDATE portfolio SET shares = ?, cost_price = ? WHERE id = ?",
        (new_shares, new_cost, pos["id"]),
    )
    _record_trade(symbol, "add", shares, price, note)
    return pos["id"]


def trim(symbol: str, shares: float, price: float, note: str = "") -> int:
    """减仓。不动成本价，累加已实现盈亏。减到 0 自动转清仓。"""
    pos = get_open(symbol)
    if pos is None:
        raise ValueError(f"无 open 持仓：{symbol}")
    if shares > pos["shares"] + 1e-9:
        raise ValueError(f"减仓股数 {shares} 超过持仓 {pos['shares']}")
    pnl = (price - pos["cost_price"]) * shares
    new_shares = pos["shares"] - shares
    if new_shares <= 1e-9:
        # 全部减完 → 清仓
        execute_write(
            """UPDATE portfolio SET shares = 0, status = 'closed',
               closed_at = CURRENT_TIMESTAMP, realized_pnl = ? WHERE id = ?""",
            (pos["realized_pnl"] + pnl, pos["id"]),
        )
        _record_trade(symbol, "sell", shares, price, note or "清仓")
    else:
        execute_write(
            "UPDATE portfolio SET shares = ?, realized_pnl = ? WHERE id = ?",
            (new_shares, pos["realized_pnl"] + pnl, pos["id"]),
        )
        _record_trade(symbol, "trim", shares, price, note)
    return pos["id"]


def set_stops(symbol: str, stop_loss: float | None = None, take_profit: float | None = None) -> bool:
    """更新止损/止盈价。返回是否真的更新了一行。"""
    sets, params = [], []
    if stop_loss is not None:
        sets.append("stop_loss = ?")
        params.append(stop_loss)
    if take_profit is not None:
        sets.append("take_profit = ?")
        params.append(take_profit)
    if not sets:
        return False
    params.append(symbol)
    with get_state_conn() as conn:
        cur = conn.execute(f"UPDATE portfolio SET {', '.join(sets)} WHERE symbol = ? AND status = 'open'", params)
        conn.commit()
        return cur.rowcount > 0


def list_open() -> list[dict[str, Any]]:
    """列出所有 open 持仓。"""
    rows = execute(
        """SELECT id, symbol, name, market, shares, cost_price, opened_at,
                  stop_loss, take_profit, status, realized_pnl
           FROM portfolio WHERE status = 'open' ORDER BY opened_at DESC"""
    )
    return [_row_to_dict(r) for r in rows]


def list_closed(limit: int = 20) -> list[dict[str, Any]]:
    """列出最近 N 个已清仓持仓（归档查看）。"""
    rows = execute(
        """SELECT id, symbol, name, market, cost_price, opened_at, closed_at, realized_pnl
           FROM portfolio WHERE status = 'closed' ORDER BY closed_at DESC LIMIT ?""",
        (limit,),
    )
    return [_row_to_dict(r) for r in rows]


def get_open(symbol: str) -> dict[str, Any] | None:
    """查单只 open 持仓。"""
    rows = execute(
        """SELECT id, symbol, name, market, shares, cost_price, opened_at,
                  stop_loss, take_profit, status, realized_pnl
           FROM portfolio WHERE symbol = ? AND status = 'open'""",
        (symbol,),
    )
    return _row_to_dict(rows[0]) if rows else None


def trade_history(symbol: str, limit: int = 50) -> list[dict[str, Any]]:
    """查某标的全部交易历史。"""
    rows = execute(
        "SELECT id, symbol, action, shares, price, executed_at, note FROM trade_history WHERE symbol = ? ORDER BY executed_at DESC LIMIT ?",
        (symbol, limit),
    )
    return [_row_to_dict(r) for r in rows]


def _record_trade(symbol: str, action: str, shares: float, price: float, note: str) -> None:
    execute_write(
        "INSERT INTO trade_history (symbol, action, shares, price, note) VALUES (?, ?, ?, ?, ?)",
        (symbol, action, shares, price, note or None),
    )
