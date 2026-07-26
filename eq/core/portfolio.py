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

from eq.data.market import detect_market, normalize_symbol
from eq.db import execute, execute_write, get_state_conn


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return dict(row)


def _norm(symbol: str) -> str:
    """符号规整。``600519``/``600519.sh``/``SH600519`` 都落到同一行持仓，
    否则同一只票会因写法不同建出多条 open 持仓。"""
    try:
        return normalize_symbol(symbol)
    except ValueError:
        return str(symbol).strip().upper()


def open_position(
    symbol: str,
    shares: float,
    price: float,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    note: str = "",
) -> int:
    """建仓。若已存在 open 持仓则改用 add 加仓，避免 UNIQUE 冲突。"""
    symbol = _norm(symbol)
    if shares <= 0:
        raise ValueError(f"建仓股数必须为正：{shares}")
    if price <= 0:
        raise ValueError(f"成交价必须为正：{price}")
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
    symbol = _norm(symbol)
    if shares <= 0:
        raise ValueError(f"加仓股数必须为正：{shares}")
    if price <= 0:
        raise ValueError(f"成交价必须为正：{price}")
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
    symbol = _norm(symbol)
    if shares <= 0:
        raise ValueError(f"减仓股数必须为正：{shares}")
    if price <= 0:
        raise ValueError(f"成交价必须为正：{price}")
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
    symbol = _norm(symbol)
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
    symbol = _norm(symbol)
    rows = execute(
        """SELECT id, symbol, name, market, shares, cost_price, opened_at,
                  stop_loss, take_profit, status, realized_pnl
           FROM portfolio WHERE symbol = ? AND status = 'open'""",
        (symbol,),
    )
    return _row_to_dict(rows[0]) if rows else None


def trade_history(symbol: str, limit: int = 50) -> list[dict[str, Any]]:
    """查某标的全部交易历史。"""
    symbol = _norm(symbol)
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


def summary() -> dict[str, Any]:
    """持仓体检：一次性算全持仓的盈亏/占比/距止损止盈距离/今日涨跌。

    返回字典：
        ``positions``: 每只持仓的明细（含当前价/总市值/浮盈/浮盈%/
            距止损%/距止盈%/今日涨跌%/仓位占比%/风险敞口）
        ``total_market_value``: 全持仓总市值
        ``total_cost``: 全持仓成本
        ``total_unrealized_pnl``: 全持仓浮盈合计
        ``total_realized_pnl``: 全持仓已实现盈亏合计
        ``total_today_pnl``: 全持仓今日盈亏合计（按市值 * 今日涨跌%）
        ``max_weight_pct`` / ``max_weight_symbol``: 最大单票集中度
        ``risk_at_stop``: 全部触发止损时的合计亏损（未设止损的按 0 计）
        ``no_stop``: 没设止损的标的列表（裸奔仓位）
        ``stale``: 行情拉取失败、只能用成本价占位的标的列表
    """
    from eq.data.market import get_snapshots

    rows = list_open()
    positions: list[dict[str, Any]] = []
    total_mv = 0.0
    total_cost = 0.0
    total_unrealized = 0.0
    total_realized = 0.0
    total_today = 0.0
    risk_at_stop = 0.0
    no_stop: list[str] = []
    stale: list[str] = []

    # 并发拉全部行情：持仓 20 只时串行要 20 次网络往返，体检命令要等半分钟
    snaps = get_snapshots([r["symbol"] for r in rows]) if rows else {}

    for r in rows:
        sym = r["symbol"]
        shares = float(r["shares"] or 0)
        cost = float(r["cost_price"] or 0)
        stop = r.get("stop_loss")
        target = r.get("take_profit")
        realized = float(r["realized_pnl"] or 0)

        snap = snaps.get(sym)
        if snap is None:
            close = cost  # 拉不到行情时退化为用成本价占位
            change_pct = 0.0
            stale.append(sym)
        else:
            close = float(snap["close"])
            change_pct = float(snap["change_pct"])

        mv = shares * close
        unrealized = (close - cost) * shares
        unrealized_pct = (close - cost) / cost * 100 if cost else 0.0
        today_pnl = mv * change_pct / 100
        dist_stop = ((close - stop) / close * 100) if (stop and close) else None
        dist_target = ((target - close) / close * 100) if (target and close) else None

        positions.append({
            "symbol": sym,
            "name": r.get("name") or "",
            "market": r.get("market") or "",
            "shares": shares,
            "cost_price": cost,
            "current_price": close,
            "market_value": mv,
            "unrealized_pnl": unrealized,
            "unrealized_pct": unrealized_pct,
            "today_pct": change_pct,
            "today_pnl": today_pnl,
            "stop_loss": stop,
            "take_profit": target,
            "dist_to_stop_pct": dist_stop,
            "dist_to_target_pct": dist_target,
            "realized_pnl": realized,
            "weight_pct": 0.0,      # 总市值算完后回填
            "risk_at_stop": (close - stop) * shares if stop else None,
            "quote_ok": snap is not None,
        })
        total_mv += mv
        total_cost += cost * shares
        total_unrealized += unrealized
        total_realized += realized
        total_today += today_pnl
        if stop:
            risk_at_stop += max(0.0, (close - float(stop)) * shares)
        else:
            no_stop.append(sym)

    # 回填仓位占比（集中度体检：单票超 30% 就该警觉了）
    for p in positions:
        p["weight_pct"] = (p["market_value"] / total_mv * 100) if total_mv else 0.0

    heaviest = max(positions, key=lambda p: p["weight_pct"], default=None)

    return {
        "positions": positions,
        "total_market_value": total_mv,
        "total_cost": total_cost,
        "total_unrealized_pnl": total_unrealized,
        "total_unrealized_pct": (total_unrealized / total_cost * 100) if total_cost else 0.0,
        "total_realized_pnl": total_realized,
        "total_today_pnl": total_today,
        "max_weight_pct": heaviest["weight_pct"] if heaviest else 0.0,
        "max_weight_symbol": heaviest["symbol"] if heaviest else "",
        "risk_at_stop": risk_at_stop,
        "no_stop": no_stop,
        "stale": stale,
    }
