"""监控规则引擎（problem 11 冶议：硬编码规则类型枚举）。

规则类型（第一版）：
- price_cross     突破/跌破某价位          params: {"level": float, "direction": "up"|"down"}
- price_pct       涨跌幅超阈值             params: {"threshold": float}  # 百分数，如 5.0
- indicator       RSI/MACD/KDJ 触发        params: {"name": "rsi", "period": 14, "level": 30|70, "action": "buy"|"sell"}
- volume_spike    成交量异常放大           params: {"multiple": float}  # 当日量 / N日均量
- limit_up        涨停                    params: {}
- limit_down      跌停                    params: {}
- news            个股新闻推送             params: {}
- event           事件日（财报/解禁/分红） params: {"event_type": str, "date": "YYYY-MM-DD"}
- flow            资金流异动（北向/龙虎）  params: {"source": "northbound"|"dragon", "threshold": float}
- stop_loss       持仓止损价触发           params: {}  # 自动关联 portfolio.stop_loss
- take_profit     持仓止盈价触发           params: {}  # 自动关联 portfolio.take_profit

规则由 monitor 命令注册，由 monitor run 命令逐根评估并触发推送。
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from typing import Any

import pandas as pd

from eq.core.notifier import dispatch
from eq.data.market import detect_market, get_recent_bars, get_snapshot
from eq.db import execute, execute_write, get_state_conn


# ----------------- 规则类型枚举（注册时校验） -----------------

RULE_TYPES = {
    "price_cross", "price_pct", "indicator", "volume_spike",
    "limit_up", "limit_down", "news", "event", "flow",
    "stop_loss", "take_profit",
}


def _validate_params(rule_type: str, params: dict[str, Any]) -> None:
    """轻量参数校验，避免显式 6 大类型类。第一版只校验关键字段存在。"""
    required = {
        "price_cross": ["level", "direction"],
        "price_pct": ["threshold"],
        "indicator": ["name", "action"],
        "volume_spike": ["multiple"],
        "flow": ["source"],
        "event": ["event_type", "date"],
    }.get(rule_type, [])
    for k in required:
        if k not in params:
            raise ValueError(f"规则 {rule_type} 缺参数 {k}")


# ----------------- CRUD -----------------

def add_rule(
    symbol: str | None,
    rule_type: str,
    params: dict[str, Any],
    channels: list[str] | None = None,
) -> int:
    """注册监控规则。symbol=None 表示全市场规则。返回 rule_id。"""
    if rule_type not in RULE_TYPES:
        raise ValueError(f"未知规则类型 {rule_type}，可选：{sorted(RULE_TYPES)}")
    _validate_params(rule_type, params)
    if channels is None:
        channels = ["desktop"]  # 默认仅桌面通知
    row_id = execute_write(
        """INSERT INTO rules (symbol, type, params, channels) VALUES (?, ?, ?, ?)""",
        (symbol, rule_type, json.dumps(params, ensure_ascii=False), json.dumps(channels)),
    )
    return row_id


def remove_rule(rule_id: int) -> bool:
    with get_state_conn() as conn:
        cur = conn.execute("DELETE FROM rules WHERE id = ?", (rule_id,))
        conn.commit()
        return cur.rowcount > 0


def set_enabled(rule_id: int, enabled: bool) -> bool:
    with get_state_conn() as conn:
        cur = conn.execute("UPDATE rules SET enabled = ? WHERE id = ?", (1 if enabled else 0, rule_id))
        conn.commit()
        return cur.rowcount > 0


def list_rules(enabled_only: bool = False) -> list[dict[str, Any]]:
    """列出所有规则，按 id 升序。"""
    q = "SELECT id, symbol, type, params, channels, enabled, created_at, last_fired_at, fire_count FROM rules"
    if enabled_only:
        q += " WHERE enabled = 1"
    q += " ORDER BY id"
    rows = execute(q)
    out = []
    for r in rows:
        d = {k: r[k] for k in r.keys()}
        d["params"] = json.loads(d["params"] or "{}")
        d["channels"] = json.loads(d["channels"] or "[]")
        out.append(d)
    return out


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {k: row[k] for k in row.keys()}


def _mark_fired(rule_id: int) -> None:
    """记录最近触发时间并累加触发次数。"""
    execute_write(
        "UPDATE rules SET last_fired_at = CURRENT_TIMESTAMP, fire_count = fire_count + 1 WHERE id = ?",
        (rule_id,),
    )


# ----------------- 评估引擎 -----------------

def run_all() -> int:
    """对所有 enabled 规则逐根评估，触发则推送。返回触发条数。"""
    rules = list_rules(enabled_only=True)
    fired = 0
    for rule in rules:
        try:
            if _evaluate(rule):
                fired += 1
        except Exception as e:
            print(f"[monitor] 规则 #{rule['id']} 评估异常：{e}")
    return fired


def _evaluate(rule: dict[str, Any]) -> bool:
    """评估单条规则，返回是否触发。"""
    handler = _HANDLERS.get(rule["type"])
    if handler is None:
        return False
    fired, title, body = handler(rule)
    if fired:
        dispatch(rule["channels"], title, body, rule_id=rule["id"])
        _mark_fired(rule["id"])
    return fired


# ----------------- 规则处理器 -----------------

def _snap(rule: dict[str, Any]) -> dict[str, Any]:
    """拉最新行情快照。symbol=None 的规则不应用本处理器。"""
    if not rule["symbol"]:
        raise ValueError("全市场规则需独立处理器")
    return get_snapshot(rule["symbol"])


def _h_price_cross(rule: dict[str, Any]) -> tuple[bool, str, str]:
    p = rule["params"]
    snap = _snap(rule)
    cur = snap["close"]
    level = float(p["level"])
    direction = p["direction"]
    fired = (cur >= level and direction == "up") or (cur <= level and direction == "down")
    title = f"价格突破 {rule['symbol']}"
    body = f"当前 {cur:.2f} {'触及上轨' if direction == 'up' else '触及下轨'} {level}\n前收 {snap['prev_close']:.2f}  涨跌 {snap['change_pct']:+.2f}%"
    return fired, title, body


def _h_price_pct(rule: dict[str, Any]) -> tuple[bool, str, str]:
    p = rule["params"]
    snap = _snap(rule)
    pct = abs(snap["change_pct"])
    fired = pct >= float(p["threshold"])
    arrow = "▲" if snap["change_pct"] >= 0 else "▼"
    title = f"涨跌幅异动 {rule['symbol']}"
    body = f"{arrow} {snap['change_pct']:+.2f}%  阈值 ±{p['threshold']}%\n收盘 {snap['close']:.2f}"
    return fired, title, body


def _h_volume_spike(rule: dict[str, Any]) -> tuple[bool, str, str]:
    p = rule["params"]
    df = get_recent_bars(rule["symbol"], days=30)
    if len(df) < 2:
        return False, "", ""
    today_vol = float(df.iloc[-1]["volume"])
    # 当日量 vs 前 N 日均量（不含今日）
    avg = float(df.iloc[:-1]["volume"].tail(20).mean())
    if avg <= 0:
        return False, "", ""
    multiple = today_vol / avg
    fired = multiple >= float(p["multiple"])
    title = f"量能放大 {rule['symbol']}"
    body = f"当日量 {today_vol:.0f}  vs 20日均量 {avg:.0f}\n倍率 {multiple:.2f}x  阈值 {p['multiple']}x"
    return fired, title, body


def _h_limit(rule: dict[str, Any], is_up: bool) -> tuple[bool, str, str]:
    snap = _snap(rule)
    market = detect_market(rule["symbol"])
    if market != "A":
        return False, "", ""  # 涨跌停规则仅适用 A 股
    # A 股涨跌停 ±10%（ST/创业板 20%，简化第一版只用 10%）
    limit_pct = 0.10
    expected = snap["prev_close"] * (1 + (limit_pct if is_up else -limit_pct))
    fired = abs(snap["close"] - expected) < 0.01 or (snap["high"] >= expected if is_up else snap["low"] <= expected)
    title = f"{'涨停' if is_up else '跌停'} {rule['symbol']}"
    body = f"收盘 {snap['close']:.2f}  前收 {snap['prev_close']:.2f}\n预期板价 {expected:.2f}"
    return fired, title, body


def _h_limit_up(rule): return _h_limit(rule, is_up=True)
def _h_limit_down(rule): return _h_limit(rule, is_up=False)


def _h_stop_loss(rule: dict[str, Any]) -> tuple[bool, str, str]:
    from eq.core.portfolio import get_open
    pos = get_open(rule["symbol"])
    if pos is None or pos["stop_loss"] is None:
        return False, "", ""
    snap = _snap(rule)
    fired = snap["low"] <= pos["stop_loss"]
    title = f"止损价触发 {rule['symbol']}"
    body = f"今日低 {snap['low']:.2f} ≤ 止损 {pos['stop_loss']:.2f}\n成本 {pos['cost_price']:.2f}  持仓 {pos['shares']:.0f} 股"
    return fired, title, body


def _h_take_profit(rule: dict[str, Any]) -> tuple[bool, str, str]:
    from eq.core.portfolio import get_open
    pos = get_open(rule["symbol"])
    if pos is None or pos["take_profit"] is None:
        return False, "", ""
    snap = _snap(rule)
    fired = snap["high"] >= pos["take_profit"]
    title = f"止盈价触发 {rule['symbol']}"
    body = f"今日高 {snap['high']:.2f} ≥ 止盈 {pos['take_profit']:.2f}\n成本 {pos['cost_price']:.2f}  持仓 {pos['shares']:.0f} 股"
    return fired, title, body


# ---------- indicator 处理器（无网络依赖，纯因子计算） ----------

def _h_indicator(rule: dict[str, Any]) -> tuple[bool, str, str]:
    """RSI/MACD/KDJ 因子触发。
    params: {"name": "rsi"|"macd"|"kdj", "period": 14, "level": 30|70, "action": "buy"|"sell"}
    """
    from eq.strategy.factors.technical import kdj, macd, rsi

    p = rule["params"]
    name = p.get("name", "rsi")
    action = p.get("action", "buy")
    period = p.get("period", 14)
    level = p.get("level", 30)  # RSI 30=超卖(buy), 70=超买(sell)
    df = get_recent_bars(rule["symbol"], days=period + 10)
    if len(df) < period + 2:
        return False, "", ""
    fired = False
    cur_val = 0.0
    prev_val = 0.0

    if name == "rsi":
        rsi_vals = rsi(df, period=period)
        cur_val = float(rsi_vals.iloc[-1])
        prev_val = float(rsi_vals.iloc[-2])
        if action == "buy":
            fired = prev_val <= level and cur_val > level  # 回升进入
        else:
            fired = prev_val >= level and cur_val < level  # 回落进入
    elif name == "macd":
        macd_df = macd(df, fast=12, slow=26, signal=9)
        dif = macd_df["dif"]
        dea = macd_df["dea"]
        if action == "buy":
            fired = dif.iloc[-2] <= dea.iloc[-2] and dif.iloc[-1] > dea.iloc[-1]  # 金叉
        else:
            fired = dif.iloc[-2] >= dea.iloc[-2] and dif.iloc[-1] < dea.iloc[-1]  # 死叉
        cur_val = float(dif.iloc[-1])
    elif name == "kdj":
        kdj_df = kdj(df, n=9, m1=3, m2=3)
        k = kdj_df["K"]
        cur_val = float(k.iloc[-1])
        prev_val = float(k.iloc[-2])
        if action == "buy":
            fired = prev_val <= level and cur_val > level
        else:
            fired = prev_val >= (100 - level) and cur_val < (100 - level)

    if not fired:
        return False, "", ""
    action_label = "买入" if action == "buy" else "卖出"
    title = f"指标触发 {rule['symbol']}"
    body = f"{name.upper()} {action_label}信号\n当前值 {cur_val:.2f}  前值 {prev_val:.2f}  阈值 {level}"
    return True, title, body


# ---------- news 处理器（akshare 东财新闻） ----------

def _h_news(rule: dict[str, Any]) -> tuple[bool, str, str]:
    """个股新闻推送。symbol 必填。"""
    if not rule["symbol"]:
        return False, "", ""
    import akshare as ak
    try:
        df = ak.stock_news_em(symbol=rule["symbol"])
    except Exception:
        return False, "", ""
    if df.empty:
        return False, "", ""
    # 最近一条新闻
    latest = df.iloc[0]
    title = f"新闻推送 {rule['symbol']}"
    body = f"{latest['新闻标题']}\n来源：{latest['文章来源']}  {latest['发布时间']}"
    return True, title, body


# ---------- event 处理器（事件日提醒） ----------

def _h_event(rule: dict[str, Any]) -> tuple[bool, str, str]:
    """事件日提醒。
    params: {"event_type": "financial_report"|"lockup_expiry"|"dividend",
             "date": "YYYY-MM-DD", "name": "中报披露"}
    """
    p = rule["params"]
    event_date = p.get("date", "")
    event_type = p.get("event_type", "")
    event_name = p.get("name", event_type)
    if not event_date:
        return False, "", ""
    try:
        target = dt.date.fromisoformat(event_date)
    except ValueError:
        return False, "", ""
    today = dt.date.today()
    diff = (target - today).days
    # 当天或明天触发
    if diff < 0:
        return False, "", ""  # 已过
    if diff > 1:
        return False, "", ""  # 还早
    title = f"事件提醒 {rule['symbol']}" if rule["symbol"] else "事件提醒"
    sym = f"{rule['symbol']} " if rule["symbol"] else ""
    when = "今天" if diff == 0 else "明天"
    body = f"{sym}{event_name} {when}（{event_date}）"
    return True, title, body


# ---------- flow 处理器（北向资金流异动） ----------

def _h_flow(rule: dict[str, Any]) -> tuple[bool, str, str]:
    """资金流异动。
    params: {"source": "northbound", "threshold": 100000000}  # 北向净流入阈值（元）
    """
    p = rule["params"]
    source = p.get("source", "northbound")
    threshold = float(p.get("threshold", 100_000_000))  # 默认 1 亿
    kwargs = {}

    if source == "northbound":
        import akshare as ak
        try:
            df = ak.stock_hsgt_hist_em(symbol="北向资金")
        except Exception:
            return False, "", ""
        if df.empty or len(df) < 2:
            return False, "", ""
        # 最新一日
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        net_inflow = float(latest["当日成交净买额"]) * 1e4  # 亿元 → 元
        prev_inflow = float(prev["当日成交净买额"]) * 1e4
        fired = abs(net_inflow) >= threshold
        direction = "流入" if net_inflow >= 0 else "流出"
        title = f"北向资金异动"
        body = (
            f"当日净{direction} {abs(net_inflow) / 1e8:.2f}亿  "
            f"（前日 {abs(prev_inflow) / 1e8:.2f}亿）\n"
            f"阈值 {threshold / 1e8:.0f}亿  累计净买额 {float(latest['历史累计净买额']) * 1e4 / 1e8:.0f}亿"
        )
        return fired, title, body
    elif source == "dragon":
        # 龙虎榜（第一版占位）
        return False, "", "龙虎榜 monitor 待集成"
    return False, "", ""


# 占位处理器（第一版返回 False，后续集成）
def _h_noop(rule: dict[str, Any]) -> tuple[bool, str, str]:
    return False, "", ""


_HANDLERS = {
    "price_cross": _h_price_cross,
    "price_pct": _h_price_pct,
    "volume_spike": _h_volume_spike,
    "limit_up": _h_limit_up,
    "limit_down": _h_limit_down,
    "stop_loss": _h_stop_loss,
    "take_profit": _h_take_profit,
    "indicator": _h_indicator,
    "news": _h_news,
    "event": _h_event,
    "flow": _h_flow,
}
