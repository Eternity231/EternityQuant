"""每日晨报组装（v0.32 新增）—— ``eq daily`` 的引擎。

散户没时间每天跑七八个命令。晨报把日常需要看的四件事装进一次调用：

1. **大盘状态**：指数相对长期均线的位置（配合 v0.30 的大盘闸门口径）
2. **持仓警报**：谁跌破/逼近止损、谁没设止损
3. **今日信号**：哪些自选股今天新触发买入/卖出（不是"当前处于买入状态"，
   而是**状态翻转的那一天**——只有翻转才需要行动）
4. **纸面记录**：新买入信号自动进 :mod:`eq.core.journal`，到期的自动结算

本模块只做数据组装，格式化在 CLI 层。所有函数都不抛网络异常——
晨报某一块拉不到数据就标注缺失，不能让整份报告失败。
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import pandas as pd

from eq.strategy import BUY, HOLD, SELL

logger = logging.getLogger(__name__)

SignalFunc = Callable[[pd.DataFrame], pd.Series]


def _held_state(sig: pd.Series) -> pd.Series:
    """信号 → 0/1 持有状态（三态 ffill；数值取 >0）。"""
    sig = pd.Series(sig)
    if sig.dtype == object or sig.isin([BUY, SELL, HOLD]).any():
        state = pd.Series(np.nan, index=sig.index, dtype="float64")
        state[sig == BUY] = 1.0
        state[sig == SELL] = 0.0
        return state.ffill().fillna(0.0)
    return (pd.to_numeric(sig, errors="coerce").fillna(0.0) > 0).astype("float64")


def detect_signal_changes(
    bars_by_symbol: dict[str, pd.DataFrame],
    strategy: SignalFunc,
) -> dict[str, str]:
    """逐标的跑策略，返回**最后一根 bar 上的状态变化**。

    只报翻转，不报存量：``enter``（今日新买入）/ ``exit``（今日转卖出）/
    ``holding``（持有中，无动作）/ ``flat``（空仓中，无动作）。

    「今天该做什么」只取决于翻转——这是晨报和回测的根本区别：
    回测关心整条曲线，晨报只关心最后一根。
    """
    out: dict[str, str] = {}
    for sym, df in bars_by_symbol.items():
        if df is None or len(df) < 30:
            continue
        try:
            state = _held_state(pd.Series(strategy(df)).reindex(df.index))
        except Exception as e:
            logger.debug("晨报信号计算失败 %s：%s", sym, e)
            continue
        today = float(state.iloc[-1])
        yesterday = float(state.iloc[-2]) if len(state) >= 2 else 0.0
        if today > 0 and yesterday <= 0:
            out[sym] = "enter"
        elif today <= 0 and yesterday > 0:
            out[sym] = "exit"
        else:
            out[sym] = "holding" if today > 0 else "flat"
    return out


def stop_breaches(
    positions: list[dict[str, Any]],
    near_pct: float = 0.02,
) -> dict[str, list[dict[str, Any]]]:
    """持仓止损体检。输入是 ``eq.core.portfolio.summary()["positions"]``。

    Returns:
        ``{"breached": [...], "near": [...], "no_stop": [...]}``——
        已跌破 / 距止损不足 ``near_pct`` / 压根没设止损。
    """
    breached, near, no_stop = [], [], []
    for p in positions:
        stop = p.get("stop_loss")
        price = p.get("current_price")
        if not stop:
            no_stop.append(p)
            continue
        if not price or price <= 0:
            continue
        if price <= stop:
            breached.append(p)
        elif (price - stop) / price <= near_pct:
            near.append(p)
    return {"breached": breached, "near": near, "no_stop": no_stop}


def market_status(
    index_bars: pd.DataFrame | None,
    ma_period: int = 200,
) -> dict[str, Any] | None:
    """大盘一行状态：现价 / 长期均线 / 偏离 / 闸门开关。拉不到指数返回 None。"""
    if index_bars is None or len(index_bars) < ma_period // 4:
        return None
    from eq.strategy.retail import market_filter

    close = index_bars["close"]
    ma = close.rolling(ma_period, min_periods=max(20, ma_period // 4)).mean()
    allow = market_filter(index_bars, ma_period=ma_period)
    last, prev = float(close.iloc[-1]), float(close.iloc[-2]) if len(close) >= 2 else float(close.iloc[-1])
    return {
        "close": last,
        "change_pct": (last - prev) / prev * 100 if prev else 0.0,
        "ma": float(ma.iloc[-1]) if pd.notna(ma.iloc[-1]) else None,
        "dist_ma_pct": (last / float(ma.iloc[-1]) - 1) * 100 if pd.notna(ma.iloc[-1]) else None,
        "gate_open": bool(allow.iloc[-1]),
        "date": str(index_bars.index[-1].date()),
    }


def build_recos(
    bars_by_symbol: dict[str, pd.DataFrame],
    changes: dict[str, str],
    benchmark_bars: pd.DataFrame | None,
) -> tuple[list[dict[str, Any]], float | None]:
    """把今日的 ``enter`` 信号打包成 journal 可记录的推荐列表。

    入场价 = 该标的最后一根 bar 的收盘价；推荐日 = 该 bar 的日期
    （用行情日期而不是今天的日历日期——周末跑晨报时不会错记日期）。
    """
    recos = []
    for sym, chg in changes.items():
        if chg != "enter":
            continue
        df = bars_by_symbol.get(sym)
        if df is None or df.empty:
            continue
        recos.append({
            "symbol": sym,
            "date": df.index[-1].date(),
            "price": float(df["close"].iloc[-1]),
        })
    bench_price = None
    if benchmark_bars is not None and len(benchmark_bars):
        bench_price = float(benchmark_bars["close"].iloc[-1])
    return recos, bench_price
