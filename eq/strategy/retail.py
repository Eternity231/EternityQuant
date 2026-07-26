"""散户长线（只做多）实用工具（v0.30 新增）。

**为什么单独开一个模块**

只做多 + 资金量小 + 没时间盯盘，这三个约束会让"通用量化建议"里
很大一部分失效甚至有害。这里放几个针对这个画像**确实有效**的东西：

1. :func:`suggest_positions`  资金量 → 该持几只。小账户分散太多会被
   最低佣金活活吃掉，这是个能算出来的硬约束，不是经验之谈。
2. :func:`market_filter` / :func:`with_market_filter`
   大盘择时闸门。只做多的账户唯一的防守手段就是空仓，所以大盘转弱时
   不持股是个自然的想法。**但实测下来它主要降回撤、不提高收益，
   且对均线长度非常敏感**——务必用 ``eq bt portfolio`` 在你自己的
   标的池上验证，别当成免费的午餐（见函数文档里的实测数据）。
3. :func:`turnover_budget` 按你的资金量算「一年最多换手几次才不亏在成本上」。

这些都不是"更聪明的选股"，而是**减少自己造成的伤害**——
对散户来说后者的边际收益通常大得多。
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import pandas as pd

from eq.backtest.cost import CostModel, for_market
from eq.strategy import BUY, HOLD, SELL

logger = logging.getLogger(__name__)

SignalFunc = Callable[[pd.DataFrame], pd.Series]


# ======================================================================
# 1) 资金量 → 持仓数
# ======================================================================

def suggest_positions(
    capital: float,
    costs: CostModel | None = None,
    *,
    max_cost_ratio: float = 0.0010,
    hard_cap: int = 20,
) -> dict[str, Any]:
    """给定资金量，算出「该持几只」和「单笔至少多少钱」。

    逻辑很直接：最低佣金 5 元意味着**单笔成交金额低于某个数，实际费率就
    会失控**。要让单边费率不超过 ``max_cost_ratio``（默认万 10），
    单笔金额至少要 ``min_commission / max_cost_ratio``。
    再用总资金除以它，就是能持有的只数上限。

    Args:
        max_cost_ratio: 能接受的单边费率上限。万 10 是个合理的心理线——
            再高的话一个来回光成本就吃掉 0.2%+，短线基本没法做

    Returns:
        ``{"capital", "min_ticket", "max_positions", "per_position",
        "cost_ratio_at_ticket", "breakeven_round_trip", "note"}``
    """
    costs = costs or for_market("A")
    if capital <= 0:
        raise ValueError("资金量必须为正")

    if costs.min_commission > 0:
        min_ticket = costs.min_commission / max(max_cost_ratio - costs.commission_rate, 1e-9)
        min_ticket = max(min_ticket, costs.min_commission / max_cost_ratio)
    else:
        min_ticket = 0.0

    n = int(capital // min_ticket) if min_ticket > 0 else hard_cap
    n = max(1, min(n, hard_cap))
    per = capital / n

    if n == 1:
        note = (f"资金 {capital:,.0f} 元只够 1 只票的合理仓位。"
                f"分散不了就别硬分散——不如买宽基 ETF（费率低、天然分散）。")
    elif n < 5:
        note = (f"只能有效持 {n} 只。集中度高是这个资金量的客观限制，"
                f"更该靠止损而不是靠分散来控风险。")
    else:
        note = f"可持 {n} 只，单只约 {per:,.0f} 元，费率控制在万 {max_cost_ratio * 1e4:.0f} 以内。"

    return {
        "capital": float(capital),
        "min_ticket": float(min_ticket),
        "max_positions": n,
        "per_position": float(per),
        "cost_ratio_at_ticket": float(costs.cost_ratio(per, "buy")),
        "breakeven_round_trip": float(costs.breakeven_pct(per)),
        "note": note,
    }


def turnover_budget(
    capital: float,
    n_positions: int,
    costs: CostModel | None = None,
    *,
    annual_cost_budget: float = 0.02,
) -> dict[str, Any]:
    """一年最多换手几次，成本才不超过资金的 ``annual_cost_budget``。

    散户最常见的自我伤害就是**换手过高**。这个函数把"少交易"从一句
    口号变成一个可执行的数字：一年 N 次，超了就是在给券商和税务打工。
    """
    costs = costs or for_market("A")
    if n_positions <= 0:
        raise ValueError("持仓数必须为正")
    per = capital / n_positions
    round_trip = costs.round_trip_ratio(per) + 2 * costs.slippage_rate
    if round_trip <= 0:
        return {"round_trips_per_year": float("inf"), "cost_per_round_trip": 0.0}
    trips = annual_cost_budget / round_trip
    return {
        "capital": float(capital),
        "n_positions": n_positions,
        "per_position": float(per),
        "cost_per_round_trip": float(round_trip),
        "round_trips_per_year": float(trips),
        "avg_hold_days": float(252 / max(trips, 1e-9)),
        "note": (f"单只 {per:,.0f} 元，一个来回成本 {round_trip * 100:.3f}%。"
                 f"要把年成本控制在 {annual_cost_budget:.0%} 以内，"
                 f"每只票一年最多换 {trips:.1f} 次，即平均持有 "
                 f"{252 / max(trips, 1e-9):.0f} 个交易日。"),
    }


# ======================================================================
# 2) 大盘择时闸门
# ======================================================================

def market_filter(
    index_bars: pd.DataFrame,
    *,
    ma_period: int = 200,
    confirm_days: int = 3,
) -> pd.Series:
    """大盘闸门：指数在长期均线之上才允许持股。

    只做多的账户唯一的防守手段是空仓，所以「指数跌破长期均线就清仓」
    是个流传很广的规则。**但本项目实测不支持"它必然更好"这个说法。**

    在 15 只 A 股 × 900 根 bar（2022-11~2026-07）、被动等权持有上的对照：

    ========  =========  =========  =========
    闸门        总收益     最大回撤    换手/年
    ========  =========  =========  =========
    无          -17.84%    -32.89%       0.8
    MA100       -26.07%    -29.76%       2.6
    MA150       -31.50%    -34.98%       2.5
    MA200       -22.15%    -26.13%       1.6
    MA250       -16.28%    -20.43%       1.0
    ========  =========  =========  =========

    结论：**回撤基本都改善了（这是它的本职），但收益没有变好**，
    而且换手上升（在均线附近来回被打），对均线长度极其敏感——
    MA150 和 MA250 差了 15 个百分点，这种参数敏感度本身就是危险信号。

    所以它是**风险管理工具**，不是收益增强工具。用之前请在你自己的
    标的池上跑 ``eq bt portfolio`` 验证。

    Args:
        ma_period: 均线长度。200 日（约一年）是经典值；A 股波动大，
            120/150 日反应更快但假信号更多
        confirm_days: 需要连续站上/跌破几天才切换，避免在均线附近反复横跳

    Returns:
        布尔序列，True = 允许持股
    """
    close = index_bars["close"]
    ma = close.rolling(ma_period, min_periods=max(20, ma_period // 4)).mean()
    above = close > ma
    if confirm_days > 1:
        # 连续 N 天同向才认
        up = above.rolling(confirm_days).sum() == confirm_days
        down = (~above).rolling(confirm_days).sum() == confirm_days
        state = pd.Series(np.nan, index=close.index, dtype="float64")
        state[up] = 1.0
        state[down] = 0.0
        allow = state.ffill().fillna(0.0) > 0
    else:
        allow = above.fillna(False)
    return allow.rename("market_ok")


def with_market_filter(
    strategy: SignalFunc,
    index_bars: pd.DataFrame,
    *,
    ma_period: int = 200,
    confirm_days: int = 3,
) -> SignalFunc:
    """给任意策略套上大盘闸门：大盘转弱时**强制清仓**并停止买入。

    注意是「强制清仓」不是「不再买入」——只做多的账户在熊市里，
    继续持有已有仓位和新买入一样危险。
    """
    allow = market_filter(index_bars, ma_period=ma_period, confirm_days=confirm_days)

    def _fn(df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(strategy(df)).reindex(df.index)
        ok = allow.reindex(df.index.union(allow.index)).ffill().reindex(df.index).fillna(False)
        is_ternary = sig.dtype == object or sig.isin([BUY, SELL, HOLD]).any()
        if is_ternary:
            # 关键：闸门要作用在**持仓状态**上，不能直接改三态信号。
            # 三态信号只在穿越时发 BUY，若熊市里把 BUY 吞掉、牛市回来时
            # 策略正处在"已持有"的静默状态，就再也不会重新入场——
            # 闸门会变成一个只会卖不会买的单向阀门（实测收益因此暴跌）。
            state = pd.Series(np.nan, index=df.index, dtype="float64")
            state[sig == BUY] = 1.0
            state[sig == SELL] = 0.0
            held = state.ffill().fillna(0.0)
        else:
            held = pd.to_numeric(sig, errors="coerce").fillna(0.0).clip(0.0, 1.0)
        # 闸门关闭 → 目标仓位归零；闸门重开 → 恢复策略本来的持仓状态
        return (held * ok.astype("float64")).rename("market_filtered")

    _fn.__name__ = f"mkt({getattr(strategy, '__name__', '?')})"
    return _fn


def load_index_bars(symbol: str = "000300.SH", days: int = 900) -> pd.DataFrame:
    """拉一条指数做大盘闸门用。缺省沪深 300。

    A 股常用：``000300.SH`` 沪深300 / ``000905.SH`` 中证500 /
    ``000001.SH`` 上证指数。
    """
    from eq.data.market import get_recent_bars

    return get_recent_bars(symbol, days=days)


# ======================================================================
# 3) 一体化建议
# ======================================================================

def advise(capital: float, market: str = "A",
           annual_cost_budget: float = 0.02) -> dict[str, Any]:
    """给定资金量，一次算出持仓数 / 单笔金额 / 换手预算。"""
    costs = for_market(market)
    pos = suggest_positions(capital, costs)
    budget = turnover_budget(capital, pos["max_positions"], costs,
                             annual_cost_budget=annual_cost_budget)
    return {"costs": costs, "positions": pos, "turnover": budget}


def format_advice(a: dict[str, Any]) -> str:
    """把 :func:`advise` 的结果排成人话。"""
    p, t, c = a["positions"], a["turnover"], a["costs"]
    lines = [f"\n资金 {p['capital']:,.0f} 元（{c.label}成本模型）\n"]
    lines.append(f"  建议持仓数        {p['max_positions']} 只")
    lines.append(f"  单只金额          {p['per_position']:,.0f} 元")
    lines.append(f"  单笔最低金额      {p['min_ticket']:,.0f} 元（低于此值费率失控）")
    lines.append(f"  单边实际费率      万 {p['cost_ratio_at_ticket'] * 1e4:.1f}")
    lines.append(f"  一个来回要涨      {p['breakeven_round_trip'] * 100:.3f}% 才回本")
    lines.append(f"\n  换手预算（年成本≤{2:.0f}%）")
    lines.append(f"    每只每年最多换  {t['round_trips_per_year']:.1f} 次")
    lines.append(f"    平均持有        {t['avg_hold_days']:.0f} 个交易日")
    lines.append(f"\n  {p['note']}")
    return "\n".join(lines) + "\n"
