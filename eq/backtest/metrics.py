"""回测绩效指标（双引擎共用）。

此前两个引擎各自复制了一份 ``_compute_metrics``，同样的两个 bug 也复制了两份：

1. ``(1 + total_return) ** (1 / years)`` 在总收益 ≤ -100% 时是负数开分数次方，
   结果是 ``nan``（复数），报表里直接显示 ``nan%``。
2. ``years = max(n_days / 252, 1e-9)``——回测 20 天时 years≈0.079，年化按
   12.6 次方外推，一个 +3% 的短窗口回测会报出 +45% 的"年化"。下限拉到
   1/12 年（约 21 个交易日），短于此不做年化外推。

顺带补齐了散户实际要看的几个指标：Sortino / Calmar / 盈亏比 / 最长回撤天数。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
# 年化外推的最小窗口：少于 21 个交易日（约 1 个月）不外推，直接报区间收益
_MIN_YEARS = 1.0 / 12


def annualize(total_return: float, n_periods: int, periods_per_year: int = TRADING_DAYS) -> float:
    """把区间总收益年化。亏光（≤ -100%）或窗口过短时安全退化。"""
    if not np.isfinite(total_return) or total_return <= -1.0:
        return -1.0
    years = n_periods / periods_per_year
    if years < _MIN_YEARS:
        return float(total_return)  # 窗口太短，不做年化外推
    return float((1.0 + total_return) ** (1.0 / years) - 1.0)


def max_drawdown(equity: pd.Series) -> tuple[float, int]:
    """最大回撤（负数）与最长回撤持续期（bar 数）。"""
    if equity.empty:
        return 0.0, 0
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0, np.nan)
    max_dd = float(dd.min()) if len(dd.dropna()) else 0.0
    # 最长回撤：连续处于水下（equity < peak）的最大长度
    underwater = (equity < peak).to_numpy()
    longest = cur = 0
    for flag in underwater:
        cur = cur + 1 if flag else 0
        longest = max(longest, cur)
    return (max_dd if np.isfinite(max_dd) else 0.0), longest


def compute_metrics(equity: pd.Series, trades: pd.DataFrame) -> dict:
    """从权益曲线 + 交易明细算全套绩效指标。"""
    equity = equity.dropna()
    if equity.empty or float(equity.iloc[0]) == 0:
        return {
            "total_return": 0.0, "annual_return": 0.0, "sharpe": 0.0, "sortino": 0.0,
            "calmar": 0.0, "max_drawdown": 0.0, "max_dd_days": 0, "volatility": 0.0,
            "win_rate": 0.0, "profit_factor": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "num_trades": 0, "num_bars": 0,
        }

    total_return = float(equity.iloc[-1]) / float(equity.iloc[0]) - 1.0
    n = len(equity)
    annual_return = annualize(total_return, n)

    daily = equity.pct_change().dropna()
    std = float(daily.std())
    vol = std * np.sqrt(TRADING_DAYS) if std > 0 else 0.0
    sharpe = float(daily.mean() / std * np.sqrt(TRADING_DAYS)) if std > 0 else 0.0
    # Sortino：只惩罚下行波动（散户更关心的"亏得多不多"）
    downside = daily[daily < 0]
    dstd = float(downside.std()) if len(downside) > 1 else 0.0
    sortino = float(daily.mean() / dstd * np.sqrt(TRADING_DAYS)) if dstd > 0 else 0.0

    max_dd, dd_days = max_drawdown(equity)
    calmar = float(annual_return / abs(max_dd)) if max_dd < 0 else 0.0

    win_rate = profit_factor = avg_win = avg_loss = 0.0
    if trades is not None and not trades.empty and "pnl_pct" in trades.columns:
        pnl = pd.to_numeric(trades["pnl_pct"], errors="coerce").dropna()
        if len(pnl):
            wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
            win_rate = float(len(wins) / len(pnl))
            avg_win = float(wins.mean()) if len(wins) else 0.0
            avg_loss = float(losses.mean()) if len(losses) else 0.0
            gross_loss = float(-losses.sum())
            profit_factor = float(wins.sum() / gross_loss) if gross_loss > 0 else float("inf") if len(wins) else 0.0

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "max_dd_days": dd_days,
        "volatility": float(vol),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "num_trades": int(len(trades)) if trades is not None else 0,
        "num_bars": n,
    }
