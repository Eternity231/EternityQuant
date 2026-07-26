"""ML 预测 → 可回测的策略（v0.29 新增）。

**为什么需要这座桥**

项目里有两套彼此隔离的评估体系：

- **ML 层**：训练模型 → 算 Rank IC / ICIR（``eq.strategy.factors.evaluation``）
- **策略层**：跑回测 → 算夏普 / 回撤（``eq.backtest``）

问题是 **IC 高不等于能赚钱**。IC 衡量的是「预测值和未来收益的秩相关」，
它完全不考虑：交易成本、换手率、能同时持有几只、涨跌停买不进、
信号衰减速度。一个 IC=0.05 但需要每天换手 100% 的模型，
扣掉印花税和最低佣金后大概率是亏的。

所以模型训完必须**真的跑一遍回测**。本模块把 ``ml_predictions`` 表里的
预测分数转成两种可回测的形态：

- :func:`ml_score_matrix`  日期 × 标的的分数矩阵 → 直接喂
  :func:`~eq.backtest.portfolio.run_portfolio`（横截面选股，ML 的主场）
- :func:`ml_signal_for`    绑定单只标的的 ``SignalFunc`` → 喂
  :mod:`eq.backtest.robust` 做单标的稳健性检验

**横截面才是 ML 选股的正确用法**：模型输出的是「这只票相对其他票的强弱」，
「买分数最高的 N 只」是截面排序问题，不是逐标的的择时问题。
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import pandas as pd

from eq.db import execute
from eq.strategy import BUY, HOLD, SELL

logger = logging.getLogger(__name__)


# ======================================================================
# 读预测
# ======================================================================

def load_predictions(
    model_id: str | None = None,
    symbols: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """从 ``ml_predictions`` 表读预测，返回长表 ``[date, symbol, score]``。

    ``model_id=None`` 时取当前激活的模型（任一 universe）。
    """
    q = "SELECT date, symbol, score, model_id FROM ml_predictions WHERE 1=1"
    params: list[Any] = []
    if model_id:
        q += " AND model_id = ?"
        params.append(model_id)
    else:
        q += " AND model_id IN (SELECT id FROM ml_models WHERE is_active = 1)"
    if start:
        q += " AND date >= ?"
        params.append(start)
    if end:
        q += " AND date <= ?"
        params.append(end)
    q += " ORDER BY date, symbol"
    rows = execute(q, tuple(params))
    if not rows:
        return pd.DataFrame(columns=["date", "symbol", "score", "model_id"])
    df = pd.DataFrame([dict(r) for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    if symbols:
        from eq.data.market import normalize_symbol

        want = {normalize_symbol(s) for s in symbols}
        df = df[df["symbol"].map(lambda s: _norm(s) in want)]
    return df.dropna(subset=["score"])


def _norm(sym: str) -> str:
    from eq.data.market import normalize_symbol

    try:
        return normalize_symbol(sym)
    except ValueError:
        return str(sym).strip().upper()


def predictions_wide(model_id: str | None = None, **kw) -> pd.DataFrame:
    """预测长表 → 日期 × 标的的宽表。"""
    long = load_predictions(model_id, **kw)
    if long.empty:
        return pd.DataFrame()
    long = long.copy()
    long["symbol"] = long["symbol"].map(_norm)
    return long.pivot_table(index="date", columns="symbol", values="score", aggfunc="last")


# ======================================================================
# 横截面：分数矩阵
# ======================================================================

def ml_score_matrix(
    predictions: pd.DataFrame | str | None = None,
    *,
    top_n: int = 10,
    index: pd.DatetimeIndex | None = None,
    hold_days: int = 1,
    min_stocks: int = 3,
) -> pd.DataFrame:
    """把预测分数转成组合引擎能用的**分数矩阵**（0~1，非选中的为 0）。

    Args:
        predictions: 宽表 DataFrame，或 model_id 字符串，或 None（取激活模型）
        top_n: 每日选分数最高的 N 只
        index: 目标日期轴（通常是行情的交易日）。预测频率低于行情时会 ffill
        hold_days: 选出来后至少持有几天。预测是逐日给的，每天重选会导致
            极高换手——``hold_days=5`` 表示每 5 天才重选一次
        min_stocks: 当日有预测的标的少于此数就不建仓（截面太窄，排序无意义）

    Returns:
        日期 × 标的的分数矩阵。选中的标的分数按当日截面 rank 归一到 (0,1]，
        便于 ``allocation="score"`` 按强弱定权重。
    """
    if predictions is None or isinstance(predictions, str):
        wide = predictions_wide(predictions)
    else:
        wide = predictions
    if wide is None or wide.empty:
        raise ValueError("没有可用的预测数据（先跑 eq ml predict-batch 写入 ml_predictions）")

    wide = wide.sort_index()
    if hold_days > 1:
        # 只在每 hold_days 个预测日重选，中间沿用——直接抑制换手
        keep = np.zeros(len(wide), dtype=bool)
        keep[::hold_days] = True
        wide = wide.where(pd.Series(keep, index=wide.index), other=np.nan).ffill()

    out = pd.DataFrame(0.0, index=wide.index, columns=wide.columns)
    for day, row in wide.iterrows():
        valid = row.dropna()
        if len(valid) < min_stocks:
            continue
        top = valid.nlargest(min(top_n, len(valid)))
        if len(top) == 0:
            continue
        # 截面内归一：最高分 1.0，最低分接近 0（但 >0，保证被视为「想持有」）
        r = top.rank(method="first")
        out.loc[day, top.index] = (r / r.max()).to_numpy()

    if index is not None:
        out = out.reindex(pd.DatetimeIndex(index).union(out.index)).ffill().reindex(index)
    return out.fillna(0.0)


# ======================================================================
# 单标的：SignalFunc
# ======================================================================

def ml_signal_for(
    symbol: str,
    model_id: str | None = None,
    *,
    buy_quantile: float = 0.7,
    sell_quantile: float = 0.4,
    lookback: int = 120,
) -> Callable[[pd.DataFrame], pd.Series]:
    """绑定单只标的的 ``SignalFunc``，可喂给 ``eq bt robust`` / 单标的回测。

    单标的没有截面可比，所以用**该标的自身分数的历史分位**做阈值：
    分数进入自身历史前 30% 买入、跌回后 40% 卖出。

    Args:
        buy_quantile/sell_quantile: 滚动分位阈值
        lookback: 滚动分位的窗口
    """
    sym = _norm(symbol)
    long = load_predictions(model_id, symbols=[sym])
    series = pd.Series(dtype="float64")
    if not long.empty:
        s = long[long["symbol"].map(_norm) == sym].set_index("date")["score"]
        series = s[~s.index.duplicated(keep="last")].sort_index()

    def _fn(df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(HOLD, index=df.index, name=f"ml({sym})")
        if series.empty:
            return sig
        sc = series.reindex(df.index.union(series.index)).ffill().reindex(df.index)
        if sc.notna().sum() < 5:
            return sig
        hi = sc.rolling(lookback, min_periods=10).quantile(buy_quantile)
        lo = sc.rolling(lookback, min_periods=10).quantile(sell_quantile)
        above = sc >= hi
        below = sc <= lo
        sig[above & ~above.shift(1, fill_value=False)] = BUY
        sig[below & ~below.shift(1, fill_value=False)] = SELL
        return sig

    _fn.__name__ = f"ml_signal({sym})"
    return _fn


# ======================================================================
# 一键：模型 → 组合回测
# ======================================================================

def backtest_model(
    model_id: str | None = None,
    *,
    bars: dict[str, pd.DataFrame] | None = None,
    top_n: int = 10,
    hold_days: int = 5,
    days: int = 500,
    portfolio_cfg=None,
) -> dict[str, Any]:
    """把一个训练好的模型**真的跑一遍组合回测**。

    这是回答「IC 0.05 到底能不能赚钱」的唯一办法——IC 不含成本、
    不含换手、不含持仓数约束。

    Returns:
        ``{"result": PortfolioResult, "top_n":…, "hold_days":…, "n_symbols":…}``
    """
    from eq.backtest.portfolio import PortfolioConfig, run_portfolio

    wide = predictions_wide(model_id)
    if wide.empty:
        raise ValueError("该模型没有预测记录，先跑 eq ml predict-batch")

    if bars is None:
        from concurrent.futures import ThreadPoolExecutor

        from eq.data.market import get_recent_bars

        syms = list(wide.columns)

        def _one(s):
            try:
                return s, get_recent_bars(s, days=days)
            except Exception:
                return s, None

        with ThreadPoolExecutor(max_workers=8) as pool:
            bars = {s: d for s, d in pool.map(_one, syms) if d is not None and len(d) >= 30}
    if not bars:
        raise ValueError("拉不到预测标的的行情")

    index = pd.DatetimeIndex(sorted(set().union(*(d.index for d in bars.values()))))
    scores = ml_score_matrix(wide, top_n=top_n, index=index, hold_days=hold_days)
    scores = scores[[c for c in scores.columns if c in bars]]

    cfg = portfolio_cfg or PortfolioConfig(max_positions=top_n, allocation="score")
    res = run_portfolio(bars, scores, cfg)
    return {"result": res, "top_n": top_n, "hold_days": hold_days,
            "n_symbols": len(bars), "n_pred_days": len(wide)}
