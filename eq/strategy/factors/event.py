"""事件类因子（v0.37）：解禁 / 股东户数 / 融资融券 / 北向持股。

这四项是 v0.35 删掉的深度研究报告里**唯一值得回收**的部分——它们不是给人看的
资讯，是**可交易的数据**：解禁日期是已知的未来供给冲击，股东户数下降是筹码集中，
融资余额是杠杆资金的方向，北向持股是外资的仓位。

和 :mod:`technical` 的区别：技术因子只用 OHLCV，本模块的输入是**外部事件表**，
于是多出一个技术因子完全没有的风险——**前视偏差**。

## 前视偏差：这个模块最要命的地方

外部数据几乎都有两个日期：

- **报告期 / 基准日**（如「2025 年三季度末股东户数」）
- **公告日 / 可获得日**（如「2025-10-24 披露」）

按报告期对齐，等于让 9 月 30 日的策略用上 10 月 24 日才公布的数字——回测里
这种因子的 IC 会非常漂亮，实盘一分钱赚不到。所有对齐一律走
:func:`align_events`，它**只认公告日**，并且强制 ``公告日 <= 交易日``。

## 现状（重要）

**这些因子的预测力尚未验证。** 本模块只保证「算得对、不穿越」，
不保证「有 alpha」。用法是先拉数据、再跑截面 IC，自己看数字决定要不要用：

    from eq.strategy.factors.event import evaluate_factor
    print(evaluate_factor(panel, bars_by_symbol, horizon=5))

IC 站不住就该扔掉，别因为「听起来有道理」就塞进策略。
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "align_events", "days_to_event", "event_pressure",
    "holder_change", "balance_momentum", "build_panel", "evaluate_factor",
]


# ---------- 对齐：唯一允许的外部数据接入口 ----------

def align_events(
    index: pd.DatetimeIndex,
    events: pd.DataFrame,
    value_col: str,
    publish_col: str = "publish_date",
    ffill_limit: int | None = None,
) -> pd.Series:
    """把「公告日 → 数值」的事件表对齐到交易日索引上，**严格不穿越**。

    每个交易日取的是「公告日 ≤ 该交易日」中最近的一条。公告日当天就算可用
    ——A 股公告普遍是盘后或开盘前发布，当天收盘价已经反映了它。

    Args:
        index: 目标交易日索引（通常是 bars.index）
        events: 至少包含 ``publish_col`` 和 ``value_col`` 两列
        ffill_limit: 一条公告最多向后延用多少个交易日；``None`` 表示不限。
            低频数据（股东户数一季一次）设个上限可以避免拿半年前的数字当当前状态。

    Returns:
        与 ``index`` 等长的 Series；还没有任何公告的日期为 NaN。
    """
    out = pd.Series(np.nan, index=index, dtype=float)
    if events is None or len(events) == 0:
        return out
    if publish_col not in events.columns or value_col not in events.columns:
        logger.debug("事件表缺列（需要 %s / %s），返回全 NaN", publish_col, value_col)
        return out

    ev = events[[publish_col, value_col]].copy()
    ev[publish_col] = pd.to_datetime(ev[publish_col], errors="coerce")
    ev[value_col] = pd.to_numeric(ev[value_col], errors="coerce")
    ev = ev.dropna(subset=[publish_col]).sort_values(publish_col)
    if ev.empty:
        return out

    # merge_asof 的 direction="backward" 正是「取最近一条已公告的」
    left = pd.DataFrame({"d": pd.to_datetime(index)}).sort_values("d")
    merged = pd.merge_asof(left, ev, left_on="d", right_on=publish_col,
                           direction="backward")

    if ffill_limit is not None:
        # 「这条公告已经生效了几个交易日」= 同一个公告日在结果里出现的第几行。
        #
        # 不能用「值是不是 NaN」来判断新鲜度：merge_asof 出来的值本身就是
        # 前向填充过的，notna() 恒为真，算出来的 age 永远是 0，限制形同虚设
        # （这个 BUG 被 test_align_ffill_limit_expires_stale_data 抓到过）。
        pub = merged[publish_col]
        age = pub.groupby(pub).cumcount()
        merged.loc[age > ffill_limit, value_col] = np.nan

    s = pd.Series(merged[value_col].to_numpy(), index=merged["d"].to_numpy())
    return pd.Series(s.reindex(pd.to_datetime(index)).to_numpy(), index=index)


# ---------- 解禁：已知的未来供给冲击 ----------

def days_to_event(
    index: pd.DatetimeIndex,
    event_dates,
    max_days: int = 120,
) -> pd.Series:
    """到下一个事件（解禁日）还有几个自然日。没有将来事件时为 NaN。

    解禁的特殊之处在于**日期本身是提前公开的**，所以「还有 N 天解禁」
    这个因子不存在前视问题——不需要公告日对齐。

    ``max_days`` 之外的事件视为太远、不计入（截断可以避免因子被
    「一年后有个大解禁」这种噪声主导）。
    """
    idx = pd.to_datetime(index)
    out = pd.Series(np.nan, index=index, dtype=float)
    ds = pd.to_datetime(pd.Series(list(event_dates)), errors="coerce").dropna()
    if ds.empty:
        return out
    ds = ds.sort_values().to_numpy()
    for i, d in enumerate(idx):
        future = ds[ds >= np.datetime64(d)]
        if len(future) == 0:
            continue
        gap = (future[0] - np.datetime64(d)) / np.timedelta64(1, "D")
        if gap <= max_days:
            out.iloc[i] = float(gap)
    return out


def event_pressure(
    index: pd.DatetimeIndex,
    events: pd.DataFrame,
    date_col: str = "date",
    ratio_col: str = "ratio",
    window: int = 60,
    decay: float = 30.0,
) -> pd.Series:
    """未来 ``window`` 天内解禁压力：按占流通股比例加权、按距离指数衰减求和。

    比 :func:`days_to_event` 多考虑了两件事：**规模**（解禁 1% 和 30% 完全不同）
    和**叠加**（一个月内三次小解禁 ≈ 一次大解禁）。

    权重 ``exp(-gap/decay)``：越近的解禁压力越大。``decay=30`` 表示 30 天外
    的解禁权重衰减到 1/e。
    """
    idx = pd.to_datetime(index)
    out = pd.Series(0.0, index=index, dtype=float)
    if events is None or len(events) == 0 or date_col not in events.columns:
        return out
    ev = events.copy()
    ev[date_col] = pd.to_datetime(ev[date_col], errors="coerce")
    ev["_r"] = (pd.to_numeric(ev[ratio_col], errors="coerce")
                if ratio_col in ev.columns else 1.0)
    ev = ev.dropna(subset=[date_col])
    if ev.empty:
        return out
    dates = ev[date_col].to_numpy()
    ratios = ev["_r"].fillna(0.0).to_numpy()
    for i, d in enumerate(idx):
        gap = (dates - np.datetime64(d)) / np.timedelta64(1, "D")
        m = (gap >= 0) & (gap <= window)
        if m.any():
            out.iloc[i] = float((ratios[m] * np.exp(-gap[m] / decay)).sum())
    return out


# ---------- 股东户数：筹码集中度 ----------

def holder_change(
    index: pd.DatetimeIndex,
    events: pd.DataFrame,
    value_col: str = "holders",
    publish_col: str = "publish_date",
    ffill_limit: int = 90,
) -> pd.Series:
    """股东户数的环比变化率，**取负**——户数下降（筹码集中）为正分。

    方向约定：本项目所有因子统一「大 = 看多」，这样截面 IC 的符号才有可比性。
    户数减少通常被解读为筹码向少数人集中，所以取负。

    ``ffill_limit=90`` 个交易日：股东户数一般一季一披露，超过一个季度还没有
    新数据就当它失效，不要拿去年的数字当今天的状态。
    """
    raw = align_events(index, events, value_col, publish_col, ffill_limit=ffill_limit)
    # 只在数值真的换了一期时才产生变化（同一期内 pct_change 恒为 0）
    changed = raw[raw.notna()].drop_duplicates()
    if len(changed) < 2:
        return pd.Series(np.nan, index=index, dtype=float)
    pct = changed.pct_change()
    return -pct.reindex(index).ffill()


# ---------- 融资余额 / 北向持股：资金方向 ----------

def balance_momentum(
    index: pd.DatetimeIndex,
    events: pd.DataFrame,
    value_col: str,
    publish_col: str = "publish_date",
    window: int = 20,
) -> pd.Series:
    """余额类数据（融资余额、北向持股）的 ``window`` 日变化率。

    用变化率而不是绝对额：绝对额和市值高度相关，截面上等于在赌大小盘风格，
    不是在赌资金流向。
    """
    raw = align_events(index, events, value_col, publish_col)
    if raw.notna().sum() < window + 1:
        return pd.Series(np.nan, index=index, dtype=float)
    base = raw.shift(window)
    return (raw - base) / base.abs().replace(0, np.nan)


# ---------- 组装成截面面板 + 验证 ----------

def build_panel(factor_by_symbol: dict[str, pd.Series]) -> pd.Series:
    """把 ``{symbol: 因子序列}`` 拼成带 (datetime, instrument) MultiIndex 的 Series。

    这是 :mod:`evaluation` 那套截面工具要的格式。
    """
    parts = []
    for sym, s in factor_by_symbol.items():
        if s is None or len(s) == 0:
            continue
        f = pd.Series(np.asarray(s, dtype=float), index=pd.to_datetime(s.index))
        f.index = pd.MultiIndex.from_product([f.index, [sym]],
                                             names=["datetime", "instrument"])
        parts.append(f)
    if not parts:
        return pd.Series(dtype=float,
                         index=pd.MultiIndex.from_arrays(
                             [pd.DatetimeIndex([]), []],
                             names=["datetime", "instrument"]))
    return pd.concat(parts).sort_index()


def forward_returns(bars_by_symbol: dict[str, pd.DataFrame], horizon: int = 5) -> pd.Series:
    """未来 ``horizon`` 日收益，同样是 (datetime, instrument) 面板。

    用 ``close[t+h]/close[t] - 1``；末尾 h 根没有标签，会是 NaN 并在评估时被丢掉。
    """
    parts = []
    for sym, df in bars_by_symbol.items():
        if df is None or len(df) <= horizon or "close" not in df.columns:
            continue
        c = df["close"].astype(float)
        r = c.shift(-horizon) / c - 1
        r.index = pd.MultiIndex.from_product([pd.to_datetime(r.index), [sym]],
                                             names=["datetime", "instrument"])
        parts.append(r)
    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts).sort_index()


def evaluate_factor(
    factor: pd.Series,
    bars_by_symbol: dict[str, pd.DataFrame],
    horizon: int = 5,
) -> dict[str, Any]:
    """把事件因子送进和 ML 模型**同一套**验收流程：逐日截面 Rank IC。

    用同一把尺很重要——事件因子和 ML 预测最终要放在一起比较，
    换口径就没法比了。
    """
    from eq.strategy.factors.evaluation import evaluate

    label = forward_returns(bars_by_symbol, horizon=horizon)
    return evaluate(factor, label)


# ---------- 数据获取（akshare，全部走公告日） ----------

def fetch_lockups(symbol: str) -> pd.DataFrame:
    """拉个股解禁明细。返回 ``date`` / ``ratio``（占流通股比例，%）两列。

    取不到时返回空表——上层因子函数对空表都是返回 NaN/0，不会炸。
    """
    from eq.data.market import bare_code

    code = bare_code(symbol)
    try:
        import akshare as ak

        df = ak.stock_restricted_release_detail_em(
            start_date=(dt.date.today() - dt.timedelta(days=365)).strftime("%Y%m%d"),
            end_date=(dt.date.today() + dt.timedelta(days=365)).strftime("%Y%m%d"),
        )
    except Exception as e:
        logger.debug("解禁数据拉取失败 %s：%s", symbol, e)
        return pd.DataFrame(columns=["date", "ratio"])
    return _pick_symbol(df, code, date_keys=("解禁时间", "解禁日期"),
                        value_keys=("占流通股比例", "占总股本比例"),
                        out_cols=("date", "ratio"))


def fetch_holders(symbol: str) -> pd.DataFrame:
    """拉股东户数。返回 ``publish_date`` / ``holders``。

    **必须取公告日**：东财这张表同时有「股东户数统计截止日」（报告期）和
    「公告日期」，用前者对齐就是穿越。
    """
    from eq.data.market import bare_code

    try:
        import akshare as ak

        df = ak.stock_zh_a_gdhs_detail_em(symbol=bare_code(symbol))
    except Exception as e:
        logger.debug("股东户数拉取失败 %s：%s", symbol, e)
        return pd.DataFrame(columns=["publish_date", "holders"])
    return _normalize(df, date_keys=("公告日期",), value_keys=("股东户数-本次",
                                                              "股东户数"),
                      out_cols=("publish_date", "holders"))


def _normalize(df, date_keys, value_keys, out_cols) -> pd.DataFrame:
    """从中文列名的 akshare 表里挑出日期列和数值列，统一改名。"""
    empty = pd.DataFrame(columns=list(out_cols))
    if df is None or len(df) == 0:
        return empty
    dcol = next((k for k in date_keys if k in df.columns), None)
    vcol = next((k for k in value_keys if k in df.columns), None)
    if dcol is None or vcol is None:
        logger.debug("列名对不上（有 %s）", list(df.columns)[:8])
        return empty
    out = df[[dcol, vcol]].copy()
    out.columns = list(out_cols)
    out[out_cols[0]] = pd.to_datetime(out[out_cols[0]], errors="coerce")
    out[out_cols[1]] = pd.to_numeric(out[out_cols[1]], errors="coerce")
    return out.dropna().sort_values(out_cols[0]).reset_index(drop=True)


def _pick_symbol(df, code: str, date_keys, value_keys, out_cols) -> pd.DataFrame:
    """全市场表里过滤出本股，再走 :func:`_normalize`。"""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=list(out_cols))
    ccol = next((k for k in ("代码", "股票代码", "证券代码") if k in df.columns), None)
    if ccol is None:
        return pd.DataFrame(columns=list(out_cols))
    mine = df[df[ccol].astype(str).str.zfill(6) == code]
    return _normalize(mine, date_keys, value_keys, out_cols)
