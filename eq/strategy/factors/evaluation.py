"""因子/模型评估指标（v0.25 新增）。

**为什么要单独写这个模块**

原三条训练链路（``MLPAlphaNet`` / ``RecurrentAlphaNet`` / ``DeepAlphaTrainer``）算 IC 都是::

    score = cov(pred, label) / (pred.std() * label.std())

这是把**所有日期所有股票混在一起**算的 Pearson 相关（pooled IC）。
量化里说的 IC 几乎总是指另一个东西：**每个交易日先做一次横截面相关，
再对日期序列求均值**。两者差别不是细节：

- pooled IC 会被「日期之间的均值漂移」污染。极端情况下模型完全不会选股、
  只会预测大盘涨跌，pooled IC 照样很漂亮，但它对选股毫无用处。
- 真正决定策略能不能赚钱的是 **ICIR = mean(daily_IC) / std(daily_IC)**，
  即信号的稳定性。pooled IC 根本算不出这个数。

所以本模块提供业界标准口径：

- :func:`daily_ic`         每日横截面 IC 序列（默认 Spearman = Rank IC）
- :func:`ic_summary`       IC 均值/标准差/ICIR/t 统计量/胜率
- :func:`quantile_returns` 按预测分数分 N 组的未来收益 + 多空价差
- :func:`evaluate`         一次算全套，返回结构化报告
- :func:`format_report`    打印用

所有函数都在**没有日期索引时优雅降级**为 pooled 口径，并在返回值里
用 ``pooled=True`` 明确标注——降级过的数字不能和正常口径比。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 一个交易日至少要有这么多只股票才算得出有意义的横截面相关
MIN_STOCKS_PER_DAY = 5
TRADING_DAYS = 252


def _as_series(x, name: str) -> pd.Series:
    """把 DataFrame/ndarray/Series 统一成一维 Series，尽量保住索引。"""
    if isinstance(x, pd.Series):
        return x.rename(name)
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(f"{name} 期望单列，收到 {x.shape[1]} 列")
        return x.iloc[:, 0].rename(name)
    arr = np.asarray(x).reshape(-1)
    return pd.Series(arr, name=name)


def _aligned_frame(pred, label) -> pd.DataFrame:
    """对齐 pred/label 成两列 DataFrame，丢掉任一为 NaN 的行。"""
    p = _as_series(pred, "pred")
    lb = _as_series(label, "label")
    if len(p) != len(lb) and not (p.index.equals(lb.index)):
        # 索引不同且长度不同 → 只能按索引 join
        df = pd.concat([p, lb], axis=1, join="inner")
    elif p.index.equals(lb.index):
        df = pd.concat([p, lb], axis=1)
    else:
        # 长度相同但索引不同（一方是 RangeIndex）：按位置对齐
        df = pd.DataFrame({"pred": p.to_numpy(), "label": lb.to_numpy()}, index=p.index)
    return df.replace([np.inf, -np.inf], np.nan).dropna()


def _date_level(df: pd.DataFrame) -> pd.Index | None:
    """取出「日期」这一层索引。拿不到返回 None（调用方降级为 pooled）。"""
    idx = df.index
    if isinstance(idx, pd.MultiIndex):
        # qlib 的约定是 (datetime, instrument)
        for level in range(idx.nlevels):
            vals = idx.get_level_values(level)
            if pd.api.types.is_datetime64_any_dtype(vals):
                return vals
        return idx.get_level_values(0)
    if pd.api.types.is_datetime64_any_dtype(idx):
        return idx
    return None


def daily_ic(
    pred,
    label,
    method: str = "spearman",
    min_stocks: int = MIN_STOCKS_PER_DAY,
) -> pd.Series:
    """每日横截面 IC 序列。

    Args:
        pred/label: 带 (datetime, instrument) MultiIndex 的 Series/DataFrame
        method: ``spearman``（Rank IC，抗异常值，推荐）| ``pearson``
        min_stocks: 当日股票数少于此值就跳过（横截面相关无意义）

    Returns:
        index=日期、value=当日 IC 的 Series。无日期索引时返回**单元素**
        Series（pooled IC），调用方可用 ``len(...) == 1`` 识别。
    """
    df = _aligned_frame(pred, label)
    if df.empty:
        return pd.Series(dtype=float)

    dates = _date_level(df)
    if dates is None:
        # 降级：没有日期信息，只能算 pooled
        logger.debug("pred/label 无日期索引，IC 降级为 pooled 口径")
        c = df["pred"].corr(df["label"], method=method)
        return pd.Series([float(c) if pd.notna(c) else 0.0], index=[pd.NaT])

    grouped = df.groupby(dates.to_numpy(), sort=True)
    out: dict[Any, float] = {}
    for day, g in grouped:
        if len(g) < min_stocks:
            continue
        # 当日预测或标签全是同一个值 → 相关无定义，记 0（中性）而不是 NaN，
        # 否则"模型输出塌缩成常数"这种坏情况会被 dropna 悄悄藏起来
        if g["pred"].nunique() < 2 or g["label"].nunique() < 2:
            out[day] = 0.0
            continue
        c = g["pred"].corr(g["label"], method=method)
        out[day] = float(c) if pd.notna(c) else 0.0
    if not out:
        return pd.Series(dtype=float)
    s = pd.Series(out).sort_index()
    s.index = pd.to_datetime(s.index)
    return s


def ic_summary(ic: pd.Series) -> dict[str, float]:
    """把每日 IC 序列汇总成一组指标。

    - ``ic_mean``   平均 IC。A 股日频选股，0.03 已算可用，0.05+ 是好因子
    - ``ic_std``    IC 波动
    - ``icir``      = ic_mean / ic_std。**比 IC 本身更重要**，衡量信号稳定性；
                    经验上 ICIR > 0.3 才值得上实盘
    - ``t_stat``    = icir * sqrt(n_days)，|t| > 2 才谈得上统计显著
    - ``ic_win_rate`` IC > 0 的交易日占比，0.55+ 算稳
    - ``ic_ir_annual`` 年化 ICIR（= icir * sqrt(252)），和夏普可比
    """
    ic = pd.Series(ic).dropna()
    n = len(ic)
    if n == 0:
        return {"ic_mean": 0.0, "ic_std": 0.0, "icir": 0.0, "t_stat": 0.0,
                "ic_win_rate": 0.0, "ic_ir_annual": 0.0, "n_days": 0}
    mean = float(ic.mean())
    std = float(ic.std(ddof=1)) if n > 1 else 0.0
    # 常数序列的样本标准差在浮点下是 ~1e-18 而非严格 0，直接 mean/std
    # 会算出 1e15 这种荒谬的 ICIR。用绝对阈值判定"无波动"。
    if std < 1e-12:
        std = 0.0
    icir = mean / std if std > 0 else 0.0
    return {
        "ic_mean": mean,
        "ic_std": std,
        "icir": icir,
        "t_stat": icir * np.sqrt(n) if std > 0 else 0.0,
        "ic_win_rate": float((ic > 0).mean()),
        "ic_ir_annual": icir * np.sqrt(TRADING_DAYS) if std > 0 else 0.0,
        "n_days": n,
    }


def quantile_returns(
    pred,
    label,
    n_groups: int = 5,
    min_stocks: int = MIN_STOCKS_PER_DAY,
) -> dict[str, Any]:
    """按预测分数分组，看每组的平均未来收益（分层测试）。

    这是判断因子「单调性」的标准手段：好因子的分组收益应该单调递增，
    且 top 组减 bottom 组的多空价差显著为正。IC 高但分层不单调的因子
    通常是被少数极端值撑起来的，实盘不可用。

    Returns:
        ``{"group_mean": [...], "long_short": float, "monotonic": bool,
        "long_short_series": pd.Series, "n_days": int}``
    """
    df = _aligned_frame(pred, label)
    if df.empty:
        return {"group_mean": [], "long_short": 0.0, "monotonic": False,
                "long_short_series": pd.Series(dtype=float), "n_days": 0}

    dates = _date_level(df)
    if dates is None:
        keys = np.zeros(len(df))  # 全当同一天
    else:
        keys = dates.to_numpy()

    per_day_groups: list[np.ndarray] = []
    ls_by_day: dict[Any, float] = {}
    for day, g in df.groupby(keys, sort=True):
        if len(g) < max(min_stocks, n_groups):
            continue
        # rank → 等频分组（pct rank 比 qcut 更抗重复值）
        r = g["pred"].rank(method="first", pct=True)
        bucket = np.clip((r * n_groups).astype(int), 0, n_groups - 1)
        means = g["label"].groupby(bucket).mean()
        means = means.reindex(range(n_groups))
        per_day_groups.append(means.to_numpy(dtype=float))
        if pd.notna(means.iloc[-1]) and pd.notna(means.iloc[0]):
            ls_by_day[day] = float(means.iloc[-1] - means.iloc[0])

    if not per_day_groups:
        return {"group_mean": [], "long_short": 0.0, "monotonic": False,
                "long_short_series": pd.Series(dtype=float), "n_days": 0}

    group_mean = np.nanmean(np.vstack(per_day_groups), axis=0)
    ls_series = pd.Series(ls_by_day).sort_index()
    diffs = np.diff(group_mean)
    return {
        "group_mean": [float(v) for v in group_mean],
        "long_short": float(group_mean[-1] - group_mean[0]),
        # 允许一次反向（现实中很少完美单调），全程递增才算强单调
        "monotonic": bool(np.sum(diffs < 0) <= 1),
        "long_short_series": ls_series,
        "n_days": len(per_day_groups),
    }


def evaluate(pred, label, n_groups: int = 5, method: str = "spearman") -> dict[str, Any]:
    """一次算全套评估指标。

    Returns:
        含 ``rank_ic`` / ``pearson_ic`` / 分层结果 / ``pooled`` 标志的字典。
        ``pooled=True`` 表示输入没有日期索引、只能给 pooled 口径——
        这种数字不可与正常口径横向比较。
    """
    df = _aligned_frame(pred, label)
    pooled = _date_level(df) is None
    ic = daily_ic(df["pred"], df["label"], method=method)
    summ = ic_summary(ic)
    q = quantile_returns(df["pred"], df["label"], n_groups=n_groups)
    # Pearson 版本一并给出，和 Rank IC 差很多时说明信号被极端值主导
    pic = daily_ic(df["pred"], df["label"], method="pearson")
    return {
        "pooled": pooled,
        "n_samples": int(len(df)),
        "ic_method": method,
        **summ,
        "pearson_ic_mean": float(pd.Series(pic).dropna().mean()) if len(pic) else 0.0,
        "group_mean": q["group_mean"],
        "long_short": q["long_short"],
        "monotonic": q["monotonic"],
        "n_group_days": q["n_days"],
        "ic_series": ic,
    }


def format_report(report: dict[str, Any], title: str = "模型评估") -> str:
    """把 :func:`evaluate` 的结果格式化成人能读的文本块。"""
    lines = [f"\n===== {title} ====="]
    if report.get("pooled"):
        lines.append("  ⚠ 输入无日期索引，以下为 pooled 口径（偏乐观，不可与按日口径比较）")
    lines.append(
        f"  样本 {report.get('n_samples', 0):,}   有效交易日 {report.get('n_days', 0)}"
    )
    icir = report.get("icir", 0.0)
    t = report.get("t_stat", 0.0)
    lines.append(
        f"  Rank IC  {report.get('ic_mean', 0):+.4f}  (std {report.get('ic_std', 0):.4f})"
        f"   ICIR {icir:+.3f}   t={t:+.2f}   胜率 {report.get('ic_win_rate', 0):.1%}"
    )
    lines.append(f"  Pearson IC {report.get('pearson_ic_mean', 0):+.4f}")
    gm = report.get("group_mean") or []
    if gm:
        groups = "  ".join(f"Q{i+1} {v:+.4f}" for i, v in enumerate(gm))
        lines.append(f"  分层收益  {groups}")
        lines.append(
            f"  多空价差  {report.get('long_short', 0):+.4f}"
            f"   单调性 {'✓' if report.get('monotonic') else '✗'}"
        )
    # 结论行：给一句人话判断
    lines.append(f"  判定：{verdict(report)}")
    return "\n".join(lines) + "\n"


def verdict(report: dict[str, Any]) -> str:
    """给一句「这个模型能不能用」的直白判断。"""
    if report.get("pooled"):
        return "无法判定（缺日期索引，只能算 pooled IC）"
    n = report.get("n_days", 0)
    if n < 20:
        return f"样本期太短（{n} 个交易日），结论不可信"
    ic, icir, t = report.get("ic_mean", 0), report.get("icir", 0), report.get("t_stat", 0)
    if abs(t) < 2:
        return f"不显著（|t|={abs(t):.2f} < 2），与随机无异"
    if ic <= 0:
        return f"IC 为负（{ic:+.4f}），信号方向可能反了"
    if icir < 0.2:
        return f"有信号但极不稳定（ICIR {icir:.2f} < 0.2），不建议实盘"
    if ic < 0.02:
        return f"信号偏弱（IC {ic:.4f}），扣掉成本大概率不剩什么"
    if not report.get("monotonic", False):
        return f"IC 尚可（{ic:.4f}）但分层不单调，可能被极端值主导"
    return f"可用（IC {ic:.4f}，ICIR {icir:.2f}，分层单调）"
