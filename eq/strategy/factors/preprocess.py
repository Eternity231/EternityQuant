"""特征/标签预处理（v0.38）—— 从 qlib 手里接管这一层。

原来这些是通过 ``{"class": "RobustZScoreNorm", ...}`` 这种字典配置交给 qlib
handler 内部执行的。接管过来有三个实在的好处：

1. **能测**。qlib 的处理器要跑起来得先有 qlib + 一整套 .bin 数据；这里是纯
   pandas，几行造个 DataFrame 就能验证边界行为（全 NaN 列、单股票截面、
   MAD=0 的常数列……这些以前全靠祈祷）。
2. **fit 窗口变显式**。归一化的统计量必须只用训练段拟合，否则验证/测试段的
   信息会顺着均值方差漏进训练——这是最隐蔽的一类泄漏。以前它藏在
   handler 的 ``fit_start_time`` / ``fit_end_time`` 参数里，看不见；
   现在 :meth:`Pipeline.fit` 收什么就是拟合什么。
3. **少一层 qlib 依赖**。qlib 0.9.7 在本项目里已经贴了 5 个 monkey patch
   绕它的 bug，能少用一块是一块。

## 数值口径

按 qlib 的处理器语义实现，但**不保证逐位一致**（qlib 未安装，无法对拍）。
项目当前已注册模型数为 0，不存在新旧模型可比性问题。若日后要和 qlib 结果
对照，用 :func:`compare_with_qlib` 在装了 qlib 的机器上跑一次。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "process_inf", "robust_zscore", "fillna", "dropna_label",
    "cs_rank_norm", "cs_zscore_norm", "Pipeline", "default_pipeline",
]

# 正态分布下 MAD 到标准差的换算系数：1/Φ⁻¹(0.75) ≈ 1.4826
_MAD_TO_STD = 1.4826
# 均匀分布 U(0,1) 的标准差是 1/√12，取倒数把 rank 拉成单位方差
_UNIFORM_SCALE = float(np.sqrt(12.0))


# ---------- 逐列处理（不跨截面） ----------

def process_inf(df: pd.DataFrame) -> pd.DataFrame:
    """±inf 换成该列**有限值的均值**。

    直接留着 inf 会让 BatchNorm1d 的梯度当场爆掉；换成 0 又会把
    「这个值极大」误导成「这个值中性」，取有限值均值是折中。
    """
    out = df.copy()
    mask = np.isinf(out.to_numpy(dtype="float64", na_value=np.nan))
    if not mask.any():
        return out
    arr = out.to_numpy(dtype="float64", na_value=np.nan).copy()
    arr[mask] = np.nan
    with np.errstate(invalid="ignore"):
        # 整列都是 inf/NaN 时 nanmean 会警告并返回 NaN，下一行统一兜成 0
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            means = np.nanmean(np.where(np.isfinite(arr), arr, np.nan), axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    rows, cols = np.where(mask)
    arr[rows, cols] = means[cols]
    return pd.DataFrame(arr, index=out.index, columns=out.columns)


def fillna(df: pd.DataFrame, value: float = 0.0) -> pd.DataFrame:
    """缺失填固定值。归一化之后填 0 就是「填成截面均值」，语义上合理。"""
    return df.fillna(value)


def dropna_label(features: pd.DataFrame, label: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """丢掉标签缺失的样本。

    标签缺失通常出现在**每只票序列的末尾** h 根（未来收益还不知道），
    留着会让模型学到一堆假的 0。特征缺失不丢——那由 fillna 处理。
    """
    keep = label.notna()
    return features.loc[keep.to_numpy()], label.loc[keep.to_numpy()]


# ---------- 截面处理（按日期分组） ----------

def _date_level(obj) -> np.ndarray | None:
    """取出 (datetime, instrument) MultiIndex 的日期层。取不到返回 None。"""
    idx = obj.index
    if isinstance(idx, pd.MultiIndex):
        for name in ("datetime", "date"):
            if name in (idx.names or []):
                return idx.get_level_values(name).to_numpy()
        return idx.get_level_values(0).to_numpy()
    if isinstance(idx, pd.DatetimeIndex):
        return idx.to_numpy()
    return None


def cs_rank_norm(obj):
    """横截面 rank 归一化：当日排名 → 百分位 → 中心化 → 拉成单位方差。

    这一步决定了模型学的是「今天这批票里哪只更好」而不是「明天大盘涨不涨」。
    结果与原始数值的**量纲无关**，只保留当日的相对次序，所以对异常值免疫。

    没有日期索引时退化成全样本 rank，并打 debug 日志——那多半意味着
    调用方丢了索引，是个 BUG 的前兆。
    """
    dates = _date_level(obj)
    if dates is None:
        logger.debug("cs_rank_norm 拿不到日期层，退化为全样本 rank")
        r, n = obj.rank(), obj.notna().sum()
    else:
        g = obj.groupby(dates)
        r, n = g.rank(), g.transform("count")

    # **精确居中**，而不是 qlib 那样的 rank(pct=True) - 0.5：
    # rank(pct=True) 的取值是 1/n, 2/n, …, 1，均值是 (n+1)/(2n) 而不是 1/2，
    # 减 0.5 之后每天残留 +1/(2n) 的偏移——n=30 时是 +0.058。
    # 偏移只跟当天股票数有关，于是「当天上市了多少只票」被编码进了因子值，
    # 变成一个纯粹由样本量制造的跨日水平差。
    # 用 (rank - (n+1)/2) / n 对任意 n 都严格居中。
    # n 可能是标量（无日期索引的退化分支）也可能是 Series（逐日计数）
    n = n.replace(0, np.nan) if hasattr(n, "replace") else (float(n) or np.nan)
    return ((r - (n + 1) / 2) / n) * _UNIFORM_SCALE


def cs_zscore_norm(obj):
    """横截面 z-score：``(x - 当日均值) / 当日标准差``。

    比 :func:`cs_rank_norm` 多保留了**幅度**信息（涨 10% 和涨 1% 不一样），
    代价是对异常值敏感。树模型更适合这个，神经网络更适合 rank。
    """
    dates = _date_level(obj)
    if dates is None:
        logger.debug("cs_zscore_norm 拿不到日期层，退化为全样本 z-score")
        std = obj.std()
        return (obj - obj.mean()) / (std if np.ndim(std) else (std or 1.0))
    g = obj.groupby(dates)
    mean, std = g.transform("mean"), g.transform("std")
    return (obj - mean) / std.replace(0, np.nan)


# ---------- 有状态：在训练段拟合，在全段应用 ----------

def robust_zscore(df: pd.DataFrame, median: pd.Series, mad: pd.Series,
                  clip_outlier: bool = True) -> pd.DataFrame:
    """用给定的中位数/MAD 做稳健 z-score。

    用中位数和 MAD 而不是均值和标准差：金融特征的尾部很厚，
    一个极端值就能把标准差抬上去、把所有正常值压成 0 附近。

    ``clip_outlier`` 把结果截到 ±3——注意这**必须在归一化之后**做，
    先截原值需要先知道尺度，是循环依赖。
    """
    scale = (mad * _MAD_TO_STD).replace(0, np.nan)
    out = (df - median) / scale
    if clip_outlier:
        out = out.clip(-3.0, 3.0)
    return out


@dataclass
class Pipeline:
    """特征处理管线：``fit`` 只吃训练段，``transform`` 用于全段。

    **为什么要显式分开**：归一化的统计量（中位数、MAD）如果在全样本上算，
    验证段和测试段的分布信息就顺着这两个数漏进了训练——模型没直接看到未来
    数据，但看到了「未来数据的统计摘要」。这类泄漏不会让回测崩，只会让它
    好看一点点，非常难发现。
    """

    clip_outlier: bool = True
    fillna_value: float = 0.0
    median_: pd.Series | None = field(default=None, repr=False)
    mad_: pd.Series | None = field(default=None, repr=False)
    columns_: list[str] = field(default_factory=list, repr=False)

    def fit(self, features: pd.DataFrame) -> Pipeline:
        """在**训练段特征**上拟合中位数与 MAD。"""
        clean = process_inf(features)
        self.columns_ = list(clean.columns)
        self.median_ = clean.median()
        self.mad_ = (clean - self.median_).abs().median()
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.median_ is None or self.mad_ is None:
            raise RuntimeError("Pipeline 未 fit，先在训练段上调 fit()")
        missing = [c for c in self.columns_ if c not in features.columns]
        if missing:
            raise ValueError(f"特征列缺失 {missing[:5]}（共 {len(missing)} 列），"
                             "训练和推理的特征集必须一致")
        out = process_inf(features[self.columns_])
        out = robust_zscore(out, self.median_, self.mad_, self.clip_outlier)
        return fillna(out, self.fillna_value)

    def fit_transform(self, features: pd.DataFrame) -> pd.DataFrame:
        return self.fit(features).transform(features)


def default_pipeline(clip_outlier: bool = True) -> Pipeline:
    """项目默认的特征管线：ProcessInf → RobustZScore(clip) → Fillna(0)。"""
    return Pipeline(clip_outlier=clip_outlier)


def normalize_label(label: pd.Series, method: str = "rank") -> pd.Series:
    """标签的截面归一化。``rank``（默认，抗异常）| ``zscore``（保留幅度）| ``none``。"""
    if method == "rank":
        return cs_rank_norm(label)
    if method == "zscore":
        return cs_zscore_norm(label)
    if method == "none":
        return label
    raise ValueError(f"未知标签归一化方式 {method}（可选 rank/zscore/none）")


# ---------- 和 qlib 对拍（需要装了 qlib 的机器） ----------

def compare_with_qlib(features: pd.DataFrame, fit_slice: slice | None = None) -> dict[str, Any]:
    """在装了 qlib 的机器上比对本模块与 qlib 处理器的数值差异。

    本项目的开发环境没装 qlib，所以这层是照 qlib 的**语义**实现的，
    没做过逐位对拍。真要迁移历史模型或对照旧结果时，跑一下这个函数。

    Returns:
        ``{"max_abs_diff": float, "mean_abs_diff": float, "cols": int}``；
        qlib 不可用时返回 ``{"error": ...}``。
    """
    try:
        from qlib.data.dataset.processor import ProcessInf, RobustZScoreNorm
    except Exception as e:  # pragma: no cover - 取决于环境
        return {"error": f"qlib 不可用：{e}"}

    fit_df = features if fit_slice is None else features.iloc[fit_slice]
    mine = default_pipeline().fit(fit_df).transform(features)

    qdf = ProcessInf()(features.copy())
    rz = RobustZScoreNorm(fit_start_time=None, fit_end_time=None, clip_outlier=True)
    try:
        rz.fit(fit_df)
    except Exception:
        pass
    theirs = rz(qdf).fillna(0.0)

    diff = (mine - theirs.reindex_like(mine)).abs()
    return {"max_abs_diff": float(np.nanmax(diff.to_numpy())),
            "mean_abs_diff": float(np.nanmean(diff.to_numpy())),
            "cols": int(mine.shape[1])}
