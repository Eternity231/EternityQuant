"""训练集切分与可复现性（v0.25 新增）。

**两个必须修的问题**

1. **标签泄漏（purge）**

   标签是 ``Ref($close, -h) / Ref($close, -1) - 1``——日期 T 的标签用到了
   T+h 的收盘价。所以如果训练集止于 2020-08-31、验证集起于 2020-09-01，
   训练集最后 h 天的标签**已经看过验证期的价格**。

   原来三条链路的切分全是「训练止 = 验证起 - 1 天」，等于每次切分都漏一次。
   h=5 时泄漏 5 天，对短窗口验证（原默认验证集只有 2020-09-01~09-25 共 19 个
   交易日）意味着 **1/4 的验证期被训练集看过**。

   正确做法：训练集尾部剔除 h 个交易日（purge），这就是 :func:`purged_split`。

2. **选模型的集合 = 报成绩的集合**

   ``fit()`` 用验证 IC 做 early stopping 和 best_state 选择，训练函数又把
   同一个 ``best_score``（**200 个 epoch 里的最大值**）当作模型成绩报出去。
   这是在选择集上报最大值——一个纯噪声的统计量跑 200 轮取最大，很容易
   "得到" IC=+0.03~0.05。

   正确做法：切出第三段 test，只在 test 上报成绩。:func:`purged_split`
   默认就给三段。

另附 :func:`set_seed`——原来没有任何随机种子控制，同一条命令两次跑出来的
IC 能差一大截，根本没法判断"调参到底有没有用"。
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------- 可复现性 ----------

def set_seed(seed: int = 42, deterministic: bool = False) -> int:
    """统一设置 random / numpy / torch(+cuda) 的随机种子。

    Args:
        deterministic: 额外开启 cuDNN 确定性算法。会慢 10~30%，
            但同一条命令两次跑出的结果逐位一致。调参对比时建议开。

    Returns:
        实际使用的 seed（方便记录进 metrics）。
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # 有些算子没有确定性实现，warn_only 避免直接抛错中断训练
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:  # pragma: no cover - 旧版 torch 无此 API
                pass
    except ImportError:  # pragma: no cover - 没装 torch 也能用
        pass
    return seed


# ---------- 切分 ----------

@dataclass
class Split:
    """一次切分的结果。mask 是与输入索引等长的布尔数组。"""

    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray | None = None
    # 每段的日期区间（闭区间），便于打日志和写进 ml_models 表
    bounds: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    purged_days: int = 0

    def sizes(self) -> dict[str, int]:
        out = {"train": int(self.train.sum()), "valid": int(self.valid.sum())}
        if self.test is not None:
            out["test"] = int(self.test.sum())
        return out

    def describe(self) -> str:
        parts = []
        for seg in ("train", "valid", "test"):
            mask = getattr(self, seg)
            if mask is None:
                continue
            lo, hi = self.bounds.get(seg, (None, None))
            rng = f" {_d(lo)}~{_d(hi)}" if lo is not None else ""
            parts.append(f"{seg}={int(mask.sum())}{rng}")
        return "  ".join(parts) + f"  (purge={self.purged_days}日)"


def _d(x) -> str:
    try:
        return pd.Timestamp(x).date().isoformat()
    except Exception:
        return str(x)


def _dates_of(index) -> np.ndarray:
    """从任意索引里取出「日期」数组（长度与索引一致）。"""
    if isinstance(index, pd.MultiIndex):
        for level in range(index.nlevels):
            vals = index.get_level_values(level)
            if pd.api.types.is_datetime64_any_dtype(vals):
                return vals.to_numpy()
        return index.get_level_values(0).to_numpy()
    if isinstance(index, pd.Index):
        return index.to_numpy()
    return np.asarray(index)


def purged_split(
    index,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    embargo_days: int = 5,
    with_test: bool = True,
) -> Split:
    """按**时间**切 train/valid/test，并在每个边界前 purge 掉 ``embargo_days``。

    切分是按「唯一交易日」而非样本条数做的——样本按标的展开后，
    简单地取前 80% 条会切成「前 80% 只股票」而不是「前 80% 个交易日」
    （港股链路原来正是这个 bug）。

    Args:
        index: 样本索引。可以是 DatetimeIndex、(datetime, instrument) MultiIndex，
            或一个与样本等长的日期数组。
        embargo_days: purge 的交易日数，**应设为标签的 horizon**。
        with_test: False 时只切 train/valid（沿用旧行为，用于 walk-forward 内层）。

    Returns:
        :class:`Split`
    """
    dates = _dates_of(index)
    if len(dates) == 0:
        empty = np.zeros(0, dtype=bool)
        return Split(train=empty, valid=empty, test=empty if with_test else None)

    uniq = np.array(sorted(pd.unique(dates)))
    n = len(uniq)
    if n < 3:
        raise ValueError(f"只有 {n} 个不同日期，无法做时间切分（至少需要 3 个）")

    if not with_test:
        test_ratio = 0.0
    if valid_ratio <= 0 or valid_ratio + test_ratio >= 1:
        raise ValueError(f"valid_ratio+test_ratio 必须在 (0,1) 内，收到 {valid_ratio}+{test_ratio}")

    n_test = int(round(n * test_ratio))
    n_valid = max(1, int(round(n * valid_ratio)))
    n_train = n - n_valid - n_test
    if n_train < 1:
        raise ValueError(f"切分后训练集为空（共 {n} 日，valid={n_valid}, test={n_test}）")

    train_days = uniq[:n_train]
    valid_days = uniq[n_train:n_train + n_valid]
    test_days = uniq[n_train + n_valid:] if n_test > 0 else np.array([])

    # purge：把每段尾部 embargo_days 个交易日剔掉，切断标签跨段引用
    e = max(0, int(embargo_days))
    if e > 0:
        train_days = train_days[:-e] if len(train_days) > e else train_days[:0]
        if len(test_days) > 0:
            valid_days = valid_days[:-e] if len(valid_days) > e else valid_days[:0]
    if len(train_days) == 0:
        raise ValueError(
            f"purge {e} 日后训练集为空（原 {n_train} 日）。"
            f"减小 embargo_days 或加长训练区间。"
        )
    if len(valid_days) == 0:
        raise ValueError(f"purge {e} 日后验证集为空（原 {n_valid} 日）。")

    def _mask(days) -> np.ndarray:
        return np.isin(dates, days) if len(days) else np.zeros(len(dates), dtype=bool)

    bounds = {}
    for seg, days in (("train", train_days), ("valid", valid_days), ("test", test_days)):
        if len(days):
            bounds[seg] = (days[0], days[-1])

    return Split(
        train=_mask(train_days),
        valid=_mask(valid_days),
        test=_mask(test_days) if n_test > 0 else None,
        bounds=bounds,
        purged_days=e,
    )


def walk_forward_windows(
    index,
    n_splits: int = 5,
    valid_days: int = 60,
    embargo_days: int = 5,
    expanding: bool = True,
    min_train_days: int = 120,
) -> list[Split]:
    """滚动前向验证窗口。

    单次固定切分只测了「一段特定行情」；A 股风格切换频繁，
    单次切分的 IC 很可能只是恰好赶上一段友好的行情。滚动验证给出
    「跨多个时间段的 IC 分布」，这才是判断模型稳不稳的依据。

    Args:
        n_splits: 滚动几次
        valid_days: 每次验证窗口的交易日数
        expanding: True = 训练集不断扩张（推荐，数据越多越好）；
            False = 固定长度滑窗（适合怀疑存在结构性突变时）
        min_train_days: 训练窗口最少交易日，不足则跳过该窗口

    Returns:
        :class:`Split` 列表（每个只有 train/valid，test 为 None）
    """
    dates = _dates_of(index)
    uniq = np.array(sorted(pd.unique(dates)))
    n = len(uniq)
    e = max(0, int(embargo_days))

    needed = min_train_days + e + valid_days
    if n < needed:
        logger.debug("交易日 %d < 滚动验证所需 %d，返回空窗口列表", n, needed)
        return []

    # 从末尾往前排 n_splits 个验证窗口
    windows: list[Split] = []
    for k in range(n_splits):
        v_end = n - k * valid_days
        v_start = v_end - valid_days
        if v_start <= 0:
            break
        train_end = v_start - e            # purge
        if train_end < min_train_days:
            break
        train_start = 0 if expanding else max(0, train_end - min_train_days)

        train_days = uniq[train_start:train_end]
        valid_d = uniq[v_start:v_end]
        if len(train_days) < min_train_days or len(valid_d) == 0:
            continue

        windows.append(Split(
            train=np.isin(dates, train_days),
            valid=np.isin(dates, valid_d),
            test=None,
            bounds={"train": (train_days[0], train_days[-1]),
                    "valid": (valid_d[0], valid_d[-1])},
            purged_days=e,
        ))
    windows.reverse()  # 时间正序，读日志更自然
    return windows


def split_bounds_to_qlib_segments(split: Split) -> dict[str, tuple[str, str]]:
    """把 :class:`Split` 的日期边界转成 qlib ``DatasetH`` 要的 segments 字典。"""
    out = {}
    for seg, (lo, hi) in split.bounds.items():
        out[seg] = (_d(lo), _d(hi))
    return out
