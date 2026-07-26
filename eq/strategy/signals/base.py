"""信号强度契约（v0.27 新增）。

**原来的问题**

信号只有 ``BUY`` / ``SELL`` / ``HOLD`` 三个字符串。这个契约有两个硬伤：

1. **没法表达强弱**。RSI 刚跌破 30 和跌到 12，都只能输出一个 BUY，
   仓位上无从区分——而这恰恰是最该区分的地方。
2. **没法组合**。想做「5 个策略投票，3 票以上才买」，三态字符串
   加不起来，只能写一堆 if/else。

所以 v0.27 引入**分数**这一层：``score ∈ [-1, +1]``，
-1 = 最强看空、0 = 中性、+1 = 最强看多。

- 策略内部算分数 → :func:`score_to_signal` 转成三态给回测引擎（契约不变）
- 组合策略直接对分数加权求和（见 :mod:`eq.strategy.signals.composite`）
- 仓位管理按分数大小定仓（见 :mod:`eq.strategy.risk`）

三态与分数可以互转，老策略一行不用改。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from eq.strategy import BUY, HOLD, SELL

# 默认阈值：分数超过 +0.3 才算买、低于 -0.3 才算卖，中间一律观望。
# 不用 0 做阈值是因为噪声会让分数在 0 附近反复穿越，导致极高换手。
DEFAULT_BUY_TH = 0.3
DEFAULT_SELL_TH = -0.3


def score_to_signal(
    score: pd.Series,
    buy_th: float = DEFAULT_BUY_TH,
    sell_th: float = DEFAULT_SELL_TH,
    *,
    on_cross_only: bool = True,
) -> pd.Series:
    """分数序列 → BUY/SELL/HOLD 三态。

    Args:
        on_cross_only: True（默认）只在**穿越阈值那一根**发信号，
            之后保持 HOLD 由引擎维持仓位。False 则只要在阈值之外就持续发。

            为什么默认只在穿越时发：回测引擎把 BUY 解释成「目标仓位=满仓」、
            SELL 解释成「清仓」，持续发 BUY 不会重复买入，但会让
            ``trades`` 明细里出现大量无意义的重复记录，也让 ``--sweep``
            这类横评的"交易笔数"失真。
    """
    score = pd.Series(score).astype("float64")
    out = pd.Series(HOLD, index=score.index, name="signal")
    long_zone = score >= buy_th
    short_zone = score <= sell_th
    if on_cross_only:
        prev_long = long_zone.shift(1, fill_value=False)
        prev_short = short_zone.shift(1, fill_value=False)
        out[long_zone & ~prev_long] = BUY
        out[short_zone & ~prev_short] = SELL
    else:
        out[long_zone] = BUY
        out[short_zone] = SELL
    return out


def signal_to_score(sig: pd.Series) -> pd.Series:
    """三态 → 分数（BUY=+1 / SELL=-1 / HOLD=0），供老策略参与组合投票。"""
    sig = pd.Series(sig)
    return pd.Series(
        np.select([sig == BUY, sig == SELL], [1.0, -1.0], default=0.0),
        index=sig.index, dtype="float64", name="score",
    )


def clip_score(s: pd.Series) -> pd.Series:
    """把任意实数压到 [-1, 1]，NaN 记 0（中性）。"""
    return pd.Series(s).astype("float64").fillna(0.0).clip(-1.0, 1.0)


def normalize_score(s: pd.Series, period: int = 60) -> pd.Series:
    """把一个无界指标滚动标准化成 [-1, 1] 的分数。

    用滚动 z-score 再除以 2 压缩（±2σ → ±1）。滚动而非全样本，
    是为了避免用到未来数据。
    """
    s = pd.Series(s).astype("float64")
    mean = s.rolling(period, min_periods=max(2, period // 4)).mean()
    std = s.rolling(period, min_periods=max(2, period // 4)).std()
    return clip_score((s - mean) / std.replace(0, np.nan) / 2.0)


def scaled_by(score: pd.Series, gate: pd.Series) -> pd.Series:
    """用一个 0~1 的闸门序列缩放分数（如用趋势强度给趋势信号加权）。"""
    g = pd.Series(gate).astype("float64").fillna(0.0).clip(0.0, 1.0)
    return clip_score(pd.Series(score).astype("float64").fillna(0.0) * g)
