"""信号子包：组合因子出买卖决策（BUY/SELL/HOLD 或 0~1 连续仓位）。

分层：

- ``base.py``      分数契约（score ∈ [-1,1]）与三态互转
- ``trend.py``     趋势类：EMA 交叉、ADX 趋势
- ``reversal.py``  反转类：RSI / 布林 / z-score / KDJ / CCI
- ``breakout.py``  突破类：唐奇安 / 肯特纳 / SuperTrend / 波动率突破
- ``composite.py`` 组合类：多策略投票、市场状态自适应、前置过滤

策略是函数：``Callable[[pd.DataFrame], pd.Series]``。
两个回测引擎共享此接口，返回三态或连续仓位都认。
"""

from eq.strategy.signals.base import (
    clip_score, normalize_score, score_to_signal, signal_to_score,
)
from eq.strategy.signals.breakout import (
    atr_channel_score, breakout_composite, donchian_breakout, keltner_breakout,
    supertrend_follow, turtle_score, volatility_breakout,
)
from eq.strategy.signals.composite import (
    filtered, make_filtered, make_regime_adaptive, make_vote, market_regime,
    not_overextended, regime_adaptive, squeeze_filter, volatility_filter,
    volume_filter, vote,
)
from eq.strategy.signals.reversal import (
    bollinger_break, cci_reversal, kdj_cross, reversion_composite, rsi_reversal,
    rsi_score, zscore_reversion,
)
from eq.strategy.signals.trend import adx_trend, ema_cross

__all__ = [
    # 趋势
    "ema_cross", "adx_trend",
    # 反转
    "rsi_reversal", "bollinger_break", "zscore_reversion", "kdj_cross",
    "cci_reversal", "rsi_score", "reversion_composite",
    # 突破
    "donchian_breakout", "keltner_breakout", "supertrend_follow",
    "volatility_breakout", "turtle_score", "atr_channel_score", "breakout_composite",
    # 组合
    "vote", "make_vote", "regime_adaptive", "make_regime_adaptive",
    "market_regime", "filtered", "make_filtered",
    "volume_filter", "volatility_filter", "squeeze_filter", "not_overextended",
    # 分数
    "score_to_signal", "signal_to_score", "clip_score", "normalize_score",
]
