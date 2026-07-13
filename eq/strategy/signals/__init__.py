"""信号子包：组合因子出买卖决策（BUY/SELL/HOLD）。

- trend.py     趋势类（EMA 交叉、ADX 趋势）
- reversal.py  反转类（RSI 超买超卖、布林突破）

策略是函数：Callable[[pd.DataFrame], pd.Series]（problem 10 冶议）。
向量化引擎和事件驱动引擎共享此接口，零适配器。
"""

from eq.strategy.signals.trend import ema_cross, adx_trend
from eq.strategy.signals.reversal import rsi_reversal, bollinger_break

__all__ = ["ema_cross", "adx_trend", "rsi_reversal", "bollinger_break"]
