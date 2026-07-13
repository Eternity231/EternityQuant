"""策略层：因子 + 信号两层分层（problem 14 冶议）。

- factors/：单数列计算，可复用、可单独评估（IC/IR）
- signals/：组合因子出买卖决策（BUY/SELL/HOLD 或 float 置信度）
"""

from eq.strategy.types import BUY, SELL, HOLD, FactorFunc, SignalFunc

__all__ = ["BUY", "SELL", "HOLD", "FactorFunc", "SignalFunc"]
