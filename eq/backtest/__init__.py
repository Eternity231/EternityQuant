"""回测引擎子包。

双引擎并存（problem 9 冶议）：
- vectorized.py：向量化回测，研发阶段用（快）
- event_driven.py：事件驱动回测，上线前用（准）

两引擎共享 signal(df) -> df 接口（problem 10 冶议），零适配器。
"""

from eq.backtest.types import BacktestConfig, BacktestResult
from eq.backtest.vectorized import VectorizedBacktester
from eq.backtest.event_driven import EventDrivenBacktester

__all__ = ["BacktestConfig", "BacktestResult", "VectorizedBacktester", "EventDrivenBacktester"]
