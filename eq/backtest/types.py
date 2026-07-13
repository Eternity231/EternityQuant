"""回测配置和结果数据契约（双引擎共享）。"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from eq.strategy import BUY, SELL, HOLD


@dataclass
class BacktestConfig:
    """回测配置。

    - initial_cash: 初始现金
    - commission_bps: 单边手续费（万分之）
    - slippage_bps: 单边滑点（万分之）
    - allow_short: 第一版禁卖空
    - engine: 'vectorized' | 'event_driven'
    """
    initial_cash: float = 1_000_000.0
    commission_bps: float = 2.5     # A 股典型 0.025%
    slippage_bps: float = 5.0       # 散户打 ±0.05%
    allow_short: bool = False
    engine: Literal["vectorized", "event_driven"] = "vectorized"


@dataclass
class BacktestResult:
    """回测结果。metrics 由引擎填充；equity_curve 是逐日权益；trades 是明细。"""

    config: BacktestConfig
    equity_curve: pd.Series          # index = date, value = total equity
    trades: pd.DataFrame             # columns: entry_date, exit_date, entry_price, exit_price, shares, pnl
    metrics: dict = field(default_factory=dict)

    def summary(self) -> str:
        """格式化关键指标为简短文本。"""
        m = self.metrics
        return (
            f"总收益 {m.get('total_return', 0):+.2%}  "
            f"年化 {m.get('annual_return', 0):+.2%}  "
            f"夏普 {m.get('sharpe', 0):+.2f}  "
            f"最大回撤 {m.get('max_drawdown', 0):+.2%}  "
            f"胜率 {m.get('win_rate', 0):.1%}  "
            f"交易 {m.get('num_trades', 0)} 笔"
        )
