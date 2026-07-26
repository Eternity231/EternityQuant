"""回测配置和结果数据契约（双引擎共享）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd


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
    # v0.29：真实成本模型（印花税仅卖出、最低佣金、过户费）。
    # 传字符串（"a_share"/"hk"/"us"/"crypto"）或 CostModel 实例；
    # None 时退回上面两个 bps 的旧行为，老代码不受影响。
    cost_model: object | None = None
    # v0.30：执行延迟（bar 数）。0 = 收盘价看到信号、同一根收盘价成交——
    # 这对散户是**做不到**的：你看到收盘价时已经收盘了。
    # 1 = 次日成交（真实约束），会让所有回测数字下降，但那才是能拿到的收益。
    execution_delay: int = 0

    def resolve_costs(self):
        """拿到实际生效的 :class:`~eq.backtest.cost.CostModel`。"""
        from eq.backtest.cost import from_bps, get_cost_model

        m = get_cost_model(self.cost_model)
        return m if m is not None else from_bps(self.commission_bps, self.slippage_bps)


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

    def detail(self) -> str:
        """完整指标（含 Sortino / Calmar / 盈亏比 / 回撤持续期）。"""
        m = self.metrics
        pf = m.get("profit_factor", 0)
        pf_txt = "∞" if pf == float("inf") else f"{pf:.2f}"
        return (
            f"  总收益   {m.get('total_return', 0):>+8.2%}    年化     {m.get('annual_return', 0):>+8.2%}\n"
            f"  夏普     {m.get('sharpe', 0):>+8.2f}    Sortino  {m.get('sortino', 0):>+8.2f}\n"
            f"  Calmar   {m.get('calmar', 0):>+8.2f}    年化波动 {m.get('volatility', 0):>8.2%}\n"
            f"  最大回撤 {m.get('max_drawdown', 0):>+8.2%}    回撤天数 {m.get('max_dd_days', 0):>8d}\n"
            f"  胜率     {m.get('win_rate', 0):>8.1%}    盈亏比   {pf_txt:>8}\n"
            f"  平均盈   {m.get('avg_win', 0):>+8.2%}    平均亏   {m.get('avg_loss', 0):>+8.2%}\n"
            f"  交易     {m.get('num_trades', 0):>8d} 笔  回测 bar {m.get('num_bars', 0):>6d} 根"
        )
