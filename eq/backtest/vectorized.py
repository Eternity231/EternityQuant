"""向量化回测引擎（problem 9 冶议：研发阶段用，快）。

策略是 Callable[[pd.DataFrame], pd.Series]，返回值 ∈ {BUY, SELL, HOLD}（problem 10 冶议）。

第一版简化假设：
- 信号触发的当根 close 全仓进出（无仓位管理、无分批）
- 手续费 + 滑点按 bps 应用
- 不做空、不加杠杆、不留现金外资产
- 涨停日不买、跌停日不卖（后处理校正，而非事件驱动）
"""

from __future__ import annotations

import datetime as dt
from typing import Callable

import numpy as np
import pandas as pd

from eq.backtest.types import BacktestConfig, BacktestResult
from eq.strategy import BUY, SELL, HOLD

SignalFunc = Callable[[pd.DataFrame], pd.Series]


class VectorizedBacktester:
    """向量化回测器。"""

    def run(self, df: pd.DataFrame, signal: SignalFunc, config: BacktestConfig | None = None) -> BacktestResult:
        cfg = config or BacktestConfig()
        cfg.engine = "vectorized"

        # 1. 信号生成
        sig = signal(df)
        # 对齐索引，避免信号 df 长度不匹配
        sig = sig.reindex(df.index).fillna(HOLD)

        # 2. 涨跌停后处理：涨停不可买，跌停不可卖（A 股 ±10% 简化）
        close = df["close"]
        prev_close = close.shift(1).fillna(close)
        limit_up = close >= prev_close * 1.099      # 留 0.1% 浮动容忍
        limit_down = close <= prev_close * 0.901
        sig = sig.where(~((sig == BUY) & limit_up), HOLD)
        sig = sig.where(~((sig == SELL) & limit_down), HOLD)

        # 3. 持仓状态：BUY → 持仓 1，SELL → 持仓 0，HOLD → 维持前态
        pos_target = pd.Series(np.nan, index=df.index, dtype=float)
        pos_target[sig == BUY] = 1.0
        pos_target[sig == SELL] = 0.0
        pos = pos_target.ffill().fillna(0.0)

        # 4. 成本调整后的等比收益（持仓期间）
        commission = cfg.commission_bps / 1e4
        slippage = cfg.slippage_bps / 1e4
        # 切换时的换手成本：从 pos.shift(1) 到 pos 的变化幅度
        turn = (pos - pos.shift(1).fillna(0)).abs()
        cost_ratio = turn * (commission + slippage)

        asset_return = close.pct_change().fillna(0)
        # 策略净收益 = 持仓 * 资产收益 - 换手 * 成本
        strategy_return = pos.shift(1).fillna(0) * asset_return - cost_ratio
        equity = cfg.initial_cash * (1 + strategy_return).cumprod()

        # 5. 交易明细：每次 pos 变化即一笔
        trades = self._extract_trades(df, sig, pos, cfg)

        # 6. 关键指标
        metrics = self._compute_metrics(equity, trades, cfg)

        return BacktestResult(
            config=cfg,
            equity_curve=equity.rename("equity"),
            trades=trades,
            metrics=metrics,
        )

    def _extract_trades(self, df: pd.DataFrame, sig: pd.Series, pos: pd.Series, cfg: BacktestConfig) -> pd.DataFrame:
        """从 pos 变化提取买卖点。第一版用简化配对：BUY 到下一个 SELL 之间为一次交易。"""
        trades = []
        in_pos = False
        entry_date = None
        entry_price = None
        prev_pos = 0.0
        for i, idx in enumerate(df.index):
            cur_pos = pos.iloc[i]
            if cur_pos != prev_pos:
                price = df["close"].iloc[i]
                date = idx
                if cur_pos > 0 and not in_pos:
                    # 买入
                    in_pos = True
                    entry_date = date
                    entry_price = price * (1 + cfg.slippage_bps / 1e4)
                elif cur_pos == 0 and in_pos:
                    # 卖出
                    exit_price = price * (1 - cfg.slippage_bps / 1e4)
                    pnl = (exit_price - entry_price) / entry_price - 2 * cfg.commission_bps / 1e4
                    trades.append({
                        "entry_date": entry_date,
                        "exit_date": date,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "shares": 100,  # 简化：固定 100 股
                        "pnl_pct": pnl,
                    })
                    in_pos = False
            prev_pos = cur_pos
        return pd.DataFrame(trades)

    def _compute_metrics(self, equity: pd.Series, trades: pd.DataFrame, cfg: BacktestConfig) -> dict:
        total_return = equity.iloc[-1] / equity.iloc[0] - 1
        # 年化：假设日线 252 个交易日
        n_days = len(equity)
        years = max(n_days / 252, 1e-9)
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        # 夏普：日收益 / std * sqrt(252)，无风险利率 0 简化
        daily_ret = equity.pct_change().fillna(0)
        sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0
        # 最大回撤
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        max_dd = drawdown.min()
        # 胜率
        if not trades.empty:
            win_rate = (trades["pnl_pct"] > 0).mean()
        else:
            win_rate = 0.0
        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "num_trades": len(trades),
        }
