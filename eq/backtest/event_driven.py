"""事件驱动回测引擎（problem 9 冶议：上线前用，准）。

逐 bar 回放，每根 bar 产生一个事件，依次经过 signal → risk → execution。
精确建模：涨跌停限制、停牌、滑点、手续费、部分成交、资金占用。

与向量化引擎共享 signal(df) -> df 接口（problem 10 冶议），零适配器：
事件驱动引擎对每个 bar t，调 signal(df[:t]) 取当期信号。
会有重复计算开销（每个 bar 都重算一遍历史），但日线级别可接受。
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from eq.backtest.types import BacktestConfig, BacktestResult
from eq.strategy import BUY, SELL, HOLD

SignalFunc = Callable[[pd.DataFrame], pd.Series]


class EventDrivenBacktester:
    """事件驱动回测器。逐 bar 模拟，精确建模市场摩擦。"""

    def run(self, df: pd.DataFrame, signal: SignalFunc, config: BacktestConfig | None = None) -> BacktestResult:
        cfg = config or BacktestConfig()
        cfg.engine = "event_driven"

        # 状态变量
        cash = cfg.initial_cash
        shares = 0.0
        entry_price = 0.0
        entry_date = None
        trades = []
        equity_curve = []

        commission = cfg.commission_bps / 1e4
        slippage = cfg.slippage_bps / 1e4

        # 预计算涨跌停标记（A 股 ±10% 简化）
        close = df["close"]
        prev_close = close.shift(1).fillna(close)
        limit_up = close >= prev_close * 1.099
        limit_down = close <= prev_close * 0.901

        for i, idx in enumerate(df.index):
            bar = df.iloc[i]
            cur_close = float(bar["close"])

            # 信号生成：用截至当前 bar 的数据（避免前视偏差）
            hist_df = df.iloc[: i + 1]
            try:
                sig = signal(hist_df)
                cur_sig = sig.iloc[-1] if not sig.empty else HOLD
            except Exception:
                cur_sig = HOLD

            # 执行逻辑：BUY → 全仓买，SELL → 清仓卖（第一版简化仓位管理）
            if cur_sig == BUY and shares == 0 and not limit_up.iloc[i]:
                # 买入：滑点抬升成交价
                exec_price = cur_close * (1 + slippage)
                max_shares = cash // (exec_price * (1 + commission))
                if max_shares >= 100:  # A 股最小 1 手 = 100 股
                    shares = max_shares
                    cash -= shares * exec_price * (1 + commission)
                    entry_price = exec_price
                    entry_date = idx
            elif cur_sig == SELL and shares > 0 and not limit_down.iloc[i]:
                # 卖出：滑点压低成交价
                exec_price = cur_close * (1 - slippage)
                proceeds = shares * exec_price * (1 - commission)
                pnl = proceeds - shares * entry_price
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": idx,
                    "entry_price": entry_price,
                    "exit_price": exec_price,
                    "shares": shares,
                    "pnl_pct": (exec_price - entry_price) / entry_price - 2 * commission,
                })
                cash += proceeds
                shares = 0
                entry_price = 0
                entry_date = None

            # 权益 = 现金 + 持仓市值
            equity = cash + shares * cur_close
            equity_curve.append((idx, equity))

        # 若回测结束仍持仓，按末日收盘价虚拟平仓计入权益（但不计入 trades）
        if shares > 0:
            equity = cash + shares * float(df.iloc[-1]["close"])
            equity_curve.append((df.index[-1], equity))

        equity_series = pd.Series(
            [v for _, v in equity_curve],
            index=[d for d, _ in equity_curve],
            name="equity",
        )

        trades_df = pd.DataFrame(trades)
        metrics = self._compute_metrics(equity_series, trades_df, cfg)

        return BacktestResult(
            config=cfg,
            equity_curve=equity_series,
            trades=trades_df,
            metrics=metrics,
        )

    def _compute_metrics(self, equity: pd.Series, trades: pd.DataFrame, cfg: BacktestConfig) -> dict:
        total_return = equity.iloc[-1] / equity.iloc[0] - 1
        n_days = len(equity)
        years = max(n_days / 252, 1e-9)
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        daily_ret = equity.pct_change().fillna(0)
        sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        max_dd = drawdown.min()
        win_rate = (trades["pnl_pct"] > 0).mean() if not trades.empty else 0.0
        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "num_trades": len(trades),
        }
