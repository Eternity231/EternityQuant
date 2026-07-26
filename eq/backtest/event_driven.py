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

from eq.backtest.metrics import compute_metrics
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
        entry_i = 0
        trades = []
        equity_curve = []

        costs = cfg.resolve_costs()
        slippage = costs.slippage_rate
        # 兼容旧写法：没有最低佣金/印花税时，commission 仍是一个常数比例
        commission = costs.commission_rate + costs.transfer_fee + costs.platform_fee

        def _buy_cost(value: float) -> float:
            """买入这笔的总费用（绝对金额）。"""
            return costs.trade_cost(value, "buy")

        def _sell_cost(value: float) -> float:
            """卖出这笔的总费用（含印花税——A 股印花税只在卖出收）。"""
            return costs.trade_cost(value, "sell")

        # 预计算涨跌停标记（A 股 ±10% 简化）
        close = df["close"]
        prev_close = close.shift(1).fillna(close)
        limit_up = close >= prev_close * 1.099
        limit_down = close <= prev_close * 0.901

        for i, idx in enumerate(df.index):
            bar = df.iloc[i]
            cur_close = float(bar["close"])

            # 信号生成：用截至当前 bar 的数据（避免前视偏差）
            # execution_delay=1 时改用 i-1 根的信号，模拟「今天看到、明天下单」
            sig_i = i - max(0, int(getattr(cfg, "execution_delay", 0)))
            if sig_i < 0:
                equity_curve.append((idx, cash + shares * cur_close))
                continue
            hist_df = df.iloc[: sig_i + 1]
            try:
                sig = signal(hist_df)
                cur_sig = sig.iloc[-1] if len(sig) else HOLD
            except Exception:
                cur_sig = HOLD

            # v0.27：策略可返回连续目标仓位（0~1 的数），此时按目标仓位调仓，
            # 而不是只有满仓/空仓两档
            if isinstance(cur_sig, (int, float, np.floating)) and not isinstance(cur_sig, bool):
                equity_now = cash + shares * cur_close
                target_frac = float(np.clip(cur_sig, 0.0, 1.0))
                exec_price = cur_close * (1 + slippage)
                target_shares = costs.round_lots(
                    equity_now * target_frac / (exec_price * (1 + commission)))
                delta = target_shares - shares
                if delta > 0 and not limit_up.iloc[i]:
                    value = delta * exec_price
                    cost = value + _buy_cost(value)
                    if cost <= cash:
                        if shares == 0:
                            entry_price, entry_date, entry_i = exec_price, idx, i
                        else:
                            # 加仓：入场价按加权平均更新，否则 pnl_pct 会算错
                            entry_price = (entry_price * shares + exec_price * delta) / (shares + delta)
                        cash -= cost
                        shares += delta
                elif delta < 0 and not limit_down.iloc[i]:
                    sell_price = cur_close * (1 - slippage)
                    qty = min(-delta, shares)
                    gross = qty * sell_price
                    proceeds = gross - _sell_cost(gross)
                    cash += proceeds
                    shares -= qty
                    if shares == 0 and entry_price > 0:
                        basis = qty * entry_price
                        trades.append({
                            "entry_date": entry_date, "exit_date": idx,
                            "entry_price": entry_price, "exit_price": sell_price,
                            "shares": qty,
                            "pnl": proceeds - (basis + _buy_cost(basis)),
                            "pnl_pct": (proceeds - basis - _buy_cost(basis)) / basis,
                            "hold_bars": i - entry_i,
                        })
                        entry_price, entry_date = 0.0, None
                equity_curve.append((idx, cash + shares * cur_close))
                continue

            # 执行逻辑：BUY → 全仓买，SELL → 清仓卖（三态信号的原有语义）
            if cur_sig == BUY and shares == 0 and not limit_up.iloc[i]:
                # 买入：滑点抬升成交价
                exec_price = cur_close * (1 + slippage)
                max_shares = costs.round_lots(cash / (exec_price * (1 + commission)))
                # 买完还要留出费用，不够就再减一手
                while max_shares > 0 and (max_shares * exec_price
                                          + _buy_cost(max_shares * exec_price)) > cash:
                    max_shares -= max(costs.lot_size, 1)
                if max_shares >= max(costs.lot_size, 1):
                    shares = max_shares
                    value = shares * exec_price
                    cash -= value + _buy_cost(value)
                    entry_price = exec_price
                    entry_date = idx
                    entry_i = i
            elif cur_sig == SELL and shares > 0 and not limit_down.iloc[i]:
                # 卖出：滑点压低成交价
                exec_price = cur_close * (1 - slippage)
                gross = shares * exec_price
                proceeds = gross - _sell_cost(gross)
                basis = shares * entry_price
                cost_basis = basis + _buy_cost(basis)
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": idx,
                    "entry_price": entry_price,
                    "exit_price": exec_price,
                    "shares": shares,
                    "pnl": proceeds - cost_basis,     # 含全部费用的真实盈亏（元）
                    "pnl_pct": (proceeds - cost_basis) / cost_basis,
                    "hold_bars": i - entry_i,
                })
                cash += proceeds
                shares = 0
                entry_price = 0
                entry_date = None

            # 权益 = 现金 + 持仓市值
            equity = cash + shares * cur_close
            equity_curve.append((idx, equity))

        # 注：循环里最后一根 bar 已经把持仓市值计进权益了，
        # 此前这里又 append 了一次末日权益，导致权益曲线末尾出现重复索引
        # （重复日期会让 to_parquet / 画图 / pct_change 全部出问题）。

        equity_series = pd.Series(
            [v for _, v in equity_curve],
            index=[d for d, _ in equity_curve],
            name="equity",
        )

        trades_df = pd.DataFrame(trades)
        metrics = compute_metrics(equity_series, trades_df)
        # 未平仓的浮盈单独报出来，避免"总收益里有一部分还没落袋"看不见
        metrics["open_position_shares"] = float(shares)

        return BacktestResult(
            config=cfg,
            equity_curve=equity_series,
            trades=trades_df,
            metrics=metrics,
        )
