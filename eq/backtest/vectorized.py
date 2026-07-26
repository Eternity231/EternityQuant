"""向量化回测引擎（problem 9 冶议：研发阶段用，快）。

策略是 Callable[[pd.DataFrame], pd.Series]，返回值 ∈ {BUY, SELL, HOLD}（problem 10 冶议）。

第一版简化假设：
- 信号触发的当根 close 全仓进出（无仓位管理、无分批）
- 手续费 + 滑点按 bps 应用
- 不做空、不加杠杆、不留现金外资产
- 涨停日不买、跌停日不卖（后处理校正，而非事件驱动）
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from eq.backtest.metrics import compute_metrics
from eq.backtest.types import BacktestConfig, BacktestResult
from eq.strategy import BUY, SELL, HOLD

SignalFunc = Callable[[pd.DataFrame], pd.Series]


class VectorizedBacktester:
    """向量化回测器。"""

    def run(self, df: pd.DataFrame, signal: SignalFunc, config: BacktestConfig | None = None) -> BacktestResult:
        cfg = config or BacktestConfig()
        cfg.engine = "vectorized"

        # 1. 信号生成
        raw = pd.Series(signal(df)).reindex(df.index)
        # v0.27：策略可以直接返回**连续目标仓位**（0~1 的数值序列）而不只是三态。
        # 这样 eq.strategy.risk 的波动率定仓/ATR 止损才有地方落地——
        # 原来只有 BUY=满仓 / SELL=空仓 两档，仓位管理无从谈起。
        continuous = not (raw.dtype == object or raw.isin([BUY, SELL, HOLD]).any())

        close = df["close"]
        prev_close = close.shift(1).fillna(close)
        # 2. 涨跌停后处理：涨停不可买，跌停不可卖（A 股 ±10% 简化）
        limit_up = close >= prev_close * 1.099      # 留 0.1% 浮动容忍
        limit_down = close <= prev_close * 0.901

        if continuous:
            target = raw.astype("float64").fillna(0.0).clip(0.0, 1.0)
            # 涨停日不许加仓、跌停日不许减仓 → 维持前一根的仓位
            prev_pos = target.shift(1).fillna(0.0)
            blocked = (limit_up & (target > prev_pos)) | (limit_down & (target < prev_pos))
            pos = target.where(~blocked, prev_pos)
            # 交易明细仍按「从 0 变正 / 从正变 0」配对，沿用三态语义
            sig = pd.Series(HOLD, index=df.index)
            sig[(pos > 0) & (prev_pos <= 0)] = BUY
            sig[(pos <= 0) & (prev_pos > 0)] = SELL
        else:
            sig = raw.fillna(HOLD)
            sig = sig.where(~((sig == BUY) & limit_up), HOLD)
            sig = sig.where(~((sig == SELL) & limit_down), HOLD)
            # 3. 持仓状态：BUY → 持仓 1，SELL → 持仓 0，HOLD → 维持前态
            pos_target = pd.Series(np.nan, index=df.index, dtype=float)
            pos_target[sig == BUY] = 1.0
            pos_target[sig == SELL] = 0.0
            pos = pos_target.ffill().fillna(0.0)

        # 3.5 执行延迟：信号在 T 根收盘产生，散户最快 T+1 才能成交。
        # delay=0 等于假设你能在收盘瞬间以收盘价成交——这个假设会系统性
        # 高估收益，尤其对高频策略（信号越频繁，抢不到的成交越多）。
        delay = max(0, int(getattr(cfg, "execution_delay", 0)))
        if delay:
            pos = pos.shift(delay).fillna(0.0)
            sig = sig.shift(delay).fillna(HOLD)

        # 4. 成本调整后的等比收益（持仓期间）
        costs = cfg.resolve_costs()
        prev_pos = pos.shift(1).fillna(0)
        turn = (pos - prev_pos).abs()
        asset_return = close.pct_change().fillna(0)

        if costs.min_commission > 0:
            # 有最低佣金时费率取决于**成交金额**，不再是常数，
            # 只能逐 bar 推进（用上一根的权益估算本次成交金额）。
            # 这正是最低佣金要单独处理的原因：小额交易实际费率远高于名义费率。
            eq_arr = np.empty(len(df))
            equity_now = cfg.initial_cash
            turn_arr = turn.to_numpy()
            delta_arr = (pos - prev_pos).to_numpy()
            ret_arr = (prev_pos * asset_return).to_numpy()
            for i in range(len(df)):
                gross = equity_now * (1.0 + ret_arr[i])
                t = turn_arr[i]
                if t > 0:
                    value = equity_now * t
                    side = "buy" if delta_arr[i] > 0 else "sell"
                    gross -= costs.trade_cost(value, side) + value * costs.slippage_rate
                equity_now = max(gross, 0.0)
                eq_arr[i] = equity_now
            equity = pd.Series(eq_arr, index=df.index)
            strategy_return = equity.pct_change().fillna(equity.iloc[0] / cfg.initial_cash - 1)
        else:
            # 无最低佣金 → 费率是常数，纯向量化
            flat = costs.commission_rate + costs.slippage_rate + costs.transfer_fee \
                + costs.platform_fee
            cost_ratio = turn * flat
            # 印花税只在卖出收，单独加
            sell_turn = (prev_pos - pos).clip(lower=0)
            cost_ratio = cost_ratio + sell_turn * costs.stamp_duty_sell \
                + (pos - prev_pos).clip(lower=0) * costs.stamp_duty_buy
            strategy_return = prev_pos * asset_return - cost_ratio
            equity = cfg.initial_cash * (1 + strategy_return).cumprod()

        # 5. 交易明细：每次 pos 变化即一笔
        trades = self._extract_trades(df, sig, pos, cfg)

        # 6. 关键指标
        metrics = compute_metrics(equity, trades)

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
        entry_i = 0
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
                    entry_i = i
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
                        "hold_bars": i - entry_i,
                    })
                    in_pos = False
            prev_pos = cur_pos
        return pd.DataFrame(trades)
