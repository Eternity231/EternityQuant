"""散户实用工具 + 执行延迟（v0.30）。"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from eq.backtest.cost import A_SHARE, US_STOCK
from eq.backtest.portfolio import PortfolioConfig, run_portfolio
from eq.backtest.types import BacktestConfig
from eq.backtest.vectorized import VectorizedBacktester
from eq.strategy import BUY, HOLD, SELL
from eq.strategy import retail as R
from eq.strategy.signals import ema_cross


def _bars(seed=0, n=400, trend=0.05, vol=1.5, start="2024-01-01"):
    rng = np.random.default_rng(seed)
    c = np.maximum(100 + np.cumsum(rng.normal(trend, vol, n)), 5.0)
    nz = np.abs(rng.normal(0, 0.8, n))
    return pd.DataFrame({"open": c, "high": c + nz, "low": c - nz, "close": c,
                         "volume": np.full(n, 1e6)},
                        index=pd.bdate_range(start, periods=n))


# ====================== 资金量 → 持仓数 ======================

def test_suggest_positions_scales_with_capital():
    small = R.suggest_positions(30_000)
    big = R.suggest_positions(2_000_000)
    assert small["max_positions"] < big["max_positions"]
    assert small["per_position"] < big["per_position"]


def test_suggest_positions_caps_cost_ratio():
    """建议的单只金额下，实际费率应不超过设定上限。"""
    for cap in (50_000, 200_000, 1_000_000):
        a = R.suggest_positions(cap, max_cost_ratio=0.0010)
        assert a["cost_ratio_at_ticket"] <= 0.0011, f"{cap} 的费率 {a['cost_ratio_at_ticket']}"


def test_tiny_capital_gets_one_position_and_etf_advice():
    a = R.suggest_positions(5_000)
    assert a["max_positions"] == 1
    assert "ETF" in a["note"]


def test_no_min_commission_market_is_not_constrained():
    """美股零佣金无最低，持仓数不该被成本约束卡住。"""
    a = R.suggest_positions(20_000, US_STOCK)
    assert a["min_ticket"] == 0.0
    assert a["max_positions"] == 20


def test_suggest_positions_rejects_bad_capital():
    with pytest.raises(ValueError, match="必须为正"):
        R.suggest_positions(0)


def test_turnover_budget_shrinks_for_small_accounts():
    """小账户单笔金额小 → 单次来回成本高 → 允许的换手次数更少。"""
    small = R.turnover_budget(30_000, 5, A_SHARE)
    big = R.turnover_budget(3_000_000, 5, A_SHARE)
    assert small["round_trips_per_year"] < big["round_trips_per_year"]
    assert small["avg_hold_days"] > big["avg_hold_days"]


def test_turnover_budget_math():
    b = R.turnover_budget(1_000_000, 10, A_SHARE, annual_cost_budget=0.02)
    assert b["cost_per_round_trip"] * b["round_trips_per_year"] == pytest.approx(0.02)


def test_turnover_budget_rejects_zero_positions():
    with pytest.raises(ValueError, match="必须为正"):
        R.turnover_budget(100_000, 0)


def test_advise_and_format():
    a = R.advise(200_000)
    assert {"costs", "positions", "turnover"} <= set(a)
    out = R.format_advice(a)
    assert "建议持仓数" in out and "换手预算" in out


# ====================== 大盘闸门 ======================

def _index_up_then_down(n=400):
    up = np.linspace(100, 200, n // 2)
    down = np.linspace(200, 120, n - n // 2)
    c = np.concatenate([up, down])
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c,
                         "volume": np.full(n, 1e6)},
                        index=pd.bdate_range("2024-01-01", periods=n))


def test_market_filter_turns_off_in_downtrend():
    idx = _index_up_then_down()
    ok = R.market_filter(idx, ma_period=60, confirm_days=3)
    assert ok.iloc[150:190].mean() > 0.8, "上涨段应允许持股"
    assert not ok.iloc[-30:].any(), "下跌段末尾应禁止持股"


def test_market_filter_confirm_days_reduces_flapping():
    """确认天数越多，开关切换次数越少。"""
    rng = np.random.default_rng(0)
    c = 100 + np.cumsum(rng.normal(0, 2.0, 400))
    idx = pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c,
                        "volume": np.full(400, 1e6)},
                       index=pd.bdate_range("2024-01-01", periods=400))
    flips1 = R.market_filter(idx, ma_period=50, confirm_days=1).diff().abs().sum()
    flips5 = R.market_filter(idx, ma_period=50, confirm_days=5).diff().abs().sum()
    assert flips5 <= flips1


def test_with_market_filter_zeroes_position_when_gate_shut():
    idx = _index_up_then_down()
    bars = _bars(n=len(idx), trend=0.05)
    bars.index = idx.index
    fn = R.with_market_filter(lambda d: pd.Series(BUY, index=d.index), idx, ma_period=60)
    pos = fn(bars)
    assert pos.iloc[-20:].sum() == 0, "闸门关闭时目标仓位必须为 0"
    assert pos.iloc[150:190].mean() > 0.8


def test_market_filter_restores_position_when_gate_reopens():
    """回归：闸门作用在持仓状态上，不能作用在三态信号上。

    三态信号只在穿越时发 BUY。若在闸门关闭期间吞掉 BUY，
    闸门重开时策略正处于"已持有"的静默状态，就再也不会重新入场——
    闸门会退化成只会卖不会买的单向阀门。
    """
    n = 300
    # 指数：跌 → 涨（闸门先关后开）
    c = np.concatenate([np.linspace(200, 120, n // 2), np.linspace(120, 260, n - n // 2)])
    idx = pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c,
                        "volume": np.full(n, 1e6)},
                       index=pd.bdate_range("2024-01-01", periods=n))
    bars = _bars(n=n)
    bars.index = idx.index

    # 策略：第 5 根发一次 BUY 之后再不发信号（一直持有）
    def once(d):
        s = pd.Series(HOLD, index=d.index)
        s.iloc[5] = BUY
        return s

    pos = R.with_market_filter(once, idx, ma_period=60)(bars)
    assert pos.iloc[-20:].mean() > 0.9, "闸门重开后必须恢复持仓"


def test_with_market_filter_handles_continuous_positions():
    idx = _index_up_then_down()
    bars = _bars(n=len(idx))
    bars.index = idx.index
    fn = R.with_market_filter(lambda d: pd.Series(0.5, index=d.index), idx, ma_period=60)
    pos = fn(bars)
    assert pos.max() <= 0.5
    assert pos.iloc[-20:].sum() == 0


# ====================== 执行延迟 ======================

def test_execution_delay_shifts_positions():
    """延迟 1 根 = 用昨天的信号，收益应等于把信号整体后移一根。"""
    bars = _bars(seed=3, n=300)
    cfg0 = BacktestConfig(cost_model=None, commission_bps=0, slippage_bps=0)
    cfg1 = dataclasses.replace(cfg0, execution_delay=1)
    r0 = VectorizedBacktester().run(bars, ema_cross, cfg0)
    r1 = VectorizedBacktester().run(bars, ema_cross, cfg1)
    assert r0.metrics["total_return"] != r1.metrics["total_return"]


def test_execution_delay_zero_is_legacy_behaviour():
    bars = _bars(seed=4)
    a = VectorizedBacktester().run(bars, ema_cross, BacktestConfig()).metrics
    b = VectorizedBacktester().run(
        bars, ema_cross, BacktestConfig(execution_delay=0)).metrics
    assert a["total_return"] == pytest.approx(b["total_return"])


def test_execution_delay_removes_same_bar_advantage():
    """构造一个「用当根收盘信息」的策略：延迟后优势必须消失。

    这正是执行延迟要防的事——散户看到收盘价时已经收盘了。
    """
    bars = _bars(seed=5, n=300, vol=2.0)
    ret = bars["close"].pct_change()

    def same_bar(d):
        # 当根上涨就满仓（用了当根收盘价，现实中拿不到）
        return (ret.reindex(d.index) > 0).astype(float)

    cfg = BacktestConfig(cost_model=None, commission_bps=0, slippage_bps=0)
    no_delay = VectorizedBacktester().run(bars, same_bar, cfg).metrics["total_return"]
    delayed = VectorizedBacktester().run(
        bars, same_bar, dataclasses.replace(cfg, execution_delay=1)).metrics["total_return"]
    assert no_delay > delayed, "延迟必须削掉「用当根收盘价」带来的虚假优势"


@pytest.mark.parametrize("engine", ["vectorized", "event_driven"])
def test_both_engines_accept_execution_delay(engine):
    from eq.backtest.event_driven import EventDrivenBacktester

    bars = _bars(seed=6)
    cls = VectorizedBacktester if engine == "vectorized" else EventDrivenBacktester
    m = cls().run(bars, ema_cross, BacktestConfig(execution_delay=1)).metrics
    assert np.isfinite(m["total_return"])


def test_portfolio_execution_delay():
    uni = {f"S{i}": _bars(seed=i, n=300) for i in range(5)}
    base = PortfolioConfig(initial_cash=500_000, max_positions=3,
                           cost_model=None, commission_bps=0, slippage_bps=0)
    r0 = run_portfolio(uni, ema_cross, dataclasses.replace(base, execution_delay=0))
    r1 = run_portfolio(uni, ema_cross, dataclasses.replace(base, execution_delay=1))
    assert r0.metrics["total_return"] != r1.metrics["total_return"]


def test_portfolio_defaults_to_delayed_execution():
    """组合层面默认就该是 T+1——调仓要动一篮子票，更不可能在收盘瞬间完成。"""
    assert PortfolioConfig().execution_delay == 1
