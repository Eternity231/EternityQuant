"""交易成本模型（v0.29）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eq.backtest import cost as C
from eq.backtest.types import BacktestConfig
from eq.backtest.vectorized import VectorizedBacktester
from eq.strategy.signals import ema_cross


def _bars(n=300, seed=0):
    rng = np.random.default_rng(seed)
    c = np.maximum(100 + np.cumsum(rng.normal(0.05, 1.5, n)), 5.0)
    nz = np.abs(rng.normal(0, 0.8, n))
    return pd.DataFrame({"open": c, "high": c + nz, "low": c - nz, "close": c,
                         "volume": np.full(n, 1e6)},
                        index=pd.bdate_range("2024-01-01", periods=n))


# ====================== 最低佣金 ======================

def test_min_commission_binds_on_small_trades():
    """核心：5000 元的单子按万 2.5 只有 1.25 元，但最低要收 5 元。"""
    m = C.A_SHARE
    assert m.commission(5_000) == 5.0
    assert m.commission(100_000) == pytest.approx(100_000 * 0.00025)


def test_small_trades_pay_far_higher_effective_rate():
    m = C.A_SHARE
    small = m.cost_ratio(2_000, "buy")
    large = m.cost_ratio(200_000, "buy")
    assert small > large * 5, f"小额 {small:.5f} 应远高于大额 {large:.5f}"


def test_cost_ratio_monotonic_decreasing_until_threshold():
    m = C.A_SHARE
    ratios = [m.cost_ratio(v, "buy") for v in (1_000, 5_000, 20_000, 100_000)]
    assert ratios == sorted(ratios, reverse=True)


# ====================== 印花税只在卖出 ======================

def test_a_share_stamp_duty_is_sell_side_only():
    m = C.A_SHARE
    v = 100_000
    buy, sell = m.trade_cost(v, "buy"), m.trade_cost(v, "sell")
    assert sell - buy == pytest.approx(v * 0.001), "差额应正好是千分之一印花税"
    assert buy < sell


def test_hk_stamp_duty_is_both_sides():
    m = C.HK_STOCK
    v = 1_000_000
    assert m.trade_cost(v, "buy") == pytest.approx(m.trade_cost(v, "sell"))


def test_us_has_no_commission_but_sec_fee_on_sell():
    m = C.US_STOCK
    assert m.commission(100_000) == 0.0
    assert m.trade_cost(100_000, "sell") > m.trade_cost(100_000, "buy")


# ====================== 盈亏平衡 ======================

def test_breakeven_includes_both_sides_and_slippage():
    m = C.A_SHARE
    v = 100_000
    expected = m.round_trip_ratio(v) + 2 * m.slippage_rate
    assert m.breakeven_pct(v) == pytest.approx(expected)
    assert m.breakeven_pct(v) > 0.002        # A 股来回至少 0.2%


def test_breakeven_much_worse_for_tiny_accounts():
    assert C.A_SHARE.breakeven_pct(2_000) > C.A_SHARE.breakeven_pct(200_000) * 2


def test_zero_and_negative_values_are_safe():
    for m in (C.A_SHARE, C.HK_STOCK, C.US_STOCK, C.CRYPTO, C.FLAT):
        assert m.trade_cost(0, "buy") == 0.0
        assert m.cost_ratio(0, "buy") == 0.0
        assert m.trade_cost(-5, "sell") == 0.0
        assert m.breakeven_pct(0) >= 0


# ====================== 整手 ======================

def test_round_lots():
    assert C.A_SHARE.round_lots(1234) == 1200      # A 股 100 股一手
    assert C.A_SHARE.round_lots(50) == 0
    assert C.US_STOCK.round_lots(1234.7) == 1234   # 美股可买零股


# ====================== 预设与解析 ======================

def test_presets_and_lookup():
    assert C.get_cost_model("a_share") is C.A_SHARE
    assert C.get_cost_model("A") is C.A_SHARE
    assert C.get_cost_model(None) is None
    assert C.get_cost_model(C.HK_STOCK) is C.HK_STOCK
    with pytest.raises(ValueError, match="未知成本模型"):
        C.get_cost_model("火星股市")


def test_for_market():
    assert C.for_market("A") is C.A_SHARE
    assert C.for_market("hk") is C.HK_STOCK
    assert C.for_market("CRYPTO") is C.CRYPTO
    assert C.for_market("未知") is C.FLAT


def test_from_bps_matches_legacy_rates():
    m = C.from_bps(2.5, 5.0)
    assert m.commission_rate == pytest.approx(0.00025)
    assert m.slippage_rate == pytest.approx(0.0005)
    assert m.min_commission == 0.0 and m.stamp_duty_sell == 0.0


def test_compare_costs_table():
    df = C.compare_costs([5_000, 100_000])
    assert len(df) == 2 and "成交金额" in df.columns


# ====================== 与回测引擎的集成 ======================

def test_config_defaults_to_legacy_behaviour():
    """cost_model=None 时必须和旧的 bps 行为完全一致（不破坏老代码）。"""
    bars = _bars()
    legacy = VectorizedBacktester().run(
        bars, ema_cross, BacktestConfig(commission_bps=2.5, slippage_bps=5.0)).metrics
    explicit = VectorizedBacktester().run(
        bars, ema_cross, BacktestConfig(cost_model=C.from_bps(2.5, 5.0))).metrics
    assert legacy["total_return"] == pytest.approx(explicit["total_return"], abs=1e-12)


def test_real_costs_are_worse_than_flat_bps():
    """A 股真实成本含印花税，一定比旧的对称 bps 模型更贵。"""
    bars = _bars()
    flat = VectorizedBacktester().run(bars, ema_cross, BacktestConfig()).metrics
    real = VectorizedBacktester().run(
        bars, ema_cross, BacktestConfig(cost_model="a_share")).metrics
    assert real["total_return"] < flat["total_return"]


def test_min_commission_hurts_small_accounts_more():
    """同一策略，小账户被最低佣金吃掉的比例应显著更高。"""
    bars = _bars()

    def gap(cash):
        flat = VectorizedBacktester().run(
            bars, ema_cross, BacktestConfig(initial_cash=cash)).metrics["total_return"]
        real = VectorizedBacktester().run(
            bars, ema_cross,
            BacktestConfig(initial_cash=cash, cost_model="a_share")).metrics["total_return"]
        return flat - real

    assert gap(3_000) > gap(1_000_000) * 1.5


def test_event_driven_respects_cost_model():
    from eq.backtest.event_driven import EventDrivenBacktester

    bars = _bars()
    flat = EventDrivenBacktester().run(bars, ema_cross, BacktestConfig()).metrics
    real = EventDrivenBacktester().run(
        bars, ema_cross, BacktestConfig(cost_model="a_share")).metrics
    assert real["total_return"] <= flat["total_return"]


def test_event_driven_buys_whole_lots_under_a_share_model():
    from eq.backtest.event_driven import EventDrivenBacktester

    res = EventDrivenBacktester().run(
        _bars(), lambda d: pd.Series(1.0, index=d.index),
        BacktestConfig(initial_cash=123_456, cost_model="a_share"))
    if not res.trades.empty:
        assert (res.trades["shares"] % 100 == 0).all(), "A 股必须整手交易"
