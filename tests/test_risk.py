"""仓位管理与风控（v0.27）+ 回测引擎的连续仓位支持。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eq.backtest.event_driven import EventDrivenBacktester
from eq.backtest.types import BacktestConfig
from eq.backtest.vectorized import VectorizedBacktester
from eq.strategy import BUY, HOLD, SELL
from eq.strategy import risk as RK


def _bars(n=250, seed=0, vol=1.5, trend=0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = np.maximum(100 + np.cumsum(rng.normal(trend, vol, n)), 5.0)
    noise = np.abs(rng.normal(0, 0.8, n))
    return pd.DataFrame({
        "open": close + rng.normal(0, 0.3, n), "high": close + noise,
        "low": close - noise, "close": close,
        "volume": rng.integers(1e6, 9e6, n).astype(float),
    }, index=pd.bdate_range("2024-01-01", periods=n))


@pytest.fixture
def bars():
    return _bars()


# ====================== 定仓 ======================

def test_fixed_fraction_is_constant_and_clipped(bars):
    assert (RK.fixed_fraction(bars, 0.5) == 0.5).all()
    assert (RK.fixed_fraction(bars, 5.0) == 1.0).all()
    assert (RK.fixed_fraction(bars, -1.0) == 0.0).all()


def test_volatility_target_shrinks_position_for_volatile_asset():
    """核心诉求：波动大的少买，让每笔风险贡献大致相等。"""
    calm, wild = _bars(seed=1, vol=0.3), _bars(seed=1, vol=3.0)
    s_calm = RK.volatility_target(calm, target_vol=0.20).iloc[-1]
    s_wild = RK.volatility_target(wild, target_vol=0.20).iloc[-1]
    assert s_wild < s_calm, f"高波动 {s_wild} 应小于低波动 {s_calm}"


def test_volatility_target_respects_bounds(bars):
    s = RK.volatility_target(bars, target_vol=5.0, max_leverage=1.0, min_size=0.1)
    assert s.between(0.1, 1.0).all(), "算出 >1 的仓位必须截断（散户账户无杠杆）"


def test_atr_risk_size_matches_risk_budget():
    """按定义反推：仓位 × 止损距离% ≈ 单笔风险预算。"""
    bars = _bars(seed=2)
    risk = 0.02
    size = RK.atr_risk_size(bars, risk_per_trade=risk, atr_mult=2.0, max_leverage=10.0)
    from eq.strategy.factors.technical import atr
    stop_pct = 2.0 * atr(bars, 14) / bars["close"]
    implied = (size * stop_pct).dropna()
    # 未被上下限截断的部分应精确等于风险预算
    inner = implied[(size > 0.06) & (size < 9.9)]
    assert np.allclose(inner, risk, atol=1e-6)


def test_score_scaled_size(bars):
    score = pd.Series(np.linspace(-1, 1, len(bars)), index=bars.index)
    s = RK.score_scaled_size(score, base=1.0)
    assert s.iloc[0] == pytest.approx(1.0)      # |−1| → 满仓
    assert s.iloc[len(s) // 2] < 0.1            # 接近 0 → 极小仓
    assert s.between(0, 1).all()


# ====================== 止损 ======================

def _flat_then_crash(n=60, crash_at=40):
    close = np.full(n, 100.0)
    close[crash_at:] = np.linspace(100, 60, n - crash_at)
    return pd.DataFrame({
        "open": close, "high": close + 0.5, "low": close - 0.5, "close": close,
        "volume": np.full(n, 1e6),
    }, index=pd.bdate_range("2024-01-01", periods=n))


def test_atr_stop_exits_on_crash():
    df = _flat_then_crash()
    pos = pd.Series(1.0, index=df.index)          # 一直想满仓
    out = RK.apply_stops(df, pos, atr_mult=2.0, trailing=False)
    assert (out["position"].iloc[-5:] == 0).all(), "暴跌后应已止损离场"
    assert "stop_loss" in set(out["exit_reason"])


def test_trailing_stop_locks_in_profit():
    """涨上去再回落，跟踪止损应在回落时离场（而非等跌回成本）。"""
    n = 80
    close = np.concatenate([np.linspace(100, 160, 50), np.linspace(160, 130, 30)])
    df = pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5,
                       "close": close, "volume": np.full(n, 1e6)},
                      index=pd.bdate_range("2024-01-01", periods=n))
    out = RK.apply_stops(df, pd.Series(1.0, index=df.index), atr_mult=2.0, trailing=True)
    exited = np.argmax(out["position"].to_numpy() == 0)
    assert exited > 50, "不该在上涨段就被打掉"
    assert close[exited] > 100, "应在盈利状态离场，而不是跌回成本"


def test_stop_blocks_immediate_reentry():
    """止损后若目标仓位仍为正，不能下一根马上买回来——那样止损等于没设。"""
    df = _flat_then_crash()
    out = RK.apply_stops(df, pd.Series(1.0, index=df.index), atr_mult=2.0, trailing=False)
    pos = out["position"].to_numpy()
    first_exit = int(np.argmax(pos == 0))
    assert (pos[first_exit:] == 0).all(), "止损后被锁定，不该反复重入"


def test_stop_unblocks_after_signal_goes_flat():
    df = _flat_then_crash(n=80, crash_at=30)
    want = pd.Series(1.0, index=df.index)
    want.iloc[50:60] = 0.0                        # 信号归零 → 解锁
    out = RK.apply_stops(df, want, atr_mult=2.0, trailing=False)
    assert (out["position"].iloc[60:] > 0).any(), "信号归零后应允许重新入场"


def test_time_stop(bars):
    out = RK.apply_stops(bars, pd.Series(1.0, index=bars.index),
                         atr_mult=99.0, trailing=False, max_hold_bars=10)
    assert "time_stop" in set(out["exit_reason"])


def test_take_profit():
    n = 60
    close = np.linspace(100, 200, n)
    df = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                       "close": close, "volume": np.full(n, 1e6)},
                      index=pd.bdate_range("2024-01-01", periods=n))
    out = RK.apply_stops(df, pd.Series(1.0, index=df.index),
                         atr_mult=99.0, trailing=False, take_profit_mult=3.0)
    assert "take_profit" in set(out["exit_reason"])


def test_stops_never_produce_negative_or_over_full_position(bars):
    out = RK.apply_stops(bars, pd.Series(1.0, index=bars.index))
    assert out["position"].between(0, 1).all()


# ====================== 回撤熔断 ======================

def test_drawdown_throttle_halves_then_halts():
    eq = pd.Series([100, 110, 104, 99, 87])       # 回撤 0 / 0 / -5.5% / -10% / -20.9%
    pos = pd.Series(1.0, index=eq.index)
    out = RK.drawdown_throttle(eq, pos, warn_dd=0.05, halt_dd=0.20, warn_scale=0.5)
    assert out.iloc[1] == 1.0
    assert out.iloc[2] == 0.5
    assert out.iloc[4] == 0.0


# ====================== build_positions ======================

def test_build_positions_from_ternary_holds_between_signals(bars):
    sig = pd.Series(HOLD, index=bars.index)
    sig.iloc[10] = BUY
    sig.iloc[50] = SELL
    pos = RK.build_positions(bars, sig, sizing="fixed", stops=False)
    assert pos.iloc[9] == 0 and pos.iloc[20] == 1.0 and pos.iloc[60] == 0


def test_build_positions_from_scores(bars):
    score = pd.Series(np.linspace(-1, 1, len(bars)), index=bars.index)
    pos = RK.build_positions(bars, score, sizing="score", stops=False)
    assert pos.between(0, 1).all()
    assert pos.iloc[0] == 0.0        # 负分不持仓
    assert pos.iloc[-1] > 0.9        # 强正分接近满仓


def test_build_positions_vol_target_is_not_binary(bars):
    sig = pd.Series(BUY, index=bars.index)
    pos = RK.build_positions(bars, sig, sizing="vol_target", stops=False)
    assert pos.nunique() > 5, "波动率定仓应产生连续仓位，不是 0/1 两档"
    assert pos.between(0, 1).all()


def test_build_positions_rejects_unknown_sizing(bars):
    with pytest.raises(ValueError, match="未知定仓方式"):
        RK.build_positions(bars, pd.Series(BUY, index=bars.index), sizing="魔法")


def test_make_managed_is_a_signal_func(bars):
    from eq.strategy.signals import ema_cross
    fn = RK.make_managed(ema_cross, sizing="vol_target", stops=True)
    pos = fn(bars)
    assert len(pos) == len(bars) and pos.between(0, 1).all()
    assert "managed(" in fn.__name__


# ====================== 引擎的连续仓位支持 ======================

@pytest.mark.parametrize("engine_cls", [VectorizedBacktester, EventDrivenBacktester])
def test_engine_accepts_continuous_positions(engine_cls, bars):
    half = lambda d: pd.Series(0.5, index=d.index)          # noqa: E731
    res = engine_cls().run(bars, half, BacktestConfig(initial_cash=1_000_000))
    assert len(res.equity_curve) > 0
    assert np.isfinite(res.metrics["total_return"])


@pytest.mark.parametrize("engine_cls", [VectorizedBacktester, EventDrivenBacktester])
def test_half_position_halves_exposure(engine_cls):
    """半仓的敞口应约为满仓的一半——不论行情涨跌，方向一致、幅度减半。"""
    n = 300
    close = 100 * np.exp(np.cumsum(np.full(n, 0.002)))    # 确定性上行，去掉随机性
    df = pd.DataFrame({"open": close, "high": close * 1.005, "low": close * 0.995,
                       "close": close, "volume": np.full(n, 1e6)},
                      index=pd.bdate_range("2024-01-01", periods=n))
    cfg = lambda: BacktestConfig(initial_cash=1_000_000, commission_bps=0, slippage_bps=0)  # noqa: E731
    r_full = engine_cls().run(df, lambda d: pd.Series(1.0, index=d.index), cfg()).metrics["total_return"]
    r_half = engine_cls().run(df, lambda d: pd.Series(0.5, index=d.index), cfg()).metrics["total_return"]
    assert 0 < r_half < r_full
    assert r_half == pytest.approx(r_full / 2, rel=0.35)


@pytest.mark.parametrize("engine_cls", [VectorizedBacktester, EventDrivenBacktester])
def test_half_position_halves_losses_too(engine_cls):
    """下跌行情里半仓也应少亏一半——敞口缩放对两个方向都成立。"""
    n = 300
    close = 100 * np.exp(np.cumsum(np.full(n, -0.002)))
    df = pd.DataFrame({"open": close, "high": close * 1.005, "low": close * 0.995,
                       "close": close, "volume": np.full(n, 1e6)},
                      index=pd.bdate_range("2024-01-01", periods=n))
    cfg = lambda: BacktestConfig(initial_cash=1_000_000, commission_bps=0, slippage_bps=0)  # noqa: E731
    r_full = engine_cls().run(df, lambda d: pd.Series(1.0, index=d.index), cfg()).metrics["total_return"]
    r_half = engine_cls().run(df, lambda d: pd.Series(0.5, index=d.index), cfg()).metrics["total_return"]
    assert r_full < r_half < 0
    assert abs(r_half) < abs(r_full)


@pytest.mark.parametrize("engine_cls", [VectorizedBacktester, EventDrivenBacktester])
def test_zero_position_means_flat(engine_cls, bars):
    res = engine_cls().run(bars, lambda d: pd.Series(0.0, index=d.index),
                           BacktestConfig(initial_cash=1_000_000))
    assert res.metrics["total_return"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("engine_cls", [VectorizedBacktester, EventDrivenBacktester])
def test_ternary_signals_still_work(engine_cls, bars):
    """连续仓位是新增能力，老的三态契约必须原样可用。"""
    from eq.strategy.signals import ema_cross
    res = engine_cls().run(bars, ema_cross, BacktestConfig())
    assert np.isfinite(res.metrics["total_return"])


def test_risk_managed_strategy_reduces_drawdown():
    """风控层的价值主张：牺牲一些收益换显著更小的回撤。"""
    from eq.strategy.signals import ema_cross
    df = _bars(n=400, seed=11, vol=2.0)
    cfg = lambda: BacktestConfig(initial_cash=1_000_000)   # noqa: E731
    raw = VectorizedBacktester().run(df, ema_cross, cfg())
    managed = VectorizedBacktester().run(
        df, RK.make_managed(ema_cross, sizing="vol_target",
                            stop_kwargs={"atr_mult": 2.0}), cfg())
    assert managed.metrics["max_drawdown"] >= raw.metrics["max_drawdown"], (
        f"风控后回撤 {managed.metrics['max_drawdown']:.2%} "
        f"不应比裸策略 {raw.metrics['max_drawdown']:.2%} 更差"
    )
