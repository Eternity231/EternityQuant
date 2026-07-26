"""回测引擎与绩效指标（v0.24 新增，纯逻辑无网络）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eq.backtest.metrics import annualize, compute_metrics, max_drawdown
from eq.backtest.types import BacktestConfig
from eq.backtest.event_driven import EventDrivenBacktester
from eq.backtest.vectorized import VectorizedBacktester
from eq.strategy import BUY, HOLD, SELL
from eq.strategy.signals import adx_trend, bollinger_break, ema_cross, rsi_reversal


# ---------- metrics ----------

def test_annualize_handles_total_wipeout():
    """总收益 ≤ -100% 时 (1+r)**(1/y) 是负数开分数次方 → nan。此前报表直接显示 nan%。"""
    assert annualize(-1.0, 252) == -1.0
    assert annualize(-1.5, 252) == -1.0
    assert not np.isnan(annualize(-1.0, 252))


def test_annualize_no_extrapolation_on_short_window():
    """20 个 bar 的 +3% 不该外推成 +45% 的"年化"。"""
    short = annualize(0.03, 20)
    assert short == pytest.approx(0.03)
    # 一年以上正常年化
    assert annualize(0.10, 252) == pytest.approx(0.10, abs=1e-6)
    assert annualize(0.21, 504) == pytest.approx(0.10, abs=1e-3)


def test_max_drawdown_and_duration():
    eq = pd.Series([100, 110, 90, 95, 120], dtype=float)
    dd, days = max_drawdown(eq)
    assert dd == pytest.approx((90 - 110) / 110)
    assert days == 2  # 两根 bar 处于水下


def test_compute_metrics_on_empty():
    m = compute_metrics(pd.Series(dtype=float), pd.DataFrame())
    assert m["total_return"] == 0.0
    assert m["num_trades"] == 0


def test_compute_metrics_keys():
    eq = pd.Series(np.linspace(100, 120, 300))
    trades = pd.DataFrame({"pnl_pct": [0.05, -0.02, 0.03]})
    m = compute_metrics(eq, trades)
    for k in ("total_return", "annual_return", "sharpe", "sortino", "calmar",
              "max_drawdown", "max_dd_days", "volatility", "win_rate",
              "profit_factor", "avg_win", "avg_loss", "num_trades", "num_bars"):
        assert k in m, f"缺指标 {k}"
    assert m["win_rate"] == pytest.approx(2 / 3)
    assert m["profit_factor"] == pytest.approx(0.08 / 0.02)


# ---------- 信号 ----------

@pytest.fixture
def trending_bars():
    """120 根带趋势和回撤的行情，够跑所有内置策略。"""
    n = 120
    idx = pd.bdate_range("2025-01-01", periods=n)
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0.15, 1.8, n))
    close = np.maximum(close, 5.0)
    return pd.DataFrame({
        "open": close + rng.normal(0, 0.3, n),
        "high": close + np.abs(rng.normal(0, 1.0, n)),
        "low": close - np.abs(rng.normal(0, 1.0, n)),
        "close": close,
        "volume": rng.integers(1_000_000, 9_000_000, n).astype(float),
    }, index=idx)


@pytest.mark.parametrize("fn", [ema_cross, adx_trend, rsi_reversal, bollinger_break])
def test_signals_do_not_raise_and_return_valid_values(fn, trending_bars):
    """adx_trend 此前必崩：bool Series shift 退化成 object，`~prev_above` TypeError。"""
    sig = fn(trending_bars)
    assert len(sig) == len(trending_bars)
    assert set(sig.unique()) <= {BUY, SELL, HOLD}


# ---------- 引擎 ----------

@pytest.mark.parametrize("engine_cls", [VectorizedBacktester, EventDrivenBacktester])
def test_engine_runs_and_reports(engine_cls, trending_bars):
    res = engine_cls().run(trending_bars, ema_cross, BacktestConfig(initial_cash=1_000_000))
    assert len(res.equity_curve) > 0
    assert np.isfinite(res.metrics["total_return"])
    assert np.isfinite(res.metrics["annual_return"])
    assert isinstance(res.summary(), str)
    assert isinstance(res.detail(), str)


def test_event_driven_equity_index_has_no_duplicates(trending_bars):
    """此前收盘仍持仓时会再 append 一次末日权益，权益曲线尾部索引重复。"""
    res = EventDrivenBacktester().run(trending_bars, ema_cross, BacktestConfig())
    assert not res.equity_curve.index.duplicated().any()
    assert len(res.equity_curve) == len(trending_bars)


def test_engines_agree_roughly(trending_bars):
    """同一策略同一段行情，两引擎的总收益方向应一致（量级允许有差）。"""
    cfg = BacktestConfig(initial_cash=1_000_000)
    v = VectorizedBacktester().run(trending_bars, ema_cross, cfg).metrics["total_return"]
    e = EventDrivenBacktester().run(trending_bars, ema_cross, BacktestConfig(initial_cash=1_000_000)).metrics["total_return"]
    assert abs(v - e) < 0.10, f"两引擎差异过大 v={v:.4f} e={e:.4f}"
