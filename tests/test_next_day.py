"""次日高点预测与限价止盈（v0.31）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eq.backtest.cost import A_SHARE, CostModel
from eq.strategy import next_day as N

_FREE = CostModel(name="free", commission_rate=0.0, slippage_rate=0.0)


def _bars(n=300, seed=0, trend=0.0, vol=1.5):
    """合成日线。**必须保证 low ≤ open,close ≤ high**——
    否则会造出现实中不存在的 K 线，让「成交价不超过最高价」这类不变量失效。"""
    rng = np.random.default_rng(seed)
    c = np.maximum(100 + np.cumsum(rng.normal(trend, vol, n)), 5.0)
    o = c + rng.normal(0, 0.3, n)
    hi = np.maximum(c, o) + np.abs(rng.normal(0, 1.0, n))
    lo = np.minimum(c, o) - np.abs(rng.normal(0, 1.0, n))
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c,
                         "volume": np.full(n, 1e6)},
                        index=pd.bdate_range("2024-01-01", periods=n))


def _explicit(rows):
    """手工构造 OHLC，用于精确验证成交规则。"""
    return pd.DataFrame(rows, index=pd.bdate_range("2024-01-01", periods=len(rows)))


# ====================== 基线分布 ======================

def test_baseline_stats_shape():
    s = N.baseline_stats(_bars())
    for k in ("n", "mfe_mean", "mfe_median", "mae_median",
              "close_ret_mean", "pct_up_close", "mfe_over_mae"):
        assert k in s
    assert s["n"] > 0


def test_mfe_is_above_and_mae_below_entry():
    s = N.baseline_stats(_bars(seed=1))
    assert s["mfe_median"] > 0, "次日最高价理应高于今收"
    assert s["mae_median"] < 0, "次日最低价理应低于今收"


def test_baseline_handles_short_input():
    assert N.baseline_stats(_bars(n=1))["n"] == 0


def test_format_baseline_runs():
    out = N.format_baseline(N.baseline_stats(_bars()), "测试")
    assert "MFE" in out and "MAE" in out


# ====================== 特征与标签 ======================

def test_build_dataset_labels_are_correct():
    b = _bars(n=50, seed=2)
    ds = N.build_dataset(b, "TEST")
    exp_mfe = b["high"].shift(-1) / b["close"] - 1
    assert ds["y_mfe"].dropna().round(10).equals(exp_mfe.dropna().round(10))
    assert ds["symbol"].iloc[0] == "TEST"


def test_features_use_no_future_data():
    """把最后一根之后的数据改掉，不应影响任何**特征**（只影响标签）。"""
    b = _bars(n=100, seed=3)
    a = N.build_dataset(b)
    b2 = b.copy()
    b2.iloc[-1, :] = b2.iloc[-1, :] * 5      # 篡改最后一根
    c = N.build_dataset(b2)
    feats = [x for x in a.columns if not x.startswith("y_") and x != "entry_close"]
    # 倒数第二根及以前的特征必须完全不变
    pd.testing.assert_frame_equal(a[feats].iloc[:-1], c[feats].iloc[:-1])


def test_dataset_has_no_inf():
    ds = N.build_dataset(_bars(seed=4))
    assert not np.isinf(ds.select_dtypes("number").to_numpy()).any()


# ====================== 限价成交规则 ======================

def test_fill_at_limit_when_high_touches():
    b = _explicit([
        {"open": 100, "high": 100, "low": 100, "close": 100, "volume": 1e6},
        {"open": 100, "high": 105, "low": 99, "close": 101, "volume": 1e6},
        {"open": 101, "high": 101, "low": 101, "close": 101, "volume": 1e6},
    ])
    r = N.simulate_limit(b, 0.02, costs=_FREE)          # 限价 102
    t = r.trades.iloc[0]
    assert t["reason"] == "limit"
    assert t["exit"] == pytest.approx(102.0)
    assert t["gross"] == pytest.approx(0.02)


def test_gap_open_fills_better_than_limit():
    """跳空高开时应以开盘价成交（优于限价），而不是限价。"""
    b = _explicit([
        {"open": 100, "high": 100, "low": 100, "close": 100, "volume": 1e6},
        {"open": 106, "high": 108, "low": 105, "close": 107, "volume": 1e6},
        {"open": 107, "high": 107, "low": 107, "close": 107, "volume": 1e6},
    ])
    t = N.simulate_limit(b, 0.02, costs=_FREE).trades.iloc[0]   # 限价 102
    assert t["reason"] == "gap_open"
    assert t["exit"] == pytest.approx(106.0), "应拿到开盘价 106，而不是限价 102"


def test_unfilled_exits_at_close():
    b = _explicit([
        {"open": 100, "high": 100, "low": 100, "close": 100, "volume": 1e6},
        {"open": 100, "high": 101, "low": 97, "close": 98, "volume": 1e6},
        {"open": 98, "high": 98, "low": 98, "close": 98, "volume": 1e6},
    ])
    t = N.simulate_limit(b, 0.05, costs=_FREE).trades.iloc[0]   # 限价 105，摸不到
    assert t["reason"] == "close"
    assert t["exit"] == pytest.approx(98.0)
    assert t["gross"] == pytest.approx(-0.02)


def test_stop_loss_takes_precedence_conservatively():
    """同日既触止损又触止盈时无法判断先后，保守按先止损。"""
    b = _explicit([
        {"open": 100, "high": 100, "low": 100, "close": 100, "volume": 1e6},
        {"open": 100, "high": 110, "low": 94, "close": 105, "volume": 1e6},
        {"open": 105, "high": 105, "low": 105, "close": 105, "volume": 1e6},
    ])
    t = N.simulate_limit(b, 0.05, stop_pct=0.03, costs=_FREE).trades.iloc[0]
    assert t["reason"] == "stop"
    assert t["exit"] <= 97.0


def test_limit_up_day_is_not_bought():
    """T 日涨停封板买不进。"""
    b = _explicit([
        {"open": 100, "high": 100, "low": 100, "close": 100, "volume": 1e6},
        {"open": 110, "high": 110, "low": 110, "close": 110, "volume": 1e6},  # +10% 封板
        {"open": 111, "high": 115, "low": 110, "close": 112, "volume": 1e6},
        {"open": 112, "high": 112, "low": 112, "close": 112, "volume": 1e6},
    ])
    r = N.simulate_limit(b, 0.02, costs=_FREE)
    assert pd.Timestamp(b.index[1]) not in r.trades.index, "涨停日不该建仓"


def test_entry_mask_filters_days():
    b = _bars(n=100, seed=5)
    mask = pd.Series(False, index=b.index)
    mask.iloc[10:20] = True
    r = N.simulate_limit(b, 0.01, entry=mask, costs=_FREE)
    assert r.stats["n_trades"] <= 10


def test_series_target_allows_dynamic_limits():
    b = _bars(n=100, seed=6)
    tgt = pd.Series(np.linspace(0.005, 0.05, len(b)), index=b.index)
    r = N.simulate_limit(b, tgt, costs=_FREE)
    assert r.stats["n_trades"] > 0
    # 限价应随目标递增
    assert r.trades["limit"].iloc[-1] / r.trades["entry"].iloc[-1] > \
           r.trades["limit"].iloc[0] / r.trades["entry"].iloc[0]


def test_no_valid_days_returns_empty():
    b = _bars(n=100)
    r = N.simulate_limit(b, 0.01, entry=pd.Series(False, index=b.index))
    assert r.stats["n_trades"] == 0
    assert r.trades.empty


# ====================== 成本与统计 ======================

def test_costs_reduce_net_return():
    b = _bars(n=200, seed=7)
    free = N.simulate_limit(b, 0.01, costs=_FREE).stats["mean_net"]
    real = N.simulate_limit(b, 0.01, costs=A_SHARE).stats["mean_net"]
    assert real < free
    assert free - real == pytest.approx(
        A_SHARE.round_trip_ratio(10_000) + 2 * A_SHARE.slippage_rate, abs=1e-9)


def test_higher_target_lowers_fill_rate():
    b = _bars(n=300, seed=8)
    lo = N.simulate_limit(b, 0.005, costs=_FREE).stats["fill_rate"]
    hi = N.simulate_limit(b, 0.05, costs=_FREE).stats["fill_rate"]
    assert hi < lo, "限价挂得越高，成交率越低"


def test_gross_never_exceeds_mfe():
    """成交价不可能高于次日最高价——这是回测正确性的硬约束。"""
    b = _bars(n=300, seed=9)
    t = N.simulate_limit(b, 0.01, costs=_FREE).trades
    assert (t["gross"] <= t["mfe"] + 1e-9).all()


def test_capture_rate_is_bounded():
    r = N.simulate_limit(_bars(n=300, seed=10), 0.01, costs=_FREE)
    assert 0.0 <= r.stats["capture_rate"] <= 1.0 + 1e-9


def test_summary_string():
    r = N.simulate_limit(_bars(n=200), 0.01)
    assert "成交率" in r.summary()


# ====================== 档位扫描 ======================

def test_scan_targets_table():
    df = N.scan_targets(_bars(n=300, seed=11), [0.005, 0.01, 0.02], costs=_FREE)
    assert len(df) == 3
    assert {"限价档", "fill_rate", "mean_net", "annualized"} <= set(df.columns)
    assert df["fill_rate"].is_monotonic_decreasing


# ====================== 核心结论的回归 ======================

def test_a_share_cost_exceeds_typical_next_day_edge():
    """本模块最重要的结论：A 股一个来回的成本，
    高于「次日收盘收益」的典型均值一个数量级——所以无条件日内来回必亏。"""
    fee = A_SHARE.round_trip_ratio(10_000) + 2 * A_SHARE.slippage_rate
    b = _bars(n=500, seed=12, trend=0.0)
    daily_edge = abs((b["close"].shift(-1) / b["close"] - 1).mean())
    assert fee > daily_edge * 5, f"成本 {fee:.4%} vs 日均漂移 {daily_edge:.4%}"


def test_unconditional_daily_round_trip_loses_money():
    b = _bars(n=500, seed=13, trend=0.0)
    for t in (0.005, 0.01, 0.02):
        assert N.simulate_limit(b, t, costs=A_SHARE).stats["mean_net"] < 0
