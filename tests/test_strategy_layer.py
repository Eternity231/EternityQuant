"""策略层：新增因子 / 信号 / 组合 / 风控（v0.27）。纯逻辑，无网络无 GPU。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eq.strategy import BUY, HOLD, SELL
from eq.strategy.factors import technical as T
from eq.strategy.signals import base as B
from eq.strategy.signals import breakout as BO
from eq.strategy.signals import composite as C
from eq.strategy.signals import reversal as R


def _bars(n=250, seed=0, trend=0.0, vol=1.5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(trend, vol, n))
    close = np.maximum(close, 5.0)
    noise = np.abs(rng.normal(0, 0.8, n))
    return pd.DataFrame({
        "open": close + rng.normal(0, 0.3, n),
        "high": close + noise,
        "low": close - noise,
        "close": close,
        "volume": rng.integers(1_000_000, 9_000_000, n).astype(float),
    }, index=pd.bdate_range("2024-01-01", periods=n))


@pytest.fixture
def bars():
    return _bars()


# ====================== 新因子 ======================

@pytest.mark.parametrize("fn", [
    T.true_range, T.atr, T.natr, T.realized_vol, T.cci, T.williams_r,
    T.roc, T.momentum, T.zscore, T.bollinger_bandwidth, T.trend_strength,
])
def test_series_factors_shape_and_finite(fn, bars):
    s = fn(bars)
    assert isinstance(s, pd.Series) and len(s) == len(bars)
    assert np.isfinite(s.dropna()).all(), f"{fn.__name__} 产生了 inf"


@pytest.mark.parametrize(("fn", "cols"), [
    (T.donchian, {"upper", "lower", "mid"}),
    (T.keltner, {"upper", "mid", "lower"}),
    (T.supertrend, {"line", "trend"}),
    (T.stochastic, {"K", "D"}),
])
def test_frame_factors_columns(fn, cols, bars):
    out = fn(bars)
    assert isinstance(out, pd.DataFrame) and set(out.columns) == cols
    assert len(out) == len(bars)


def test_atr_is_positive_and_tracks_volatility():
    calm, wild = _bars(seed=1, vol=0.3), _bars(seed=1, vol=3.0)
    assert (T.atr(calm).dropna() > 0).all()
    assert T.atr(wild).mean() > T.atr(calm).mean() * 2


def test_true_range_covers_gaps():
    """跳空时 TR 应大于当日高低差。"""
    df = pd.DataFrame({
        "open": [10, 20.0], "high": [11, 21.0], "low": [9, 19.0], "close": [10, 20.0],
        "volume": [1.0, 1.0],
    }, index=pd.bdate_range("2024-01-01", periods=2))
    tr = T.true_range(df)
    assert tr.iloc[1] == pytest.approx(11.0)   # |21-10| 跳空，而非 21-19=2


def test_donchian_shifts_to_avoid_lookahead():
    """通道上下轨必须 shift(1)，否则今日高点在窗口里，"突破"永不成立。"""
    bars = _bars(seed=2)
    d = T.donchian(bars, 20)
    manual_no_shift = bars["high"].rolling(20).max()
    assert not d["upper"].equals(manual_no_shift)
    assert d["upper"].iloc[25] == pytest.approx(manual_no_shift.iloc[24])


def test_supertrend_trend_is_bipolar_and_persists():
    st = T.supertrend(_bars(seed=3))
    assert set(np.unique(st["trend"])) <= {-1.0, 1.0}
    # 状态机应该有持续性，不该每根都翻
    flips = (st["trend"].diff() != 0).sum()
    assert flips < len(st) / 4


def test_zscore_is_centered():
    z = T.zscore(_bars(seed=4), 20).dropna()
    assert abs(z.mean()) < 1.0 and z.std() > 0.3


# ====================== 分数契约 ======================

def test_score_to_signal_only_fires_on_cross():
    score = pd.Series([0.0, 0.5, 0.6, 0.7, 0.0, -0.5, -0.6])
    sig = B.score_to_signal(score, 0.3, -0.3)
    assert list(sig) == [HOLD, BUY, HOLD, HOLD, HOLD, SELL, HOLD]


def test_score_to_signal_continuous_mode():
    score = pd.Series([0.0, 0.5, 0.6, -0.5])
    sig = B.score_to_signal(score, 0.3, -0.3, on_cross_only=False)
    assert list(sig) == [HOLD, BUY, BUY, SELL]


def test_signal_score_roundtrip():
    sig = pd.Series([BUY, HOLD, SELL, HOLD])
    assert list(B.signal_to_score(sig)) == [1.0, 0.0, -1.0, 0.0]


def test_clip_score_bounds_and_nan():
    out = B.clip_score(pd.Series([-5.0, 0.2, 5.0, np.nan]))
    assert list(out) == [-1.0, 0.2, 1.0, 0.0]


def test_normalize_score_is_bounded():
    s = B.normalize_score(pd.Series(np.random.default_rng(0).normal(0, 10, 300)), 60)
    assert s.between(-1, 1).all()


# ====================== 新信号 ======================

_SIGNALS = [
    BO.donchian_breakout, BO.keltner_breakout, BO.supertrend_follow,
    BO.volatility_breakout, BO.breakout_composite,
    R.zscore_reversion, R.kdj_cross, R.cci_reversal, R.reversion_composite,
]


@pytest.mark.parametrize("fn", _SIGNALS)
def test_signals_return_valid_ternary(fn, bars):
    sig = fn(bars)
    assert len(sig) == len(bars)
    assert set(sig.unique()) <= {BUY, SELL, HOLD}


@pytest.mark.parametrize("fn", _SIGNALS)
def test_signals_survive_short_and_flat_input(fn):
    """指标未成形的短序列、以及完全无波动的序列，都不能崩。"""
    for df in (_bars(n=8, seed=5),
               pd.DataFrame({"open": [10.0] * 60, "high": [10.0] * 60,
                             "low": [10.0] * 60, "close": [10.0] * 60,
                             "volume": [1e6] * 60},
                            index=pd.bdate_range("2024-01-01", periods=60))):
        sig = fn(df)
        assert len(sig) == len(df)


@pytest.mark.parametrize("fn", [BO.turtle_score, BO.atr_channel_score, R.rsi_score])
def test_score_signals_are_bounded(fn, bars):
    s = fn(bars)
    assert s.between(-1, 1).all(), f"{fn.__name__} 越界"


def test_donchian_breakout_fires_on_new_high():
    """构造一段单调上涨，突破策略必须给出买入信号。"""
    n = 80
    close = np.linspace(10, 30, n)
    df = pd.DataFrame({"open": close, "high": close + 0.1, "low": close - 0.1,
                       "close": close, "volume": np.full(n, 1e6)},
                      index=pd.bdate_range("2024-01-01", periods=n))
    assert (BO.donchian_breakout(df, 20, 10) == BUY).any()


def test_zscore_reversion_buys_the_dip():
    """构造一段横盘后急跌，均值回归必须在低点买入。"""
    close = np.concatenate([np.full(60, 100.0) + np.random.default_rng(0).normal(0, 1, 60),
                            np.linspace(100, 80, 15)])
    n = len(close)
    df = pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5,
                       "close": close, "volume": np.full(n, 1e6)},
                      index=pd.bdate_range("2024-01-01", periods=n))
    assert (R.zscore_reversion(df, 20, entry=1.5) == BUY).any()


# ====================== 组合 ======================

def test_vote_equal_weight(bars):
    always_buy = lambda d: pd.Series(BUY, index=d.index)      # noqa: E731
    always_sell = lambda d: pd.Series(SELL, index=d.index)    # noqa: E731
    s = C.vote(bars, [always_buy, always_buy, always_sell], as_score=True)
    assert s.iloc[-1] == pytest.approx(1 / 3)


def test_vote_weights_are_normalised(bars):
    buy = lambda d: pd.Series(BUY, index=d.index)             # noqa: E731
    sell = lambda d: pd.Series(SELL, index=d.index)           # noqa: E731
    # 权重 3:1 → (3-1)/4 = 0.5，传绝对值即可，函数内部归一化
    s = C.vote(bars, [buy, sell], weights=[3, 1], as_score=True)
    assert s.iloc[-1] == pytest.approx(0.5)


def test_vote_min_agree_suppresses_lone_voice(bars):
    buy = lambda d: pd.Series(BUY, index=d.index)             # noqa: E731
    hold = lambda d: pd.Series(HOLD, index=d.index)           # noqa: E731
    lone = C.vote(bars, [buy, hold, hold], min_agree=2, as_score=True)
    assert (lone == 0).all(), "只有一票同意时应压成中性"
    pair = C.vote(bars, [buy, buy, hold], min_agree=2, as_score=True)
    assert (pair > 0).all()


def test_vote_rejects_bad_input(bars):
    with pytest.raises(ValueError, match="至少要传"):
        C.vote(bars, [])
    with pytest.raises(ValueError, match="权重个数"):
        C.vote(bars, [lambda d: pd.Series(HOLD, index=d.index)], weights=[1, 2])
    with pytest.raises(ValueError, match="不能为 0"):
        C.vote(bars, [lambda d: pd.Series(HOLD, index=d.index)], weights=[0])


def test_vote_mixes_ternary_and_score_strategies(bars):
    tern = lambda d: pd.Series(BUY, index=d.index)                       # noqa: E731
    score = lambda d: pd.Series(0.5, index=d.index)                      # noqa: E731
    s = C.vote(bars, [tern, score], as_score=True)
    assert s.iloc[-1] == pytest.approx(0.75)


def test_make_vote_produces_usable_signal_func(bars):
    fn = C.make_vote([BO.donchian_breakout, R.rsi_reversal])
    out = fn(bars)
    assert set(out.unique()) <= {BUY, SELL, HOLD}
    assert "vote(" in fn.__name__


def test_market_regime_labels(bars):
    reg = C.market_regime(bars)
    assert set(reg.unique()) <= {"trend", "range", "transition"}
    assert len(reg) == len(bars)


def test_market_regime_detects_strong_trend():
    n = 150
    close = np.linspace(10, 60, n)          # 强单边
    df = pd.DataFrame({"open": close, "high": close + 0.1, "low": close - 0.1,
                       "close": close, "volume": np.full(n, 1e6)},
                      index=pd.bdate_range("2024-01-01", periods=n))
    assert (C.market_regime(df) == "trend").sum() > n * 0.5


def test_regime_adaptive_switches_by_regime(bars):
    t = lambda d: pd.Series(BUY, index=d.index)      # noqa: E731
    r = lambda d: pd.Series(SELL, index=d.index)     # noqa: E731
    reg = C.market_regime(bars)
    out = C.regime_adaptive(bars, t, r, transition="hold")
    assert (out[reg == "trend"] == BUY).all()
    assert (out[reg == "range"] == SELL).all()


def test_regime_adaptive_flat_transition_exits(bars):
    t = lambda d: pd.Series(BUY, index=d.index)      # noqa: E731
    r = lambda d: pd.Series(BUY, index=d.index)      # noqa: E731
    out = C.regime_adaptive(bars, t, r, transition="flat")
    reg = C.market_regime(bars)
    entering = (reg == "transition") & (reg.shift(1) != "transition")
    if entering.any():
        assert (out[entering] == SELL).all(), "进入过渡区应清仓"


def test_filters_return_boolean_masks(bars):
    for f in (C.volume_filter, C.volatility_filter, C.squeeze_filter, C.not_overextended):
        m = f(bars)
        assert len(m) == len(bars)
        assert m.dropna().isin([True, False]).all(), f.__name__


def test_filtered_blocks_buys_but_not_sells(bars):
    always = lambda d: pd.Series(BUY, index=d.index)          # noqa: E731
    never_ok = lambda d: pd.Series(False, index=d.index)      # noqa: E731
    out = C.filtered(bars, always, [never_ok])
    assert (out == HOLD).all()

    sells = lambda d: pd.Series(SELL, index=d.index)          # noqa: E731
    out2 = C.filtered(bars, sells, [never_ok], apply_to="buy")
    assert (out2 == SELL).all(), "离场信号不该被过滤掉——该跑时要跑得掉"
    out3 = C.filtered(bars, sells, [never_ok], apply_to="both")
    assert (out3 == HOLD).all()


def test_filtered_without_filters_is_passthrough(bars):
    sig = BO.donchian_breakout(bars)
    assert C.filtered(bars, BO.donchian_breakout, []).equals(sig)
