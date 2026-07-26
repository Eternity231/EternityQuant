"""事件因子（v0.37）——重点是**不许穿越**。

外部数据有报告期和公告日两个日期，按报告期对齐的因子回测起来 IC 特别漂亮，
实盘一分钱赚不到。这套用例专门构造「报告期早、公告日晚」的数据来验证。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eq.strategy.factors import event as ev


@pytest.fixture
def days():
    return pd.bdate_range("2025-01-01", periods=60)


# ---------- 对齐：核心防线 ----------

def test_align_uses_publish_date_not_report_date(days):
    """报告期 1/10、公告日 2/20 的数据，2/19 必须还看不到。"""
    events = pd.DataFrame({
        "report_date": [pd.Timestamp("2025-01-10")],
        "publish_date": [pd.Timestamp("2025-02-20")],
        "holders": [50000.0],
    })
    s = ev.align_events(days, events, "holders")
    before = s[s.index < "2025-02-20"]
    after = s[s.index >= "2025-02-20"]
    assert before.isna().all(), "公告前必须无值——有值就是穿越"
    assert (after.dropna() == 50000.0).all()


def test_align_publish_day_itself_is_available(days):
    """公告日当天算可用：A 股公告多在盘后/盘前，当天收盘价已反映。"""
    events = pd.DataFrame({"publish_date": [pd.Timestamp("2025-02-03")], "v": [7.0]})
    s = ev.align_events(days, events, "v")
    assert s.loc[pd.Timestamp("2025-02-03")] == 7.0


def test_align_takes_the_latest_available(days):
    events = pd.DataFrame({
        "publish_date": pd.to_datetime(["2025-01-06", "2025-02-10", "2025-03-05"]),
        "v": [1.0, 2.0, 3.0],
    })
    s = ev.align_events(days, events, "v")
    assert s.loc[pd.Timestamp("2025-01-20")] == 1.0
    assert s.loc[pd.Timestamp("2025-02-20")] == 2.0
    assert s.loc[pd.Timestamp("2025-03-10")] == 3.0


def test_align_out_of_order_events_are_sorted(days):
    """事件表乱序也要对——merge_asof 未排序会直接报错或给错值。"""
    events = pd.DataFrame({
        "publish_date": pd.to_datetime(["2025-03-05", "2025-01-06", "2025-02-10"]),
        "v": [3.0, 1.0, 2.0],
    })
    s = ev.align_events(days, events, "v")
    assert s.loc[pd.Timestamp("2025-02-20")] == 2.0


def test_align_ffill_limit_expires_stale_data(days):
    """超过 ffill_limit 个交易日的陈旧数据置空，不能拿去年的当今天。"""
    events = pd.DataFrame({"publish_date": [pd.Timestamp("2025-01-02")], "v": [1.0]})
    s = ev.align_events(days, events, "v", ffill_limit=10)
    assert pd.isna(s.iloc[0]), "公告日(01-02)之前(01-01)本来就该是空"
    assert s.iloc[1:12].notna().all(), "公告日起 10 个交易日内有效"
    assert s.iloc[12:].isna().all(), "第 11 个交易日之后过期"


@pytest.mark.parametrize("bad", [None, pd.DataFrame()])
def test_align_empty_is_all_nan(days, bad):
    assert ev.align_events(days, bad, "v").isna().all()


def test_align_missing_columns_is_all_nan(days):
    assert ev.align_events(days, pd.DataFrame({"x": [1]}), "v").isna().all()


# ---------- 解禁 ----------

def test_days_to_event_counts_forward_only(days):
    s = ev.days_to_event(days, [pd.Timestamp("2025-02-14")])
    assert s.loc[pd.Timestamp("2025-02-04")] == 10
    assert s.loc[pd.Timestamp("2025-02-14")] == 0
    assert pd.isna(s.loc[pd.Timestamp("2025-02-17")]), "过去的解禁不该再计入"


def test_days_to_event_truncates_far_future(days):
    s = ev.days_to_event(days, [pd.Timestamp("2026-01-01")], max_days=30)
    assert s.isna().all(), "一年后的解禁太远，不该进因子"


def test_event_pressure_scales_with_size(days):
    small = pd.DataFrame({"date": [pd.Timestamp("2025-02-10")], "ratio": [1.0]})
    big = pd.DataFrame({"date": [pd.Timestamp("2025-02-10")], "ratio": [30.0]})
    d = pd.Timestamp("2025-02-03")
    assert ev.event_pressure(days, big).loc[d] > ev.event_pressure(days, small).loc[d]


def test_event_pressure_decays_with_distance(days):
    e = pd.DataFrame({"date": [pd.Timestamp("2025-03-17")], "ratio": [10.0]})
    p = ev.event_pressure(days, e)
    assert p.loc[pd.Timestamp("2025-03-14")] > p.loc[pd.Timestamp("2025-02-14")]


def test_event_pressure_accumulates(days):
    one = pd.DataFrame({"date": [pd.Timestamp("2025-02-10")], "ratio": [5.0]})
    three = pd.DataFrame({"date": pd.to_datetime(["2025-02-10", "2025-02-17", "2025-02-24"]),
                          "ratio": [5.0, 5.0, 5.0]})
    d = pd.Timestamp("2025-02-03")
    assert ev.event_pressure(days, three).loc[d] > ev.event_pressure(days, one).loc[d]


def test_event_pressure_ignores_past(days):
    past = pd.DataFrame({"date": [pd.Timestamp("2024-06-01")], "ratio": [50.0]})
    assert (ev.event_pressure(days, past) == 0).all()


# ---------- 股东户数 ----------

def test_holder_change_sign_convention(days):
    """户数下降（筹码集中）应给**正**分——全项目统一「大 = 看多」。"""
    events = pd.DataFrame({
        "publish_date": pd.to_datetime(["2025-01-06", "2025-03-06"]),
        "holders": [100000.0, 80000.0],       # 户数减少 20%
    })
    s = ev.holder_change(days, events)
    val = s.loc[pd.Timestamp("2025-03-20")]
    assert val > 0, f"户数下降该给正分，实得 {val}"
    assert val == pytest.approx(0.2, abs=1e-6)


def test_holder_change_rising_is_negative(days):
    events = pd.DataFrame({
        "publish_date": pd.to_datetime(["2025-01-06", "2025-03-06"]),
        "holders": [80000.0, 100000.0],
    })
    assert ev.holder_change(days, events).loc[pd.Timestamp("2025-03-20")] < 0


def test_holder_change_no_lookahead(days):
    """第二期公告前，因子不该已经知道户数要降。"""
    events = pd.DataFrame({
        "publish_date": pd.to_datetime(["2025-01-06", "2025-03-06"]),
        "holders": [100000.0, 80000.0],
    })
    s = ev.holder_change(days, events)
    assert s[s.index < "2025-03-06"].fillna(0).eq(0).all(), "公告前不能有变化信号"


def test_holder_change_single_period_is_nan(days):
    events = pd.DataFrame({"publish_date": [pd.Timestamp("2025-01-06")],
                           "holders": [100000.0]})
    assert ev.holder_change(days, events).isna().all()


# ---------- 余额动量 ----------

def test_balance_momentum_measures_change_not_level(days):
    """两只票余额差 1000 倍但涨幅相同时，因子值应该一样（不赌大小盘）。"""
    small = pd.DataFrame({"publish_date": days, "v": np.linspace(1e6, 1.2e6, len(days))})
    big = pd.DataFrame({"publish_date": days, "v": np.linspace(1e9, 1.2e9, len(days))})
    a = ev.balance_momentum(days, small, "v", window=20)
    b = ev.balance_momentum(days, big, "v", window=20)
    pd.testing.assert_series_equal(a.dropna(), b.dropna(), check_exact=False, rtol=1e-9)


def test_balance_momentum_sign(days):
    up = pd.DataFrame({"publish_date": days, "v": np.linspace(100, 200, len(days))})
    down = pd.DataFrame({"publish_date": days, "v": np.linspace(200, 100, len(days))})
    assert ev.balance_momentum(days, up, "v", window=20).dropna().iloc[-1] > 0
    assert ev.balance_momentum(days, down, "v", window=20).dropna().iloc[-1] < 0


def test_balance_momentum_too_short_is_nan():
    """样本比窗口还短时返回全 NaN，不能凑合算一个假动量。"""
    short_idx = pd.bdate_range("2025-01-01", periods=8)
    e = pd.DataFrame({"publish_date": short_idx, "v": np.arange(8.0)})
    assert ev.balance_momentum(short_idx, e, "v", window=20).isna().all()


def test_balance_momentum_flat_series_is_zero(days):
    """余额一直没动 = 零动量（不是 NaN）——低频数据前向填充后就是这种形态。"""
    e = pd.DataFrame({"publish_date": days[:5], "v": [1.0] * 5})
    out = ev.balance_momentum(days, e, "v", window=20).dropna()
    assert len(out) > 0 and (out == 0).all()


# ---------- 面板与验收 ----------

def test_build_panel_shape(days):
    panel = ev.build_panel({
        "600519.SH": pd.Series(1.0, index=days),
        "000001.SZ": pd.Series(2.0, index=days),
    })
    assert isinstance(panel.index, pd.MultiIndex)
    assert list(panel.index.names) == ["datetime", "instrument"]
    assert len(panel) == 2 * len(days)


def test_build_panel_empty():
    assert len(ev.build_panel({})) == 0


def test_forward_returns_has_no_lookahead():
    """标签是 close[t+h]/close[t]-1，末尾 h 根必须是 NaN。"""
    idx = pd.bdate_range("2025-01-01", periods=10)
    bars = {"A": pd.DataFrame({"close": np.arange(1.0, 11.0)}, index=idx)}
    r = ev.forward_returns(bars, horizon=3)
    assert r.isna().sum() == 3
    # t=0: close 1 → 4，收益 3.0
    assert r.iloc[0] == pytest.approx(3.0)


def test_evaluate_factor_runs_same_metric_as_ml():
    """事件因子和 ML 预测要能直接比，必须走同一套 evaluate。"""
    idx = pd.bdate_range("2025-01-01", periods=40)
    rng = np.random.default_rng(0)
    bars, factors = {}, {}
    for i in range(12):
        sym = f"S{i:02d}"
        c = pd.Series(100 + np.cumsum(rng.normal(size=len(idx))), index=idx)
        bars[sym] = pd.DataFrame({"close": c})
        factors[sym] = pd.Series(rng.normal(size=len(idx)), index=idx)
    rep = ev.evaluate_factor(ev.build_panel(factors), bars, horizon=5)
    assert "ic_mean" in rep and "icir" in rep
    assert abs(rep["ic_mean"]) < 0.5, "随机因子的 IC 该接近 0"


# ---------- 取数容错 ----------

def test_fetchers_return_empty_frame_on_failure(monkeypatch):
    """akshare 挂了要返回空表，不能把异常抛给上层策略。"""
    import sys
    import types

    mod = types.ModuleType("akshare")

    def _boom(*a, **k):
        raise RuntimeError("网络炸了")

    mod.stock_restricted_release_detail_em = _boom
    mod.stock_zh_a_gdhs_detail_em = _boom
    monkeypatch.setitem(sys.modules, "akshare", mod)

    assert ev.fetch_lockups("600519.SH").empty
    assert ev.fetch_holders("600519.SH").empty


def test_normalize_picks_publish_date_column():
    df = pd.DataFrame({
        "股东户数统计截止日": ["2025-09-30"],   # 报告期——不能用
        "公告日期": ["2025-10-24"],             # 公告日——要用这个
        "股东户数-本次": [12345],
    })
    out = ev._normalize(df, ("公告日期",), ("股东户数-本次",),
                        ("publish_date", "holders"))
    assert out.loc[0, "publish_date"] == pd.Timestamp("2025-10-24")
    assert out.loc[0, "holders"] == 12345
