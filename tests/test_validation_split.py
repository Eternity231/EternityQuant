"""训练集切分与可复现性（v0.25 新增，纯逻辑无网络无 GPU）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eq.strategy.factors import validation as va


def _panel_index(n_days=200, n_stocks=10, start="2024-01-01"):
    dates = pd.bdate_range(start, periods=n_days)
    return pd.MultiIndex.from_product(
        [dates, [f"S{i:02d}" for i in range(n_stocks)]],
        names=["datetime", "instrument"],
    )


# ---------- set_seed ----------

def test_set_seed_makes_numpy_reproducible():
    va.set_seed(123)
    a = np.random.rand(5)
    va.set_seed(123)
    assert np.allclose(a, np.random.rand(5))


def test_set_seed_returns_seed():
    assert va.set_seed(7) == 7


def test_set_seed_makes_torch_reproducible():
    torch = pytest.importorskip("torch")
    va.set_seed(2024)
    a = torch.randn(8)
    va.set_seed(2024)
    assert torch.allclose(a, torch.randn(8))


# ---------- purged_split ----------

def test_purged_split_is_by_date_not_by_row():
    """核心 bug 回归：样本按标的展开后，取前 80% 行 = 取前 80% 只股票。

    正确切分必须让 train 的所有日期都早于 valid 的所有日期。
    """
    idx = _panel_index(n_days=200, n_stocks=10)
    sp = va.purged_split(idx, valid_ratio=0.15, test_ratio=0.15, embargo_days=5)
    dates = idx.get_level_values(0)
    assert dates[sp.train].max() < dates[sp.valid].min()
    assert dates[sp.valid].max() < dates[sp.test].min()
    # 每段都应包含全部 10 只股票（不是按标的切的）
    for mask in (sp.train, sp.valid, sp.test):
        assert idx.get_level_values(1)[mask].nunique() == 10


def test_purged_split_enforces_embargo_gap():
    """train 末日与 valid 首日之间必须隔开 embargo_days 个交易日。"""
    idx = _panel_index(n_days=200, n_stocks=4)
    embargo = 5
    sp = va.purged_split(idx, valid_ratio=0.2, test_ratio=0.2, embargo_days=embargo)
    dates = pd.DatetimeIndex(sorted(pd.unique(idx.get_level_values(0))))
    train_end = pd.Timestamp(sp.bounds["train"][1])
    valid_start = pd.Timestamp(sp.bounds["valid"][0])
    gap = dates.get_loc(valid_start) - dates.get_loc(train_end)
    assert gap == embargo + 1, f"实际间隔 {gap} 个交易日，应为 {embargo + 1}"


def test_purged_split_zero_embargo_is_contiguous():
    idx = _panel_index(n_days=100, n_stocks=3)
    sp = va.purged_split(idx, valid_ratio=0.2, test_ratio=0.2, embargo_days=0)
    dates = pd.DatetimeIndex(sorted(pd.unique(idx.get_level_values(0))))
    gap = dates.get_loc(pd.Timestamp(sp.bounds["valid"][0])) - dates.get_loc(pd.Timestamp(sp.bounds["train"][1]))
    assert gap == 1
    assert sp.purged_days == 0


def test_purged_split_masks_are_disjoint_and_shorter_than_full():
    idx = _panel_index(n_days=200, n_stocks=5)
    sp = va.purged_split(idx, valid_ratio=0.15, test_ratio=0.15, embargo_days=5)
    assert not (sp.train & sp.valid).any()
    assert not (sp.valid & sp.test).any()
    assert not (sp.train & sp.test).any()
    # purge 掉的样本不属于任何一段
    assert sp.train.sum() + sp.valid.sum() + sp.test.sum() < len(idx)


def test_purged_split_without_test():
    idx = _panel_index(n_days=100, n_stocks=3)
    sp = va.purged_split(idx, valid_ratio=0.2, embargo_days=3, with_test=False)
    assert sp.test is None
    assert "test" not in sp.bounds
    assert set(sp.sizes()) == {"train", "valid"}


def test_purged_split_accepts_plain_datetime_index():
    idx = pd.bdate_range("2024-01-01", periods=100)
    sp = va.purged_split(idx, valid_ratio=0.2, test_ratio=0.2, embargo_days=2)
    assert sp.train.sum() > 0 and sp.valid.sum() > 0 and sp.test.sum() > 0


def test_purged_split_accepts_raw_date_array():
    dates = np.repeat(pd.bdate_range("2024-01-01", periods=60).to_numpy(), 4)
    sp = va.purged_split(dates, valid_ratio=0.2, test_ratio=0.2, embargo_days=2)
    assert sp.train.sum() > 0


def test_purged_split_raises_when_too_few_dates():
    with pytest.raises(ValueError, match="至少需要 3 个"):
        va.purged_split(pd.bdate_range("2024-01-01", periods=2))


def test_purged_split_raises_when_embargo_eats_train():
    idx = _panel_index(n_days=12, n_stocks=2)
    with pytest.raises(ValueError, match="训练集为空"):
        va.purged_split(idx, valid_ratio=0.2, test_ratio=0.2, embargo_days=50)


def test_purged_split_rejects_bad_ratios():
    idx = _panel_index(n_days=100, n_stocks=2)
    with pytest.raises(ValueError):
        va.purged_split(idx, valid_ratio=0.6, test_ratio=0.5)
    with pytest.raises(ValueError):
        va.purged_split(idx, valid_ratio=0.0)


def test_split_describe_and_sizes():
    idx = _panel_index(n_days=150, n_stocks=4)
    sp = va.purged_split(idx, embargo_days=5)
    s = sp.sizes()
    assert s["train"] > s["valid"] and s["test"] > 0
    d = sp.describe()
    assert "train=" in d and "purge=5" in d


def test_qlib_segments_conversion():
    idx = _panel_index(n_days=150, n_stocks=2)
    seg = va.split_bounds_to_qlib_segments(va.purged_split(idx, embargo_days=5))
    assert set(seg) == {"train", "valid", "test"}
    for lo, hi in seg.values():
        assert len(lo) == 10 and lo.count("-") == 2   # YYYY-MM-DD
        assert lo <= hi


# ---------- walk_forward_windows ----------

def test_walk_forward_windows_are_time_ordered_and_purged():
    idx = _panel_index(n_days=600, n_stocks=5)
    wins = va.walk_forward_windows(idx, n_splits=4, valid_days=60, embargo_days=5)
    assert len(wins) == 4
    dates = pd.DatetimeIndex(sorted(pd.unique(idx.get_level_values(0))))
    prev_valid_start = None
    for w in wins:
        assert w.test is None
        tr_end = pd.Timestamp(w.bounds["train"][1])
        va_start = pd.Timestamp(w.bounds["valid"][0])
        assert tr_end < va_start
        gap = dates.get_loc(va_start) - dates.get_loc(tr_end)
        assert gap == 5 + 1
        # 时间正序
        if prev_valid_start is not None:
            assert va_start > prev_valid_start
        prev_valid_start = va_start


def test_walk_forward_expanding_grows_train():
    idx = _panel_index(n_days=600, n_stocks=3)
    wins = va.walk_forward_windows(idx, n_splits=4, valid_days=60, embargo_days=5, expanding=True)
    sizes = [w.train.sum() for w in wins]
    assert sizes == sorted(sizes), "expanding 模式下训练集应逐窗增大"


def test_walk_forward_rolling_keeps_train_size_constant():
    idx = _panel_index(n_days=600, n_stocks=3)
    wins = va.walk_forward_windows(idx, n_splits=4, valid_days=60, embargo_days=5,
                                   expanding=False, min_train_days=200)
    sizes = {w.train.sum() for w in wins}
    assert len(sizes) == 1, "滑窗模式下训练集大小应恒定"


def test_walk_forward_returns_empty_when_data_too_short():
    idx = _panel_index(n_days=50, n_stocks=3)
    assert va.walk_forward_windows(idx, n_splits=5, valid_days=60, min_train_days=120) == []


def test_walk_forward_stops_early_rather_than_producing_bad_windows():
    """数据只够 2 个窗口时不该硬凑 5 个。"""
    idx = _panel_index(n_days=320, n_stocks=3)
    wins = va.walk_forward_windows(idx, n_splits=5, valid_days=60,
                                   embargo_days=5, min_train_days=120)
    assert 0 < len(wins) < 5
    for w in wins:
        assert w.train.sum() > 0 and w.valid.sum() > 0


# ---------- qlib segments 构造（_build_segments，无 qlib 时走自然日兜底） ----------

def test_build_segments_creates_three_purged_segments():
    from eq.strategy.factors.ml_workflow import _build_segments

    seg = _build_segments("2015-01-01", "2020-08-31", "2020-09-01", "2021-06-30",
                          test_ratio=0.2, embargo_days=5)
    assert set(seg) == {"train", "valid", "test"}
    # train 尾部被 purge：不再等于原 train_end
    assert seg["train"][1] < "2020-08-31"
    # 时间严格递进
    assert seg["train"][1] < seg["valid"][0]
    assert seg["valid"][1] < seg["test"][0]
    assert seg["test"][1] == "2021-06-30"


def test_build_segments_zero_embargo_keeps_original_train_end():
    from eq.strategy.factors.ml_workflow import _build_segments

    seg = _build_segments("2015-01-01", "2020-08-31", "2020-09-01", "2021-06-30",
                          test_ratio=0.2, embargo_days=0)
    assert seg["train"] == ("2015-01-01", "2020-08-31")


def test_build_segments_no_test_when_ratio_zero():
    from eq.strategy.factors.ml_workflow import _build_segments

    seg = _build_segments("2015-01-01", "2020-08-31", "2020-09-01", "2021-06-30",
                          test_ratio=0.0, embargo_days=5)
    assert set(seg) == {"train", "valid"}
    assert seg["valid"] == ("2020-09-01", "2021-06-30")


def test_build_segments_skips_test_when_valid_window_too_short(capsys):
    """原默认验证区间只有 2020-09-01~09-25，再切 test 两段都没样本。"""
    from eq.strategy.factors.ml_workflow import _build_segments

    seg = _build_segments("2015-01-01", "2020-08-31", "2020-09-01", "2020-09-05",
                          test_ratio=0.2, embargo_days=5)
    assert set(seg) == {"train", "valid"}
    assert "太短" in capsys.readouterr().out


def test_lgb_params_are_regularized():
    """低信噪比金融数据必须强正则——原配置连 L1/L2 都没有。"""
    from eq.strategy.factors.ml_workflow import LGB_PARAMS

    assert LGB_PARAMS["lambda_l1"] > 100
    assert LGB_PARAMS["lambda_l2"] > 100
    assert 0 < LGB_PARAMS["subsample"] < 1
    assert 0 < LGB_PARAMS["colsample_bytree"] < 1
    # v0.38：轮数/早停从 params 里挪出来了——lightgbm 原生 API 把它们当
    # train() 的参数而不是超参字典的键
    from eq.strategy.factors.gbdt import DEFAULT_EARLY_STOP, DEFAULT_ROUNDS
    assert DEFAULT_EARLY_STOP > 0 and DEFAULT_ROUNDS > DEFAULT_EARLY_STOP
