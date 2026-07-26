"""港股训练链路的切分正确性（v0.25 新增）。

用合成行情跑真训练（CPU，小模型），验证 v0.25 修掉的三个 IC 虚高来源：
按行切分 / 无 purge / 验证集既选模型又报成绩。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from eq.data import hk_market as hk  # noqa: E402
from eq.strategy.factors import validation as va  # noqa: E402


def _fake_ohlcv(seed: int, n: int = 400) -> pd.DataFrame:
    """造一只合成港股日线：几何随机游走 + 少量自相关。"""
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0003, 0.02, n)
    ret[1:] += 0.05 * ret[:-1]                      # 一点点动量，让模型有东西可学
    close = 20 * np.exp(np.cumsum(ret))
    idx = pd.bdate_range("2023-01-02", periods=n)
    noise = np.abs(rng.normal(0, 0.006, n))
    return pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.004, n)),
        "high": close * (1 + noise),
        "low": close * (1 - noise),
        "close": close,
        "volume": rng.integers(1e6, 9e6, n).astype(float),
    }, index=idx)


@pytest.fixture
def fake_hk(monkeypatch):
    """把港股下载换成内存合成数据：12 只票，日期区间完全相同。"""
    codes = [f"{i:05d}" for i in range(1, 13)]
    data = {c: _fake_ohlcv(seed=i) for i, c in enumerate(codes)}
    monkeypatch.setattr(hk, "download_hk_stock", lambda code, s, e: data.get(code, pd.DataFrame()))
    monkeypatch.setattr(hk, "list_hk_stocks", lambda limit=200: codes[:limit])
    return codes


def _tiny(**kw):
    """小到能在 CPU 上秒级跑完的训练配置。"""
    base = dict(hidden_size=8, num_layers=1, batch_size=512, max_steps=3,
                device="cpu", walk_forward=False, verbose=False, seed=0)
    base.update(kw)
    return base


# ---------- 核心回归：按时间切，不是按行切 ----------

def test_split_is_by_time_not_by_symbol(fake_hk, monkeypatch):
    """原 bug：样本按标的依次 append，``int(len(X)*0.8)`` 切出来的
    "验证集"是最后那批股票，两段时间完全重叠。

    这里拦截 purged_split，检查它拿到的索引确实是 (datetime, instrument) 面板，
    且切出来的三段在时间上严格不重叠、每段都覆盖全部标的。
    """
    seen = {}
    orig = va.purged_split

    def _spy(index, **kw):
        sp = orig(index, **kw)
        seen["index"] = index
        seen["split"] = sp
        return sp

    monkeypatch.setattr(hk, "purged_split", _spy, raising=False)
    monkeypatch.setattr("eq.strategy.factors.validation.purged_split", _spy)
    hk.train_hk(symbols=fake_hk, horizon=5, **_tiny())

    idx = seen["index"]
    sp = seen["split"]
    assert isinstance(idx, pd.MultiIndex), "切分依据必须是带日期的面板索引"
    dates = idx.get_level_values(0)
    syms = idx.get_level_values(1)

    # 时间上严格递进
    assert dates[sp.train].max() < dates[sp.valid].min()
    assert dates[sp.valid].max() < dates[sp.test].min()
    # 每段都应包含全部 12 只股票——按行切的话 train 会缺掉最后几只
    assert syms[sp.train].nunique() == 12
    assert syms[sp.valid].nunique() == 12
    assert syms[sp.test].nunique() == 12


def test_purge_gap_equals_horizon(fake_hk, monkeypatch):
    """标签用 T+horizon 的价格，所以 train 尾部必须 purge horizon 个交易日。"""
    seen = {}
    orig = va.purged_split
    monkeypatch.setattr("eq.strategy.factors.validation.purged_split",
                        lambda index, **kw: seen.setdefault("sp", orig(index, **kw)))
    horizon = 7
    hk.train_hk(symbols=fake_hk, horizon=horizon, **_tiny())
    assert seen["sp"].purged_days == horizon


def test_result_reports_test_ic_separately_from_valid(fake_hk):
    r = hk.train_hk(symbols=fake_hk, horizon=5, **_tiny())
    assert "ic" in r and "valid_ic" in r
    assert r["test_report"] is not None
    # 测试段报告必须是按日横截面口径，不是 pooled
    assert r["test_report"]["pooled"] is False
    assert r["test_report"]["n_days"] > 0


def test_no_test_segment_falls_back_to_valid(fake_hk):
    r = hk.train_hk(symbols=fake_hk, horizon=5, **_tiny(test_ratio=0.0))
    assert r["test_report"] is None
    assert r["ic"] == pytest.approx(r["valid_ic"])


def test_reports_trading_days_and_samples(fake_hk):
    r = hk.train_hk(symbols=fake_hk, horizon=5, **_tiny())
    assert r["symbols"] == 12
    assert r["trading_days"] > 300
    assert r["train_samples"] > 0


# ---------- 标签横截面归一化 ----------

def test_cs_normalized_label_is_centered_rank(fake_hk, monkeypatch):
    """开启后标签应变成每日 rank-0.5，即每日均值≈0、范围约 [-0.5, 0.5]。"""
    captured = {}
    from eq.strategy.factors.ml_workflow import _SimpleSeqModel

    def _spy_fit(self, xt, yt, xv, yv, early_stop=20):
        captured.setdefault("y", np.concatenate([np.asarray(yt), np.asarray(yv)]))
        return orig(self, xt, yt, xv, yv, early_stop=early_stop)

    orig = _SimpleSeqModel.fit
    monkeypatch.setattr(_SimpleSeqModel, "fit", _spy_fit)

    hk.train_hk(symbols=fake_hk, horizon=5, **_tiny(cs_normalize_label=True))
    y = captured["y"]
    assert y.min() > -0.55 and y.max() < 0.55
    assert abs(float(np.mean(y))) < 0.05

    captured.clear()
    hk.train_hk(symbols=fake_hk, horizon=5, **_tiny(cs_normalize_label=False))
    y_raw = captured["y"]
    # 原始 h 日收益尺度完全不同（远小于 0.5 的量级分布）
    assert float(np.std(y_raw)) < 0.3


# ---------- Walk-Forward ----------

def test_walk_forward_reports_cross_window_stats(fake_hk):
    r = hk.train_hk(symbols=fake_hk, horizon=5,
                    **_tiny(walk_forward=True, max_steps=2))
    assert r["wf_windows"] >= 1
    for k in ("wf_ic_mean", "wf_ic_std", "wf_ic_min", "wf_ic_max"):
        assert k in r and np.isfinite(r[k])
    assert r["wf_ic_min"] <= r["wf_ic_mean"] <= r["wf_ic_max"]


def test_walk_forward_skipped_when_history_too_short(monkeypatch):
    """只有 150 天历史时不该硬凑窗口，应跳过而不是报错。"""
    codes = ["00001", "00002", "00003"]
    data = {c: _fake_ohlcv(seed=i, n=200) for i, c in enumerate(codes)}
    monkeypatch.setattr(hk, "download_hk_stock", lambda code, s, e: data.get(code, pd.DataFrame()))
    r = hk.train_hk(symbols=codes, horizon=5, **_tiny(walk_forward=True, max_steps=2))
    assert "ic" in r  # 没炸就行


# ---------- 可复现性 ----------

def test_same_seed_gives_same_result(fake_hk):
    a = hk.train_hk(symbols=fake_hk, horizon=5, **_tiny(seed=7))
    b = hk.train_hk(symbols=fake_hk, horizon=5, **_tiny(seed=7))
    assert a["valid_ic"] == pytest.approx(b["valid_ic"], abs=1e-9)


def test_empty_symbols_raises(monkeypatch):
    monkeypatch.setattr(hk, "download_hk_stock", lambda code, s, e: pd.DataFrame())
    with pytest.raises(RuntimeError, match="无有效样本"):
        hk.train_hk(symbols=["00001"], horizon=5, **_tiny())
