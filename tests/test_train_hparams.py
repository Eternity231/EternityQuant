"""训练超参与早停口径（v0.36）。

两件事：
1. 早停必须用**每日横截面 Rank IC**（和 evaluate 验收同一把尺），
   不能用池化 Pearson IC——两者在截面数据上会挑出不同的 checkpoint。
2. lr/weight_decay 按优化器给默认值：Lion 的更新量是 sign(...)，
   每个坐标恒定走 ±lr，需要比 AdamW 小一个量级。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eq.strategy.factors.ml_workflow import _make_valid_scorer, resolve_opt_hparams


class _FakeTensor:
    """只需支撑 scorer 用到的 detach().cpu().numpy()，不拖 torch 进来。"""

    def __init__(self, arr):
        self._a = np.asarray(arr, dtype=float)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._a


def _panel(n_days: int, n_stocks: int):
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    idx = pd.MultiIndex.from_product([dates, [f"S{i:03d}" for i in range(n_stocks)]],
                                     names=["datetime", "instrument"])
    return idx


# ---------- 1. 早停口径 ----------

def test_scorer_uses_daily_cross_section_not_pooled():
    """构造一个两种口径**结论相反**的例子，钉死用的是每日截面。

    做法：每天给标签加一个很大的日内共同偏移（相当于当天的大盘涨跌），
    预测值 = 那个偏移本身。这样：
    - 池化口径：预测和标签高度正相关（都被大盘主导）→ IC 接近 +1
    - 每日截面：当天所有股票的预测完全相同，选不出任何票 → IC 恒为 0

    「能预测大盘、但当天选不出好票」的模型正是不该被早停选中的那种。
    """
    idx = _panel(n_days=20, n_stocks=30)
    rng = np.random.default_rng(0)
    market = pd.Series(idx.get_level_values(0).factorize()[0], index=idx).astype(float)
    label = market * 10 + rng.normal(size=len(idx))     # 大盘主导
    pred = market.to_numpy(dtype=float)                  # 只会预测大盘

    scorer = _make_valid_scorer(pd.DataFrame(index=idx), label)
    daily = scorer(_FakeTensor(pred))

    pooled = float(np.corrcoef(pred, label.to_numpy())[0, 1])
    assert pooled > 0.9, f"构造前提不成立，池化 IC 应该很高：{pooled}"
    assert abs(daily) < 0.05, f"每日截面口径应接近 0，实得 {daily}"


def test_scorer_rewards_real_cross_sectional_skill():
    """反过来：当天真能排出好票的预测，每日口径给高分。"""
    idx = _panel(n_days=15, n_stocks=40)
    rng = np.random.default_rng(1)
    label = pd.Series(rng.normal(size=len(idx)), index=idx)
    pred = label.to_numpy() + rng.normal(scale=0.2, size=len(idx))   # 带噪但同序

    scorer = _make_valid_scorer(pd.DataFrame(index=idx), label)
    assert scorer(_FakeTensor(pred)) > 0.8


def test_scorer_is_rank_based_not_pearson():
    """Rank IC 抗异常值：塞一个极端离群预测不该把分数打崩。"""
    idx = _panel(n_days=10, n_stocks=30)
    rng = np.random.default_rng(2)
    label = pd.Series(rng.normal(size=len(idx)), index=idx)
    pred = label.to_numpy().copy()
    pred[0] = 1e9                                    # 单点离群
    score = _make_valid_scorer(pd.DataFrame(index=idx), label)(_FakeTensor(pred))
    assert score > 0.9, f"Rank 口径下单个离群点不该拉垮 IC：{score}"


def test_scorer_falls_back_to_pooled_without_dates():
    """喂 numpy（无日期索引）时退回池化口径，不能报错。"""
    rng = np.random.default_rng(3)
    y = rng.normal(size=200)
    scorer = _make_valid_scorer(np.zeros((200, 5)), y)
    assert scorer(_FakeTensor(y)) > 0.9


def test_collapsed_prediction_scores_zero_and_loses():
    """输出塌缩成常数时打 0（不是 NaN）——NaN 会被 dropna 悄悄藏起来。

    关键是它必须**输给**任何有真实截面能力的 checkpoint，
    否则早停会停在一个恒定输出的废模型上。
    """
    idx = _panel(n_days=8, n_stocks=25)
    rng = np.random.default_rng(4)
    label = pd.Series(rng.normal(size=len(idx)), index=idx)
    scorer = _make_valid_scorer(pd.DataFrame(index=idx), label)

    collapsed = scorer(_FakeTensor(np.ones(len(idx))))
    skilled = scorer(_FakeTensor(label.to_numpy() + rng.normal(scale=0.3, size=len(idx))))
    assert collapsed == 0.0
    assert skilled > collapsed


def test_scorer_length_mismatch_is_rejected():
    idx = _panel(n_days=5, n_stocks=10)
    label = pd.Series(np.zeros(len(idx)), index=idx)
    assert _make_valid_scorer(pd.DataFrame(index=idx), label)(
        _FakeTensor(np.zeros(3))) == -float("inf")


# ---------- 2. 优化器相关默认超参 ----------

def test_lion_lr_is_an_order_smaller_than_adamw():
    """Lion 走 sign 更新，步长与梯度无关，lr 必须比 AdamW 小一个量级。"""
    lion_lr, lion_wd = resolve_opt_hparams("lion")
    adamw_lr, adamw_wd = resolve_opt_hparams("adamw")
    assert lion_lr <= adamw_lr / 3, f"lion {lion_lr} vs adamw {adamw_lr}"
    assert lion_wd >= adamw_wd * 3, "Lion 的 weight_decay 该同步放大"


def test_explicit_values_win_over_defaults():
    """显式传参必须照做——否则 --lr 又变成一个静默失效的开关。"""
    assert resolve_opt_hparams("lion", lr=0.05) == (0.05, 1e-4)
    assert resolve_opt_hparams("lion", weight_decay=0.0)[1] == 0.0
    assert resolve_opt_hparams("adamw", lr=0.01, weight_decay=0.02) == (0.01, 0.02)


def test_unknown_optimizer_falls_back_to_adamw_defaults():
    assert resolve_opt_hparams("sam") == resolve_opt_hparams("adamw")


@pytest.mark.parametrize("name", ["LION", "Lion", "lion"])
def test_optimizer_name_is_case_insensitive(name):
    assert resolve_opt_hparams(name) == resolve_opt_hparams("lion")


# ---------- 3. 训练完还得存得下来 ----------

def test_model_stays_picklable_after_fit():
    """fit 之后模型必须仍可 pickle——训练完是要存盘的。

    早停打分器是个闭包，一旦挂成 ``self._scorer`` 就会让整个模型不可 pickle，
    而且**训练全程都不报错**，只在最后 ``_pkl.dump`` 那一刻炸。
    """
    torch = pytest.importorskip("torch")
    import pickle

    from eq.strategy.factors.ml_workflow import MLPAlphaNet

    idx = _panel(n_days=6, n_stocks=20)
    rng = np.random.default_rng(7)
    x = pd.DataFrame(rng.normal(size=(len(idx), 8)), index=idx)
    y = pd.Series(rng.normal(size=len(idx)), index=idx)

    m = MLPAlphaNet(input_dim=8, hidden=8, max_steps=2, batch_size=32,
                    device="cpu", optimizer="adamw", seed=1)
    m.fit(x, y, x, y, early_stop=2)
    assert isinstance(pickle.loads(pickle.dumps(m)), MLPAlphaNet)
    assert torch is not None


def test_lion_default_lr_reaches_the_optimizer():
    """默认走 Lion 时，优化器里实际的 lr 必须是 Lion 那档，不是 AdamW 的 1e-3。"""
    pytest.importorskip("torch")
    from eq.strategy.factors.ml_workflow import MLPAlphaNet, resolve_opt_hparams

    m = MLPAlphaNet(input_dim=4, hidden=4, device="cpu", optimizer="lion")
    expected = resolve_opt_hparams("lion")[0]
    assert m.opt.param_groups[0]["lr"] == expected
    assert MLPAlphaNet(input_dim=4, hidden=4, device="cpu",
                       optimizer="adamw").opt.param_groups[0]["lr"] > expected


# ---------- 4. CLI 真把参数透传下去 ----------

def test_cli_train_exposes_lr_and_weight_decay():
    from typer.testing import CliRunner

    from eq.cli import app
    out = CliRunner().invoke(app, ["ml", "train", "--help"]).output
    assert "--lr" in out and "--weight-decay" in out


def test_both_torch_call_sites_forward_the_knobs():
    """两个 ``wf_train_torch(`` 调用点都必须透传 optimizer/lr/weight_decay。

    源码级检查，不需要装 qlib——这类"忘了透传某个参数"的 BUG（--dropout、
    --optimizer 都栽过）只在真跑训练时才暴露，而真跑训练要 qlib + 数据 + GPU，
    单测环境里没有。用文本检查换取「任何环境都能守住」。
    """
    import re
    from pathlib import Path

    import eq.cli as cli_mod
    src = Path(cli_mod.__file__).read_text(encoding="utf-8")
    calls = re.findall(r"wf_train_torch\((.*?)\n            \)", src, re.S)
    assert len(calls) == 2, f"预期 2 个 torch 训练调用点，实得 {len(calls)}"
    for i, body in enumerate(calls):
        for knob in ("optimizer=", "lr=", "weight_decay="):
            assert knob in body, f"第 {i + 1} 个调用点没透传 {knob}"


def test_cli_passes_optimizer_for_plain_torch_algos(monkeypatch):
    """端到端确认 gru 分支真把 optimizer/lr 传下去（需要 qlib，装了才跑）。

    ``wf_train_torch`` 是 cli 里函数内导入的别名，所以要打在源头模块上。
    """
    pytest.importorskip("qlib")
    from typer.testing import CliRunner

    from eq.cli import app
    from eq.strategy.factors import ml_workflow

    seen: dict = {}

    def _fake(**kw):
        seen.update(kw)
        return {"model_id": "x", "metrics": {"ic": 0.0}, "model_path": "p"}

    monkeypatch.setattr(ml_workflow, "train_torch", _fake)
    CliRunner().invoke(app, [
        "ml", "train", "gru", "--optimizer", "adamw", "--lr", "0.005",
        "--weight-decay", "0.02",
    ])
    assert seen.get("optimizer") == "adamw", "gru 分支必须透传 optimizer"
    assert seen.get("lr") == 0.005
    assert seen.get("weight_decay") == 0.02


def test_cli_lr_negative_means_use_default(monkeypatch):
    """不传 --lr 时要传 None 下去（让 resolve_opt_hparams 按优化器决定），
    不能把哨兵值 -1 原样透传成学习率。"""
    pytest.importorskip("qlib")
    from typer.testing import CliRunner

    from eq.cli import app
    from eq.strategy.factors import ml_workflow

    seen: dict = {}
    monkeypatch.setattr(ml_workflow, "train_torch",
                        lambda **kw: (seen.update(kw), {"model_id": "x", "metrics": {}})[1])
    CliRunner().invoke(app, ["ml", "train", "gru"])
    assert seen.get("lr") is None and seen.get("weight_decay") is None
