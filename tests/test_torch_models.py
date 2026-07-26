"""自写 torch 模型的构造与前向（v0.25 新增）。

覆盖 v0.25 修的两个「参数看起来生效、实际没生效」的问题：
``--dropout`` 对 MLP 完全无效、``_reshape`` 静默丢特征。
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from eq.strategy.factors.ml_workflow import _SimpleMLP, _SimpleSeqModel  # noqa: E402


def _dropouts(module: nn.Module) -> list[float]:
    return [m.p for m in module.modules() if isinstance(m, nn.Dropout)]


# ---------- dropout 打通 ----------

def test_mlp_dropout_is_applied():
    """原来 _SimpleMLP 里 nn.Dropout(0.05) 硬编码、构造函数根本不收 dropout，
    CLI 的 --dropout 对 MLP 路径完全无效。"""
    m = _SimpleMLP(input_dim=20, hidden=(16, 8), device="cpu", dropout=0.42)
    ps = _dropouts(m.net)
    assert ps, "MLP 应该含 Dropout 层"
    assert all(p == pytest.approx(0.42) for p in ps), f"实际 {ps}"


def test_mlp_dropout_zero_is_respected():
    m = _SimpleMLP(input_dim=20, hidden=(16,), device="cpu", dropout=0.0)
    assert all(p == 0.0 for p in _dropouts(m.net))


def test_mlp_default_dropout_is_not_the_old_hardcoded_005():
    assert _SimpleMLP(input_dim=20, hidden=(8,), device="cpu").net is not None
    assert _dropouts(_SimpleMLP(input_dim=20, hidden=(8,), device="cpu").net)[0] > 0.05


def test_seq_model_dropout_reaches_rnn():
    m = _SimpleSeqModel(input_dim=60, seq_len=6, input_size=10,
                        hidden_size=8, num_layers=2, device="cpu", dropout=0.37)
    assert m.net.dropout == pytest.approx(0.37)


def test_seq_model_single_layer_forces_zero_dropout():
    """PyTorch RNN 单层时 dropout 无意义（会 warn），应置 0。"""
    m = _SimpleSeqModel(input_dim=60, seq_len=6, input_size=10,
                        hidden_size=8, num_layers=1, device="cpu", dropout=0.5)
    assert m.net.dropout == 0


# ---------- _reshape 不再丢特征 ----------

def test_reshape_no_longer_drops_features():
    """原来 cut = 6*26 = 156 < 158，x[:, :cut] 静默丢掉最后 2 维，
    而 docstring 却写着"保证 158 维全保留"。"""
    m = _SimpleSeqModel(input_dim=158, seq_len=6, input_size=26,
                        hidden_size=8, num_layers=1, device="cpu")
    assert m.seq_len * m.input_size >= 158, "重塑后的容量必须覆盖全部输入维度"
    assert m.input_size == 27  # ceil(158/6)


def test_reshape_preserves_all_values_and_pads_with_zero():
    m = _SimpleSeqModel(input_dim=158, seq_len=6, input_size=26,
                        hidden_size=8, num_layers=1, device="cpu")
    x = torch.arange(158, dtype=torch.float32).unsqueeze(0)
    out = m._reshape(x)
    assert out.shape == (1, 6, 27)
    flat = out.reshape(-1)
    # 前 158 个值原样保留
    assert torch.allclose(flat[:158], x[0])
    # 其余是补的零
    assert torch.all(flat[158:] == 0)


def test_reshape_exact_fit_needs_no_padding():
    m = _SimpleSeqModel(input_dim=60, seq_len=6, input_size=10,
                        hidden_size=8, num_layers=1, device="cpu")
    assert m.input_size == 10
    out = m._reshape(torch.arange(60, dtype=torch.float32).unsqueeze(0))
    assert out.shape == (1, 6, 10)
    assert torch.allclose(out.reshape(-1), torch.arange(60, dtype=torch.float32))


def test_reshape_warns_and_bumps_when_capacity_short(capsys):
    _SimpleSeqModel(input_dim=158, seq_len=6, input_size=26,
                    hidden_size=8, num_layers=1, device="cpu")
    assert "丢" in capsys.readouterr().out


# ---------- 种子 ----------

def test_seed_makes_init_reproducible():
    a = _SimpleMLP(input_dim=20, hidden=(16,), device="cpu", seed=99)
    b = _SimpleMLP(input_dim=20, hidden=(16,), device="cpu", seed=99)
    for pa, pb in zip(a.net.parameters(), b.net.parameters(), strict=True):
        assert torch.allclose(pa, pb)


def test_different_seeds_give_different_init():
    a = _SimpleMLP(input_dim=20, hidden=(16,), device="cpu", seed=1)
    b = _SimpleMLP(input_dim=20, hidden=(16,), device="cpu", seed=2)
    assert not all(torch.allclose(pa, pb)
                   for pa, pb in zip(a.net.parameters(), b.net.parameters(), strict=True))


# ---------- 端到端 fit/predict ----------

def _xy(n=400, d=60, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)).astype("float32")
    y = (x[:, :5].sum(axis=1) * 0.3 + rng.normal(0, 0.5, n)).astype("float32")
    return x, y


def test_mlp_fit_predict_shapes():
    x, y = _xy()
    m = _SimpleMLP(input_dim=60, hidden=(16,), max_steps=5, batch_size=128,
                   device="cpu", dropout=0.2, seed=0)
    m.fit(x[:300], y[:300], x[300:], y[300:], early_stop=5)
    assert m.predict(x[300:]).shape == (100,)


def test_seq_fit_predict_shapes():
    x, y = _xy()
    m = _SimpleSeqModel(input_dim=60, seq_len=6, input_size=10, hidden_size=8,
                        num_layers=1, max_steps=5, batch_size=128,
                        device="cpu", dropout=0.2, seed=0)
    m.fit(x[:300], y[:300], x[300:], y[300:], early_stop=5)
    assert m.predict(x[300:]).shape == (100,)


def test_models_learn_a_real_signal():
    """信号足够强时 best_score 应明显为正——保证训练循环本身没坏。"""
    x, y = _xy(n=1200, seed=3)
    m = _SimpleMLP(input_dim=60, hidden=(32,), max_steps=40, batch_size=256,
                   device="cpu", dropout=0.1, lr=3e-3, seed=0)
    m.fit(x[:900], y[:900], x[900:], y[900:], early_stop=40)
    assert m.best_score > 0.2, f"best IC={m.best_score}"


def test_predict_handles_single_sample():
    """BatchNorm1d 需要 batch≥2，单样本预测不能崩。"""
    x, y = _xy(n=200)
    m = _SimpleMLP(input_dim=60, hidden=(8,), max_steps=2, batch_size=64,
                   device="cpu", seed=0)
    m.fit(x[:150], y[:150], x[150:], y[150:], early_stop=2)
    assert m.predict(x[:1]).shape == (1,)
