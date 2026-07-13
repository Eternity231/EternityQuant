"""qlib workflow 真集成：Alpha158 特征 + LightGBM 训练 + 批量预测。

替代 v0.1 的 ml predict 手工录入，对接真实训练 pipeline。
qlib 数据集截至 2020-09-25，训练区间用 2015-01-01~2020-08-31，验证 2020-09。
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pandas as pd

from eq.db import DEFAULT_HOME, execute_write
from eq.strategy.factors.ml import activate, register_model

_QLIB_MODELS_DIR = DEFAULT_HOME / "ml_models"


def _ensure_dir() -> Path:
    _QLIB_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return _QLIB_MODELS_DIR


def _qlib_init() -> None:
    """qlib init + torch DLL 预热（Windows + cu132 坑：先 torch.cuda.init 再 qlib.init）。

    还修 qlib 0.9.7 的 ReduceLROnPlateau 版本判断 bug：
    qlib 用 `str(torch.__version__).split('+')[0] <= '2.6.0'` 做字符串比较，
    对 torch 2.13.0 误判（'2.13.0' <= '2.6.0' 字典序为真），走错老分支传 verbose=True。
    monkey patch 绕开：让 ReduceLROnPlateau 接受并忽略 verbose 参数。
    """
    import torch  # noqa: F401
    if torch.cuda.is_available():
        torch.cuda.init()  # 预热 DLL，避免 c10.dll 延迟加载失败

    # monkey patch ReduceLROnPlateau 接受 verbose 参数（qlib 0.9.7 版本判断 bug 绕开）
    _orig_reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau.__init__

    def _patched_reduce_lr(self, *args, **kwargs):
        kwargs.pop("verbose", None)  # 新版 torch 不再支持 verbose，忽略
        return _orig_reduce_lr(self, *args, **kwargs)

    torch.optim.lr_scheduler.ReduceLROnPlateau.__init__ = _patched_reduce_lr

    import qlib
    from qlib.config import REG_CN
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)


def train(
    universe: str = "csi300",
    train_start: str = "2015-01-01",
    train_end: str = "2020-08-31",
    valid_start: str = "2020-09-01",
    valid_end: str = "2020-09-25",
    horizon: int = 5,
    algo: str = "lightgbm",
    device: str = "cpu",
    name: str | None = None,
) -> dict[str, Any]:
    """走 qlib 标准 pipeline 训练一个 LightGBM 模型。

    Args:
        device: "cpu" | "gpu" | "cuda"（cuda 需编译时开 USE_CUDA=1，本机不可用）
    Returns:
        {"model_id": str, "metrics": dict, "model_path": str}
    """
    _qlib_init()
    from qlib.data import D

    from qlib.contrib.data.handler import Alpha158
    from qlib.contrib.model import LGBModel
    from qlib.utils import init_instance_by_config

    # 1. 标的池（csi300 默认；qlib 本地数据支持）
    try:
        instruments = D.instruments(market=universe)
        inst_list = D.list_instruments(instruments=instruments, start_time=train_start, end_time=valid_end)
        inst_list = list(inst_list) if not isinstance(inst_list, list) else inst_list
        if not inst_list:
            raise ValueError(f"universe {universe} 无数据")
    except Exception as e:
        raise ValueError(f"qlib instruments 拉取失败：{e}") from e

    # 2. Alpha158 handler
    # infer_processors 必须为空或只含 inference-safe processor，learn_processors 才加归一化
    learn_procs = [{"class": "DropnaLabel"}, {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}}]
    label_expr = [f"Ref($close, -{horizon}) / Ref($close, -1) - 1"]
    handler = Alpha158(
        instruments=universe,
        start_time=train_start,
        end_time=valid_end,
        fit_start_time=train_start,
        fit_end_time=train_end,
        infer_processors=[],
        learn_processors=learn_procs,
        label=label_expr,
    )

    # 3. 数据集切片
    from qlib.data.dataset import DatasetH
    segments = {
        "train": (train_start, train_end),
        "valid": (valid_start, valid_end),
    }
    dataset = DatasetH(handler=handler, segments=segments)

    # 4. 训练 LightGBM（device 透传：cpu|gpu|cuda）
    if algo != "lightgbm":
        raise NotImplementedError(f"algo {algo} 待集成，第一版只支持 lightgbm")
    model = LGBModel(
        loss="mse", num_leaves=64, learning_rate=0.05, n_estimators=200, colsample_bytree=0.9,
        device=device,
    )
    model.fit(dataset)

    # 5. 评估（predict 直接接 dataset + segment="valid"）
    valid_pred = model.predict(dataset, segment="valid")
    valid_data = dataset.prepare("valid", col_set="label")
    valid_label = valid_data
    # IC 指标
    valid_pred_df = valid_pred if isinstance(valid_pred, pd.DataFrame) else pd.DataFrame(valid_pred)
    valid_label_df = valid_label if isinstance(valid_label, pd.DataFrame) else pd.DataFrame(valid_label)
    aligned = valid_pred_df.align(valid_label_df, axis=0, join="inner")
    pred_series = aligned[0].iloc[:, 0] if not aligned[0].empty else pd.Series(dtype=float)
    label_series = aligned[1].iloc[:, 0] if not aligned[1].empty else pd.Series(dtype=float)
    if pred_series.empty or label_series.empty:
        ic = 0.0
    else:
        cov = pred_series.cov(label_series)
        std_p = pred_series.std()
        std_l = label_series.std()
        ic = cov / (std_p * std_l) if std_p > 0 and std_l > 0 else 0.0

    # 6. 模型存盘（pickle 直存，绕开 qlib dump API 复杂性）
    import pickle as _pkl
    model_path = _ensure_dir() / f"lgbm_{universe}_{horizon}d.pkl"
    with open(model_path, "wb") as f:
        _pkl.dump(model, f)

    # 7. 登记 ml_models 表
    model_name = name or f"{universe}_{algo}_h{horizon}_{dt.date.today().strftime('%Y%m%d')}"
    features = ["Alpha158(158 个 qlib 标准特征)"]
    model_id = register_model(
        name=model_name,
        universe=universe,
        features=features,
        algo=algo,
        horizon=horizon,
        train_period=f"{train_start}~{train_end}",
        valid_period=f"{valid_start}~{valid_end}",
        metrics={"ic": ic, "algo": algo, "horizon": horizon, "device": device},
        model_path=str(model_path),
        notes="qlib workflow 真集成训练",
    )
    return {"model_id": model_id, "metrics": {"ic": ic}, "model_path": str(model_path)}


# ---------- qlib PyTorch 模型（走 CUDA，3060 主场） ----------

_TORCH_ALGOS = {"alstm", "gru", "lstm", "mlp"}


def _build_torch_model(algo: str, device: str):
    """按 algo 名造一个 qlib PyTorch 模型实例。device='cuda' 时 GPU=0。

    注意：qlib DNNModelPytorch/ALSTM/GRU 在 torch 2.13 + Alpha158 默认配置下 loss 全 nan
    （BatchNorm1d 遇全 NaN 列梯度爆），所以这只返回 qlib 原生模型供尝试，主路径走自写 MLP。
    """
    from qlib.contrib.model import ALSTM, GRU, LSTM, DNNModelPytorch

    gpu_id = 0 if device == "cuda" else -1  # GPU=-1 走 CPU
    common = dict(
        d_feat=6, hidden_size=64, num_layers=2, dropout=0.0,
        n_epochs=50, lr=0.001, batch_size=2000, early_stop=10,
        loss="mse", optimizer="adam", GPU=gpu_id,
    )
    if algo == "alstm":
        return ALSTM(**common)
    if algo == "gru":
        return GRU(**common)
    if algo == "lstm":
        return LSTM(**common)
    if algo == "mlp":
        # 走自写 MLP 路径，不返 qlib DNNModelPytorch
        return None
    raise NotImplementedError(f"algo {algo} 待集成，可选：{sorted(_TORCH_ALGOS)}")


# ---------- 自写最简 MLP（走 torch.cuda，绕开 qlib DNNModelPytorch nan 坑） ----------

class _SimpleMLP:
    """最简 MLP：158 -> 256 -> 1，走 BatchNorm1d + Adam，支持 CUDA。

    qlib DNNModelPytorch 在 torch 2.13 + Alpha158 默认配置下 loss 全 nan（BatchNorm1d 坑），
    自写此绕开，只取 qlib handler 的 feature 和 label 做数据，训练用原生 torch。
    """

    def __init__(self, input_dim: int = 158, hidden: int = 256, lr: float = 1e-3, max_steps: int = 300, batch_size: int = 2000, device: str = "cuda"):
        import torch
        import torch.nn as nn
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.05),
            nn.Linear(hidden, 1),
        ).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.best_score = -float("inf")
        self.best_state = None
        self.best_step = 0

    def fit(self, x_train, y_train, x_valid, y_valid, early_stop: int = 30):
        import torch
        import numpy as np
        from torch.utils.data import DataLoader, TensorDataset

        def _to_tensor(df):
            if hasattr(df, "values"):
                return torch.from_numpy(df.values).float()
            return torch.from_numpy(np.asarray(df)).float()

        xt = _to_tensor(x_train).to(self.device)
        yt = _to_tensor(y_train).squeeze(-1).to(self.device)
        xv = _to_tensor(x_valid).to(self.device)
        yv = _to_tensor(y_valid).squeeze(-1).to(self.device)

        stop = 0
        for step in range(self.max_steps):
            self.net.train()
            idx = torch.randperm(len(xt), device=self.device)
            for i in range(0, len(idx), self.batch_size):
                b = idx[i:i + self.batch_size]
                if len(b) < 2:  # BatchNorm1d 需 >1 样本
                    break
                pred = self.net(xt[b]).squeeze(-1)
                loss = self.loss_fn(pred, yt[b])
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            # eval
            self.net.eval()
            with torch.no_grad():
                vp = self.net(xv).squeeze(-1) if len(xv) >= 2 else torch.zeros(1, device=self.device)
                vl = self.loss_fn(vp, yv).item() if len(xv) >= 2 else float("inf")
            # IC 作 score（越高越好）
            if len(xv) >= 2 and vp.std().item() > 0 and yv.std().item() > 0:
                score = torch.cov(torch.stack([vp, yv]))[0, 1].item() / (vp.std().item() * yv.std().item())
            else:
                score = -float("inf")
            if score > self.best_score:
                self.best_score = score
                self.best_step = step
                self.best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                stop = 0
            else:
                stop += 1
                if stop >= early_stop:
                    break
        if self.best_state is not None:
            self.net.load_state_dict(self.best_state)

    def predict(self, x):
        import torch
        import numpy as np
        self.net.eval()
        with torch.no_grad():
            xt = torch.from_numpy(np.asarray(x if not hasattr(x, "values") else x.values)).float().to(self.device)
            if len(xt) < 2:
                xt = xt.unsqueeze(0).repeat(2, 1)  # BatchNorm1d 需 >=2
                pred = self.net(xt).squeeze(-1)[0:1]
            else:
                pred = self.net(xt).squeeze(-1)
            return pred.cpu().numpy()


def train_torch(
    universe: str = "csi300",
    train_start: str = "2015-01-01",
    train_end: str = "2020-08-31",
    valid_start: str = "2020-09-01",
    valid_end: str = "2020-09-25",
    horizon: int = 5,
    algo: str = "gru",
    device: str = "cuda",  # 默认 cuda（真 CUDA，3060 主场）
    name: str | None = None,
) -> dict[str, Any]:
    """走 qlib PyTorch pipeline 训练 ALSTM/GRU/LSTM/MLP，用 CUDA。

    Args:
        algo: alstm | gru | lstm | mlp（mlp 走自写 _SimpleMLP 绕开 qlib nan 坑）
        device: cuda | cpu
    Returns:
        {"model_id": str, "metrics": dict, "model_path": str}
    """
    _qlib_init()

    # Alpha158 handler（feature 158 维）
    # infer_processors 用默认（ProcessInf + ZScoreNorm + Fillna），跳过会让 feature 含 NaN/Inf 喂给 BatchNorm1d 梯度爆
    from qlib.contrib.data.handler import Alpha158, _DEFAULT_INFER_PROCESSORS
    label_expr = [f"Ref($close, -{horizon}) / Ref($close, -1) - 1"]
    handler = Alpha158(
        instruments=universe,
        start_time=train_start,
        end_time=valid_end,
        fit_start_time=train_start,
        fit_end_time=train_end,
        infer_processors=_DEFAULT_INFER_PROCESSORS,
        learn_processors=[{"class": "DropnaLabel"}, {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}}],
        label=label_expr,
    )

    from qlib.data.dataset import DatasetH
    segments = {"train": (train_start, train_end), "valid": (valid_start, valid_end)}
    dataset = DatasetH(handler=handler, segments=segments)

    if algo == "mlp":
        # 自写 MLP 路径：从 dataset 取 feature 和 label，用 torch.cuda 训练
        train_data = dataset.prepare("train", col_set=["feature", "label"])
        valid_data = dataset.prepare("valid", col_set=["feature", "label"])
        x_train, y_train = train_data["feature"], train_data["label"]
        x_valid, y_valid = valid_data["feature"], valid_data["label"]
        if hasattr(y_train, "values"):
            y_train = y_train.squeeze() if y_train.ndim > 1 else y_train
        if hasattr(y_valid, "values"):
            y_valid = y_valid.squeeze() if y_valid.ndim > 1 else y_valid
        model = _SimpleMLP(input_dim=158, hidden=256, lr=1e-3, max_steps=300, batch_size=2000, device=device)
        model.fit(x_train, y_train, x_valid, y_valid, early_stop=30)
        ic = float(model.best_score)
        epochs = model.best_step + 1

        # 存盘（pickle 整个 _SimpleMLP 实例，含 net state_dict）
        import pickle as _pkl
        model_path = _ensure_dir() / f"torch_{algo}_{universe}_{horizon}d.pkl"
        with open(model_path, "wb") as f:
            _pkl.dump(model, f)

        model_id = register_model(
            name=name or f"{universe}_{algo}_h{horizon}_{dt.date.today().strftime('%Y%m%d')}",
            universe=universe,
            features=["Alpha158(158 个 qlib 标准特征)"],
            algo=algo,
            horizon=horizon,
            train_period=f"{train_start}~{train_end}",
            valid_period=f"{valid_start}~{valid_end}",
            metrics={"ic": ic, "algo": algo, "horizon": horizon, "device": device, "epochs": epochs},
            model_path=str(model_path),
            notes=f"自写 _SimpleMLP 真集成训练（{device}），绕开 qlib DNNModelPytorch nan 坑",
        )
        return {"model_id": model_id, "metrics": {"ic": ic, "epochs": epochs}, "model_path": str(model_path)}

    # qlib 原生 ALSTM/GRU/LSTM 路径
    model = _build_torch_model(algo, device)
    if model is None:
        raise NotImplementedError(f"algo {algo} 构造失败")
    evals_result: dict = {}
    model.fit(dataset, evals_result=evals_result)
    valid_scores = evals_result.get("valid", [])
    ic = float(valid_scores[-1]) if valid_scores else 0.0

    import pickle as _pkl
    model_path = _ensure_dir() / f"torch_{algo}_{universe}_{horizon}d.pkl"
    with open(model_path, "wb") as f:
        _pkl.dump(model, f)

    model_id = register_model(
        name=name or f"{universe}_{algo}_h{horizon}_{dt.date.today().strftime('%Y%m%d')}",
        universe=universe,
        features=["Alpha158(158 个 qlib 标准特征)"],
        algo=algo,
        horizon=horizon,
        train_period=f"{train_start}~{train_end}",
        valid_period=f"{valid_start}~{valid_end}",
        metrics={"ic": ic, "algo": algo, "horizon": horizon, "device": device, "epochs": len(valid_scores)},
        model_path=str(model_path),
        notes=f"qlib PyTorch {algo} 真集成训练（{device}）",
    )
    return {"model_id": model_id, "metrics": {"ic": ic, "epochs": len(valid_scores)}, "model_path": str(model_path)}


def predict_batch(
    model_id: str,
    universe: str = "csi300",
    predict_date: str | None = None,
    top_n: int = 50,
) -> pd.DataFrame:
    """用指定模型批量预测全 universe，写入 ml_predictions 表，返回前 N 名。

    predict_date 缺省用 qlib 数据末日 + 1 日（受数据集截至 2020-09 限制）。
    """
    _qlib_init()
    from qlib.data import D

    from qlib.contrib.data.handler import Alpha158
    from qlib.contrib.model import LGBModel

    # 拉模型元数据
    from eq.db import execute
    meta_rows = execute("SELECT universe, horizon, model_path FROM ml_models WHERE id = ?", (model_id,))
    if not meta_rows:
        raise KeyError(f"模型 {model_id} 不存在")
    meta = {k: meta_rows[0][k] for k in meta_rows[0].keys()}
    horizon = int(meta["horizon"])
    model_path = meta["model_path"]
    universe = meta["universe"] or universe

    # 拉末日数据作为 predict_date
    if predict_date is None:
        # qlib 数据末日 2020-09-25，predict 用 2020-09-25
        predict_date = "2020-09-25"

    # 重新构造 handler 取特征（predict 不需要真 label，用占位表达式避免 horizon 未来数据问题）
    # label 用 Ref($close,-1)/Ref($close,-1)-1 恒为 0 的占位，handler 能跑通，predict 只用 feature
    handler = Alpha158(
        instruments=universe,
        start_time=predict_date,
        end_time=predict_date,
        fit_start_time="2015-01-01",
        fit_end_time=predict_date,
        infer_processors=[],
        label=["Ref($close, -1) / Ref($close, -1) - 1"],
    )
    from qlib.data.dataset import DatasetH
    dataset = DatasetH(handler=handler, segments={"test": (predict_date, predict_date)})

    # 加载模型并预测（pickle 直加载绕开 qlib LGBModel.load 触发的 torch DLL 链）
    import pickle as _pkl
    with open(model_path, "rb") as f:
        model = _pkl.load(f)
    pred = model.predict(dataset, segment="test")
    # pred 是 pd.Series，index 是 MultiIndex(datetime, instrument)，values 是预测分数
    if pred is None or (isinstance(pred, pd.Series) and pred.empty):
        return pd.DataFrame(columns=["symbol", "score"])
    # 转 DataFrame，取 predict_date 当日
    pred_df = pred.to_frame("score") if isinstance(pred, pd.Series) else pred
    if isinstance(pred_df.index, pd.MultiIndex):
        # level 0=datetime, level 1=instrument（qlib 0.9.7 顺序）
        if predict_date in pred_df.index.get_level_values(0):
            pred_df = pred_df.xs(predict_date, level=0)
        else:
            pred_df = pred_df.groupby(level=1).last()  # 取最近一日
        pred_df = pred_df.reset_index()
        # instrument 列名可能是 "instrument" 或 index 名
        inst_col = "instrument" if "instrument" in pred_df.columns else pred_df.columns[0]
        pred_df = pred_df.rename(columns={inst_col: "symbol"})
    else:
        pred_df = pred_df.reset_index()
    pred_df = pred_df[["symbol", "score"]].sort_values("score", ascending=False).head(top_n).reset_index(drop=True)

    # 转 EternityQuant 符号格式：SH600519 → 600519.SH
    def _to_eq_code(s: str) -> str:
        if s.startswith("SH"):
            return s[2:] + ".SH"
        if s.startswith("SZ"):
            return s[2:] + ".SZ"
        return s
    pred_df["symbol"] = pred_df["symbol"].map(_to_eq_code)

    # 写入 ml_predictions 表
    target_date = dt.date.fromisoformat(predict_date)
    for _, row in pred_df.iterrows():
        execute_write(
            "INSERT INTO ml_predictions (model_id, symbol, date, score) VALUES (?, ?, ?, ?)",
            (model_id, row["symbol"], target_date.isoformat(), float(row["score"])),
        )

    return pred_df
