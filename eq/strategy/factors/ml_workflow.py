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
    """qlib init + torch DLL 预热（Windows + cu132 坑：先 torch.cuda.init 再 qlib.init）。"""
    import torch  # noqa: F401
    if torch.cuda.is_available():
        torch.cuda.init()  # 预热 DLL，避免 c10.dll 延迟加载失败
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
