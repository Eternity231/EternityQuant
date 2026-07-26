"""qlib workflow 真集成：Alpha158 特征 + LightGBM 训练 + 批量预测。

替代 v0.1 的 ml predict 手工录入，对接真实训练 pipeline。
qlib 数据集截至 2020-09-25，训练区间用 2015-01-01~2020-08-31，验证 2020-09。
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np  # _patched_corr_load 里用到（此前漏 import，补丁一触发就 NameError）
import pandas as pd
import torch  # _LionOpt 继承 torch.optim.Optimizer 须顶层可见

from eq.db import DEFAULT_HOME, execute_write
from eq.strategy.factors.ml import activate, register_model

_QLIB_MODELS_DIR = DEFAULT_HOME / "ml_models"


def _ensure_dir() -> Path:
    _QLIB_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return _QLIB_MODELS_DIR


# ---------- 自写 Lion 优化器（EvoLved Sign Momentum，不依赖外部 lion-pytorch 包） ----------
#
# Lion 由谷歌通过 AutoML 符号搜索「进化」出，彻底舍弃二阶矩估计，仅追一阶动量，
# 并强制用符号函数决定更新方向。相比 AdamW：
#   显存减半（无二阶矩状态）+ 天然正则化（符号操作抗噪），对低信噪比数据
#   （如股票收益率）噪声不敏感，倾向顺应大趋势忽略微小抖动。
# 参考: Chen et al. 2023 "Symbolic Discovery of Optimization Rules for Deep Neural Networks"
class _LionOpt(torch.optim.Optimizer):
    """最简 Lion 优化器：lr·sign(w·b + (1-w)·m)·lr_proj 更新，动量 m 指数衰减。

    仅追一阶动量 m（与参数同形），无二阶矩 v → 显存比 AdamW 省一半。
    weight_decay 内嵌进 update（不依赖外部调度），与 AdamW 实现一致。

    继承 torch.optim.Optimizer：兼容 ReduceLROnPlateau / warmup 等 scheduler
    的 isinstance(opt, Optimizer) 检查，param_groups/state_dict 由父类管。
    """

    def __init__(self, params, lr=1e-3, weight_decay=1e-6, momentum_decay=0.95):
        import torch
        # 父类要 defaults + param_groups；self.m 动量单独管（不进 optimizer.state）
        defaults = {"lr": lr, "weight_decay": weight_decay, "momentum_decay": momentum_decay}
        super().__init__(params, defaults)
        self.lr = lr
        self.wd = weight_decay
        self.m_decay = momentum_decay
        # 一阶动量 m 与各 param 同形，按 param_groups 组织便于 step 时按组遍历
        self.m = [[torch.zeros_like(p) for p in g["params"]] for g in self.param_groups]

    def zero_grad(self, set_to_none: bool = True):
        # 父类已有，但 Optimizer.zero_grad 走 self.param_groups，行为一致；显式覆盖保留接口
        super().zero_grad(set_to_none=set_to_none)

    def _get_lr(self):
        # warmup / scheduler 会改 param_groups[0]["lr"]，读最新值
        return float(self.param_groups[0]["lr"])

    @torch.no_grad()
    def step(self, closure=None):
        lr = self._get_lr()
        wd = float(self.param_groups[0].get("weight_decay", self.wd))
        gi = 0  # param_groups 与 self.m 的同步索引
        for group in self.param_groups:
            m_decay = float(group.get("momentum_decay", self.m_decay))
            for j, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                g = p.grad
                m = self.m[gi][j]
                # Lion 核心: u = sign(β·m + (1-β)·g);  m = β·m + (1-β)·g
                m.mul_(m_decay).add_(g, alpha=1 - m_decay)
                update = m.sign()  # 符号方向
                # weight_decay 内嵌: p -= lr·(update + wd·p)
                if wd > 0:
                    p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)
            gi += 1

    def state_dict(self):
        # 父类 state 存 defaults；额外存 m 动量
        sd = super().state_dict()
        sd["m"] = [[m.clone() for m in group_m] for group_m in self.m]
        return sd

    def load_state_dict(self, sd):
        super().load_state_dict(sd)
        if "m" in sd:
            for gi, group_m in enumerate(sd["m"]):
                for j, m_new in enumerate(group_m):
                    self.m[gi][j].copy_(m_new)



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
    from pathlib import Path as _P
    from qlib.config import REG_CN
    from eq.data.paths import QLIB_CN_DATA_DIR, ensure_data_dirs
    ensure_data_dirs()
    # qlib 0.9.7 要求 provider_uri 是 dict（key=freq），传字符串会报
    # "does not contain data for day"——内部按 freq 查 key 找不到。
    _qlib_uri = {"day": str(QLIB_CN_DATA_DIR)}
    qlib.init(provider_uri=_qlib_uri, region=REG_CN)

    # monkey patch: 修 qlib 0.9.7 issue #1949
    #
    # 根因:
    #   LocalDatasetProvider.features() 调用 DatasetD.dataset() 时把 disk_cache
    #   作为第6个位置参数传入。DatasetD 是 Wrapper，通过 __getattr__ 委托给
    #   LocalDatasetProvider.dataset()，其签名为:
    #     def dataset(self, instruments, fields, start_time, end_time, freq, inst_processors=[])
    #   所以 disk_cache 被当作 inst_processors 位置参数，同时
    #   inst_processors=inst_processors 又作为关键字参数传入——双重复值报错。
    #
    # 修复: monkey patch LocalDatasetProvider.dataset() 接受 *args, 遇
    #   inst_processors 双重复值时丢弃位置参数，保留关键字参数。
    from qlib.data.data import LocalDatasetProvider
    _orig_dataset = LocalDatasetProvider.dataset

    def _patched_dataset(self, *args, **kwargs):
        if len(args) > 5 and "inst_processors" in kwargs:
            # disk_cache 被当作第6个位置参数（inst_processors），
            # 同时 inst_processors 又以关键字传入——丢弃位置参数
            args = args[:5]  # 只保留 instruments, fields, start_time, end_time, freq
        return _orig_dataset(self, *args, **kwargs)

    LocalDatasetProvider.dataset = _patched_dataset

    # monkey patch: 修 qlib Corr._load_internal 中 series_right 为空 (0,) 的 bug
    # Corr._load_internal 在 super()._load_internal() 之后重新加载
    # feature_left/feature_right，但第二次加载时 series_right 返回空数组，
    # 导致 np.isclose 广播失败 (632,) vs (0,)。
    from qlib.data.ops import Corr as _Corr
    _orig_corr_load = _Corr._load_internal

    def _patched_corr_load(self, instrument, start_index, end_index, *args):
        res = _orig_corr_load(self, instrument, start_index, end_index, *args)
        series_left = self.feature_left.load(instrument, start_index, end_index, *args)
        series_right = self.feature_right.load(instrument, start_index, end_index, *args)
        if len(series_right) == 0:
            return res  # 无法过滤，直接返回原始结果
        res.loc[
            np.isclose(series_left.rolling(self.N, min_periods=1).std(), 0, atol=2e-05)
            | np.isclose(series_right.rolling(self.N, min_periods=1).std(), 0, atol=2e-05)
        ] = np.nan
        return res

    _Corr._load_internal = _patched_corr_load


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
    # --- v0.25 训练策略参数 ---
    test_ratio: float = 0.2,
    embargo_days: int | None = None,
    seed: int = 42,
    feature_set: str = "Alpha158",
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """走 qlib 标准 pipeline 训练一个 LightGBM 模型。

    Args:
        device: "cpu" | "gpu" | "cuda"（cuda 需编译时开 USE_CUDA=1，本机不可用）
        test_ratio/embargo_days/seed: 见 :func:`train_torch`
        params: 覆盖默认 LightGBM 超参（见 :data:`LGB_PARAMS`）
    Returns:
        {"model_id": str, "metrics": dict, "model_path": str}
    """
    _qlib_init()
    from qlib.data import D

    from qlib.contrib.model import LGBModel

    from eq.strategy.factors.validation import set_seed

    set_seed(seed)
    embargo = horizon if embargo_days is None else int(embargo_days)

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
    # 权威处理器链（qlib benchmarks/LightGBM 同款配置，参考
    # examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml）：
    #   infer_processors (特征):
    #     ProcessInf → RobustZScoreNorm(clip_outlier=True) → Fillna
    #     - ProcessInf: 处理 Inf（替换为列均值），避免 BatchNorm1d 梯度爆
    #     - RobustZScoreNorm: MAD（中位绝对偏差）抗异常 z-score，clip_outlier 截断 3σ 外
    #     - Fillna: NaN 填 0
    #   learn_processors (标签):
    #     DropnaLabel → CSZScoreNorm (横截面 z-score，去截面均值/方差影响)
    infer_procs = infer_processors()
    learn_procs = learn_processors(rank_norm=False)   # 树模型保留幅度信息
    label_expr = [f"Ref($close, -{horizon}) / Ref($close, -1) - 1"]
    handler = _resolve_handler(feature_set)(
        instruments=universe,
        start_time=train_start,
        end_time=valid_end,
        fit_start_time=train_start,
        fit_end_time=train_end,
        infer_processors=infer_procs,
        learn_processors=learn_procs,
        label=label_expr,
    )

    # 3. 数据集切片：train / valid / test，段间 purge horizon 天
    from qlib.data.dataset import DatasetH
    segments = _build_segments(train_start, train_end, valid_start, valid_end,
                               test_ratio=test_ratio, embargo_days=embargo)
    print("  [切分] " + "  ".join(f"{k}={v[0]}~{v[1]}" for k, v in segments.items())
          + f"  (purge={embargo}日, seed={seed})", flush=True)
    dataset = DatasetH(handler=handler, segments=segments)

    # 4. 训练 LightGBM（device 透传：cpu|gpu|cuda）
    if algo != "lightgbm":
        raise NotImplementedError(f"algo {algo} 待集成，第一版只支持 lightgbm")
    lgb_kwargs = dict(LGB_PARAMS)
    lgb_kwargs.update(params or {})
    lgb_kwargs["device"] = device
    lgb_kwargs.setdefault("seed", seed)
    model = LGBModel(**lgb_kwargs)
    # 有 valid 段时交给 LightGBM 自己早停（qlib LGBModel 内部用 valid 做 early stopping）
    model.fit(dataset)

    # 5. 评估：valid（选择集，偏乐观）+ test（独立，真成绩）
    valid_report = _eval_segment(model, dataset, "valid", segments)
    test_report = _eval_on_test(model, dataset, segments)
    ic = (test_report or valid_report or {}).get("ic_mean", 0.0)

    # 6. 模型存盘（pickle 直存，绕开 qlib dump API 复杂性）
    import pickle as _pkl
    model_path = _ensure_dir() / f"lgbm_{universe}_{horizon}d.pkl"
    with open(model_path, "wb") as f:
        _pkl.dump(model, f)

    # 7. 登记 ml_models 表
    model_name = name or f"{universe}_{algo}_h{horizon}_{dt.date.today().strftime('%Y%m%d')}"
    model_id = register_model(
        name=model_name,
        universe=universe,
        features=[feature_set],
        algo=algo,
        horizon=horizon,
        train_period=f"{segments['train'][0]}~{segments['train'][1]}",
        valid_period=f"{segments['valid'][0]}~{segments['valid'][1]}",
        metrics={"ic": ic, "valid_ic": (valid_report or {}).get("ic_mean", 0.0),
                 "algo": algo, "horizon": horizon, "device": device, "seed": seed,
                 "embargo_days": embargo, "feature_set": feature_set,
                 **{f"test_{k}": v for k, v in (test_report or {}).items()
                    if k != "ic_series" and not isinstance(v, list)}},
        model_path=str(model_path),
        notes=f"qlib LightGBM（{feature_set}，官方调优超参，purge={embargo}日）",
    )
    return {"model_id": model_id,
            "metrics": {"ic": ic, "valid_ic": (valid_report or {}).get("ic_mean", 0.0),
                        "test": test_report},
            "model_path": str(model_path)}


# ---------- 训练策略公共件（v0.25） ----------

_FEATURE_SETS = {"alpha158": "Alpha158", "alpha360": "Alpha360"}

# LightGBM 超参：直接采用 qlib 官方 benchmark 调优结果
# （examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml）。
#
# 原来的配置是 num_leaves=64, lr=0.05, n_estimators=200, colsample=0.9，
# **没有任何 L1/L2 正则、没有行采样**。股票日频截面数据信噪比极低
# （单因子 IC 通常 0.02~0.05），这种配置几乎必然过拟合训练段：
# 官方调出来的 lambda_l1=205 / lambda_l2=580 是常规 GBDT 任务的几百倍，
# 正是为了压住这种低信噪比数据上的过拟合。
LGB_PARAMS: dict[str, Any] = {
    "loss": "mse",
    "learning_rate": 0.0421,
    "colsample_bytree": 0.8879,
    "subsample": 0.8789,
    "lambda_l1": 205.6999,
    "lambda_l2": 580.9769,
    "max_depth": 8,
    "num_leaves": 210,
    "feature_fraction": 0.8879,
    "bagging_fraction": 0.8789,
    "bagging_freq": 1,
    "min_data_in_leaf": 50,
    "num_boost_round": 1000,      # 配合 early stopping，不会真跑满
    "early_stopping_rounds": 50,
    "num_threads": 20,
}


def _resolve_handler(feature_set: str):
    """按名取 qlib handler 类。

    - ``Alpha158``：158 个横截面因子，适合 LightGBM / MLP
    - ``Alpha360``：6 个价量字段 × 60 天 = 360，**真时序**，
      是 qlib 官方 GRU/LSTM/ALSTM benchmark 的配置
    """
    key = str(feature_set).strip().lower()
    if key not in _FEATURE_SETS:
        raise ValueError(f"未知特征集 {feature_set}，可选：Alpha158 / Alpha360")
    from qlib.contrib.data import handler as _h

    return getattr(_h, _FEATURE_SETS[key])


def _shift_trading_days(date_str: str, n: int) -> str:
    """在 qlib 交易日历上把日期前后挪 n 个交易日（n<0 往前）。

    日历读不到时退化为自然日——purge 会略微保守，但绝不会漏 purge。
    """
    import datetime as _dt

    try:
        from qlib.data import D

        cal = list(D.calendar(start_time="1990-01-01", end_time="2099-12-31", freq="day"))
        target = pd.Timestamp(date_str)
        # bisect：找到 <= target 的最后一个交易日下标
        import bisect

        i = bisect.bisect_right(cal, target) - 1
        j = min(max(i + n, 0), len(cal) - 1)
        return pd.Timestamp(cal[j]).date().isoformat()
    except Exception:
        d = _dt.date.fromisoformat(str(date_str)[:10])
        # 自然日换算：1 个交易日 ≈ 1.45 自然日，向上取整更保守
        return (d + _dt.timedelta(days=int(n * 1.5))).isoformat()


def _build_segments(
    train_start: str, train_end: str, valid_start: str, valid_end: str,
    test_ratio: float = 0.2, embargo_days: int = 5,
) -> dict[str, tuple[str, str]]:
    """构造 qlib ``DatasetH`` 的 segments：train / valid / test，段间 purge。

    - train 尾部 purge ``embargo_days`` 个交易日（标签用到 T+h 的价格）
    - 原 valid 区间尾部按 ``test_ratio`` 切出 test，valid 尾部同样 purge
    """
    segs: dict[str, tuple[str, str]] = {}
    e = max(0, int(embargo_days))
    segs["train"] = (train_start, _shift_trading_days(train_end, -e) if e else train_end)

    if test_ratio and test_ratio > 0:
        v0, v1 = pd.Timestamp(valid_start), pd.Timestamp(valid_end)
        span = (v1 - v0).days
        if span >= 10:  # 太短就不切了，切了两段都没样本
            cut = v0 + pd.Timedelta(days=int(span * (1 - test_ratio)))
            segs["valid"] = (valid_start, _shift_trading_days(cut.date().isoformat(), -e) if e else cut.date().isoformat())
            segs["test"] = ((cut + pd.Timedelta(days=1)).date().isoformat(), valid_end)
            return segs
        print(f"  [warn] 验证区间只有 {span} 天，太短无法再切 test，退化为 train/valid 两段", flush=True)
    segs["valid"] = (valid_start, valid_end)
    return segs


def _prepare_xy(dataset, segment: str):
    """从 qlib dataset 取一段的 (feature, label)，**保留 (datetime, instrument) 索引**。

    原 ``_align_dropna`` 直接 ``.values`` 转 numpy，把 MultiIndex 丢了——
    丢了日期就没法算「每日横截面 IC」，只能退回被日间漂移污染的 pooled IC。
    这里保留索引，只在喂给 torch 时才转 numpy。
    """
    import numpy as _np

    data = dataset.prepare(segment, col_set=["feature", "label"])
    x = data["feature"]
    y = data["label"]
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    # feature 有 NaN/Inf 或 label 有 NaN 的样本一起剔掉（按索引对齐）
    x = x.replace([_np.inf, -_np.inf], _np.nan)
    mask = y.notna() & x.notna().all(axis=1)
    return x[mask], y[mask]


def _eval_segment(model, dataset, segment: str, segments: dict[str, tuple[str, str]],
                  quiet: bool = True) -> dict[str, Any] | None:
    """在指定段上评估。qlib 原生模型走 ``model.predict(dataset, segment=...)``，
    自写模型走 ``model.predict(x)``——两种签名都兼容。"""
    if segment not in segments:
        return None
    from eq.strategy.factors.evaluation import evaluate, format_report

    try:
        x, y = _prepare_xy(dataset, segment)
        if len(y) == 0:
            return None
        try:
            pred = model.predict(dataset, segment=segment)   # qlib 原生签名
            pred_s = pred.iloc[:, 0] if isinstance(pred, pd.DataFrame) else pd.Series(pred)
            pred_s = pred_s.reindex(y.index).dropna()
            y = y.reindex(pred_s.index)
        except TypeError:
            pred = model.predict(x)                          # 自写模型签名
            pred_s = pd.Series(np.asarray(pred).reshape(-1), index=x.index)
    except Exception as e:
        print(f"  [warn] {segment} 段评估失败：{e}", flush=True)
        return None
    report = evaluate(pred_s, y)
    if not quiet:
        lo, hi = segments[segment]
        print(format_report(report, title=f"{segment} 段评估（{lo}~{hi}）"), flush=True)
    return report


def _eval_on_test(model, dataset, segments: dict[str, tuple[str, str]]) -> dict[str, Any] | None:
    """在**独立测试段**上评估模型，返回 :func:`evaluation.evaluate` 的报告。

    没有 test 段（区间太短）时返回 None，调用方退化为报 valid IC——
    但那个数字是选择集上的最大值，会偏高，注册进 ml_models 时用
    ``valid_ic`` 键区分。
    """
    report = _eval_segment(model, dataset, "test", segments, quiet=True)
    if report is None:
        return None
    from eq.strategy.factors.evaluation import format_report

    lo, hi = segments["test"]
    print(format_report(report, title=f"测试段评估（{lo}~{hi}，训练/选模型都没见过）"), flush=True)
    return report


# ---------- qlib PyTorch 模型（走 CUDA，CUDA GPU 主场） ----------

_TORCH_ALGOS = {"alstm", "gru", "lstm", "mlp", "deeplob", "tft"}


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


# ---------- 多种子集成 ----------

class SeedEnsemble:
    """同配置、不同随机种子的一组模型，预测取**标准化后**的平均。

    低信噪比数据上单次训练的方差极大：同一份数据同一套超参，换个种子
    test IC 能差出一倍。集成是这种场景下最稳的一招——它降的是方差，
    不需要任何额外的调参运气。

    **为什么先标准化再平均**：各模型的输出尺度不可比（MSE 训出来的回归值，
    不同种子收敛到的量纲不一样），直接取算术平均等于给输出方差大的模型
    更高的话语权。逐模型去均值除标准差之后再平均，每个成员的权重才一样。
    标准化是单调变换，不改变任何单个模型的截面排序。

    对 :func:`_eval_segment` 的兼容：它先试 qlib 签名 ``predict(dataset, segment=)``，
    抛 TypeError 再退回 ``predict(x)``。本类只接一个位置参数，正好走后者。
    """

    def __init__(self, models: list, seeds: list[int]):
        if not models:
            raise ValueError("集成至少要有一个模型")
        self.models = models
        self.seeds = list(seeds)
        # 成员里最好的那个 valid 分数，仅供日志参考（集成本身的分数要另外评）
        self.member_scores = [float(getattr(m, "best_score", 0.0)) for m in models]
        self.best_score = max(self.member_scores) if self.member_scores else 0.0
        self.best_step = max((int(getattr(m, "best_step", 0)) for m in models), default=0)

    def predict(self, x):
        import numpy as np

        acc = None
        for m in self.models:
            p = np.asarray(m.predict(x), dtype=float).reshape(-1)
            s = p.std()
            z = (p - p.mean()) / s if s > 0 else np.zeros_like(p)
            acc = z if acc is None else acc + z
        return acc / len(self.models)

    def __len__(self) -> int:
        return len(self.models)


# ---------- 特征/标签处理器：训练和推理必须用同一套 ----------

def infer_processors() -> list[dict]:
    """特征处理器链，**训练和预测共用**（对标 qlib benchmarks 的 Alpha158 配置）。

    - ``ProcessInf``：Inf 换成列均值，否则 BatchNorm1d 梯度直接爆
    - ``RobustZScoreNorm``：MAD（中位绝对偏差）z-score，比普通 z-score 抗异常；
      ``clip_outlier=True`` 截断 3σ 外的极值
    - ``Fillna``：剩余缺失填 0

    **为什么必须抽成一个函数**：改之前这份配置在文件里散着写了四遍，
    其中 ``predict_batch`` 那份和训练那份对不上——

    | | 训练 | 预测（旧） |
    |---|---|---|
    | ProcessInf | 有 | **没有** |
    | RobustZScoreNorm | ``fields_group="feature"``, ``clip_outlier=True`` | 裸调用，两个参数都没有 |
    | CSRankNorm | 只作用在 **label** 上 | **加在特征上**（训练时根本没有这一步）|

    也就是模型训练时吃的是 z-score 特征，上线推理时吃的是横截面 rank（[0,1] 均匀分布）
    ——分布完全不同，等于拿 A 的尺子量 B。这类 train/serve skew 不会报错、
    也不会让预测变成 NaN，只是安静地把每一次 ``eq ml predict`` 的结果打偏。
    """
    return [
        {"class": "ProcessInf", "kwargs": {}},
        {"class": "RobustZScoreNorm",
         "kwargs": {"fields_group": "feature", "clip_outlier": True}},
        {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
    ]


def learn_processors(rank_norm: bool = True) -> list[dict]:
    """标签处理器链。

    ``CSRankNorm`` 把当日截面的未来收益转成 [0,1] 均匀分布——这一步决定了
    模型学的是「今天这批票里哪只更好」而不是「明天大盘涨不涨」。
    qlib 官方 GRU/ALSTM benchmark 用的就是它；``rank_norm=False`` 退回
    ``CSZScoreNorm``（截面 z-score，保留幅度信息，但对异常值更敏感）。
    """
    norm = "CSRankNorm" if rank_norm else "CSZScoreNorm"
    return [
        {"class": "DropnaLabel"},
        {"class": norm, "kwargs": {"fields_group": "label"}},
    ]


# ---------- 优化器相关的默认超参 ----------

# Lion 的更新量是 sign(...)，每个坐标恒定走 ±lr，和梯度大小无关；
# AdamW 的更新量是 g/√v，被自适应缩放过。所以同一个 lr 在 Lion 下的实际步长
# 大得多——Lion 论文（Chen et al. 2023, Symbolic Discovery of Optimization
# Algorithms）明确建议 **lr 取 AdamW 的 1/3~1/10、weight_decay 放大 3~10 倍**
# （大致保持 lr×wd 乘积不变）。
#
# 本项目 v0.32 把默认优化器切成 Lion，但 lr/weight_decay 还留着 AdamW 那套
# （1e-3 / 1e-5），等于让 Lion 用约 10 倍的步长跑——低信噪比的金融数据上，
# 这会让它在极小值附近反复横跳，早停挑到的多半是噪声高点。
_OPT_DEFAULTS = {
    "lion":  {"lr": 1e-4, "weight_decay": 1e-4},
    "adamw": {"lr": 1e-3, "weight_decay": 1e-5},
}


def resolve_opt_hparams(optimizer: str, lr: float | None = None,
                        weight_decay: float | None = None) -> tuple[float, float]:
    """按优化器给出 (lr, weight_decay)。显式传值优先，None 才用默认。

    显式优先是关键：``--lr`` 传了就必须照做，否则用户根本没法调参
    （改动前这两个值在调用点写死，命令行想调也调不了）。
    """
    d = _OPT_DEFAULTS.get(str(optimizer).lower(), _OPT_DEFAULTS["adamw"])
    return (d["lr"] if lr is None else float(lr),
            d["weight_decay"] if weight_decay is None else float(weight_decay))


# ---------- 训练期打分：早停必须用「和验收同一把尺」 ----------

def _make_valid_scorer(x_valid, y_valid):
    """造一个 ``pred_tensor -> float`` 的验证集打分函数，口径＝**每日横截面 Rank IC**。

    修的是一个隐蔽但影响很大的错配：早停按**池化 Pearson IC**挑 checkpoint
    （把整个验证集当一坨算一次相关），而模型最终是按 :func:`evaluation.evaluate`
    的**每日横截面 Rank IC** 验收和使用的。两个口径在截面数据上根本不是一回事——
    池化 IC 会把「不同日期的收益水平差」算成预测力，于是早停可能挑中一个
    「能区分牛市日和熊市日、但当天选不出好票」的 checkpoint。选的和考的不一致，
    调参再多也是在优化错的东西。

    统一改成直接调 :func:`evaluation.daily_ic`——全项目只留一个 IC 定义，
    以后改口径不会再出现两处不同步。

    验证集没有 (datetime, instrument) 索引时（比如直接喂 numpy），
    ``daily_ic`` 自己会退回池化口径，此时行为和改动前一致。
    """
    import numpy as np
    import pandas as pd

    idx = getattr(x_valid, "index", None)
    y_arr = np.asarray(y_valid).reshape(-1)
    if idx is not None and len(idx) == len(y_arr):
        label = pd.Series(y_arr, index=idx)
    else:
        label = pd.Series(y_arr)

    from eq.strategy.factors.evaluation import daily_ic

    def _score(pred) -> float:
        # 既接 torch 张量（训练循环里），也接 numpy 数组（集成评分时）
        if hasattr(pred, "detach"):
            pred = pred.detach().cpu().numpy()
        arr = np.asarray(pred, dtype=float).reshape(-1)
        if len(arr) != len(label) or len(arr) < 2:
            return -float("inf")
        ics = daily_ic(pd.Series(arr, index=label.index), label)
        if len(ics) == 0:
            return -float("inf")
        m = float(ics.mean())
        return m if m == m else -float("inf")   # NaN → -inf，别让它当上最佳

    return _score


# ---------- 自写最简 MLP（走 torch.cuda，绕开 qlib DNNModelPytorch nan 坑） ----------

class MLPAlphaNet:
    """最简 MLP：158 -> 256 -> 1，走 BatchNorm1d + Adam，支持 CUDA。

    qlib DNNModelPytorch 在 torch 2.13 + Alpha158 默认配置下 loss 全 nan（BatchNorm1d 坑），
    自写此绕开，只取 qlib handler 的 feature 和 label 做数据，训练用原生 torch。
    """

    def __init__(self, input_dim: int = 158, hidden: int | tuple = 256, lr: float | None = None,
                 max_steps: int = 300, batch_size: int = 2000, device: str = "cuda",
                 optimizer: str = "lion", dropout: float = 0.3, seed: int | None = None,
                 weight_decay: float | None = None):
        import torch
        import torch.nn as nn

        if seed is not None:
            from eq.strategy.factors.validation import set_seed
            set_seed(seed)
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        # lr/weight_decay 缺省时按优化器给（Lion 要比 AdamW 小一个量级，见 _OPT_DEFAULTS）
        lr, weight_decay = resolve_opt_hparams(optimizer, lr, weight_decay)
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.dropout = dropout
        self.seed = seed
        # hidden 支持 int（单隐层）或 tuple（多隐层，如 (512,256,128)）
        if isinstance(hidden, int):
            hidden_layers = [hidden]
        else:
            hidden_layers = list(hidden)
        # 构造 158 → h1 → h2 → ... → 1 的 Sequential，每层 Linear+BN+ReLU+Dropout
        # dropout 此前硬编码 0.05 且不接受参数——CLI 的 --dropout 对 MLP 完全无效。
        # 金融数据信噪比极低，0.05 基本等于没正则。
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers).to(self.device)
        # 默认 Lion：显存省半 + 符号操作抗噪，适合低信噪比金融数据；
        # optimizer="adamw" 可切回（reduce_lr/warmup 都走 param_groups[0]["lr"] 兼容）
        if optimizer.lower() == "lion":
            self.opt = _LionOpt(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = nn.MSELoss()
        self.best_score = -float("inf")
        self.best_state = None
        self.best_step = 0

    def fit(self, x_train, y_train, x_valid, y_valid, early_stop: int = 30):
        import numpy as np
        import torch

        def _to_tensor(df):
            if hasattr(df, "values"):
                return torch.from_numpy(df.values).float()
            return torch.from_numpy(np.asarray(df)).float()

        xt = _to_tensor(x_train).to(self.device)
        yt = _to_tensor(y_train).squeeze(-1).to(self.device)
        xv = _to_tensor(x_valid).to(self.device)
        yv = _to_tensor(y_valid).squeeze(-1).to(self.device)
        # 早停打分要用**每日横截面 Rank IC**，所以得留住验证集的 (日期, 标的) 索引
        # ——转成 tensor 就丢了。拿不到索引时 scorer 自动退回池化口径。
        # 只做局部变量：它是闭包，挂到 self 上模型就 pickle 不了（存盘会炸）。
        scorer = _make_valid_scorer(x_valid, y_valid)

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
            score = scorer(vp)
            # 每 10 步或新最佳打进度
            mem_mb = torch.cuda.memory_allocated() / 1e6 if self.device.type == "cuda" else 0.0
            if step % 10 == 0 or score > self.best_score:
                best_mark = "✓" if score > self.best_score else " "
                print(f"  [MLP step {step:3d}] loss={loss.item():.4f} valid_loss={vl:.4f} IC={score:+.4f} {best_mark} mem={mem_mb:.0f}MB", flush=True)
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
        print(f"  [MLP 训练完成] best IC={self.best_score:+.4f} @step {self.best_step} (early_stop={stop}/{early_stop})", flush=True)

    def predict(self, x):
        import torch
        import numpy as np
        self.net.eval()
        with torch.no_grad():
            xt = torch.from_numpy(np.asarray(x if not hasattr(x, "values") else x.values)).float().to(self.device)
            if xt.dim() == 1:
                xt = xt.unsqueeze(0)  # (158,) → (1,158)
            if len(xt) < 2:
                # BatchNorm1d 需 batch≥2，扩成 2 样本预测后取首
                xt = xt.repeat(2, 1)
                pred = self.net(xt).squeeze(-1)[0:1]
            else:
                pred = self.net(xt).squeeze(-1)
            return pred.cpu().numpy()


# ---------- 自写时序模型（LSTM/GRU + 动态学习率，走 torch.cuda） ----------

class RecurrentAlphaNet:
    """自写时序模型：支持 LSTM/GRU，ReduceLROnPlateau 动态学习率，AdamW 优化器。

    把 Alpha158 的 158 维特征重塑成 (batch, seq_len=6, input_size=26) 喂给 RNN。

    研究结论（2024-2025 文献综合）：
    - GRU > LSTM（2门 vs 3门，参数少=不过拟合，S&P 500 84%方向准确率）
    - 浅层 2-3 层够（深层过拟合）
    - Adam lr=0.001 标配，ReduceLROnPlateau 稳
    - Dropout 0.1-0.2 最佳（太高反而伤）
    """

    def __init__(self, input_dim: int = 158, seq_len: int = 6, input_size: int = 26,
                 hidden_size: int = 64, num_layers: int = 2, cell_type: str = "gru",
                 lr: float | None = None, max_steps: int = 200, batch_size: int = 4000,
                 device: str = "cuda", dropout: float = 0.1, use_scheduler: bool = True,
                 optimizer: str = "lion", seed: int | None = None,
                 weight_decay: float | None = None):
        import math

        import torch
        import torch.nn as nn

        if seed is not None:
            from eq.strategy.factors.validation import set_seed
            set_seed(seed)
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.seq_len = seq_len
        # 特征丢失修复：原来 cut = seq_len*input_size = 6*26 = 156 < input_dim=158，
        # `_reshape` 走的是 x[:, :cut]，**静默丢掉最后 2 维特征**——而它的 docstring
        # 却写着"保证 158 维全保留"。这里把 input_size 上调到能覆盖 input_dim 的最小值，
        # 不足部分由 _reshape 补零（补零对 RNN 贡献为 0，不污染信号）。
        need = math.ceil(input_dim / seq_len) if seq_len > 0 else input_size
        if seq_len * input_size < input_dim:
            print(
                f"  [warn] seq_len({seq_len})×input_size({input_size})={seq_len * input_size}"
                f" < input_dim({input_dim})，会丢 {input_dim - seq_len * input_size} 维特征；"
                f"已自动把 input_size 调到 {need}（不足部分补零）",
                flush=True,
            )
            input_size = need
        self.input_dim = input_dim
        self.input_size = input_size
        # lr 缺省按优化器给（Lion 比 AdamW 小一个量级，见 _OPT_DEFAULTS）。
        # 必须在 self.lr 赋值**之前**解析——warmup 用的是 self.lr，
        # 拿到 None 会在 self.lr/10 处炸。
        lr, _ = resolve_opt_hparams(optimizer, lr, None)
        self.lr = lr
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.cell_type = cell_type
        self.use_scheduler = use_scheduler

        rnn_cls = nn.GRU if cell_type == "gru" else nn.LSTM
        # 输入归一化：Alpha158 原始特征虽经 RobustZScoreNorm，但重塑后尺度仍可能让
        # LSTM 的 tanh 饱和（所有样本输出几乎相同 → std=0 → IC=-inf 死循环）。
        # 加 BatchNorm1d 对重塑后每步 26 维做 BN（reshape 后用 view 摊平 2D 喂 BN，
        # 避开 BatchNorm1d 对 3D 张量按第二维算的坑），防输出塌缩。
        self.input_bn = nn.BatchNorm1d(input_size).to(self.device)
        self.net = rnn_cls(input_size, hidden_size, num_layers=num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0).to(self.device)
        # Xavier 初始化：默认初始化权重过小 + tanh 饱和 + BN 收敛，导致 LSTM 输出
        # 塌缩成常数（pred.std()=0 → IC=0 恒不更新）。Xavier 拉开初始权重尺度，
        # 配合首步 warmup 让首步就有差异化输出，训练能真推进。
        for name, p in self.net.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                nn.init.xavier_normal_(p)
            elif "bias_ih" in name or "bias_hh" in name:
                nn.init.zeros_(p)
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, 1),
        ).to(self.device)
        # weight_decay **不**跟着 Lion 那套放大 10 倍：RNN 对 weight_decay 极敏感，
        # 过强会把权重压向 0、加剧输出塌缩（实测过 pred.std()=0 恒 IC=0），
        # 所以这里独立定在 1e-6，显式传参可覆盖。（lr 已在上面解析过）
        weight_decay = 1e-6 if weight_decay is None else float(weight_decay)
        self.weight_decay = weight_decay
        # 默认 Lion：显存省半 + 符号操作抗噪，适合低信噪比金融数据；
        # optimizer="adamw" 可切回（warmup/scheduler 都走 param_groups[0]["lr"] 兼容）。
        _all_params = list(self.net.parameters()) + list(self.head.parameters()) + list(self.input_bn.parameters())
        if optimizer.lower() == "lion":
            self.opt = _LionOpt(_all_params, lr=lr, weight_decay=weight_decay)
        else:
            self.opt = torch.optim.AdamW(_all_params, lr=lr, weight_decay=weight_decay)
        if use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.opt, mode="max", factor=0.5, patience=5, min_lr=1e-6,
            )
        self.loss_fn = nn.MSELoss()
        self.best_score = -float("inf")
        self.best_state = None
        self.best_step = 0
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self._warmup_steps = 5  # 首 5 步 lr 从 1e-4 线性升到 lr，防首步塌缩

    def _reshape(self, x):
        """把 (batch, input_dim) 重塑成 (batch, seq_len, input_size)。

        构造函数已保证 ``seq_len * input_size >= input_dim``，所以这里
        只会补零、不会截断——不再静默丢特征。

        .. warning::
           对 **Alpha158** 而言这个"时间轴"是假的：158 个特征之间没有时间
           顺序（是 KBAR/MA/STD/BETA/RSQR 等各类横截面因子），把它 view 成
           (6, 26) 后 RNN 在"时间轴"上跑的是任意特征分组，GRU/LSTM 实际退化
           成一个参数共享的 MLP。真要做时序建模应该用 **Alpha360**
           （6 个价量字段 × 60 天，天然时序），即 qlib 官方 GRU/LSTM
           benchmark 的配置——见 ``train_torch(feature_set="Alpha360")``。
        """
        import torch
        cut = self.seq_len * self.input_size
        if x.size(1) < cut:
            pad = torch.zeros(x.size(0), cut - x.size(1), device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        return x[:, :cut].view(x.size(0), self.seq_len, self.input_size)

    def fit(self, x_train, y_train, x_valid, y_valid, early_stop: int = 20):
        import torch
        import numpy as np

        def _to_tensor(df):
            if hasattr(df, "values"):
                return torch.from_numpy(df.values).float()
            return torch.from_numpy(np.asarray(df)).float()

        xt = _to_tensor(x_train).to(self.device)
        yt = _to_tensor(y_train).squeeze(-1).to(self.device)
        xv = _to_tensor(x_valid).to(self.device)
        # 同 MLP：早停口径统一到每日横截面 Rank IC（见 _make_valid_scorer）。
        # 打分改由 scorer 用 pandas 侧的 y_valid 完成，不再需要 yv 张量。
        # 同样只做局部变量：闭包不可 pickle，挂上去模型存不了盘。
        scorer = _make_valid_scorer(x_valid, y_valid)

        stop = 0
        for step in range(self.max_steps):
            # warmup：首 _warmup_steps 步 lr 从 self.lr/10 线性升到 self.lr，
            # 防首步权重过大 + BN 未收敛导致 LSTM 输出塌缩成常数（pred.std()=0）。
            # 起点写成相对值：原来硬编码 1e-4，Lion 默认 lr 也是 1e-4 时
            # warmup 变成 1e-4→1e-4 的空转，等于没预热。
            if step < self._warmup_steps:
                _lr0 = self.lr / 10
                warmup_lr = _lr0 + (self.lr - _lr0) * (step + 1) / self._warmup_steps
                for g in self.opt.param_groups:
                    g["lr"] = warmup_lr
            self.net.train()
            self.head.train()
            self.input_bn.train()
            idx = torch.randperm(len(xt), device=self.device)
            for i in range(0, len(idx), self.batch_size):
                b = idx[i:i + self.batch_size]
                if len(b) < 2:
                    break
                xb = self._reshape(xt[b])
                # reshape 后 (batch, 6, 26) → view 摊平 2D 傂 BN 再 view 回
                xb = self.input_bn(xb.reshape(-1, self.input_size)).reshape(xb.size(0), self.seq_len, self.input_size)
                out, _ = self.net(xb)
                pred = self.head(out[:, -1, :]).squeeze(-1)
                loss = self.loss_fn(pred, yt[b])
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.opt.step()
            # eval
            self.net.eval()
            self.head.eval()
            self.input_bn.eval()
            with torch.no_grad():
                xv_r = self._reshape(xv)
                xv_r = self.input_bn(xv_r.reshape(-1, self.input_size)).reshape(xv_r.size(0), self.seq_len, self.input_size)
                out, _ = self.net(xv_r)
                vp = self.head(out[:, -1, :]).squeeze(-1)
                # 输出塌缩（pred.std()=0）时打 0 而不是 -inf：-inf 会让 best_score
                # 永不更新，warmup 阶段还没散开的模型会被判死刑，训练推进不下去。
                score = scorer(vp)
                if score == -float("inf") or vp.std().item() == 0:
                    score = 0.0
            # 动态学习率（IC 不再提升时降 LR）
            if self.use_scheduler:
                self.scheduler.step(score)
            # 进度
            mem_mb = torch.cuda.memory_allocated() / 1e6 if self.device.type == "cuda" else 0.0
            cur_lr = self.opt.param_groups[0]["lr"]
            if step % 5 == 0 or score > self.best_score:
                best_mark = "✓" if score > self.best_score else " "
                print(f"  [{self.cell_type.upper()} step {step:3d}] IC={score:+.4f} {best_mark} best={self.best_score:+.4f}@{self.best_step} lr={cur_lr:.2e} mem={mem_mb:.0f}MB", flush=True)
            if score > self.best_score:
                self.best_score = score
                self.best_step = step
                self.best_state = {
                    "net": {k: v.clone() for k, v in self.net.state_dict().items()},
                    "head": {k: v.clone() for k, v in self.head.state_dict().items()},
                    "input_bn": {k: v.clone() for k, v in self.input_bn.state_dict().items()},
                }
                stop = 0
            else:
                stop += 1
                if stop >= early_stop:
                    break
        if self.best_state is not None:
            self.net.load_state_dict(self.best_state["net"])
            self.head.load_state_dict(self.best_state["head"])
            self.input_bn.load_state_dict(self.best_state["input_bn"])
        print(f"  [{self.cell_type.upper()} 训练完成] best IC={self.best_score:+.4f} @step {self.best_step} (early_stop={stop}/{early_stop})", flush=True)

    def predict(self, x):
        import torch
        import numpy as np
        self.net.eval()
        self.head.eval()
        self.input_bn.eval()
        with torch.no_grad():
            xt = torch.from_numpy(np.asarray(x if not hasattr(x, "values") else x.values)).float().to(self.device)
            if xt.dim() == 1:
                xt = xt.unsqueeze(0)  # (158,) → (1,158)
            if len(xt) < 2:
                # BatchNorm1d 需 batch≥2，扩成 2 样本预测后取首
                xt = xt.repeat(2, 1)
                xr = self._reshape(xt)
                xr = self.input_bn(xr.reshape(-1, self.input_size)).reshape(xr.size(0), self.seq_len, self.input_size)
                out, _ = self.net(xr)
                pred = self.head(out[:, -1, :]).squeeze(-1)[0:1]
            else:
                xr = self._reshape(xt)
                xr = self.input_bn(xr.reshape(-1, self.input_size)).reshape(xr.size(0), self.seq_len, self.input_size)
                out, _ = self.net(xr)
                pred = self.head(out[:, -1, :]).squeeze(-1)
            return pred.cpu().numpy()


# ---- 向后兼容别名 ----
# 训练好的模型是 pickle 整个实例存盘的，pickle 记的是「模块路径 + 类名」。
# 直接改名会让 v0.26 之前存下来的所有 .pkl 加载不了（AttributeError），
# 所以旧名字必须留着指向新类。
_SimpleMLP = MLPAlphaNet
_SimpleSeqModel = RecurrentAlphaNet
_SimpleLSTM = RecurrentAlphaNet


def train_torch(
    universe: str = "csi300",
    train_start: str = "2015-01-01",
    train_end: str = "2020-08-31",
    valid_start: str = "2020-09-01",
    valid_end: str = "2020-09-25",
    horizon: int = 5,
    algo: str = "gru",
    device: str = "cuda",  # 默认 cuda（真 CUDA，CUDA GPU 主场）
    hidden_size: int = 0,
    num_layers: int = 0,
    batch_size: int = 0,
    name: str | None = None,
    # --- 高级参数（DeepLOB/TFT） ---
    optimizer: str = "lion",  # 默认 Lion：显存省半 + 符号操作抗噪，适合低信噪比金融数据
    loss_type: str = "sharpe",
    dropout: float = 0.3,
    adversarial: bool = False,
    orthogonalize: bool = False,
    seq_len: int = 0,
    num_heads: int = 4,
    gpu_ids: str | list[int] | None = None,  # 多卡并行
    # --- v0.25 训练策略参数 ---
    test_ratio: float = 0.2,
    embargo_days: int | None = None,
    seed: int = 42,
    feature_set: str = "Alpha158",
    # v0.36：学习率/权重衰减此前写死在调用点，命令行给了也没用。
    # None = 按优化器取默认（见 resolve_opt_hparams）
    lr: float | None = None,
    weight_decay: float | None = None,
    # v0.37：多种子集成。低信噪比数据上单次训练方差极大，同超参换个种子
    # test IC 能差一倍；集成降的是方差，不靠调参运气。见 SeedEnsemble。
    n_seeds: int = 1,
) -> dict[str, Any]:
    """走 qlib PyTorch pipeline 训练 ALSTM/GRU/LSTM/MLP/DeepLOB/TFT，用 CUDA。

    Args:
        algo: alstm | gru | lstm | mlp | deeplob | tft
        device: cuda | cpu
        optimizer: adamw | sam | lookahead | lion（仅 DeepLOB/TFT）
        loss_type: sharpe | mse | ic（仅 DeepLOB/TFT）
        dropout: Dropout 率（量化建议 0.3-0.4）
        adversarial: FGSM 对抗训练（仅 DeepLOB/TFT）
        orthogonalize: 特征正交化去 Beta
        seq_len: DeepLOB/TFT 输入窗口
        num_heads: TFT 注意力头数
        test_ratio: 从 [valid_start, valid_end] 尾部切出的**独立测试段**比例。
            0 = 沿用旧行为（只有 train/valid，报告的 IC 会偏高）。
        embargo_days: 段间 purge 的交易日数，缺省 = horizon。标签是
            ``Ref($close,-h)/Ref($close,-1)-1``，训练集尾部 h 天的标签
            已经看过验证期价格，不 purge 就是泄漏。
        seed: 随机种子。此前没有种子控制，同一条命令两次跑出的 IC 能差一大截。
        feature_set: ``Alpha158``（默认）| ``Alpha360``。RNN 类模型（gru/lstm/alstm）
            建议用 Alpha360——它是 6 个价量字段 × 60 天的真时序张量，
            而 Alpha158 被 reshape 成 (6,26) 的"时间轴"是假的。

    Returns:
        {"model_id", "metrics", "model_path"}。``metrics`` 里
        ``ic`` 是**测试段**的 Rank IC（test_ratio>0 时），
        ``valid_ic`` 才是旧口径的验证集最优值。
    """
    _qlib_init()
    from eq.strategy.factors.evaluation import evaluate, format_report
    from eq.strategy.factors.validation import purged_split, set_seed

    set_seed(seed)
    embargo = horizon if embargo_days is None else int(embargo_days)

    # Alpha158 handler（feature 158 维）
    # 权威处理器链（对标 qlib examples/benchmarks/GRU/workflow_config_gru_Alpha158.yaml）：
    #   infer_processors (特征): ProcessInf → RobustZScoreNorm(clip) → Fillna
    #     - RobustZScoreNorm 用 MAD（中位绝对偏差）抗异常，比 ZScoreNorm 稳
    #     - clip_outlier=True 截断 3σ 外的极值
    #   learn_processors (标签): DropnaLabel → CSRankNorm
    #     - CSRankNorm 横截面排序归一化，把未来收益转成 [0,1] 均匀分布
    #     - 这是官方 GRU/ALSTM/LSTM benchmark 的标准配置，比 CSZScoreNorm 更抗异常
    handler_cls = _resolve_handler(feature_set)
    infer_procs = infer_processors()
    learn_procs = learn_processors(rank_norm=True)
    label_expr = [f"Ref($close, -{horizon}) / Ref($close, -1) - 1"]
    handler = handler_cls(
        instruments=universe,
        start_time=train_start,
        end_time=valid_end,
        fit_start_time=train_start,
        fit_end_time=train_end,
        infer_processors=infer_procs,
        learn_processors=learn_procs,
        label=label_expr,
    )

    from qlib.data.dataset import DatasetH
    # 三段切分：从原 valid 区间尾部切出独立 test。
    # 之所以必须有 test：fit() 用 valid IC 做 early stopping + best_state 选择，
    # 旧代码又把同一个 best_score（200 个 epoch 里的最大值）当模型成绩报出去——
    # 在选择集上报最大值，纯噪声也能"跑出" IC=0.03~0.05。
    segments = _build_segments(
        train_start, train_end, valid_start, valid_end,
        test_ratio=test_ratio, embargo_days=embargo,
    )
    print("  [切分] " + "  ".join(f"{k}={v[0]}~{v[1]}" for k, v in segments.items())
          + f"  (purge={embargo}日, seed={seed}, 特征集={feature_set})", flush=True)
    dataset = DatasetH(handler=handler, segments=segments)

    # --- 高级模型路径：DeepLOB / TFT ---
    if algo in ("deeplob", "tft"):
        from eq.strategy.factors.advanced_models import (
            DeepAlphaTrainer, DeepLOB, TemporalFusionTransformer,
        )

        x_train, y_train = _prepare_xy(dataset, "train")
        x_valid, y_valid = _prepare_xy(dataset, "valid")

        # 构建模型
        _seq_len = seq_len if seq_len > 0 else (120 if algo == "deeplob" else 60)
        _hidden = hidden_size if hidden_size > 0 else (64 if algo == "deeplob" else 256)
        _batch = batch_size if batch_size > 0 else (512 if algo == "deeplob" else 256)
        input_size = 26  # 6×26 时序重塑的每步维度

        if algo == "deeplob":
            model = DeepLOB(
                seq_len=_seq_len, input_dim=input_size,
                lstm_hidden=_hidden, dropout=dropout,
                raw_input_dim=158,  # Alpha158 原始 158 维 → 投影到 seq_len * input_size
            )
            notes = f"DeepLOB CNN+BiLSTM+Attention（seq={_seq_len}, hidden={_hidden}, dropout={dropout}）"
        else:
            model = TemporalFusionTransformer(
                input_dim=158,  # Alpha158 原始 158 维
                hidden_dim=_hidden,
                num_heads=num_heads,
                dropout=dropout,
                max_seq_len=1,  # 单时间步，不做时序展开
            )
            notes = f"TFT（seq={_seq_len}, hidden={_hidden}, heads={num_heads}, dropout={dropout}）"

        # 用 AdvancedTrainer 训练
        trainer = DeepAlphaTrainer(
            model=model,
            optimizer_type=optimizer,
            loss_type=loss_type,
            learning_rate=1e-4,
            weight_decay=0.01,
            max_steps=300,
            batch_size=_batch,
            early_stop=30,
            use_adversarial=adversarial,
            adversarial_epsilon=0.01,
            orthogonalize=orthogonalize,
            use_scheduler=True,
            device=device,
            gpu_ids=gpu_ids,
            verbose=True,
            seed=seed,
        )
        result = trainer.fit(x_train, y_train, x_valid, y_valid)
        valid_ic = float(result["best_ic"])
        test_report = _eval_on_test(model, dataset, segments)
        ic = test_report["ic_mean"] if test_report else valid_ic

        # 存盘
        import pickle as _pkl
        model_path = _ensure_dir() / f"torch_{algo}_{universe}_{horizon}d.pkl"
        with open(model_path, "wb") as f:
            _pkl.dump(model, f)

        model_id = register_model(
            name=name or f"{universe}_{algo}_h{horizon}_{dt.date.today().strftime('%Y%m%d')}",
            universe=universe,
            features=[f"{feature_set}"],
            algo=algo,
            horizon=horizon,
            train_period=f"{segments['train'][0]}~{segments['train'][1]}",
            valid_period=f"{segments['valid'][0]}~{segments['valid'][1]}",
            metrics={"ic": ic, "valid_ic": valid_ic, "algo": algo, "horizon": horizon,
                     "device": device, "optimizer": optimizer, "loss": loss_type,
                     "adversarial": adversarial, "orthogonalize": orthogonalize,
                     "seed": seed, "embargo_days": embargo, "feature_set": feature_set,
                     **{f"test_{k}": v for k, v in (test_report or {}).items()
                        if k != "ic_series" and not isinstance(v, list)}},
            model_path=str(model_path),
            notes=notes,
        )
        return {"model_id": model_id,
                "metrics": {"ic": ic, "valid_ic": valid_ic, "epochs": result["best_step"] + 1,
                            "test": test_report},
                "model_path": str(model_path)}

    # --- 原有 PyTorch 模型路径（MLP / LSTM / GRU / ALSTM） ---
    if algo in ("mlp", "lstm", "gru", "alstm"):
        # 自写 MLP/LSTM 路径：从 dataset 取 feature 和 label，用 torch.cuda 训练
        # _prepare_xy 保留 (datetime, instrument) 索引，测试段才能算每日横截面 IC
        x_train, y_train = _prepare_xy(dataset, "train")
        x_valid, y_valid = _prepare_xy(dataset, "valid")
        n_feat = x_train.shape[1]
        print(f"  [DIAG] 样本 train={len(y_train)} valid={len(y_valid)}  "
              f"特征 {n_feat} 维  yt.std={float(y_train.std()):.4f}", flush=True)
        if float(y_train.std()) == 0 or len(y_train) < 10:
            print(f"  [DIAG] 警告: 标签无方差或样本过少（{len(y_train)}），无信号可学", flush=True)

        # 模型工厂：同一套超参、只换种子，供多种子集成复用（数据集只建一次）
        if algo == "mlp":
            def _make(_seed: int):
                return MLPAlphaNet(
                    input_dim=n_feat, hidden=(512, 256, 128), lr=lr, max_steps=300,
                    batch_size=8000, device=device, optimizer=optimizer,
                    dropout=dropout, seed=_seed, weight_decay=weight_decay,
                )
            notes = f"自写 MLPAlphaNet（{device}, {optimizer}, dropout={dropout}, {feature_set}）"
        else:
            # 研究结论：GRU > LSTM（2 门 vs 3 门，参数少不易过拟合），浅层 2-3 层最优
            cell = "lstm" if algo == "lstm" else "gru"
            _hs = hidden_size if hidden_size > 0 else 64
            _nl = num_layers if num_layers > 0 else 2
            _bs = batch_size if batch_size > 0 else 4000
            # Alpha360 是 (60 天 × 6 字段) 的真时序；Alpha158 只能假装 (6, ceil(158/6))
            if feature_set.lower() == "alpha360":
                _seq, _in = 60, 6
            else:
                _seq, _in = 6, 26

            def _make(_seed: int):
                return RecurrentAlphaNet(
                    input_dim=n_feat, seq_len=_seq, input_size=_in,
                    hidden_size=_hs, num_layers=_nl, cell_type=cell,
                    lr=lr, max_steps=200, batch_size=_bs, device=device,
                    dropout=dropout, use_scheduler=True, optimizer=optimizer,
                    seed=_seed, weight_decay=weight_decay,
                )
            notes = (f"自写 {cell.upper()}（{device}, {optimizer}, dropout={dropout}, "
                     f"{feature_set}, seq={_seq}×{_in}）")

        _es = 20 if algo != "mlp" else 30
        _n = max(1, int(n_seeds))
        _seeds = [seed + i for i in range(_n)]
        _members = []
        for _i, _s in enumerate(_seeds):
            if _n > 1:
                print(f"  [集成 {_i + 1}/{_n}] seed={_s}", flush=True)
            _m = _make(_s)
            _m.fit(x_train, y_train, x_valid, y_valid, early_stop=_es)
            _members.append(_m)
        if _n > 1:
            model = SeedEnsemble(_members, _seeds)
            # 集成的 valid 分数要重新算——成员各自的 best_score 是各自的，
            # 平均之后是另一个模型，不能拿成员最好的那个冒充集成成绩
            valid_ic = _make_valid_scorer(x_valid, y_valid)(model.predict(x_valid))
            print(f"  [集成完成] {_n} 个种子  成员 valid IC "
                  f"{[round(s, 4) for s in model.member_scores]}  "
                  f"集成 {valid_ic:+.4f}", flush=True)
            notes += f"｜{_n} 种子集成"
        else:
            model = _members[0]
            valid_ic = float(model.best_score)
        epochs = model.best_step + 1

        test_report = _eval_on_test(model, dataset, segments)
        ic = test_report["ic_mean"] if test_report else valid_ic

        # 存盘（pickle 整个模型实例，含 net state_dict）
        import pickle as _pkl
        # 文件名带种子数：集成模型不能覆盖同 algo 的单模型，否则想对比两者时
        # 先训的那个已经被后训的冲掉了
        _suffix = f"_x{_n}" if _n > 1 else ""
        model_path = _ensure_dir() / f"torch_{algo}_{universe}_{horizon}d{_suffix}.pkl"
        with open(model_path, "wb") as f:
            _pkl.dump(model, f)

        model_id = register_model(
            name=name or f"{universe}_{algo}_h{horizon}_{dt.date.today().strftime('%Y%m%d')}",
            universe=universe,
            features=[feature_set],
            algo=algo,
            horizon=horizon,
            train_period=f"{segments['train'][0]}~{segments['train'][1]}",
            valid_period=f"{segments['valid'][0]}~{segments['valid'][1]}",
            metrics={"ic": ic, "valid_ic": valid_ic, "algo": algo, "horizon": horizon,
                     "device": device, "epochs": epochs, "seed": seed,
                     "n_seeds": _n,
                     # 成员各自的 valid IC：集成没跑赢最好的成员时能一眼看出来
                     **({"member_ics": [round(s, 4) for s in model.member_scores]}
                        if _n > 1 else {}),
                     "embargo_days": embargo, "dropout": dropout, "feature_set": feature_set,
                     **{f"test_{k}": v for k, v in (test_report or {}).items()
                        if k != "ic_series" and not isinstance(v, list)}},
            model_path=str(model_path),
            notes=notes,
        )
        return {"model_id": model_id,
                "metrics": {"ic": ic, "valid_ic": valid_ic, "epochs": epochs, "test": test_report},
                "model_path": str(model_path)}

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


def _score_to_trend(pred_df: pd.DataFrame) -> pd.DataFrame:
    """把回归分数映射成趋势标签 + 概率，不改模型架构。

    用 csi500 池内分数的横截面分位段（quintile）切 5 档：
      分位 ≥80% → 强多（Strong Bullish）
      60~80%   → 弱多（Mild Bullish）
      40~60%   → 中性（Neutral）
      20~40%   → 弱空（Mild Bearish）
      ≤20%     → 强空（Strong Bearish）

    trend_prob 是该股票落在某档的「信心分」：极端档（强多/强空）用距 80/20 分位
    的标准化距离；中性档用距 50% 的距离反算。值域 [0, 1]，越大越肯定。

    Args:
        pred_df: 含 score 列的 DataFrame（横截面分数）
    Returns:
        加 trend / trend_prob 两列的 DataFrame（按 score 降序）
    """
    import numpy as _np
    df = pred_df.copy()
    s = df["score"].astype(float)
    # 横截面分位数（quantile）切档
    q80 = s.quantile(0.80)
    q60 = s.quantile(0.60)
    q40 = s.quantile(0.40)
    q20 = s.quantile(0.20)
    labels = []
    probs = []
    for v in s:
        if v >= q80:
            labels.append("强多")
            # 距 q80 的标准化距离，上限到池内 max
            mx = s.max()
            probs.append(float((v - q80) / (mx - q80 + 1e-9)) if mx > q80 else 0.5)
        elif v >= q60:
            labels.append("弱多")
            probs.append(float((v - q60) / (q80 - q60 + 1e-9)))
        elif v >= q40:
            labels.append("中性")
            # 距 50% 分位越近越肯定中性
            q50 = s.quantile(0.50)
            probs.append(float(1 - abs(v - q50) / (q60 - q40 + 1e-9)))
        elif v >= q20:
            labels.append("弱空")
            probs.append(float((q40 - v) / (q40 - q20 + 1e-9)))
        else:
            labels.append("强空")
            mn = s.min()
            probs.append(float((q20 - v) / (q20 - mn + 1e-9)) if q20 > mn else 0.5)
    df["trend"] = labels
    df["trend_prob"] = [_np.clip(p, 0.0, 1.0) for p in probs]
    return df


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
    meta = dict(meta_rows[0])
    model_path = meta["model_path"]
    universe = meta["universe"] or universe

    # 拉末日数据作为 predict_date
    if predict_date is None:
        # qlib 数据末日 2020-09-25，predict 用 2020-09-25
        predict_date = "2020-09-25"

    # 重新构造 handler 取特征（predict 不需要真 label，用占位表达式避免 horizon 未来数据问题）
    # label 用 Ref($close,-1)/Ref($close,-1)-1 恒为 0 的占位，handler 能跑通，predict 只用 feature
    # infer_processors 必须与训练时**逐字**一致，所以直接调用同一个函数。
    # 旧版在这里手抄了一份，抄错了三处（缺 ProcessInf、丢 clip_outlier、
    # 多了个训练时没有的 CSRankNorm），详见 infer_processors() 的 docstring。
    handler = Alpha158(
        instruments=universe,
        start_time=predict_date,
        end_time=predict_date,
        fit_start_time="2015-01-01",
        fit_end_time=predict_date,
        infer_processors=infer_processors(),
        label=["Ref($close, -1) / Ref($close, -1) - 1"],
    )
    from qlib.data.dataset import DatasetH
    dataset = DatasetH(handler=handler, segments={"test": (predict_date, predict_date)})

    # 加载模型（pickle 直加载绕开 qlib LGBModel.load 触发的 torch DLL 链）
    import pickle as _pkl
    with open(model_path, "rb") as f:
        model = _pkl.load(f)

    # 按 algo 分路预测：
    # - LightGBM（qlib LGBModel）：model.predict(dataset, segment="test") 返回 pd.Series
    # - 自写 LSTM/MLP（_SimpleLSTM/MLPAlphaNet）：从 dataset 取 feature DataFrame，model.predict(x) 返回 ndarray
    from eq.db import execute as _execute
    algo_row = _execute("SELECT algo FROM ml_models WHERE id = ?", (model_id,))
    algo = algo_row[0]["algo"] if algo_row else "lightgbm"

    if algo in ("lstm", "gru", "alstm", "mlp", "deeplob", "tft"):
        # 自写模型路径：取 feature，喂 model.predict(x)
        test_data = dataset.prepare("test", col_set="feature")
        # test_data 可能是 DataFrame（index 是 MultiIndex datetime, instrument）或 dict
        if isinstance(test_data, dict):
            test_data = test_data.get("feature", pd.DataFrame())
        if test_data is None or test_data.empty:
            return pd.DataFrame(columns=["symbol", "score"])
        # 调自写模型 predict（接 DataFrame，返回 ndarray）
        scores = model.predict(test_data)
        # 构造 pred_df，index 复用 test_data 的 MultiIndex
        pred_df = pd.DataFrame({"score": scores}, index=test_data.index)
        if isinstance(pred_df.index, pd.MultiIndex):
            if predict_date in pred_df.index.get_level_values(0):
                pred_df = pred_df.xs(predict_date, level=0)
            else:
                pred_df = pred_df.groupby(level=1).last()
            pred_df = pred_df.reset_index()
            inst_col = "instrument" if "instrument" in pred_df.columns else pred_df.columns[0]
            pred_df = pred_df.rename(columns={inst_col: "symbol"})
        else:
            pred_df = pred_df.reset_index()
    else:
        # LightGBM 路径：qlib LGBModel.predict(dataset, segment) 返回 pd.Series
        pred = model.predict(dataset, segment="test")
        if pred is None or (isinstance(pred, pd.Series) and pred.empty):
            return pd.DataFrame(columns=["symbol", "score"])
        pred_df = pred.to_frame("score") if isinstance(pred, pd.Series) else pred
        if isinstance(pred_df.index, pd.MultiIndex):
            if predict_date in pred_df.index.get_level_values(0):
                pred_df = pred_df.xs(predict_date, level=0)
            else:
                pred_df = pred_df.groupby(level=1).last()
            pred_df = pred_df.reset_index()
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

    # 分数 → 趋势标签 + 概率（强多/弱多/中性/弱空/强空），输出列加 trend / trend_prob
    pred_df = _score_to_trend(pred_df)
    return pred_df


# ---------- LSTM 超参搜索 ----------

_SEARCH_GRID = {
    "hidden_size": [128, 256, 512],
    "num_layers": [2, 3, 4],
    "lr": [1e-3, 5e-4],
    "batch_size": [2000, 4000],
}


def search_lstm(
    universe: str = "csi300",
    horizon: int = 5,
    train_start: str = "2015-01-01",
    train_end: str = "2020-08-31",
    valid_start: str = "2020-09-01",
    valid_end: str = "2020-09-25",
    device: str = "cuda",
    fast: bool = True,
    auto_train: bool = False,
    algo: str = "gru",
) -> list[dict]:
    """网格搜索 LSTM/GRU 超参，每组合跑短训练（max_steps=50）。
    
    auto_train=True 时，搜索完成后自动用最佳参数全量训练并激活。

    Returns:
        [{"hidden_size":128, "num_layers":2, "lr":0.001, "batch_size":2000,
          "ic":0.12, "epochs":23}, ...]
    """
    _qlib_init()
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset import DatasetH

    # handler（支持 csi300/csi500/all/watchlist）
    if universe not in ("csi300", "csi500", "all", "csi800", "watchlist"):
        raise ValueError(f"universe {universe} 暂不支持，可选 csi300/csi500/all/watchlist")
    actual_univ = universe if universe != "all" else "csi500"  # all = csi500 + 中证1000
    learn_procs = learn_processors(rank_norm=False)
    label_expr = [f"Ref($close, -{horizon}) / Ref($close, -1) - 1"]
    handler = Alpha158(
        instruments=actual_univ,
        start_time=train_start, end_time=valid_end,
        fit_start_time=train_start, fit_end_time=train_end,
        infer_processors=infer_processors(),   # 统一成和训练/预测同一份
        learn_processors=learn_procs,
        label=label_expr,
    )

    segments = {"train": (train_start, train_end), "valid": (valid_start, valid_end)}
    dataset = DatasetH(handler=handler, segments=segments)

    results = []
    total = len(_SEARCH_GRID["hidden_size"]) * len(_SEARCH_GRID["num_layers"]) * len(_SEARCH_GRID["lr"]) * len(_SEARCH_GRID["batch_size"])
    idx = 0
    for hidden_size in _SEARCH_GRID["hidden_size"]:
        for num_layers in _SEARCH_GRID["num_layers"]:
            for lr in _SEARCH_GRID["lr"]:
                for batch_size in _SEARCH_GRID["batch_size"]:
                    idx += 1
                    max_steps = 50 if fast else 200
                    early_stop = 10 if fast else 20
                    print(f"[{idx}/{total}] hidden={hidden_size} layers={num_layers} lr={lr} batch={batch_size}", flush=True)
                    try:
                        cell = "lstm" if algo == "lstm" else "gru"
                        model = RecurrentAlphaNet(
                            input_dim=158, seq_len=6, input_size=26,
                            hidden_size=hidden_size, num_layers=num_layers,
                            cell_type=cell,
                            lr=lr, max_steps=max_steps, batch_size=batch_size,
                            device=device, use_scheduler=True,
                        )
                        train_data = dataset.prepare("train", col_set=["feature", "label"])
                        valid_data = dataset.prepare("valid", col_set=["feature", "label"])
                        x_train, y_train = train_data["feature"], train_data["label"]
                        x_valid, y_valid = valid_data["feature"], valid_data["label"]
                        if hasattr(y_train, "values") and y_train.ndim > 1:
                            y_train = y_train.squeeze()
                        if hasattr(y_valid, "values") and y_valid.ndim > 1:
                            y_valid = y_valid.squeeze()
                        model.fit(x_train, y_train, x_valid, y_valid, early_stop=early_stop)
                        ic = float(model.best_score)
                        results.append({
                            "hidden_size": hidden_size, "num_layers": num_layers,
                            "lr": lr, "batch_size": batch_size,
                            "ic": ic, "epochs": model.best_step + 1,
                        })
                        print(f"  ✓ IC={ic:+.4f} @step {model.best_step+1}", flush=True)
                    except Exception as e:
                        print(f"  ✗ FAIL: {repr(e)[:100]}", flush=True)

    results.sort(key=lambda r: r["ic"], reverse=True)
    print(f"\n{'='*60}")
    print(f"  搜索完成 {len(results)}/{total} 组合")
    print("  Top3：")
    for i, r in enumerate(results[:3]):
        print(f"  #{i+1}: hidden={r['hidden_size']} layers={r['num_layers']} "
              f"lr={r['lr']} batch={r['batch_size']}  IC={r['ic']:+.4f}")
    print(f"{'='*60}\n")

    if auto_train and results:
        best = results[0]
        hs, nl, lr, bs = best["hidden_size"], best["num_layers"], best["lr"], best["batch_size"]
        best_step = best["epochs"]
        print(f"自动训练最佳参数: hidden={hs} layers={nl} lr={lr} batch={bs}  (搜索 best_step={best_step})", flush=True)
        model = RecurrentAlphaNet(
            input_dim=158, seq_len=6, input_size=26,
            hidden_size=hs, num_layers=nl, cell_type="lstm" if algo == "lstm" else "gru",
            lr=lr, max_steps=best_step + 10, batch_size=bs, device=device,  # 不多跑，留 10 步余量
            use_scheduler=True,
        )
        train_data = dataset.prepare("train", col_set=["feature", "label"])
        valid_data = dataset.prepare("valid", col_set=["feature", "label"])
        x_tr, y_tr = train_data["feature"], train_data["label"]
        x_va, y_va = valid_data["feature"], valid_data["label"]
        if hasattr(y_tr, "values") and y_tr.ndim > 1:
            y_tr = y_tr.squeeze()
        if hasattr(y_va, "values") and y_va.ndim > 1:
            y_va = y_va.squeeze()
        model.fit(x_tr, y_tr, x_va, y_va, early_stop=20)
        ic_full = float(model.best_score)
        print(f"\n全量训练完成: IC={ic_full:+.4f} (搜索阶段 IC={best['ic']:+.4f})", flush=True)

    return results
