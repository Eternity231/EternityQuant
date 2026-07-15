"""港股数据管道（akshare Sina 源）。全套：下载 → 特征计算 → 训练 → 预测。

数据源限制（大陆网络）：
- Sina 源 stock_hk_spot：2799 只约 83s（全市场快照）
- Sina 源 stock_hk_hist：单只日线约 2-5s
- 东财/腾讯/雅虎全被限流（RemoteDisconnected）

港股特征 ~60 维（MA/MACD/布林/KDJ/RSI/波动率/成交量等），
重塑成 (seq=6, dim=10) 喂给 _SimpleSeqModel(GRU)。
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from eq.data.paths import (
    HK_DAILY_DIR as _HK_DAILY_DIR,
    HK_5M_DIR as _HK_5M_DIR,
    HK_1M_DIR as _HK_1M_DIR,
    HK_FEAT_DIR as _HK_FEAT_DIR,
    HK_MODELS_DIR as _HK_MODELS_DIR,
    ensure_data_dirs,
)


def _ensure_dirs() -> None:
    ensure_data_dirs()


def _dl_one(code: str, start: str, end: str) -> tuple:
    """模块级工作函数，失败自动重试 3 次（Sina 源限流常见）。"""
    import time as _t
    for attempt in range(3):
        try:
            if attempt > 0:
                _t.sleep(3 * attempt)  # 退避 3s, 6s
            df = download_hk_stock(code, start, end)
            if df is not None and len(df) > 5:
                return (code, len(df))
        except:
            pass
    return (code, 0)


# ========== 第 1 步：数据下载 ==========

def list_hk_stocks(limit: int = 200) -> list[str]:
    """拉港股列表（Sina 源，全量 2799 只约 83s，默认取前 200 烱门）。"""
    import akshare as ak
    try:
        df = ak.stock_hk_spot()
        if df.empty:
            return []
        code_col = "代码" if "代码" in df.columns else df.columns[0]
        codes = df[code_col].astype(str).str.zfill(5).tolist()[:limit]
        return codes
    except Exception:
        return []


def parse_hk_codes_from_file(path: str | Path, verbose: bool = True) -> list[str]:
    """从混合品种表文件中解析出港股代码（5 位数字）。

    支持 Table.txt 这种「代码 制表符 名称」格式，自动剔除：
    - A 股（SH/SZ 前缀）
    - 美股（纯字母代码）
    - 指数/ETF/外汇（字母混合）
    - 已带 -W 后缀的二次上市标记

    Returns:
        5 位港股代码列表（去重保序），如 ["00700", "09988", ...]
    """
    import re
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"品种表文件不存在: {p}")

    codes: list[str] = []
    seen: set[str] = set()
    # 港股代码：5 位纯数字（akshare stock_hk_daily 接收的就是 5 位字符串）
    hk_pattern = re.compile(r"^\d{5}$")

    # 港 A 品种表常见 GBK 编码，先读 bytes 再试解码
    raw_bytes = p.read_bytes()
    text: str | None = None
    for enc in ("utf-8", "gbk", "gb18030", "latin-1"):
        try:
            text = raw_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise RuntimeError(f"无法解码品种表文件: {p}")

    for line in text.splitlines():
            # 分割：制表符或空白
            parts = re.split(r"[\s\t]+", line.strip())
            if not parts:
                continue
            raw = parts[0].strip()
            # 剔除明显带市场前缀的（SH/SZ/BJ）或字母代码（美股/指数/ETF/外汇）
            if not raw:
                continue
            if any(raw.startswith(pfx) for pfx in ("SH", "SZ", "BJ", "sh", "sz", "bj")):
                continue
            if not raw.isdigit():
                continue
            # 5 位数字 → 港股
            if hk_pattern.match(raw):
                if raw not in seen:
                    seen.add(raw)
                    codes.append(raw)

    if verbose:
        print(f"  品种表 {p.name}: 解析出 {len(codes)} 只港股  前 5: {codes[:5]}", flush=True)
    return codes


def download_hk_stock(symbol: str, start: str, end: str) -> pd.DataFrame:
    """拉单只港股日线（Sina 源）。存到本地 CSV 缓存。symbol：00700（纯代码）。"""
    import akshare as ak
    _ensure_dirs()
    cache_path = _HK_FEAT_DIR / f"{symbol}.csv"
    # 检查缓存（避免反复拉 2-5s/只）
    if cache_path.exists():
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        if len(df) > 10:
            # 只要缓存有数据就返回，不要求完全覆盖区间
            return df
    try:
        df = ak.stock_hk_hist(symbol=symbol, period="daily", start_date=start, end_date=end, adjust="qfq")
        if df.empty:
            return pd.DataFrame()
        col_map = {"开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume"}
        df = df.rename(columns=col_map)
        df = df.set_index("日期")
        df.index = pd.to_datetime(df.index)
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        df.to_csv(cache_path)
        return df
    except Exception:
        # 下载失败时，如果有缓存就返回缓存数据
        if cache_path.exists():
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if len(df) > 10:
                return df
        return pd.DataFrame()


def update_hk_data(
    start: str | None = None,
    end: str | None = None,
    top_n: int = 200,
    workers: int = 3,
    codes: list[str] | None = None,
    verbose: bool = True,
) -> dict:
    """下载港股日线数据到本地缓存。

    Args:
        codes: 显式指定代码列表（5 位数字字符串）。传 None 则用 list_hk_stocks 拉热门。
    Returns:
        {"codes": int, "days": int, "cache_dir": str}
    """
    if end is None:
        end = dt.date.today().isoformat()
    if start is None:
        start = (dt.date.today() - dt.timedelta(days=730)).isoformat()  # 默认 2 年

    if codes is None:
        codes = list_hk_stocks(limit=top_n)
    else:
        # 显式清单：补 0 到 5 位（如 "9988" → "09988"）
        codes = [str(c).strip().zfill(5) for c in codes if str(c).strip()]
        if top_n > 0:
            codes = codes[:top_n]

    if not codes:
        return {"codes": 0, "days": 0, "cache_dir": str(_HK_FEAT_DIR)}

    import time as _t, multiprocessing as _mp

    _t0 = _t.time()
    ok = 0
    with _mp.Pool(processes=workers) as pool:
        results = pool.starmap(_dl_one, [(code, start, end) for code in codes])
        total_days = 0
        for i, (code, days) in enumerate(results):
            if days > 0:
                ok += 1
                total_days += days
            if verbose and (i + 1) % 20 == 0:
                _pct = (i + 1) / len(codes) * 100
                _elapsed = _t.time() - _t0
                print(f"\r  进度 {_pct:5.1f}%  ({i+1}/{len(codes)})  ✓{ok}  已用 {_elapsed:.0f}s", end="", flush=True)
    if verbose:
        print(f"\n港股数据下载完成：{ok}/{len(codes)} 只 ✓，共约 {total_days} 日线", flush=True)

    return {"codes": ok, "days": total_days, "cache_dir": str(_HK_FEAT_DIR)}


# ========== 第 2 步：特征计算（~60 维）==========

def compute_features_hk(df: pd.DataFrame) -> pd.DataFrame:
    """港股 ~60 维技术特征，替代 Alpha158（港股无法用 qlib）。

    分组（可重塑为 seq=6, dim=10）：
    - 价格动量（12 维）：ret1/2/3/5/10/20 + 相对 MA 位置
    - 均线（6 维）：ma3/5/10/20/60 + 均线条数
    - MACD（3 维）：dif/dea/hist
    - 布林带（3 维）：upper/lower/width
    - 波动率（5 维）：atr + volatility5/10/20 + 归一化
    - 成交量（5 维）：volume_ma5/20 + volume_ratio + 量价相关
    - 震荡指标（6 维）：rsi14 + k/d + 乖离率
    - 形态（~20 维）：高低价差/上影/下影/各种比
    """
    df = df.copy()
    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"]

    # --- 价格动量 ---
    for p in [1, 2, 3, 5, 10, 20]:
        df[f"ret{p}"] = close.pct_change(p)
    df["close_ma5"] = close / close.rolling(5).mean()
    df["close_ma10"] = close / close.rolling(10).mean()
    df["close_ma20"] = close / close.rolling(20).mean()
    df["close_ma60"] = close / close.rolling(60).mean()

    # --- 均线 ---
    for p in [3, 5, 10, 20, 60]:
        df[f"ma{p}"] = close.rolling(p).mean()
    df["ma_cross"] = df["ma5"] - df["ma20"]  # 金叉死叉信号

    # --- MACD ---
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df["dif"] = ema12 - ema26
    df["dea"] = df["dif"].ewm(span=9).mean()
    df["hist"] = 2 * (df["dif"] - df["dea"])

    # --- 布林带 ---
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["bb_upper"] = (ma20 + 2 * std20) / close - 1
    df["bb_lower"] = (ma20 - 2 * std20) / close - 1
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (ma20 / close + 1)

    # --- 波动率 ---
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean() / close
    ret1 = close.pct_change()
    for p in [5, 10, 20]:
        df[f"volatility{p}"] = ret1.rolling(p).std() * np.sqrt(p)

    # --- 成交量 ---
    for p in [5, 20]:
        df[f"vma{p}"] = vol.rolling(p).mean()
    df["v_ratio"] = vol / df["vma5"]
    df["v_price_corr"] = vol.rolling(10).corr(close)  # 量价相关性

    # --- 震荡指标 ---
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["rsi14"] = 100 - 100 / (1 + gain / (loss + 1e-10))

    # KDJ
    llv = low.rolling(9).min()
    hhv = high.rolling(9).max()
    rsv = (close - llv) / (hhv - llv + 1e-10) * 100
    df["k"] = rsv.ewm(span=3).mean()
    df["d"] = df["k"].ewm(span=3).mean()
    df["j"] = 3 * df["k"] - 2 * df["d"]

    # 乖离率
    df["bias5"] = (close - df["ma5"]) / df["ma5"]
    df["bias10"] = (close - df["ma10"]) / df["ma10"]

    # --- 形态特征 ---
    df["hl_ratio"] = (high - low) / close
    df["co_ratio"] = (close - open_) / (high - low + 1e-10)  # 实体占比
    df["up_shadow"] = (high - close) / (close - low + 1e-10)  # 上影线
    df["low_shadow"] = (open_ - low) / (high - low + 1e-10)  # 下影线
    df["body"] = abs(close - open_) / (high - low + 1e-10)  # 实体比例
    # 连续涨跌
    df["up_count"] = (ret1 > 0).astype(int).rolling(5).sum()
    df["down_count"] = (ret1 < 0).astype(int).rolling(5).sum()

    return df.dropna()


# ========== 第 3 步：训练 ==========

def train_hk(
    symbols: list[str] | None = None,
    top_n: int = 100,
    start: str | None = None,
    end: str | None = None,
    horizon: int = 5,
    hidden_size: int = 128,
    num_layers: int = 2,
    cell_type: str = "gru",
    batch_size: int = 512,
    max_steps: int = 200,
    device: str = "cuda",
    dropout: float = 0.3,
    walk_forward: bool = True,
    name: str | None = None,
    verbose: bool = True,
    gpu_ids: str | list[int] | None = None,  # 多卡并行
) -> dict:
    """港股 GRU 训练（不走 qlib，自写特征 + _SimpleSeqModel）。

    walk_forward=True 时用滚动前向验证（Walk-Forward Validation），
    每 60 天滚动一次，模拟实盘。

    Returns:
        {"model_id": str, "ic": float, "model_path": str}
    """
    if end is None:
        end = dt.date.today().isoformat()
    if start is None:
        start = (dt.date.today() - dt.timedelta(days=730)).isoformat()

    # 获取股票列表
    if symbols is None:
        symbols = list_hk_stocks(limit=top_n)
    if not symbols:
        raise RuntimeError("无港股数据，先跑 eq hk update-data")

    # 下载 + 算特征 + 构训练集
    all_features = []
    all_labels = []
    symbols_ok = []

    for code in symbols:
        df = download_hk_stock(code, start, end)
        if df.empty or len(df) < 120:
            continue
        feat_df = compute_features_hk(df)
        if feat_df.empty:
            continue
        # label：horizon 日后收益
        feat_df["label"] = feat_df["close"].shift(-horizon) / feat_df["close"] - 1
        feat_df = feat_df.dropna()
        if len(feat_df) < 60:
            continue
        # 特征列（排除 price/volume 原始列 + label）
        exclude = {"open", "high", "low", "close", "volume", "label", "vma5", "vma20"}
        feat_cols = [c for c in feat_df.columns if c not in exclude]
        # 每只票取最近 time_steps 天的特征（时序样本）
        time_steps = 6
        feat_dim = len(feat_cols)
        # 切分成滑动窗口样本
        for i in range(time_steps, len(feat_df)):
            all_features.append(feat_df[feat_cols].iloc[i - time_steps:i].values.flatten())
            all_labels.append(feat_df["label"].iloc[i])
        symbols_ok.append(code)

    if not all_features:
        raise RuntimeError(f"特征计算后无有效样本（{len(symbols)} 只股票）")

    # 转换为 numpy 数组
    import numpy as _np
    X = _np.array(all_features, dtype=_np.float32)
    y = _np.array(all_labels, dtype=_np.float32)

    seq_len = time_steps
    input_size = len(feat_cols)

    # 导入模型
    from eq.strategy.factors.ml_workflow import _SimpleSeqModel

    # Walk-Forward Validation：滚动 60 天窗口，每滚一次训一次，取平均 IC
    if walk_forward and len(X) > 240:
        window = 60  # 验证窗口 60 天
        step = 30    # 每 30 天滚一次
        wf_ics = []
        if verbose:
            print(f"  Walk-Forward Validation: 窗口={window}天 步长={step}天", flush=True)
        for wf_start in range(window, len(X) - window, step):
            wf_train_x = X[:wf_start]
            wf_train_y = y[:wf_start]
            wf_valid_x = X[wf_start:wf_start + window]
            wf_valid_y = y[wf_start:wf_start + window]
            if len(wf_train_x) < 120 or len(wf_valid_x) < 10:
                continue
            wf_model = _SimpleSeqModel(
                input_dim=seq_len * input_size, seq_len=seq_len, input_size=input_size,
                hidden_size=hidden_size, num_layers=num_layers, cell_type=cell_type,
                lr=1e-3, max_steps=100, batch_size=batch_size,
                device=device, dropout=dropout, use_scheduler=True,
            )
            wf_model.fit(wf_train_x, wf_train_y, wf_valid_x, wf_valid_y, early_stop=15)
            wf_ics.append(float(wf_model.best_score))
        if wf_ics:
            avg_ic = sum(wf_ics) / len(wf_ics)
            if verbose:
                print(f"  Walk-Forward IC: mean={avg_ic:+.4f}  "
                      f"min={min(wf_ics):+.4f}  max={max(wf_ics):+.4f}  "
                      f"({len(wf_ics)} 窗口)", flush=True)

    # 固定切分验证（与 Walk-Forward 对比）
    split = int(len(X) * 0.8)
    x_train, y_train = X[:split], y[:split]
    x_valid, y_valid = X[split:], y[split:]

    if verbose:
        print(f"港股数据集：{len(x_train)} 训练 + {len(x_valid)} 验证  "
              f"（{len(symbols_ok)} 只股票，{len(feat_cols)} 维特征，dropout={dropout}）", flush=True)

    # 训练
    model = _SimpleSeqModel(
        input_dim=seq_len * input_size, seq_len=seq_len, input_size=input_size,
        hidden_size=hidden_size, num_layers=num_layers, cell_type=cell_type,
        lr=1e-3, max_steps=max_steps, batch_size=batch_size,
        device=device, dropout=dropout, use_scheduler=True,
    )
    model.fit(x_train, y_train, x_valid, y_valid, early_stop=20)
    ic = float(model.best_score)

    # 存盘
    import pickle as _pkl
    _ensure_dirs()
    model_name = name or f"hk_{cell_type}_h{horizon}_{dt.date.today().strftime('%Y%m%d')}"
    model_path = _HK_MODELS_DIR / f"{model_name}.pkl"
    with open(model_path, "wb") as f:
        _pkl.dump(model, f)

    # 登记 ml_models 表（复用 A 股的表结构）
    name_field = model_name

    result = {
        "model_name": name_field,
        "ic": ic,
        "model_path": str(model_path),
        "symbols": len(symbols_ok),
        "train_samples": len(x_train),
    }
    if verbose:
        print(f"\n港股训练完成：IC={ic:+.4f}  {len(symbols_ok)} 只股票  {model_path.name}", flush=True)
    return result


# ========== 第 4 步：预测 ==========

def predict_hk_top(
    model_path: str,
    symbols: list[str] | None = None,
    top_n: int = 10,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """用已训练模型跑港股批量预测，返回 TopN。

    Returns:
        DataFrame [symbol, score], 按 score 降序
    """
    if end is None:
        end = dt.date.today().isoformat()
    if start is None:
        start = (dt.date.today() - dt.timedelta(days=90)).isoformat()
    if symbols is None:
        symbols = list_hk_stocks(limit=200)

    import pickle as _pkl
    with open(model_path, "rb") as f:
        model = _pkl.load(f)

    results = []
    for code in symbols:
        df = download_hk_stock(code, start, end)
        if df.empty or len(df) < model.seq_len + 10:
            continue
        feat_df = compute_features_hk(df)
        if feat_df.empty:
            continue
        exclude = {"open", "high", "low", "close", "volume", "label", "vma5", "vma20"}
        feat_cols = [c for c in feat_df.columns if c not in exclude]
        # 取最后 seq_len 行
        last_feats = feat_df[feat_cols].iloc[-model.seq_len:].values.flatten()
        if len(last_feats) != model.seq_len * len(feat_cols):
            continue
        score = model.predict(pd.DataFrame([last_feats]))[0]
        results.append({"symbol": f"{code}.HK", "score": float(score)})

    df_result = pd.DataFrame(results).sort_values("score", ascending=False).head(top_n).reset_index(drop=True)
    return df_result


# ============================================================================
# 方案 A：多频率分别训练 + 集成预测
# ============================================================================

def _load_hk_cache(code: str, freq: str = "daily") -> pd.DataFrame:
    """读本地港股缓存 CSV，不存在返回空 DataFrame。

    Args:
        freq: daily | 5m | 1m
    """
    import os
    if freq == "daily":
        path = _HK_DAILY_DIR / f"{code}.csv"
    elif freq == "5m":
        path = _HK_5M_DIR / f"{code}.csv"
    elif freq == "1m":
        path = _HK_1M_DIR / f"{code}.csv"
    else:
        return pd.DataFrame()
    if not path.exists() or path.stat().st_size < 100:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if df.empty or len(df) < 10:
            return pd.DataFrame()
        # 统一列名（yfinance 下发的是首字母大写 Open/Close）
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
        })
        need = {"open", "high", "low", "close", "volume"}
        if not need.issubset(set(df.columns)):
            return pd.DataFrame()
        return df[["open", "high", "low", "close", "volume"]].astype(float)
    except Exception:
        return pd.DataFrame()


def compute_features_hk_minute(df: pd.DataFrame, freq: str = "5m") -> pd.DataFrame:
    """港股分钟线特征（~30 维，适配分钟尺度）。

    与日线特征 compute_features_hk 的区别：
    - 均线窗口按分钟尺度调短（MA3/5/10/30）
    - MACD 用 12/26/9（分钟级常用）
    - RSI/KDJ 保留，但滚动窗口缩短
    - 剔除 MA60 等长窗口（分钟线没有足够样本）

    Args:
        freq: 5m | 1m（仅影响标签 horizon 的物理含义）
    """
    df = df.copy()
    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"]

    # --- 价格动量 ---
    for p in [1, 2, 3, 5, 10, 30]:
        df[f"ret{p}"] = close.pct_change(p)
    df["close_ma5"] = close / close.rolling(5).mean()
    df["close_ma10"] = close / close.rolling(10).mean()
    df["close_ma30"] = close / close.rolling(30).mean()

    # --- 均线 ---
    for p in [3, 5, 10, 30]:
        df[f"ma{p}"] = close.rolling(p).mean()
    df["ma_cross"] = df["ma5"] - df["ma10"]

    # --- MACD（12/26/9 与日线同尺度） ---
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df["dif"] = ema12 - ema26
    df["dea"] = df["dif"].ewm(span=9).mean()
    df["hist"] = 2 * (df["dif"] - df["dea"])

    # --- 布林带 ---
    ma10 = close.rolling(10).mean()
    std10 = close.rolling(10).std()
    df["bb_upper"] = (ma10 + 2 * std10) / close - 1
    df["bb_lower"] = (ma10 - 2 * std10) / close - 1
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (ma10 / close + 1)

    # --- 波动率 ---
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(10).mean() / close
    ret1 = close.pct_change()
    for p in [5, 10, 30]:
        df[f"volatility{p}"] = ret1.rolling(p).std() * np.sqrt(p)

    # --- 成交量 ---
    for p in [5, 30]:
        df[f"vma{p}"] = vol.rolling(p).mean()
    df["v_ratio"] = vol / df["vma5"]
    df["v_price_corr"] = vol.rolling(10).corr(close)

    # --- 邇荡指标 ---
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["rsi14"] = 100 - 100 / (1 + gain / (loss + 1e-10))

    llv = low.rolling(9).min()
    hhv = high.rolling(9).max()
    rsv = (close - llv) / (hhv - llv + 1e-10) * 100
    df["k"] = rsv.ewm(span=3).mean()
    df["d"] = df["k"].ewm(span=3).mean()

    # --- 形态特征 ---
    df["hl_ratio"] = (high - low) / close
    df["co_ratio"] = (close - open_) / (high - low + 1e-10)
    df["up_shadow"] = (high - close) / (close - low + 1e-10)
    df["body"] = abs(close - open_) / (high - low + 1e-10)

    return df.dropna()


def train_hk_minute(
    symbols: list[str] | None = None,
    top_n: int = 100,
    freq: str = "5m",           # 5m | 1m
    horizon: int = 30,          # 分钟线预测窗口：5m×30=2.5h, 1m×30=30min
    hidden_size: int = 128,
    num_layers: int = 2,
    cell_type: str = "gru",
    batch_size: int = 512,
    max_steps: int = 200,
    device: str = "cuda",
    dropout: float = 0.3,
    walk_forward: bool = True,
    name: str | None = None,
    verbose: bool = True,
    gpu_ids: str | list[int] | None = None,
) -> dict:
    """港股分钟线 GRU 训练（走本地缓存 5m/1m 目录，不走网络下载）。

    Args:
        freq: 5m | 1m
        horizon: 预测窗口（根根频率单位为根数）
    Returns:
        {"model_name": str, "ic": float, "model_path": str, "symbols": int, "train_samples": int}
    """
    if symbols is None:
        # 从日线缓存目录反推票池（确保日线有该票）
        symbols = sorted({f.stem for f in _HK_DAILY_DIR.glob("*.csv") if f.stat().st_size > 1000})
        if not symbols:
            raise RuntimeError("无港股日线缓存，先跑 eq hk update-data")
    symbols = symbols[:top_n]

    all_features: list = []
    all_labels: list = []
    symbols_ok: list[str] = []

    for code in symbols:
        df = _load_hk_cache(code, freq=freq)
        if df.empty or len(df) < 120:
            continue
        feat_df = compute_features_hk_minute(df, freq=freq)
        if feat_df.empty:
            continue
        # 标签：horizon 根分钟后的收益
        feat_df["label"] = feat_df["close"].shift(-horizon) / feat_df["close"] - 1
        feat_df = feat_df.dropna()
        if len(feat_df) < 60:
            continue

        exclude = {"open", "high", "low", "close", "volume", "label", "vma5", "vma30"}
        feat_cols = [c for c in feat_df.columns if c not in exclude]
        time_steps = 6
        for i in range(time_steps, len(feat_df)):
            all_features.append(feat_df[feat_cols].iloc[i - time_steps:i].values.flatten())
            all_labels.append(feat_df["label"].iloc[i])
        symbols_ok.append(code)

    if not all_features:
        raise RuntimeError(f"分钟线({freq})特征计算后无有效样本（{len(symbols)} 只股票）")

    import numpy as _np
    X = _np.array(all_features, dtype=_np.float32)
    y = _np.array(all_labels, dtype=_np.float32)

    seq_len = time_steps
    input_size = len(feat_cols)

    from eq.strategy.factors.ml_workflow import _SimpleSeqModel

    # Walk-Forward Validation
    if walk_forward and len(X) > 240:
        window = 60
        step = 30
        wf_ics = []
        if verbose:
            print(f"  Walk-Forward Validation({freq}): 窗口={window} 步长={step}", flush=True)
        for wf_start in range(window, len(X) - window, step):
            wf_train_x = X[:wf_start]
            wf_train_y = y[:wf_start]
            wf_valid_x = X[wf_start:wf_start + window]
            wf_valid_y = y[wf_start:wf_start + window]
            if len(wf_train_x) < 120 or len(wf_valid_x) < 10:
                continue
            wf_model = _SimpleSeqModel(
                input_dim=seq_len * input_size, seq_len=seq_len, input_size=input_size,
                hidden_size=hidden_size, num_layers=num_layers, cell_type=cell_type,
                lr=1e-3, max_steps=100, batch_size=batch_size,
                device=device, dropout=dropout, use_scheduler=True,
            )
            wf_model.fit(wf_train_x, wf_train_y, wf_valid_x, wf_valid_y, early_stop=15)
            wf_ics.append(float(wf_model.best_score))
        if wf_ics and verbose:
            avg_ic = sum(wf_ics) / len(wf_ics)
            print(f"  Walk-Forward IC({freq}): mean={avg_ic:+.4f}  ({len(wf_ics)} 窗口)", flush=True)

    # 固定切分验证
    split = int(len(X) * 0.8)
    x_train, y_train = X[:split], y[:split]
    x_valid, y_valid = X[split:], y[split:]

    if verbose:
        print(f"分钟线数据集({freq})：{len(x_train)} 训练 + {len(x_valid)} 验证  "
              f"（{len(symbols_ok)} 只，{len(feat_cols)} 维特征）", flush=True)

    model = _SimpleSeqModel(
        input_dim=seq_len * input_size, seq_len=seq_len, input_size=input_size,
        hidden_size=hidden_size, num_layers=num_layers, cell_type=cell_type,
        lr=1e-3, max_steps=max_steps, batch_size=batch_size,
        device=device, dropout=dropout, use_scheduler=True,
    )
    model.fit(x_train, y_train, x_valid, y_valid, early_stop=20)
    ic = float(model.best_score)

    import pickle as _pkl
    _ensure_dirs()
    model_name = name or f"hk_{cell_type}_{freq}_h{horizon}_{dt.date.today().strftime('%Y%m%d')}"
    model_path = _HK_MODELS_DIR / f"{model_name}.pkl"
    with open(model_path, "wb") as f:
        _pkl.dump(model, f)

    result = {
        "model_name": model_name,
        "ic": ic,
        "model_path": str(model_path),
        "symbols": len(symbols_ok),
        "train_samples": len(x_train),
        "freq": freq,
        "horizon": horizon,
    }
    if verbose:
        print(f"\n分钟线训练完成({freq})：IC={ic:+.4f}  {len(symbols_ok)} 只  {model_path.name}", flush=True)
    return result


def predict_hk_ensemble(
    model_daily: str,
    model_5m: str | None = None,
    model_1m: str | None = None,
    symbols: list[str] | None = None,
    top_n: int = 10,
    weights: dict | None = None,
    lookback_days: int = 90,
) -> pd.DataFrame:
    """港股多频率集成预测（方案 A 核心）。

    Args:
        model_daily: 日线模型路径（必传）
        model_5m: 5 分钟线模型路径（可选）
        model_1m: 1 分钟线模型路径（可选）
        weights: {"daily": w1, "5m": w2, "1m": w3} 加权集成；默认均等
    Returns:
        DataFrame [symbol, score_daily, score_5m, score_1m, score]，按 score 降序
    """
    import pickle as _pkl

    if symbols is None:
        symbols = sorted({f.stem for f in _HK_DAILY_DIR.glob("*.csv") if f.stat().st_size > 1000})
        if not symbols:
            raise RuntimeError("无港股日线缓存，先跑 eq hk update-data")

    if weights is None:
        weights = {"daily": 1.0, "5m": 1.0 if model_5m else 0.0, "1m": 1.0 if model_1m else 0.0}
    total_w = sum(weights.values())
    if total_w <= 0:
        raise RuntimeError("集成权重全为 0")

    models: dict[str, object] = {}
    for key, path in [("daily", model_daily), ("5m", model_5m), ("1m", model_1m)]:
        if path and Path(path).exists():
            with open(path, "rb") as f:
                models[key] = _pkl.load(f)

    if "daily" not in models:
        raise RuntimeError(f"日线模型必传且必须存在: {model_daily}")

    start = (dt.date.today() - dt.timedelta(days=lookback_days)).isoformat()
    end = dt.date.today().isoformat()

    results: list[dict] = []
    for code in symbols:
        row: dict = {"symbol": f"{code}.HK"}
        valid_any = False

        # 日线分支（缓存不够时在线补拉）
        if "daily" in models:
            df = _load_hk_cache(code, freq="daily")
            if not df.empty and len(df) < 120:
                try:
                    df = download_hk_stock(code, start, end)
                except Exception:
                    pass
            if not df.empty and len(df) >= 60:
                feat_df = compute_features_hk(df)
                if not feat_df.empty:
                    exclude = {"open", "high", "low", "close", "volume", "label", "vma5", "vma20"}
                    feat_cols = [c for c in feat_df.columns if c not in exclude]
                    m = models["daily"]
                    last = feat_df[feat_cols].iloc[-m.seq_len:].values.flatten()
                    if len(last) == m.seq_len * len(feat_cols):
                        row["score_daily"] = float(m.predict(pd.DataFrame([last]))[0])
                        valid_any = True

        # 5 分钟线分支
        if "5m" in models:
            df = _load_hk_cache(code, freq="5m")
            if not df.empty and len(df) >= 60:
                feat_df = compute_features_hk_minute(df, freq="5m")
                if not feat_df.empty:
                    exclude = {"open", "high", "low", "close", "volume", "label", "vma5", "vma30"}
                    feat_cols = [c for c in feat_df.columns if c not in exclude]
                    m = models["5m"]
                    last = feat_df[feat_cols].iloc[-m.seq_len:].values.flatten()
                    if len(last) == m.seq_len * len(feat_cols):
                        row["score_5m"] = float(m.predict(pd.DataFrame([last]))[0])
                        valid_any = True

        # 1 分钟线分支
        if "1m" in models:
            df = _load_hk_cache(code, freq="1m")
            if not df.empty and len(df) >= 60:
                feat_df = compute_features_hk_minute(df, freq="1m")
                if not feat_df.empty:
                    exclude = {"open", "high", "low", "close", "volume", "label", "vma5", "vma30"}
                    feat_cols = [c for c in feat_df.columns if c not in exclude]
                    m = models["1m"]
                    last = feat_df[feat_cols].iloc[-m.seq_len:].values.flatten()
                    if len(last) == m.seq_len * len(feat_cols):
                        row["score_1m"] = float(m.predict(pd.DataFrame([last]))[0])
                        valid_any = True

        if valid_any:
            results.append(row)

    if not results:
        return pd.DataFrame(columns=["symbol", "score"])

    df = pd.DataFrame(results)

    def _w(row, key):
        v = row.get(f"score_{key}")
        return v if pd.notna(v) else 0.0

    df["score"] = 0.0
    for key in ("daily", "5m", "1m"):
        if key in models:
            df["score"] += df.apply(lambda r: _w(r, key) * weights[key], axis=1)
    df["score"] = df["score"] / total_w

    cols = ["symbol", "score"]
    for k in ("daily", "5m", "1m"):
        if k in models:
            cols.insert(cols.index("score"), f"score_{k}")
    df = df[cols].sort_values("score", ascending=False).head(top_n).reset_index(drop=True)
    return df
