"""Alpha158 特征集本地实现（v0.39）—— 脱离 qlib 的最后一块。

qlib 在本项目里最后的角色就是算这 158 个特征。接管之后，训练链路**直接吃
项目自己的 OHLCV DataFrame**，不再需要 qlib 的 .bin 数据、表达式引擎和
那几个绕 bug 的 monkey patch。

## 输入输出

输入 ``{symbol: OHLCV DataFrame}``（就是 ``get_recent_bars`` 的返回格式），
输出带 ``(datetime, instrument)`` MultiIndex 的特征面板——和
:mod:`eq.strategy.factors.preprocess` / :mod:`evaluation` 直接对接。

## 关于「和 qlib 是否逐位一致」

**没有对拍过**：开发机上没装 qlib，qlib 数据目录也是空的。下面每个公式都按
qlib Alpha158 的**公开定义**实现，并在 docstring 里写清了口径，但不保证逐位相同。
已知有意不同的地方在 :func:`_idx_max` 等处单独标注。

要对拍就在装了 qlib 的机器上跑 :func:`compare_with_qlib`。项目当前已注册模型
数为 0，所以即使有差异也不存在新旧模型可比性问题。

## 分组

- **KBAR（9 个）**：单根 K 线的形态，不含时序
- **价格（4 个）**：开高低和 VWAP 相对收盘价的位置
- **滚动（29 个 × 5 个窗口 = 145 个）**：窗口 ``[5, 10, 20, 30, 60]``

合计 158。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["WINDOWS", "alpha158", "alpha158_single", "feature_names", "compare_with_qlib"]

WINDOWS: tuple[int, ...] = (5, 10, 20, 30, 60)
_EPS = 1e-12


# ======================================================================
# 滚动算子
# ======================================================================

def _slope_parts(s: pd.Series, d: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """滚动线性回归（对时间下标）的中间量：斜率、y 均值、y 方差。

    自变量固定是窗口内的位置 0,1,…,d-1，所以它的均值/方差是常数，
    只有协方差需要滚动算。协方差用一个恒等式做到完全向量化：

        Σᵢ i·y  =  Σⱼ j·yⱼ − (n−d+1)·Σⱼ yⱼ        （j 是绝对下标）

    左边是窗口内位置加权和，右边两项都是普通 rolling sum。
    否则就得 ``rolling.apply(polyfit)``，在 158 特征 × 几百只票的规模上慢到没法用。
    """
    n = len(s)
    j = pd.Series(np.arange(n, dtype="float64"), index=s.index)
    sum_y = s.rolling(d).sum()
    sum_jy = (j * s).rolling(d).sum()
    # 窗口左端的绝对下标
    left = j - (d - 1)
    sum_iy = sum_jy - left * sum_y

    mean_x = (d - 1) / 2.0
    var_x = (d * d - 1) / 12.0
    mean_y = sum_y / d
    cov = sum_iy / d - mean_x * mean_y
    slope = cov / var_x
    var_y = s.rolling(d).var(ddof=0)
    return slope, mean_y, var_y


def _slope(s: pd.Series, d: int) -> pd.Series:
    return _slope_parts(s, d)[0]


def _rsquare(s: pd.Series, d: int) -> pd.Series:
    """滚动回归的 R²。y 无波动时（常数序列）回归无意义，记 0。"""
    slope, _, var_y = _slope_parts(s, d)
    var_x = (d * d - 1) / 12.0
    r2 = (slope ** 2) * var_x / var_y.replace(0, np.nan)
    return r2.clip(0.0, 1.0)


def _resi(s: pd.Series, d: int) -> pd.Series:
    """当前点相对滚动回归直线的残差。度量「有多偏离趋势」。"""
    slope, mean_y, _ = _slope_parts(s, d)
    # 当前点在窗口内的位置是 d-1，直线在该处的值 = mean_y + slope*(d-1-mean_x)
    return s - (mean_y + slope * ((d - 1) - (d - 1) / 2.0))


def _idx_max(s: pd.Series, d: int) -> pd.Series:
    """距窗口内最高点过去了几天（今天最高 = 0）。

    **口径说明**：Alpha158 对这个特征的描述是「当前日期与前一个最高价日期
    之间的天数」，所以这里返回的是**回溯距离**而不是 argmax 的原始下标。
    两者相差一个 ``d-1-·`` 的翻转；qlib 内部用哪个未经对拍确认，
    但「距离」的语义更符合它自己的文档，也更符合直觉（越小＝越新的高点）。
    """
    return s.rolling(d).apply(lambda a: d - 1 - int(np.argmax(a)), raw=True)


def _idx_min(s: pd.Series, d: int) -> pd.Series:
    return s.rolling(d).apply(lambda a: d - 1 - int(np.argmin(a)), raw=True)


def _rank_pct(s: pd.Series, d: int) -> pd.Series:
    """当前值在过去 d 天里的百分位（0~1）。"""
    return s.rolling(d).rank(pct=True)


def _corr(a: pd.Series, b: pd.Series, d: int) -> pd.Series:
    """滚动相关。任一边在窗口内是常数时相关无定义，置 NaN 交给后续 fillna。"""
    with np.errstate(invalid="ignore", divide="ignore"):
        out = a.rolling(d).corr(b)
    return out.replace([np.inf, -np.inf], np.nan)


# ======================================================================
# 特征构造
# ======================================================================

def _kbar(df: pd.DataFrame) -> dict[str, pd.Series]:
    """单根 K 线的形态特征（9 个），不含任何时序信息。

    两套分母：除以 ``open`` 得到相对开盘价的幅度，除以 ``high-low`` 得到
    在当日振幅中的占比。后者对不同波动水平的股票更可比。
    """
    o, h, low, c = df["open"], df["high"], df["low"], df["close"]
    hl = (h - low) + _EPS
    up = h - np.maximum(o, c)          # 上影线
    dn = np.minimum(o, c) - low        # 下影线
    return {
        "KMID": (c - o) / o,
        "KLEN": (h - low) / o,
        "KMID2": (c - o) / hl,
        "KUP": up / o,
        "KUP2": up / hl,
        "KLOW": dn / o,
        "KLOW2": dn / hl,
        "KSFT": (2 * c - h - low) / o,      # 收盘价在当日区间里偏上还是偏下
        "KSFT2": (2 * c - h - low) / hl,
    }


def _price(df: pd.DataFrame) -> dict[str, pd.Series]:
    """开/高/低/均价相对收盘价的位置（4 个）。除以 close 是为了去掉价格量纲。"""
    c = df["close"]
    out = {"OPEN0": df["open"] / c, "HIGH0": df["high"] / c, "LOW0": df["low"] / c}
    if "vwap" in df.columns:
        out["VWAP0"] = df["vwap"] / c
    elif "amount" in df.columns:
        # 没有 vwap 列时用成交额/成交量现算，比直接丢掉一个特征好
        vol = df["volume"].replace(0, np.nan)
        out["VWAP0"] = (df["amount"] / vol) / c
    else:
        # 连成交额都没有就退化成 (high+low+close)/3 的典型价
        out["VWAP0"] = ((df["high"] + df["low"] + c) / 3) / c
    return out


def _rolling(df: pd.DataFrame, d: int) -> dict[str, pd.Series]:
    """一个窗口下的 29 个滚动特征。

    绝大多数都除以了当前 ``close`` 或 ``volume``——这一步是**去量纲**，
    让 5 元的票和 500 元的票、成交量差三个数量级的票能放在同一个截面里比。
    不做的话模型会先学会「区分大盘股和小盘股」，那不是 alpha。
    """
    c, h, low, v = df["close"], df["high"], df["low"], df["volume"]
    c_prev = c.shift(1)
    dc = c - c_prev
    vol_safe = v + _EPS
    dv = v - v.shift(1)
    abs_dc_sum = dc.abs().rolling(d).sum() + _EPS
    abs_dv_sum = dv.abs().rolling(d).sum() + _EPS

    max_h, min_l = h.rolling(d).max(), low.rolling(d).min()
    sump = dc.clip(lower=0).rolling(d).sum() / abs_dc_sum
    sumn = (-dc).clip(lower=0).rolling(d).sum() / abs_dc_sum
    cntp = (c > c_prev).rolling(d).mean()
    cntn = (c < c_prev).rolling(d).mean()
    vsump = dv.clip(lower=0).rolling(d).sum() / abs_dv_sum
    vsumn = (-dv).clip(lower=0).rolling(d).sum() / abs_dv_sum
    # 成交额加权的波动：|涨跌幅| × 成交量，度量「放量时波动大不大」
    wv = (c / c_prev - 1).abs() * v

    return {
        f"ROC{d}": c.shift(d) / c,
        f"MA{d}": c.rolling(d).mean() / c,
        f"STD{d}": c.rolling(d).std() / c,
        f"BETA{d}": _slope(c, d) / c,
        f"RSQR{d}": _rsquare(c, d),
        f"RESI{d}": _resi(c, d) / c,
        f"MAX{d}": max_h / c,
        f"MIN{d}": min_l / c,
        f"QTLU{d}": c.rolling(d).quantile(0.8) / c,
        f"QTLD{d}": c.rolling(d).quantile(0.2) / c,
        f"RANK{d}": _rank_pct(c, d),
        f"RSV{d}": (c - min_l) / (max_h - min_l + _EPS),
        f"IMAX{d}": _idx_max(h, d) / d,
        f"IMIN{d}": _idx_min(low, d) / d,
        f"IMXD{d}": (_idx_max(h, d) - _idx_min(low, d)) / d,
        f"CORR{d}": _corr(c, np.log(vol_safe), d),
        f"CORD{d}": _corr(c / c_prev, np.log(v / v.shift(1) + 1), d),
        f"CNTP{d}": cntp,
        f"CNTN{d}": cntn,
        f"CNTD{d}": cntp - cntn,
        f"SUMP{d}": sump,
        f"SUMN{d}": sumn,
        f"SUMD{d}": sump - sumn,
        f"VMA{d}": v.rolling(d).mean() / vol_safe,
        f"VSTD{d}": v.rolling(d).std() / vol_safe,
        f"WVMA{d}": wv.rolling(d).std() / (wv.rolling(d).mean() + _EPS),
        f"VSUMP{d}": vsump,
        f"VSUMN{d}": vsumn,
        f"VSUMD{d}": vsump - vsumn,
    }


def alpha158_single(df: pd.DataFrame, windows=WINDOWS) -> pd.DataFrame:
    """对**单只**标的算全套特征，返回 index 与输入对齐的 DataFrame。

    要求列含 ``open/high/low/close/volume``；``amount`` / ``vwap`` 可选。
    """
    need = {"open", "high", "low", "close", "volume"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"缺少必需列 {sorted(missing)}")

    src = df.astype("float64")
    feats: dict[str, pd.Series] = {}
    feats.update(_kbar(src))
    feats.update(_price(src))
    for d in windows:
        feats.update(_rolling(src, d))
    out = pd.DataFrame(feats, index=df.index)
    return out.replace([np.inf, -np.inf], np.nan)


def alpha158(
    bars: dict[str, pd.DataFrame],
    windows=WINDOWS,
    min_bars: int | None = None,
) -> pd.DataFrame:
    """对一篮子标的算特征，拼成 ``(datetime, instrument)`` 面板。

    Args:
        min_bars: 少于这么多根 bar 的标的直接跳过。缺省取 ``max(windows) + 10``
            ——窗口都填不满的标的，特征绝大部分是 NaN，留着只会污染截面统计。
    """
    if min_bars is None:
        min_bars = max(windows) + 10
    parts = []
    for sym, df in sorted(bars.items()):
        if df is None or len(df) < min_bars:
            logger.debug("跳过 %s：只有 %s 根 bar（需要 %d）", sym,
                         0 if df is None else len(df), min_bars)
            continue
        try:
            f = alpha158_single(df, windows)
        except Exception as e:
            logger.warning("标的 %s 特征计算失败：%s", sym, e)
            continue
        f.index = pd.MultiIndex.from_arrays(
            [pd.to_datetime(f.index), [sym] * len(f)],
            names=["datetime", "instrument"])
        parts.append(f)
    if not parts:
        return pd.DataFrame(
            index=pd.MultiIndex.from_arrays([pd.DatetimeIndex([]), []],
                                            names=["datetime", "instrument"]),
            columns=feature_names(windows), dtype="float64")
    return pd.concat(parts).sort_index()


def feature_names(windows=WINDOWS) -> list[str]:
    """特征名清单，顺序与 :func:`alpha158_single` 的输出一致。"""
    idx = pd.date_range("2024-01-01", periods=max(windows) + 5)
    fake = pd.DataFrame({c: np.linspace(1, 2, len(idx)) for c in
                         ("open", "high", "low", "close", "volume")}, index=idx)
    return list(alpha158_single(fake, windows).columns)


def forward_return(bars: dict[str, pd.DataFrame], horizon: int = 5) -> pd.Series:
    """标签：未来 ``horizon`` 日收益 ``close[t+h]/close[t] - 1``，同样是面板格式。

    末尾 h 根没有标签（未来还没发生），会是 NaN——由
    :func:`preprocess.dropna_label` 丢掉，不能填 0。
    """
    parts = []
    for sym, df in sorted(bars.items()):
        if df is None or len(df) <= horizon:
            continue
        c = df["close"].astype("float64")
        r = c.shift(-horizon) / c - 1
        r.index = pd.MultiIndex.from_arrays(
            [pd.to_datetime(r.index), [sym] * len(r)],
            names=["datetime", "instrument"])
        parts.append(r)
    return pd.concat(parts).sort_index() if parts else pd.Series(dtype="float64")


# ======================================================================
# 与 qlib 对拍
# ======================================================================

def compare_with_qlib(symbol: str, start: str, end: str,
                      tol: float = 1e-6) -> dict[str, Any]:
    """在装了 qlib + 有 .bin 数据的机器上，逐特征比对本实现与 qlib Alpha158。

    开发机上没装 qlib，所以本模块只按公开定义实现、没做过逐位对拍。
    真要拿本地特征和 qlib 时代的结果对照，先跑这个。

    Returns:
        ``{"matched": [...], "mismatched": {name: max_abs_diff}, "missing": [...]}``；
        qlib 不可用时返回 ``{"error": ...}``。
    """
    try:
        from qlib.contrib.data.handler import Alpha158 as QAlpha158
    except Exception as e:  # pragma: no cover - 取决于环境
        return {"error": f"qlib 不可用：{e}"}

    from eq.data.market import get_recent_bars

    try:
        handler = QAlpha158(instruments=[symbol], start_time=start, end_time=end,
                            infer_processors=[], learn_processors=[])
        theirs = handler.fetch(col_set="feature")
    except Exception as e:  # pragma: no cover
        return {"error": f"qlib handler 失败：{e}"}

    bars = {symbol: get_recent_bars(symbol, days=2000)}
    mine = alpha158(bars)

    common = [c for c in mine.columns if c in theirs.columns]
    missing = [c for c in mine.columns if c not in theirs.columns]
    matched, mismatched = [], {}
    for col in common:
        a = mine[col].droplevel("instrument")
        b = theirs[col].droplevel("instrument") if isinstance(
            theirs.index, pd.MultiIndex) else theirs[col]
        joined = pd.concat([a, b], axis=1, join="inner").dropna()
        if joined.empty:
            continue
        diff = float((joined.iloc[:, 0] - joined.iloc[:, 1]).abs().max())
        if diff <= tol:
            matched.append(col)
        else:
            mismatched[col] = diff
    return {"matched": matched, "mismatched": mismatched, "missing": missing,
            "n_common": len(common)}
