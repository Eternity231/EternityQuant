"""技术选股器（v0.24 新增）。

扫描一批标的的日线，逐个跑技术条件，输出命中的标的 + 命中原因。
和 :mod:`eq.core.scanner`（按涨幅/成交量排行）互补：scanner 回答"今天谁涨得多"，
screener 回答"谁现在处在我要的技术形态上"。

内置条件（可多选，默认全部满足才算命中，``mode="any"`` 时任一满足即可）：

===================  ====================================================
条件名                含义
===================  ====================================================
``rsi_oversold``     RSI(14) < 阈值（默认 30）——超卖
``rsi_overbought``   RSI(14) > 阈值（默认 70）——超买
``golden_cross``     EMA5 上穿 EMA20（最近 N 根内）
``death_cross``      EMA5 下穿 EMA20（最近 N 根内）
``above_ma``         收盘价站上 MA(period)
``below_ma``         收盘价跌破 MA(period)
``volume_spike``     当日量 / 20 日均量 ≥ 倍数（默认 2.0）
``macd_golden``      MACD DIF 上穿 DEA（最近 N 根内）
``near_high``        距 N 日新高不足 X%（默认 60 日 / 3%）
``near_low``         距 N 日新低不足 X%
``breakout``         收盘创 N 日新高（默认 20 日）
``pullback``         上升趋势（价 > MA60）中回踩 MA20 附近（±2%）
===================  ====================================================

用法::

    from eq.core.screener import screen
    hits = screen(["600519.SH", "000001.SZ"], ["rsi_oversold", "above_ma"])
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import pandas as pd

from eq.data.market import get_recent_bars, normalize_symbol
from eq.strategy.factors.technical import bollinger, ema, macd, rsi

logger = logging.getLogger(__name__)

# 条件名 -> (函数, 默认参数, 中文说明)
Condition = Callable[[pd.DataFrame, dict[str, Any]], tuple[bool, str]]


def _c_rsi_oversold(df, p):
    level = float(p.get("rsi_level", 30))
    v = float(rsi(df, int(p.get("rsi_period", 14))).iloc[-1])
    return v < level, f"RSI={v:.1f}<{level:g}"


def _c_rsi_overbought(df, p):
    level = float(p.get("rsi_level", 70))
    v = float(rsi(df, int(p.get("rsi_period", 14))).iloc[-1])
    return v > level, f"RSI={v:.1f}>{level:g}"


def _cross(fast: pd.Series, slow: pd.Series, lookback: int, up: bool) -> tuple[bool, int]:
    """fast 在最近 lookback 根内是否穿越 slow。返回 (是否, 几根之前)。"""
    diff = (fast - slow).dropna()
    if len(diff) < 2:
        return False, -1
    window = diff.iloc[-(lookback + 1):]
    vals = window.to_numpy()
    for k in range(len(vals) - 1, 0, -1):
        crossed = (vals[k - 1] <= 0 < vals[k]) if up else (vals[k - 1] >= 0 > vals[k])
        if crossed:
            return True, len(vals) - 1 - k
    return False, -1


def _c_golden_cross(df, p):
    lb = int(p.get("lookback", 3))
    ok, ago = _cross(ema(df, int(p.get("fast", 5))), ema(df, int(p.get("slow", 20))), lb, up=True)
    return ok, f"EMA金叉({ago}根前)" if ok else ""


def _c_death_cross(df, p):
    lb = int(p.get("lookback", 3))
    ok, ago = _cross(ema(df, int(p.get("fast", 5))), ema(df, int(p.get("slow", 20))), lb, up=False)
    return ok, f"EMA死叉({ago}根前)" if ok else ""


def _c_macd_golden(df, p):
    m = macd(df)
    ok, ago = _cross(m["dif"], m["dea"], int(p.get("lookback", 3)), up=True)
    return ok, f"MACD金叉({ago}根前)" if ok else ""


def _c_macd_death(df, p):
    m = macd(df)
    ok, ago = _cross(m["dif"], m["dea"], int(p.get("lookback", 3)), up=False)
    return ok, f"MACD死叉({ago}根前)" if ok else ""


def _c_above_ma(df, p):
    n = int(p.get("ma_period", 20))
    ma = df["close"].rolling(n).mean().iloc[-1]
    c = float(df["close"].iloc[-1])
    if pd.isna(ma):
        return False, ""
    return c > ma, f"价{c:.2f}>MA{n}({ma:.2f})"


def _c_below_ma(df, p):
    n = int(p.get("ma_period", 20))
    ma = df["close"].rolling(n).mean().iloc[-1]
    c = float(df["close"].iloc[-1])
    if pd.isna(ma):
        return False, ""
    return c < ma, f"价{c:.2f}<MA{n}({ma:.2f})"


def _c_volume_spike(df, p):
    mult = float(p.get("volume_multiple", 2.0))
    if len(df) < 21:
        return False, ""
    avg = float(df["volume"].iloc[:-1].tail(20).mean())
    if avg <= 0:
        return False, ""
    ratio = float(df["volume"].iloc[-1]) / avg
    return ratio >= mult, f"量比{ratio:.2f}x"


def _c_near_high(df, p):
    n, tol = int(p.get("high_period", 60)), float(p.get("tolerance_pct", 3.0))
    hi = float(df["high"].tail(n).max())
    c = float(df["close"].iloc[-1])
    if hi <= 0:
        return False, ""
    gap = (hi - c) / hi * 100
    return gap <= tol, f"距{n}日高{gap:.1f}%"


def _c_near_low(df, p):
    n, tol = int(p.get("low_period", 60)), float(p.get("tolerance_pct", 3.0))
    lo = float(df["low"].tail(n).min())
    c = float(df["close"].iloc[-1])
    if lo <= 0:
        return False, ""
    gap = (c - lo) / lo * 100
    return gap <= tol, f"距{n}日低{gap:.1f}%"


def _c_breakout(df, p):
    n = int(p.get("breakout_period", 20))
    if len(df) < n + 1:
        return False, ""
    prior_high = float(df["close"].iloc[-(n + 1):-1].max())
    c = float(df["close"].iloc[-1])
    return c > prior_high, f"创{n}日新高({c:.2f}>{prior_high:.2f})"


def _c_pullback(df, p):
    """上升趋势中回踩：价 > MA60，且贴近 MA20（±tolerance）。"""
    tol = float(p.get("tolerance_pct", 2.0))
    if len(df) < 60:
        return False, ""
    ma20 = float(df["close"].rolling(20).mean().iloc[-1])
    ma60 = float(df["close"].rolling(60).mean().iloc[-1])
    c = float(df["close"].iloc[-1])
    if pd.isna(ma20) or pd.isna(ma60) or ma20 <= 0:
        return False, ""
    near = abs(c - ma20) / ma20 * 100 <= tol
    return (c > ma60 and near), f"多头回踩MA20(偏离{abs(c - ma20) / ma20 * 100:.1f}%)"


def _c_bollinger_squeeze(df, p):
    """布林带收口（带宽处于近 N 日最窄的分位）——变盘前兆。"""
    n = int(p.get("squeeze_period", 60))
    q = float(p.get("squeeze_quantile", 0.2))
    b = bollinger(df, int(p.get("boll_period", 20)))
    width = ((b["upper"] - b["lower"]) / b["mid"].replace(0, pd.NA)).dropna()
    if len(width) < n // 2:
        return False, ""
    recent = width.tail(n)
    cur = float(width.iloc[-1])
    thresh = float(recent.quantile(q))
    return cur <= thresh, f"布林收口(带宽{cur:.3f}≤{thresh:.3f})"


CONDITIONS: dict[str, tuple[Condition, str]] = {
    "rsi_oversold": (_c_rsi_oversold, "RSI 超卖"),
    "rsi_overbought": (_c_rsi_overbought, "RSI 超买"),
    "golden_cross": (_c_golden_cross, "EMA 金叉"),
    "death_cross": (_c_death_cross, "EMA 死叉"),
    "macd_golden": (_c_macd_golden, "MACD 金叉"),
    "macd_death": (_c_macd_death, "MACD 死叉"),
    "above_ma": (_c_above_ma, "站上均线"),
    "below_ma": (_c_below_ma, "跌破均线"),
    "volume_spike": (_c_volume_spike, "放量"),
    "near_high": (_c_near_high, "接近新高"),
    "near_low": (_c_near_low, "接近新低"),
    "breakout": (_c_breakout, "突破新高"),
    "pullback": (_c_pullback, "多头回踩"),
    "squeeze": (_c_bollinger_squeeze, "布林收口"),
}


def screen(
    symbols: list[str],
    conditions: list[str],
    *,
    mode: str = "all",
    params: dict[str, Any] | None = None,
    days: int = 120,
    workers: int = 8,
    use_cache: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """对一批标的跑技术条件筛选。

    Args:
        symbols: 标的列表
        conditions: :data:`CONDITIONS` 里的条件名
        mode: ``"all"`` 全部满足 / ``"any"`` 任一满足
        params: 条件参数覆盖（如 ``{"rsi_level": 25, "ma_period": 60}``）
        days: 每只拉多少根日线（部分条件需要 60 根以上）
        use_cache: 走本地 bar_cache（默认走，重复筛选几乎不打网络）

    Returns:
        命中列表，按命中数降序：
        ``{symbol, close, change_pct, matched: [...], reasons: [...], score}``
    """
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        raise ValueError(f"未知筛选条件 {unknown}，可选：{sorted(CONDITIONS)}")
    if not conditions:
        raise ValueError("至少要指定一个筛选条件")
    if mode not in ("all", "any"):
        raise ValueError(f"mode 只能是 all / any，收到 {mode}")

    p = dict(params or {})
    symbols = list(dict.fromkeys(normalize_symbol(s) for s in symbols))
    total = len(symbols)
    done = 0

    def _one(sym: str) -> dict[str, Any] | None:
        try:
            df = get_recent_bars(sym, days=days, use_cache=use_cache)
        except Exception as e:
            logger.debug("筛选取数失败 %s：%s", sym, e)
            return None
        if df is None or len(df) < 25:
            return None
        matched, reasons = [], []
        for name in conditions:
            fn = CONDITIONS[name][0]
            try:
                ok, why = fn(df, p)
            except Exception as e:
                logger.debug("条件 %s 在 %s 上算失败：%s", name, sym, e)
                ok, why = False, ""
            if ok:
                matched.append(name)
                if why:
                    reasons.append(why)
        hit = len(matched) == len(conditions) if mode == "all" else bool(matched)
        if not hit:
            return None
        close = float(df["close"].iloc[-1])
        prev = float(df["close"].iloc[-2]) if len(df) >= 2 else close
        return {
            "symbol": sym,
            "close": close,
            "change_pct": (close - prev) / prev * 100 if prev else 0.0,
            "matched": matched,
            "reasons": reasons,
            "score": len(matched),
        }

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, max(total, 1)))) as pool:
        for res in pool.map(_one, symbols):
            done += 1
            if on_progress:
                on_progress(done, total)
            if res:
                results.append(res)

    results.sort(key=lambda d: (-d["score"], -d["change_pct"]))
    return results


def format_screen(hits: list[dict[str, Any]], conditions: list[str], mode: str = "all") -> str:
    """格式化筛选结果为文本表格。"""
    cond_label = " + ".join(CONDITIONS[c][1] for c in conditions if c in CONDITIONS)
    join = "全部满足" if mode == "all" else "任一满足"
    if not hits:
        return f"\n筛选条件：{cond_label}（{join}）\n  无标的命中\n"
    lines = [f"\n筛选条件：{cond_label}（{join}）— 命中 {len(hits)} 只\n"]
    lines.append(f"{'代码':<14} {'最新价':>10} {'涨跌幅':>9} {'命中':>5}  命中原因")
    lines.append("-" * 96)
    for h in hits:
        lines.append(
            f"{h['symbol']:<14} {h['close']:>10.2f} {h['change_pct']:>+8.2f}% "
            f"{h['score']:>5}  {', '.join(h['reasons'])[:52]}"
        )
    return "\n".join(lines) + "\n"
