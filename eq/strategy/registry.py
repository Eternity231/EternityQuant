"""内置策略注册表（v0.33）——CLI 和看板共用的唯一一份。

原来这张表只长在 ``eq/cli.py`` 里，看板的回测页自己另抄了一份、只列了 4 个策略；
新加的 supertrend / donchian / managed_trend 在网页上永远看不到。
放这里之后两边都从同一处取，加策略只需要改一个地方。

**为什么是函数不是模块级字典**：组合策略（trend_vote / regime_switch / managed_trend）
要先构造出基础策略再包一层，模块级求值会把这些 import 提到 ``eq.strategy`` 导入时，
拖慢所有只想用一个指标函数的调用方。构造结果用 ``lru_cache`` 缓存，重复调用不重复建。
"""

from __future__ import annotations

from functools import lru_cache

from eq.strategy.types import SignalFunc

__all__ = ["builtin_strategies", "list_strategies", "resolve", "CATEGORIES"]

# 分类只用于展示（CLI 帮助、看板下拉分组），不影响调用
CATEGORIES: dict[str, tuple[str, ...]] = {
    "趋势": ("ema_cross", "adx_trend", "supertrend"),
    "反转": ("rsi_reversal", "bollinger_break", "zscore_reversion",
             "kdj_cross", "cci_reversal", "reversion_pack"),
    "突破": ("donchian", "keltner", "vol_breakout", "breakout_pack"),
    "组合/风控": ("trend_vote", "regime_switch", "trend_vol_filtered", "managed_trend"),
}


@lru_cache(maxsize=1)
def builtin_strategies() -> dict[str, SignalFunc]:
    """内置策略表（延迟构造：组合策略要引用其他策略）。

    v0.27 从 4 个扩到 14 个，按套路分四类——原来四个全是「单指标交叉」，
    结构同质，横评出来的差异主要来自行情碰巧适合谁，而不是策略本身。
    """
    from eq.strategy.risk import make_managed
    from eq.strategy.signals import (
        adx_trend, bollinger_break, breakout_composite, cci_reversal,
        donchian_breakout, ema_cross, kdj_cross, keltner_breakout, make_regime_adaptive,
        make_vote, reversion_composite, rsi_reversal, supertrend_follow,
        volatility_breakout, volume_filter, zscore_reversion,
    )
    from eq.strategy.signals.composite import make_filtered

    trend_pack = make_vote([ema_cross, adx_trend, supertrend_follow], min_agree=2)
    reversion_pack = make_vote([rsi_reversal, zscore_reversion, bollinger_break], min_agree=2)
    return {
        # --- 趋势 ---
        "ema_cross": ema_cross,
        "adx_trend": adx_trend,
        "supertrend": supertrend_follow,
        # --- 反转 ---
        "rsi_reversal": rsi_reversal,
        "bollinger_break": bollinger_break,
        "zscore_reversion": zscore_reversion,
        "kdj_cross": kdj_cross,
        "cci_reversal": cci_reversal,
        "reversion_pack": reversion_composite,
        # --- 突破 ---
        "donchian": donchian_breakout,
        "keltner": keltner_breakout,
        "vol_breakout": volatility_breakout,
        "breakout_pack": breakout_composite,
        # --- 组合 / 择时 / 风控 ---
        "trend_vote": trend_pack,
        "regime_switch": make_regime_adaptive(trend_pack, reversion_pack),
        "trend_vol_filtered": make_filtered(trend_pack, [volume_filter]),
        "managed_trend": make_managed(trend_pack, sizing="vol_target", stops=True),
    }


def list_strategies() -> list[str]:
    """所有内置策略名，已排序。"""
    return sorted(builtin_strategies())


def resolve(name: str) -> SignalFunc:
    """按名字取策略函数。未知名字抛 ``KeyError``，消息里带可选项。"""
    table = builtin_strategies()
    try:
        return table[name]
    except KeyError as e:
        raise KeyError(f"未知策略 {name!r}，可选：{', '.join(sorted(table))}") from e
