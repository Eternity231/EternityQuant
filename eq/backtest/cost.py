"""交易成本模型（v0.29 新增）。

**原来的问题**

``BacktestConfig`` 只有 ``commission_bps`` / ``slippage_bps`` 两个数，双边同费率。
这和真实成本差得很远，而且差的方向对散户特别不利：

============  ========================================================
真实成本项      实际规则
============  ========================================================
佣金           万 2.5 左右，但**每笔最低 5 元**——这条最要命
印花税         千分之一，**仅卖出**收（买入不收）
过户费         万分之 0.1，沪深双边都收
============  ========================================================

**最低佣金**意味着小额交易的实际费率远高于名义费率：
2 万元的单子按万 2.5 是 5 元，正好卡在最低线；**5000 元的单子仍要交 5 元，
实际费率是万 10——名义费率的 4 倍**。一个高频小额策略在旧模型下看着能赚，
按真实成本可能直接亏光。

印花税只在卖出收，也意味着**买卖成本不对称**，旧模型的对称假设会低估卖出成本。

本模块给出各市场的成本预设，并保留旧的纯 bps 模式作向后兼容。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class CostModel:
    """一个市场的交易成本规则。

    所有费率都是**小数**（0.00025 = 万 2.5），不是 bps。

    Attributes:
        commission_rate: 佣金费率（双边）
        min_commission: 每笔最低佣金（账户币种）。这是散户小额交易的主要成本来源
        stamp_duty_sell: 印花税，仅卖出
        stamp_duty_buy:  印花税，买入（A 股为 0，港股买卖都收）
        transfer_fee:    过户费/交收费（双边）
        platform_fee:    交易所规费等其他按比例收的（双边）
        slippage_rate:   滑点（单边）
        lot_size:        最小交易单位（A 股 100 股一手）
    """

    name: str = "flat"
    label: str = "固定费率"
    commission_rate: float = 0.00025
    min_commission: float = 0.0
    stamp_duty_sell: float = 0.0
    stamp_duty_buy: float = 0.0
    transfer_fee: float = 0.0
    platform_fee: float = 0.0
    slippage_rate: float = 0.0005
    lot_size: int = 1

    def commission(self, value: float) -> float:
        """佣金（含最低佣金约束）。``value`` 是成交金额。"""
        if value <= 0:
            return 0.0
        return max(value * self.commission_rate, self.min_commission)

    def trade_cost(self, value: float, side: Side = "buy") -> float:
        """一笔成交的**总成本**（绝对金额，不含滑点）。"""
        if value <= 0:
            return 0.0
        cost = self.commission(value)
        cost += value * (self.stamp_duty_sell if side == "sell" else self.stamp_duty_buy)
        cost += value * (self.transfer_fee + self.platform_fee)
        return cost

    def cost_ratio(self, value: float, side: Side = "buy") -> float:
        """这笔成交的**实际费率**（成本 / 成交金额）。

        小额交易因为最低佣金会显著高于名义费率——用这个函数能直接看出来。
        """
        if value <= 0:
            return 0.0
        return self.trade_cost(value, side) / value

    def round_trip_ratio(self, value: float) -> float:
        """买入 + 卖出一个来回的总费率，便于快速估算「至少要涨多少才回本」。"""
        if value <= 0:
            return 0.0
        return (self.trade_cost(value, "buy") + self.trade_cost(value, "sell")) / value

    def breakeven_pct(self, value: float) -> float:
        """一个来回的盈亏平衡涨幅（含滑点）。"""
        return self.round_trip_ratio(value) + 2 * self.slippage_rate

    def round_lots(self, shares: float) -> float:
        """按最小交易单位向下取整。"""
        if self.lot_size <= 1:
            return float(int(shares))
        return float(int(shares // self.lot_size) * self.lot_size)


# ======================================================================
# 各市场预设
# ======================================================================

A_SHARE = CostModel(
    name="a_share", label="A 股",
    commission_rate=0.00025,   # 万 2.5（各家券商不同，这是常见档）
    min_commission=5.0,        # 每笔最低 5 元——小额交易的实际费率会高很多
    stamp_duty_sell=0.001,     # 印花税千分之一，仅卖出
    stamp_duty_buy=0.0,
    transfer_fee=0.00001,      # 过户费万分之 0.1，沪深双边
    slippage_rate=0.0005,
    lot_size=100,              # 一手 100 股
)

HK_STOCK = CostModel(
    name="hk", label="港股",
    commission_rate=0.00025,
    min_commission=50.0,       # 港股最低佣金通常更高（港币）
    stamp_duty_sell=0.001,     # 港股印花税买卖**双边**都收
    stamp_duty_buy=0.001,
    transfer_fee=0.00005,      # 交收费
    platform_fee=0.00005,      # 交易征费 + 交易费
    slippage_rate=0.0008,      # 流动性一般不如 A 股主板
    lot_size=1,                # 每手股数各股不同，这里不强制
)

US_STOCK = CostModel(
    name="us", label="美股",
    commission_rate=0.0,       # 主流券商零佣金
    min_commission=0.0,
    stamp_duty_sell=0.0000278,  # SEC fee，仅卖出
    stamp_duty_buy=0.0,
    platform_fee=0.0,
    slippage_rate=0.0005,
    lot_size=1,
)

CRYPTO = CostModel(
    name="crypto", label="加密",
    commission_rate=0.001,     # 现货 taker 常见 0.1%
    min_commission=0.0,
    slippage_rate=0.001,       # 深度差，滑点更大
    lot_size=1,
)

FLAT = CostModel(name="flat", label="固定费率（旧行为）")

PRESETS: dict[str, CostModel] = {
    "a_share": A_SHARE, "a": A_SHARE,
    "hk": HK_STOCK, "us": US_STOCK,
    "crypto": CRYPTO, "flat": FLAT,
}


def get_cost_model(name: str | CostModel | None) -> CostModel | None:
    """按名字取预设。``None`` 原样返回（调用方退回纯 bps 行为）。"""
    if name is None or isinstance(name, CostModel):
        return name
    key = str(name).strip().lower()
    if key not in PRESETS:
        raise ValueError(f"未知成本模型 {name}，可选：{sorted(set(PRESETS))}")
    return PRESETS[key]


def for_market(market: str) -> CostModel:
    """按市场代码取成本模型（A/HK/US/CRYPTO）。"""
    return {"A": A_SHARE, "HK": HK_STOCK, "US": US_STOCK,
            "CRYPTO": CRYPTO}.get(str(market).upper(), FLAT)


def from_bps(commission_bps: float, slippage_bps: float) -> CostModel:
    """把旧的 bps 配置包成 CostModel，保证行为完全一致。"""
    return CostModel(name="flat", label="固定费率（旧行为）",
                     commission_rate=commission_bps / 1e4,
                     slippage_rate=slippage_bps / 1e4)


def compare_costs(values: list[float] | None = None,
                  models: list[CostModel] | None = None) -> "object":
    """不同成交金额下的实际费率对照表，直观看出最低佣金的影响。"""
    import pandas as pd

    values = values or [2_000, 5_000, 10_000, 20_000, 50_000, 200_000]
    models = models or [A_SHARE, HK_STOCK, US_STOCK, CRYPTO]
    rows = []
    for v in values:
        row = {"成交金额": v}
        for m in models:
            row[f"{m.label} 单边"] = round(m.cost_ratio(v, "buy") * 1e4, 2)
            row[f"{m.label} 来回%"] = round(m.breakeven_pct(v) * 100, 3)
        rows.append(row)
    return pd.DataFrame(rows)
