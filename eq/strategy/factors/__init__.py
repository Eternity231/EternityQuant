"""因子子包：单数列计算，纯 pandas 向量化实现。

- technical.py：RSI / EMA / MACD / ADX / KDJ / 布林
- volume.py  ：OBV / 量比 / 换手率
- fundamental.py：PE / PB / ROE（跨表，后续集成）
- ml.py：qlib 预测输出作 ML 因子（problem 16）
"""

from eq.strategy.factors import ml  # noqa: F401  # 让 from eq.strategy.factors import ml 生效

