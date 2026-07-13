"""因子和信号的共享类型契约（problem 10/14 冶议）。

FactorFunc：输入价格 DataFrame，返回单数列（pd.Series）
SignalFunc：输入价格 DataFrame，返回信号数列（pd.Series，值 BUY/SELL/HOLD 或 float 0~1 置信度）

两引擎共享此接口，零适配器。
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

# 因子：输入价格数据，返回单数列
FactorFunc = "pd.Series | pd.DataFrame"  # 实为 Callable[[pd.DataFrame], pd.Series]，用字符串避免运行时类型开销

# 信号值枚举
SignalValue = Literal["BUY", "SELL", "HOLD"]

# 信号函数：输入价格数据，返回信号数列
SignalFunc = "pd.Series"  # 实为 Callable[[pd.DataFrame], pd.Series]

# 常用信号值（用于向量比较）
BUY = "BUY"
SELL = "SELL"
HOLD = "HOLD"
