"""数据层子包：按市场选主源拉行情，失败切 fallback。

市场主源策略（问题 12）：
- A股 (.SH/.SZ/.BJ)  → baostock（待集成） → fallback akshare
- 港股 (.HK) / 美股 (.US) → yfinance → fallback akshare
- 加密  → OKX（待集成） → fallback ccxt

第一版：yfinance + akshare 直调（SDK），baostock/okx 后续加。
"""
