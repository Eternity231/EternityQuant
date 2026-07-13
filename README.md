# EternityQuant

个人散户量化助手 —— 不交易，只提醒和辅助决策。

## 当前能力（v0.1 最短闭环）

```bash
eq watch 600519.SH       # 查个股快照（行情+关键指标）
eq --help                # 看所有命令
```

## 架构原则

- **EternityQuant 自写全部核心引擎**（数据层、信号引擎、回测、监控、推送）。
- **vibe-trading 仅作数据补全和一次性查询的助手**，不是硬依赖 —— 挂了框架照样跑。
- **qlib 作信号引擎**，预测值作为因子喂给信号层。

## 技术栈

- CLI：`typer`
- 定时：`APScheduler`（后续）
- Web：`Streamlit`（后续）
- 数据源：A股 baostock，港美股 yfinance，加密 okx；fallback akshare
- 回测：双引擎（向量化 + 事件驱动），共享 `signal(df) -> df` 接口

## 配置

- `~/.eternityquant/config.yml`：静态配置
- `~/.eternityquant/.env`：密钥（tushare token、企业微信 webhook）
- `~/.eternityquant/eternityquant.db`：状态库（watchlist/portfolio/rules/signals/ml_*）
- `~/.eternityquant/market_cache.db`：行情缓存（可随时删）
- `~/.eternityquant/backtests/<run_id>.parquet`：回测详细数据外存

## 路线图

1. ✅ CLI + 数据层 + watch 命令
2. ⏳ scan 全市场扫描
3. ⏳ watchlist / portfolio 增删查
4. ⏳ rules 监控引擎 + 推送通道（企业微信 + 桌面通知）
5. ⏳ factors + signals 两层分层 + 向量化回测
6. ⏳ qlib 集成作 ML 因子
7. ⏳ Streamlit 仪表盘
8. ⏳ 事件驱动回测引擎（第二引擎）

## License

MIT
