# EternityQuant

个人散户量化助手 —— 不交易，只提醒和辅助决策。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/colab_eternityquant_train.ipynb)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](notebooks/kaggle_eternityquant_train.ipynb)

当前版本 **v0.15**（16 个 commit 全实测固化，35 单元测试）。

## 当前能力速览

```bash
eq watch 600519.SH                       # 个股快照（A/HK/US/CRYPTO）
eq scan A --by change_pct --top 30       # 四市场扫描（A/HK/US/CRYPTO）
eq research 600519.SH                   # 个股深度研究（14 板块）
eq watchlist add 600519.SH --reason 白酒龙头
eq portfolio buy 600519.SH 100 1680     # 建仓 100 股 @1680
eq monitor add 600519.SH price_cross '{"level":1700,"direction":"up"}' --channels desktop
eq scheduler add 每日收盘扫描 '0 16 * * 1-5' scan_report --params '{"market":"A","top_n":20}'
eq backtest 600519.SH trend_ema --engine vectorized
eq bt list / show <run_id> / remove <run_id>
eq ml train csi300 5 --algo lightgbm --device cpu      # LightGBM CPU
eq ml train csi300 5 --algo lightgbm --device gpu      # LightGBM GPU（OpenCL）
eq ml train csi300 5 --algo mlp --device cuda          # 自写 MLP 走 3060 CUDA
eq ml train csi300 5 --algo lstm --device cuda          # 自写 LSTM 走 CUDA（量化选股最佳，6×26 时序重塑）
eq ml update-data --start 2020-09-28 --universe csi300  # qlib 数据续到最新（baostock）
eq ml activate <model_id>
eq ml predict-batch <model_id> --top 10                # 批量预测入 ml_predictions 表（v0.14 支持自写模型）
eq dash                                 # 启动 Streamlit 6 页看板
eq --help                               # 看所有命令
```

## 架构原则

- **EternityQuant 自写全部核心引擎**（数据层、信号引擎、回测、监控、推送）。
- **vibe-trading MCP 仅作数据补全和一次性查询的助手**，不是硬依赖 —— 挂了框架照样跑。
  - 港/美/加密行情源被中国大陆网络限流时，`eq research` 输出显式 MCP 补全建议（💡），agent 里跑会自动调 vibe-trading `get_*` 系列补全。
- **qlib 作信号引擎**，预测值作为因子喂给信号层。

## 技术栈

- CLI：`typer`
- 定时：`APScheduler`（`eq scheduler daemon` 常驻）
- Web：`Streamlit`（6 页看板：概览/持仓/自选/监控/ML/深度研究）
- 数据源：A股 baostock（TCP 稳）/ 港美 yfinance / 加密 okx / fallback akshare
- 回测：双引擎（向量化 + 事件驱动），共享 `signal(df) -> df` 接口
- ML：qlib Alpha158 特征 + LightGBM（CPU/GPU）+ 自写 MLP（CUDA，3060 主场）

## 配置分层

- `~/.eternityquant/config.yml`：静态配置（可选）
- `~/.eternityquant/.env`：密钥（tushare token、企业微信 webhook）
- `~/.eternityquant/eternityquant.db`：状态库（10 表：watchlist/portfolio/trade_history/rules/signals/ml_models/ml_predictions/ml_runs/scheduled_jobs/backtest_runs）
- `~/.eternityquant/market_cache.db`：行情缓存（可随时删）
- `~/.eternityquant/backtests/<run_id>.parquet`：回测详细数据外存
- `~/.eternityquant/ml_models/*.pkl`：ML 模型文件
- `~/.qlib/qlib_data/cn_data`：qlib A 股本地数据集（2001~2020-09，196MB）

## CLI 命令全貌

| 命令组 | 功能 | 版本 |
|--------|------|------|
| `eq watch <symbol>` | 个股快照（A/HK/US/CRYPTO） | v0.1 |
| `eq scan <market> --by --top` | 四市场扫描（A/HK/US/CRYPTO） | v0.4 |
| `eq research <symbol> --sections` | 个股深度研究（14 板块） | v0.10 |
| `eq watchlist add/list/remove/find` | 自选股 CRUD | v0.1 |
| `eq portfolio buy/add/trim/sell/list/stops/history` | 持仓全生命周期（成本价加权 + 已实现盈亏） | v0.1 |
| `eq monitor add/list/run/enable/disable` | 监控规则（10 种硬编码类型） | v0.1/v0.5 |
| `eq scheduler add/list/run/daemon` | 定时推送（APScheduler） | v0.2 |
| `eq backtest ... --engine vectorized/event_driven` | 双引擎回测，自动外存 parquet | v0.1/v0.3 |
| `eq bt list/show/remove` | 回测历史管理 | v0.3 |
| `eq ml train/activate/list/info/predict/predict-batch/update-data` | ML 因子（LightGBM + PyTorch + 数据更新） | v0.6~v0.15 |
| `eq dash` | Streamlit 6 页看板 | v0.1/v0.11 |

## 监控规则 10 种类型

| 类型 | 说明 | params 示例 | 版本 |
|------|------|------------|------|
| `price_cross` | 突破/跌破某价位 | `{"level":1700,"direction":"up"}` | v0.1 |
| `price_pct` | 涨跌幅超阈值 | `{"threshold":5.0}` | v0.1 |
| `indicator` | RSI/MACD/KDJ 因子触发 | `{"name":"rsi","period":14,"level":30,"action":"buy"}` | v0.5 |
| `volume_spike` | 成交量异常放大 | `{"multiple":3.0}` | v0.1 |
| `limit_up` / `limit_down` | 涨跌停（仅 A 股） | `{}` | v0.1 |
| `news` | 个股新闻推送 | `{}` | v0.5 |
| `event` | 事件日提醒（财报/解禁/分红） | `{"event_type":"financial_report","date":"2026-07-14","name":"中报披露"}` | v0.5 |
| `flow` | 北向资金流异动 | `{"source":"northbound","threshold":100000000}` | v0.5 |
| `stop_loss` / `take_profit` | 持仓止损/止盈价触发 | `{}`（自动关联 portfolio） | v0.1 |

## ML 因子层

### IC 指标说明

**IC（Information Coefficient，信息系数）** 是量化选股中最核心的因子评价指标，衡量**因子预测值**与**未来真实收益**之间的相关性。

| 指标 | 公式 | 含义 |
|------|------|------|
| **Pearson IC** | `corr(pred, actual)` | 预测值与真实收益的线性相关系数。正值越大越好，+0.10 以上即有显著预测力 |
| **Rank IC** | `corr(rank(pred), rank(actual))` | 秩相关系数（Spearman），更稳健，对异常值不敏感 |
| **ICIR** | `mean(IC) / std(IC)` | IC 的稳定性指标，衡量因子预测力是否持续，> 0.5 为优秀 |
| **Rank ICIR** | `mean(Rank IC) / std(Rank IC)` | Rank IC 的稳定性 |

**IC 解读参考：**
- `IC > 0.10`：因子有实际预测力，可用于选股
- `IC > 0.15`：因子显著，Alpha 收益可观
- `IC > 0.20`：因子非常强（量化私募竞赛级）
- `IC 为负`：因子反向有效（可做反向信号）

> 本框架在训练过程中每步都计算验证集 IC 并打印，训练结束后以最佳 IC 作为模型指标。LightGBM 基线 IC ≈ +0.0985，自写 MLP 可达 +0.1654。

### 四条训练路径

| algo | device | 说明 | IC（CSI300+Alpha158+5年，2020-09 数据） |
|------|--------|------|---------------------------|
| `lightgbm` | `cpu` | 基线，qlib LGBModel | +0.0985 |
| `lightgbm` | `gpu` | OpenCL 后端（默认编译含） | +0.0985 |
| `mlp` | `cuda` | 自写 _SimpleMLP（158→512→256→128→1），真 CUDA | +0.1654 |
| `lstm` | `cuda` | **自写 _SimpleLSTM（6×26 时序重塑，2 层 hidden=128），量化选股最佳** | 待续数据后测 |

LSTM 路径把 Alpha158 的 158 维特征重塑成 (batch, seq_len=6, input_size=26) 的时序张量喂给 LSTM——这是量化选股的正确做法（学"过去 6 日形态"），比 MLP 把特征当独立向量强。3060 12GB CUDA 主场。

### 数据更新器（v0.15）

qlib 本地数据集截至 2020-09-25，`eq ml update-data` 续到最新：

```bash
eq ml update-data --start 2020-09-28 --universe csi300   # baostock 拉 6 年日线，约 30-60 分钟
```

baostock 拉日线 → 转 qlib .bin 格式（float32，按日历顺序）续期 + 日历续期。续完后 `eq ml train` 用最新数据训练，`predict-batch` 出的就是今天的分数。

### Colab / Kaggle 云训练适配

EternityQuant 支持在 **Google Colab** 和 **Kaggle** 的免费 GPU 上训练模型，利用 T4/P100 的 CUDA 加速。

**📍 笔记本地址：**

| 平台 | 笔记本 | GPU | 显存 |
|------|--------|-----|------|
| [Colab](https://colab.research.google.com) | [`notebooks/colab_eternityquant_train.ipynb`](notebooks/colab_eternityquant_train.ipynb) | T4 | 16 GB |
| [Kaggle](https://kaggle.com) | [`notebooks/kaggle_eternityquant_train.ipynb`](notebooks/kaggle_eternityquant_train.ipynb) | T4/P100 | 16 GB |

**云端 vs 本地训练对比：**

| 维度 | 本地（3060 12GB） | Colab（T4 16GB） | Kaggle（T4/P100 16GB） |
|------|-------------------|------------------|----------------------|
| GPU | RTX 3060 | Tesla T4 | T4 / P100 |
| 显存 | 12 GB | 16 GB | 16 GB |
| CUDA 核心 | 3584 | 2560 | 2560 / 3584 |
| 训练速度 | 1×（基准） | ~0.9× | ~0.9× / ~1.2× |
| 使用限制 | 无限制 | 每天有限额 | 每周 30h GPU |
| 数据持久化 | 本地磁盘 | Google Drive | Kaggle Dataset |

**云训练流程：**

1. **打开笔记本** → Colab 或 Kaggle
2. **运行环境准备** → 安装依赖 + 克隆代码
3. **准备数据** → 方案 A：从云存储挂载（推荐）/ 方案 B：在线拉取
4. **训练模型** → LightGBM / MLP / GRU / LSTM
5. **导出模型** → 下载 `.pkl` 文件
6. **回本地导入** → `eq ml register` + `eq ml activate`

**💡 建议：** 在 Colab 中训练 GRU/LSTM，在本地运行 `eq ml predict-batch` 做预测。训练好的模型文件通过 pickle 跨平台兼容。

### 环坑修复记录

- **torch DLL 预热**：Windows + torch 2.13+cu132 坑，qlib 集成链触发 torch 延迟加载 `c10.dll` 失败。`cli.py` 顶层 + `ml_workflow._qlib_init()` 均先 `torch.cuda.init()` 预热。
- **qlib ReduceLROnPlateau 版本判断 bug**：qlib 0.9.7 用 `str(torch.__version__).split('+')[0] <= '2.6.0'` 做字符串比较，对 torch 2.13.0 误判（字典序 `'2.13.0' <= '2.6.0'` 为真），走错老分支传 `verbose=True`。monkey patch 绕开：让 `ReduceLROnPlateau.__init__` 接受并忽略 `verbose` 参数。
- **qlib DNNModelPytorch loss 全 nan**：torch 2.13 + Alpha158 默认配置下 BatchNorm1d �遇全 NaN 列梯度爆。自写 `_SimpleMLP`（158→256→1，BatchNorm1d+Adam+Dropout）绕开，直 API 路径走 `torch.cuda`。

## 个股深度研究 14 板块

按市场自动选板块：

| 市场 | 板块数 | 板块列表 |
|------|--------|----------|
| A 股 | 11 | snapshot/financial/fund_flow/news/research/block_trades/margin/shareholders/lockup/northbound/sector |
| 港股 | 4 | snapshot/profile/news/fund_flow |
| 美股 | 6 | snapshot/profile/sec_filings/news/financial/options |
| 加密 | 1 | snapshot |

港/美/加密数据源被中国大陆网络限流时，输出 vibe-trading MCP 补全建议（💡）。

## Streamlit 6 页看板

```bash
eq dash --port 8501    # 启动本地看板
```

| 页 | 功能 |
|----|------|
| 概览 | 持仓+自选+监控触发汇总 |
| 持仓 | 当前持仓+已清仓记录 |
| 自选 | 自选股列表 |
| 监控规则 | 规则列表+触发统计 |
| ML 模型 | 模型列表+激活+predict-batch Top10+一键入自选+预测历史 |
| 深度研究 | 输入 symbol → 14 板块深度研究（结构化展开） |

## 路线图（全完成）

1. ✅ CLI + 数据层 + watch 命令（v0.1）
2. ✅ 定时推送服务固化（v0.2，APScheduler）
3. ✅ 回测结果外存 parquet + backtest_runs 表（v0.3）
4. ✅ 多市场扫描（v0.4，A/HK/US/CRYPTO）
5. ✅ 四个监控处理器（v0.5，indicator/news/event/flow，10 种规则全落地）
6. ✅ qlib workflow 真集成（v0.6，Alpha158 + LightGBM）
7. ✅ predict-batch 跑通 + torch DLL 预热（v0.7）
8. ✅ LightGBM GPU 训练（v0.8，`--device gpu`）
9. ✅ qlib PyTorch CUDA 集成（v0.9，自写 MLP 走 3060）
10. ✅ 个股深度研究（v0.10，跨市场 14 板块 + MCP 补全建议）
11. ✅ Streamlit 看板加 ML 交互 + 深度研究页（v0.11）
12. ✅ 单元测试固化 + CLI CUDA 泄漏修复（v0.12，35 测试）
13. ✅ 自写 LSTM + CUDA 训练进度 log（v0.13，6×26 时序重塑）
14. ✅ predict-batch 支持自写 LSTM/MLP 模型（v0.14，按 algo 分路）
15. ✅ qlib 数据更新器（v0.15，baostock 续到最新）

## 剩余候选（未做）

- GRU/ALSTM 真单元差异（当前复用 LSTM 路径）
- PyPI 打包 publish（让 `pip install eternityquant` 能装）

## License

MIT
