# EternityQuant

个人散户量化助手 —— 不交易，只提醒和辅助决策。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/colab_eternityquant_train.ipynb)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](notebooks/kaggle_eternityquant_train.ipynb)

当前版本 **v0.32**（日常闭环：`eq daily` 每日晨报 + `eq paper` 纸面战绩——前向的、没法作弊的验证）。

## 当前能力速览

```bash
eq doctor                                # 环境体检（依赖/数据/数据库/通道/连通性）
eq watch 600519.SH                       # 个股快照（A/HK/US/CRYPTO，自动走本地缓存）
eq watch 600519                          # 符号随手写：裸码/小写/SH 前缀都认（v0.24）
eq scan A --by change_pct --top 30       # 四市场扫描（A/HK/US/CRYPTO）
eq screen golden_cross,volume_spike --from watchlist   # 技术选股（14 种条件，v0.24）
eq export --format excel                 # 导出全部数据为 Excel（v0.24）
eq cache stats / warm / clear            # 行情本地缓存管理（v0.24）
eq watchlist add 600519.SH --reason 白酒龙头
eq watchlist quotes                      # 一屏看完所有自选实时行情（并发，v0.24）
eq portfolio buy 600519.SH 100 1680     # 建仓 100 股 @1680
eq portfolio summary                     # 持仓体检 + 集中度/止损覆盖率风险提示（v0.24）
eq monitor add 600519.SH price_cross '{"level":1700,"direction":"up"}' --channels desktop --cooldown 60
eq monitor signals                       # 最近触发的信号历史（v0.24）
eq scheduler add 每日收盘扫描 '0 16 * * 1-5' scan_report --params '{"market":"A","top_n":20}'
eq backtest 600519.SH ema_cross --engine vectorized --detail
eq backtest 600519.SH x --sweep          # 全策略横评并按夏普排名（v0.24）
eq bt list / show <run_id> / remove <run_id>
eq bt compare --symbol 600519.SH         # 多次回测并排比较（v0.24）
eq ml train csi300 5 --algo lightgbm --device cpu      # LightGBM CPU
eq ml train csi300 5 --algo lightgbm --device gpu      # LightGBM GPU（OpenCL）
eq ml train csi300 5 --algo mlp --device cuda          # 自写 MLP 走 CUDA GPU CUDA
eq ml train csi300 5 --algo lstm --device cuda          # 自写 LSTM 走 CUDA（量化选股最佳，6×26 时序重塑）
eq ml train csi300 5 --algo deeplob --device cuda       # DeepLOB: CNN+BiLSTM+Attention（顶会论文复现）
eq ml train csi300 5 --algo tft --device cuda           # Temporal Fusion Transformer（Google 多时间跨度预测）
eq ml train csi300 5 --algo gru --device cuda --optimizer sam --loss sharpe  # SAM 优化器 + 可微夏普比率
eq ml train csi300 5 --algo tft --device cuda --adversarial --orthogonalize   # 对抗训练 + 特征正交化
eq ml train csi300 5 --algo gru --device cuda --optimizer lion --loss ic      # Lion 优化器（Google 进化发现）
eq ml train csi300 5 --algo tft --device cuda --gpus "0,1,2,3"              # 多卡并行（4 张 GPU）
eq ml update-data --start 2020-09-28 --universe csi300  # qlib 数据续到最新（腾讯 API）
eq ml update-data -u csi500 -x SH600519,SZ000001         # 单股 + 预设指数合并下载训练（v0.22）
eq ml update-data -u watchlist                            # 从 D:\idmxz\Table.txt 读自选股下载（v0.22）
eq ml activate <model_id>
eq ml predict-batch <model_id> --top 10                # 批量预测入 ml_predictions 表（v0.14 支持自写模型）
eq data a                                 # A 股日线收集（腾讯 API → qlib .bin）
eq data a -u csi500 -x SH600519              # 单股 + 预设指数合并下载（v0.22）
eq data a -u watchlist                       # 从自选股文件下载（v0.22）
eq data hk                                # 港股日线收集（akshare，全历史 2004~2026）
eq data hk-5min                           # 港股 5 分钟线（yfinance，最近 30 天）
eq data hk-1min                           # 港股 1 分钟线（yfinance，最近 7 天）
eq data us                                # 美股日线（yfinance）
eq data all                               # 全量数据收集
eq dash                                 # 启动 Streamlit 9 页看板
eq --help                               # 看所有命令
```

## 架构原则

- **EternityQuant 自写全部核心引擎**（数据层、信号引擎、回测、监控、推送）。
- **不依赖任何 AI 助手或外部服务**（v0.34 起）—— 纯 Python 进程，装好依赖就能跑，
  所有数据直连公开 SDK（akshare / yfinance / baostock / 腾讯 / 新浪 / 东财）。
  没有 MCP、没有 API key、没有需要登录的账号。
- **qlib 作信号引擎**，预测值作为因子喂给信号层。

## 技术栈

- CLI：`typer`
- 定时：`APScheduler`（`eq scheduler daemon` 常驻）
- Web：`Streamlit`（10 页看板，见下）
- 数据源：A股 baostock（TCP 稳）/ 港美 yfinance / 加密 okx / fallback akshare
- 回测：双引擎（向量化 + 事件驱动），共享 `signal(df) -> df` 接口
- ML：qlib Alpha158 特征 + LightGBM（CPU/GPU）+ 自写 MLP（CUDA，CUDA GPU 主场）

## 配置分层

- `~/.eternityquant/config.yml`：静态配置（可选）
- `~/.eternityquant/.env`：密钥（tushare token、企业微信 webhook）
- `~/.eternityquant/eternityquant.db`：状态库（10 表：watchlist/portfolio/trade_history/rules/signals/ml_models/ml_predictions/ml_runs/scheduled_jobs/backtest_runs）
- `~/.eternityquant/market_cache.db`：行情缓存（可随时删）
- `~/.eternityquant/backtests/<run_id>.parquet`：回测详细数据外存
- `~/.eternityquant/ml_models/*.pkl`：ML 模型文件

### 统一数据目录（v0.20）

**所有市场数据集中在 `~/.eternityquant/data/` 下**，按市场分子目录：

```
.eternityquant/data/
├─ a/                          # A 股（qlib .bin 格式）
│  └─ qlib_cn_data/            # qlib provider_uri
│     ├─ calendars/day.txt     # 交易日历
│     ├─ features/sh600000/    # 每只股票一个目录，含 {open,high,low,close,volume,factor,change}.day.bin
│     ├─ instruments/csi300.txt # 成分股列表
│     └─ all_codes.txt
├─ hk/                         # 港股
│  ├─ daily/                   # 日线 CSV（akshare Sina 源，2004~2026 全历史）
│  ├─ 5m/                      # 5 分钟线 CSV（yfinance，最近 30 天）
│  ├─ 1m/                      # 1 分钟线 CSV（yfinance，最近 7 天）
│  ├─ features/                # 计算后的特征 CSV（训练用）
│  └─ models/                  # 港股模型 pkl
└─ us/                         # 美股
   └─ daily/                   # 日线 CSV（yfinance）
```

**旧目录自动迁移**：第一次访问时 `eq.data.paths.migrate_legacy_data_layout()` 会把散落的 `.qlib_data/cn_data`、`.eternityquant/hk_data`、`.eternityquant/us_data` 复制到统一目录（旧目录保留，不破坏现有脚本）。

## CLI 命令全貌

| 命令组 | 功能 | 版本 |
|--------|------|------|
| `eq watch <symbol>` | 个股快照（A/HK/US/CRYPTO） | v0.1 |
| `eq scan <market> --by --top` | 四市场扫描（A/HK/US/CRYPTO） | v0.4 |
| `eq screen <条件> --from --mode` | 技术选股器（14 种条件，可一键入自选） | v0.24 |
| `eq doctor [--network]` | 环境体检（依赖/数据/库/通道/连通性） | v0.24 |
| `eq export --datasets --format` | 导出 7 类数据为 CSV / Excel | v0.24 |
| `eq cache stats/warm/clear` | 行情本地缓存管理 | v0.24 |
| `eq data sources [--test]` | 数据源注册表：13 个源，本机自检可用性 | v0.26 |
| `eq watchlist add/import/list/remove/find/quotes` | 自选股 CRUD + 批量实时行情 | v0.1/v0.24 |
| `eq portfolio buy/add/trim/sell/list/stops/history/summary/closed` | 持仓全生命周期 + 风险体检 | v0.1/v0.24 |
| `eq monitor add/list/run/enable/disable/signals/cooldown` | 监控规则（11 种类型 + 冷却期 + 信号历史） | v0.1/v0.5/v0.24 |
| `eq scheduler add/list/run/daemon` | 定时推送（APScheduler） | v0.2 |
| `eq backtest ... --engine --sweep --detail` | 双引擎回测 + 全策略横评，自动外存 parquet | v0.1/v0.3/v0.24 |
| `eq bt list/show/remove/compare` | 回测历史管理 + 并排比较 | v0.3/v0.24 |
| `eq bt robust <策略>` | 稳健性验证：多标的分布 + 滚动样本外 + 随机基准 | v0.28 |
| `eq bt optimize <标的> <策略>` | 参数寻优：样本内选参 + 样本外验证 + 高原检测 | v0.28 |
| `eq bt portfolio <策略>` | 组合级回测：资金分配 + 持仓约束 + 真实成本 | v0.29 |
| `eq bt costs` | 各市场真实费率对照（最低佣金对小额交易的影响） | v0.29 |
| `eq bt ml <model_id>` | 把 ML 模型真的跑一遍组合回测 | v0.29 |
| `eq advise <资金>` | 按资金量算持仓数/单笔金额/换手预算 | v0.30 |
| `eq nextday` | 次日高点研究：MFE/MAE 分布 + 限价档位扫描 | v0.31 |
| `eq daily` | 每日晨报：大盘 + 止损警报 + 信号翻转 + 纸面记录 | v0.32 |
| `eq paper` | 纸面战绩：前向推荐 vs 基准的超额 t 检验 | v0.32 |
| `eq ml train/activate/list/info/predict/predict-batch/update-data` | ML 因子（LightGBM + PyTorch + 数据更新） | v0.6~v0.15 |
| `eq data a/hk/hk-5min/hk-1min/us/all` | 统一数据收集（A股/港股日线/分钟线/美股） | v0.19 |
| `eq ml train-local` | 训练（不用 qlib）：本地 Alpha158 + 项目自己的行情缓存 | v0.39 |
| `eq ml predict-local` | 用 train-local 的模型给一批标的打分（不用 qlib）| v0.39 |
| `eq ml factor-scan` | 单因子基准：同一测试段上逐个评估 158 个因子，给模型成绩当参照 | v0.40 |
| `eq ml baseline` | 无参数基准：验证段选因子、测试段评估等权合成，和模型 test IC 直接可比 | v0.41 |
| `eq ml backtest-local` | 把本地模型跑成权益曲线（只在测试段，含 A 股真实成本 + 零成本对照）| v0.42 |
| `eq dash` | Streamlit 9 页看板 | v0.1/v0.11/v0.24/v0.33 |
| `eq theme <图片>` | 看板换肤：自动取色 + 背景图 + 侧栏看板娘 | v0.33 |

### 符号写法（v0.24 起全面容错）

所有命令的 symbol 参数都会先经 `eq.data.market.normalize_symbol()` 规整，随手写就行：

| 你输入 | 规整为 |
|--------|--------|
| `600519.sh` / ` 600519.SH ` / `"600519.SH"` | `600519.SH` |
| `600519` / `000001` / `300750` / `920000` | `600519.SH` / `000001.SZ` / `300750.SZ` / `920000.BJ` |
| `SH600519` / `sz000001`（qlib 风格） | `600519.SH` / `000001.SZ` |
| `700` / `0700.HK` / `09988.hk` | `00700.HK` / `00700.HK` / `09988.HK` |
| `AAPL` / `aapl.us` | `AAPL.US` |

同一只票的不同写法会落到**同一行**自选/持仓，不再重复建仓。


## 策略层（v0.27 重构）

**原来薄在哪**：只有 4 个策略（EMA 金叉 / ADX / RSI / 布林），全是「单指标 + 交叉」
这一个套路；信号只有 BUY/SELL/HOLD 三态，没法表达强弱；回测只有满仓/空仓两档，
没有仓位管理也没有止损。

v0.27 补齐三层：

### 1. 因子库 6 → 21

| 组 | 因子 |
|----|------|
| 波动率 | `true_range` `atr` `natr` `realized_vol` |
| 通道 | `donchian` `keltner` `supertrend` `bollinger_bandwidth` |
| 动量 | `roc` `momentum` `zscore` `trend_strength` |
| 摆荡 | `cci` `williams_r` `stochastic` |
| 原有 | `rsi` `ema` `macd` `adx` `kdj` `bollinger` |

### 2. 策略 4 → 17，且不再同质

```bash
eq backtest 600519.SH x --sweep      # 17 个策略横评
```

| 类别 | 策略 |
|------|------|
| 趋势 | `ema_cross` `adx_trend` `supertrend` |
| 反转 | `rsi_reversal` `bollinger_break` `zscore_reversion` `kdj_cross` `cci_reversal` `reversion_pack` |
| 突破 | `donchian`（海龟）`keltner` `vol_breakout` `breakout_pack` |
| 组合/择时/风控 | `trend_vote` `regime_switch` `trend_vol_filtered` `managed_trend` |

**信号分数化**：策略可返回 `score ∈ [-1,+1]` 而不只是三态。RSI 刚跌破 30 和跌到 12
现在能区分，仓位也能按强弱定。三态与分数可互转，老策略一行不用改。

**多策略投票**（`vote` / `make_vote`）：加权合成 + `min_agree` 要求至少 N 票同向，
压掉单一策略的噪声。

**市场状态自适应**（`regime_adaptive`）：这是对「哪个策略最好」的正解 ——
没有最好的策略，只有适合当前状态的。ADX>25 用趋势跟随、<20 用均值回归，
中间过渡区默认空仓观望。

**前置过滤**（`filtered`）：给任意策略加放量/波动率/收口/乖离闸门。
注意只过滤**买入**信号 —— 离场信号被过滤掉就跑不掉了。

### 3. 仓位管理与风控（`eq.strategy.risk`，全新）

| 功能 | 说明 |
|------|------|
| `volatility_target` | 波动率目标定仓：`仓位 = 目标波动 / 标的波动`。让每笔交易风险贡献相等，而不是被高波动标的绑架 |
| `atr_risk_size` | 按「单笔最大亏损 = 账户 x%」反推仓位（海龟法则） |
| `score_scaled_size` | 按信号分数强弱定仓 |
| `apply_stops` | ATR 止损 / 跟踪止损 / 时间止损 / 止盈，逐 bar 状态机 |
| `drawdown_throttle` | 回撤熔断：超阈值降仓、再超清仓 |
| `build_positions` | 串起来：信号 → 0~1 连续目标仓位 |

**回测引擎已支持连续仓位** —— 策略返回数值序列时按目标仓位调仓（含加仓时的
加权平均成本），返回三态时沿用原语义。

真实数据上的效果（茅台，2024-07 ~ 2026-07，500 根 bar）：

```
方案                                   总收益       夏普     最大回撤   交易
裸 ema_cross                        -18.85%     -0.54    -28.47%    16
三策略投票(≥2票)                     -12.91%     -0.18    -32.57%     1
投票 + 波动率定仓                     -8.62%     -0.16    -22.83%     1
投票 + 定仓 + ATR止损                 -3.87%     -1.00     -3.94%     2
投票 + 定仓 + 跟踪止损(1.5×ATR)       -1.22%     -0.40     -2.09%     2
```

信号本身在这段行情上是亏的（四个方案总收益都为负），但**回撤从 -28.5% 压到 -2.1%**
—— 这正是风控层该做的事：不负责让你赚钱，负责让你亏得起。

### 模型类改名（v0.27）

`_SimpleMLP` / `_SimpleSeqModel` / `AdvancedTrainer` 这类「模糊限定词 + 类型」的名字
换成了按职责命名的 `MLPAlphaNet` / `RecurrentAlphaNet` / `DeepAlphaTrainer`。

**旧名字保留为别名** —— 训练好的模型是 pickle 整个实例存盘的，pickle 记的是
「模块路径 + 类名」，直接改名会让 v0.27 之前存下的所有 `.pkl` 加载不了。






## 日常闭环（v0.32，`eq daily` / `eq paper`）

20 多个命令做出来了，但散户真正的问题是「**每天到底跑什么**」。答案是一条命令：

```bash
eq daily        # 每天收盘后跑一次
```

一次输出四件事：

1. **大盘状态**：沪深300 现价、距 MA200、闸门开关
2. **持仓警报**：谁跌破止损（‼ 按纪律该走了）、谁逼近止损、谁没设止损
3. **今日信号**：自选+持仓里哪些**今天新触发**买入/卖出——只报翻转不报存量，
   因为只有翻转才需要行动
4. **纸面记录**：新买入信号自动记入日志，到期的自动结算，附战绩牌

### 纸面日志：唯一没法作弊的验证

回测再漂亮也可能是过拟合（v0.28 演示过：样本内夏普 +0.57 → 样本外 -2.40）。
唯一诚实的验证是**前向的**：从今天起把每个推荐记下来，持有 N 个交易日后
用真实行情结算，和同期沪深300 比超额。记录之日起的表现没法作弊——
没有窥探未来，没有幸存者偏差，没有挑区间。

```bash
eq paper                    # 战绩牌
eq paper -n 20 --settle     # 先结算到期的，再看最近 20 笔明细
```

战绩牌的核心不是收益率，是**超额收益的 t 统计量**：

```
纸面战绩（trend_vote）
  已结算 37 笔   在途 12 笔   胜率 54%   平均收益 +0.85%
  vs 基准：超额均值 +0.42%   跑赢占比 57%   t=+1.31
  判定：暂时领先但不显著（|t|<2），继续攒样本——按当前水平约需 89 笔
```

|t| ≥ 2 之前，一切领先都可能只是运气。战绩牌会直接告诉你还要攒多少笔
才能下结论——把「别急着下结论」从一句劝告变成一个倒计时。

**这个闭环回答的正是最重要的那个问题**（v0.30 提出的）：
你的选股到底能不能跑赢宽基。回测说了不算，纸面日志说了算。

## 次日高点预测（v0.31，`eq nextday`）

T 日收盘买入、T+1 卖出（A 股 T+1 下最短的合法持有期），目标卖在次日高点附近。

**先破除一个幻觉**：「按第二天最高价卖出」不是策略，是**未来函数** ——
你事先不知道高点在哪。能执行的只有一种形式：提前挂**限价单**。
于是问题变成「限价该挂多高」。`simulate_limit` 正确建模了成交：
跳空高开按开盘价成交（优于限价）、摸到按限价、没摸到收盘平仓、
T 日涨停封板买不进。

### 实测结论（10 只 A 股 × 600 bar）

```
次日最高/今收-1（MFE）  中位 +0.766%
次日最低/今收-1（MAE）  中位 -0.764%
上行/下行空间比 0.93     ← 略微不利
次日收盘收益            均值 +0.034%，上涨占比 46.7%

限价档     成交率   捕获率   净收益/笔    年化中位
+0.3%      75.6%   30.1%   -0.2970%    -52.1%
+1.0%      40.8%   29.9%   -0.3155%    -52.8%
+5.0%       3.0%   27.0%   -0.2855%    -48.9%
```

**所有档位净收益都在 -0.29% 左右，年化 -50%。** 原因是算术的：
一个来回成本 0.302%，而次日收盘收益均值只有 +0.034%。
限价挂多高都改变不了这个结构 —— 限价止盈砍掉上涨、保留下跌，是负期望操作。

### ML 能预测次日高点吗？能，但没用

训了个 LightGBM 预测 MFE，测试集上预测与真实 MFE 相关性 **+0.237** —— 模型确实有效。
按预测值分 5 组：

```
组     预测MFE    实际MFE    实际MAE    次日收盘收益
Q1    +0.407%   +0.786%   -0.761%      -0.014%
Q5    +2.033%   +1.799%   -1.452%      +0.081%
```

**Q5 的 MFE 确实高得多，但 MAE 同步恶化，次日收盘收益几乎没改善。**
模型学到的是**波动率**，不是**方向** —— 波动大不等于能赚钱。
用「预测 MFE 前 20%」做入场筛选后，净收益仍是 -0.27%~-0.30%/笔。

### 要让它成立需要什么

必须有**方向性 alpha**：把选出股票的次日收盘收益均值抬高 **0.27 个百分点**以上
（约 0.16 个标准差）。这是个很高的门槛 —— 先用 `eq bt robust` 确认你的信号
有没有这个能力，再考虑做日内来回。

顺带修了数据层一个缺口：`_norm_bars` 现在校验 OHLC 自洽性
（13 个数据源质量参差，偶尔给出 `open > high` 这种不可能的 K 线，
会让限价回测算出高于当日最高价的成交价）。

## 散户 · 只做多（v0.30）

针对「资金量小 + 只做多 + 没时间盯盘」这个画像。下面每条都有实测数据支撑，
**包括一条结论是负面的**。

### 1. 执行延迟：先把虚高的数字挤掉

原来的引擎是「收盘价看到信号 → 同一根收盘价成交」。散户看到收盘价时已经收盘了，
实际最快次日才能交易。加上 `execution_delay=1` 后（15 只 A 股 × 900 bar）：

```
方案                        总收益      夏普     最大回撤
基线（当日收盘成交）         +2.51%    +0.13    -29.50%
+ 执行延迟 T+1              -1.19%    +0.07    -30.76%
```

**同一个策略，仅仅因为改成次日成交就从 +2.5% 变成 -1.2%。**
`PortfolioConfig.execution_delay` 默认就是 1 —— 组合调仓要动一篮子票，
更不可能在收盘瞬间完成。单标的引擎默认仍是 0（不破坏老结果），
但**做决策前请务必设成 1**。

### 2. 资金量决定你能持几只（`eq advise`）

```bash
eq advise 100000
```

最低佣金 5 元是硬约束：单笔金额太小，实际费率会成倍上升。

```
资金 100,000 元
  建议持仓数     14 只        单只金额    7,143 元
  单笔最低金额   6,667 元     单边费率    万 7.1
  一个来回要涨   0.342% 才回本
  换手预算：每只每年最多换 5.8 次，平均持有 43 个交易日
```

资金 3 万时只能有效持 4 只；5 千时连 1 只的合理仓位都不够 —— 那种情况下
宽基 ETF 比自己选股更合理（费率低、天然分散）。

### 3. 大盘闸门：**实测不支持"它更好"**

「指数跌破 200 日均线就清仓」流传很广。本项目在 15 只 A 股 × 900 bar
（2022-11~2026-07）被动等权持有上的对照：

```
闸门        总收益     最大回撤   换手/年
无         -17.84%    -32.89%      0.8
MA100      -26.07%    -29.76%      2.6
MA150      -31.50%    -34.98%      2.5
MA200      -22.15%    -26.13%      1.6
MA250      -16.28%    -20.43%      1.0
```

**回撤基本都改善了（这是它的本职），但收益没有变好**，换手上升，
且对均线长度极其敏感 —— MA150 和 MA250 差 15 个百分点，这种参数敏感度
本身就是危险信号。

所以它是**风险管理工具，不是收益增强工具**。用之前请在你自己的标的池上
`eq bt portfolio --market-filter 000300.SH` 验证。

### 更重要的一个观察

同一窗口里**沪深300 自己涨了 +24%**，而上面这 15 只精选股的等权组合是 **-17.8%**。
问题出在选股，不在择时 —— 再精妙的仓位管理和大盘闸门也救不回来。
对多数散户来说，**先确认自己的选股能跑赢宽基**，再谈优化。

## 组合回测 / 真实成本 / ML 接入（v0.29）

### 1. 真实交易成本（`eq bt costs`）

原来只有双边固定 bps。A 股实际是**佣金万 2.5 且每笔最低 5 元** +
**印花税千 1 仅卖出** + 过户费万 0.1 双边。最低佣金这条对散户影响最大：

```
成交金额   A 股单边(万分之)   来回盈亏平衡
  2,000        25.1           0.702%
  5,000        10.1           0.402%
 20,000         2.6           0.252%
200,000         2.6           0.252%
```

2000 元的单子实际费率是 20000 元单子的 **10 倍**。同一个策略：

```
      本金      旧 bps 模型   A 股真实成本   被吃掉
     3,000       48.75%       42.83%      -5.92%
   200,000       48.75%       47.07%      -1.68%
```

预设：`a_share` / `hk`（印花税双边）/ `us`（零佣金 + SEC fee）/ `crypto` / `flat`（旧行为）。
`cost_model=None` 时完全退回旧的 bps 行为，老代码不受影响。

### 2. 组合级回测（`eq bt portfolio`）

之前所有回测都是单标的、满仓/空仓，和「一笔钱同时管十几只票」差得很远。

```bash
eq bt portfolio trend_vote --from watchlist --max-positions 6 --alloc inverse_vol
eq bt portfolio trend_vote --from watchlist --compare      # 对比三种资金分配
```

- **三种分配**：等权 / 波动率反比（风险平价简化版）/ 信号强弱加权
- **约束**：最大持仓数、单票权重上限、最小权重（太小的仓位会被最低佣金吃掉）、现金缓冲
- **调仓节奏**：signal / daily / weekly / monthly
- **输出**：权益曲线、逐日权重、逐标的贡献、换手率、已实现/未实现拆分

分散化收益是实打实的 —— `trend_vote` 在**茅台单只上 -12.7%**，
同样的策略做成 **6 只的组合是 +24.1%**（夏普 0.68，回撤 -15%）。

也支持直接喂**日期 × 标的的分数矩阵**（横截面策略，ML 选股就属于这类）。

### 3. ML 模型接入回测（`eq bt ml`）

项目里原本有两套隔离的评估体系：ML 层算 IC/ICIR，策略层算夏普/回撤。
问题是 **IC 高不等于能赚钱** —— IC 完全不考虑交易成本、换手率、持仓数约束。

用一个合成模型（Rank IC = +0.259，ICIR = 0.84，按 IC 标准是很好的因子）实测：

```
持有期    总收益     夏普      回撤    换手x/年   交易
  1     -4.48%   -0.13   -20.04%    253.0    812
  5    +46.75%   +1.61    -6.59%     56.1    178
 20     +5.01%   +0.27   -15.51%     12.6     38
```

**同一个模型、同样的 IC**，每天换手就亏 4.5%，持有 5 天就赚 46.8%。
差别全在换手率 —— 253x/年的换手被印花税和最低佣金吃光了。
模型训完必须真的跑一遍回测，光看 IC 会做出完全错误的判断。

```bash
eq bt ml <model_id> --top 10 --hold 5      # 每期选前 10 只，持有 5 天
```

## 策略稳健性验证（v0.28）

**为什么必须有**：`eq backtest --sweep` 是在**一只标的、一段区间**上跑的，
这种结果基本没有决策价值 —— 17 个策略里挑最好的那个本身就是一次多重比较，
纯噪声下也必然有一个"看起来很好"。ML 层在 v0.25 修过同一类问题
（验证集既选模型又报成绩、没有 purge），策略层这里补齐对应的东西。

```bash
eq bt robust trend_vote --from watchlist --days 600    # 多标的 + 滚动样本外 + 随机基准
eq bt optimize 600036.SH ema_cross --days 800          # 样本内选参 + 样本外验证
```

### 三个视角

**1. 多标的分布** —— 看中位数和盈利占比，不看单点

```
多标的稳健性（8 只标的）
  sharpe 中位数 +0.45   均值 +0.24   区间 [-1.45, +1.03]
  盈利标的占比 75%   收益中位数 +15.43%   最差回撤 -49.64%
```

一个策略在 8 只票上有 6 只赚钱、夏普中位数 0.45，比在 1 只票上夏普 1.5 有说服力得多。
均值会被极端样本带偏，所以主看中位数。

**2. Walk-Forward 样本外** —— 段间 purge，检验跨时段稳定性

```
窗口    测试区间                     总收益      夏普      回撤   交易
1     2025-04-30~2025-07-28       +2.35%   +0.70   -6.37%     0
...
  sharpe 中位数 +0.00   为正的窗口占比 40%   窗口间标准差 1.59
  判定：多数窗口不赚钱（中位数 +0.00），策略在这个标的上不成立
```

为什么要 embargo：策略用的指标（如 60 日均线）在训练段末尾和测试段开头是
**同一批数据算出来的**，紧邻切分会让"样本外"沾到样本内的信息。

**3. 随机基准** —— 回答「这策略比瞎买强吗」

```
真实夏普 +0.60   随机均值 +0.04 ± 0.44   百分位 88   p=0.120
判定：略优于随机但不显著
```

关键是**交易频率匹配**：拿高频策略去比低频随机基准，差异全来自交易成本
而不是选时能力。

### 参数寻优必须带样本外

直接在全样本上网格搜索再报最优值，等价于 ML 里"验证集既选模型又报成绩"。
`eq bt optimize` 把数据切两段：前段选参数，后段（没见过）报成绩。

实测（招商银行 800 根 bar，ema_cross 25 组参数）：

```
最优参数：{'fast': 5, 'slow': 20}
  样本内 sharpe +0.571  →  样本外 -2.399   衰减 520%   参数高原分 0.25
  判定：样本外为负（-2.40），典型过拟合
```

没有这一步，人就会得出「fast=5/slow=20 最优，夏普 0.57」然后拿去实盘。

**参数高原分**是判断过拟合最实用的一招：取表现前 20% 的参数组，看它们在参数
空间里是否彼此靠近。1 = 最优点周围是一片高地（真有效），0 = 孤立尖峰
（旁边一格就掉下去，拟合的是噪声，换个市场必然失效）。

## 数据源注册表（v0.26）

此前数据源写死在 `market.py` 里：A 股 `baostock→akshare`，港/美 `yfinance→akshare`。
问题是**不同人的网络环境差别极大**——东财/腾讯/新浪国内直连很快、海外可能完全不通；
yfinance 在国内常被限流。写死优先级注定有人不合适。

v0.26 改成注册表 + 自检：**13 个源**按优先级自动 failover，
再用 `eq data sources --test` 在**你自己的机器上**实测一遍，把真正通的排到前面。

```bash
eq data sources              # 看有哪些源、各自支持什么
eq data sources --test       # 在本机实测（结果存进 .eternityquant/source_health.json）
eq data sources --test -m A  # 只测 A 股

eq watch 600519 --realtime           # 走实时源，盘中返回当前价
eq watch 00700.HK -r -S tencent      # 强制指定源
eq watchlist quotes --realtime       # 批量接口，几十只一次请求问完
```

| 源 | 市场 | 能力 | 说明 |
|----|------|------|------|
| `sina` 新浪财经 | 快照 A/HK/US；**K 线仅 A** | bars/snapshot/**batch** | 免费无 key，快照实测最快（~0.05s）。K 线端点不认港/美代码 |
| `tencent` 腾讯财经 | A/HK/US | bars/snapshot/**batch** | 免费无 key，日 K 前复权，三市场全覆盖 |
| `eastmoney` 东方财富 | A/HK/US | bars/snapshot/spot | A 股全市场 5500+ 只一次拉完 |
| `netease` 网易财经 | A | bars | 全历史 CSV（1990 至今），带换手率/市值 |
| `yahoo` Yahoo(直连) | A/HK/US/CRYPTO | bars/snapshot | 不经 yfinance，省依赖省开销 |
| `binance` | CRYPTO | bars/spot | 加密 K 线 + 24h 全市场 |
| `okx` (ccxt) | CRYPTO | spot | 1200+ 交易对 |
| `coingecko` | CRYPTO | snapshot | 免费无 key |
| `baostock` / `yfinance` / `akshare` | — | — | 原有源，包进注册表统一 failover |
| `tdx` 通达信(mootdx) | A | bars | TDX 二进制协议，无 HTTP 限流 |
| `tushare` Tushare Pro | A | bars | 需 `.env` 配 `TUSHARE_TOKEN` |

**批量快照**：新浪/腾讯支持一次请求问几十只。自选/持仓那种「一屏几十只」的场景，
从「N 次网络往返」压成 **1 次**——实测 15 只 0.12s。`realtime=True` 时自动启用，
批量没覆盖到的少数标的再逐只补。

**加密市场是最大受益者**：此前只有 OKX 的 spot（连日线都没有），
现在有 binance/yahoo/yfinance 三个 bars 源，`eq backtest BTC-USDT` 才真正能跑。

新增一个源只要写好 fetch 函数并 `register()` 一下，调用方完全不用改。

### 声明必须按能力分，不能按源分

第一版把 `markets` 写成对**所有能力**统一，这是错的 —— 真实的源不这么整齐：

- 新浪：快照覆盖 A/HK/US，但 K 线端点（`CN_MarketDataService`）**只认 A 股**，
  `hk00700` / `rt_hk00700` / `00700` 各种写法实测一律返回 `null`
- 腾讯美股 K 线**必须带交易所后缀**：`usAAPL` 只返回 1 根（当日），
  `usAAPL.OQ`（NASDAQ）才返回完整历史；`.N` 是 NYSE
- 东财美股 secid 同理：`105`=NASDAQ / `106`=NYSE / `107`=AMEX

谎报支持的后果不只是「调用失败」——它还会占住 failover 链的前排位置，
把真正能用的源挤到后面。所以 `DataSource` 加了 `cap_markets`
做分能力覆盖声明，美股 K 线则改成挨个试交易所后缀。

## 行情本地缓存（v0.24）

`market_cache.db` 的 `bar_cache` 表建库起就存在，但此前**没有任何代码读写它**——
每次 `eq watch` / `monitor run` / `backtest` / `screen` 都在重新打网络拉同一段日线。
v0.24 把它接上：

```bash
eq cache warm --from watchlist -d 400   # 预热：把自选日线一次拉到本地
eq cache stats --detail                 # 看缓存了多少标的多少行、占多大
eq cache clear -s 600519.SH             # 清单只 / 不带 -s 清全部
```

- 默认 TTL 6 小时，命中就不打网络（`--no-cache` 可强制走网络）
- **全部数据源都挂时退化用过期缓存**，而不是直接报错
- 预热之后 `eq screen` / `eq backtest` 基本可以离线跑

## 技术选股器（v0.24）

```bash
# 自选股里找 EMA 金叉 + 放量的
eq screen golden_cross,volume_spike --from watchlist

# A 股成交额前 300 里找超卖反弹（任一条件满足即命中），命中的打标签入自选
eq screen rsi_oversold,near_low --from A --top 300 --mode any --add-to-watchlist 抄底候选

# 自定义条件参数
eq screen rsi_oversold,above_ma --params '{"rsi_level":25,"ma_period":60}'
```

| 条件 | 含义 | 主要参数 |
|------|------|---------|
| `rsi_oversold` / `rsi_overbought` | RSI 超卖 / 超买 | `rsi_level`, `rsi_period` |
| `golden_cross` / `death_cross` | EMA 金叉 / 死叉 | `fast`, `slow`, `lookback` |
| `macd_golden` / `macd_death` | MACD 金叉 / 死叉 | `lookback` |
| `above_ma` / `below_ma` | 站上 / 跌破均线 | `ma_period` |
| `volume_spike` | 放量（当日量 / 20 日均量） | `volume_multiple` |
| `near_high` / `near_low` | 接近 N 日新高 / 新低 | `high_period`, `tolerance_pct` |
| `breakout` | 收盘创 N 日新高 | `breakout_period` |
| `pullback` | 上升趋势中回踩 MA20 | `tolerance_pct` |
| `squeeze` | 布林带收口（变盘前兆） | `squeeze_period`, `squeeze_quantile` |

## 数据收集

`eq data` 命令组统一管理多市场数据收集。**所有数据落到 `~/.eternityquant/data/{a,hk,us}/` 下统一目录**（v0.20 起取代散落的 `.qlib_data/cn_data`、`.eternityquant/hk_data`、`.eternityquant/us_data`）。

```bash
# === A 股（qlib 本地数据集，腾讯 API 续期） ===
eq data a --start 2026-01-01 --universe csi300 --workers 10   # 沪深 300
eq data a --start 2026-01-01 --universe csi500 --workers 10   # 中证 500
eq data a --start 2026-01-01 --universe watchlist --workers 10  # 从 D:\idmxz\Table.txt 读自选股（v0.22）
# 单只股票 + 预设指数合并下载（v0.22）：
eq data a --start 2015-01-01 --universe csi500 --extra SH600519,SZ000001 --workers 5
# 首次使用需先解压 qlib_cn_data 到 data/a/qlib_cn_data/，再续期：
eq ml update-data --start 2020-09-28 --universe csi300        # 腾讯 API 拉 6 年日线

# === 港股（akshare Sina 源，全历史 2004~2026） ===
eq data hk --top 73                    # 港股日线 → data/hk/daily/
eq data hk-5min --top 73               # 港股 5 分钟线 → data/hk/5m/（yfinance，最近 30 天）
eq data hk-1min --top 73               # 港股 1 分钟线 → data/hk/1m/（yfinance，最近 7 天）

# === 美股（yfinance） ===
eq data us --top 31                    # 美股日线 → data/us/daily/

# === 全量（按上面顺序串行） ===
eq data all --top 73                   # 全量收集
```

| 数据源 | 市场 | 类型 | 历史长度 | 落盘位置 |
|--------|------|------|---------|---------|
| 腾讯 API `web.ifzq.gtimg.cn` | A 股 | 日线 | 2001~2026 | `data/a/qlib_cn_data/` |
| akshare `stock_hk_daily` | 港股 | 日线 | 2004~2026 | `data/hk/daily/` |
| yfinance | 港股 | 5m/1m | 30天/7天 | `data/hk/{5m,1m}/` |
| yfinance | 美股 | 日线 | ~2年 | `data/us/daily/` |

### A 股数据抓取三项特性（v0.22）

1. **单只股票 + 预设指数合并下载训练** — `--universe` 指定指数成分股（csi300/csi500/all），`--extra`/`-x` 逗号分隔额外股票代码，两者合并去重后写入同一份 `instruments/<universe>.txt`，训练时 qlib 自动读取合并池。亦支持 `--universe watchlist` 从 `D:\idmxz\Table.txt` 提取 A 股代码。

2. **跳过较晚股票没上市的时间不重试** — 腾讯 API 返回空数据时判定为未上市/已退市/区间在上市前，**直接写全 NaN 跳过不重试**；只有网络异常才指数退避重试 3 次。未上市股在 `instruments` 文件中可用区间为空，qlib 训练时自动忽略。

3. **下载先后顺序无关，结果一致** — 每次更新 `[start, end]` 区间都整段重算并**覆盖写** `.bin` 文件（`_write_bin` 用 `"wb"` 整段写，替代旧追加模式）；`calendars/day.txt` 合并去重升序后整段覆盖写；`instruments/<universe>.txt` 从 `close.day.bin` 扫描首/尾非 NaN 索引映射回日历日期生成。因此无论先下 2016 再下 2026、还是反之，最终 `.bin` 与 `instruments` 文件内容逐字节一致。

### 从零开始的完整数据流程

```bash
# 1. A 股：先准备 qlib 数据集（解压 cn_data.tar.gz 或下载官方 datasets）
#    放到 ~/.eternityquant/data/a/qlib_cn_data/
#    再续期到最新：
eq ml update-data --start 2020-09-28 --universe csi300 --workers 5
# 或合并单股 + 指数一起下载训练：
eq ml update-data --start 2015-01-01 --universe csi500 --extra SH600519 --workers 5

# 2. 港股：拉日线 + 分钟线
eq data hk --top 73
eq data hk-5min --top 73
eq data hk-1min --top 73

# 3. 美股：拉日线
eq data us --top 31

# 4. 验证：查看统一数据目录
ls ~/.eternityquant/data/a/qlib_cn_data/features/ | head    # A 股 .bin
ls ~/.eternityquant/data/hk/daily/ | wc -l                  # 港股日线文件数
ls ~/.eternityquant/data/us/daily/ | wc -l                  # 美股日线文件数
```

> 💡 **旧数据自动迁移**：第一次运行任何 `eq data` 或 `eq ml` 命令时，`eq.data.paths.migrate_legacy_data_layout()` 会自动把旧散落目录的数据复制到统一目录，旧目录保留不破坏现有脚本。

## 监控规则 11 种类型

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

**冷却期（v0.24）**：`--cooldown <分钟>` 让一条规则触发后 N 分钟内不再重复推。
定时任务每 5 分钟跑一次 `monitor_run` 时，一条"跌破 1700"的规则此前会每 5 分钟推一遍。

```bash
eq monitor add 600519.SH price_cross '{"level":1700,"direction":"up"}' --cooldown 60
eq monitor cooldown 3 30    # 给已有规则 #3 改成 30 分钟冷却，0 = 取消
eq monitor signals -n 30    # 看最近触发历史（signals 表，v0.24 起真的会写）
```

**涨跌停幅度按板块区分（v0.24）**：`limit_up`/`limit_down` 此前一律按 ±10% 算，
创业板（30xxxx）/科创板（688xxx）实际 ±20%、北交所 ±30%，这些板块的规则永远触发不了。

## ML 因子层

### ⚠️ v0.25 之前的 IC 不可用于调参

v0.25 对三条训练链路（A股 qlib / 港股 / AdvancedTrainer）做了一轮方法学审查，
发现**报出来的 IC 系统性虚高**，在此之上调超参等于在拟合噪声。四个来源：

| # | 问题 | 后果 |
|---|------|------|
| 1 | **验证集既选模型又报成绩**。`fit()` 用验证 IC 做 early stopping + `best_state` 选择，训练函数又把同一个 `best_score`（200 个 epoch 里的**最大值**）当模型成绩报出去 | 在选择集上报最大值。纯噪声跑 200 轮也能"得到" IC=0.03~0.05 |
| 2 | **IC 算的是 pooled Pearson**，即所有日期所有股票混在一起算相关 | 业界的 IC 指每日横截面相关再对日期平均。一个只会预测大盘、零选股能力的模型，pooled IC 能到 0.9+（见 `tests/test_evaluation.py::test_pooled_ic_is_inflated_vs_daily_ic`） |
| 3 | **没有 purge**。标签是 `Ref($close,-h)/Ref($close,-1)-1`，训练集尾部 h 天的标签已经看过验证期价格 | 每次切分都漏一次。旧默认验证区间只有 19 个交易日，h=5 时 **1/4 的验证期被训练集看过** |
| 4 | **港股链路按行切分**。样本按标的依次 append，`int(len(X)*0.8)` 切出来的"验证集"是**最后那批股票**，两段时间范围完全重叠 | 训练集见过验证期的全部行情 |

合成数据上的实测对比（12 只票 / 330 个交易日 / GRU）：

```
[旧] 按行切 80/20  train 2023-04-03~2024-07-05 (330日, 10只)
                   valid 2023-04-03~2024-07-05 (330日,  3只)  ← 时间完全重叠
[新] 按时间切+purge train 2023-04-03~2024-01-18
                   valid 2024-01-26~2024-03-28
                   test  2024-04-05~2024-07-05   (purge=5日)

旧口径 验证段 pooled IC（epoch 最大值）  +0.1582
新口径 测试段 Rank IC（模型没见过）      +0.0405   ← 约 3.9× 虚高
```

修完之后：`eq ml train` / `eq hk train` 报的是**独立测试段的 Rank IC + ICIR + t 统计量**，
旧口径的验证段数值仍以 `valid_ic` 保留但标注"仅供参考"。

```bash
# 新增的训练策略参数
eq ml train csi300 5 --algo lightgbm --test-ratio 0.2 --embargo 5 --seed 42
eq ml train csi300 5 --algo gru --features Alpha360   # RNN 用真时序特征集
eq hk train --test-ratio 0.2 --cs-norm --seed 42
```

| 参数 | 作用 |
|------|------|
| `--test-ratio` | 从验证区间尾部切出的**独立测试段**占比（默认 0.2）。0 = 沿用旧行为 |
| `--embargo` | 段间 purge 的交易日数，缺省 = `horizon` |
| `--seed` | 随机种子。此前无种子控制，同一条命令两次跑出的 IC 能差一大截，根本没法判断调参有没有用 |
| `--features` | `Alpha158`（截面因子）\| `Alpha360`（6 价量字段 × 60 天，**真时序**） |
| `--cs-norm` | （港股）标签按日横截面 rank 归一化 |

### 早停口径对齐（v0.36）—— v0.25 那次审查的漏网之鱼

v0.25 修好了**报告**口径（`evaluate` 改成每日横截面 Rank IC），但**早停**口径
一直没动：`MLPAlphaNet.fit` / `RecurrentAlphaNet.fit` 里挑 `best_state` 用的
仍然是内联算的**池化 Pearson IC**。也就是——

> 拿一把尺选 checkpoint，拿另一把尺给它打分。

这正是上表问题 #2 在训练循环里的残留。后果不是数字虚高（成绩早就按 Rank IC 报了），
而是**选错模型**：池化口径下「能区分牛市日和熊市日、但当天一只票都选不出来」的
checkpoint 得分很高，于是早停可能就停在它身上。选的和考的不一致，超参调得再细
也是在优化错的目标。

v0.36 把两处内联计算都换成直接调 `evaluation.daily_ic`——全项目只留一个 IC 定义。
`tests/test_train_hparams.py` 里构造了一个两种口径结论**相反**的例子钉住这件事：
标签由「日内共同偏移」主导时，只会预测大盘的模型 pooled IC > 0.9，每日截面 IC ≈ 0。

### 学习率与优化器（v0.36）

Lion 的更新量是 `sign(...)`，每个坐标恒定走 ±lr，和梯度大小无关；AdamW 的更新量
被 `g/√v` 自适应缩放过。所以**同一个 lr 在 Lion 下的实际步长大得多**——Lion 论文
（Chen et al. 2023）明确建议 lr 取 AdamW 的 1/3~1/10、weight_decay 放大 3~10 倍，
大致保持 lr×wd 乘积不变。

v0.32 把默认优化器切成了 Lion，但 lr / weight_decay 还留着 AdamW 那套
（1e-3 / 1e-5），等于让 Lion 用约 10 倍的步长跑。v0.36 改成按优化器给默认值：

| 优化器 | lr | weight_decay |
|--------|-----|--------------|
| lion（默认） | 1e-4 | 1e-4 |
| adamw | 1e-3 | 1e-5 |

RNN 路径的 weight_decay 例外，独立定在 1e-6——它对权重衰减极敏感，过强会把权重
压向 0、加剧输出塌缩（实测出现过 `pred.std()=0` 恒 IC=0）。

顺带修的两个静默失效：

- `--lr` / `--weight-decay` **此前根本不存在**，两个值写死在调用点，想调也调不了
- `--optimizer` 对 gru/lstm/mlp **静默失效**——那个分支压根没把它传下去
  （和之前 `--dropout` 一模一样的毛病）。`tests/test_train_hparams.py` 加了一条
  源码级守卫，两个调用点少透传任何一个参数都会失败

```bash
eq ml train gru --optimizer adamw --lr 5e-4 --weight-decay 1e-5
eq ml train mlp                      # 不传就按优化器取默认
```

**这两项改动都还没在真实行情上做过 A/B**（本机 qlib 数据目录是空的）。
理由是原理性的，不是实测的。想验证就固定 `--seed` 跑两次对比 `test_ic`：

```bash
eq ml train gru --seed 42 --lr 1e-3 --name lr1e-3   # 旧默认
eq ml train gru --seed 42 --name lr-default          # 新默认
eq ml list                                           # 比 test_ic，不是 valid_ic
```

### 多种子集成（v0.37）

低信噪比数据上单次训练的方差极大：同一份数据、同一套超参，换个种子 test IC
能差出一倍。集成降的是**方差**，不需要任何调参运气，是这类场景里最稳的一招。

```bash
eq ml train gru --seeds 5      # 跑 5 个种子取平均，训练时间线性增加
```

成员预测**先标准化再平均**：各模型输出尺度不可比（MSE 训出来的回归值量纲不同），
直接算术平均等于给输出方差大的成员更高话语权。标准化是单调变换，不改变任何
单个模型的截面排序。注册表里同时记 `n_seeds` 和 `member_ics`——集成没跑赢
最好的成员时能一眼看出来。

### 训练与推理的特征处理必须一致（v0.37 修）

`predict_batch` 里的 `infer_processors` 是手抄的，和训练那份对不上三处：

| | 训练 | 预测（修复前） |
|---|---|---|
| ProcessInf | 有 | **没有** |
| RobustZScoreNorm | `fields_group="feature"`, `clip_outlier=True` | 裸调用，两个参数都没有 |
| CSRankNorm | 只作用在 **label** | **加在特征上**（训练时根本没这步）|

也就是模型训练时吃的是 z-score 特征，上线推理时吃的是横截面 rank（[0,1] 均匀分布）
——分布完全不同。这类 train/serve skew 不报错、不产生 NaN，只是安静地把每一次
`eq ml predict` 的结果打偏。现在统一由 `ml_workflow.infer_processors()` 提供，
全文件只此一份，测试里有源码级守卫。

### 评估口径（v0.25）

`eq.strategy.factors.evaluation` 提供业界标准口径，训练结束自动打印：

- **Rank IC**：每日横截面 Spearman 相关，再对日期求均值
- **ICIR** = `mean(daily_IC) / std(daily_IC)` —— **比 IC 本身更重要**，衡量信号稳定性，经验上 > 0.3 才值得上实盘
- **t 统计量** = `ICIR × sqrt(n_days)`，`|t| > 2` 才谈得上统计显著
- **分层收益**：按预测分数分 5 组的未来收益 + 多空价差 + 单调性检查。IC 高但分层不单调 = 被少数极端值撑起来的，实盘不可用

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
| `deeplob` | `cuda` | **DeepLOB: CNN(1×2)+BiLSTM(64)+Attention** — 顶会论文复现 | 见实盘结果 |
| `tft` | `cuda` | **Temporal Fusion Transformer: 多头注意力+GRN** — Google 论文复现 | 快速测试 +0.2106 |

### 数据标准化（v0.20，对标 qlib 官方 benchmarks）

**为什么必须标准化**：Alpha158 的 158 维原始特征尺度差异巨大（价格元级、成交量千万级、收益率百分比级），不标准化会污染 LightGBM 的树分裂、让 BatchNorm1d 梯度爆。这是之前 `train()` 路径 `infer_processors=[]` 的核心 bug。

本框架采用 qlib 官方 benchmarks（GRU/ALSTM/LightGBM）同款处理器链，参考 [microsoft/qlib examples/benchmarks](https://github.com/microsoft/qlib/tree/main/examples/benchmarks)：

**特征处理器（`infer_processors`）**：

| 处理器 | 作用 | 权威依据 |
|--------|------|---------|
| `ProcessInf` | 把 Inf 替换为列均值 | qlib `_DEFAULT_INFER_PROCESSORS` |
| `RobustZScoreNorm(clip_outlier=True)` | **MAD（中位绝对偏差）z-score**，截断 3σ 外极值 | qlib GRU benchmark；MAD 比 std 抗异常 |
| `Fillna` | NaN 填 0 | qlib Fillna processor |

**标签处理器（`learn_processors`）**：

| 路径 | 标签处理 | 权威依据 |
|------|---------|---------|
| `train()` (LightGBM) | `DropnaLabel` → `CSZScoreNorm` | qlib LightGBM benchmark 配置 |
| `train_torch()` (GRU/ALSTM/LSTM/MLP/TFT/DeepLOB) | `DropnaLabel` → **`CSRankNorm`** | qlib GRU/ALSTM benchmark 标准配置 |

**为什么 `train_torch` 用 `CSRankNorm` 而非 `CSZScoreNorm`**：
- `CSZScoreNorm`（横截面 z-score）对异常值敏感，一只暴涨股会拉偏全截面均值
- `CSRankNorm`（横截面排序归一化）把未来收益转成 `[0, 1]` 均匀分布，**天然免疫异常值**
- qlib 官方 GRU/ALSTM/LSTM benchmark 全部用 `CSRankNorm`，这是社区验证的最佳实践

**标准化三原则**（webfetch 权威资料综合）：
1. **只用训练集统计量**：`fit_start_time` / `fit_end_time` 限定处理器学习参数的范围，绝不能用测试集均值/方差（数据泄露）
2. **横截面优先**：跨股票比较时用横截面 z-score（`CSZScoreNorm`）或横截面排序（`CSRankNorm`），而非时序 z-score
3. **抗异常值**：用 MAD（`RobustZScoreNorm`）替代 std（`ZScoreNorm`），或用 rank 替代 magnitude

### 高级训练参数（v0.16+）

`eq ml train` 新增一系列机构级训练参数，对标华尔街量化团队：

```bash
# 优化器选择
eq ml train csi300 5 --algo tft --optimizer adamw    # AdamW（解耦权重衰减，默认）
eq ml train csi300 5 --algo gru  --optimizer sam      # SAM（Sharpness-Aware Minimization，平坦极小值搜索）
eq ml train csi300 5 --algo mlp  --optimizer lookahead # Lookahead（k步前看，1步后收）
eq ml train csi300 5 --algo lstm --optimizer lion      # Lion（Google 进化搜索，只看梯度符号）

# 损失函数
eq ml train csi300 5 --algo tft --loss sharpe   # 可微夏普比率（直接优化风险调整收益，默认）
eq ml train csi300 5 --algo gru --loss mse      # 均方误差（传统回归损失）
eq ml train csi300 5 --algo mlp --loss ic       # 负 IC 损失（最大化信息系数）

# 对抗训练 + 特征正交化
eq ml train csi300 5 --algo deeplob --adversarial          # FGSM 对抗训练（忽略微小价格波动）
eq ml train csi300 5 --algo tft --orthogonalize             # 特征正交化去 Beta（学纯 Alpha）
eq ml train csi300 5 --algo gru --adversarial --orthogonalize # 两者结合

# 高级网络参数
eq ml train csi300 5 --algo deeplob --dropout 0.4 --seq-len 120  # DeepLOB: 120 步窗口, 40% dropout
eq ml train csi300 5 --algo tft --dropout 0.3 --heads 4 --hidden 256  # TFT: 4头注意力, 256隐藏
```

#### 优化器对比

| 优化器 | 论文 | 核心思想 | 金融优势 | 推荐场景 |
|--------|------|---------|---------|---------|
| **AdamW** | Loshchilov & Hutter, 2019 | 解耦权重衰减 | 真正落实正则化，防止过拟合历史噪音 | 所有模型基线 |
| **SAM** | Foret et al., ICLR 2021 | 寻找平坦极小值 (Flat Minima) | 市场环境漂移时损失仍保持低水平，防"见光死" | 实盘前最后优化 |
| **Lookahead** | Zhang et al., NeurIPS 2019 | 双权重：快权探索，慢权稳定 | 极大降低局部噪音带偏概率，方差更稳定 | 训练过程不稳定时 |
| **Lion** | Chen et al., NeurIPS 2023 | 只看梯度符号，忽略幅度 | 天然免疫闪崩等极端异常值，节省显存 | 大 Batch Size 训练 |

### 机构级模型架构

#### DeepLOB — CNN + BiLSTM + Attention

论文 [Zhang et al., 2019]: 针对金融微观结构设计的专用架构。

```
Input(158) → Projection(120) → Conv3×2(16,16,16) → BiLSTM(64) → Attention → FC(1)
```

- CNN 1×2 卷积核：捕捉同档位买卖价差/量价不平衡的空间特征
- BiLSTM：双向时序建模，捕捉过去 120 个时间步的微观动量
- 注意力机制：自动加权重要时间步，而非简单取最后一步
- 超参：`--seq-len 120 --dropout 0.3 --hidden 64`

#### Temporal Fusion Transformer (TFT)

论文 [Lim et al., 2019]: Google 多时间跨度预测，目前中低频时序最先进模型之一。

```
Input(158) → Linear(256) → LSTM Encoder → GRN → Multi-Head Attention(4) → FC(1)
```

- GRN (Gated Residual Network)：门控残差网络，特征选择+非线性变换
- 多头注意力：4 头并行，捕捉不同周期的因子共振
- 位置编码：可学习位置编码，建模时序顺序
- 超参：`--hidden 256 --heads 4 --dropout 0.3`

### 损失函数

| 损失函数 | 公式 | 说明 |
|---------|------|------|
| **可微夏普比率** (Sharpe) | `-E[R] / sqrt(Var[R] + ε)` | 直接优化组合风险调整收益，默认推荐 |
| 均方误差 (MSE) | `mean((pred - actual)²)` | 传统回归损失，不直接优化收益率 |
| 负 IC (IC) | `-corr(pred, actual)` | 最大化信息系数，因子评价标准 |

### 特征正交化 + 对抗训练

**特征正交化**：将截面特征相对于市场基准回归取残差，确保模型学习纯 Alpha 而非 Beta。

**FGSM 对抗训练**：在训练数据中注入梯度方向的微小扰动，强制模型忽略微小价格噪音，极大提升实盘异常行情鲁棒性。

### 多卡并行训练

支持 `nn.DataParallel` 多 GPU 并行，自动将 batch 切分到多张 GPU 上计算梯度：

```bash
# 双卡
eq ml train csi300 5 --algo tft --device cuda --gpus "0,1"

# 四卡
eq ml train csi300 5 --algo deeplob --device cuda --gpus "0,1,2,3"

# 港股训练多卡
eq hk train --top 73 --cell gru --device cuda --gpus "0,1"
```

| GPU 配置 | batch_size 建议 | 加速比 |
|---------|----------------|--------|
| 单卡 CUDA GPU | 512 | 1×（基准） |
| 双卡 4090 | 1024 | ~1.8× |
| 四卡 A100 | 4096 | ~3.5× |

LSTM 路径把 Alpha158 的 158 维特征重塑成 (batch, seq_len=6, input_size=26) 的时序张量喂给 LSTM——这是量化选股的正确做法（学"过去 6 日形态"），比 MLP 把特征当独立向量强。CUDA GPU 12GB CUDA 主场。

### 数据更新器（v0.15，v0.22 增三项特性）

qlib 本地数据集截至 2020-09-25，`eq ml update-data` 续到最新：

```bash
eq ml update-data --start 2020-09-28 --universe csi300   # 腾讯 API 拉 6 年日线，约 30-60 分钟
# v0.22 新增：单股 + 预设指数合并下载训练
eq ml update-data --start 2015-01-01 --universe csi500 --extra SH600519,SZ000001
# v0.22 新增：从自选股文件读取（D:\idmxz\Table.txt）
eq ml update-data --start 2015-01-01 --universe watchlist
```

腾讯 API（`web.ifzq.gtimg.cn`，国内直连无需梯子）拉日线 → 转 qlib `.bin` 格式（float32，按日历顺序）续期 + 日历续期。续完后 `eq ml train` 用最新数据训练，`predict-batch` 出的就是今天的分数。

**v0.22 三项特性**（详见前文「A 股数据抓取三项特性」节）：

1. **单股 + 预设指数合并下载训练** — `--extra`/`-x` 指定额内股票，与 `--universe` 成分股合并去重写入同一份 `instruments/<universe>.txt`。
2. **跳过较晚股票没上市的时间不重试** — 腾讯返回空 → 判定未上市/已退市，直接写全 NaN 跳过；只有网络异常才重试 3 次。
3. **下载先后顺序无关，结果一致** — `_write_bin` 覆盖写 + `_generate_instruments` 从 `.bin` 推断区间，多次下载同区间得到逐字节一致的 `.bin` 与 `instruments` 文件。

### Colab / Kaggle 云训练适配

EternityQuant 支持在 **Google Colab** 和 **Kaggle** 的免费 GPU 上训练模型，利用 T4/P100 的 CUDA 加速。

**📍 笔记本地址：**

| 平台 | 笔记本 | GPU | 显存 |
|------|--------|-----|------|
| [Colab](https://colab.research.google.com) | [`notebooks/colab_eternityquant_train.ipynb`](notebooks/colab_eternityquant_train.ipynb) | T4 | 16 GB |
| [Kaggle](https://kaggle.com) | [`notebooks/kaggle_eternityquant_train.ipynb`](notebooks/kaggle_eternityquant_train.ipynb) | T4/P100 | 16 GB |

**云端 vs 本地训练对比：**

| 维度 | 本地（CUDA GPU 12GB） | Colab（T4 16GB） | Kaggle（T4/P100 16GB） |
|------|-------------------|------------------|----------------------|
| GPU | CUDA GPU | Tesla T4 | T4 / P100 |
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

## Streamlit 9 页看板

```bash
eq dash --port 8501    # 启动本地看板
```

| 页 | 功能 |
|----|------|
| 概览 | 持仓实时盈亏 + 自选/规则/缓存统计 + 最近触发信号 |
| 晨报 | 大盘闸门 + 持仓止损警报 + 今日信号翻转 + 纸面战绩（网页版 `eq daily`，v0.33） |
| 持仓 | 持仓体检（实时浮盈 + 仓位分布柱状图 + 风险提示）+ 已清仓记录 |
| 自选 | 自选列表 + 一键拉实时行情 + 表单加自选 |
| 选股 | 技术选股（14 条件多选，命中一键入自选） |
| 回测 | 跑回测/全策略横评（17 个内置策略）+ 权益曲线（含买入持有基准）+ 历史记录管理 |
| 监控规则 | 规则列表+触发统计 |
| ML 模型 | 模型列表+激活+predict-batch Top10+一键入自选+预测历史 |
| 下载管理 | A/港/美股下载 + 缓存占用统计与清理 |

### 换肤（v0.33）

给一张图，看板自动变成它的配色：

```bash
eq theme "D:\pic\封面.jpg"           # 存进 .eternityquant/.env，之后 eq dash 一直生效
eq theme                             # 留空 = 看当前配置 + 提取出的色板
eq theme --primary "#a86034"         # 手动指定主色，不自动取色
eq theme --clear                     # 恢复默认外观
eq dash -i x.jpg --opacity 0.7       # 只这一次用（加 --save 才写配置）
eq dash --no-theme                   # 本次禁用主题（排查显示问题）
```

| 环境变量 | 作用 | 默认 |
|----------|------|------|
| `EQ_DASH_IMAGE` | 背景图路径 | 空（不换肤） |
| `EQ_DASH_OPACITY` | 遮罩不透明度，越大图越淡 | 0.88 |
| `EQ_DASH_MASCOT` | 侧栏看板娘卡片 | on |
| `EQ_DASH_PRIMARY` | 手动指定强调色（`#rrggbb`），不自动取色 | 空 |

取色用 PIL 中值切分量化，挑饱和度够高的一档做强调色，按图片平均亮度自动切亮/暗两套。

**配色走 Streamlit 原生 `--theme.*` 参数，不是 CSS 硬覆盖**——指标数值、下拉框、
`st.dataframe`（canvas 渲染）的文字颜色由 Streamlit 自己管，只改容器背景会在暗色图上
撞出「深底深字」看不见（实测过）。CSS 只负责它做不到的：背景图、看板娘、毛玻璃卡片。

## 路线图（全完成）

1. ✅ CLI + 数据层 + watch 命令（v0.1）
2. ✅ 定时推送服务固化（v0.2，APScheduler）
3. ✅ 回测结果外存 parquet + backtest_runs 表（v0.3）
4. ✅ 多市场扫描（v0.4，A/HK/US/CRYPTO）
5. ✅ 四个监控处理器（v0.5，indicator/news/event/flow，10 种规则全落地）
6. ✅ qlib workflow 真集成（v0.6，Alpha158 + LightGBM）
7. ✅ predict-batch 跑通 + torch DLL 预热（v0.7）
8. ✅ LightGBM GPU 训练（v0.8，`--device gpu`）
9. ✅ qlib PyTorch CUDA 集成（v0.9，自写 MLP 走 CUDA GPU）
10. ✅ 个股深度研究（v0.10~v0.34；v0.35 删除——数据 dump 不喂任何下游，手机 App 做得更好，见下「砍掉的东西」）
11. ✅ Streamlit 看板加 ML 交互（v0.11）
12. ✅ 单元测试固化 + CLI CUDA 泄漏修复（v0.12，35 测试）
13. ✅ 自写 LSTM + CUDA 训练进度 log（v0.13，6×26 时序重塑）
14. ✅ predict-batch 支持自写 LSTM/MLP 模型（v0.14，按 algo 分路）
15. ✅ qlib 数据更新器（v0.15，腾讯 API 续到最新）
16. ✅ 高级优化器（AdamW/SAM/Lookahead/Lion）+ 可微夏普损失 + 对抗训练 + 特征正交化（v0.16）
17. ✅ DeepLOB（CNN+BiLSTM+Attention）+ TFT（Google 多时间跨度）顶会架构复现（v0.17）
18. ✅ 港股全链路（数据收集 → 特征 → 自写 GRU 训练 → 预测）（v0.18）
19. ✅ 统一数据目录 `data/{a,hk,us}/` + 旧目录自动迁移（v0.20）
20. ✅ 自选股 universe（v0.22，从 `D:\idmxz\Table.txt` 读取 A 股代码）
21. ✅ 数据抓取三项特性（v0.22）：
    - 单股 + 预设指数合并下载训练（`--extra`/`-x` + `--universe`）
    - 跳过未上市/已退市时间不重试（腾讯返回空 → 写全 NaN 跳过）
    - 下载先后顺序无关（覆盖写 `.bin` + 从 `.bin` 推断 instruments 区间）
22. ✅ 修复导入 bug：`eq.web.run_dashboard` 改从 `runner` 导入；`search_lstm` universe 校验补 `watchlist`（v0.22）
23. ✅ 全量代码审查 + BUG 修复 + 实用功能扩充（v0.24，测试 35 → 154）
24. ✅ 训练策略方法学修正（v0.25，测试 → 213）：独立测试段 + purge + 每日横截面 Rank IC/ICIR + 随机种子 + LightGBM 官方调优超参
31. ✅ 日常闭环（v0.32，测试 → 549）：`eq daily` 晨报（大盘/止损警报/信号翻转检测）+ `eq paper` 纸面日志（前向记录、到期结算、vs 沪深300 超额 t 检验）
30. ✅ 次日高点预测与限价止盈（v0.31，测试 → 522）：正确建模限价成交、MFE/MAE 分布、档位扫描；实测结论为**负**（成本 > 次日边际），并证明 ML 预测 MFE 学到的是波动率而非方向；顺带补 OHLC 自洽性校验
29. ✅ 散户长线工具（v0.30，测试 → 496）：执行延迟（T+1）、资金量→持仓数/换手预算、大盘闸门（附否定其有效性的实测数据）
28. ✅ 组合回测 + 真实成本 + ML 接入（v0.29，测试 → 475）：A 股/港/美真实费率（含最低佣金与单边印花税）、组合级回测（三种资金分配 + 持仓约束 + 分散化）、ML 预测直接跑组合回测
27. ✅ 策略稳健性验证（v0.28，测试 → 424）：多标的分布、Walk-Forward 样本外（带 purge）、参数高原检测、样本内外分离的参数寻优、频率匹配的随机基准
26. ✅ 策略层重构（v0.27，测试 → 394）：因子 6→21、策略 4→17、信号分数化、多策略投票、市场状态自适应、仓位管理与风控、引擎支持连续仓位
25. ✅ 数据源注册表（v0.26，测试 → 294）：13 个源自动 failover + `eq data sources --test` 本机自检 + 新浪/腾讯批量快照

### v0.26 顺带修的解析 bug

新增源时用「同一只票、两个独立源」交叉验证，抓到几个只看单源发现不了的问题：

| 位置 | 问题 | 后果 |
|------|------|------|
| `tencent_snapshot` | 成交量一律 ×100 当「手」换「股」 | A 股对，但港股/美股字段本身就是股数——港股成交量虚报 100 倍 |
| `tencent_snapshot` | 日期按固定分隔符切 | 三个市场格式不同（A 股 `20260724161433` / 港股 `2026/07/24 16:08:10` / 美股 `2026-07-24 16:00:01`），港股切出 `2026-/0-7/` |
| `sina_snapshot`（美股） | 直接取时间戳前 10 位当日期 | 新浪美股时间戳是**北京时间**，美东 07-24 21:46 显示成北京 07-25 09:46，日期比真实交易日多一天 |
| `eastmoney_spot` | 自己重写了一遍板块判断 | `920xxx`（北交所）被判成 `.SH`——直接复用 `normalize_symbol` 才对 |
| `_norm_bars` | 没统一 dtype | 各源成交量有的 int、有的 str、有的 Int64，下游算术会因 dtype 不一致出岔子 |

## v0.25 训练策略修的问题

除上面「IC 不可用于调参」四条外，还修了这些：

| 位置 | 问题 | 后果 |
|------|------|------|
| `_SimpleMLP` | `nn.Dropout(0.05)` 硬编码，构造函数根本不收 `dropout` 参数 | CLI 的 `--dropout 0.4` 对 MLP 路径**完全无效**，用户以为调了其实没调 |
| `_SimpleSeqModel` | `train_torch` 调用时没传 `dropout`，一直用类默认 0.1 | 同上 |
| `_SimpleSeqModel._reshape` | `cut = 6×26 = 156 < input_dim = 158`，走 `x[:, :cut]` **静默丢掉最后 2 维**——而 docstring 写着"保证 158 维全保留" | 注释与实现矛盾，特征悄悄丢失 |
| `AdvancedTrainer.fit` | 模型输出塌缩成常数时 `score = -inf`，`best_score` 永不更新 → `best_state` 一直是 `None` → 训练结束根本不加载最优权重 | **返回一个从未被选中过的模型**（同类坑在 `_SimpleSeqModel` 已修过，这里是漏网的一处） |
| `ml_data_updater` | `.bin` 只写了 `open/high/low/close/volume/factor/change`，**没有 `vwap`** | Alpha158 的 `VWAP0` 列恒为 0（被 `Fillna` 悄悄填掉，158 维里 1 维是死的）；Alpha360 的 60 列依赖 `$vwap`，完全不可用。已补 `vwap ≈ (H+L+C)/3` |
| `train()` LightGBM | `num_leaves=64, lr=0.05, n_estimators=200`，**无任何 L1/L2 正则、无行采样** | 股票日频截面数据信噪比极低（单因子 IC 通常 0.02~0.05），这种配置几乎必然过拟合。已换成 qlib 官方 benchmark 调优结果（`lambda_l1=205.7, lambda_l2=580.98`，是常规 GBDT 任务的几百倍——正是为了压住低信噪比数据上的过拟合） |
| `train_hk` | Walk-Forward 在**行下标**上滚，滚的是股票不是时间；且算完的 `avg_ic` 只打印、不返回 | 滚动验证完全无效 |
| `train_hk` | 标签用原始 h 日收益，未做横截面归一化 | 港股日收益方差绝大部分是市场 beta，模型学的是"预测大盘"而非选股 |

### 仍未验证的部分

**A 股 qlib 链路（`eq ml train` 的 lightgbm/gru/lstm/mlp 路径）的改动只经过静态检查，
没有端到端跑过** —— 本机没装 qlib，`data/a/qlib_cn_data/` 也是空的。
可验证的部分（评估指标、切分、模型类、港股链路）都有单测覆盖并实际跑通。
首次用新版跑 A 股训练时请留意 `[切分]` 那行日志，确认三段区间符合预期。

## v0.24 修复的 BUG

一轮全文件审查的产出。按影响面排序：

| # | 位置 | 问题 | 后果 |
|---|------|------|------|
| 1 | `eq/db.py` | `with get_state_conn() as conn` 用的是原生 `sqlite3.Connection`，其上下文管理器**只 commit/rollback、不关连接** | 每次 CRUD 泄漏一个连接 + 文件句柄；scheduler daemon / Streamlit 长会话最终耗尽句柄 |
| 2 | `eq/cli.py` | `if __name__ == "__main__": app()` 写在**文件中段** | `python -m eq.cli` 时 `scheduler`/`hk`/`data`/`dash` 四个命令组全部"不存在" |
| 3 | `eq/strategy/signals/trend.py` | bool Series `.shift(1)` 退化成 object dtype，`~prev_above` 抛 `TypeError` | `adx_trend` 策略在任何数据上必崩，从未跑通过 |
| 4 | `eq/strategy/factors/ml_workflow.py` | `_patched_corr_load` 用了 `np` 但模块没 `import numpy` | qlib Corr 补丁一触发就 `NameError`，训练中断 |
| 5 | `eq/data/market.py` | yfinance 港股要 4 位零填充（`0700.HK`），代码传的是项目内 5 位格式 | 港股主源必然查无此票，每次都白跑一遍再退 akshare |
| 6 | `eq/data/market.py` | baostock 用**进程级全局 socket**，新增的并发取数会互相踩踏 | `WinError 10038` + 批量拉取随机失败 |
| 7 | `eq/backtest/*.py` | `(1+total_return) ** (1/years)` 在总收益 ≤ -100% 时是负数开分数次方 | 年化收益显示 `nan%` |
| 8 | `eq/backtest/*.py` | `years = max(n_days/252, 1e-9)`，20 个 bar 也照样年化外推 | 短窗口回测报出荒谬的"年化 +45%" |
| 9 | `eq/backtest/event_driven.py` | 收盘仍持仓时又 append 一次末日权益 | 权益曲线尾部**索引重复**，`to_parquet`/画图/`pct_change` 全出问题 |
| 10 | `eq/core/monitor.py` | `_h_news` 把 `600519.SH` 传给 akshare（它只认裸码 `600519`） | `news` 规则永远不触发 |
| 11 | `eq/core/monitor.py` | 涨跌停一律按 ±10% 判 | 创业板/科创板（±20%）、北交所（±30%）的涨跌停规则永远差 10~20 个点触发不了 |
| 12 | `eq/core/monitor.py` | 无冷却机制 | 定时任务每 5 分钟跑一次，同一条规则每 5 分钟推一遍 |
| 13 | `eq/core/scanner.py` | `_norm_cols` 用 `df[list(col_map.values())]` 硬取列 | 上游 akshare 改列名 / 某市场没成交额列 → `KeyError` 整个命令挂掉 |
| 14 | `eq/core/scanner.py` | `sort_by` 不在列里时既不排序**也不截断** | `eq scan US --by volume` 把全表都吐出来 |
| 15 | `eq/core/scanner.py` | 美股 `split(".")[1]` 遇到裸代码得到 `NaN` | 符号变成 `nan.US`，后续全炸 |
| 16 | `eq/web/dashboard.py` | 缓存目录写死相对路径 `Path("data")` | 缓存清理/占用统计恒为空（取决于 streamlit 工作目录） |
| 17 | `eq/web/dashboard.py` | 港股/美股下载传了不存在的 `--codes-file` 参数 | 只要填了品种表/代码，下载必然失败 |
| 18 | `eq/web/dashboard.py` | `pd.DataFrame(list[sqlite3.Row])`（Row 是序列不是映射） | 预测历史表列名全变成 `0/1/2` |
| 19 | `eq/data/collector.py` | 美股 yfinance fallback 分支 `reset_index()`，与东财分支落盘结构不一致 | 下游 `read_csv(index_col=0)` 读到行号而非日期 |
| 20 | `eq/cli.py` | `pf_summary` 定义两次、`ml_app` 建两次并重复 `add_typer` | 后者静默覆盖前者，行为取决于 click 解析顺序 |
| 21 | `eq/cli.py` | `wl_import` 硬要求行内含 `\t`，且调了一个结果没用的 `_watchlist_instruments()` | 空格分隔的品种表一只都导不进来 |
| 22 | `eq/data/market.py` | `detect_market` 只认严格大写全后缀格式 | `600519`、`600519.sh`、`SH600519` 一律 `ValueError: 无法识别市场` |

另修：`get_recent_bars` 不再返回约 2 倍于 `days` 的数据（现按 `days` 截断）；
持仓/自选符号统一规整（同一只票不同写法不再重复建仓）；
建仓/加仓/减仓补零负值校验；`signals` 表从"建了没人写"变成真的记录触发历史；
sqlite3 date/timestamp 适配器显式注册（消除 Python 3.12+ 废弃告警）；
Streamlit `use_container_width` → `width="stretch"`（旧参数已过移除期）。

## 脱离 qlib（v0.38 第一步 / v0.39 第二步）

qlib 在本项目里贴了 5 个 monkey patch 绕它的 bug（ReduceLROnPlateau 的版本号
字符串比较 `'2.13.0' <= '2.6.0'` 字典序为真、issue #1949 的位置参数重复传值、
`Corr._load_internal` 空序列崩溃、`provider_uri` 必须是 dict）。两步做完之后，
**存在一条完全不碰 qlib 的训练链路**。

```bash
eq ml train-local --from watchlist --algo lightgbm --seeds 3
```

这条命令从 `eq data a` 下下来的行情缓存直接出模型，全程不 import qlib。

| 原来由 qlib 提供 | 现在 | 位置 |
|---|---|---|
| `contrib.model.LGBModel` | 原生 lightgbm | `factors/gbdt.py`（v0.38）|
| `contrib.model.ALSTM/GRU/LSTM/DNN` | **删除**（早就是死代码） | —（v0.38）|
| 5 个处理器 | 自写 | `factors/preprocess.py`（v0.38）|
| ReduceLROnPlateau 补丁 | **删除** | —（v0.38）|
| **Alpha158 特征** | 自写 158 个 | `factors/alpha.py`（v0.39）|
| **数据层（.bin + 表达式引擎）** | 直接吃项目的行情缓存 | `factors/local_train.py`（v0.39）|

老的 `eq ml train` 仍然走 qlib，两条路共用切分/预处理/模型/评估/集成/注册，
所以成绩可以直接比——比的是特征实现，不是别的东西。

完整用法：

```bash
eq data a -u watchlist                        # 先把行情下下来
eq ml train-local --from A --top 300 --seeds 3  # 训练（A 股成交额前 300）
eq ml predict-local <model_id> --dry-run      # 先看看，不写库
eq ml predict-local <model_id> --top 10       # 满意了再落 ml_predictions
```

**候选池不能太小。** 截面选股是「今天这批票里挑哪只」——只有 5 只自选股的话，
模型每天只在 5 个名字之间排序，IC 的噪声大到没有意义。建议 ≥100 只。

### 模型成绩要有参照物（v0.40）

单看一个 test IC 数字没法判断好坏。实跑 A 股成交额前 300、284k 样本的结果：

```
valid IC +0.0840   test IC +0.0110
test ICIR +0.048   t 值 +0.62   胜率 48%
```

valid 比 test 高 **7.6 倍**——valid 是早停挑 checkpoint 用的，是迭代过程中的最大值，
天然虚高，这个项目已经反复量到同一现象。而 test IC +0.011、t=0.62、胜率不到一半，
**和 0 分不开**：按 ICIR=0.048 反推，要让 t 达到 2 需要约 1740 个交易日（≈7 年）的测试段。

问题是：这是「模型不行」还是「这段行情本来就难」？`eq ml factor-scan` 在**同一个
测试段**上逐个评估 158 个单因子（不训练，很快），给出参照：

```bash
eq ml factor-scan --from A --top 300
```

- 模型明显高于最强单因子 → 模型学到了组合效应
- 模型还不如某个单因子 → 158 维模型跑输一个公式，管线有问题
- 两者都接近 0 → 这段行情/这批票就是难

**多重检验陷阱**：这是从 158 个因子里挑最大值，不是单次检验。零假设下扫 158 个因子，
最大 |t| 本来就期望在 3 附近——在**纯随机数据**上实测能挑出 `t=4.85` 的"因子"。
Bonferroni 校正后排第一的那个要 `|t| > 3.6` 才谈得上显著。这个命令是给模型当参照物的，
不是用来挖因子的。

### 冻结股票池（v0.42.1）

`--from A --top 300` **每次都重新联网扫市场**，两次调用很可能拿到不同的 300 只票。
跑对照实验（同一批票换不同参数）时这会让结果彼此不可比，而且**不会有任何报错**。

```bash
--from file:.eternityquant/ml_universe.txt   # 每行一个代码，# 后为注释
```

注释必须**按行**剥——先整体 split 的话，`# 说明文字` 里除了 `#` 之外的每个词
都会被当成股票代码（这个 bug 真的发生过，测试里钉住了）。

### IC 换算成钱（v0.42）

IC 是抽象数字——不含手续费、印花税、最低佣金、换手，也不含「只能买整手」。
`eq ml backtest-local` 把模型跑成权益曲线，并给一条**零成本对照**：

```bash
eq ml backtest-local <model_id> --from A --top 300 --positions 10 -r weekly
```

合成数据上的实测输出：

```
毛收益（零成本） +2.33%   换手 58.8x/年
净收益（含成本） -0.65%
成本吃掉         2.98%
```

**成本吃掉的比模型赚的还多。** 这是日频选股的常态：A 股卖出印花税 0.1%，
换手 58x/年意味着光印花税就是 5.8% 的年化拖累，再加佣金和滑点。
所以在 IC 0.01~0.04 这个量级上，**降换手比调模型有效得多**——
`--rebalance monthly`、加大持仓数、或训练时拉长 `--horizon`。

**回测只在测试段跑，起点从模型存的 `split_bounds` 里读。** 在训练段上回测会给出
一条又漂亮又毫无意义的曲线，而且不报错、不崩，只会让人高兴——所以边界定不下来时
直接拒绝执行，不猜。测试里有一条专门验证起点晚于训练段末尾。

### factor-scan 慢到以为卡死（v0.41.1 修）

300 只票跑 `factor-scan` 要十几分钟才出结果。原因是 158 个因子各调一次
`evaluate()`，而 `evaluate` 每次要跑三遍数据（Rank IC + 分层收益 + Pearson IC），
每遍都按日 groupby。

改成一次算完整张 `日期 x 因子` 的 IC 矩阵：`groupby(日期).rank()` 把 158 列同时
排名，逐日去均值后 Rank IC = 中心化秩的相关，写成几个 groupby 求和。
实测 **422 秒 -> 1.45 秒（约 290 倍）**，数值和逐列 spearman 差 1.11e-16（机器精度）。

`baseline` 的因子选择也走同一条路。

### 一个自己造的 KeyError（v0.41.1 修）

`factor-scan` 直接崩在 `KeyError: 't_nw'`：CLI 要显示这一列，但 `factor_scan`
的 `rows.append` 里根本没加。根因是我用 `str.replace` 改代码却**没断言匹配成功**
——不匹配时它静默 no-op。更糟的是空表分支加了这列、非空分支没加，
同一个函数返回两种 schema。

而当时的用例断言的是**旧列名**，所以它"通过"了，没拦住。现在加了三条：
`t_nw` 必须存在且非空、空表与非空分支列名必须一致、向量化实现与逐列 spearman 必须逐位一致。

### 重叠标签让 t 值虚高（v0.41 修）

`horizon=5` 的标签是 `close[t+5]/close[t]-1`，**相邻交易日的标签共用 4 天行情**。
于是每日 IC 强烈自相关，而 `t = ICIR x sqrt(n_days)` 把这些天当成独立样本，
系统性高估显著性——粗略地说高估 `sqrt(horizon)` 倍，h=5 时约 2.2 倍。

这不是小事。实跑 `factor-scan` 报出来的最强因子 RSQR20 是 `t=4.60`，
看着远超 Bonferroni 阈值 3.6；修正后大概率落到 2 附近，**跨过了「显著」和
「不显著」的分界线**。

现在所有报告都多一个 `t_stat_nw`（Newey-West，滞后阶 `horizon-1`），
用**实测的自相关**而不是拍一个 `sqrt(h)`。

实测：5 日重叠序列 t 2.76 -> 1.49（比值 0.54，理论约 0.45）；
本来就无自相关的序列 1.67 -> 1.68（不误伤）。

**这个 bug 是写测试时撞出来的**——合成基准在纯随机数据上拿到 IC -0.0965，
按独立同分布算是 4.6 个标准误。第一反应是工具有问题，查下来是 t 值算法本身有偏。

### 早停口径对齐 LightGBM（v0.41，无实测收益）

v0.36 给自写 torch 模型改成按 Rank IC 早停时，我在 LightGBM 这里写了句
「早停只能用 LightGBM 的 mse，那是它内部的事」——**那句话是错的**，`feval` 就是干这个的。
现在 `ic_early_stop=True`（默认）关掉内置 metric，用每日截面 Rank IC 早停。

**但要说清楚：这没有带来可测的收益。** 5 个种子的合成数据 A/B：

| | 早停轮次 | test IC | test ICIR |
|---|---|---|---|
| MSE 早停 | 60.4 | +0.2411 | +2.020 |
| Rank IC 早停 | 26.8 | +0.2409 | +2.017 |

口径对齐是对的（选和考应该同一把尺），训练轮次也减半了，但它**不解释**
模型跑输单因子的现象。我一度把这个当成原因，是过早下结论。

### 小样本上的正则塌缩（v0.39.1 实测修复）

拿 5 只自选股跑 `train-local`，输出是一串 `IC +0.0000 / ICIR 0.000 / 胜率 0%`。
看着像「这批票没信号」，实际是**模型根本没长出来**：qlib 那套官方超参
（`lambda_l1=205.7`）是在 csi300 约 40 万样本上调的，而 LightGBM 的 L1 是对叶子内
**梯度和**做软阈值——`|Σg| ≤ lambda_l1` 时叶子输出直接归零。几千个样本时任何分裂
都过不了这个门槛，模型退化成单个常数叶子，`best_iteration=1`，同一天所有票分数
相同，截面 IC 恒等于 0。

塌缩与否同时取决于两件事：**样本量**和**标签尺度**。本项目默认对标签做截面 rank
归一化，压到 ±1.7，梯度和比原始收益率标签小得多，更容易被削平（实测：同样 336 个
样本，原始收益率标签不塌缩、rank 标签塌缩）。

两处修复：

1. `gbdt.scale_params_to_size()` 按训练样本量线性缩放 `lambda_l1/l2`，
   并把 `num_leaves` 压到 `n/100`（4000 个样本配 210 片叶子，平均每片不到 20 条，
   纯粹在记噪声）。实测同一份数据：塌缩 → valid IC +0.126。
2. 训练后自检。IC=0 有两种截然不同的原因——**模型没长出来**（要调参）和
   **这批票真没信号**（要换票），不区分的话用户没法处置。现在会直接打出
   「模型预测塌缩成常数」「候选池只有 N 只」「训练样本仅 N 条」。

**模型和预处理管线是一起存的**（`{"model", "pipeline", "features", "horizon"}`）。
推理时必须复用训练时拟合好的管线——重新 fit 一个，归一化统计量就来自推理数据
而不是训练段，那就是 train/serve skew 又回来了，而且不会报任何错。
测试里有一条专门把 `Pipeline.fit` 换成抛异常来钉死这点。

### 自写 Alpha158 的验证方式

没装 qlib、也没有 .bin 数据，所以**不是**和 qlib 逐位对拍，而是验证性质：

- **无前视**（最关键）：在序列尾部追加或篡改未来数据，已有行的特征值必须一字不变
- **算子正确**：等差数列上 MA/BETA/RSQR/RESI 都有闭式解，直接手算比对
- **尺度不变**：价格和成交量整体放大 1000 倍，特征值不变
  （这是「5 元的票和 500 元的票能否放进同一截面」的前提）
- **边界不炸**：停牌零成交量、一字板常数序列、超短历史

想和 qlib 对拍就在装了 qlib 的机器上跑 `alpha.compare_with_qlib(symbol, start, end)`，
它逐特征报最大绝对差。

**已知有意不同**：`IMAX/IMIN` 返回的是「距最高/最低点过去了几天」（今天创新高＝0），
按 Alpha158 自己的文字描述实现；qlib 内部用的是 argmax 原始下标还是回溯距离未经确认。

### 顺带修的几个问题

- **rank 归一化没精确居中**（v0.38）：qlib 的 `rank(pct=True) - 0.5` 残留 `+1/(2n)`
  偏移（n=30 时 +0.058），等于把「当天有多少只票」编码进因子值，制造出纯粹由
  样本量产生的跨日水平差。改用 `(rank-(n+1)/2)/n`。
- **预测路径的 lightgbm 分支已经坏了**（v0.38）：调的是 qlib LGBModel 的
  `predict(dataset, segment=)` 签名，换原生 lightgbm 后会直接 TypeError。
- **fit 窗口从隐式变显式**（v0.38）：归一化统计量只能用训练段拟合，
  以前藏在 handler 的 `fit_start_time` 里看不见。

### 还没验证的部分

- `preprocess.py` / `alpha.py` 是照 qlib 的**语义**实现的，没做逐位对拍。
- 老的 `eq ml train`（走 qlib handler）那几行在本机跑不了，改动未经端到端验证。
  新的 `eq ml train-local` 有 19 个用例覆盖，其中一条专门把 qlib 变成不可 import
  来确认整条链路真的不依赖它。
- 项目当前已注册模型数为 0，数值口径变化不存在新旧可比性问题。

### 要彻底删掉 qlib 依赖还差什么

`eq ml train` / `predict-batch` / `update-data` 这几条老路仍然用 qlib 的 .bin。
如果 `train-local` 用下来没问题，可以把老路一起摘掉、`.bin` 数据也不用维护了。
建议先两条路并行跑一段，比较 test IC 之后再决定。

## 事件因子（v0.37）

v0.35 删掉深度研究时留了个回收清单：解禁 / 股东户数 / 融资融券 / 北向持股
这四项不是给人看的资讯，是**可交易的数据**。v0.37 把它们写成了因子
（`eq/strategy/factors/event.py`），不是报告。

| 因子 | 含义 | 方向 |
|------|------|------|
| `days_to_event` | 距下一次解禁还有几天 | 日期本身提前公开，无前视问题 |
| `event_pressure` | 未来 N 天解禁压力（按比例加权、按距离指数衰减） | 越大压力越大 |
| `holder_change` | 股东户数环比变化，**取负** | 户数下降＝筹码集中＝正分 |
| `balance_momentum` | 融资余额 / 北向持股的 N 日变化率 | 用变化率不用绝对额，否则是在赌大小盘 |

### 这个模块最要命的地方：前视偏差

外部数据几乎都有两个日期——**报告期**（2025 年三季度末股东户数）和
**公告日**（2025-10-24 披露）。按报告期对齐，等于让 9 月 30 日的策略用上
10 月 24 日才公布的数字：回测 IC 会非常漂亮，实盘一分钱赚不到。

所有对齐一律走 `align_events()`，它**只认公告日**且强制 `公告日 <= 交易日`。
`tests/test_event_factors.py` 专门构造「报告期早、公告日晚」的数据来验证这条。

（写这个模块时自己就踩了一次：`ffill_limit` 用「值是不是 NaN」判断数据新鲜度，
但 `merge_asof` 出来的值本身已经前向填充过，`notna()` 恒为真，限制形同虚设。
改成按公告日分组计龄。）

### 现状：预测力**尚未验证**

本模块只保证「算得对、不穿越」，**不保证有 alpha**。用法是先拉数据、再跑截面 IC：

```python
from eq.strategy.factors.event import build_panel, evaluate_factor
print(evaluate_factor(build_panel(factors), bars, horizon=5))
```

走的是和 ML 模型**同一套** `evaluation.evaluate`（逐日截面 Rank IC），
两者可以直接比较。IC 站不住就该扔掉，别因为「听起来有道理」就塞进策略。

## 行业集中度约束（v0.37）

单票权重上限管不住行业集中——10 只票全是白酒，每只 10% 也照样是满仓一个行业，
一条政策就能把整个组合带走。

```python
cfg = PortfolioConfig(max_positions=10, max_per_industry=2)
run_portfolio(bars, strategy, cfg, industries={"600519.SH": "白酒", ...})
```

约束发生在**选股阶段**而不是权重分配阶段：等到分配权重时候选池已经全是同行业了，
再怎么调权重也分散不了。行业归属缺失的标的**不受限制**——数据拿不到不该等于禁止买入。
默认 `max_per_industry=0`（不限制），行为与旧版一致。

## 砍掉的东西

### 个股深度研究（v0.10 加，v0.35 删）

`eq research <symbol>` + 看板的深度研究页，15 个板块的基本面/资金/新闻/研报 dump。
删掉的理由，按重要性排：

1. **输出不喂任何下游。** 整个项目是一条链——数据 → 因子 → 信号 → 回测 →
   纸面日志 → 晨报，每个模块都被下游消费。只有 research 是终端节点：打印完就结束，
   没有任何因子、信号、回测、筛选器读它的结果。
2. **手机 App 做得更好。** 公司画像、财报、新闻、股东户数这些，东财/雪球/富途
   都是实时的、排版好的、随手能翻的。我们这版是文本 dump。
3. **全仓维护成本最高。** 918 行代码挂在约 10 个 akshare 接口 + yfinance 上，
   全是最易变的第三方 schema。代码里那些「挨个试两个函数名」的循环就是被打怕的痕迹。
4. **它喂的是拍脑袋决策。** 本项目反复测出来的结论是收益漏在主观判断上，
   纸面日志就是为了约束这个；一个新闻/研报 dump 恰好在助长反面习惯。

**如果要回收**：`lockup`（解禁日期＝已知的未来供给冲击）、`shareholders`（股东户数
变化）、`margin`、`northbound` 这四个是真正可交易的数据。但正确做法是写成因子、
跑逐日截面 Rank IC 验证有没有预测力，而不是从 `git show ec62374:eq/core/research.py`
里把 print 函数捞回来（仓库没打 tag，所以这里写的是 commit hash）。

## 剩余候选（未做）

- GRU/ALSTM 真单元差异（当前复用 LSTM 路径）
- PyPI 打包 publish（让 `pip install eternityquant` 能装）
- `eq/data/hk_market.py`（875 行）与 `ml_workflow.py`（1203 行）内部仍有重复逻辑可抽

## License

MIT
