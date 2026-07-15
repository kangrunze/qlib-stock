# 配置参考

> 文档版本: v4.0  
> 更新日期: 2026-07-10  
> 变更摘要: 新增 RankLGBModel 参数名映射说明、stock_universe 过滤策略修复、qlib_lgb 超参更新为 Optuna 最优值

---

## 配置体系概览

项目使用两层配置，职责明确：

| 配置文件 | 路径 | 用途 | 消费方 |
|----------|------|------|--------|
| **settings.yaml** | `config/settings.yaml` | 全局配置：数据路径、输出路径、策略参数、市场阶段分析参数 | `run.py`（argparse 默认值）、`data_center/`（数据路径）、`cmd_regime` |
| **workflow_config.yaml** | `qlib_pipeline/workflow_config.yaml` | Qlib 工作流配置：数据加载、特征处理器、模型超参、回测参数 | `qlib_pipeline/` 各模块（train/backtest/rolling/tscv/regime/sensitivity） |

> ⚠ `settings.yaml:data_source.qlib_dir` 必须与 `workflow_config.yaml:qlib.provider_uri` 保持一致。修改数据目录时请两处同步修改。

---

## 一、settings.yaml 配置详解

### 1.1 data_source — 数据源配置

```yaml
data_source:
  data_format: "csv"           # 数据格式: csv / parquet / qlib
  csv_dir: "D:/data"           # CSV 原始数据目录
  qlib_dir: "D:/download/qlib_bin" # Qlib bin 数据目录（需与 workflow_config.yaml:qlib.provider_uri 一致）
  start_date: "2005-01-01"     # 数据起始日期
  max_workers: 10              # 并发下载线程数
  retry_max: 3                 # API 请求最大重试次数
  retry_delay: 2               # 重试间隔（秒）
  incremental_window: 30       # 增量更新覆盖最近N天（应对复权修正）
```

| 字段 | 消费方 | 说明 |
|------|--------|------|
| `data_format` | `DataStore` | 切换底层数据格式 |
| `csv_dir` | `run.py data`、`CsvDataLoader` | 原始 CSV 数据目录 |
| `qlib_dir` | `run.py data`、`daily_update.py`、`cmd_regime` | Qlib bin 数据目录 |
| `max_workers` | `akshare_client`、`run.py update` | 并发下载线程数 |
| `incremental_window` | `daily_update.py` | 增量更新覆盖天数 |

### 1.2 output — 输出路径配置

```yaml
output:
  charts: "output/qlib_charts"
  picks: "output/picks"
  drift: "output/drift"
  ic_stability: "output/ic_stability"
  regime: "output/regime"
  tscv: "output/tscv"
  rolling: "output/rolling"
  sensitivity: "output/sensitivity"
  key_years: "output/key_years"
  validation: "output/validation"
  benchmark: "output/benchmark"
```

所有命令的 `--output-dir` 默认值均从此处读取，可通过命令行参数覆盖。

### 1.3 strategy — 策略全局参数

```yaml
strategy:
  top_k: 30                    # 选股推荐数量（argparse --pick-topk/--topk 默认值）
```

### 1.4 regime — 市场阶段分析配置

```yaml
regime:
  benchmark_code: "SH000300"   # 基准指数代码（沪深300指数，Phase C 修复后使用真实指数）
  benchmark_field: "close"     # 基准价格字段
  ma_window: 60                # 均线窗口
  vol_window: 60               # 波动率窗口
```

消费方：`run.py:cmd_regime()` 使用这些配置读取基准数据和计算参数。

---

## 二、workflow_config.yaml 配置详解

`qlib_pipeline/workflow_config.yaml` 专注于 Qlib 框架相关配置，包含以下核心段落：

### 2.1 qlib — Qlib 引擎初始化

```yaml
qlib:
  provider_uri: "D:/download/qlib_bin"   # Qlib bin 数据目录
  region: "cn"                        # 中国区
```

> ⚠ `provider_uri` 必须与 `config/settings.yaml:data_source.qlib_dir` 保持一致。

### 2.2 data_source — 数据源（仅用于数据转换）

```yaml
data_source:
  data_format: "csv"
  csv_dir: "D:/data"
  start_date: "2005-01-01"
  max_workers: 10
  retry_max: 3
  retry_delay: 2
```

消费方：`run_qlib_workflow.py`（数据下载和 CSV→bin 转换）。注意此处不包含 `qlib_dir`（已在 `qlib.provider_uri` 定义，避免重复）。

### 2.3 stock_universe — 选股池过滤

```yaml
stock_universe:
  exclude_boards: ["北交所"]
  min_listed_days: 250
  exclude_st: true
  exclude_suspended: true
```

消费方与生效范围：

| 消费方 | 作用阶段 | 生效条件 |
|--------|---------|---------|
| `run_qlib_workflow.py` | CSV 转 bin | 始终生效，写入 `instruments/all.txt` |
| `qlib_pipeline/train.py:_build_stock_universe_instruments()` | 训练/回测 | **仅当 `data_handler.instruments="all"` 时生效** |

> ⚠ **重要变更（2026-07-10）**：指数成分股池（`csi300`/`csi500`/`csi800`）已自带质量过滤，
> 且其 instruments 文件的 `start_date` 是指数纳入日期而非上市日期，
> 对其应用 `min_listed_days` 会错误调整纳入日期导致训练数据为空（`ValueError: Empty data from dataset`）。
> 因此 `stock_universe` 过滤在训练阶段**仅对 `"all"` 全市场生效**，指数成分股池自动跳过。

### 2.4 data_handler — 数据处理器

```yaml
data_handler:
  freq: "day"
  start_time: "2020-01-01"         # 数据加载起始
  end_time: "2026-07-03"            # 数据加载截止
  fit_start_time: "2020-01-01"     # 标准化拟合起始
  fit_end_time: "2023-06-30"       # 标准化拟合截止（防泄露）
  instruments: "csi300"             # 股票池: csi300 / csi500 / all
```

消费方：`qlib_pipeline/train.py:build_task()` → 作为 Alpha158/360 handler 的 kwargs。

### 2.5 dataset — 数据集定义

```yaml
dataset:
  handler: "Alpha158"                # 特征处理器: Alpha158 / Alpha360
  segments:
    train: ["2020-01-01", "2024-06-30"]
    valid: ["2024-07-01", "2025-06-30"]
    test:  ["2025-07-01", "2026-06-25"]
```

消费方：`qlib_pipeline/train.py:build_task()` → 作为 DatasetH 的 segments 参数。

### 2.6 qlib_lgb — Qlib 原生 LGBModel / RankLGBModel

```yaml
qlib_lgb:
  class: "LGBModel"                        # Qlib 模型类名（loss=rank 时自动切换为 RankLGBModel）
  module_path: "qlib.contrib.model.gbdt"   # Qlib 模块路径（loss=rank 时自动切换为 qlib_pipeline.model）
  kwargs:
    loss: "rank"                           # mse / rank（rank 使用 LambdaRank 排序目标）
    # 以下超参为 2026-07-10 Optuna 7-trial 搜索最优值（valid IC=0.0122）
    colsample_bytree: 0.8734341669478973
    learning_rate: 0.026163773612656583
    subsample: 0.8059350619095792
    lambda_l1: 0.05639417059186902
    lambda_l2: 1.7897519441564924e-08
    max_depth: 3                           # 浅树防过拟合（Optuna 确认 3 优于 5/8）
    num_leaves: 59
    num_threads: 8                         # 训练线程数（Windows 实测稳定值）
    min_child_samples: 90                  # 大值防过拟合
    early_stopping_rounds: 100             # 早停轮数（从默认 50 增至 100，给更多训练空间）
```

消费方：`qlib_pipeline/train.py:build_task()` → 构建 Qlib task dict 的 model 段。

#### loss 函数路由

| `loss` 值 | 实际模型类 | 模块路径 | 适用场景 |
|-----------|-----------|---------|---------|
| `"mse"` | `LGBModel` | `qlib.contrib.model.gbdt` | 回归模式，预测具体收益率数值 |
| `"rank"` | `RankLGBModel` | `qlib_pipeline.model` | LambdaRank 排序模式，优化股票排序（推荐用于选股） |

#### RankLGBModel 参数名映射（重要）

`RankLGBModel` 内部使用 `lgb.train()` 而非 sklearn API，**不识别 sklearn 风格的参数名**。
配置文件中的 sklearn API 参数名会被自动转换为 LightGBM 原生名：

| 配置文件（sklearn API） | 转换后（LightGBM 原生） | 说明 |
|------------------------|------------------------|------|
| `subsample` | `bagging_fraction` | 行采样比例 |
| `colsample_bytree` | `feature_fraction` | 列采样比例 |
| `subsample_freq` | `bagging_freq` | 行采样频率（默认自动设为 1 使 bagging_fraction 生效） |

> ⚠ **历史 Bug 说明**：2026-07-10 修复前，`subsample`/`colsample_bytree` 未做名称转换，
> 被 `lgb.train()` 静默忽略，导致 Optuna 搜索这两个参数时所有 trial 等效。
> 修复后参数才真正生效，回测超额收益从 -12.55% 提升到 +22.79%。

#### 超参调优历史

| 日期 | 配置 | valid IC | 回测超额(含成本) | IR | 备注 |
|------|------|---------|----------------|-----|------|
| 2026-07-06 前 | mse, depth=8, leaves=210 | — | — | — | 过拟合严重 |
| 2026-07-10 (30-trial) | rank, depth=5, lr=0.028 | 0.0027 | -12.55% | -1.167 | best_iter=1，单棵树太弱 |
| 2026-07-10 (lr=0.01) | rank, depth=5, lr=0.01 | — | -18.63% | -1.517 | NDCG 与回测收益不相关 |
| **2026-07-10 (7-trial)** | **rank, depth=3, min_child=90** | **0.0122** | **+22.79%** | **2.202** | 浅树+强正则，best_iter=3 |

### 2.7 backtest — 回测引擎

```yaml
backtest:
  executor:
    class: "SimulatorExecutor"
    module_path: "qlib.backtest.executor"
  strategy:
    class: "TopkDropoutStrategy"
    module_path: "qlib.contrib.strategy.signal_strategy"
    kwargs:
      topk: 30                           # 选股数量
      n_drop: 5                           # 每期替换数
  backtest:
    start_time: "2025-07-01"
    end_time: "2026-06-25"
    account: 100000000
    benchmark: "SH000300"             # 基准指数代码（沪深300指数）
    exchange_kwargs:
      freq: "day"
      limit_threshold: 0.099         # 涨跌停阈值（9.9%，留容差避免浮点误拒）
      deal_price: "open"             # 成交价（次日开盘，消除前视偏差，不推荐 close）
      open_cost: 0.0005              # 开仓手续费率（万 2.5）
      close_cost: 0.0015             # 平仓手续费率（含印花税万 10）
      min_cost: 5                    # 最低佣金（元）
```

> ⚠ **2026-07-09 回测引擎修复**：
> - `deal_price` 从 `close` 改为 `open`，消除 T+1 制度下的前视偏差
> - `limit_threshold` 从 `0.095` 改为 `0.099`，避免接近涨停但未触板的订单被错误拒绝

消费方：`run.py:cmd_full()` / `cmd_backtest()` → 传给 Qlib PortAnaRecord。

### 2.8 experiment — 实验配置

```yaml
experiment:
  name: "qlib_pipeline"
  mlflow_tracking_uri: "mlruns/"
  log_artifacts: true
  random_seed: 42
```

消费方：`qlib_pipeline/train.py`（MLflow 实验追踪）。

### 2.9 labels — 预测目标定义

```yaml
labels:
  types:
    - name: "ret_20d"
      horizon: 20
      description: "未来20日收益率（适合月度调仓）"
    - name: "ret_60d"
      horizon: 60
      description: "未来60日收益率（适合季度调仓）"
    - name: "ret_120d"
      horizon: 120
      description: "未来120日收益率（适合半年调仓）"
    - name: "alpha_20d"
      horizon: 20
      description: "相对基准超额收益（20日）"
    - name: "alpha_60d"
      horizon: 60
      description: "相对基准超额收益（60日）"
    - name: "alpha_120d"
      horizon: 120
      description: "相对基准超额收益（120日）"
    - name: "xs_ret_20d"
      horizon: 20
      description: "相对全池中位数收益（20日）"
    - name: "xs_ret_60d"
      horizon: 60
      description: "相对全池中位数收益（60日）"
    - name: "xs_ret_120d"
      horizon: 120
      description: "相对全池中位数收益（120日）"
    - name: "up_down_60d"
      horizon: 60
      description: "方向判断标签（60日，诊断用途）"
  primary: "xs_ret_20d"                # ★ 主 Label（2026-07-10 更新：从 ret_20d 改为 xs_ret_20d）
```

消费方：`qlib_pipeline/train.py:build_task()` → 传给 Alpha158/360 handler 生成 label 列。

> ⚠ **primary 标签选择**：2026-07-10 第 8.4 节 E6 实验确认 `xs_ret_20d`（截面中位数减法）
> 优于 `ret_20d`（绝对收益）。`xs_ret_Nd` 标签会自动注入 `CSMedianSubtract` learn_processor，
> 作为 Qlib learn_processor 使用（训练+推理一致），确保标签口径统一。

---

## 三、配置优先级

命令行参数 > workflow_config.yaml > 代码内兜底默认值

```bash
# 命令行参数覆盖 workflow_config.yaml
python run.py full --handler Alpha360 --loss rank --topk 30

# 使用自定义配置文件
python run.py full --config my_config.yaml
```