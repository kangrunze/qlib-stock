# 配置参考

> 文档版本: v3.2  
> 更新日期: 2026-07-06

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

消费方：`run_qlib_workflow.py`（CSV 转 bin 时写入 `instruments/all.txt`）。

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

### 2.6 qlib_lgb — Qlib 原生 LGBModel

```yaml
qlib_lgb:
  class: "LGBModel"
  module_path: "qlib.contrib.model.gbdt"
  kwargs:
    loss: "mse"                       # mse / rank
    colsample_bytree: 0.8879
    learning_rate: 0.0421
    subsample: 0.8789
    lambda_l1: 205.6999
    lambda_l2: 580.9768
    max_depth: 8
    num_leaves: 210
    num_threads: 20
```

消费方：`qlib_pipeline/train.py:build_task()` → 构建 Qlib task dict 的 model 段。

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
      topk: 50
      n_drop: 5
  backtest:
    start_time: "2025-07-01"
    end_time: "2026-06-25"
    account: 100000000
    benchmark: "SH000300"             # 基准指数代码（沪深300指数）
    exchange_kwargs:
      open_cost: 0.0005
      close_cost: 0.0015
```

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
  primary: "ret_20d"
```

消费方：`qlib_pipeline/train.py:build_task()` → 传给 Alpha158/360 handler 生成 label 列。

---

## 三、配置优先级

命令行参数 > workflow_config.yaml > 代码内兜底默认值

```bash
# 命令行参数覆盖 workflow_config.yaml
python run.py full --handler Alpha360 --loss rank --topk 30

# 使用自定义配置文件
python run.py full --config my_config.yaml
```