# 配置参考

> 文档版本: v3.0  
> 更新日期: 2026-07-06

---

## 配置体系概览

项目使用两层配置：

| 配置文件 | 路径 | 用途 | 消费方 |
|----------|------|------|--------|
| **settings.yaml** | `config/settings.yaml` | 全局配置：数据路径、模型参数、策略、输出路径等 | `run.py`、`data_center/`、`model/` |
| **workflow_config.yaml** | `qlib_pipeline/workflow_config.yaml` | Qlib 工作流配置：数据加载、特征处理器、回测参数 | `qlib_pipeline/` 各模块 |

`settings.yaml` 是唯一的全局配置源，所有硬编码参数均已提取到此文件中。`workflow_config.yaml` 专注 Qlib 框架相关配置。

---

## 一、settings.yaml 配置详解

### 1.1 data_source — 数据源配置

```yaml
data_source:
  data_format: "csv"           # 数据格式: csv / parquet / qlib
  csv_dir: "D:/data"           # CSV 原始数据目录
  qlib_dir: "D:/trae/qlib_bin" # Qlib bin 数据目录
  start_date: "2005-01-01"     # 数据起始日期
  max_workers: 10              # 并发下载线程数
  retry_max: 3                 # API 请求最大重试次数
  retry_delay: 2               # 重试间隔（秒）
  incremental_window: 30       # 增量更新覆盖最近N天
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
```

所有命令的 `--output-dir` 默认值均从此处读取，可通过命令行参数覆盖。

### 1.3 stock_universe — 选股池配置

```yaml
stock_universe:
  exclude_boards: ["北交所"]
  min_listed_days: 250
  exclude_st: true
  exclude_suspended: true
```

| 字段 | 说明 |
|------|------|
| `exclude_boards` | 排除板块列表 |
| `min_listed_days` | 上市至少 N 个交易日 |
| `exclude_st` | 排除 ST 股票 |
| `exclude_suspended` | 排除停牌股票 |

### 1.4 dataset_split — 数据集划分

```yaml
dataset_split:
  train_start: "2013-01-01"
  train_end: "2021-06-30"
  valid_start: "2021-07-01"
  valid_end: "2022-12-31"
  test_start: "2023-01-01"
  test_end: "2023-12-31"
  backtest_start: "2024-01-01"
  backtest_end: "2025-06-30"
```

### 1.5 features — 特征工程配置

```yaml
features:
  enabled_categories:
    - "trend"           # 趋势类: MA/EMA/MACD/ADX/CCI
    - "momentum"        # 动量类: RSI/KDJ/WR/MOM
    - "volatility"      # 波动类: ATR/STD/BB_width
    - "volume_price"    # 量价类: OBV/MFI/VWAP
    - "statistical"     # 统计类: Skew/Kurt/Beta/Alpha
    - "cross_section"   # 横截面: 排名分位数
  ic_threshold: 0.02
  corr_threshold: 0.95
```

### 1.6 labels — 标签配置

```yaml
labels:
  types:
    - name: "ret_20d"        # 未来20日收益率
      horizon: 20
    - name: "ret_60d"        # 未来60日收益率
      horizon: 60
    - name: "alpha_20d"      # 20日超额收益 vs 中证500
      horizon: 20
      benchmark: "000905"
    - name: "alpha_60d"      # 60日超额收益 vs 中证500
      horizon: 60
      benchmark: "000905"
  primary: "alpha_20d"
```

### 1.7 model — 模型配置

```yaml
model:
  types: ["lightgbm", "xgboost", "catboost"]
  ensemble_weights: [0.5, 0.3, 0.2]
  early_stopping_rounds: 50
  enable_optuna: true

  lightgbm:                  # LightGBM 参数
    objective: "regression"
    metric: "rmse"
    boosting_type: "gbdt"
    num_leaves: 128
    max_depth: 10
    learning_rate: 0.05
    n_estimators: 1000
    subsample: 0.8
    colsample_bytree: 0.8
    reg_alpha: 0.1
    reg_lambda: 0.1
    min_child_samples: 20
    random_state: 42
    n_jobs: -1

  xgboost:                   # XGBoost 参数
    objective: "reg:squarederror"
    eval_metric: "rmse"
    max_depth: 8
    learning_rate: 0.05
    n_estimators: 1000
    subsample: 0.8
    colsample_bytree: 0.8
    reg_alpha: 0.1
    reg_lambda: 0.1
    random_state: 42
    n_jobs: -1

  catboost:                  # CatBoost 参数
    loss_function: "RMSE"
    iterations: 1000
    learning_rate: 0.05
    depth: 8
    l2_leaf_reg: 3.0
    random_seed: 42
    thread_count: -1

  optuna:                    # 超参搜索
    n_trials: 50
    cv_folds: 5
    direction: "maximize"
    timeout: 3600

  shap:                      # SHAP 可解释性
    max_display: 20
    sample_size: 1000

  training:                  # 训练配置
    test_size: 0.2
    validation_size: 0.2
    random_state: 42
    shuffle: false

  registry:                  # 模型注册
    storage_path: "models/"
    metadata_filename: "meta.json"
    model_filename: "model.pkl"
```

消费方：`model/lgb_model.py`、`model/xgb_model.py`、`model/cat_model.py`、`model/ensemble.py` 均从 `settings.yaml` 的 `model` section 读取参数。

### 1.8 strategy — 策略配置

```yaml
strategy:
  top_k: 30                          # 选股数量
  rebalance_freq: "monthly"          # 调仓频率: daily/weekly/monthly/quarterly
  weight_method: "equal"             # 权重方法: equal/score_weighted/risk_parity
  industry_neutral: true             # 行业中性化
  max_stocks_per_industry: 5         # 单行业最大持仓数
```

### 1.9 risk — 风险控制

```yaml
risk:
  max_position_pct: 0.10             # 单票最大仓位 10%
  max_industry_pct: 0.30             # 单行业最大仓位 30%
  stop_loss_pct: 0.08                # 止损比例 8%
```

### 1.10 backtest — 回测配置

```yaml
backtest:
  initial_cash: 1000000              # 初始资金 100万
  commission_buy: 0.00025            # 买入手续费 万2.5
  commission_sell: 0.00125           # 卖出手续费 含印花税
  slippage_bps: 0.001                # 滑点 0.1%
  benchmark: "000905"                # 基准指数 中证500
```

### 1.11 regime — 市场阶段分析

```yaml
regime:
  benchmark_code: "SH600000"         # 基准股票代码
  benchmark_field: "close"           # 基准价格字段
  ma_window: 60                      # 均线窗口
  vol_window: 60                     # 波动率窗口
```

消费方：`run.py:cmd_regime()` 使用这些配置读取基准数据和计算参数。

---

## 二、workflow_config.yaml 配置详解

`qlib_pipeline/workflow_config.yaml` 专注于 Qlib 框架相关配置，包含以下核心段落：

### 2.1 qlib — Qlib 引擎初始化

```yaml
qlib:
  provider_uri: "D:/trae/qlib_bin"   # Qlib bin 数据目录
  region: "cn"                        # 中国区
```

`provider_uri` 必须与 `settings.yaml` 中 `data_source.qlib_dir` 一致。

### 2.2 data_handler — 数据处理器

```yaml
data_handler:
  class: "DataHandlerLP"
  kwargs:
    start_time: "2012-01-01"         # 数据加载起始
    end_time: "2026-06-25"            # 数据加载截止
    fit_start_time: "2012-01-01"     # 标准化拟合起始
    fit_end_time: "2021-06-30"       # 标准化拟合截止（防泄露）
    instruments: "all"                # 股票池: all / csi300 / csi500
```

### 2.3 dataset — 数据集定义

```yaml
dataset:
  class: "DatasetH"
  handler: "Alpha158"                 # 特征处理器: Alpha158 / Alpha360
  kwargs:
    segments:
      train: ["2013-01-01", "2021-06-30"]
      valid: ["2021-07-01", "2022-12-31"]
      test:  ["2023-01-01", "2023-12-31"]
```

### 2.4 qlib_lgb — Qlib 原生 LightGBM

```yaml
qlib_lgb:
  class: "LGBModel"
  kwargs:
    loss: "mse"                       # mse / rank
    learning_rate: 0.042
    num_leaves: 256
    max_depth: 8
    lambda_l1: 206
    lambda_l2: 581
    colsample_bytree: 0.8
    subsample: 0.8
    early_stopping_rounds: 50
    num_threads: 4
```

### 2.5 backtest — 回测引擎

```yaml
backtest:
  strategy:
    class: "TopkDropoutStrategy"
    kwargs:
      topk: 50
      n_drop: 5
  backtest:
    start_time: "2024-01-01"
    end_time: "2025-06-30"
    account: 10000000
    benchmark: "SH600000"             # 必须使用完整代码格式
    exchange_kwargs:
      open_cost: 0.0005
      close_cost: 0.0015
```

---

## 三、配置优先级

命令行参数 > workflow_config.yaml > settings.yaml

```bash
# 命令行参数覆盖 workflow_config.yaml
python run.py full --handler Alpha360 --loss rank --topk 30

# 使用自定义配置文件
python run.py full --config my_config.yaml
```