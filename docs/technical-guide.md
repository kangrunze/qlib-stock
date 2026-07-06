# 技术架构说明

> 文档版本: v3.0  
> 更新日期: 2026-07-06

---

## 目录

1. [系统架构](#1-系统架构)
2. [数据流](#2-数据流)
3. [核心模块](#3-核心模块)
4. [训练管线](#4-训练管线)
5. [回测引擎](#5-回测引擎)
6. [稳健性验证体系](#6-稳健性验证体系)
7. [配置管理设计](#7-配置管理设计)

---

## 1. 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                        run.py (统一入口)                      │
│  cmd_train │ cmd_full │ cmd_backtest │ cmd_pick │ cmd_data   │
│  cmd_rolling│cmd_drift│ cmd_ic_stab │ cmd_tscv │ cmd_regime  │
│  cmd_sensitivity│cmd_key_years│cmd_update│cmd_validate│      │
│  cmd_optuna │ cmd_explain │                                  │
├─────────────────────────────────────────────────────────────┤
│                    qlib_pipeline (核心管线)                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │
│  │ train.py │ │backtest.py│ │dataset.py│ │ic_stability.py│  │
│  ├──────────┤ ├──────────┤ ├──────────┤ ├──────────────┤  │
│  │rolling.py│ │ drift.py │ │ tscv.py  │ │  regime.py   │  │
│  │sensitivity│ │validate.py│ │model.py │ │numpy_compat  │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                      research (研究层)                        │
│  ┌──────────────┐ ┌──────────────┐ ┌────────────────────┐  │
│  │significance  │ │ risk_model   │ │   attribution      │  │
│  │portfolio_    │ │ capacity     │ │ experiment_tracker │  │
│  │constructor   │ │ explain      │ │                    │  │
│  └──────────────┘ └──────────────┘ └────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                      tuning (调参层)                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ optuna_search.py                                     │  │
│  └──────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                      data_center (数据层)                     │
│  ┌──────────────┐ ┌──────────────┐ ┌────────────────────┐  │
│  │daily_update.py│ │data_store.py │ │  csv_loader.py    │  │
│  └──────────────┘ └──────────────┘ └────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    config (配置层)                            │
│  ┌──────────────────┐ ┌──────────────────────────────┐     │
│  │ settings.yaml    │ │ workflow_config.yaml         │     │
│  │ (全局配置)        │ │ (Qlib 工作流配置)             │     │
│  └──────────────────┘ └──────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### 分层职责

| 层 | 职责 | 关键文件 |
|----|------|----------|
| 入口层 | 命令行解析、子命令路由、配置加载 | `run.py` |
| 管线层 | Qlib 训练、回测、稳健性分析 | `qlib_pipeline/` |
| 研究层 | 统计检验、归因、行业约束、SHAP 可解释性 | `research/` |
| 调参层 | Optuna 贝叶斯超参数搜索 | `tuning/` |
| 数据层 | 数据下载、格式转换、增量更新 | `data_center/` |
| 配置层 | 全局参数管理、消冗余 | `config/` |

---

## 2. 数据流

### 2.1 全量数据流

```
AKShare API
    │
    ▼
┌──────────┐     run_qlib_workflow.py     ┌──────────────┐
│ D:/data/ │ ──────────────────────────▶  │ D:/download/     │
│ *.csv    │   CSV → Qlib bin 格式转换     │ qlib_bin/    │
└──────────┘                              │ ├ calendars/ │
                                          │ ├ instruments/│
                                          │ └ features/  │
                                          └──────────────┘
                                                 │
                                          qlib.init()
                                                 │
                                                 ▼
                                          ┌──────────────┐
                                          │ Qlib Dataset │
                                          │ (Alpha158/360)│
                                          └──────────────┘
```

### 2.2 增量更新流

```
python run.py update
    │
    ▼
┌────────────────────┐
│ daily_update.py    │
│ 1. 从 features/ 发 │
│    现股票列表       │
│ 2. AKShare 增量下载 │
│ 3. 更新 parquet     │
│ 4. 更新 .day.bin    │
│ 5. 更新 calendars/  │
│ 6. 更新 instruments/│
└────────────────────┘
```

### 2.3 Qlib bin 文件格式

每个 `.day.bin` 文件存储单只股票单个字段的时间序列：

```
[start_date_idx, val_0, val_1, val_2, ...]
```

- 第一个值为全局日历中的起始索引（`int32`）
- 后续值为浮点数时间序列
- 通过全局日历 (`calendars/day.txt`) 定位日期

---

## 3. 核心模块

### 3.1 run.py — 统一入口

所有功能通过子命令调用，命令路由在 `main()` 中：

```python
cmd_map = {
    "train": cmd_train, "backtest": cmd_backtest, "full": cmd_full,
    "pick": cmd_pick, "data": cmd_data, "update": cmd_update,
    "rolling": cmd_rolling, "drift": cmd_drift, "ic-stability": cmd_ic_stability,
    "tscv": cmd_tscv, "regime": cmd_regime,
    "sensitivity": cmd_sensitivity, "key-years": cmd_key_years,
    "validate-picks": cmd_validate_picks,
    "optuna": cmd_optuna, "explain": cmd_explain,
}
```

启动时通过 `_load_settings()` 加载 `config/settings.yaml`，所有 argparse 默认值从配置读取。

### 3.2 train.py — 训练管线

提供三个核心函数：

| 函数 | 用途 |
|------|------|
| `init_qlib_env(config)` | 初始化 Qlib 引擎，设置单线程模式 |
| `build_task(config)` | 从 workflow_config 构建 task dict（dataset + model） |
| `run_train(config, experiment_name)` | 完整训练流程：构建 → 训练 → 保存到 MLflow Recorder |

### 3.3 data_store.py — 统一数据存储

通过工厂模式屏蔽底层数据格式差异：

```python
store = DataStore()  # 自动根据 settings.yaml 的 data_format 创建后端
```

支持三种后端：
- `CsvDataStore` — 从 CSV 文件读取
- `ParquetDataStore` — 从 Parquet 文件读取（DuckDB 后端）
- `QlibDataStore` — 从 Qlib bin 格式读取

### 3.4 daily_update.py — 增量更新

核心逻辑：
1. 从 `features/` 目录发现股票列表（无需依赖 CSV）
2. 通过 AKShare 下载最近 N 天数据
3. 计算全局日期索引，定位 bin 文件追加位置
4. 更新 `calendars/day.txt`、`instruments/all.txt`、`workflow_config.yaml`

---

## 4. 训练管线

### 4.1 训练流程

```
workflow_config.yaml
    │
    ▼
build_task(config)
    │
    ├── dataset: DatasetH(handler=Alpha158, segments={train/valid/test})
    └── model: LGBModel(loss=mse, ...)
    │
    ▼
model.fit(dataset)          # 训练 + early stopping
    │
    ▼
R.save_objects(trained_model=model)  # 保存到 MLflow Recorder
```

### 4.2 时间体系

```
|<--- data_handler.start/end (全部加载数据) ----------------------->|
|<--- fit (标准化拟合) -->|                                        |
|<--- train (训练) -->|<-- valid (早停) -->|<-- test (评估) -->|
                                                       |<-- backtest -->|
```

关键约束：
- `fit_end_time` 不得覆盖验证/测试集（防数据泄露）
- `backtest.end_time` 不能超出日历最后交易日
- 训练集需要比实际训练期早至少 1 年（技术指标 lookback）

### 4.3 模型预测

Qlib 的 `LGBModel.predict()` 签名为 `predict(dataset, segment="test")`，内部自动调用 `dataset.prepare()`。不要传入 DataFrame 或手动 prepare。

---

## 5. 回测引擎

### 5.1 回测流程

```
已训练模型 + Dataset
    │
    ▼
SignalRecord.generate()     # 生成预测信号
    │
    ▼
PortAnaRecord.generate()    # 模拟交易（TopkDropoutStrategy）
    │
    ▼
report_normal.pkl           # 回测报告
port_analysis.pkl           # 组合分析
pred.pkl                    # 预测信号
```

### 5.2 回测输出

| 文件 | 内容 |
|------|------|
| `qlib_01_cumulative_return.png` | 累计收益曲线 |
| `qlib_02_pred_distribution.png` | 预测值分布 |
| `qlib_03_monthly_heatmap.png` | 月度收益热力图 |
| `qlib_04_drawdown.png` | 回撤曲线 |
| `qlib_05_rolling_sharpe.png` | 滚动夏普比率 |

### 5.3 Benchmark 注意事项

- 使用 `SH000300`（沪深300指数）作为回测基准
- 基准指数数据必须存在于 `features/SH000300/close.day.bin`
- 回测引擎会从 `backtest.backtest.benchmark` 读取并使用该指数的真实价格曲线

---

## 6. 稳健性验证体系

### 6.1 Phase 1 — 基础稳健性

| 分析 | 指标 | 实现 |
|------|------|------|
| Walk-Forward 验证 | 各窗口收益、夏普、回撤 | `rolling.py` |
| 特征漂移检测 | PSI（群体稳定性指数） | `drift.py` |
| IC 稳定性 | ICIR、IC 衰减曲线、分层 IC | `ic_stability.py` |

### 6.2 Phase 3 — 深化鲁棒性

| 分析 | 指标 | 实现 |
|------|------|------|
| Purged TSCV | 防泄露的时序交叉验证 | `tscv.py` |
| 市场阶段分析 | 牛/熊/震荡市分阶段 IC | `regime.py` |
| 敏感性分析 | 超参数对 IC 的影响 | `sensitivity.py` |
| 关键年份回测 | 指定年份独立回测 | `regime.py:key_year_backtest()` |

### 6.3 cmd_regime 实现细节

市场阶段分析通过从 bin 文件直接读取基准股票价格数据，配合日历构造纯 DatetimeIndex 的 Series，避免 Qlib `D.features()` 返回 MultiIndex 导致的日期匹配问题。

```python
# 从 bin 文件直接读取
benchmark_path = Path(qlib_dir) / "features" / benchmark_code / f"{benchmark_field}.day.bin"
# 配合 calendars/day.txt 构造纯日期索引
dates = all_dates[start_idx:start_idx + len(close_values)]
price = pd.Series(close_values, index=pd.to_datetime(dates))
```

---

## 7. 配置管理设计

### 7.1 设计原则

1. **单一配置源**：`config/settings.yaml` 是唯一的全局配置，所有硬编码参数均已提取
2. **分层配置**：`settings.yaml`（全局）+ `workflow_config.yaml`（Qlib 专用）
3. **消冗余**：已删除 `model_config.yaml`、`stock_universe.yaml`、`workflow_config_midlong.yaml` 等冗余配置文件，模型超参统一在 `workflow_config.yaml` 的 `qlib_lgb` 段管理
4. **命令行覆盖**：所有配置项可通过命令行参数临时覆盖

### 7.2 配置消费关系

```
settings.yaml
    ├── run.py              → argparse 默认值
    ├── daily_update.py     → qlib_dir
    ├── data_store.py       → data_format, csv_dir, qlib_dir
    ├── csv_loader.py       → csv_dir, start_date
    └── cmd_regime          → regime.*

workflow_config.yaml
    ├── train.py            → init_qlib_env, build_task
    ├── backtest.py         → 回测参数
    ├── rolling.py          → Qlib 初始化
    ├── tscv.py             → Qlib 初始化
    ├── regime.py           → Qlib 初始化
    └── sensitivity.py      → Qlib 初始化
```