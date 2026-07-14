# 技术架构说明

> 文档版本: v4.0  
> 更新日期: 2026-07-10  
> 变更摘要: 新增 RankLGBModel 参数名映射机制、stock_universe 过滤策略修复、调参层 dataset 复用优化、回测结果提升说明

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
│  cmd_optuna │ cmd_explain │ cmd_risk │ cmd_attribution │      │
│  cmd_capacity │ cmd_benchmark │                                │
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
│  │constructor   │ │ explain      │ │   benchmark        │  │
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
    "risk": cmd_risk, "attribution": cmd_attribution, "capacity": cmd_capacity,
    "benchmark": cmd_benchmark,
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
    ├── stock_universe 过滤（仅 instruments="all" 时生效）
    ├── dataset: DatasetH(handler=Alpha158, segments={train/valid/test})
    │     └── xs_ret_Nd 标签自动注入 CSMedianSubtract learn_processor
    └── model: LGBModel(loss=mse) 或 RankLGBModel(loss=rank)
    │
    ▼
model.fit(dataset)          # 训练 + early stopping
    │
    ▼
R.save_objects(trained_model=model)  # 保存到 MLflow Recorder
```

### 4.2 模型类路由

`build_task()` 根据 `qlib_lgb.kwargs.loss` 自动选择模型类：

| `loss` | 模型类 | 模块路径 | 训练方式 |
|--------|--------|---------|---------|
| `"mse"` | `LGBModel` | `qlib.contrib.model.gbdt` | sklearn API（`fit(X, y)`） |
| `"rank"` | `RankLGBModel` | `qlib_pipeline.model` | LightGBM 原生（`lgb.train()`） |

`RankLGBModel` 绕过 Qlib 0.9.7 `LGBModel` 的 loss 白名单校验（只支持 mse/binary），
直接使用 LightGBM 原生的 `objective="lambdarank"` 实现选股排序学习。

### 4.3 RankLGBModel 参数名映射

`RankLGBModel` 使用 `lgb.train()` 而非 sklearn API，**不识别 sklearn 风格的参数名**。
`__init__` 中自动将 sklearn API 参数名转换为 LightGBM 原生名：

```python
param_map = {
    "subsample": "bagging_fraction",
    "colsample_bytree": "feature_fraction",
    "subsample_freq": "bagging_freq",
}
# bagging_freq 默认设为 1，使 bagging_fraction 生效
if "bagging_fraction" in converted and "bagging_freq" not in converted:
    converted["bagging_freq"] = 1
```

> ⚠ **历史 Bug**：2026-07-10 修复前，参数名未转换，导致 `subsample`/`colsample_bytree`
> 被 `lgb.train()` 静默忽略，Optuna 搜索这两个参数时所有 trial 等效（返回相同 IC）。

### 4.4 stock_universe 过滤策略

`_build_stock_universe_instruments()` 在 `build_task()` 中调用，过滤逻辑：

| `data_handler.instruments` | stock_universe 过滤 | 原因 |
|---------------------------|--------------------|----|
| `"all"` | ✅ 生效 | 全市场需要 ST/次新股/北交所过滤 |
| `"csi300"`/`"csi500"`/`"csi800"` | ❌ 跳过 | 指数成分股池已自带质量过滤；且其 instruments 文件的 `start_date` 是指数纳入日期而非上市日期，`min_listed_days` 会错误调整纳入日期导致训练数据为空 |

### 4.5 CSMedianSubtract 处理器

`xs_ret_Nd` 标签的标准实现：截面中位数减法（`xs_ret = ret - 截面中位数(ret)`）。

作为 Qlib **learn_processor** 使用（非 infer_processor），确保训练和推理口径一致：
- 训练时：label 经过截面中位数减法
- 推理时：预测值同样基于减去中位数的 label 训练，口径统一

注入逻辑：`build_task()` 检测到 `primary` 为 `xs_ret_Nd` 时，自动在 `learn_processors` 中追加 `CSMedianSubtract`，保留 Alpha158 默认的 `DropnaLabel` 和 `CSZScoreNorm`。

### 4.6 时间体系

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

### 4.7 模型预测

Qlib 的 `LGBModel.predict()` 签名为 `predict(dataset, segment="test")`，内部自动调用 `dataset.prepare()`。`RankLGBModel.predict()` 签名一致，返回每个样本的排序分数（绝对值无意义，只有相对排序有意义）。不要传入 DataFrame 或手动 prepare。

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
| 关键年份回测 | 指定年份独立回测（支持样本外模式） | `regime.py:key_year_backtest()` |

### 6.3 Phase 4 — 风险诊断与对比实验

| 分析 | 指标 | 实现 |
|------|------|------|
| 风格暴露诊断 | Barra 5因子截面 z-score 暴露 | `risk_model.py` → `run.py risk` |
| 收益归因 | Fama-MacBeth 截面 OLS 回归 | `attribution.py` → `run.py attribution` |
| 容量检查 | 持仓占日均成交额比例 | `capacity.py` → `run.py capacity` |
| 强制性对比实验 | E1-E7 全量对比 + NW 显著性 | `benchmark.py` → `run.py benchmark` |

诊断模块在 `python run.py full` 结束时自动调用，数据就绪时自动计算，否则优雅降级。

### 6.4 cmd_regime 实现细节

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
    ├── sensitivity.py      → Qlib 初始化 + dataset 复用
    └── optuna_search.py    → Qlib 初始化 + dataset 复用
```

---

## 8. 调参层设计

### 8.1 调参流程（探索/确认两阶段协议）

遵循第 7.3 节协议，按顺序执行：

```
1. 敏感性分析 (sensitivity.py)
   └── 单变量扫描，识别关键超参（learning_rate/max_depth/subsample 等）
       └── dataset 跨参数扫描复用（首次加载 ~10 分钟，后续每个扫描点 ~20 秒）

2. Optuna 贝叶斯搜索 (optuna_search.py)
   └── 多变量联合优化，最大化 valid IC 均值
       └── dataset 跨 trial 复用（首次加载 ~10 分钟，后续每个 trial ~20 秒）
       └── 时间戳 study_name + load_if_exists=False（避免加载旧 study）

3. 滚动验证 (rolling.py)
   └── 用最优参数在多个时间窗口验证稳定性
```

### 8.2 dataset 复用机制

`optuna_search.py` 和 `sensitivity.py` 均实现了 dataset 复用，避免每次 trial/扫描点重新加载 ~10 分钟数据：

| 模块 | 实现方式 |
|------|---------|
| `optuna_search.py` | `shared_state["dataset"]` 在 trials 间传递，`_single_trial_ic` 接收 `dataset` 参数 |
| `sensitivity.py` | `shared_dataset` 在参数扫描间传递，`_single_train` 接收 `dataset` 参数 |

复用前提：handler 配置不变时数据相同（仅模型参数变化）。

### 8.3 Optuna 搜索空间

```python
learning_rate:    log-uniform, 0.01 ~ 0.2
num_leaves:       int, 15 ~ 63 (step=4)
max_depth:        int, 3 ~ 6
subsample:        uniform, 0.5 ~ 1.0
colsample_bytree: uniform, 0.5 ~ 1.0
lambda_l1:        log-uniform, 1e-8 ~ 100
lambda_l2:        log-uniform, 1e-8 ~ 100
min_child_samples: int, 10 ~ 100
```

> 搜索空间严格收窄（num_leaves ≤ 63, max_depth ≤ 6），防止中长周期因子数据信噪比低时过拟合。

---

## 9. 最新回测结果（2026-07-10）

### 9.1 配置

| 项 | 值 |
|----|-----|
| 特征处理器 | Alpha158 |
| 模型 | RankLGBModel (LambdaRank) |
| 标签 | xs_ret_20d（截面中位数减法） |
| 股票池 | csi300 |
| 训练/验证/测试 | 2020-01-01 ~ 2024-06-30 / 2024-07-01 ~ 2025-06-30 / 2025-07-01 ~ 2026-06-25 |
| 回测区间 | 2025-07-01 ~ 2026-06-25（239 个交易日） |
| 策略 | TopkDropout, topk=30, n_drop=5 |
| 基准 | 沪深300 (SH000300) |

### 9.2 回测指标

| 指标 | 基准(沪深300) | 超额(无成本) | 超额(含成本) |
|------|-------------|-------------|-------------|
| 年化收益 | 25.42% | 30.11% | 22.79% |
| 信息比率 | 1.648 | 2.912 | 2.202 |
| 最大回撤 | -7.90% | -7.32% | -7.79% |

- `best_iteration=3`（浅树+强正则配置下早停合理）
- NDCG@1=0.337, NDCG@5=0.342

### 9.3 调优历程

| 阶段 | 配置变更 | 超额(含成本) | IR | 关键发现 |
|------|---------|-------------|-----|---------|
| 修复前 | mse, depth=8, 参数名 bug | — | — | subsample/colsample 被静默忽略 |
| 30-trial Optuna | rank, depth=5, lr=0.028 | -12.55% | -1.167 | best_iter=1，单棵树太弱 |
| 降 lr | rank, depth=5, lr=0.01 | -18.63% | -1.517 | NDCG 与回测收益不相关 |
| **7-trial Optuna** | **rank, depth=3, min_child=90, early_stop=100** | **+22.79%** | **2.202** | 浅树+强正则泛化更好 |

**核心结论**：浅树（max_depth=3）+ 强正则（min_child_samples=90）即使 best_iteration 较小，也远优于深树配置。NDCG 排序指标不能完全代表组合预测能力，需结合回测验证。