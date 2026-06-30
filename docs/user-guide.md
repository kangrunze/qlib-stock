# A股量化选股系统 — 用户使用指南

> 文档版本: v2.2  
> 更新日期: 2026-06-30  
> 适用项目: `d:/project/qlib-stock`

---

## 目录

1. [项目概述](#1-项目概述)
2. [文件架构](#2-文件架构)
3. [环境准备](#3-环境准备)
4. [数据管理](#4-数据管理)
5. [命令行使用指南](#5-命令行使用指南)
6. [配置体系详解](#6-配置体系详解)
7. [模型训练与回测](#7-模型训练与回测)
8. [常见问题与修复记录](#8-常见问题与修复记录)
9. [中长线低波动策略](#9-中长线低波动策略)
10. [AI 辅助调参闭环](#10-ai-辅助调参闭环)
11. [附录](#11-附录)

---

## 1. 项目概述

本项目是一个基于 Microsoft Qlib 的 A 股量化选股系统，核心能力包括：

- **特征工程**：内置 Alpha158（158 因子）和 Alpha360（360 因子）
- **模型训练**：LightGBM / XGBoost / CatBoost，支持排序学习（rank）
- **回测引擎**：Qlib 原生 TopkDropoutStrategy，支持日/周/月线回测
- **可视化**：Plotly + Kaleido 生成分析图表
- **选股推荐**：自动输出 Top-K 股票推荐列表

**技术栈**：Python 3.10、Qlib 0.9.7、LightGBM、Plotly、Pandas、NumPy 1.x

---

## 2. 文件架构

```
qlib-stock/
├── run.py                          # 统一入口（训练/回测/数据/选股等）
├── run_qlib_workflow.py            # CSV → Qlib bin 数据转换
├── qlib_pipeline/
│   ├── workflow_config.yaml        # 默认配置（日常快速训练）
│   ├── workflow_config_midlong.yaml # 中长线低波动配置
│   ├── train.py                    # 训练管线（build_task / init_qlib_env / run_train）
│   ├── dataset.py                  # 配置加载器（load_workflow_config）
│   ├── backtest.py                 # 回测分析（图表生成 / 选股推荐输出）
│   ├── model/                      # 自研模型定义（LGB / XGB / CatBoost）
│   ├── rolling.py                  # Walk-Forward 滚动验证
│   ├── drift.py                    # 特征漂移检测（PSI）
│   ├── ic_stability.py             # IC 稳定性分析
│   ├── tscv.py                     # Purged K-Fold 时序交叉验证
│   ├── regime.py                   # 市场阶段分析
│   └── sensitivity.py              # 超参数敏感性分析
├── qlib_data/cn_data/              # Qlib bin 格式数据目录
│   ├── calendars/day.txt           # 交易日历
│   ├── instruments/all.txt         # 全部股票列表
│   └── features/600000/            # 每只股票独立目录
│       ├── open.day.bin
│       ├── close.day.bin
│       ├── high.day.bin
│       ├── low.day.bin
│       ├── volume.day.bin
│       └── amount.day.bin
├── output/
│   ├── qlib_charts/                # 回测图表输出
│   └── picks/                      # 选股推荐 CSV
└── docs/
    ├── stock-ai-docs.html          # 技术文档（HTML）
    ├── factor-model-guide/         # 因子与模型说明文档
    └── user-guide.md               # 本文档
```

---

## 3. 环境准备

### 3.1 Python 版本

必须使用 TRAE 内置的 Python 3.10.11，不要混用系统 python：

```bash
# 正确
"c:/Users/22179/AppData/Roaming/TRAE SOLO CN/ModularData/ai-agent/vm/tools/python/python.exe" run.py full

# 错误（可能引用 numpy 2.x，与 Qlib 不兼容）
python run.py full
```

### 3.2 关键依赖

| 包 | 版本要求 | 说明 |
|---|---|---|
| pyqlib | 0.9.7 | 核心量化框架 |
| lightgbm | 最新 | 梯度提升模型 |
| numpy | 1.x | **禁止 numpy 2.x**，会导致 Qlib 崩溃 |
| plotly | 最新 | 图表渲染 |
| kaleido | 最新 | Plotly 静态导出 |
| pandas | 最新 | 数据处理 |

> Qlib 0.9.7 有 numpy 2.x 兼容性问题，项目通过 `qlib_pipeline/numpy_compat.py` 进行了补丁处理。

### 3.3 Windows 特殊配置

项目已在 `train.py` 中强制单线程模式，避免多进程崩溃：

```python
os.environ["NUMEXPR_MAX_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
C.joblib_backend = "threading"
C.dataset_process_n_worker = 1
```

---

## 4. 数据管理

### 4.1 数据来源

原始 CSV 数据放在 `D:/data/`，格式为 `{market}.{code}.csv`：

```
D:/data/
├── sh.600000.csv    # 浦发银行
├── sh.600519.csv    # 贵州茅台
├── sz.000001.csv    # 平安银行
└── ...
```

CSV 必须包含以下列：`code, date, open, high, low, close, volume, amount`

### 4.2 数据转换

```bash
# 全量转换（5423 只股票，约 10-20 分钟）
python run.py data --convert

# 小样本快速测试（100 只，约 1 分钟）
python run.py data --convert --sample 100

# 转换为周线
python run.py data --convert --freq week

# 转换为月线
python run.py data --convert --freq month

# 检查数据完整性
python run.py data --check
```

> `run.py full` **不会**自动触发数据转换，必须手动先跑 `data --convert`。

### 4.3 数据频率切换

Qlib 支持 `day / week / month` 三种频率：

```bash
# 日线（默认）
python run.py data --convert --freq day
python run.py full

# 周线
python run.py data --convert --freq week
# 修改 workflow_config.yaml: data_handler.freq = "week"
python run.py full
```

---

## 5. 命令行使用指南

`run.py` 是唯一的命令行入口，支持以下子命令：

### 5.1 核心流程

```bash
# 一键跑通: 训练 + 回测 + 图表 + 选股推荐
python run.py full

# 仅训练模型
python run.py train

# 仅回测（需已有训练结果）
python run.py backtest --rid <recorder_id>

# 选股推荐（从已有回测结果中提取）
python run.py pick --rid <recorder_id> [--date 2025-12-31]
```

### 5.2 数据管理

```bash
python run.py data --download        # 从 AKShare 下载数据
python run.py data --convert         # CSV → Qlib bin
python run.py data --convert --sample 100   # 小样本测试
python run.py data --convert --freq week    # 周线转换
python run.py data --check           # 数据完整性检查
```

### 5.3 稳健性验证（Phase 1）

```bash
# 滚动 Walk-Forward 验证
python run.py rolling --n-folds 6

# 特征漂移 + 概念漂移检测（PSI）
python run.py drift

# IC 稳定性分析（ICIR / IC 衰减 / 分层 IC）
python run.py ic-stability
```

### 5.4 鲁棒性深化（Phase 3）

```bash
# Purged K-Fold 时序交叉验证
python run.py tscv --n-splits 5

# 市场阶段稳定性分析
python run.py regime

# 超参数敏感性分析
python run.py sensitivity

# 关键年份独立回测
python run.py key-years --years 2020,2022,2024
```

### 5.5 命令行参数覆盖

所有配置项都可以通过命令行临时覆盖，不修改配置文件：

```bash
python run.py full --handler Alpha360 --loss rank --topk 30

# 使用自定义配置文件
python run.py full --config qlib_pipeline/workflow_config_midlong.yaml
```

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--handler` | 特征处理器 | `Alpha158` |
| `--model-type` | 模型类型 | `lgb` |
| `--loss` | 损失函数 (`mse` / `rank`) | `mse` |
| `--topk` | 持仓股票数 | `50` |
| `--pick-topk` | 选股推荐数量 | `30` |
| `--experiment` | 实验名称 | `qlib_pipeline` |
| `--output-dir` | 图表输出目录 | `output/qlib_charts` |
| `--config` | 配置文件路径 | `qlib_pipeline/workflow_config.yaml` |

---

## 6. 配置体系详解

配置统一在 `workflow_config.yaml` 中管理，通过 `--config` 切换。配置分为 17 个段落：

### 6.1 配置项映射表

| 配置段 | 消费方 | 说明 |
|---|---|---|
| `qlib` | `train.py:init_qlib_env()` | Qlib 引擎初始化 |
| `data_source` | `run_qlib_workflow.py` | 数据下载/转换 |
| `stock_universe` | `run_qlib_workflow.py` | CSV 转 bin 时的过滤规则 |
| `data_handler` | `train.py:build_task()` | Alpha158/360 的 kwargs |
| `dataset` | `train.py:build_task()` | DatasetH 的 segments |
| `model` | `train.py`（旧管线） | 自研模型超参（run.py 不使用） |
| `qlib_lgb` | `train.py:build_task()` | Qlib 原生 LGBModel（run.py 实际使用） |
| `optuna` | `model/optuna_tuner.py` | 超参搜索 |
| `shap` | `model/shap_analysis.py` | SHAP 可解释性 |
| `strategy` | Qlib Pipeline / 自研回测 | 选股策略 |
| `risk` | 自研回测引擎 | 风控参数 |
| `backtest` | `run.py:cmd_full()` | 回测引擎 + 交易参数 |
| `experiment` | `train.py:run_train()` | MLflow 实验名 |
| `training` | `train.py`（旧管线） | 训练划分参数 |
| `registry` | `model/model_registry.py` | 模型存储路径 |
| `features` | `feature_engine/` | 自定义因子 |
| `labels` | `train.py:build_task()` | 预测目标定义 |

### 6.2 时间体系说明

配置中包含四层时间，含义完全不同：

```
|<--- data_handler.start/end (全部加载数据) ----------------------->|
|<--- fit (标准化拟合) -->|                                        |
|<--- train (训练) -->|<-- valid (早停) -->|<-- test (评估) -->|
                                                          |<-- backtest -->|
```

| 时间字段 | 作用 | 约束 |
|---|---|---|
| `data_handler.start_time / end_time` | Qlib 从 bin 加载的全部数据范围 | 至少比训练集早 1 年（技术指标需要 lookback） |
| `data_handler.fit_start_time / fit_end_time` | 标准化器拟合范围 | `fit_end_time` 不得覆盖验证/测试集（防泄露） |
| `dataset.segments` | 训练/验证/测试三段切分 | 连续不重叠，纯时间序列 |
| `backtest.backtest.start_time / end_time` | 回测模拟交易时间段 | 通常与 test 段对齐，不能超出日历最后交易日 |

### 6.3 关键可调参数速查

| 参数 | 配置段 | 默认值 | 影响 |
|---|---|---|---|
| `handler` | `dataset` | `Alpha158` | 因子数量（Alpha360 更丰富但训练更慢） |
| `loss` | `qlib_lgb.kwargs` | `mse` | `rank` 更适合选股，`mse` 适合预测绝对收益 |
| `topk` | `strategy.kwargs` | `50` | 持仓股票数，越大越分散 |
| `n_drop` | `strategy.kwargs` | `5` | 每期替换数，越大换手率越高 |
| `learning_rate` | `qlib_lgb.kwargs` | `0.042` | 学习率，越小越稳定但训练越慢 |
| `lambda_l1 / lambda_l2` | `qlib_lgb.kwargs` | `206 / 581` | 正则化，越大越抑制过拟合 |
| `max_depth` | `qlib_lgb.kwargs` | `8` | 树深度，越大模型越复杂 |
| `instruments` | `data_handler` | `csi300` | 股票池：`csi300` / `csi500` / `csi800` / `all` |
| `rebalance_freq` | `strategy` | `monthly` | 调仓频率：`daily` / `weekly` / `monthly` |
| `industry_neutral` | `strategy` | `false` | 是否行业中性化 |

---

## 7. 模型训练与回测

### 7.1 训练流程

```bash
# 1. 初始化 Qlib（加载 bin 数据）
# 2. 构建 DatasetH（Alpha158/360 + 训练/验证/测试三段划分）
# 3. 创建 LGBModel（LightGBM）
# 4. 训练模型（early stopping）
# 5. 保存模型和参数到 MLflow Recorder

python run.py train
# 输出: recorder_id = xxxxxxxx
```

训练日志解读：

```
[20]    train's l2: 0.993201    valid's l2: 0.994234
[200]   train's l2: 0.980401    valid's l2: 0.986770
```

- `train's l2`：训练集 L2 损失（越小越好）
- `valid's l2`：验证集 L2 损失
- 两个值都在 0.98~0.99 之间是正常的（未标准化的原始 MSE）
- 关键看趋势：如果 train 持续下降但 valid 不再下降，说明过拟合

### 7.2 回测流程

```bash
# 1. 加载已训练模型
# 2. 在测试集上生成预测信号
# 3. 使用 TopkDropoutStrategy 模拟交易
# 4. 计算收益、回撤、夏普比等指标
# 5. 保存结果到 MLflow Recorder

python run.py backtest --rid <recorder_id>
```

### 7.3 输出结果

回测完成后，输出目录结构：

```
output/
├── qlib_charts/
│   ├── qlib_01_cumulative_return.png    # 累计收益曲线
│   ├── qlib_02_pred_distribution.png   # 预测值分布
│   ├── qlib_03_monthly_heatmap.png     # 月度收益热力图
│   ├── qlib_04_drawdown.png            # 回撤曲线
│   └── qlib_05_rolling_sharpe.png      # 滚动夏普比率
└── picks/
    └── stock_picks_20260625.csv         # 选股推荐列表
```

---

## 8. 常见问题与修复记录

### 8.1 `ValueError: The benchmark ['SH000300'] does not exist`

**现象**：回测时报错，说基准指数不存在。

**根因**：Qlib 源码 `create_account_instance` 的 bug — 当 `benchmark=None` 时，它传入空字典 `{}` 而非 `None`，导致 `_cal_benchmark` 回退到 `CSI300_BENCH` 默认值 `SH000300`。但 `SH000300` 是指数代码，不是普通股票，不在 features 目录中。

**修复**：改为使用实际存在的股票代码作为 benchmark：

```yaml
backtest:
  backtest:
    benchmark: "600000"    # 浦发银行，存在 features/600000/ 目录中
```

> 在 `run_qlib_workflow.py` 中也将硬编码的 `benchmark = "SH000300"` 改为 `benchmark = None`。

### 8.2 `FileNotFoundError` during `shutil.rmtree(features_dir)`

**现象**：数据转换时 `shutil.rmtree` 清理旧 features 目录失败。

**根因**：Windows 对大目录的 `shutil.rmtree` 可能因路径长度或文件锁失败。

**修复**：添加 try/except 回退到逐文件删除：

```python
if features_dir.exists():
    try:
        shutil.rmtree(features_dir)
    except Exception:
        # 回退：逐个删除文件和目录
        for root, dirs, files in os.walk(str(features_dir), topdown=False):
            for name in files:
                try: os.remove(os.path.join(root, name))
                except OSError: pass
            for name in dirs:
                try: os.rmdir(os.path.join(root, name))
                except OSError: pass
```

### 8.3 `IndexError: index 1569 is out of bounds for axis 0 with size 1569`

**现象**：回测循环访问日历索引越界。

**根因**：回测 `end_time` 超出日历最后一个交易日。日历有 1569 条记录（索引 0~1568），但回测尝试访问索引 1569。

**修复**：调整回测结束日期到实际最后交易日之前：

```yaml
backtest:
  backtest:
    end_time: "2026-06-25"    # 日历最后交易日是 2026-06-26，回测必须在之前结束
```

### 8.4 Pandas `FutureWarning: 'M' is deprecated`

**现象**：`resample_daily_to_freq()` 中使用 `"M"` 触发弃用警告。

**修复**：改为 `"ME"`（Month End）：

```python
resample_rule = "W-FRI" if freq == "week" else "ME"
```

### 8.5 训练日志 `valid's l2` 停止下降但 `train's l2` 仍在下降

**现象**：过拟合迹象。训练集损失持续下降，验证集不再改善。

**应对**：
- 增大 `lambda_l1` / `lambda_l2`（正则化）
- 降低 `max_depth` / `num_leaves`
- 增大 `subsample` / `colsample_bytree`
- 减小 `learning_rate`

---

## 9. 中长线低波动策略

### 9.1 设计目标

- 年化收益：15%~25%
- 最大回撤：控制在 15% 以内
- 夏普比率：> 1.5
- 月频调仓，换手率 < 20%/月

### 9.2 配置文件

已预置配置文件：`qlib_pipeline/workflow_config_midlong.yaml`

### 9.3 与默认配置的 8 处关键差异

| 配置项 | 默认值 | 中长线值 | 改动原因 |
|---|---|---|---|
| `handler` | `Alpha158` | `Alpha360` | 因子更丰富，Rank IC 高 2~3 个百分点 |
| `loss` | `mse` | `rank` | 排序学习（Lambdarank），只关心排序不关心绝对值 |
| `topk` | 50 | 25 | 集中持仓，避免收益被摊薄 |
| `n_drop` | 5 | 3 | 月度换手仅 12%，低摩擦成本 |
| `lambda_l1` | 205.7 | 300.0 | 加大 L1，稀疏化特征权重 |
| `lambda_l2` | 580.9 | 800.0 | 加大 L2，抑制过拟合 |
| `训练期` | 4.5 年 | 9.5 年 | 覆盖完整牛熊周期（2015 牛市、2018 熊市） |
| `industry_neutral` | `false` | `true` | 行业中性化，分散系统性风险 |

### 9.4 使用方式

```bash
# 使用中长线配置一键跑通
python run.py full --config qlib_pipeline/workflow_config_midlong.yaml

# 快速探索（100 只股票，5 分钟出结果）
python run.py data --convert --sample 100
python run.py full --config qlib_pipeline/workflow_config_midlong.yaml
```

### 9.5 为什么是 rank 而不是 mse

`mse` 会让模型花大量精力拟合收益率的绝对值（如 ±1% 的差异），而 `rank` 只关心"哪只股票涨得更多"。在预测长周期收益时，排序信号更稳健，不容易被单日异常波动干扰。

---

## 10. AI 辅助调参闭环

### 10.1 核心思路

建立 **"实验 → 记录 → 分析 → 建议 → 再实验"** 的自动化循环：

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   ① AI 分析历史记录 → 建议下一组参数                  │
│         ↓                                            │
│   ② 写入新 config 文件                               │
│         ↓                                            │
│   ③ 执行 python run.py full --config xxx.yaml        │
│         ↓                                            │
│   ④ 提取指标，追加到实验记录                          │
│         ↓                                            │
│   ⑤ 更新调参经验（如发现新规律）                      │
│         ↓                                            │
│   ⑥ 回到 ①，循环                                     │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 10.2 实验记录维度

每次实验需要结构化记录：

| 维度 | 内容 | 用途 |
|---|---|---|
| 参数快照 | handler、loss、topk、n_drop、learning_rate、lambda_l1/l2、max_depth | 复现实验 |
| 训练指标 | IC、ICIR、Rank IC、训练耗时 | 判断模型质量 |
| 回测指标 | 年化收益、夏普、最大回撤、超额收益、换手率 | 最终评判 |
| 时间戳 | 实验日期、数据截止日 | 追溯数据版本 |
| 备注 | 市场阶段观察（如"回测期是牛市，结果可能虚高"） | 避免幸存者偏差 |

### 10.3 实验分档策略

| 实验类型 | 数据量 | 耗时 | 用途 |
|---|---|---|---|
| 快速探索 | `--sample 100` | ~5 分钟 | 验证参数组合是否合理、有无 bug |
| 候选验证 | 全量 csi300 | ~30 分钟 | 验证快速探索中表现好的参数 |
| 最终验证 | 全量 + rolling + tscv | ~2 小时 | 稳健性验证，确认非过拟合 |

**建议流程**：

```
AI 建议新参数 → 快速探索（100 只）→ 看结果
    ↓ 表现好
候选验证（全量 csi300）→ 看结果
    ↓ 表现好
最终验证（全量 + rolling + tscv）→ 记录入库
```

### 10.4 AI 分析能力

积累一定量实验后，AI 可以做：

1. **相关性分析** — 哪些参数对收益/回撤影响最大？例如发现 `lambda_l2` 从 200 提到 800 后回撤降了 3%，但收益也降了 2%，量化取舍
2. **趋势发现** — Alpha360 比 Alpha158 的 IC 稳定高 2%，但训练慢 3 倍，是否值得？
3. **失效模式识别** — 某些参数组合在牛市表现好、熊市表现差，标记"回测期是大牛市，结果可能虚高"
4. **边界探索** — 已有实验覆盖了哪些参数空间？哪些区域还没试过？主动建议"topk=15 还没试过"

### 10.5 使用示例

```bash
# 用户说：帮我试试降低 topk 到 20
# AI 操作：
#   1. 复制当前配置 → workflow_config_exp_001.yaml
#   2. 修改 topk: 50 → 20
#   3. 运行 python run.py data --convert --sample 100
#   4. 运行 python run.py full --config workflow_config_exp_001.yaml
#   5. 提取回测指标，追加到实验记录
#   6. 对比历史结果，给出分析建议
```

---

## 11. 附录

### 11.1 命令速查表

```bash
# === 首次使用 ===
python run.py data --convert                    # 转数据
python run.py full                               # 跑全流程

# === 日常迭代 ===
python run.py train                              # 仅训练
python run.py backtest --rid <ID>               # 仅回测
python run.py pick --rid <ID>                   # 仅选股

# === 参数覆盖 ===
python run.py full --handler Alpha360 --loss rank --topk 25
python run.py full --config qlib_pipeline/workflow_config_midlong.yaml

# === 稳健性验证 ===
python run.py rolling --n-folds 6
python run.py drift
python run.py ic-stability
python run.py tscv --n-splits 5
python run.py regime
python run.py sensitivity
```

### 11.2 重要文件路径

| 文件 | 路径 |
|---|---|
| 默认配置 | `qlib_pipeline/workflow_config.yaml` |
| 中长线配置 | `qlib_pipeline/workflow_config_midlong.yaml` |
| 统一入口 | `run.py` |
| 数据转换 | `run_qlib_workflow.py` |
| 训练管线 | `qlib_pipeline/train.py` |
| 回测分析 | `qlib_pipeline/backtest.py` |
| Qlib 数据 | `qlib_data/cn_data/` |
| 图表输出 | `output/qlib_charts/` |
| 选股推荐 | `output/picks/` |

### 11.3 技术文档索引

- `docs/stock-ai-docs.html` — 完整技术文档（HTML，含架构图和 FAQ）
- `docs/factor-model-guide/factor-model-guide.html` — 因子与模型说明文档
- `docs/user-guide.md` — 本文档（命令行使用和配置速查）
