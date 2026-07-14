# A股量化选股系统 — Qlib Pipeline

基于 Microsoft Qlib 的 A 股量化选股系统，支持特征工程、多模型训练、回测分析、选股推荐、稳健性验证和风险诊断。

## 最新回测结果（2026-07-10）

| 指标 | 基准(沪深300) | 超额(无成本) | 超额(含成本) |
|------|-------------|-------------|-------------|
| 年化收益 | 25.42% | 30.11% | **22.79%** |
| 信息比率 | 1.648 | 2.912 | **2.202** |
| 最大回撤 | -7.90% | -7.32% | -7.79% |

> 配置：Alpha158 + RankLGBModel(LambdaRank) + xs_ret_20d + csi300 + Top-30
> 回测区间：2025-07-01 ~ 2026-06-25（239 个交易日）

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载数据并转换
python run.py data --download
python run.py data --convert

# 3. 一键跑通：训练 + 回测 + 图表 + 选股 + 风险诊断
python run.py full

# 4. 每日增量更新
python run.py update
```

## 核心能力

| 模块 | 功能 | 命令 |
|------|------|------|
| 数据管理 | AKShare 下载 / CSV 转 Qlib bin / 增量更新 | `python run.py data` / `python run.py update` |
| 模型训练 | Qlib LGBModel / RankLGBModel(LambdaRank) + Alpha158/360 | `python run.py train` |
| 回测分析 | TopkDropoutStrategy + 图表生成 + 风格暴露诊断 | `python run.py backtest --rid <ID>` |
| 选股推荐 | Top-K 股票推荐 + 历史验证 | `python run.py pick` / `python run.py validate-picks` |
| 稳健性验证 | Walk-Forward / IC 稳定性 / 特征漂移 | `python run.py rolling` / `python run.py ic-stability` / `python run.py drift` |
| 鲁棒性深化 | TSCV / 市场阶段分析 / 敏感性分析 / 关键年份回测 | `python run.py tscv` / `python run.py regime` / `python run.py sensitivity` / `python run.py key-years` |
| 模型能力恢复 | Optuna 超参搜索 / SHAP 可解释性 | `python run.py optuna` / `python run.py explain` |
| 风险诊断 | 风格暴露 / 收益归因 / 容量检查 | `python run.py risk` / `python run.py attribution` / `python run.py capacity` |
| 对比实验 | 第 8.4 节强制性对比实验（E1-E7） | `python run.py benchmark` |

## 项目结构

```
qlib-stock/
├── run.py                       # 统一入口（所有命令）
├── config/
│   └── settings.yaml            # 全局配置（数据路径、模型参数、策略等）
├── qlib_pipeline/
│   ├── workflow_config.yaml     # Qlib 工作流配置
│   ├── train.py                 # 训练管线
│   ├── dataset.py               # 配置加载器
│   ├── backtest.py              # 回测分析
│   ├── rolling.py               # Walk-Forward 验证
│   ├── drift.py                 # 特征漂移检测
│   ├── ic_stability.py          # IC 稳定性分析
│   ├── tscv.py                  # 时序交叉验证
│   ├── regime.py                # 市场阶段分析
│   ├── sensitivity.py           # 超参数敏感性
│   ├── validate.py              # 选股验证
│   ├── model.py                 # 模型构建
│   └── numpy_compat.py          # NumPy 兼容补丁
├── research/                    # 研究分析模块
│   ├── significance.py          # 统计显著性检验（Newey-West）
│   ├── risk_model.py            # 风险模型（Barra 风格因子）
│   ├── attribution.py           # 收益归因（Fama-MacBeth 截面回归）
│   ├── portfolio_constructor.py # 行业中性化策略
│   ├── capacity.py              # 策略容量分析
│   ├── experiment_tracker.py    # 实验追踪（探索/确认协议）
│   ├── explain.py               # SHAP 可解释性
│   └── benchmark.py             # 第 8.4 节强制性对比实验（E1-E7）
├── tuning/                      # 超参数搜索
│   └── optuna_search.py         # Optuna 贝叶斯优化
├── data_center/                 # 数据层
│   ├── daily_update.py          # 每日增量更新
│   ├── data_store.py            # 统一数据存储接口
│   ├── csv_loader.py            # CSV 数据加载器
│   ├── akshare_client.py        # AKShare API 客户端
│   ├── download_history.py      # 历史数据下载
│   ├── duckdb_store.py          # DuckDB 存储后端
│   └── data_validator.py        # 数据校验
├── docs/                        # 文档
│   ├── user-guide.md            # 运行说明
│   ├── configuration.md         # 配置参考
│   └── technical-guide.md       # 技术架构
└── output/                      # 输出目录
    ├── qlib_charts/             # 回测图表
    ├── picks/                   # 选股推荐 CSV
    ├── drift/                   # 漂移检测报告
    ├── ic_stability/            # IC 稳定性报告
    ├── regime/                  # 市场阶段分析
    ├── rolling/                 # 滚动验证
    ├── tscv/                    # 交叉验证
    ├── sensitivity/             # 敏感性分析
    ├── key_years/               # 关键年份回测
    ├── optuna/                  # 超参搜索结果
    ├── shap/                    # SHAP 分析报告
    ├── benchmark/               # 对比实验报告
    └── validation/              # 选股验证
```

## 文档索引

- [运行说明](docs/user-guide.md) — 所有命令的使用方法和参数说明（v4.0）
- [配置参考](docs/configuration.md) — settings.yaml 和 workflow_config.yaml 详细说明（v4.0）
- [技术架构](docs/technical-guide.md) — 系统设计、数据流、管线说明、调参层设计、回测结果（v4.0）

## 技术栈

Python 3.10 · Qlib 0.9.7 · LightGBM (LambdaRank) · Pandas · NumPy 1.x · Plotly · AKShare · Optuna