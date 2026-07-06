# A股量化选股系统 — Qlib Pipeline

基于 Microsoft Qlib 的 A 股量化选股系统，支持特征工程、多模型训练、回测分析、选股推荐和稳健性验证。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载数据并转换
python run.py data --download
python run.py data --convert

# 3. 一键跑通：训练 + 回测 + 图表 + 选股
python run.py full

# 4. 每日增量更新
python run.py update
```

## 核心能力

| 模块 | 功能 | 命令 |
|------|------|------|
| 数据管理 | AKShare 下载 / CSV 转 Qlib bin / 增量更新 | `python run.py data` / `python run.py update` |
| 模型训练 | LightGBM / XGBoost / CatBoost + Alpha158/360 | `python run.py train` |
| 回测分析 | TopkDropoutStrategy + 图表生成 | `python run.py backtest --rid <ID>` |
| 选股推荐 | Top-K 股票推荐 + 历史验证 | `python run.py pick` / `python run.py validate-picks` |
| 稳健性验证 | Walk-Forward / IC 稳定性 / 特征漂移 | `python run.py rolling` / `python run.py ic-stability` / `python run.py drift` |
| 鲁棒性深化 | TSCV / 市场阶段分析 / 敏感性分析 | `python run.py tscv` / `python run.py regime` / `python run.py sensitivity` |

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
│   └── sensitivity.py           # 超参数敏感性
├── data_center/                 # 数据层
│   ├── daily_update.py          # 每日增量更新
│   ├── data_store.py            # 统一数据存储接口
│   └── csv_loader.py            # CSV 数据加载器
├── model/                       # 自研模型
│   ├── lgb_model.py             # LightGBM
│   ├── xgb_model.py             # XGBoost
│   ├── cat_model.py             # CatBoost
│   └── ensemble.py              # 模型集成
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
    └── validation/              # 选股验证
```

## 文档索引

- [运行说明](docs/user-guide.md) — 所有命令的使用方法和参数说明
- [配置参考](docs/configuration.md) — settings.yaml 和 workflow_config.yaml 详细说明
- [技术架构](docs/technical-guide.md) — 系统设计、数据流、管线说明

## 技术栈

Python 3.10 · Qlib 0.9.7 · LightGBM · XGBoost · CatBoost · Plotly · Pandas · NumPy 1.x