# 运行说明

> 文档版本: v3.0  
> 更新日期: 2026-07-06  
> 适用项目: `d:/project/qlib-stock`

---

## 目录

1. [环境准备](#1-环境准备)
2. [数据管理](#2-数据管理)
3. [命令参考](#3-命令参考)
4. [常用工作流](#4-常用工作流)
5. [常见问题](#5-常见问题)

---

## 1. 环境准备

### 1.1 Python 版本

Python 3.10，NumPy 必须为 1.x（Qlib 0.9.7 不兼容 NumPy 2.x）。

### 1.2 安装依赖

```bash
pip install -r requirements.txt
```

### 1.3 关键依赖

| 包 | 说明 |
|---|---|
| pyqlib | 核心量化框架 |
| lightgbm | 梯度提升模型 |
| numpy | **必须 1.x**，项目已通过 `qlib_pipeline/numpy_compat.py` 补丁处理兼容性 |
| plotly + kaleido | 图表渲染与静态导出 |
| pandas | 数据处理 |
| akshare | A 股数据下载 |

### 1.4 Windows 配置

项目已在各训练模块中强制单线程模式，避免 Windows 多进程崩溃：

```python
os.environ["NUMEXPR_MAX_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
```

---

## 2. 数据管理

### 2.1 数据目录

| 路径 | 用途 | 配置项 |
|------|------|--------|
| `D:/data/` | CSV 原始数据（AKShare 下载） | `data_source.csv_dir` |
| `D:/trae/qlib_bin/` | Qlib bin 格式数据 | `data_source.qlib_dir` |

### 2.2 下载数据

```bash
# 全量下载 AKShare 历史数据
python run.py data --download

# 小样本测试（100 只股票）
python run.py data --download --sample 100
```

### 2.3 转换数据

```bash
# 全量转换 CSV → Qlib bin（日线）
python run.py data --convert

# 小样本快速测试
python run.py data --convert --sample 100

# 转换周线 / 月线
python run.py data --convert --freq week
python run.py data --convert --freq month

# 检查数据完整性
python run.py data --check
```

### 2.4 每日增量更新

```bash
# 下载最近 3 天数据并更新 bin
python run.py update

# 下载最近 7 天
python run.py update --days 7

# 只更新 parquet 不更新 bin
python run.py update --skip-bin
```

增量更新从 `D:/trae/qlib_bin/features/` 目录自动发现股票列表，无需依赖 CSV 文件。`--days` 参数设置覆盖天数（默认 3 天），用于应对复权修正。

---

## 3. 命令参考

`run.py` 是唯一命令行入口，所有命令通过子命令调用。

### 3.1 核心流程

```bash
python run.py full                          # 一键跑通：训练+回测+图表+选股
python run.py train                         # 仅训练模型
python run.py backtest --rid <recorder_id>  # 仅回测
python run.py pick --rid <recorder_id>      # 选股推荐
```

#### full 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--handler` | 特征处理器 (`Alpha158` / `Alpha360`) | 配置决定 |
| `--loss` | 损失函数 (`mse` / `rank`) | 配置决定 |
| `--topk` | 持仓股票数 | 配置决定 |
| `--pick-topk` | 选股推荐数量 | `settings.yaml: strategy.top_k` |
| `--output-dir` | 图表输出目录 | `settings.yaml: output.charts` |
| `--config` | 配置文件路径 | `qlib_pipeline/workflow_config.yaml` |

#### backtest 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--rid` | 训练记录 ID（必填） | — |
| `--output-dir` | 图表输出目录 | `settings.yaml: output.charts` |
| `--pick-topk` | 选股推荐数量 | `settings.yaml: strategy.top_k` |

#### pick 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--rid` | 回测记录 ID（必填） | — |
| `--topk` | 推荐数量 | `settings.yaml: strategy.top_k` |
| `--date` | 指定日期 YYYY-MM-DD | 最新日期 |
| `--output-dir` | 输出目录 | `settings.yaml: output.picks` |

### 3.2 数据管理

```bash
python run.py data --download              # 下载数据
python run.py data --convert               # CSV → Qlib bin
python run.py data --convert --sample 100  # 小样本转换
python run.py data --convert --freq week   # 周线转换
python run.py data --check                 # 数据完整性检查
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--csv-dir` | CSV 数据目录 | `settings.yaml: data_source.csv_dir` |
| `--qlib-dir` | Qlib bin 目录 | `settings.yaml: data_source.qlib_dir` |
| `--freq` | 数据频率 (`day` / `week` / `month`) | `day` |
| `--sample` | 采样股票数量 | 全部 |

### 3.3 每日增量更新

```bash
python run.py update
python run.py update --days 7
python run.py update --skip-bin
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--days` / `-d` | 下载最近 N 天 | `3` |
| `--skip-bin` | 跳过 bin 更新 | `false` |
| `--workers` / `-w` | 并发线程数 | `settings.yaml: data_source.max_workers` |

### 3.4 稳健性验证（Phase 1）

```bash
python run.py rolling --n-folds 6          # Walk-Forward 滚动验证
python run.py drift                         # 特征漂移检测 (PSI)
python run.py ic-stability                  # IC 稳定性分析 (ICIR/衰减/分层)
```

| 命令 | 参数 | 说明 |
|------|------|------|
| `rolling` | `--n-folds` 折数 / `--window-years` 训练窗口年数 / `--step-months` 步长月数 | 滚动窗口训练验证 |
| `drift` | `--output-dir` | PSI 特征漂移 + 概念漂移检测 |
| `ic-stability` | `--output-dir` | ICIR、IC 衰减曲线、分层 IC |

### 3.5 鲁棒性深化（Phase 3）

```bash
python run.py tscv --n-splits 5             # Purged K-Fold 时序交叉验证
python run.py regime                        # 市场阶段稳定性分析
python run.py sensitivity                   # 超参数敏感性分析
python run.py key-years --years 2020,2022   # 关键年份独立回测
```

| 命令 | 参数 | 说明 |
|------|------|------|
| `tscv` | `--n-splits` 折数 / `--purge-days` 清除天数 / `--embargo-days` 禁运天数 | 防数据泄露的时序交叉验证 |
| `regime` | `--output-dir` | 牛市/熊市/震荡市分阶段 IC 分析 |
| `sensitivity` | `--param` 参数名 `--values` 逗号分隔值 | 单参数或全参数扫描 |
| `key-years` | `--years` 逗号分隔年份 | 指定年份独立回测 |

### 3.6 选股验证

```bash
python run.py validate-picks --picks-dir output/picks --lookback-days 20
```

读取历史选股推荐 CSV，用真实价格数据验证推荐股票的未来表现，输出命中率和各持有期收益。

---

## 4. 常用工作流

### 4.1 首次使用

```bash
python run.py data --download     # 下载全量数据
python run.py data --convert      # 转换为 Qlib bin 格式
python run.py full                # 训练 + 回测 + 选股
```

### 4.2 日常更新

```bash
python run.py update              # 增量更新数据
python run.py full                # 重新训练和选股
```

### 4.3 快速实验

```bash
python run.py data --convert --sample 100   # 100 只股票快速测试
python run.py full --handler Alpha360 --loss rank --topk 25
```

### 4.4 稳健性全面检查

```bash
python run.py rolling --n-folds 6
python run.py drift
python run.py ic-stability
python run.py tscv --n-splits 5
python run.py regime
python run.py sensitivity
```

### 4.5 参数覆盖

所有配置项可通过命令行参数临时覆盖，不修改配置文件：

```bash
python run.py full --handler Alpha360 --loss rank --topk 30 --output-dir output/my_exp
python run.py full --config qlib_pipeline/workflow_config.yaml
```

---

## 5. 常见问题

### 5.1 回测报 benchmark 不存在

**现象**：`ValueError: The benchmark ['SH000300'] does not exist`

**原因**：benchmark 必须使用 features 目录中实际存在的股票代码。

**修复**：在 `workflow_config.yaml` 中将 benchmark 设置为 `"SH600000"`（浦发银行）。

### 5.2 回测日期越界

**现象**：`IndexError: index X is out of bounds`

**原因**：回测 `end_time` 超出日历最后交易日。

**修复**：调整 `workflow_config.yaml` 中 `backtest.backtest.end_time` 到实际最后交易日之前。

### 5.3 训练过拟合

**现象**：train loss 持续下降但 valid loss 不再改善。

**应对**：
- 增大 `lambda_l1` / `lambda_l2`（正则化）
- 降低 `max_depth` / `num_leaves`
- 减小 `learning_rate`

### 5.4 数据下载中断

**现象**：AKShare 网络连接超时或中断。

**应对**：
- 重试 `python run.py update --days 7`
- 检查网络代理/VPN 设置
- 降低并发数：修改 `settings.yaml` 中 `data_source.max_workers`

### 5.5 instruments 文件损坏

**现象**：`cannot reshape array of size 1 into shape (0,)` 或 `Too many columns specified`

**原因**：`D:/trae/qlib_bin/instruments/` 下文件可能被写入二进制数据损坏。

**修复**：重新运行 `python run.py data --convert` 重建 instruments 文件。

---

## 命令速查

```bash
# 首次使用
python run.py data --download && python run.py data --convert && python run.py full

# 日常迭代
python run.py update && python run.py full

# 参数覆盖
python run.py full --handler Alpha360 --loss rank --topk 25

# 稳健性验证
python run.py rolling --n-folds 6
python run.py drift
python run.py ic-stability
python run.py tscv --n-splits 5
python run.py regime
python run.py sensitivity
python run.py key-years
python run.py validate-picks
```