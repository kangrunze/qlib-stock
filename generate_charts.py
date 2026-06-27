# -*- coding: utf-8 -*-
"""
量化分析图表生成脚本

使用项目自身 evaluate 模块（非 qlib 内置 analysis），
基于真实 CSV 数据生成 10 张分析图表。

Usage:
    python generate_charts.py --sample 50
    python generate_charts.py --sample 20
"""

import argparse
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---- 项目根目录加入 sys.path ----
_PROJECT_ROOT = Path(__file__).parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

warnings.filterwarnings("ignore")

# ---- matplotlib 后端与中文字体 ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.stats import norm, skew as scipy_skew

# 尝试设置中文字体
_CHINESE_FONTS = ["SimHei", "Microsoft YaHei", "PingFang SC", "WenQuanYi Micro Hei"]
_font_set = False
for _fn in _CHINESE_FONTS:
    try:
        matplotlib.rcParams["font.sans-serif"] = [_fn] + matplotlib.rcParams.get("font.sans-serif", [])
        matplotlib.rcParams["axes.unicode_minus"] = False
        _font_set = True
        break
    except Exception:
        continue
if not _font_set:
    matplotlib.rcParams["axes.unicode_minus"] = False

# ---- 深色主题 ----
matplotlib.rcParams.update({
    "figure.facecolor": "#0f172a",
    "axes.facecolor": "#1e293b",
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#e2e8f0",
    "text.color": "#e2e8f0",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "grid.color": "#334155",
    "grid.alpha": 0.4,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
})

# ---- 日志 ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_charts")

# ============================================================
# 导入项目模块
# ============================================================
from data_center.data_store import DataStore
from feature_engine.technical import TechnicalFactorCalculator
from feature_engine.cross_section import CrossSectionCalculator
from dataset.label_generator import LabelGenerator
from model.lgb_model import LightGBMModel
from evaluate.ic_analysis import ICAnalyzer
from evaluate.drawdown_analysis import DrawdownAnalyzer
from evaluate.sharpe_analysis import SharpeAnalyzer

# ============================================================
# 常量
# ============================================================
OUTPUT_DIR = _PROJECT_ROOT / "output" / "qlib_charts"
DPI = 150


# ============================================================
# 辅助函数
# ============================================================
def setup_output_dir():
    """创建输出目录。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("输出目录: %s", OUTPUT_DIR)


def save_fig(fig, filename: str):
    """保存图表到输出目录。"""
    path = OUTPUT_DIR / filename
    fig.savefig(str(path), dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("图表已保存: %s", path)


def build_dataset(data_dict, sample_size):
    """
    完整的流水线：
      1. 加载 CSV
      2. 计算技术因子
      3. 计算横截面因子
      4. 生成标签
      5. 合并为可用于训练的 DataFrame
    Returns:
      feature_df: pd.DataFrame  (index=date, columns=factor names)
      label_series: pd.Series   (index=date, values=forward return)
      stock_info: pd.DataFrame  (columns: date, code, close, pct_chg)
      tech_factor_dict: dict of per-stock factor DataFrames
      pred_dates: np.ndarray
      pred_returns: np.ndarray
      pred_preds: np.ndarray (model predictions)
      model: LightGBMModel (trained)
    """
    t0 = time.time()

    # ---- 1. 加载数据 ----
    logger.info("=" * 60)
    logger.info("步骤 1/%d: 加载数据 (sample=%d)", 5, sample_size)
    # DataStore 根据 settings.yaml 中 data_source.data_format 自动选择后端
    store = DataStore()
    data_dict = store.load_all(sample_size=sample_size)
    if not data_dict:
        raise RuntimeError("未加载到任何数据，请检查 settings.yaml 中的数据目录配置")
    codes = list(data_dict.keys())
    logger.info("成功加载 %d 只股票: %s ...", len(codes), codes[:5])

    # ---- 2. 计算技术因子 ----
    logger.info("=" * 60)
    logger.info("步骤 2/%d: 计算技术因子", 5)
    tech_calc = TechnicalFactorCalculator()
    tech_factor_dict = {}
    for i, (code, df) in enumerate(data_dict.items()):
        try:
            df_sorted = df.sort_values("date").set_index("date")
            factors = tech_calc.compute_all(df_sorted)
            tech_factor_dict[code] = factors
        except Exception as e:
            logger.warning("股票 %s 因子计算失败: %s", code, str(e)[:80])
        if (i + 1) % 10 == 0:
            logger.info("  已完成 %d/%d", i + 1, len(data_dict))
    logger.info("技术因子计算完成: %d 只股票", len(tech_factor_dict))

    # ---- 3. 计算横截面因子（耗时较长，仅在样本<=30时计算） ----
    logger.info("=" * 60)
    logger.info("步骤 3/%d: 计算横截面因子", 5)
    cs_factor_dict = {}
    if sample_size <= 30:
        try:
            cs_calc = CrossSectionCalculator()
            cs_factor_dict = cs_calc.compute_all(data_dict, verbose=False)
            logger.info("横截面因子计算完成: %d 只股票", len(cs_factor_dict))
        except Exception as e:
            logger.warning("横截面因子计算失败（将跳过）: %s", e)
    else:
        logger.info("样本量 %d > 30，跳过横截面因子以节省时间", sample_size)

    # ---- 4. 生成标签 ----
    logger.info("=" * 60)
    logger.info("步骤 4/%d: 生成标签", 5)
    label_dict = {}
    label_gen = LabelGenerator(benchmark_code="000905")
    try:
        label_dict = label_gen.generate_all_labels(data_dict)
    except Exception as e:
        logger.warning("标签生成失败（使用简易标签）: %s", e)

    # 如果标签为空，使用简易标签：未来 20 日收益率
    if not label_dict:
        logger.info("使用简易标签 (未来 20 日收益率)...")
        for code, df in data_dict.items():
            try:
                df2 = df.sort_values("date").copy()
                df2["label_ret_20d"] = df2["close"].shift(-20) / df2["close"] - 1
                df2 = df2[["date", "code", "label_ret_20d"]].dropna(subset=["label_ret_20d"])
                if len(df2) > 0:
                    label_dict[code] = df2
            except Exception:
                pass

    logger.info("标签生成完成: %d 只股票", len(label_dict))

    # ---- 5. 合并为训练数据集 ----
    logger.info("=" * 60)
    logger.info("步骤 5/%d: 合并数据集并训练模型", 5)
    all_rows = []
    stock_info_rows = []

    for code in data_dict:
        if code not in tech_factor_dict:
            continue
        df_orig = data_dict[code]
        df_tech = tech_factor_dict[code]

        # 获取标签
        if code in label_dict:
            lbl_df = label_dict[code].copy()
            lbl_df["date"] = pd.to_datetime(lbl_df["date"]).dt.strftime("%Y-%m-%d")
            # 找 label_ret_20d 列
            lbl_cols = [c for c in lbl_df.columns if c.startswith("label_")]
            if not lbl_cols:
                continue
            lbl_col = lbl_cols[0]
        else:
            continue

        # 合并技术因子 + 标签
        common_dates = set(df_tech.index) & set(lbl_df["date"].values)
        if not common_dates:
            continue

        for d in sorted(common_dates):
            if d not in df_tech.index:
                continue
            row = df_tech.loc[d].to_dict()
            lbl_row = lbl_df[lbl_df["date"] == d]
            if lbl_row.empty:
                continue
            row["label"] = lbl_row.iloc[0][lbl_col]
            row["date"] = d
            row["code"] = code
            # close
            orig_row = df_orig[df_orig["date"] == d]
            if not orig_row.empty:
                row["close"] = orig_row.iloc[0]["close"]
                pct = orig_row.iloc[0].get("pct_chg", np.nan)
                if pd.isna(pct):
                    pct = orig_row.iloc[0]["close"].pct_change() * 100 if len(orig_row) > 1 else 0
                stock_info_rows.append({
                    "date": d,
                    "code": code,
                    "close": orig_row.iloc[0]["close"],
                    "pct_chg": pct,
                    "label": row["label"],
                })
            all_rows.append(row)

    if not all_rows:
        raise RuntimeError("合并后数据为空，无法训练模型")

    merged_df = pd.DataFrame(all_rows)
    merged_df["date"] = pd.to_datetime(merged_df["date"])
    merged_df = merged_df.sort_values("date").reset_index(drop=True)
    stock_info_df = pd.DataFrame(stock_info_rows)
    stock_info_df["date"] = pd.to_datetime(stock_info_df["date"])

    # 提取特征列（排除元数据列）
    exclude_cols = {"date", "code", "close", "pct_chg", "label"}
    feature_cols = [c for c in merged_df.columns if c not in exclude_cols]
    logger.info("合并数据集: %d 行 x %d 特征", len(merged_df), len(feature_cols))

    # 时间划分 (80% train, 20% test)
    cutoff = merged_df["date"].quantile(0.8)
    train_df = merged_df[merged_df["date"] <= cutoff].copy()
    test_df = merged_df[merged_df["date"] > cutoff].copy()

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values
    test_dates = test_df["date"].values
    test_codes = test_df["code"].values
    test_closes = test_df["close"].values

    # 用 NaN 填充
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
    y_test = np.nan_to_num(y_test, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info("训练集: %d, 测试集: %d", len(X_train), len(X_test))

    # ---- 训练 LightGBM ----
    logger.info("训练 LightGBM 模型...")
    model = LightGBMModel(params={
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 64,
        "max_depth": 8,
        "verbose": -1,
    })
    model.fit(X_train, y_train)

    # 预测
    preds = model.predict(X_test)
    logger.info("预测完成: mean=%.6f, std=%.6f", preds.mean(), preds.std())

    elapsed = time.time() - t0
    logger.info("数据准备与模型训练完成，耗时 %.1f 秒", elapsed)

    return {
        "feature_cols": feature_cols,
        "merged_df": merged_df,
        "stock_info_df": stock_info_df,
        "test_dates": test_dates,
        "test_codes": test_codes,
        "test_closes": test_closes,
        "test_returns": y_test,
        "test_preds": preds,
        "model": model,
        "tech_factor_dict": tech_factor_dict,
        "data_dict": data_dict,
    }


# ============================================================
# Chart 01: Factor IC Analysis (Top 20)
# ============================================================
def chart_01_feature_ic(ctx):
    """因子 IC 分析柱状图（Top 20）。"""
    name = "01_feature_ic.png"
    logger.info("[%s] 生成因子 IC 分析图...", name)
    try:
        feature_cols = ctx["feature_cols"]
        test_returns = ctx["test_returns"]
        test_preds = ctx["test_preds"]
        test_dates = ctx["test_dates"]
        X_test_merged = ctx["merged_df"][ctx["merged_df"]["date"] > ctx["merged_df"]["date"].quantile(0.8)]

        # 对每个因子计算与 label 的 IC
        ic_values = {}
        unique_dates = np.unique(test_dates)
        for col in feature_cols:
            vals = X_test_merged[col].values
            valid = ~(np.isnan(vals) | np.isnan(test_returns) | np.isinf(vals) | np.isinf(test_returns))
            if valid.sum() < 20:
                continue
            # 横截面 IC per date
            daily_ics = []
            for d in unique_dates:
                mask = (test_dates == d) & valid
                if mask.sum() < 5:
                    continue
                v = vals[mask]
                r = test_returns[mask]
                c = np.corrcoef(v, r)
                ic = c[0, 1] if np.isfinite(c[0, 1]) else 0.0
                daily_ics.append(ic)
            if daily_ics:
                ic_values[col] = np.mean(daily_ics)

        if not ic_values:
            logger.warning("[%s] 无法计算 IC，跳过", name)
            return None

        # Top 20
        ic_series = pd.Series(ic_values).sort_values(ascending=False)
        top20 = ic_series.head(20)

        fig, ax = plt.subplots(figsize=(14, 8))
        colors = ["#22d3ee" if v >= 0 else "#f87171" for v in top20.values]
        bars = ax.barh(range(len(top20)), top20.values, color=colors, edgecolor="none", height=0.7)
        ax.set_yticks(range(len(top20)))
        ax.set_yticklabels(top20.index, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("IC 值")
        ax.set_title("因子 IC 分析 - Top 20 因子", fontsize=16, fontweight="bold", pad=15)
        ax.axvline(x=0, color="#64748b", linestyle="--", linewidth=0.8)
        # 数值标注
        for bar, val in zip(bars, top20.values):
            x_pos = val + 0.002 if val >= 0 else val - 0.002
            ha = "left" if val >= 0 else "right"
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f"{val:.4f}",
                    va="center", ha=ha, fontsize=8, color="#cbd5e1")
        ax.grid(axis="x", alpha=0.3)
        save_fig(fig, name)
        return name
    except Exception as e:
        logger.error("[%s] 生成失败: %s", name, e)
        return None


# ============================================================
# Chart 02: IC Time Series
# ============================================================
def chart_02_ic_series(ctx):
    """IC 时间序列折线图。"""
    name = "02_ic_series.png"
    logger.info("[%s] 生成 IC 时间序列图...", name)
    try:
        analyzer = ICAnalyzer()
        ic_df = analyzer.compute_ic(
            predictions=ctx["test_preds"],
            returns=ctx["test_returns"],
            dates=ctx["test_dates"],
        )
        if ic_df is None or ic_df.empty:
            logger.warning("[%s] IC 数据为空，跳过", name)
            return None

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(ic_df["date"], ic_df["IC"], alpha=0.35, color="#38bdf8", linewidth=0.8, label="日度 IC")
        if len(ic_df) >= 20:
            rolling = ic_df["IC"].rolling(window=20).mean()
            ax.plot(ic_df["date"], rolling, color="#facc15", linewidth=2, label="20日滚动均值")
        ax.axhline(y=0, color="#64748b", linestyle="--", linewidth=0.8)
        mean_ic = ic_df["IC"].mean()
        ax.axhline(y=mean_ic, color="#f87171", linestyle="--", linewidth=1, alpha=0.7,
                    label=f"均值 IC: {mean_ic:.4f}")
        ax.fill_between(ic_df["date"], 0, ic_df["IC"],
                         where=ic_df["IC"] > 0, color="#22c55e", alpha=0.15)
        ax.fill_between(ic_df["date"], 0, ic_df["IC"],
                         where=ic_df["IC"] <= 0, color="#ef4444", alpha=0.15)
        ax.set_xlabel("日期")
        ax.set_ylabel("IC 值")
        ax.set_title("IC 时间序列", fontsize=16, fontweight="bold", pad=15)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(alpha=0.3)
        save_fig(fig, name)
        return name
    except Exception as e:
        logger.error("[%s] 生成失败: %s", name, e)
        return None


# ============================================================
# Chart 03: Model Performance (Predicted vs Actual)
# ============================================================
def chart_03_model_performance(ctx):
    """模型表现散点图：预测值 vs 实际收益。"""
    name = "03_model_performance.png"
    logger.info("[%s] 生成模型表现散点图...", name)
    try:
        preds = ctx["test_preds"]
        returns = ctx["test_returns"]
        # 采样避免过多点
        n = len(preds)
        if n > 5000:
            idx = np.random.choice(n, 5000, replace=False)
            preds = preds[idx]
            returns = returns[idx]

        corr = np.corrcoef(preds, returns)[0, 1]

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.scatter(preds, returns, alpha=0.15, s=8, c="#38bdf8", edgecolors="none")

        # 回归线
        z = np.polyfit(preds, returns, 1)
        p = np.poly1d(z)
        x_line = np.linspace(preds.min(), preds.max(), 100)
        ax.plot(x_line, p(x_line), color="#f87171", linewidth=2.5, label="回归线")

        ax.axhline(y=0, color="#64748b", linestyle="--", linewidth=0.8)
        ax.axvline(x=0, color="#64748b", linestyle="--", linewidth=0.8)

        # 注释
        textstr = f"相关系数: {corr:.4f}\n样本量: {len(preds)}"
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment="top", color="#facc15",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#1e293b", edgecolor="#475569", alpha=0.9))

        ax.set_xlabel("预测得分")
        ax.set_ylabel("实际收益率")
        ax.set_title("模型表现 - 预测值 vs 实际收益", fontsize=16, fontweight="bold", pad=15)
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(alpha=0.3)
        save_fig(fig, name)
        return name
    except Exception as e:
        logger.error("[%s] 生成失败: %s", name, e)
        return None


# ============================================================
# Chart 04: Equity Curve Simulation
# ============================================================
def chart_04_equity_curve(ctx):
    """等权 Top-K 投资组合 vs 等权基准的净值曲线。"""
    name = "04_equity_curve.png"
    logger.info("[%s] 生成净值曲线图...", name)
    try:
        dates = ctx["test_dates"]
        preds = ctx["test_preds"]
        returns = ctx["test_returns"]

        unique_dates = sorted(np.unique(dates))
        if len(unique_dates) < 5:
            logger.warning("[%s] 交易日不足，跳过", name)
            return None

        # 每日组合收益
        strategy_daily = []
        benchmark_daily = []
        for d in unique_dates:
            mask = dates == d
            if mask.sum() < 3:
                strategy_daily.append(0.0)
                benchmark_daily.append(0.0)
                continue
            d_preds = preds[mask]
            d_rets = returns[mask]
            # 等权基准
            bench_ret = d_rets.mean()
            benchmark_daily.append(bench_ret)
            # Top-K 策略 (选预测最高的 1/3)
            k = max(3, mask.sum() // 3)
            top_idx = np.argsort(d_preds)[-k:]
            strat_ret = d_rets[top_idx].mean()
            strategy_daily.append(strat_ret)

        strategy_daily = np.array(strategy_daily)
        benchmark_daily = np.array(benchmark_daily)

        # 复合净值
        strategy_nav = np.cumprod(1 + strategy_daily)
        benchmark_nav = np.cumprod(1 + benchmark_daily)

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot(unique_dates, strategy_nav, color="#22d3ee", linewidth=2, label="Top-K 等权策略")
        ax.plot(unique_dates, benchmark_nav, color="#a78bfa", linewidth=2, label="等权基准", linestyle="--")
        ax.fill_between(unique_dates, benchmark_nav, strategy_nav,
                         where=strategy_nav >= benchmark_nav, color="#22c55e", alpha=0.15)
        ax.fill_between(unique_dates, benchmark_nav, strategy_nav,
                         where=strategy_nav < benchmark_nav, color="#ef4444", alpha=0.15)
        ax.set_xlabel("日期")
        ax.set_ylabel("累计净值")
        ax.set_title("净值曲线模拟 - Top-K 策略 vs 等权基准", fontsize=16, fontweight="bold", pad=15)
        ax.legend(loc="upper left", fontsize=11)
        ax.grid(alpha=0.3)

        # 最终收益率注释
        strat_total = strategy_nav[-1] - 1
        bench_total = benchmark_nav[-1] - 1
        ax.text(0.02, 0.02, f"策略累计: {strat_total:.2%}\n基准累计: {bench_total:.2%}",
                transform=ax.transAxes, fontsize=11, color="#facc15", va="bottom",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#1e293b", edgecolor="#475569", alpha=0.9))
        save_fig(fig, name)
        return name
    except Exception as e:
        logger.error("[%s] 生成失败: %s", name, e)
        return None


# ============================================================
# Chart 05: Drawdown Analysis
# ============================================================
def chart_05_drawdown(ctx):
    """最大回撤分析图。"""
    name = "05_drawdown.png"
    logger.info("[%s] 生成回撤分析图...", name)
    try:
        dates = ctx["test_dates"]
        preds = ctx["test_preds"]
        returns = ctx["test_returns"]

        unique_dates = sorted(np.unique(dates))
        # 构建净值序列
        strategy_daily = []
        for d in unique_dates:
            mask = dates == d
            if mask.sum() < 3:
                strategy_daily.append(0.0)
                continue
            d_preds = preds[mask]
            d_rets = returns[mask]
            k = max(3, mask.sum() // 3)
            top_idx = np.argsort(d_preds)[-k:]
            strategy_daily.append(d_rets[top_idx].mean())

        strategy_daily = np.array(strategy_daily)
        equity = np.cumprod(1 + strategy_daily)

        analyzer = DrawdownAnalyzer()
        dd_df = analyzer.compute_drawdowns(equity, dates=unique_dates)

        fig, ax = plt.subplots(figsize=(14, 6))
        x = dd_df["date"].values if "date" in dd_df.columns else range(len(dd_df))
        dd_pct = dd_df["drawdown"].values * 100

        ax.fill_between(x, 0, dd_pct, color="#ef4444", alpha=0.5)
        ax.plot(x, dd_pct, color="#f87171", linewidth=1)
        ax.axhline(y=0, color="#64748b", linestyle="-", linewidth=0.8)

        max_dd = dd_pct.min()
        ax.set_xlabel("日期")
        ax.set_ylabel("回撤 (%)")
        ax.set_title(f"回撤分析 (最大回撤: {max_dd:.2f}%)", fontsize=16, fontweight="bold", pad=15)
        ax.grid(alpha=0.3)
        save_fig(fig, name)
        return name
    except Exception as e:
        logger.error("[%s] 生成失败: %s", name, e)
        return None


# ============================================================
# Chart 06: Return Distribution
# ============================================================
def chart_06_return_distribution(ctx):
    """日收益率分布直方图。"""
    name = "06_return_distribution.png"
    logger.info("[%s] 生成收益率分布图...", name)
    try:
        dates = ctx["test_dates"]
        returns = ctx["test_returns"]

        unique_dates = sorted(np.unique(dates))
        daily_rets = []
        for d in unique_dates:
            mask = dates == d
            if mask.sum() > 0:
                daily_rets.append(returns[mask].mean())
        daily_rets = np.array(daily_rets)
        daily_rets = daily_rets[np.isfinite(daily_rets)]

        if len(daily_rets) < 10:
            logger.warning("[%s] 数据不足，跳过", name)
            return None

        fig, ax = plt.subplots(figsize=(12, 8))
        n, bins, patches = ax.hist(daily_rets, bins=60, density=True, color="#38bdf8",
                                     alpha=0.7, edgecolor="none")

        # 正态拟合
        mu, std = daily_rets.mean(), daily_rets.std()
        x = np.linspace(daily_rets.min(), daily_rets.max(), 200)
        ax.plot(x, norm.pdf(x, mu, std), color="#facc15", linewidth=2.5, label=f"正态拟合 (mu={mu:.4f}, sigma={std:.4f})")

        sk = scipy_skew(daily_rets)
        textstr = f"均值: {mu:.4f}\n标准差: {std:.4f}\n偏度: {sk:.4f}\n样本量: {len(daily_rets)}"
        ax.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment="top", horizontalalignment="right", color="#facc15",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#1e293b", edgecolor="#475569", alpha=0.9))

        ax.axvline(x=mu, color="#f87171", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("日收益率")
        ax.set_ylabel("概率密度")
        ax.set_title("日收益率分布", fontsize=16, fontweight="bold", pad=15)
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(alpha=0.3)
        save_fig(fig, name)
        return name
    except Exception as e:
        logger.error("[%s] 生成失败: %s", name, e)
        return None


# ============================================================
# Chart 07: Feature Importance
# ============================================================
def chart_07_feature_importance(ctx):
    """特征重要性水平柱状图 (Top 30)。"""
    name = "07_feature_importance.png"
    logger.info("[%s] 生成特征重要性图...", name)
    try:
        model = ctx["model"]
        imp_df = model.get_feature_importance()
        if imp_df is None or imp_df.empty:
            logger.warning("[%s] 特征重要性为空，跳过", name)
            return None

        top30 = imp_df.head(30).copy()

        fig, ax = plt.subplots(figsize=(14, 10))
        # 颜色渐变
        norm = Normalize(vmin=top30["importance"].min(), vmax=top30["importance"].max())
        cmap = plt.cm.get_cmap("plasma")
        colors = [cmap(norm(v)) for v in top30["importance"].values]

        bars = ax.barh(range(len(top30)), top30["importance"].values, color=colors, edgecolor="none", height=0.75)
        ax.set_yticks(range(len(top30)))
        ax.set_yticklabels(top30["feature"].values, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("重要性")
        ax.set_title("特征重要性 - Top 30 (LightGBM)", fontsize=16, fontweight="bold", pad=15)

        # 数值标注
        for bar, val in zip(bars, top30["importance"].values):
            ax.text(val + top30["importance"].max() * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f}", va="center", fontsize=8, color="#cbd5e1")
        ax.grid(axis="x", alpha=0.3)

        # colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30)
        cbar.set_label("重要性", color="#94a3b8")
        cbar.ax.yaxis.set_tick_params(color="#94a3b8")
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="#94a3b8")

        save_fig(fig, name)
        return name
    except Exception as e:
        logger.error("[%s] 生成失败: %s", name, e)
        return None


# ============================================================
# Chart 08: Rolling Sharpe Ratio
# ============================================================
def chart_08_rolling_sharpe(ctx):
    """滚动 60 日 Sharpe 比率折线图。"""
    name = "08_rolling_sharpe.png"
    logger.info("[%s] 生成滚动 Sharpe 图...", name)
    try:
        dates = ctx["test_dates"]
        returns = ctx["test_returns"]
        preds = ctx["test_preds"]

        unique_dates = sorted(np.unique(dates))
        if len(unique_dates) < 65:
            logger.warning("[%s] 交易日不足 65 天，跳过", name)
            return None

        # 策略日收益
        daily_rets = []
        for d in unique_dates:
            mask = dates == d
            if mask.sum() < 3:
                daily_rets.append(0.0)
                continue
            d_preds = preds[mask]
            d_rets = returns[mask]
            k = max(3, mask.sum() // 3)
            top_idx = np.argsort(d_preds)[-k:]
            daily_rets.append(d_rets[top_idx].mean())
        daily_rets = np.array(daily_rets)

        analyzer = SharpeAnalyzer(trading_days=252)
        rolling_df = analyzer.compute_rolling_sharpe(daily_rets, window=60, dates=unique_dates)

        if rolling_df is None or rolling_df.empty:
            logger.warning("[%s] 滚动 Sharpe 为空，跳过", name)
            return None

        fig, ax = plt.subplots(figsize=(14, 7))
        x = rolling_df["date"].values
        sharpe_vals = rolling_df["sharpe"].values

        ax.plot(x, sharpe_vals, color="#38bdf8", linewidth=1.5, label="60日滚动 Sharpe")
        ax.fill_between(x, 0, sharpe_vals, where=sharpe_vals > 0, color="#22c55e", alpha=0.3)
        ax.fill_between(x, 0, sharpe_vals, where=sharpe_vals <= 0, color="#ef4444", alpha=0.3)
        ax.axhline(y=0, color="#64748b", linestyle="--", linewidth=0.8)

        mean_sharpe = sharpe_vals.mean()
        ax.axhline(y=mean_sharpe, color="#facc15", linestyle="--", linewidth=1, alpha=0.7,
                    label=f"均值: {mean_sharpe:.2f}")

        ax.set_xlabel("日期")
        ax.set_ylabel("Sharpe 比率")
        ax.set_title("滚动 60 日 Sharpe 比率", fontsize=16, fontweight="bold", pad=15)
        ax.legend(loc="upper right", fontsize=11)
        ax.grid(alpha=0.3)
        save_fig(fig, name)
        return name
    except Exception as e:
        logger.error("[%s] 生成失败: %s", name, e)
        return None


# ============================================================
# Chart 09: Monthly Returns Heatmap
# ============================================================
def chart_09_monthly_returns(ctx):
    """月度收益率热力图。"""
    name = "09_monthly_returns.png"
    logger.info("[%s] 生成月度收益热力图...", name)
    try:
        dates = ctx["test_dates"]
        returns = ctx["test_returns"]
        preds = ctx["test_preds"]

        unique_dates = sorted(np.unique(dates))
        daily_rets_map = {}
        for d in unique_dates:
            mask = dates == d
            if mask.sum() < 3:
                daily_rets_map[d] = 0.0
                continue
            d_preds = preds[mask]
            d_rets = returns[mask]
            k = max(3, mask.sum() // 3)
            top_idx = np.argsort(d_preds)[-k:]
            daily_rets_map[d] = d_rets[top_idx].mean()

        # 构建 DataFrame: index=年, columns=月
        ret_series = pd.Series(daily_rets_map, index=pd.to_datetime(unique_dates))
        monthly = ret_series.resample("M").apply(lambda x: np.prod(1 + x) - 1)
        monthly.index = pd.to_datetime(monthly.index)

        pivot = pd.DataFrame({
            "year": monthly.index.year,
            "month": monthly.index.month,
            "ret": monthly.values,
        })
        pivot_table = pivot.pivot(index="year", columns="month", values="ret")
        pivot_table.columns = [f"{m}月" for m in pivot_table.columns]

        if pivot_table.empty:
            logger.warning("[%s] 月度数据为空，跳过", name)
            return None

        fig, ax = plt.subplots(figsize=(14, max(6, len(pivot_table) * 0.5 + 2)))

        # 热力图
        data = pivot_table.values
        im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-0.1, vmax=0.1)

        # 坐标
        ax.set_xticks(range(len(pivot_table.columns)))
        ax.set_xticklabels(pivot_table.columns, fontsize=10)
        ax.set_yticks(range(len(pivot_table.index)))
        ax.set_yticklabels(pivot_table.index, fontsize=10)

        # 数值标注
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if np.isfinite(val):
                    color = "#0f172a" if abs(val) > 0.06 else "#e2e8f0"
                    ax.text(j, i, f"{val:.1%}", ha="center", va="center", fontsize=8, color=color)

        cbar = fig.colorbar(im, ax=ax, pad=0.02, aspect=30, shrink=0.8)
        cbar.set_label("月度收益率", color="#94a3b8")
        cbar.ax.yaxis.set_tick_params(color="#94a3b8")
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="#94a3b8")

        ax.set_title("月度收益率热力图", fontsize=16, fontweight="bold", pad=15)
        save_fig(fig, name)
        return name
    except Exception as e:
        logger.error("[%s] 生成失败: %s", name, e)
        return None


# ============================================================
# Chart 10: Risk Metrics Dashboard
# ============================================================
def chart_10_risk_metrics(ctx):
    """风险指标仪表盘。"""
    name = "10_risk_metrics.png"
    logger.info("[%s] 生成风险指标仪表盘...", name)
    try:
        dates = ctx["test_dates"]
        returns = ctx["test_returns"]
        preds = ctx["test_preds"]

        unique_dates = sorted(np.unique(dates))
        daily_rets = []
        for d in unique_dates:
            mask = dates == d
            if mask.sum() < 3:
                daily_rets.append(0.0)
                continue
            d_preds = preds[mask]
            d_rets = returns[mask]
            k = max(3, mask.sum() // 3)
            top_idx = np.argsort(d_preds)[-k:]
            daily_rets.append(d_rets[top_idx].mean())
        daily_rets = np.array(daily_rets)
        daily_rets = daily_rets[np.isfinite(daily_rets)]

        if len(daily_rets) < 10:
            logger.warning("[%s] 数据不足，跳过", name)
            return None

        # 使用项目模块计算指标
        sharpe_analyzer = SharpeAnalyzer(trading_days=252)
        dd_analyzer = DrawdownAnalyzer()
        equity = np.cumprod(1 + daily_rets)

        try:
            sharpe_val = sharpe_analyzer.compute_sharpe(daily_rets, risk_free=0.03, annualize=True)
        except Exception:
            sharpe_val = 0.0
        try:
            sortino_val = sharpe_analyzer.compute_sortino(daily_rets, risk_free=0.03, annualize=True)
        except Exception:
            sortino_val = 0.0

        try:
            dd_result = dd_analyzer.compute_max_drawdown(equity, dates=unique_dates if len(unique_dates) == len(equity) else None)
            max_dd_val = dd_result[0]
        except Exception:
            max_dd_val = 0.0

        # 年化收益
        total_ret = np.prod(1 + daily_rets) - 1
        n_years = len(daily_rets) / 252
        annual_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0.0

        # 胜率
        win_rate = (daily_rets > 0).mean()

        metrics = [
            ("年化收益率", f"{annual_ret:.2%}", "#22d3ee"),
            ("Sharpe 比率", f"{sharpe_val:.2f}", "#22c55e" if sharpe_val > 0 else "#f87171"),
            ("Sortino 比率", f"{sortino_val:.2f}", "#22c55e" if sortino_val > 0 else "#f87171"),
            ("最大回撤", f"{max_dd_val:.2%}", "#f87171"),
            ("日胜率", f"{win_rate:.2%}", "#a78bfa"),
        ]

        fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5))
        fig.suptitle("风险指标仪表盘", fontsize=18, fontweight="bold", y=0.95, color="#e2e8f0")

        for i, (label, value, color) in enumerate(metrics):
            ax = axes[i]
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")

            # 卡片背景
            card = plt.Rectangle((0.05, 0.05), 0.9, 0.85, linewidth=2,
                                   edgecolor=color, facecolor="#1e293b",
                                   alpha=0.9, transform=ax.transAxes,
                                   clip_on=False, zorder=1)
            ax.add_patch(card)

            # 数值
            ax.text(0.5, 0.55, value, transform=ax.transAxes, fontsize=28,
                    fontweight="bold", color=color, ha="center", va="center", zorder=2)
            # 标签
            ax.text(0.5, 0.22, label, transform=ax.transAxes, fontsize=14,
                    color="#94a3b8", ha="center", va="center", zorder=2)

        plt.subplots_adjust(wspace=0.15, top=0.85, bottom=0.1)
        save_fig(fig, name)
        return name
    except Exception as e:
        logger.error("[%s] 生成失败: %s", name, e)
        return None


# ============================================================
# Main
# ============================================================
CHART_GENERATORS = [
    chart_01_feature_ic,
    chart_02_ic_series,
    chart_03_model_performance,
    chart_04_equity_curve,
    chart_05_drawdown,
    chart_06_return_distribution,
    chart_07_feature_importance,
    chart_08_rolling_sharpe,
    chart_09_monthly_returns,
    chart_10_risk_metrics,
]


def main():
    parser = argparse.ArgumentParser(description="量化分析图表生成工具")
    parser.add_argument("--sample", type=int, default=50, help="采样股票数量 (默认 50)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("量化分析图表生成工具")
    logger.info("采样股票数: %d", args.sample)
    logger.info("=" * 60)

    t_start = time.time()
    setup_output_dir()

    # ---- 构建数据集 ----
    try:
        ctx = build_dataset(data_dict=None, sample_size=args.sample)
    except Exception as e:
        logger.error("数据准备失败: %s", e)
        sys.exit(1)

    # ---- 逐张生成图表 ----
    results = []
    logger.info("=" * 60)
    logger.info("开始生成图表 (共 %d 张)", len(CHART_GENERATORS))
    logger.info("=" * 60)

    for gen_func in CHART_GENERATORS:
        try:
            result = gen_func(ctx)
            if result:
                results.append(result)
                logger.info("  [OK] %s", result)
            else:
                logger.warning("  [SKIP] %s (无数据或异常)", gen_func.__name__)
        except Exception as e:
            logger.error("  [FAIL] %s: %s", gen_func.__name__, e)

    # ---- 汇总 ----
    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("图表生成完成")
    logger.info("成功: %d / %d", len(results), len(CHART_GENERATORS))
    logger.info("总耗时: %.1f 秒", elapsed)
    logger.info("输出目录: %s", OUTPUT_DIR)
    logger.info("-" * 60)
    for r in results:
        logger.info("  %s", r)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
