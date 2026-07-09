# -*- coding: utf-8 -*-
"""
收益归因分解 — attribution.py

将超额收益拆解为"选股 Alpha 贡献"和"风格暴露贡献"两部分，
帮助判断策略收益的真实来源。

简化版做法：用持仓的风格暴露乘以对应风格因子的同期收益，得到风格贡献部分，
总超额收益减去风格贡献即为选股 Alpha。

使用场景：
  - 第 10.1 节：回测报告的风格归因分析
  - 判断策略是否在不自知地成为"单一风格因子的杠杆化押注"

Usage:
    from research.attribution import decompose_excess_return, attribution_report
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def decompose_excess_return(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    factor_exposures_df: pd.DataFrame,
    factor_returns_df: pd.DataFrame,
) -> pd.DataFrame:
    """将超额收益分解为风格贡献和 Alpha 贡献。

    Args:
        portfolio_returns: 组合日收益序列，index = date
        benchmark_returns: 基准日收益序列，index = date
        factor_exposures_df: 因子暴露 DataFrame，columns = 风格因子，index = date
        factor_returns_df: 因子收益 DataFrame，columns = 风格因子，index = date

    Returns:
        DataFrame with columns:
            - total_excess: 总超额收益
            - style_contribution: 风格贡献总和
            - alpha_contribution: 选股 Alpha
            - 各风格因子独立贡献
    """
    # 对齐日期
    common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
    common_dates = common_dates.intersection(factor_exposures_df.index)
    common_dates = common_dates.intersection(factor_returns_df.index)

    if len(common_dates) == 0:
        logger.warning("没有共同日期，无法进行归因分解")
        return pd.DataFrame()

    # 提取公共日期的数据
    port_ret = portfolio_returns.loc[common_dates]
    bench_ret = benchmark_returns.loc[common_dates]
    excess_ret = port_ret - bench_ret

    result = pd.DataFrame(index=common_dates)
    result["total_excess"] = excess_ret.values

    # 计算各风格因子贡献
    style_contribution = pd.Series(0.0, index=common_dates)

    factor_names = factor_exposures_df.columns.intersection(factor_returns_df.columns)
    for factor in factor_names:
        exposures = factor_exposures_df.loc[common_dates, factor]
        returns = factor_returns_df.loc[common_dates, factor]
        factor_contrib = exposures * returns
        result[f"style_{factor}"] = factor_contrib.values
        style_contribution = style_contribution + factor_contrib

    result["style_total"] = style_contribution.values
    result["alpha"] = result["total_excess"] - result["style_total"]

    return result


def calculate_factor_returns(
    factor_values: pd.DataFrame,
    stock_returns: pd.DataFrame,
    industry_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """用 Fama-MacBeth 截面 OLS 回归计算因子日收益。

    对每个交易日，做截面 OLS 回归（Fama-MacBeth 第一步）：
        r_i = α + Σ(β_j × F_{i,j}) + ε_i

    回归系数 β_j 即为因子 j 在该交易日的收益。
    对多日取均值即可得到因子风险溢价（Fama-MacBeth 第二步）。

    Args:
        factor_values: 因子值 DataFrame，index = date, columns = 因子名称
            每个单元格为该日期/因子下各股票的因子暴露 Series（index=stock_code）
            示例: factor_values.loc["2025-01-01", "size"] → Series(index=stock, values=exposure)
        stock_returns: 股票收益，index = date, columns = stock_code
        industry_map: 行业映射（可选，用于行业中性化）

    Returns:
        pd.DataFrame, index = date, columns = 因子名称, values = 因子日收益（OLS 回归系数）
    """
    dates = stock_returns.index.intersection(factor_values.index.unique())
    factors = list(factor_values.columns)

    if len(factors) == 0:
        logger.warning("因子列表为空")
        return pd.DataFrame()

    factor_returns = pd.DataFrame(index=dates, columns=factors, dtype=float)

    for date in dates:
        try:
            daily_ret = stock_returns.loc[date].dropna()
            if len(daily_ret) < 10:
                continue

            # 收集当天所有因子的股票级暴露
            factor_data = {}
            for factor in factors:
                if date not in factor_values.index:
                    continue
                daily_fv = factor_values.loc[date, factor]
                if isinstance(daily_fv, pd.Series):
                    factor_data[factor] = daily_fv
                elif hasattr(daily_fv, 'index'):
                    # 可能是 numpy array 或 list，尝试转为 Series
                    factor_data[factor] = pd.Series(daily_fv)

            if len(factor_data) < len(factors):
                continue  # 当天因子数据不完整，跳过

            # 对齐所有股票
            common_stocks = set(daily_ret.index)
            for fv in factor_data.values():
                common_stocks &= set(fv.index)
            common_stocks = sorted(common_stocks)

            min_samples = max(10, len(factors) * 5)  # 至少 5 倍于因子数
            if len(common_stocks) < min_samples:
                continue

            # 构建设计矩阵 X 和因变量 y
            y = daily_ret.loc[common_stocks].values.astype(float)
            X = np.column_stack([
                factor_data[f].loc[common_stocks].values.astype(float)
                for f in factors
            ])

            # 去除 NaN / Inf 行
            valid_mask = (
                ~np.isnan(X).any(axis=1) &
                ~np.isinf(X).any(axis=1) &
                ~np.isnan(y) &
                ~np.isinf(y)
            )
            X_valid = X[valid_mask]
            y_valid = y[valid_mask]

            if len(X_valid) < min_samples:
                continue

            # 截面 OLS 回归（Fama-MacBeth 第一步）
            # 添加截距项: y = α + X·β + ε
            X_with_intercept = np.column_stack([np.ones(len(X_valid)), X_valid])
            coeffs, residuals, rank, singular = np.linalg.lstsq(
                X_with_intercept, y_valid, rcond=None
            )

            # coeffs[0] = α (截距), coeffs[1:] = β_j (因子收益)
            for i, factor in enumerate(factors):
                factor_returns.loc[date, factor] = float(coeffs[i + 1])

        except Exception as e:
            logger.debug("日期 %s Fama-MacBeth 截面回归失败: %s", date, e)

    return factor_returns.dropna(how="all")


def attribution_report(
    decomposition: pd.DataFrame,
    n_periods: int = 252,
    label_horizon: int = 20,
) -> Dict:
    """生成收益归因分析的可读报告，并接入 Newey-West 显著性检验。

    对选股 Alpha 逐期收益序列做 Newey-West 调整后的 t 检验，
    处理日度收益序列的自相关，避免高估显著性。

    Args:
        decomposition: from decompose_excess_return()
        n_periods: 年化周期数（默认 252 个交易日）
        label_horizon: 标签前瞻天数，用于 Newey-West 滞后阶数

    Returns:
        dict with keys:
            - report: 格式化的报告字符串
            - annualized: 各列年化收益 Series
            - significance: Newey-West 检验结果 dict（对 alpha 序列）
    """
    if decomposition.empty:
        return {"report": "收益归因分析: 无可用数据", "annualized": pd.Series(), "significance": {}}

    # 年化各列
    style_cols = [c for c in decomposition.columns if c.startswith("style_")]
    annualized = decomposition.mean() * n_periods

    # 对选股 Alpha 逐期收益做 Newey-West 显著性检验
    significance = {}
    alpha_series = decomposition.get("alpha")
    if alpha_series is not None and len(alpha_series.dropna()) > 5:
        try:
            from research.significance import newey_west_test
            significance = newey_west_test(
                alpha_series.dropna(),
                max_lags=label_horizon,
            )
        except Exception as e:
            logger.warning("Newey-West 检验失败: %s", e)
            significance = {}

    t_nw = significance.get("t_nw")
    p_nw = significance.get("p_value_nw")
    sig = significance.get("significant")

    lines = [
        "=" * 60,
        "收益归因分析",
        "=" * 60,
        f"  n_periods 样本量: {len(decomposition)}",
        "",
        "  年化收益分解:",
        f"  总超额收益:             {annualized.get('total_excess', 0):>8.4%}",
        f"  风格贡献合计:           {annualized.get('style_total', 0):>8.4%}",
        f"  选股 Alpha:             {annualized.get('alpha', 0):>8.4%}",
        "",
        "  风格因子独立贡献:",
    ]

    for col in style_cols:
        factor_name = col.replace("style_", "")
        lines.append(f"    {factor_name:<20} {annualized.get(col, 0):>8.4%}")

    # 因子收益显著性（Newey-West 检验）
    lines.append("")
    lines.append("-" * 60)
    if t_nw is not None and p_nw is not None:
        sig_str = "是" if sig else "否"
        lines.append(
            f"  因子收益显著性: t_NW={t_nw:.2f}, p={p_nw:.3f}, 是否显著: {sig_str}"
        )
        lines.append(f"  (Newey-West 调整, 滞后阶数={significance.get('max_lags', '?')}, "
                     f"样本数={significance.get('n_samples', '?')})")
    else:
        lines.append("  因子收益显著性: 样本不足，无法检验")

    # 判断 Alpha 是否显著
    alpha = annualized.get("alpha", 0)
    style_total = annualized.get("style_total", 0)
    total_excess = annualized.get("total_excess", 0)

    lines.append("")
    lines.append("-" * 60)
    if abs(total_excess) > 0.001:
        alpha_ratio = abs(alpha / total_excess) if abs(total_excess) > 0 else 0
        if alpha_ratio > 0.7:
            lines.append("  ✓ 结论: 超额收益主要由选股 Alpha 驱动（{:.0%}）".format(alpha_ratio))
        elif alpha_ratio > 0.3:
            lines.append("  ~ 结论: 超额收益由选股 Alpha 和风格暴露共同贡献（Alpha={:.0%}）".format(alpha_ratio))
        else:
            lines.append(
                "  ⚠ 结论: 超额收益主要由风格暴露驱动（Alpha={:.0%}），"
                "策略可能是不自知的风格因子押注".format(alpha_ratio)
            )
        # 如果 Alpha 统计上不显著，追加警告
        if sig is False:
            lines.append("  ⚠ 注意: 选股 Alpha 未通过 Newey-West 显著性检验，"
                         "不应过度解读其正贡献")
    else:
        lines.append("    超额收益接近零，归因分析参考价值有限")

    lines.append("=" * 60)
    return {
        "report": "\n".join(lines),
        "annualized": annualized,
        "significance": significance,
    }