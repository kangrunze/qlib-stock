# -*- coding: utf-8 -*-
"""
Barra 风格因子暴露计算 — risk_model.py

提供简化版 Barra 风格因子暴露计算能力，用于回测结果的风险归因和监控。

核心功能：
  1. 计算组合相对于基准在五个核心风格维度上的暴露差异
  2. 风格维度：规模（Size）、估值（Value）、动量（Momentum）、波动率（Volatility）、质量（Quality）
  3. 监控：如果暴露绝对值超出预设阈值（±0.5 sigma）给出警告提示

参考 Barra Global Equity Model (GEM) 风格因子体系，本项目做简化实现。
  - Barra Global Equity Model (GEM)

Usage:
    from research.risk_model import calculate_style_exposures, check_style_constraints
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


STYLE_FACTORS = {
    "size": {
        "name": "规模",
        "description": "对数市值",
        "feature": "$volume_cap",
    },
    "value": {
        "name": "估值",
        "description": "PB 倒数 = 账面价值 / 市值",
        "feature": "$pb_inv",
    },
    "momentum": {
        "name": "动量",
        "description": "过去12个月收益，剔除最近1个月",
        "feature": "$mom_12m",
    },
    "volatility": {
        "name": "波动率",
        "description": "过去60日日收益标准差",
        "feature": "$vol_60d",
    },
    "quality": {
        "name": "质量",
        "description": "ROE",
        "feature": "$roe",
    },
}


def get_style_factor_names() -> List[str]:
    """返回支持的风格因子名称列表"""
    return list(STYLE_FACTORS.keys())


def calculate_style_exposures(
    portfolio_weights: pd.Series,
    factor_values: pd.DataFrame,
    benchmark_weights: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """计算组合相对于基准的风格因子暴露（以截面标准差为单位）。

    对每个风格因子做当日截面 z-score 标准化后，再计算组合与基准的加权暴露差，
    使得 active_exposure 是真正"以标准差为单位"的暴露差异，可直接与阈值比较。

    z-score 标准化:
        z_i = (x_i - μ) / σ
        其中 μ 和 σ 是当日所有股票在该因子上的截面均值和标准差。

    Args:
        portfolio_weights: 组合权重，index = stock_code, values = weight
        factor_values: 因子值，columns = 风格因子名称，index = stock_code
        benchmark_weights: 基准权重，index = stock_code, values = weight
            如果为 None，使用全市场平均（等权）作为基准

    Returns:
        DataFrame, columns = ["name_cn", "portfolio_exposure", "benchmark_exposure",
                              "active_exposure", "factor_mean", "factor_std"]
        index = 风格因子名称
        active_exposure 以截面标准差为单位（z-score 差值）
    """
    result = []

    # 对齐股票索引
    common_stocks = portfolio_weights.index.intersection(factor_values.index)
    if len(common_stocks) == 0:
        logger.warning("没有共同股票，无法计算风格暴露")
        return pd.DataFrame()

    port_weights_aligned = portfolio_weights.loc[common_stocks]
    port_weights_norm = port_weights_aligned / port_weights_aligned.sum()

    for style_name, style_info in STYLE_FACTORS.items():
        feat = style_info["feature"]
        if feat not in factor_values.columns:
            logger.warning("风格因子 %s 特征 %s 不存在，跳过", style_name, feat)
            continue

        fv = factor_values.loc[common_stocks, feat].astype(float)

        # ── 截面 z-score 标准化 ──
        # 去除 NaN 后计算截面均值和标准差
        fv_clean = fv.dropna()
        if len(fv_clean) < 10:
            logger.warning("风格因子 %s 有效样本不足 (%d)，跳过", style_name, len(fv_clean))
            continue

        fv_mean = fv_clean.mean()
        fv_std = fv_clean.std(ddof=1)  # 样本标准差
        if fv_std < 1e-12:
            logger.warning("风格因子 %s 截面标准差接近零，跳过标准化", style_name)
            fv_z = pd.Series(0.0, index=fv.index)
        else:
            # z-score: (x - μ) / σ  → 以标准差为单位
            fv_z = (fv - fv_mean) / fv_std

        # 组合暴露 = 加权平均（z-score 单位）
        port_exposure = (port_weights_norm * fv_z.loc[common_stocks]).sum()

        # 基准暴露
        if benchmark_weights is not None:
            bench_common = benchmark_weights.index.intersection(fv_z.index)
            if len(bench_common) > 0:
                bench_weights_aligned = benchmark_weights.loc[bench_common]
                bench_weights_norm = bench_weights_aligned / bench_weights_aligned.sum()
                bench_exposure = (bench_weights_norm * fv_z.loc[bench_common]).sum()
            else:
                bench_exposure = fv_z.mean()
        else:
            bench_exposure = fv_z.mean()

        active_exposure = port_exposure - bench_exposure

        result.append({
            "style": style_name,
            "name_cn": style_info["name"],
            "portfolio_exposure": port_exposure,
            "benchmark_exposure": bench_exposure,
            "active_exposure": active_exposure,
            "factor_mean": fv_mean,
            "factor_std": fv_std,
        })

    if not result:
        logger.warning("所有风格因子均不可用，返回空 DataFrame")
        return pd.DataFrame()
    return pd.DataFrame(result).set_index("style")


def check_style_constraints(
    exposures: pd.DataFrame,
    max_active_std: float = 0.5,
) -> Tuple[pd.DataFrame, bool]:
    """检查风格暴露是否超出约束阈值。

    Args:
        exposures: from calculate_style_exposures()
        max_active_std: 允许的最大 active 暴露（以标准差为单位）

    Returns:
        (exposures_with_status, all_ok)
            - exposures_with_status: 添加了 ok 列和提示信息
            - all_ok: True = 所有暴露都在约束内，False = 至少一个超出
    """
    exposures = exposures.copy()
    exposures["constraint_ok"] = (exposures["active_exposure"].abs() <= max_active_std)
    exposures["message"] = exposures.apply(
        lambda row: f"✓ OK" if row["constraint_ok"]
        else f"⚠ 超出约束 |active|={abs(row['active_exposure']):.2f} > {max_active_std}",
        axis=1,
    )
    all_ok = exposures["constraint_ok"].all()
    return exposures, all_ok


def get_industry_exposures(
    portfolio_weights: pd.Series,
    industry_map: Dict[str, str],
    benchmark_weights: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """计算组合相对于基准的行业暴露。

    Args:
        portfolio_weights: 组合权重
        industry_map: {stock_code: industry_name} 行业映射
        benchmark_weights: 基准权重

    Returns:
        DataFrame, 各行业暴露差异
    """
    all_industries = sorted(set(industry_map.values()))
    result = []

    # 组合行业权重
    port_industry = {}
    for stock, w in portfolio_weights.items():
        ind = industry_map.get(stock)
        if ind not in all_industries:
            continue
        port_industry[ind] = port_industry.get(ind, 0.0) + w

    # 归一化
    port_total = sum(port_industry.values())
    if port_total > 0:
        for ind in port_industry:
            port_industry[ind] /= port_total

    # 基准行业权重
    if benchmark_weights is not None:
        bench_industry = {}
        for stock, w in benchmark_weights.items():
            ind = industry_map.get(stock)
            if ind not in all_industries:
                continue
            bench_industry[ind] = bench_industry.get(ind, 0.0) + w

        bench_total = sum(bench_industry.values())
        if bench_total > 0:
            for ind in bench_industry:
                bench_industry[ind] /= bench_total
    else:
        n_inds = len(all_industries)
        bench_industry = {ind: 1.0 / n_inds for ind in all_industries}

    # 计算差异
    for ind in all_industries:
        port_w = port_industry.get(ind, 0.0)
        bench_w = bench_industry.get(ind, 0.0)
        result.append({
            "industry": ind,
            "portfolio_weight": port_w,
            "benchmark_weight": bench_w,
            "active_weight": port_w - bench_w,
        })

    df = pd.DataFrame(result).sort_values("active_weight", ascending=False)
    return df.set_index("industry")


def check_industry_constraints(
    exposures: pd.DataFrame,
    max_active_weight: float = 0.25,
) -> Tuple[pd.DataFrame, bool]:
    """检查行业暴露约束。

    Args:
        exposures: from get_industry_exposures()
        max_active_weight: 允许的最大行业偏离（百分比）

    Returns:
        (exposures_with_status, all_ok)
    """
    exposures = exposures.copy()
    exposures["constraint_ok"] = (exposures["active_weight"].abs() <= max_active_weight)
    exposures["message"] = exposures.apply(
        lambda row: f"✓ OK" if row["constraint_ok"]
        else f"⚠ 超出约束 |weight|={abs(row['active_weight']):.1%} > {max_active_weight:.1%}",
        axis=1,
    )
    all_ok = exposures["constraint_ok"].all()
    return exposures, all_ok