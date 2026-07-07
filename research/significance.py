# -*- coding: utf-8 -*-
"""
IC 显著性检验 — significance.py

提供逐日 IC 序列的统计显著性检验能力，对接 qlib_pipeline/ic_stability.py 的
compute_daily_rank_ic() 输出。

核心功能：
  1. Newey-West 调整的 t 检验 — 处理日度 IC 序列中标签重叠导致的自相关
  2. 多重检验校正 — Bonferroni / Benjamini-Hochberg (FDR)

使用场景：
  - 判断某个配置的 IC 是否显著异于零（第 8.1 节）
  - 探索阶段结束后，报告校正后的显著性阈值（第 8.2 节）

Reference:
  - Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite,
    heteroskedasticity and autocorrelation consistent covariance matrix.
  - Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate.

Usage:
    from research.significance import newey_west_test, multiple_testing_correction
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def newey_west_se(
    ic_series: pd.Series,
    max_lags: Optional[int] = None,
) -> float:
    """计算 Newey-West 调整后的标准误。

    核心思想：标准 t 检验假设样本独立同分布，但日度 IC 序列因标签窗口重叠
    存在自相关，直接用标准误会严重低估。Newey-West 通过引入滞后协方差项
    修正这一问题。

    Args:
        ic_series: 逐日 IC 序列（pd.Series）
        max_lags: 最大滞后阶数（默认 = len(series) 的 1/4 或标签周期长度）

    Returns:
        Newey-West 调整后的标准误
    """
    n = len(ic_series)
    if n < 2:
        return 0.0

    residuals = ic_series.values - ic_series.mean()
    if max_lags is None:
        max_lags = min(int(n ** 0.25), n - 1)  # 默认滞后阶数

    max_lags = min(max_lags, n - 1)

    # 方差部分（异方差一致）
    variance = np.sum(residuals ** 2) / n

    # 自协方差部分（Newey-West 修正）
    autocov_sum = 0.0
    for lag in range(1, max_lags + 1):
        weight = 1.0 - lag / (max_lags + 1)  # Bartlett kernel
        autocov = np.sum(residuals[lag:] * residuals[:-lag]) / n
        autocov_sum += 2.0 * weight * autocov

    nw_variance = variance + autocov_sum
    nw_variance = max(nw_variance, 0.0)  # 防止数值问题导致负值

    return np.sqrt(nw_variance / n)


def newey_west_test(
    ic_series: pd.Series,
    max_lags: Optional[int] = None,
    alpha: float = 0.05,
) -> Dict:
    """Newey-West 调整的 IC 显著性 t 检验。

    判断标准：只有当 |t_NW| > 2（约 95% 置信）时，才认为该配置下的 IC 均值
    显著异于零，可以进入下一步分析。

    Args:
        ic_series: 逐日 IC 序列
        max_lags: 最大滞后阶数（默认 = 标签周期长度，如 60 日标签用 60）
        alpha: 显著性水平

    Returns:
        dict with keys:
            - ic_mean: IC 均值
            - ic_std: IC 标准差（原始）
            - nw_se: Newey-West 调整后的标准误
            - t_stat: t 统计量
            - t_nw: Newey-West 调整后的 t 统计量
            - p_value_naive: 朴素 t 检验的 p 值（不可信，仅供参考）
            - p_value_nw: Newey-West 调整后的近似 p 值
            - significant: 是否通过显著性检验
            - n_samples: 样本量
            - max_lags: 使用的滞后阶数
    """
    if len(ic_series) == 0:
        return {
            "ic_mean": None, "ic_std": None, "nw_se": None,
            "t_stat": None, "t_nw": None,
            "p_value_naive": None, "p_value_nw": None,
            "significant": False, "n_samples": 0, "max_lags": 0,
        }

    n = len(ic_series)
    ic_mean = float(ic_series.mean())
    ic_std = float(ic_series.std())

    if max_lags is None:
        max_lags = min(int(n ** 0.25), n - 1)

    max_lags = max(1, min(max_lags, n - 1))

    # 朴素标准误
    naive_se = ic_std / np.sqrt(n) if ic_std > 0 else 0.0
    t_stat = ic_mean / naive_se if naive_se > 0 else 0.0

    # Newey-West 标准误
    nw_se = newey_west_se(ic_series, max_lags=max_lags)
    t_nw = ic_mean / nw_se if nw_se > 0 else 0.0

    # 近似 p 值（基于正态分布，严格来说应该用 t 分布但大样本下近似）
    from scipy.stats import norm
    p_value_naive = 2.0 * (1.0 - norm.cdf(abs(t_stat)))
    p_value_nw = 2.0 * (1.0 - norm.cdf(abs(t_nw)))

    significant = abs(t_nw) > 2.0  # 约 95% 置信

    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "nw_se": nw_se,
        "t_stat": t_stat,
        "t_nw": t_nw,
        "p_value_naive": p_value_naive,
        "p_value_nw": p_value_nw,
        "significant": significant,
        "n_samples": n,
        "max_lags": max_lags,
    }


def multiple_testing_correction(
    p_values: List[float],
    method: str = "fdr_bh",
    alpha: float = 0.05,
) -> Dict:
    """多重检验校正。

    当探索阶段比较了 N 组配置时，需要报告校正后的显著性阈值，
    否则从 N 次尝试中挑出"运气最好"的那个会有过高的假阳性率。

    Args:
        p_values: 各组实验的 p 值列表
        method: 校正方法
            - "bonferroni": Bonferroni 校正（最保守）
            - "fdr_bh": Benjamini-Hochberg FDR 校正（推荐）
        alpha: 目标显著性水平

    Returns:
        dict with:
            - corrected_threshold: 校正后的显著性阈值
            - significant_indices: 通过校正的配置索引列表
            - adjusted_p_values: 校正后的 p 值列表
    """
    n = len(p_values)
    if n == 0:
        return {"corrected_threshold": alpha, "significant_indices": [], "adjusted_p_values": []}

    if method == "bonferroni":
        corrected_threshold = alpha / n
        adjusted = [min(p * n, 1.0) for p in p_values]
        significant = [i for i, p_adj in enumerate(adjusted) if p_adj < alpha]
    elif method == "fdr_bh":
        # Benjamini-Hochberg procedure
        sorted_idx = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_idx]
        adjusted = np.ones(n)
        for i in range(n - 1, -1, -1):
            if i == n - 1:
                adjusted[sorted_idx[i]] = sorted_p[i]
            else:
                adjusted[sorted_idx[i]] = min(
                    adjusted[sorted_idx[i + 1]],
                    sorted_p[i] * n / (i + 1),
                )
        adjusted = [min(a, 1.0) for a in adjusted]
        corrected_threshold = alpha  # FDR 使用原始 alpha
        significant = [i for i, p_adj in enumerate(adjusted) if p_adj < alpha]
    else:
        raise ValueError(f"未知校正方法: {method}，支持 bonferroni / fdr_bh")

    return {
        "corrected_threshold": corrected_threshold,
        "significant_indices": significant,
        "adjusted_p_values": adjusted,
        "n_tests": n,
        "method": method,
    }


