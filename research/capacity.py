# -*- coding: utf-8 -*-
"""
流动性/容量约束检查 — capacity.py

在推荐最终选股前检查单只股票的流动性，确保按当前持仓规模
实际市场冲击成本不会过高。

核心约束：建议持仓金额不超过该股票日均成交额的 5%-10%。

Usage:
    from research.capacity import check_portfolio_capacity
"""

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def check_single_stock_capacity(
    stock_code: str,
    avg_daily_volume: float,
    position_value: float,
    max_usage_pct: float = 0.05,
) -> Tuple[bool, float]:
    """检查单只股票的流动性容量。

    Args:
        stock_code: 股票代码
        avg_daily_volume: 过去 60 日平均成交额（元）
        position_value: 拟建仓金额（元）
        max_usage_pct: 最大允许占比，建议 0.05 (5%)

    Returns:
        (acceptable: bool, usage_pct: float)
    """
    if avg_daily_volume <= 0:
        # 没有数据，假设不通过
        logger.warning("股票 %s 无成交额数据，标记为不可接受", stock_code)
        return False, 1.0

    usage_pct = position_value / avg_daily_volume if avg_daily_volume > 0 else 1.0

    if usage_pct > max_usage_pct:
        logger.warning(
            "股票 %s 流动性不足: 占用 %.1f%% > %.1f%% (position=%.0f, avg_volume=%.0f)",
            stock_code, usage_pct * 100, max_usage_pct * 100,
            position_value, avg_daily_volume,
        )
        return False, usage_pct

    return True, usage_pct


def check_portfolio_capacity(
    portfolio_weights: pd.Series,
    avg_daily_volume: pd.Series,
    account_value: float = 1e8,
    max_usage_pct: float = 0.05,
) -> Tuple[pd.DataFrame, bool]:
    """检查整个组合的流动性容量约束。

    Args:
        portfolio_weights: 组合权重，index = stock_code, values = weight
        avg_daily_volume: 股票日均成交额，index = stock_code
        account_value: 账户总资金（元）
        max_usage_pct: 单只股票最大允许占比

    Returns:
        (result_df, all_ok)
            - result_df: 每只股票的检查结果
            - all_ok: 所有股票都通过检查时 True
    """
    result = []
    all_ok = True

    for stock, w in portfolio_weights.items():
        position_value = w * account_value
        avg_vol = avg_daily_volume.get(stock, 0.0)
        ok, usage = check_single_stock_capacity(
            stock, avg_vol, position_value, max_usage_pct,
        )
        result.append({
            "stock_code": stock,
            "weight": w,
            "position_value": position_value,
            "avg_daily_volume": avg_vol,
            "usage_pct": usage,
            "max_usage_pct": max_usage_pct,
            "acceptable": ok,
        })
        if not ok:
            all_ok = False

    df = pd.DataFrame(result).sort_values("usage_pct", ascending=False)
    return df, all_ok


def get_average_volume_from_data(
    volume_df: pd.DataFrame,
    window: int = 60,
) -> pd.Series:
    """从价量数据计算滚动日均成交额。

    Args:
        volume_df: 日成交额 DataFrame（价格 × 成交量），index = date, columns = stock_code
        window: 滚动窗口大小

    Returns:
        pd.Series, index = stock_code, values = 滚动平均日均成交额
    """
    # 计算滚动窗口均值
    rolling_avg_volume = volume_df.rolling(window=window).mean()
    # 取最新值
    latest_avg = rolling_avg_volume.iloc[-1]
    return latest_avg.dropna()