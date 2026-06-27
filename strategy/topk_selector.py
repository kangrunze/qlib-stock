"""
TopK 选股模块 - 基于预测得分的TopK股票选择，支持行业中性化

功能:
    - 基于模型预测得分进行TopK选股
    - 自动过滤ST、停牌、低流动性股票
    - 行业中性化选择（可选）

使用方法:
    selector = TopKSelector()
    selected = selector.select(predictions_df, date="2024-01-15", top_k=30, industry_neutral=True)
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class TopKSelector:
    """TopK选股器，支持行业中性化。

    Attributes:
        min_daily_volume: 最低日成交量（手），低于此值视为低流动性。
        max_per_industry: 行业中性化时每个行业最大持仓数。
        exclude_st: 是否排除ST股票。
        exclude_suspended: 是否排除停牌股票。
    """

    def __init__(
        self,
        min_daily_volume: int = 10000,
        max_per_industry: int = 5,
        exclude_st: bool = True,
        exclude_suspended: bool = True,
    ):
        """初始化TopK选股器。

        Args:
            min_daily_volume: 最低日成交量过滤阈值（手）。
            max_per_industry: 行业中性化时每个行业最大股票数。
            exclude_st: 是否过滤ST股票。
            exclude_suspended: 是否过滤停牌股票。
        """
        self.min_daily_volume = min_daily_volume
        self.max_per_industry = max_per_industry
        self.exclude_st = exclude_st
        self.exclude_suspended = exclude_suspended
        logger.info(
            f"TopKSelector 初始化: min_volume={min_daily_volume}, "
            f"max_per_industry={max_per_industry}, exclude_st={exclude_st}, "
            f"exclude_suspended={exclude_suspended}"
        )

    def select(
        self,
        predictions_df: pd.DataFrame,
        date: str,
        top_k: int = 30,
        industry_neutral: bool = True,
    ) -> pd.DataFrame:
        """执行选股流程：过滤 -> 排序 -> 行业中性化选择。

        Args:
            predictions_df: 预测数据DataFrame，必须包含列:
                - stock_code: 股票代码
                - score: 预测得分
                可选列:
                - industry: 行业分类
                - is_st: 是否ST
                - is_suspended: 是否停牌
                - volume: 成交量（手）
                - close: 收盘价
            date: 选股日期 (YYYY-MM-DD)。
            top_k: 选股数量。
            industry_neutral: 是否启用行业中性化。

        Returns:
            DataFrame，包含列: stock_code, score, rank, industry, weight。
            按score降序排列，前top_k只。
        """
        logger.info(f"开始选股: date={date}, top_k={top_k}, industry_neutral={industry_neutral}")

        if predictions_df.empty:
            logger.warning("预测数据为空，返回空DataFrame")
            return pd.DataFrame(columns=["stock_code", "score", "rank", "industry", "weight"])

        df = predictions_df.copy()

        # Step 1: 应用过滤器
        df = self.apply_filters(df)

        if df.empty:
            logger.warning("过滤后无可用股票")
            return pd.DataFrame(columns=["stock_code", "score", "rank", "industry", "weight"])

        # Step 2: 按得分降序排序
        df = df.sort_values("score", ascending=False).reset_index(drop=True)

        # Step 3: 行业中性化选择
        if industry_neutral and "industry" in df.columns:
            selected = self.industry_neutral_selection(
                df, top_k=top_k, max_per_industry=self.max_per_industry
            )
        else:
            if industry_neutral:
                logger.warning("未找到industry列，跳过行业中性化")
            selected = df.head(top_k).copy()

        # Step 4: 添加rank列
        selected = selected.reset_index(drop=True)
        selected["rank"] = range(1, len(selected) + 1)

        # Step 5: 初始化等权权重
        selected["weight"] = 1.0 / len(selected)

        logger.info(f"选股完成: 选定{len(selected)}只股票")
        return selected[["stock_code", "score", "rank", "industry", "weight"]]

    def apply_filters(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """对预测数据应用过滤条件。

        过滤条件：
        1. 排除ST股票（如果启用）
        2. 排除停牌股票（如果启用）
        3. 排除低流动性股票（成交量低于阈值）
        4. 排除得分为NaN的股票

        Args:
            predictions_df: 预测数据DataFrame。

        Returns:
            过滤后的DataFrame。
        """
        df = predictions_df.copy()
        initial_count = len(df)
        filter_log: Dict[str, int] = {}

        # 排除得分为NaN的股票
        nan_count = df["score"].isna().sum()
        if nan_count > 0:
            df = df.dropna(subset=["score"])
            filter_log["score_nan"] = nan_count

        # 排除无穷大得分
        inf_count = np.isinf(df["score"]).sum()
        if inf_count > 0:
            df = df[~np.isinf(df["score"])]
            filter_log["score_inf"] = inf_count

        # 排除ST股票
        if self.exclude_st and "is_st" in df.columns:
            st_count = (df["is_st"] == True).sum()
            df = df[df["is_st"] != True]
            filter_log["st_stocks"] = st_count

        # 排除停牌股票
        if self.exclude_suspended and "is_suspended" in df.columns:
            suspended_count = (df["is_suspended"] == True).sum()
            df = df[df["is_suspended"] != True]
            filter_log["suspended"] = suspended_count

        # 排除低流动性股票
        if "volume" in df.columns:
            low_vol_count = (df["volume"] < self.min_daily_volume).sum()
            df = df[df["volume"] >= self.min_daily_volume]
            filter_log["low_volume"] = low_vol_count

        # 排除价格为0或负数的股票
        if "close" in df.columns:
            invalid_price_count = (df["close"] <= 0).sum()
            df = df[df["close"] > 0]
            filter_log["invalid_price"] = invalid_price_count

        final_count = len(df)
        logger.info(
            f"过滤结果: {initial_count} -> {final_count} "
            f"(过滤: {filter_log})"
        )

        return df

    def industry_neutral_selection(
        self,
        predictions_df: pd.DataFrame,
        top_k: int = 30,
        max_per_industry: int = 5,
    ) -> pd.DataFrame:
        """执行行业中性化选股。

        每个行业最多选max_per_industry只，总计不超过top_k只。
        优先选择得分最高的股票，同时保持行业多样性。

        Args:
            predictions_df: 已排序的预测DataFrame（按score降序）。
            top_k: 目标选股数。
            max_per_industry: 每个行业最多选入仓位。

        Returns:
            行业中性化后的选股DataFrame。
        """
        if "industry" not in predictions_df.columns:
            logger.warning("无industry列，回退到普通TopK选择")
            return predictions_df.head(top_k).copy()

        df = predictions_df.copy()
        df = df.sort_values("score", ascending=False)

        selected_list = []
        industry_counts: Dict[str, int] = {}
        unknown_count = 0
        max_unknown = max_per_industry  # 未知行业也限制数量

        for _, row in df.iterrows():
            if len(selected_list) >= top_k:
                break

            industry = row.get("industry")
            if pd.isna(industry) or str(industry).strip() == "":
                industry_key = "__UNKNOWN__"
            else:
                industry_key = str(industry)

            current_count = industry_counts.get(industry_key, 0)
            cap = max_unknown if industry_key == "__UNKNOWN__" else max_per_industry

            if current_count < cap:
                selected_list.append(row)
                industry_counts[industry_key] = current_count + 1

        result = pd.DataFrame(selected_list)
        logger.info(
            f"行业中性化结果: {len(result)}只股票, "
            f"覆盖{len(industry_counts)}个行业, "
            f"行业持仓分布: {dict(list(industry_counts.items())[:10])}"
        )
        return result


class TopKSelectorSimple:
    """简化版TopK选股器（不依赖外部数据，仅需预测得分）。

    适用于快速选股场景，不进行ST/停牌/流动性过滤。
    """

    def select(
        self,
        predictions_df: pd.DataFrame,
        top_k: int = 30,
        score_col: str = "score",
        stock_col: str = "stock_code",
    ) -> pd.DataFrame:
        """简化版选股：按得分降序取前top_k。

        Args:
            predictions_df: 预测DataFrame。
            top_k: 选股数量。
            score_col: 得分列名。
            stock_col: 股票代码列名。

        Returns:
            选股结果DataFrame。
        """
        df = predictions_df.dropna(subset=[score_col]).copy()
        df = df.sort_values(score_col, ascending=False).head(top_k).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
        df["weight"] = 1.0 / len(df)
        return df