"""
权重分配模块 - 提供多种投资组合权重分配策略

支持的方法:
    - 等权分配 (equal)
    - 得分加权分配 (score_weighted)
    - 风险平价分配 (risk_parity, 简化版-逆波动率)

使用方法:
    allocator = WeightAllocator()
    weighted = allocator.allocate(selected_stocks, method="score_weighted")
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class WeightAllocator:
    """投资组合权重分配器。

    Attributes:
        min_weight: 最小权重阈值，低于此值设为0。
        max_weight: 最大单只股票权重上限（用于裁剪）。
    """

    def __init__(self, min_weight: float = 0.0, max_weight: float = 0.15):
        """初始化权重分配器。

        Args:
            min_weight: 最小权重阈值，低于此值设为0。
            max_weight: 最大单只股票权重，用于裁剪。
        """
        self.min_weight = min_weight
        self.max_weight = max_weight

    def allocate(
        self,
        selected_stocks: pd.DataFrame,
        method: str = "equal",
        returns_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """权重分配调度器。

        Args:
            selected_stocks: 选股结果DataFrame，至少包含stock_code列。
                如果method='score_weighted'，需要score列。
            method: 分配方法，可选值:
                - "equal": 等权分配
                - "score_weighted": 按预测得分加权
                - "risk_parity": 风险平价（简化逆波动率）
            returns_data: 收益率数据（risk_parity方法需要），DataFrame，
                index为日期，columns为stock_code。

        Returns:
            包含weight列的DataFrame（在原DataFrame基础上添加）。
        """
        method_map = {
            "equal": self.allocate_equal,
            "score_weighted": self.allocate_score_weighted,
            "risk_parity": self.allocate_risk_parity,
        }

        if method not in method_map:
            raise ValueError(
                f"不支持的权重分配方法: {method}。"
                f"可选: {list(method_map.keys())}"
            )

        logger.info(f"执行权重分配: method={method}, stocks={len(selected_stocks)}")
        result = method_map[method](selected_stocks, returns_data)
        result = self._clip_and_normalize(result)
        logger.info(
            f"权重分配完成: 权重范围 [{result['weight'].min():.4f}, {result['weight'].max():.4f}]"
        )
        return result

    def allocate_equal(self, selected_stocks: pd.DataFrame, _: Any = None) -> pd.DataFrame:
        """等权分配 - 每只股票权重为 1/n。

        Args:
            selected_stocks: 选股DataFrame。
            _: 占位参数（保持接口一致）。

        Returns:
            添加了weight列的DataFrame。
        """
        df = selected_stocks.copy()
        n = len(df)
        if n == 0:
            df["weight"] = 0.0
            return df
        df["weight"] = 1.0 / n
        return df

    def allocate_score_weighted(
        self, selected_stocks: pd.DataFrame, _: Any = None
    ) -> pd.DataFrame:
        """按预测得分加权 - 权重与score成正比。

        对所有score应用softmax进行归一化，避免负分问题。

        Args:
            selected_stocks: 选股DataFrame，必须包含score列。
            _: 占位参数。

        Returns:
            添加了weight列的DataFrame。

        Raises:
            ValueError: 如果score列不存在。
        """
        if "score" not in selected_stocks.columns:
            raise ValueError("score_weighted方法需要selected_stocks包含'score'列")

        df = selected_stocks.copy()
        n = len(df)
        if n == 0:
            df["weight"] = 0.0
            return df

        scores = df["score"].values.astype(float)

        # 使用softmax进行归一化：处理负分并确保权重和为1
        # 减去最大值防止数值溢出
        scores_shifted = scores - scores.max()
        exp_scores = np.exp(scores_shifted)
        weights = exp_scores / exp_scores.sum()

        df["weight"] = weights
        return df

    def allocate_risk_parity(
        self,
        selected_stocks: pd.DataFrame,
        returns_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """风险平价分配（简化版 - 逆波动率加权）。

        权重 = (1/vol_i) / sum(1/vol_j)，波动率越低的股票权重越高。

        Args:
            selected_stocks: 选股DataFrame，必须包含stock_code列。
            returns_data: 收益率DataFrame，index为日期，columns为stock_code。
                如果为None，回退到等权分配。

        Returns:
            添加了weight列的DataFrame。
        """
        df = selected_stocks.copy()
        n = len(df)
        if n == 0:
            df["weight"] = 0.0
            return df

        if returns_data is None:
            logger.warning("无收益率数据，risk_parity回退到等权分配")
            return self.allocate_equal(df)

        # 计算每只股票的年化波动率
        stock_codes = df["stock_code"].tolist()
        volatilities = []

        for code in stock_codes:
            if code in returns_data.columns:
                ret = returns_data[code].dropna()
                if len(ret) < 20:
                    # 数据不足，使用默认高波动率
                    vol = 0.5
                else:
                    # 年化波动率
                    vol = ret.std() * np.sqrt(252)
                    vol = max(vol, 0.001)  # 防止除零
            else:
                vol = 0.5  # 默认值
            volatilities.append(vol)

        volatilities = np.array(volatilities)
        inv_vol = 1.0 / volatilities
        weights = inv_vol / inv_vol.sum()

        df["weight"] = weights
        logger.info(
            f"风险平价权重: 波动率范围 [{volatilities.min():.4f}, {volatilities.max():.4f}]"
        )
        return df

    def _clip_and_normalize(self, stocks_df: pd.DataFrame) -> pd.DataFrame:
        """裁剪权重超过max_weight的项，并重新归一化。

        Args:
            stocks_df: 包含weight列的DataFrame。

        Returns:
            裁剪并归一化后的DataFrame。
        """
        df = stocks_df.copy()
        weights = df["weight"].values.copy()

        if self.max_weight is not None and self.max_weight > 0:
            # 迭代裁剪
            for _ in range(10):  # 最多10轮迭代
                excess = weights - self.max_weight
                over_idx = excess > 0
                if not over_idx.any():
                    break
                total_excess = excess[over_idx].sum()
                weights[over_idx] = self.max_weight
                under_idx = ~over_idx
                if under_idx.any() and weights[under_idx].sum() > 0:
                    weights[under_idx] += total_excess * (
                        weights[under_idx] / weights[under_idx].sum()
                    )

        # 归一化
        total = weights.sum()
        if total > 0:
            weights = weights / total

        # 应用最小权重阈值
        weights[weights < self.min_weight] = 0.0
        # 重新归一化
        total = weights.sum()
        if total > 0:
            weights = weights / total

        df["weight"] = weights
        return df


def compute_target_shares(
    stocks_df: pd.DataFrame,
    total_capital: float,
    current_prices: Dict[str, float],
) -> pd.DataFrame:
    """根据权重和总资金计算目标持仓股数。

    辅助函数，用于将权重转化为实际交易数量。

    Args:
        stocks_df: 包含stock_code和weight列的DataFrame。
        total_capital: 总可用资金。
        current_prices: {stock_code: price} 当前价格字典。

    Returns:
        添加了target_shares, target_amount列的DataFrame。
    """
    df = stocks_df.copy()
    shares_list = []
    amounts_list = []

    for _, row in df.iterrows():
        code = row["stock_code"]
        weight = row["weight"]
        price = current_prices.get(code, 0.0)
        target_amount = total_capital * weight

        if price > 0:
            # A股以100股（1手）为单位
            target_shares = int(target_amount / price / 100) * 100
        else:
            target_shares = 0

        shares_list.append(target_shares)
        amounts_list.append(target_amount)

    df["target_shares"] = shares_list
    df["target_amount"] = amounts_list
    return df