# -*- coding: utf-8 -*-
"""组合优化器 - 支持多种优化方法"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """组合优化器 - 生成最优投资组合"""
    
    def __init__(self):
        self.optimization_results = {}
        
    def optimize(
        self,
        predictions: pd.DataFrame,
        method: str = "equal_weight",
        constraints: Optional[Dict] = None
    ) -> pd.DataFrame:
        """优化投资组合
        
        Args:
            predictions: 预测结果 (index=date, columns=stocks, values=scores)
            method: 优化方法
                - equal_weight: 等权重
                - score_weighted: 得分加权
                - mean_variance: 均值方差优化
                - risk_parity: 风险平价
            constraints: 约束条件
            
        Returns:
            权重 DataFrame (index=date, columns=stocks, values=weights)
        """
        constraints = constraints or {}
        
        if method == "equal_weight":
            weights = self._equal_weight(predictions, constraints)
        elif method == "score_weighted":
            weights = self._score_weighted(predictions, constraints)
        elif method == "mean_variance":
            weights = self._mean_variance(predictions, constraints)
        elif method == "risk_parity":
            weights = self._risk_parity(predictions, constraints)
        else:
            raise ValueError(f"不支持的优化方法: {method}")
            
        logger.info(f"组合优化完成: method={method}, 日期数={len(weights)}")
        return weights
        
    def _equal_weight(
        self,
        predictions: pd.DataFrame,
        constraints: Dict
    ) -> pd.DataFrame:
        """等权重优化"""
        top_k = constraints.get("top_k", 30)
        max_weight = constraints.get("max_weight", 0.1)
        
        weights = pd.DataFrame(index=predictions.index, columns=predictions.columns, dtype=float)
        
        for date in predictions.index:
            scores = predictions.loc[date].dropna().sort_values(ascending=False)
            top_stocks = scores.head(top_k).index.tolist()
            
            # 等权重
            weight = min(1.0 / len(top_stocks), max_weight)
            weights.loc[date, top_stocks] = weight
            
        return weights.fillna(0)
        
    def _score_weighted(
        self,
        predictions: pd.DataFrame,
        constraints: Dict
    ) -> pd.DataFrame:
        """得分加权优化"""
        top_k = constraints.get("top_k", 30)
        max_weight = constraints.get("max_weight", 0.1)
        
        weights = pd.DataFrame(index=predictions.index, columns=predictions.columns, dtype=float)
        
        for date in predictions.index:
            scores = predictions.loc[date].dropna().sort_values(ascending=False)
            top_stocks = scores.head(top_k)
            
            # 按得分加权
            total_score = top_stocks.sum()
            if total_score > 0:
                stock_weights = top_stocks / total_score
                # 限制最大权重
                stock_weights = stock_weights.clip(upper=max_weight)
                # 重新归一化
                stock_weights = stock_weights / stock_weights.sum()
                weights.loc[date, top_stocks.index] = stock_weights.values
            
        return weights.fillna(0)
        
    def _mean_variance(
        self,
        predictions: pd.DataFrame,
        constraints: Dict
    ) -> pd.DataFrame:
        """均值方差优化（简化版）"""
        top_k = constraints.get("top_k", 30)
        max_weight = constraints.get("max_weight", 0.1)
        risk_aversion = constraints.get("risk_aversion", 1.0)
        
        weights = pd.DataFrame(index=predictions.index, columns=predictions.columns, dtype=float)
        
        for date in predictions.index:
            scores = predictions.loc[date].dropna().sort_values(ascending=False)
            top_stocks = scores.head(top_k).index.tolist()
            
            # 简化版：使用得分作为预期收益的代理
            expected_returns = scores.head(top_k).values
            
            # 假设协方差矩阵为单位矩阵（简化）
            # 实际应用中应该使用历史收益协方差矩阵
            n = len(top_stocks)
            cov_matrix = np.eye(n) * 0.01
            
            # 均值方差优化公式: w = (1/λ) * Σ^(-1) * μ
            try:
                cov_inv = np.linalg.inv(cov_matrix)
                raw_weights = cov_inv @ expected_returns / risk_aversion
                
                # 归一化并限制权重
                raw_weights = np.maximum(raw_weights, 0)  # 禁止做空
                raw_weights = np.minimum(raw_weights, max_weight)
                raw_weights = raw_weights / raw_weights.sum()
                
                weights.loc[date, top_stocks] = raw_weights
            except Exception as e:
                logger.warning(f"均值方差优化失败 {date}: {e}，使用等权重")
                weights.loc[date, top_stocks] = 1.0 / n
                
        return weights.fillna(0)
        
    def _risk_parity(
        self,
        predictions: pd.DataFrame,
        constraints: Dict
    ) -> pd.DataFrame:
        """风险平价优化（简化版）"""
        top_k = constraints.get("top_k", 30)
        max_weight = constraints.get("max_weight", 0.1)
        
        weights = pd.DataFrame(index=predictions.index, columns=predictions.columns, dtype=float)
        
        for date in predictions.index:
            scores = predictions.loc[date].dropna().sort_values(ascending=False)
            top_stocks = scores.head(top_k).index.tolist()
            
            # 简化版：假设所有股票波动率相同，退化为等权重
            # 实际应用中应该使用历史波动率
            n = len(top_stocks)
            weight = min(1.0 / n, max_weight)
            weights.loc[date, top_stocks] = weight
            
        return weights.fillna(0)
        
    def backtest_portfolio(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
        transaction_cost: float = 0.001
    ) -> Dict:
        """回测投资组合
        
        Args:
            weights: 权重 DataFrame
            returns: 收益率 DataFrame
            transaction_cost: 交易成本
            
        Returns:
            回测结果字典
        """
        # 计算每日收益
        daily_returns = (weights.shift(1) * returns).sum(axis=1)
        
        # 扣除交易成本（简化版：每次调仓扣除固定成本）
        turnover = weights.diff().abs().sum(axis=1)
        costs = turnover * transaction_cost
        daily_returns = daily_returns - costs
        
        # 计算累计收益
        cumulative_returns = (1 + daily_returns).cumprod()
        
        # 计算指标
        total_return = cumulative_returns.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(daily_returns)) - 1
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe = annual_return / volatility if volatility > 0 else 0
        
        # 计算最大回撤
        rolling_max = cumulative_returns.expanding().max()
        drawdown = cumulative_returns / rolling_max - 1
        max_drawdown = drawdown.min()
        
        result = {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "cumulative_returns": cumulative_returns,
            "daily_returns": daily_returns,
            "drawdown": drawdown
        }
        
        logger.info(f"组合回测完成: 总收益={total_return:.2%}, 年化={annual_return:.2%}, 夏普={sharpe:.2f}")
        return result
