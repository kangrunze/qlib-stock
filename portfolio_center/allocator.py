# -*- coding: utf-8 -*-
"""权重分配器 - 多种权重分配方法"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class WeightAllocator:
    """权重分配器 - 分配投资组合权重"""
    
    def __init__(self, method: str = "equal"):
        """
        Args:
            method: 分配方法
                - equal: 等权重
                - score_weighted: 得分加权
                - inverse_volatility: 波动率倒数加权
                - risk_parity: 风险平价
        """
        self.method = method
        
    def allocate(
        self,
        predictions: pd.DataFrame,
        volatility: Optional[pd.DataFrame] = None,
        constraints: Optional[Dict] = None
    ) -> pd.DataFrame:
        """分配权重
        
        Args:
            predictions: 预测得分
            volatility: 波动率数据（可选）
            constraints: 约束条件
            
        Returns:
            权重 DataFrame
        """
        constraints = constraints or {}
        
        if self.method == "equal":
            weights = self._equal_weight(predictions, constraints)
        elif self.method == "score_weighted":
            weights = self._score_weighted(predictions, constraints)
        elif self.method == "inverse_volatility":
            if volatility is None:
                logger.warning("波动率数据缺失，使用等权重")
                weights = self._equal_weight(predictions, constraints)
            else:
                weights = self._inverse_volatility(predictions, volatility, constraints)
        elif self.method == "risk_parity":
            weights = self._risk_parity(predictions, volatility, constraints)
        else:
            raise ValueError(f"不支持的分配方法: {self.method}")
            
        return weights
        
    def _equal_weight(
        self,
        predictions: pd.DataFrame,
        constraints: Dict
    ) -> pd.DataFrame:
        """等权重分配"""
        top_k = constraints.get("top_k", 30)
        max_weight = constraints.get("max_weight", 0.1)
        
        weights = pd.DataFrame(0.0, index=predictions.index, columns=predictions.columns)
        
        for date in predictions.index:
            scores = predictions.loc[date].dropna().sort_values(ascending=False)
            top_stocks = scores.head(top_k).index.tolist()
            
            weight = min(1.0 / len(top_stocks), max_weight)
            weights.loc[date, top_stocks] = weight
            
        return weights
        
    def _score_weighted(
        self,
        predictions: pd.DataFrame,
        constraints: Dict
    ) -> pd.DataFrame:
        """得分加权分配"""
        top_k = constraints.get("top_k", 30)
        max_weight = constraints.get("max_weight", 0.1)
        
        weights = pd.DataFrame(0.0, index=predictions.index, columns=predictions.columns)
        
        for date in predictions.index:
            scores = predictions.loc[date].dropna().sort_values(ascending=False)
            top_scores = scores.head(top_k)
            
            # 按得分加权
            total_score = top_scores.sum()
            if total_score > 0:
                stock_weights = top_scores / total_score
                stock_weights = stock_weights.clip(upper=max_weight)
                stock_weights = stock_weights / stock_weights.sum()
                weights.loc[date, top_scores.index] = stock_weights.values
            
        return weights
        
    def _inverse_volatility(
        self,
        predictions: pd.DataFrame,
        volatility: pd.DataFrame,
        constraints: Dict
    ) -> pd.DataFrame:
        """波动率倒数加权"""
        top_k = constraints.get("top_k", 30)
        max_weight = constraints.get("max_weight", 0.1)
        
        weights = pd.DataFrame(0.0, index=predictions.index, columns=predictions.columns)
        
        for date in predictions.index:
            scores = predictions.loc[date].dropna().sort_values(ascending=False)
            top_stocks = scores.head(top_k).index.tolist()
            
            # 获取波动率
            if date in volatility.index:
                vols = volatility.loc[date, top_stocks]
            else:
                vols = volatility.iloc[-1][top_stocks]
                
            # 波动率倒数加权
            inv_vol = 1.0 / vols.replace(0, np.nan)
            inv_vol = inv_vol.fillna(inv_vol.mean())
            
            total_inv_vol = inv_vol.sum()
            if total_inv_vol > 0:
                stock_weights = inv_vol / total_inv_vol
                stock_weights = stock_weights.clip(upper=max_weight)
                stock_weights = stock_weights / stock_weights.sum()
                weights.loc[date, top_stocks] = stock_weights.values
            
        return weights
        
    def _risk_parity(
        self,
        predictions: pd.DataFrame,
        volatility: Optional[pd.DataFrame],
        constraints: Dict
    ) -> pd.DataFrame:
        """风险平价分配"""
        top_k = constraints.get("top_k", 30)
        max_weight = constraints.get("max_weight", 0.1)
        
        weights = pd.DataFrame(0.0, index=predictions.index, columns=predictions.columns)
        
        for date in predictions.index:
            scores = predictions.loc[date].dropna().sort_values(ascending=False)
            top_stocks = scores.head(top_k).index.tolist()
            
            # 如果有波动率数据，使用波动率倒数
            if volatility is not None and date in volatility.index:
                vols = volatility.loc[date, top_stocks]
                inv_vol = 1.0 / vols.replace(0, np.nan)
                inv_vol = inv_vol.fillna(inv_vol.mean())
                stock_weights = inv_vol / inv_vol.sum()
            else:
                # 否则等权重
                stock_weights = pd.Series(1.0 / len(top_stocks), index=top_stocks)
                
            stock_weights = stock_weights.clip(upper=max_weight)
            stock_weights = stock_weights / stock_weights.sum()
            weights.loc[date, top_stocks] = stock_weights.values
            
        return weights
        
    def apply_industry_neutral(
        self,
        weights: pd.DataFrame,
        industry_mapping: Dict,
        benchmark_industry_weights: Optional[Dict] = None
    ) -> pd.DataFrame:
        """应用行业中性化
        
        Args:
            weights: 原始权重
            industry_mapping: 股票行业映射
            benchmark_industry_weights: 基准行业权重
            
        Returns:
            行业中性化后的权重
        """
        neutral_weights = weights.copy()
        
        for date in weights.index:
            date_weights = weights.loc[date]
            
            # 按行业分组
            industry_stocks = {}
            for stock, weight in date_weights.items():
                if pd.isna(weight) or weight == 0:
                    continue
                industry = industry_mapping.get(stock, "unknown")
                if industry not in industry_stocks:
                    industry_stocks[industry] = []
                industry_stocks[industry].append(stock)
                
            # 调整每个行业的权重
            for industry, stocks in industry_stocks.items():
                industry_total = date_weights[stocks].sum()
                
                # 目标行业权重
                if benchmark_industry_weights and industry in benchmark_industry_weights:
                    target_weight = benchmark_industry_weights[industry]
                else:
                    target_weight = industry_total  # 保持不变
                    
                # 按比例调整行业内股票
                if industry_total > 0:
                    scale_factor = target_weight / industry_total
                    for stock in stocks:
                        neutral_weights.loc[date, stock] *= scale_factor
                        
        return neutral_weights
