# -*- coding: utf-8 -*-
"""风险管理器 - 控制组合风险暴露"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RiskManager:
    """风险管理器 - 控制投资组合的风险"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.max_position = self.config.get("max_position_pct", 0.10)
        self.max_industry = self.config.get("max_industry_pct", 0.30)
        self.stop_loss = self.config.get("stop_loss_pct", 0.08)
        self.max_drawdown = self.config.get("max_drawdown_pct", 0.20)
        
    def check_risk_limits(
        self,
        weights: pd.DataFrame,
        industry_mapping: Optional[Dict] = None
    ) -> pd.DataFrame:
        """检查并调整风险限制
        
        Args:
            weights: 原始权重
            industry_mapping: 股票行业映射 {stock: industry}
            
        Returns:
            调整后的权重
        """
        adjusted_weights = weights.copy()
        
        # 1. 限制单票最大仓位
        adjusted_weights = self._limit_position_size(adjusted_weights)
        
        # 2. 限制行业集中度
        if industry_mapping:
            adjusted_weights = self._limit_industry_exposure(adjusted_weights, industry_mapping)
        
        # 3. 确保权重归一化
        adjusted_weights = self._normalize_weights(adjusted_weights)
        
        logger.info("风险限制检查完成")
        return adjusted_weights
        
    def _limit_position_size(self, weights: pd.DataFrame) -> pd.DataFrame:
        """限制单票最大仓位"""
        return weights.clip(upper=self.max_position)
        
    def _limit_industry_exposure(
        self,
        weights: pd.DataFrame,
        industry_mapping: Dict
    ) -> pd.DataFrame:
        """限制行业集中度"""
        adjusted = weights.copy()
        
        for date in weights.index:
            date_weights = weights.loc[date]
            
            # 按行业分组
            industry_weights = {}
            for stock, weight in date_weights.items():
                if pd.isna(weight) or weight == 0:
                    continue
                industry = industry_mapping.get(stock, "unknown")
                if industry not in industry_weights:
                    industry_weights[industry] = 0
                industry_weights[industry] += weight
                
            # 检查并调整超限行业
            for industry, total_weight in industry_weights.items():
                if total_weight > self.max_industry:
                    # 按比例缩减该行业所有股票
                    scale_factor = self.max_industry / total_weight
                    for stock, weight in date_weights.items():
                        if pd.isna(weight) or weight == 0:
                            continue
                        if industry_mapping.get(stock) == industry:
                            adjusted.loc[date, stock] = weight * scale_factor
                            
        return adjusted
        
    def _normalize_weights(self, weights: pd.DataFrame) -> pd.DataFrame:
        """归一化权重"""
        row_sums = weights.sum(axis=1)
        return weights.div(row_sums, axis=0).fillna(0)
        
    def calculate_portfolio_risk(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
        cov_matrix: Optional[pd.DataFrame] = None
    ) -> Dict:
        """计算组合风险指标
        
        Args:
            weights: 权重
            returns: 历史收益率
            cov_matrix: 协方差矩阵（可选）
            
        Returns:
            风险指标字典
        """
        # 计算组合收益
        portfolio_returns = (weights.shift(1) * returns).sum(axis=1)
        
        # 波动率
        volatility = portfolio_returns.std() * np.sqrt(252)
        
        # VaR (95%)
        var_95 = np.percentile(portfolio_returns.dropna(), 5)
        
        # CVaR (Expected Shortfall)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        
        # 最大回撤
        cumulative = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = cumulative / rolling_max - 1
        max_drawdown = drawdown.min()
        
        # Beta（假设基准为等权市场）
        market_returns = returns.mean(axis=1)
        covariance = portfolio_returns.cov(market_returns)
        market_variance = market_returns.var()
        beta = covariance / market_variance if market_variance > 0 else 1.0
        
        result = {
            "volatility": volatility,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "max_drawdown": max_drawdown,
            "beta": beta,
            "daily_returns": portfolio_returns,
            "drawdown": drawdown
        }
        
        logger.info(f"风险计算完成: 波动率={volatility:.2%}, 最大回撤={max_drawdown:.2%}, Beta={beta:.2f}")
        return result
        
    def check_stop_loss(
        self,
        current_prices: pd.Series,
        entry_prices: pd.Series,
        weights: pd.Series
    ) -> pd.Series:
        """检查止损
        
        Args:
            current_prices: 当前价格
            entry_prices: 入场价格
            weights: 持仓权重
            
        Returns:
            调整后的权重（止损股票权重设为0）
        """
        returns = (current_prices - entry_prices) / entry_prices
        
        # 找出触发止损的股票
        stop_loss_mask = returns < -self.stop_loss
        
        adjusted_weights = weights.copy()
        adjusted_weights[stop_loss_mask] = 0
        
        # 重新归一化
        if adjusted_weights.sum() > 0:
            adjusted_weights = adjusted_weights / adjusted_weights.sum()
            
        n_stopped = stop_loss_mask.sum()
        if n_stopped > 0:
            logger.warning(f"触发止损: {n_stopped} 只股票")
            
        return adjusted_weights
        
    def generate_risk_report(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
        output_file: str = "output/portfolio/risk_report.csv"
    ) -> Dict:
        """生成风险报告"""
        from pathlib import Path
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        risk_metrics = self.calculate_portfolio_risk(weights, returns)
        
        # 计算每日风险指标
        daily_risk = pd.DataFrame({
            "date": risk_metrics["daily_returns"].index,
            "daily_return": risk_metrics["daily_returns"].values,
            "drawdown": risk_metrics["drawdown"].values,
            "rolling_vol_20d": risk_metrics["daily_returns"].rolling(20).std() * np.sqrt(252)
        })
        
        daily_risk.to_csv(output_path, index=False)
        
        logger.info(f"风险报告已保存到: {output_path}")
        
        return {
            "summary": {
                "volatility": risk_metrics["volatility"],
                "var_95": risk_metrics["var_95"],
                "cvar_95": risk_metrics["cvar_95"],
                "max_drawdown": risk_metrics["max_drawdown"],
                "beta": risk_metrics["beta"]
            },
            "daily_risk": daily_risk
        }
