# -*- coding: utf-8 -*-
"""因子评价器 - 计算因子IC、RankIC、PSI等指标"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEvaluator:
    """因子评价器 - 评估因子质量"""
    
    def __init__(self):
        self.evaluation_results = {}
        
    def calculate_ic(
        self,
        factor_values: pd.Series,
        forward_returns: pd.Series,
        method: str = "pearson"
    ) -> float:
        """计算IC值
        
        Args:
            factor_values: 因子值
            forward_returns: 未来收益
            method: pearson 或 spearman
            
        Returns:
            IC值
        """
        valid_mask = factor_values.notna() & forward_returns.notna()
        if valid_mask.sum() < 10:
            return np.nan
            
        factor = factor_values[valid_mask]
        returns = forward_returns[valid_mask]
        
        if method == "pearson":
            ic = factor.corr(returns)
        elif method == "spearman":
            ic = factor.corr(returns, method="spearman")
        else:
            raise ValueError(f"不支持的方法: {method}")
            
        return ic
        
    def calculate_rank_ic(
        self,
        factor_values: pd.Series,
        forward_returns: pd.Series
    ) -> float:
        """计算Rank IC"""
        return self.calculate_ic(factor_values, forward_returns, method="spearman")
        
    def calculate_ic_series(
        self,
        factor_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        method: str = "pearson"
    ) -> pd.Series:
        """计算逐期IC序列
        
        Args:
            factor_df: 因子值 DataFrame (index=date, columns=stocks)
            returns_df: 未来收益 DataFrame (index=date, columns=stocks)
            method: pearson 或 spearman
            
        Returns:
            IC序列
        """
        ic_series = []
        
        for date in factor_df.index:
            if date not in returns_df.index:
                continue
                
            factor_values = factor_df.loc[date]
            forward_returns = returns_df.loc[date]
            
            ic = self.calculate_ic(factor_values, forward_returns, method)
            ic_series.append({"date": date, "ic": ic})
            
        return pd.DataFrame(ic_series).set_index("date")["ic"]
        
    def calculate_icir(self, ic_series: pd.Series) -> float:
        """计算ICIR (IC的信息比率)"""
        if ic_series.empty or ic_series.isna().all():
            return 0.0
            
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        
        if ic_std == 0:
            return 0.0
            
        return ic_mean / ic_std
        
    def calculate_psi(
        self,
        expected: pd.Series,
        actual: pd.Series,
        bins: int = 10
    ) -> float:
        """计算PSI (群体稳定性指标)
        
        Args:
            expected: 期望分布（训练集）
            actual: 实际分布（测试集）
            bins: 分箱数
            
        Returns:
            PSI值
        """
        # 去除NaN
        expected = expected.dropna()
        actual = actual.dropna()
        
        if len(expected) == 0 or len(actual) == 0:
            return np.nan
            
        # 分箱
        breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        
        expected_counts = np.histogram(expected, bins=breakpoints)[0]
        actual_counts = np.histogram(actual, bins=breakpoints)[0]
        
        # 转换为比例
        expected_pct = expected_counts / len(expected)
        actual_pct = actual_counts / len(actual)
        
        # 避免除零
        expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
        actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)
        
        # 计算PSI
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        
        return psi
        
    def calculate_coverage(self, factor_values: pd.Series) -> float:
        """计算因子覆盖率"""
        if len(factor_values) == 0:
            return 0.0
        return factor_values.notna().mean()
        
    def calculate_nan_ratio(self, factor_values: pd.Series) -> float:
        """计算NaN比例"""
        if len(factor_values) == 0:
            return 1.0
        return factor_values.isna().mean()
        
    def evaluate_feature(
        self,
        feature_name: str,
        factor_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        train_dates: Optional[List] = None,
        test_dates: Optional[List] = None
    ) -> Dict:
        """全面评价因子
        
        Args:
            feature_name: 因子名称
            factor_df: 因子值 DataFrame
            returns_df: 未来收益 DataFrame
            train_dates: 训练集日期
            test_dates: 测试集日期
            
        Returns:
            评价结果字典
        """
        result = {
            "feature_name": feature_name,
            "coverage": {},
            "ic": {},
            "rank_ic": {},
            "icir": {},
            "psi": {}
        }
        
        # 计算覆盖率
        for col in factor_df.columns:
            result["coverage"][col] = self.calculate_coverage(factor_df[col])
            
        # 计算IC序列
        ic_series = self.calculate_ic_series(factor_df, returns_df, method="pearson")
        rank_ic_series = self.calculate_ic_series(factor_df, returns_df, method="spearman")
        
        # 总体IC统计
        result["ic"]["mean"] = ic_series.mean()
        result["ic"]["std"] = ic_series.std()
        result["ic"]["positive_ratio"] = (ic_series > 0).mean()
        
        result["rank_ic"]["mean"] = rank_ic_series.mean()
        result["rank_ic"]["std"] = rank_ic_series.std()
        result["rank_ic"]["positive_ratio"] = (rank_ic_series > 0).mean()
        
        # ICIR
        result["icir"]["icir"] = self.calculate_icir(ic_series)
        result["icir"]["rank_icir"] = self.calculate_icir(rank_ic_series)
        
        # PSI（如果有训练集和测试集）
        if train_dates is not None and test_dates is not None:
            for col in factor_df.columns:
                train_values = factor_df.loc[train_dates, col].dropna()
                test_values = factor_df.loc[test_dates, col].dropna()
                
                if len(train_values) > 0 and len(test_values) > 0:
                    psi = self.calculate_psi(train_values, test_values)
                    result["psi"][col] = psi
                    
        logger.info(f"因子评价完成: {feature_name}")
        logger.info(f"  IC均值: {result['ic']['mean']:.4f}, ICIR: {result['icir']['icir']:.4f}")
        
        return result
        
    def generate_feature_report(
        self,
        evaluations: List[Dict],
        output_file: str = "output/features/feature_report.csv"
    ) -> pd.DataFrame:
        """生成因子评价报告"""
        from pathlib import Path
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        rows = []
        for eval_result in evaluations:
            row = {
                "feature_name": eval_result["feature_name"],
                "ic_mean": eval_result["ic"]["mean"],
                "ic_std": eval_result["ic"]["std"],
                "ic_positive_ratio": eval_result["ic"]["positive_ratio"],
                "rank_ic_mean": eval_result["rank_ic"]["mean"],
                "rank_ic_std": eval_result["rank_ic"]["std"],
                "icir": eval_result["icir"]["icir"],
                "rank_icir": eval_result["icir"]["rank_icir"],
            }
            
            # 添加覆盖率平均值
            if eval_result["coverage"]:
                row["avg_coverage"] = np.mean(list(eval_result["coverage"].values()))
                
            # 添加PSI平均值
            if eval_result["psi"]:
                row["avg_psi"] = np.mean(list(eval_result["psi"].values()))
                
            rows.append(row)
            
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        
        logger.info(f"因子评价报告已保存到: {output_path}")
        return df
        
    def rank_features(
        self,
        evaluations: List[Dict],
        metric: str = "icir",
        top_k: int = 50
    ) -> List[str]:
        """因子排序
        
        Args:
            evaluations: 评价结果列表
            metric: 排序指标 (icir/ic_mean/rank_ic_mean)
            top_k: 返回前K个
            
        Returns:
            排序后的因子名称列表
        """
        scores = []
        for eval_result in evaluations:
            name = eval_result["feature_name"]
            
            if metric == "icir":
                score = eval_result["icir"]["icir"]
            elif metric == "ic_mean":
                score = abs(eval_result["ic"]["mean"])
            elif metric == "rank_ic_mean":
                score = abs(eval_result["rank_ic"]["mean"])
            else:
                raise ValueError(f"不支持的指标: {metric}")
                
            scores.append((name, score))
            
        # 排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return [name for name, _ in scores[:top_k]]
