# -*- coding: utf-8 -*-
"""实验对比器 - 对比多个实验的结果"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ExperimentComparator:
    """实验对比器 - 对比多个实验"""
    
    def __init__(self, experiment_manager):
        self.manager = experiment_manager
        
    def compare_experiments(
        self,
        experiment_ids: List[str],
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """对比多个实验
        
        Args:
            experiment_ids: 实验ID列表
            metrics: 要对比的指标列表
            
        Returns:
            对比结果 DataFrame
        """
        results = []
        
        for exp_id in experiment_ids:
            exp = self.manager.get_experiment(exp_id)
            if not exp:
                logger.warning(f"实验不存在: {exp_id}")
                continue
                
            # 加载指标
            exp_dir = self.manager.get_experiment_dir(exp_id)
            metrics_file = exp_dir / "metrics.json"
            
            if metrics_file.exists():
                import json
                with open(metrics_file, "r", encoding="utf-8") as f:
                    exp_metrics = json.load(f)
            else:
                exp_metrics = {}
                
            result = {
                "experiment_id": exp_id,
                "name": exp.get("name", ""),
                "status": exp.get("status", ""),
                "created_at": exp.get("created_at", "")
            }
            result.update(exp_metrics)
            results.append(result)
            
        df = pd.DataFrame(results)
        
        # 过滤指标列
        if metrics and not df.empty:
            cols = ["experiment_id", "name", "status", "created_at"] + \
                   [m for m in metrics if m in df.columns]
            df = df[cols]
            
        return df
        
    def find_best_experiment(
        self,
        experiment_ids: List[str],
        metric: str = "sharpe",
        higher_is_better: bool = True
    ) -> Optional[Dict]:
        """找出最佳实验
        
        Args:
            experiment_ids: 实验ID列表
            metric: 评估指标
            higher_is_better: 是否越大越好
            
        Returns:
            最佳实验信息
        """
        df = self.compare_experiments(experiment_ids, [metric])
        
        if df.empty or metric not in df.columns:
            return None
            
        # 找出最佳
        if higher_is_better:
            best_idx = df[metric].idxmax()
        else:
            best_idx = df[metric].idxmin()
            
        best = df.loc[best_idx].to_dict()
        return best
        
    def generate_comparison_report(
        self,
        experiment_ids: List[str],
        output_file: str = "output/experiments/comparison_report.csv"
    ):
        """生成对比报告"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df = self.compare_experiments(experiment_ids)
        df.to_csv(output_path, index=False)
        
        logger.info(f"对比报告已保存到: {output_path}")
        return df
