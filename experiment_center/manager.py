# -*- coding: utf-8 -*-
"""实验管理器 - 管理实验生命周期"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ExperimentManager:
    """实验管理器 - 管理所有实验"""
    
    def __init__(self, output_dir: str = "output/experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_file = self.output_dir / "experiments.jsonl"
        self.experiment_dirs = self.output_dir / "runs"
        self.experiment_dirs.mkdir(parents=True, exist_ok=True)
        
    def create_experiment(
        self,
        name: str,
        config: Dict,
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> str:
        """创建新实验
        
        Returns:
            experiment_id
        """
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}"
        
        experiment = {
            "experiment_id": experiment_id,
            "name": name,
            "description": description,
            "tags": tags or [],
            "config": config,
            "created_at": datetime.now().isoformat(),
            "status": "created"
        }
        
        # 保存到 JSONL
        with open(self.experiments_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(experiment, ensure_ascii=False) + "\n")
            
        # 创建实验目录
        exp_dir = self.experiment_dirs / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        config_file = exp_dir / "config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            
        logger.info(f"创建实验: {experiment_id}")
        return experiment_id
        
    def update_experiment_status(
        self,
        experiment_id: str,
        status: str,
        metrics: Optional[Dict] = None
    ):
        """更新实验状态"""
        experiments = self._load_experiments()
        
        for exp in experiments:
            if exp["experiment_id"] == experiment_id:
                exp["status"] = status
                exp["updated_at"] = datetime.now().isoformat()
                if metrics:
                    exp["metrics"] = metrics
                    
        # 重写文件
        with open(self.experiments_file, "w", encoding="utf-8") as f:
            for exp in experiments:
                f.write(json.dumps(exp, ensure_ascii=False) + "\n")
                
        logger.info(f"更新实验状态: {experiment_id} -> {status}")
        
    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        """获取实验详情"""
        experiments = self._load_experiments()
        for exp in experiments:
            if exp["experiment_id"] == experiment_id:
                return exp
        return None
        
    def list_experiments(
        self,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 50
    ) -> pd.DataFrame:
        """列出实验"""
        experiments = self._load_experiments()
        df = pd.DataFrame(experiments)
        
        if df.empty:
            return df
            
        # 过滤
        if status:
            df = df[df["status"] == status]
        if tags:
            df = df[df["tags"].apply(lambda x: any(t in x for t in tags))]
            
        return df.head(limit)
        
    def _load_experiments(self) -> List[Dict]:
        """加载所有实验"""
        if not self.experiments_file.exists():
            return []
            
        experiments = []
        with open(self.experiments_file, "r", encoding="utf-8") as f:
            for line in f:
                experiments.append(json.loads(line.strip()))
                
        return experiments
        
    def get_experiment_dir(self, experiment_id: str) -> Path:
        """获取实验目录"""
        return self.experiment_dirs / experiment_id
        
    def delete_experiment(self, experiment_id: str):
        """删除实验"""
        import shutil
        
        experiments = self._load_experiments()
        experiments = [e for e in experiments if e["experiment_id"] != experiment_id]
        
        with open(self.experiments_file, "w", encoding="utf-8") as f:
            for exp in experiments:
                f.write(json.dumps(exp, ensure_ascii=False) + "\n")
                
        # 删除目录
        exp_dir = self.experiment_dirs / experiment_id
        if exp_dir.exists():
            shutil.rmtree(exp_dir)
            
        logger.info(f"删除实验: {experiment_id}")
