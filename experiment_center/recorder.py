# -*- coding: utf-8 -*-
"""实验记录器 - 记录实验参数、指标、产物"""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ExperimentRecorder:
    """实验记录器 - 记录单次实验的所有信息"""
    
    def __init__(self, experiment_id: str, experiment_dir: Path):
        self.experiment_id = experiment_id
        self.experiment_dir = experiment_dir
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.params_file = self.experiment_dir / "params.json"
        self.metrics_file = self.experiment_dir / "metrics.json"
        self.artifacts_dir = self.experiment_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
    def log_params(self, **params):
        """记录参数"""
        existing = {}
        if self.params_file.exists():
            with open(self.params_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
                
        existing.update(params)
        
        with open(self.params_file, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
            
        logger.debug(f"记录参数: {list(params.keys())}")
        
    def log_metrics(self, **metrics):
        """记录指标"""
        existing = {}
        if self.metrics_file.exists():
            with open(self.metrics_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
                
        existing.update(metrics)
        existing["timestamp"] = datetime.now().isoformat()
        
        with open(self.metrics_file, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
            
        logger.debug(f"记录指标: {list(metrics.keys())}")
        
    def save_artifact(self, name: str, obj: Any, format: str = "pickle"):
        """保存产物
        
        Args:
            name: 产物名称
            obj: 对象
            format: 格式 (pickle/csv/json)
        """
        if format == "pickle":
            file_path = self.artifacts_dir / f"{name}.pkl"
            with open(file_path, "wb") as f:
                pickle.dump(obj, f)
        elif format == "csv":
            file_path = self.artifacts_dir / f"{name}.csv"
            if isinstance(obj, pd.DataFrame):
                obj.to_csv(file_path, index=False)
            else:
                pd.DataFrame(obj).to_csv(file_path, index=False)
        elif format == "json":
            file_path = self.artifacts_dir / f"{name}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"不支持的格式: {format}")
            
        logger.info(f"保存产物: {name} -> {file_path}")
        
    def load_artifact(self, name: str, format: str = "pickle") -> Any:
        """加载产物"""
        if format == "pickle":
            file_path = self.artifacts_dir / f"{name}.pkl"
            with open(file_path, "rb") as f:
                return pickle.load(f)
        elif format == "csv":
            file_path = self.artifacts_dir / f"{name}.csv"
            return pd.read_csv(file_path)
        elif format == "json":
            file_path = self.artifacts_dir / f"{name}.json"
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            raise ValueError(f"不支持的格式: {format}")
            
    def load_params(self) -> Dict:
        """加载参数"""
        if not self.params_file.exists():
            return {}
        with open(self.params_file, "r", encoding="utf-8") as f:
            return json.load(f)
            
    def load_metrics(self) -> Dict:
        """加载指标"""
        if not self.metrics_file.exists():
            return {}
        with open(self.metrics_file, "r", encoding="utf-8") as f:
            return json.load(f)
            
    def list_artifacts(self) -> List[str]:
        """列出所有产物"""
        artifacts = []
        for f in self.artifacts_dir.iterdir():
            artifacts.append(f.stem)
        return artifacts
