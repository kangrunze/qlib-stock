# -*- coding: utf-8 -*-
"""因子注册器 - 自动注册和管理因子"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class FeatureRegistry:
    """因子注册器 - 管理所有因子定义"""
    
    def __init__(self, output_dir: str = "output/features"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.output_dir / "feature_registry.json"
        self.registry = self._load_registry()
        
    def register_feature(
        self,
        name: str,
        category: str,
        description: str,
        compute_func: Optional[Callable] = None,
        params: Optional[Dict] = None,
        tags: Optional[List[str]] = None
    ):
        """注册因子
        
        Args:
            name: 因子名称
            category: 因子类别 (trend/momentum/volatility/volume_price/statistical)
            description: 因子描述
            compute_func: 计算函数（可选）
            params: 参数配置
            tags: 标签
        """
        feature_def = {
            "name": name,
            "category": category,
            "description": description,
            "params": params or {},
            "tags": tags or [],
            "registered_at": datetime.now().isoformat(),
            "has_compute_func": compute_func is not None
        }
        
        self.registry[name] = feature_def
        self._save_registry()
        
        logger.info(f"注册因子: {name} ({category})")
        
    def get_feature(self, name: str) -> Optional[Dict]:
        """获取因子定义"""
        return self.registry.get(name)
        
    def list_features(
        self,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """列出所有因子"""
        df = pd.DataFrame(self.registry.values())
        
        if df.empty:
            return df
            
        if category:
            df = df[df["category"] == category]
        if tags:
            df = df[df["tags"].apply(lambda x: any(t in x for t in tags))]
            
        return df
        
    def unregister_feature(self, name: str):
        """注销因子"""
        if name in self.registry:
            del self.registry[name]
            self._save_registry()
            logger.info(f"注销因子: {name}")
            
    def _load_registry(self) -> Dict:
        """加载注册表"""
        if not self.registry_file.exists():
            return {}
        with open(self.registry_file, "r", encoding="utf-8") as f:
            return json.load(f)
            
    def _save_registry(self):
        """保存注册表"""
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, ensure_ascii=False, indent=2)
            
    def register_builtin_features(self):
        """注册内置因子"""
        builtin_features = [
            # 趋势类
            ("MA5", "trend", "5日均线", {"window": 5}),
            ("MA10", "trend", "10日均线", {"window": 10}),
            ("MA20", "trend", "20日均线", {"window": 20}),
            ("MA60", "trend", "60日均线", {"window": 60}),
            ("MACD", "trend", "MACD指标", {"fast": 12, "slow": 26, "signal": 9}),
            ("ADX", "trend", "平均趋向指标", {"window": 14}),
            
            # 动量类
            ("RSI", "momentum", "相对强弱指标", {"window": 14}),
            ("KDJ", "momentum", "随机指标", {"n": 9, "m1": 3, "m2": 3}),
            ("MOM", "momentum", "动量指标", {"window": 10}),
            ("ROC", "momentum", "变化率指标", {"window": 12}),
            
            # 波动类
            ("ATR", "volatility", "平均真实波幅", {"window": 14}),
            ("BOLL_WIDTH", "volatility", "布林带宽度", {"window": 20, "nbdev": 2}),
            ("HIST_VOL", "volatility", "历史波动率", {"window": 20}),
            
            # 量价类
            ("OBV", "volume_price", "能量潮指标", {}),
            ("MFI", "volume_price", "资金流量指标", {"window": 14}),
            ("VWAP", "volume_price", "成交量加权平均价", {}),
            ("TURNOVER", "volume_price", "换手率", {}),
            
            # 统计类
            ("SKEW", "statistical", "偏度", {"window": 20}),
            ("KURT", "statistical", "峰度", {"window": 20}),
            ("BETA", "statistical", "Beta系数", {"window": 60}),
        ]
        
        for name, category, desc, params in builtin_features:
            self.register_feature(
                name=name,
                category=category,
                description=desc,
                params=params,
                tags=["builtin"]
            )
            
        logger.info(f"注册了 {len(builtin_features)} 个内置因子")
