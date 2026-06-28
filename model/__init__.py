# -*- coding: utf-8 -*-
"""
Model Layer - 统一模型入口

所有模型实现统一从此包导入，qlib_pipeline 仅负责调用，不重复定义模型。

Usage:
    from model import LightGBMModel, XGBoostModel, CatBoostModel
    from model import ModelEnsemble, ModelRegistry, OptunaTuner
"""

from .base_model import BaseModel
from .lgb_model import LightGBMModel
from .xgb_model import XGBoostModel
from .cat_model import CatBoostModel
from .ensemble import EnsembleModel
from .model_registry import ModelRegistry
from .optuna_tuner import OptunaTuner
from .shap_analysis import SHAPAnalyzer

__all__ = [
    "BaseModel",
    "LightGBMModel",
    "XGBoostModel",
    "CatBoostModel",
    "EnsembleModel",
    "ModelRegistry",
    "OptunaTuner",
    "SHAPAnalyzer",
]