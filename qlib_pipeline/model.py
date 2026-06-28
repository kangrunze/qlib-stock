# -*- coding: utf-8 -*-
"""
Qlib Pipeline Model 适配层

本模块不再重复定义模型，改为从 model/ 统一导入。
qlib_pipeline 仅负责 Workflow 编排，模型定义统一在 model/ 目录管理。
"""

import sys
from pathlib import Path

# Ensure project root in path
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model import LightGBMModel, XGBoostModel, CatBoostModel, EnsembleModel, ModelRegistry  # noqa: E402, F401

# Qlib 原生模型快捷创建（用于 Qlib Workflow）
from qlib.utils import init_instance_by_config  # noqa: E402


def create_lgb_model(task: dict):
    """创建 LightGBM 模型实例（从 Qlib task 配置）"""
    model_cfg = task.get("model", {})
    return init_instance_by_config(model_cfg)


def create_rank_model(loss: str = "mse", **overrides) -> dict:
    """创建 LightGBM ranking 模型配置字典"""
    model_cfg = {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": loss,
            "colsample_bytree": 0.8879,
            "learning_rate": 0.0421,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 20,
        }
    }
    model_cfg["kwargs"].update(overrides)
    return model_cfg