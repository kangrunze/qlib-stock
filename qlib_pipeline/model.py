# -*- coding: utf-8 -*-
"""
Qlib Model Configuration - model.py

Supports LightGBM (regression/ranking) models.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def create_lgb_model(task: dict):
    """
    Create LightGBM model from task config.

    Args:
        task: task dict containing model config

    Returns:
        model instance
    """
    from qlib.utils import init_instance_by_config
    model_cfg = task.get("model", {})
    logger.info("创建模型: %s (%s)", model_cfg.get("class", "LGBModel"), model_cfg.get("module_path"))
    return init_instance_by_config(model_cfg)


def create_rank_model(loss: str = "mse", **overrides) -> dict:
    """
    Create LightGBM ranking model config.

    Args:
        loss: "mse" for regression, "rank" for ranking (pairwise)
        **overrides: override model kwargs

    Returns:
        model config dict (not instantiated)
    """
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


def create_xgb_model(**overrides) -> dict:
    """Create XGBoost model config."""
    model_cfg = {
        "class": "XGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse",
            "colsample_bytree": 0.8879,
            "learning_rate": 0.0421,
            "max_depth": 8,
            "n_estimators": 1000,
            "subsample": 0.8789,
            "num_threads": 20,
        }
    }
    model_cfg["kwargs"].update(overrides)
    return model_cfg
