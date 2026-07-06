# -*- coding: utf-8 -*-
"""
Qlib Pipeline Model 适配层

提供 Qlib Workflow 模型配置的快捷创建函数。
qlib_pipeline 仅负责 Workflow 编排，模型定义统一在 model/ 目录管理。
"""

from qlib.utils import init_instance_by_config


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