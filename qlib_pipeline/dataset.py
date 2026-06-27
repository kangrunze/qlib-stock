# -*- coding: utf-8 -*-
"""
Qlib Dataset Configuration - dataset.py

Creates Qlib DatasetH with Alpha158 or Alpha360 handlers.
Uses standard train/valid/test time splits with 20-day return as primary label.
"""

import logging
from typing import Dict, Optional

import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent


def load_workflow_config(config_path: Optional[str] = None) -> dict:
    """Load workflow configuration from YAML."""
    if config_path is None:
        config_path = _PROJECT_ROOT / "qlib_pipeline" / "workflow_config.yaml"
    if not Path(config_path).exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_dataset(handler_type: str = "Alpha158", config: Optional[dict] = None, **kwargs):
    """
    Create Qlib DatasetH with specified handler.

    Args:
        handler_type: "Alpha158" or "Alpha360"
        config: workflow config dict, loaded from workflow_config.yaml if None
        **kwargs: override config values

    Returns:
        qlib.data.dataset.DatasetH instance
    """
    import qlib
    from qlib.utils import init_instance_by_config

    cfg = config or load_workflow_config()

    # Merge kwargs overrides
    for k, v in kwargs.items():
        if "." in k:
            parts = k.split(".")
            target = cfg
            for p in parts[:-1]:
                target = target.setdefault(p, {})
            target[parts[-1]] = v
        else:
            cfg[k] = v

    dataset_cfg = cfg.get("dataset", {})
    dataset_cfg["kwargs"]["handler"]["class"] = handler_type
    dataset_cfg["kwargs"]["handler"]["module_path"] = "qlib.contrib.data.handler"

    # Update handler kwargs from config
    handler_overrides = cfg.get("data_handler", {})
    if handler_overrides:
        dataset_cfg["kwargs"]["handler"]["kwargs"].update(handler_overrides)

    logger.info("创建 Dataset: handler=%s, segments=%s", handler_type, dataset_cfg["kwargs"]["segments"])

    return init_instance_by_config(dataset_cfg)


def create_dataset_from_task(task: dict):
    """Create dataset directly from a task config dict."""
    from qlib.utils import init_instance_by_config
    return init_instance_by_config(task["dataset"])
