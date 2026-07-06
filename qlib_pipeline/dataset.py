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



