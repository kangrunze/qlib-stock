# -*- coding: utf-8 -*-
"""
Qlib Training Pipeline - train.py

完整训练流程（Qlib 标准 Workflow）：
  1. Init Qlib with data
  2. Create dataset (Alpha158/Alpha360)
  3. Train model (LightGBM via Qlib LGBModel)
  4. Save model + metadata
  5. (Optional) Run prediction and backtest

模型定义统一在 model/ 目录，本模块仅负责 Workflow 编排。
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# MUST import numpy_compat BEFORE qlib
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from qlib_pipeline.numpy_compat import *  # noqa - must be before qlib import

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord

from qlib_pipeline.dataset import load_workflow_config
from qlib_pipeline.model import create_rank_model  # Qlib LGB shortcut

logger = logging.getLogger(__name__)


def build_task(config: dict) -> dict:
    """
    Build complete Qlib task config from workflow config.

    从统一配置构建 Qlib task dict（包含 model + dataset），
    模型定义从 model/ 包导入，配置从 workflow_config.yaml 读取。

    Args:
        config: workflow config dict

    Returns:
        task dict for Qlib workflow
    """
    handler = config.get("dataset", {}).get("handler", "Alpha158")
    handler_cfg = config.get("data_handler", {}).copy()

    # Data handler
    handler_cfg = {
        "class": handler,
        "module_path": "qlib.contrib.data.handler",
        "kwargs": handler_cfg,
    }

    # Segments
    segments = config.get("dataset", {}).get("segments", {})

    # Model — 使用 qlib_lgb 配置（Qlib 原生 LGBModel）
    model_cfg = config.get("qlib_lgb", create_rank_model())

    task = {
        "model": {
            "class": model_cfg.get("class", "LGBModel"),
            "module_path": model_cfg.get("module_path", "qlib.contrib.model.gbdt"),
            "kwargs": model_cfg.get("kwargs", {}),
        },
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": handler_cfg,
                "segments": segments,
            },
        },
    }

    return task


def init_qlib_env(config: dict):
    """Initialize Qlib with Windows-compatible single-thread mode."""
    import os
    os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    provider_uri = config.get("qlib", {}).get("provider_uri", "d:/project/qlib-stock/qlib_data/cn_data")
    logger.info("初始化 Qlib: %s", provider_uri)
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    from qlib.config import C
    C.joblib_backend = "threading"
    C.maxtasksperchild = None
    C.dataset_process_n_worker = 1
    C.min_data_shift = 1


def run_train(config_path: str = None, config: dict = None,
              experiment_name: str = "qlib_train", **overrides):
    """
    Run complete training pipeline.

    Args:
        config_path: path to workflow_config.yaml
        config: pre-loaded config dict (takes precedence)
        experiment_name: mlflow experiment name
        **overrides: override config values (e.g. handler="Alpha360")

    Returns:
        tuple (model, dataset, recorder_id)
    """
    if config is None:
        config = load_workflow_config(config_path)

    # Apply overrides
    for k, v in overrides.items():
        if "." in k:
            parts = k.split(".")
            target = config
            for p in parts[:-1]:
                target = target.setdefault(p, {})
            target[parts[-1]] = v
        else:
            config[k] = v

    init_qlib_env(config)

    # Build task
    task = build_task(config)
    handler_type = task["dataset"]["kwargs"]["handler"]["class"]
    logger.info("Task: handler=%s, model=%s, segments=%s",
                handler_type, task["model"]["class"],
                task["dataset"]["kwargs"]["segments"])

    # Create dataset
    dataset = init_instance_by_config(task["dataset"])
    logger.info("Dataset 创建完成")

    # Create and train model
    model = init_instance_by_config(task["model"])
    logger.info("开始训练...")

    with R.start(experiment_name=experiment_name):
        R.log_params(**flatten_dict(task))
        R.log_params(handler_type=handler_type)
        model.fit(dataset)
        R.save_objects(trained_model=model)
        rid = R.get_recorder().id
        logger.info("训练完成, recorder_id=%s", rid)

    return model, dataset, rid


def run_prediction(model, dataset, experiment_name: str = "qlib_pred",
                   train_rid: str = None):
    """Run prediction and generate signal records."""
    logger.info("运行预测...")
    with R.start(experiment_name=experiment_name):
        if train_rid:
            recorder = R.get_recorder(recorder_id=train_rid)
            model = recorder.load_object("trained_model")
        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()
        ba_rid = recorder.id
        logger.info("预测完成, ba_rid=%s", ba_rid)
    return ba_rid


def run_backtest(model, dataset, config: dict = None,
                 train_rid: str = None):
    """Run backtest and generate analysis."""
    config = config or load_workflow_config()
    port_config = config.get("backtest", {})

    logger.info("运行回测: %s ~ %s",
                port_config.get("backtest", {}).get("start_time"),
                port_config.get("backtest", {}).get("end_time"))

    with R.start(experiment_name="backtest_analysis"):
        if train_rid:
            recorder = R.get_recorder(recorder_id=train_rid, experiment_name="qlib_train")
            model = recorder.load_object("trained_model")
        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()
        par = PortAnaRecord(recorder, port_config, "day")
        par.generate()
        ba_rid = recorder.id

    recorder = R.get_recorder(recorder_id=ba_rid, experiment_name="backtest_analysis")
    pred_df = recorder.load_object("pred.pkl")
    report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

    logger.info("回测完成")
    return pred_df, report_normal_df, analysis_df