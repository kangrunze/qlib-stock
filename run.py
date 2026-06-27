#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A股量化选股系统 - Qlib Pipeline 统一入口

一键跑通完整流程：
  数据 → Alpha158/Alpha360 特征 → LightGBM 训练 → 信号预测 → 回测分析 → 图表

Usage:
  # 一键跑通完整流程 (训练+回测+图表)
  python run.py full

  # 只训练
  python run.py train

  # 只回测（需已有训练结果）
  python run.py backtest --rid <recorder_id>

  # 数据管理
  python run.py data --download          # 下载 AKShare 数据
  python run.py data --convert           # CSV 转 Qlib bin
  python run.py data --check             # 检查数据完整性

  # 参数覆盖
  python run.py train --handler Alpha360 --loss rank --topk 30
  python run.py full --handler Alpha158 --loss mse --topk 50
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# 项目根目录
_PROJECT_ROOT = Path(__file__).parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# MUST patch numpy BEFORE importing qlib
from qlib_pipeline.numpy_compat import *  # noqa

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R

from qlib_pipeline.dataset import load_workflow_config
from qlib_pipeline.train import build_task, run_train
from qlib_pipeline.backtest import generate_report_charts, print_summary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)5s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run")


def parse_args():
    parser = argparse.ArgumentParser(description="Qlib Pipeline 统一入口")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # === full: 一键跑通 ===
    full_parser = subparsers.add_parser("full", help="一键跑通: 训练+回测+图表")
    full_parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    full_parser.add_argument("--handler", type=str, default=None, choices=["Alpha158", "Alpha360"],
                        help="特征处理器: Alpha158 或 Alpha360")
    full_parser.add_argument("--model-type", type=str, default=None, choices=["lgb", "xgb"],
                        help="模型类型: lgb 或 xgb")
    full_parser.add_argument("--loss", type=str, default=None, choices=["mse", "rank"],
                        help="损失函数: mse(回归) 或 rank(排序)")
    full_parser.add_argument("--topk", type=int, default=None, help="选股数量")
    full_parser.add_argument("--experiment", type=str, default="qlib_pipeline", help="实验名称")
    full_parser.add_argument("--output-dir", type=str, default="output/qlib_charts", help="图表输出目录")

    # === train: 仅训练 ===
    train_parser = subparsers.add_parser("train", help="仅训练模型")
    train_parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    train_parser.add_argument("--handler", type=str, default=None, choices=["Alpha158", "Alpha360"],
                        help="特征处理器: Alpha158 或 Alpha360")
    train_parser.add_argument("--model-type", type=str, default=None, choices=["lgb", "xgb"],
                        help="模型类型: lgb 或 xgb")
    train_parser.add_argument("--loss", type=str, default=None, choices=["mse", "rank"],
                        help="损失函数: mse(回归) 或 rank(排序)")
    train_parser.add_argument("--experiment", type=str, default="qlib_train", help="实验名称")

    # === backtest: 仅回测 ===
    bt_parser = subparsers.add_parser("backtest", help="仅回测 (需已有训练结果)")
    bt_parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    bt_parser.add_argument("--rid", type=str, required=True, help="训练 recorder_id")
    bt_parser.add_argument("--experiment", type=str, default="qlib_train", help="训练实验名称")
    bt_parser.add_argument("--output-dir", type=str, default="output/qlib_charts", help="图表输出目录")
    bt_parser.add_argument("--topk", type=int, default=None, help="选股数量")

    # === data: 数据管理 ===
    data_parser = subparsers.add_parser("data", help="数据下载/转换/检查")
    data_parser.add_argument("--download", action="store_true", help="下载 AKShare 数据到 Parquet")
    data_parser.add_argument("--convert", action="store_true", help="CSV 转 Qlib bin 格式")
    data_parser.add_argument("--check", action="store_true", help="检查 Qlib 数据完整性")
    data_parser.add_argument("--sample", type=int, default=None, help="仅处理前N只股票")
    data_parser.add_argument("--csv-dir", type=str, default="D:/data", help="CSV 数据目录")
    data_parser.add_argument("--qlib-dir", type=str, default="d:/project/qlib-stock/qlib_data/cn_data", help="Qlib 数据目录")

    return parser.parse_args()


def apply_cli_overrides(config: dict, args) -> dict:
    """Apply CLI argument overrides to config dict."""
    if hasattr(args, "handler") and args.handler:
        config.setdefault("dataset", {})["handler"] = args.handler
    if hasattr(args, "model_type") and args.model_type:
        config.setdefault("model", {})["type"] = args.model_type
    if hasattr(args, "loss") and args.loss:
        config.setdefault("model", {}).setdefault("lgb", {}).setdefault("kwargs", {})["loss"] = args.loss
    if hasattr(args, "topk") and args.topk:
        config.setdefault("backtest", {}).setdefault("strategy", {}).setdefault("kwargs", {})["topk"] = args.topk
    return config


def init_qlib(config: dict):
    """Initialize Qlib with config (Windows-compatible single-thread mode)."""
    provider_uri = config.get("qlib", {}).get("provider_uri", "d:/project/qlib-stock/qlib_data/cn_data")
    logger.info("初始化 Qlib: provider_uri=%s", provider_uri)

    # Windows 兼容性：强制单线程，避免 multiprocessing 子进程重新导入主模块崩溃
    os.environ["NUMEXPR_MAX_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    qlib.init(provider_uri=provider_uri, region=REG_CN)

    # 强制使用 threading backend（避免 spawn 新进程）
    from qlib.config import C
    C.joblib_backend = "threading"
    C.maxtasksperchild = None
    # 限制数据加载并发数，避免 Windows 句柄耗尽
    C.dataset_process_n_worker = 1
    C.min_data_shift = 1

    return provider_uri


def cmd_train(args):
    """执行训练命令。"""
    config = load_workflow_config(args.config)
    config = apply_cli_overrides(config, args)
    init_qlib(config)

    model, dataset, rid = run_train(
        config=config,
        experiment_name=args.experiment,
    )
    logger.info("=" * 60)
    logger.info("训练完成! recorder_id=%s", rid)
    logger.info("=" * 60)
    return model, dataset, rid


def cmd_backtest(args):
    """执行回测命令。"""
    config = load_workflow_config(args.config)
    config = apply_cli_overrides(config, args)

    from qlib.workflow import R

    # 先初始化 Qlib（不依赖 handler）
    init_qlib(config)

    # 从训练 recorder 中读取保存的 handler 类型，确保与训练一致
    train_recorder = R.get_recorder(recorder_id=args.rid, experiment_name=args.experiment)
    params = train_recorder.list_params()
    saved_handler = params.get("handler_type", config.get("dataset", {}).get("handler", "Alpha158"))
    config.setdefault("dataset", {})["handler"] = saved_handler
    logger.info("回测使用 handler: %s (来自训练记录)", saved_handler)

    task = build_task(config)
    dataset = init_instance_by_config(task["dataset"])

    from qlib.workflow.record_temp import SignalRecord, PortAnaRecord

    port_config = config.get("backtest", {})

    with R.start(experiment_name=f"{args.experiment}_backtest"):
        recorder = R.get_recorder(recorder_id=args.rid, experiment_name=args.experiment)
        model = recorder.load_object("trained_model")

        # 注入 model 和 dataset 到策略配置（TopkDropoutStrategy 需要）
        port_config = port_config.copy()
        port_config.setdefault("strategy", {}).setdefault("kwargs", {})["model"] = model
        port_config.setdefault("strategy", {}).setdefault("kwargs", {})["dataset"] = dataset

        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()
        par = PortAnaRecord(recorder, port_config, "day")
        par.generate()
        ba_rid = recorder.id

    # Load results
    recorder = R.get_recorder(recorder_id=ba_rid, experiment_name=f"{args.experiment}_backtest")
    pred_df = recorder.load_object("pred.pkl")
    report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

    print_summary(report_normal_df, analysis_df)

    charts = generate_report_charts(
        pred_df, report_normal_df, analysis_df,
        output_dir=args.output_dir
    )
    logger.info("图表已保存到: %s", args.output_dir)
    return pred_df, report_normal_df, analysis_df


def cmd_full(args):
    """执行一键跑通命令。"""
    config = load_workflow_config(args.config)
    config = apply_cli_overrides(config, args)
    init_qlib(config)

    handler = config.get("dataset", {}).get("handler", "Alpha158")
    model_type = config.get("model", {}).get("type", "lgb")

    logger.info("=" * 60)
    logger.info("Qlib Pipeline 一键跑通")
    logger.info("=" * 60)

    # ====== TRAIN ======
    logger.info("-" * 40)
    logger.info("阶段 1/3: 训练模型")
    logger.info("-" * 40)

    task = build_task(config)
    logger.info("特征处理器: %s", handler)
    logger.info("模型类型: %s", task["model"]["class"])
    logger.info("时间划分: %s", task["dataset"]["kwargs"]["segments"])

    dataset = init_instance_by_config(task["dataset"])
    model = init_instance_by_config(task["model"])

    with R.start(experiment_name=args.experiment):
        R.log_params(**flatten_dict(task))
        model.fit(dataset)
        R.save_objects(trained_model=model)
        rid = R.get_recorder().id

    logger.info("训练完成, recorder_id=%s", rid)

    # ====== BACKTEST ======
    logger.info("-" * 40)
    logger.info("阶段 2/3: 信号预测 + 回测")
    logger.info("-" * 40)

    from qlib.workflow.record_temp import SignalRecord, PortAnaRecord

    port_config = config.get("backtest", {})

    with R.start(experiment_name=f"{args.experiment}_backtest"):
        recorder = R.get_recorder(recorder_id=rid, experiment_name=args.experiment)
        model_bt = recorder.load_object("trained_model")

        # 注入 model 和 dataset 到策略配置（TopkDropoutStrategy 需要）
        port_config = port_config.copy()
        port_config.setdefault("strategy", {}).setdefault("kwargs", {})["model"] = model_bt
        port_config.setdefault("strategy", {}).setdefault("kwargs", {})["dataset"] = dataset

        recorder = R.get_recorder()
        sr = SignalRecord(model_bt, dataset, recorder)
        sr.generate()
        par = PortAnaRecord(recorder, port_config, "day")
        par.generate()
        ba_rid = recorder.id

    recorder = R.get_recorder(recorder_id=ba_rid, experiment_name=f"{args.experiment}_backtest")
    pred_df = recorder.load_object("pred.pkl")
    report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

    print_summary(report_normal_df, analysis_df)

    # ====== CHARTS ======
    logger.info("-" * 40)
    logger.info("阶段 3/3: 生成图表")
    logger.info("-" * 40)

    charts = generate_report_charts(
        pred_df, report_normal_df, analysis_df,
        output_dir=args.output_dir
    )
    logger.info("图表已保存到: %s", args.output_dir)

    logger.info("=" * 60)
    logger.info("Pipeline 完成!")
    logger.info("=" * 60)
    return rid, ba_rid


def cmd_data(args):
    """执行数据管理命令。"""
    if args.download:
        logger.info("=" * 60)
        logger.info("启动 AKShare 数据下载")
        logger.info("=" * 60)
        from data_center.download_history import download_full_history
        download_full_history(sample=args.sample)
        return

    if args.convert:
        logger.info("=" * 60)
        logger.info("启动 CSV 转 Qlib bin")
        logger.info("=" * 60)
        from run_qlib_workflow import create_qlib_bin_data
        csv_dir = Path(args.csv_dir)
        qlib_dir = Path(args.qlib_dir)
        create_qlib_bin_data(csv_dir, qlib_dir, sample=args.sample)
        return

    if args.check:
        logger.info("=" * 60)
        logger.info("检查 Qlib 数据完整性")
        logger.info("=" * 60)
        qlib_dir = Path(args.qlib_dir)
        calendar_file = qlib_dir / "calendars" / "day.txt"
        instruments_file = qlib_dir / "instruments" / "all.txt"
        features_dir = qlib_dir / "features"

        checks = []
        if calendar_file.exists():
            with open(calendar_file) as f:
                days = len(f.readlines())
            checks.append(f"  交易日历: {days} 天")
        else:
            checks.append("  交易日历: 缺失!")

        if instruments_file.exists():
            with open(instruments_file) as f:
                stocks = len(f.readlines())
            checks.append(f"  股票列表: {stocks} 只")
        else:
            checks.append("  股票列表: 缺失!")

        if features_dir.exists():
            feat_stocks = len(list(features_dir.iterdir()))
            checks.append(f"  特征目录: {feat_stocks} 只股票")
        else:
            checks.append("  特征目录: 缺失!")

        for c in checks:
            logger.info(c)
        return

    logger.info("请指定 --download、--convert 或 --check")


def main():
    # 修复 mlflow 文件存储兼容性
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

    args = parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "full":
        cmd_full(args)
    elif args.command == "data":
        cmd_data(args)
    else:
        # 默认执行 full
        logger.info("未指定子命令，默认执行 full 流程")
        # 构造一个假的 args 对象用于 full
        args.config = None
        args.handler = None
        args.model_type = None
        args.loss = None
        args.topk = None
        args.experiment = "qlib_pipeline"
        args.output_dir = "output/qlib_charts"
        cmd_full(args)


if __name__ == "__main__":
    main()
