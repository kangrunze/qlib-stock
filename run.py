#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A股量化选股系统 - Qlib Pipeline 统一入口

一键跑通完整流程：
  数据 → Alpha158/Alpha360 特征 → LightGBM 训练 → 信号预测 → 回测分析 → 图表 → 选股推荐

Usage:
  # === 核心流程 ===
  python run.py full                          # 一键跑通: 训练+回测+图表+选股推荐
  python run.py train                         # 仅训练
  python run.py backtest --rid <recorder_id>  # 仅回测
  python run.py pick [--date 2025-12-31]      # 选股推荐（需已有预测结果）

  # === 数据管理 ===
  python run.py data --download               # 下载 AKShare 数据
  python run.py data --convert                # CSV 转 Qlib bin (日线)
  python run.py data --convert --freq week    # CSV 转 Qlib bin (周线)
  python run.py data --convert --freq month   # CSV 转 Qlib bin (月线)
  python run.py data --check                  # 检查数据完整性

  # === Phase 1: 稳健性 ===
  python run.py rolling --n-folds 6           # 滚动 Walk-Forward 验证
  python run.py drift                         # 特征漂移 + 概念漂移检测
  python run.py ic-stability                  # ICIR/IC衰减/分层IC 分析

  # === Phase 3: 鲁棒性 ===
  python run.py tscv --n-splits 5             # Purged K-Fold TSCV
  python run.py regime                        # 市场阶段稳定性分析
  python run.py sensitivity                   # 超参数敏感性分析
  python run.py key-years                     # 关键年份独立回测

  # === Phase 4: 模型能力恢复 ===
  python run.py optuna --n-trials 100         # Optuna 超参数搜索
  python run.py explain [--rid <id>]          # SHAP 可解释性分析

  # === 参数覆盖 ===
  python run.py train --handler Alpha360 --loss rank --topk 30
  python run.py full --handler Alpha360 --topk 30 --output-dir output/my_run
"""

import argparse
import copy
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# 项目根目录
_PROJECT_ROOT = Path(__file__).parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---- 加载全局配置 ----
def _load_settings() -> dict:
    """加载 config/settings.yaml"""
    settings_path = _PROJECT_ROOT / "config" / "settings.yaml"
    with open(settings_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

_settings = _load_settings()
_output_cfg = _settings.get("output", {})
_ds_cfg = _settings.get("data_source", {})
_strategy_cfg = _settings.get("strategy", {})
_regime_cfg = _settings.get("regime", {})

# MUST patch numpy BEFORE importing qlib
from qlib_pipeline.numpy_compat import *  # noqa

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R

from qlib_pipeline.dataset import load_workflow_config
from qlib_pipeline.train import build_task, run_train, init_qlib_env
from qlib_pipeline.backtest import generate_report_charts, print_summary, print_stock_picks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)5s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run")


def _log_config_summary(config: dict, prefix: str = ""):
    """输出配置摘要日志（中文）"""
    p = prefix + "  " if prefix else ""
    logger.info("%s╔══════════════════════════════════════════════════════════════╗", p)
    logger.info("%s║                   配置摘要                                  ║", p)
    logger.info("%s╠══════════════════════════════════════════════════════════════╣", p)

    # 数据处理器 & 模型
    handler = config.get("dataset", {}).get("handler", "Alpha158")
    model_cfg = config.get("qlib_lgb", {})
    model_name = model_cfg.get("class", "LGBModel")
    loss = model_cfg.get("kwargs", {}).get("loss", "mse")
    logger.info("%s║ 特征处理器: %-10s  模型: %-12s  损失函数: %-6s ║", p, handler, model_name, loss)

    # 股票池
    dh = config.get("data_handler", {})
    instruments = dh.get("instruments", "csi300")
    logger.info("%s║ 股票池: %-20s                                ║", p, instruments)

    # 数据加载时间范围
    dh_start = dh.get("start_time", "N/A")
    dh_end = dh.get("end_time", "N/A")
    logger.info("%s║ 数据加载范围: %s ~ %s                    ║", p, dh_start, dh_end)

    # 拟合时间范围
    fit_start = dh.get("fit_start_time", "N/A")
    fit_end = dh.get("fit_end_time", "N/A")
    logger.info("%s║ 拟合范围:    %s ~ %s                    ║", p, fit_start, fit_end)

    # 训练/验证/测试划分
    segs = config.get("dataset", {}).get("segments", {})
    train = segs.get("train", ["N/A", "N/A"])
    valid = segs.get("valid", ["N/A", "N/A"])
    test = segs.get("test", ["N/A", "N/A"])
    logger.info("%s║ 训练集: %s ~ %s                               ║", p, train[0], train[1])
    logger.info("%s║ 验证集: %s ~ %s                               ║", p, valid[0], valid[1])
    logger.info("%s║ 测试集: %s ~ %s                               ║", p, test[0], test[1])

    # 回测参数
    bt = config.get("backtest", {}).get("backtest", {})
    bt_start = bt.get("start_time", "N/A")
    bt_end = bt.get("end_time", "N/A")
    account = bt.get("account", "N/A")
    benchmark = bt.get("benchmark", "N/A")
    logger.info("%s║ 回测范围: %s ~ %s  初始资金: %s  基准: %s ║", p, bt_start, bt_end, account, benchmark)

    # 策略参数
    strat = config.get("backtest", {}).get("strategy", {}).get("kwargs", {})
    topk = strat.get("topk", config.get("strategy", {}).get("kwargs", {}).get("topk", "N/A"))
    n_drop = strat.get("n_drop", "N/A")
    logger.info("%s║ 策略: TopkDropout  topk=%s  n_drop=%s                       ║", p, topk, n_drop)

    # 交易成本
    ex = bt.get("exchange_kwargs", {})
    open_cost = ex.get("open_cost", "N/A")
    close_cost = ex.get("close_cost", "N/A")
    logger.info("%s║ 交易成本: 开仓 %.4f  平仓 %.4f                             ║", p, open_cost, close_cost)

    logger.info("%s╚══════════════════════════════════════════════════════════════╝", p)


def parse_args():
    parser = argparse.ArgumentParser(description="Qlib Pipeline 统一入口")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # === full: 一键跑通 ===
    full_parser = subparsers.add_parser("full", help="一键跑通: 训练+回测+图表+选股推荐")
    full_parser.add_argument("--config", type=str, default=None)
    full_parser.add_argument("--handler", type=str, default=None,
                        choices=["Alpha158", "Alpha360"])
    full_parser.add_argument("--loss", type=str, default=None,
                        choices=["mse", "rank"])
    full_parser.add_argument("--topk", type=int, default=None)
    full_parser.add_argument("--pick-topk", type=int, default=_strategy_cfg.get("top_k", 30),
                             help="选股推荐数量")
    full_parser.add_argument("--phase", type=str, default="exploration",
                             choices=["exploration", "confirmation"],
                             help="实验阶段: exploration=探索阶段, confirmation=确认阶段（只允许运行一次）")
    full_parser.add_argument("--experiment", type=str, default="qlib_pipeline")
    full_parser.add_argument("--output-dir", type=str, default=_output_cfg.get("charts", "output/qlib_charts"))

    # === train: 仅训练 ===
    train_parser = subparsers.add_parser("train", help="仅训练模型")
    train_parser.add_argument("--config", type=str, default=None)
    train_parser.add_argument("--handler", type=str, default=None,
                        choices=["Alpha158", "Alpha360"])
    train_parser.add_argument("--loss", type=str, default=None,
                        choices=["mse", "rank"])
    train_parser.add_argument("--experiment", type=str, default="qlib_train")

    # === backtest: 仅回测 ===
    bt_parser = subparsers.add_parser("backtest", help="仅回测 (需已有训练结果)")
    bt_parser.add_argument("--config", type=str, default=None)
    bt_parser.add_argument("--rid", type=str, required=True)
    bt_parser.add_argument("--experiment", type=str, default="qlib_train")
    bt_parser.add_argument("--output-dir", type=str, default=_output_cfg.get("charts", "output/qlib_charts"))
    bt_parser.add_argument("--topk", type=int, default=None)
    bt_parser.add_argument("--pick-topk", type=int, default=_strategy_cfg.get("top_k", 30),
                        help="选股推荐数量")

    # === pick: 选股推荐 ===
    pick_parser = subparsers.add_parser("pick", help="选股推荐（需已有回测结果）")
    pick_parser.add_argument("--config", type=str, default=None)
    pick_parser.add_argument("--rid", type=str, required=True)
    pick_parser.add_argument("--experiment", type=str, default="qlib_pipeline")
    pick_parser.add_argument("--topk", type=int, default=_strategy_cfg.get("top_k", 30))
    pick_parser.add_argument("--date", type=str, default=None,
                        help="指定日期 YYYY-MM-DD（默认取最新）")
    pick_parser.add_argument("--output-dir", type=str, default=_output_cfg.get("picks", "output/picks"))

    # === data: 数据管理 ===
    data_parser = subparsers.add_parser("data", help="数据下载/转换/检查")
    data_parser.add_argument("--download", action="store_true")
    data_parser.add_argument("--convert", action="store_true")
    data_parser.add_argument("--check", action="store_true")
    data_parser.add_argument("--sample", type=int, default=None)
    data_parser.add_argument("--csv-dir", type=str, default=_ds_cfg.get("csv_dir", "D:/data"))
    data_parser.add_argument("--qlib-dir", type=str,
                        default=_ds_cfg.get("qlib_dir", "D:/trae/qlib_bin"))
    data_parser.add_argument("--freq", type=str, default="day",
                        choices=["day", "week", "month"],
                        help="数据频率: day(日线) | week(周线) | month(月线)")

    # === update: 每日增量更新 ===
    update_parser = subparsers.add_parser("update", help="每日增量更新（下载最近N天数据并更新bin）")
    update_parser.add_argument("--days", "-d", type=int, default=3,
                        help="下载最近多少天的数据（默认: 3）")
    update_parser.add_argument("--skip-bin", action="store_true",
                        help="跳过 bin 文件更新（只更新 parquet）")
    update_parser.add_argument("--workers", "-w", type=int, default=None,
                        help="并发线程数（默认使用配置值）")

    # === Phase 1: rolling ===
    rolling_parser = subparsers.add_parser("rolling", help="滚动 Walk-Forward 验证")
    rolling_parser.add_argument("--config", type=str, default=None)
    rolling_parser.add_argument("--n-folds", type=int, default=6)
    rolling_parser.add_argument("--window-years", type=float, default=3.0)
    rolling_parser.add_argument("--step-months", type=int, default=6)
    rolling_parser.add_argument("--output-dir", type=str, default=_output_cfg.get("rolling", "output/rolling"))

    # === Phase 1: drift ===
    drift_parser = subparsers.add_parser("drift", help="特征漂移检测 (PSI)")
    drift_parser.add_argument("--config", type=str, default=None)
    drift_parser.add_argument("--output-dir", type=str, default=_output_cfg.get("drift", "output/drift"))

    # === Phase 1: ic-stability ===
    ic_parser = subparsers.add_parser("ic-stability", help="IC 稳定性分析 (ICIR/衰减/分层)")
    ic_parser.add_argument("--config", type=str, default=None)
    ic_parser.add_argument("--output-dir", type=str, default=_output_cfg.get("ic_stability", "output/ic_stability"))
    ic_parser.add_argument("--phase", type=str, default="exploration",
                             choices=["exploration", "confirmation"],
                             help="实验阶段: exploration=探索阶段, confirmation=确认阶段（只允许运行一次）")
    ic_parser.add_argument("--experiment", type=str, default="ic_stability")

    # === Phase 3: tscv ===
    tscv_parser = subparsers.add_parser("tscv", help="Purged K-Fold TSCV")
    tscv_parser.add_argument("--config", type=str, default=None)
    tscv_parser.add_argument("--n-splits", type=int, default=5)
    tscv_parser.add_argument("--purge-days", type=int, default=5)
    tscv_parser.add_argument("--embargo-days", type=int, default=0)
    tscv_parser.add_argument("--output-dir", type=str, default=_output_cfg.get("tscv", "output/tscv"))

    # === Phase 3: regime ===
    regime_parser = subparsers.add_parser("regime", help="市场阶段稳定性分析")
    regime_parser.add_argument("--config", type=str, default=None)
    regime_parser.add_argument("--output-dir", type=str, default=_output_cfg.get("regime", "output/regime"))

    # === Phase 3: sensitivity ===
    sens_parser = subparsers.add_parser("sensitivity", help="超参数敏感性分析")
    sens_parser.add_argument("--config", type=str, default=None)
    sens_parser.add_argument("--param", type=str, default=None,
                        help="单参数扫描 (e.g. learning_rate)")
    sens_parser.add_argument("--values", type=str, default=None,
                        help="逗号分隔值 (e.g. 0.01,0.05,0.10)")
    sens_parser.add_argument("--output-dir", type=str, default=_output_cfg.get("sensitivity", "output/sensitivity"))

    # === Phase 3: key-years ===
    ky_parser = subparsers.add_parser("key-years", help="关键年份独立回测")
    ky_parser.add_argument("--config", type=str, default=None)
    ky_parser.add_argument("--years", type=str, default=None,
                      help="逗号分隔年份 (e.g. 2020,2022,2024)")
    ky_parser.add_argument("--rid", type=str, default=None,
                      help="预训练模型 recorder_id（传入则使用样本外模式，在未训练年份上做纯独立回测）")
    ky_parser.add_argument("--output-dir", type=str, default=_output_cfg.get("key_years", "output/key_years"))

    # === validate-picks: 验证历史推荐 ===
    vp_parser = subparsers.add_parser("validate-picks", help="验证历史选股推荐的实际表现")
    vp_parser.add_argument("--picks-dir", type=str, default=_output_cfg.get("picks", "output/picks"))
    vp_parser.add_argument("--lookback-days", type=int, default=20)
    vp_parser.add_argument("--output-dir", type=str, default=_output_cfg.get("validation", "output/validation"))

    # === Phase 4: 风险诊断（风格暴露 / 收益归因 / 容量） ===
    risk_parser = subparsers.add_parser("risk", help="风格暴露诊断（需要 Phase A 基本面数据）")
    risk_parser.add_argument("--config", type=str, default=None)
    risk_parser.add_argument("--rid", type=str, default=None,
                        help="回测 recorder_id（从回测结果中提取持仓权重）")
    risk_parser.add_argument("--output-dir", type=str, default=_output_cfg.get("charts", "output/qlib_charts"))

    attr_parser = subparsers.add_parser("attribution", help="收益归因分析（需要 Phase A 基本面数据）")
    attr_parser.add_argument("--config", type=str, default=None)
    attr_parser.add_argument("--rid", type=str, default=None,
                        help="回测 recorder_id（从回测结果中提取收益序列）")
    attr_parser.add_argument("--output-dir", type=str, default=_output_cfg.get("charts", "output/qlib_charts"))

    cap_parser = subparsers.add_parser("capacity", help="组合容量检查（需要 Phase A 成交额数据）")
    cap_parser.add_argument("--config", type=str, default=None)
    cap_parser.add_argument("--rid", type=str, default=None,
                        help="回测 recorder_id（从回测结果中提取持仓）")
    cap_parser.add_argument("--account", type=float, default=1e8,
                        help="管理规模（元，默认 1 亿）")
    cap_parser.add_argument("--output-dir", type=str, default=_output_cfg.get("charts", "output/qlib_charts"))

    # === Phase 4: 强制性对比实验（第 8.4 节） ===
    bm_parser = subparsers.add_parser("benchmark", help="第 8.4 节强制性对比实验（E1-E5）")
    bm_parser.add_argument("--config", type=str, default=None)
    bm_parser.add_argument("--experiments", type=str, default=None,
                      help="逗号分隔实验列表，如 E1,E3；不指定则运行全部")
    bm_parser.add_argument("--output-dir", type=str, default="output/benchmark")

    # === Phase 4: 超参数搜索 ===
    optuna_parser = subparsers.add_parser("optuna", help="Optuna 超参数搜索")
    optuna_parser.add_argument("--config", type=str, default=None)
    optuna_parser.add_argument("--n-trials", type=int, default=100)
    optuna_parser.add_argument("--timeout", type=int, default=3600)
    optuna_parser.add_argument("--study-name", type=str, default="lgb_optimization")
    optuna_parser.add_argument("--output-dir", type=str, default="output/optuna")
    optuna_parser.add_argument("--phase", type=str, default="exploration",
                               choices=["exploration", "confirmation"])

    # === Phase 4: SHAP 可解释性分析 ===
    explain_parser = subparsers.add_parser("explain", help="SHAP 可解释性分析")
    explain_parser.add_argument("--config", type=str, default=None)
    explain_parser.add_argument("--rid", type=str, default=None)
    explain_parser.add_argument("--experiment", type=str, default="qlib_pipeline")
    explain_parser.add_argument("--output-dir", type=str, default="output/shap")
    explain_parser.add_argument("--segment", type=str, default="test")
    explain_parser.add_argument("--max-samples", type=int, default=2000)
    explain_parser.add_argument("--full", action="store_true",
                                help="输出完整报告（含单样本解释和特征重要性）")

    return parser.parse_args()


def apply_cli_overrides(config: dict, args) -> dict:
    if hasattr(args, "handler") and args.handler:
        config.setdefault("dataset", {})["handler"] = args.handler
    if hasattr(args, "loss") and args.loss:
        config.setdefault("qlib_lgb", {}).setdefault("kwargs", {})["loss"] = args.loss
    if hasattr(args, "topk") and args.topk:
        config.setdefault("backtest", {}).setdefault("strategy", {}).setdefault("kwargs", {})["topk"] = args.topk
    return config


# ===== Core Commands =====

def cmd_train(args):
    logger.info("[命令] train — 启动模型训练流程")
    config = load_workflow_config(args.config)
    config = apply_cli_overrides(config, args)
    _log_config_summary(config)

    # ── 数据预检 ──
    if not _check_data_availability(config):
        logger.error("数据预检未通过，流程终止。请先准备数据后再运行。")
        logger.error("  数据准备命令: python run.py data --convert")
        return

    try:
        logger.info("[步骤 1/3] 初始化 Qlib 环境 ...")
        init_qlib_env(config)
        logger.info("[步骤 1/3] ✓ Qlib 环境初始化完成")
    except Exception as e:
        logger.error("Qlib 环境初始化失败: %s", e)
        logger.error("  → 请检查 workflow_config.yaml 中的 qlib.provider_uri 路径")
        return

    try:
        # 输出训练时间段
        segments = config.get("dataset", {}).get("segments", {})
        train_seg = segments.get("train", [])
        valid_seg = segments.get("valid", [])
        test_seg = segments.get("test", [])
        handler_cfg = config.get("data_handler", {})
        logger.info(
            "[步骤 2/3] 开始模型训练 — 训练集: %s~%s, 验证集: %s~%s, 测试集: %s~%s",
            train_seg[0] if len(train_seg) > 0 else "N/A",
            train_seg[1] if len(train_seg) > 1 else "N/A",
            valid_seg[0] if len(valid_seg) > 0 else "N/A",
            valid_seg[1] if len(valid_seg) > 1 else "N/A",
            test_seg[0] if len(test_seg) > 0 else "N/A",
            test_seg[1] if len(test_seg) > 1 else "N/A",
        )
        model, dataset, rid = run_train(config=config, experiment_name=args.experiment)
        logger.info("[步骤 2/3] ✓ 模型训练完成")
    except Exception as e:
        logger.error("[步骤 2/3] 模型训练失败: %s", e)
        logger.error("  → 排查: 运行 python run.py data --check 确认数据完整性")
        logger.error("  → 检查 workflow_config.yaml 中时间段配置是否在数据范围内")
        raise

    logger.info("[步骤 3/3] 保存训练结果 ...")
    logger.info("=" * 60)
    logger.info("训练完成! recorder_id=%s", rid)
    logger.info("=" * 60)
    return model, dataset, rid


def cmd_backtest(args):
    logger.info("[命令] backtest — 启动回测流程 (recorder_id=%s)", args.rid)
    config = load_workflow_config(args.config)
    config = apply_cli_overrides(config, args)
    _log_config_summary(config)

    logger.info("[步骤 1/4] 初始化 Qlib 环境并加载训练记录 ...")
    from qlib.workflow import R
    init_qlib_env(config)
    train_recorder = R.get_recorder(recorder_id=args.rid, experiment_name=args.experiment)
    params = train_recorder.list_params()
    saved_handler = params.get("handler_type", config.get("dataset", {}).get("handler", "Alpha158"))
    config.setdefault("dataset", {})["handler"] = saved_handler
    logger.info("[步骤 1/4] ✓ 加载训练记录成功, handler=%s", saved_handler)

    logger.info("[步骤 2/4] 构建数据集 ...")
    task = build_task(config)
    dataset = init_instance_by_config(task["dataset"])
    logger.info("[步骤 2/4] ✓ 数据集构建完成")

    logger.info("[步骤 3/4] 执行信号预测与回测 ...")
    from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
    port_config = config.get("backtest", {})
    bt_cfg = port_config.get("backtest", {})
    logger.info("  → 回测时间范围: %s ~ %s, 初始资金: %s",
                bt_cfg.get("start_time", "N/A"),
                bt_cfg.get("end_time", "N/A"),
                bt_cfg.get("account", "N/A"))
    experiment_bt = f"{args.experiment}_backtest"
    with R.start(experiment_name=experiment_bt):
        recorder = R.get_recorder(recorder_id=args.rid, experiment_name=args.experiment)
        model = recorder.load_object("trained_model")
        port_config = copy.deepcopy(port_config)
        s_kwargs = port_config.setdefault("strategy", {}).setdefault("kwargs", {})
        s_kwargs["model"] = model
        s_kwargs["dataset"] = dataset
        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()
        par = PortAnaRecord(recorder, port_config, "day")
        par.generate()
        ba_rid = recorder.id
    logger.info("[步骤 3/4] ✓ 回测完成, backtest_recorder_id=%s", ba_rid)

    logger.info("[步骤 4/4] 生成分析报告与图表 ...")
    recorder = R.get_recorder(recorder_id=ba_rid, experiment_name=experiment_bt)
    pred_df = recorder.load_object("pred.pkl")
    report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
    print_summary(report_normal_df, analysis_df, config=config)
    generate_report_charts(pred_df, report_normal_df, analysis_df, output_dir=args.output_dir)
    logger.info("[步骤 4/4] ✓ 图表已保存到: %s", args.output_dir)

    # 输出选股推荐
    if hasattr(args, "pick_topk") and args.pick_topk > 0:
        logger.info("[附加] 生成选股推荐 Top-%d ...", args.pick_topk)
        print_stock_picks(pred_df, top_k=args.pick_topk, output_dir="output/picks",
                         recorder_id=ba_rid, config_snapshot=config)
        logger.info("[附加] ✓ 选股推荐已保存到 output/picks")

    logger.info("=" * 60)
    logger.info("回测流程全部完成!")
    logger.info("=" * 60)
    return pred_df, report_normal_df, analysis_df


def _check_data_availability(config: dict) -> bool:
    """检查 Qlib bin 数据是否可用，缺失时给出明确提示"""
    import qlib
    provider_uri = config.get("qlib", {}).get("provider_uri", "")
    provider_path = Path(provider_uri)

    # 检查 1: 数据目录是否存在
    if not provider_path.exists():
        logger.error("数据目录不存在: %s", provider_path)
        logger.error("  → 请先运行: python run.py data --convert")
        return False

    # 检查 2: 日历文件
    cal_file = provider_path / "calendars" / "day.txt"
    if not cal_file.exists():
        logger.error("交易日历缺失: %s", cal_file)
        logger.error("  → 请先运行: python run.py data --convert")
        return False
    with open(cal_file) as f:
        cal_lines = len(f.readlines())
    if cal_lines < 10:
        logger.error("交易日历数据过少（仅 %d 行），数据可能不完整", cal_lines)
        logger.error("  → 请检查 CSV 原始数据是否充足")
        return False

    # 检查 3: 股票列表
    inst_file = provider_path / "instruments" / "all.txt"
    if not inst_file.exists():
        logger.error("股票列表缺失: %s", inst_file)
        logger.error("  → 请先运行: python run.py data --convert")
        return False
    with open(inst_file) as f:
        stock_lines = len(f.readlines())

    # 检查 4: features 目录
    features_dir = provider_path / "features"
    if not features_dir.exists() or not any(features_dir.iterdir()):
        logger.error("特征目录为空: %s", features_dir)
        logger.error("  → 请先运行: python run.py data --convert")
        return False
    stock_count = len([d for d in features_dir.iterdir() if d.is_dir()])

    logger.info("[数据预检] ✓ 数据可用: %d 只股票, %d 个交易日", stock_count, cal_lines)

    # 检查 5: 时间范围校验（配置的时间段是否超出数据日历）
    try:
        from qlib.data import D
        calendar = D.calendar(start_time=None, end_time=None)
        cal_start, cal_end = pd.Timestamp(str(calendar[0])), pd.Timestamp(str(calendar[-1]))

        segments = config.get("dataset", {}).get("segments", {})
        for seg_name, seg in segments.items():
            seg_end = pd.Timestamp(seg[1] if isinstance(seg, (list, tuple)) else seg.get("end"))
            if seg_end > cal_end:
                logger.error("配置的 %s 段结束日期 %s 超出数据日历范围（最后交易日 %s）",
                             seg_name, seg_end.strftime("%Y-%m-%d"), cal_end.strftime("%Y-%m-%d"))
                return False

        bt = config.get("backtest", {}).get("backtest", {})
        if bt.get("end_time"):
            bt_end = pd.Timestamp(bt["end_time"])
            if bt_end > cal_end:
                logger.error("回测 end_time %s 超出数据日历范围（最后交易日 %s）",
                             bt_end.strftime("%Y-%m-%d"), cal_end.strftime("%Y-%m-%d"))
                return False
        if bt.get("start_time"):
            bt_start = pd.Timestamp(bt["start_time"])
            if bt_start < cal_start:
                logger.error("回测 start_time %s 早于数据日历起始日期（最早交易日 %s）",
                             bt_start.strftime("%Y-%m-%d"), cal_start.strftime("%Y-%m-%d"))
                return False
    except Exception as e:
        logger.warning("无法读取 Qlib 日历用于时间范围校验: %s", e)

    return True


def _run_risk_diagnostics(config, ba_rid, pred_df, report_normal_df):
    """一键跑通后自动追加风险诊断（风格暴露 / 容量 / 归因）。

    所有诊断优雅降级：Phase A 数据未就绪时仅提示框架，不报错。
    """
    logger.info("-" * 40)
    logger.info("[阶段 4/4] 风险诊断（风格暴露 / 容量 / 归因）")
    logger.info("-" * 40)

    # ── 4a. 风格暴露诊断 ──
    try:
        from research.risk_model import STYLE_FACTORS, calculate_style_exposures, check_style_constraints
        logger.info("[4a] 风格暴露诊断 ...")

        # 从选股结果提取最新日期的持仓
        if pred_df is not None and not pred_df.empty:
            pick_date = pred_df.index.get_level_values("datetime")[-1] if hasattr(pred_df.index, "get_level_values") else pred_df.index[-1][0]
            pick_stocks = list(pred_df.loc[pick_date].index)[:30] if hasattr(pred_df, "loc") else []
            needed_features = [v["feature"] for v in STYLE_FACTORS.values()]

            # 尝试加载因子值
            from qlib.data import D
            factor_data = {}
            for feat in needed_features:
                try:
                    fv = D.features(pick_stocks, [feat], start_time=pick_date, end_time=pick_date)
                    if fv is not None and not fv.empty:
                        factor_data[feat] = fv.iloc[:, 0] if fv.shape[1] > 0 else None
                except Exception:
                    pass

            if len(factor_data) >= 3:
                factor_df = pd.DataFrame(factor_data)
                weights = pd.Series(1.0 / len(pick_stocks), index=pick_stocks)
                exposures = calculate_style_exposures(weights, factor_df)
                if not exposures.empty:
                    logger.info("  ✓ 风格暴露计算完成 (%d 个维度):", len(exposures))
                    for _, row in exposures.iterrows():
                        flag = " ⚠ 超标" if abs(row["active_exposure"]) > 0.5 else ""
                        logger.info("    %s: active=%+.3fσ%s", row["name_cn"], row["active_exposure"], flag)
                    violations = check_style_constraints(exposures)
                    if violations:
                        logger.warning("  ⚠ 风格暴露超标: %s", ", ".join(violations))
                    else:
                        logger.info("  ✓ 所有风格暴露在阈值内 (|active| ≤ 0.5σ)")
                else:
                    _log_phase_a_hint("风格暴露", needed_features)
            else:
                _log_phase_a_hint("风格暴露", needed_features)
        else:
            logger.info("  (选股结果为空，跳过风格暴露诊断)")
    except ImportError:
        logger.info("  (research.risk_model 不可用)")
    except Exception as e:
        logger.warning("  (风格暴露诊断失败: %s)", e)

    # ── 4b. 容量检查 ──
    try:
        from research.capacity import check_portfolio_capacity
        logger.info("[4b] 容量检查 ...")

        if pred_df is not None and not pred_df.empty:
            pick_date = pred_df.index.get_level_values("datetime")[-1] if hasattr(pred_df.index, "get_level_values") else pred_df.index[-1][0]
            pick_stocks = list(pred_df.loc[pick_date].index)[:30] if hasattr(pred_df, "loc") else []

            if not pick_stocks:
                logger.info("  (选股列表为空，跳过容量检查)")
            else:
                try:
                    from qlib.data import D
                    vol_start = str(pd.Timestamp(pick_date) - pd.Timedelta(days=120))
                    vol_end = str(pick_date)
                    volume_data = {}
                    for stock in pick_stocks[:10]:
                        try:
                            vol = D.features([stock], ["$volume"], start_time=vol_start, end_time=vol_end)
                            if vol is not None and not vol.empty:
                                volume_data[stock] = float(vol.mean().iloc[0]) if vol.shape[1] > 0 else 0.0
                        except Exception:
                            pass

                    if len(volume_data) >= 3:
                        volume_series = pd.Series(volume_data)
                        weights = pd.Series(1.0 / len(pick_stocks), index=pick_stocks[:len(volume_series)])
                        result_df, all_ok = check_portfolio_capacity(weights, volume_series)
                        if all_ok:
                            logger.info("  ✓ 容量检查通过 (持仓 ≤ 5%% 日均成交额)")
                        else:
                            n_fail = (~result_df["acceptable"]).sum()
                            logger.warning("  ⚠ %d/%d 只股票超出流动性约束", n_fail, len(result_df))
                    else:
                        _log_phase_a_hint("容量检查", ["$volume", "日均成交额"])
                except Exception:
                    _log_phase_a_hint("容量检查", ["$volume", "日均成交额"])
        else:
            logger.info("  (选股结果为空，跳过容量检查)")
    except ImportError:
        logger.info("  (research.capacity 不可用)")
    except Exception as e:
        logger.warning("  (容量检查失败: %s)", e)

    # ── 4c. 收益归因 ──
    try:
        from research.attribution import decompose_excess_return
        logger.info("[4c] 收益归因 ...")

        if report_normal_df is not None and not report_normal_df.empty:
            port_ret = report_normal_df.get("return")
            bench_ret = report_normal_df.get("benchmark_return")
            if port_ret is not None and bench_ret is not None:
                try:
                    from research.risk_model import STYLE_FACTORS
                    from qlib.data import D
                    needed_features = [v["feature"] for v in STYLE_FACTORS.values()]
                    sample_stocks = list(port_ret.index.get_level_values("instrument").unique())[:10]
                    factor_values = {}
                    for feat in needed_features:
                        try:
                            fv = D.features(
                                list(sample_stocks), [feat],
                                start_time=str(port_ret.index.get_level_values("datetime").min())[:10],
                                end_time=str(port_ret.index.get_level_values("datetime").max())[:10],
                            )
                            if fv is not None and not fv.empty:
                                factor_values[feat] = fv
                        except Exception:
                            pass
                    if len(factor_values) >= 3:
                        from research.attribution import calculate_factor_returns
                        decomposition = decompose_excess_return(
                            port_ret, bench_ret,
                            factor_exposures_df=pd.DataFrame(factor_values),
                            factor_returns_df=calculate_factor_returns(
                                pd.DataFrame(factor_values), port_ret.to_frame(),
                            ),
                        )
                        if not decomposition.empty:
                            logger.info("  ✓ 收益归因完成")
                        else:
                            _log_phase_a_hint("收益归因", ["$roe", "$pb_inv", "$mom_12m"])
                    else:
                        _log_phase_a_hint("收益归因", ["$roe", "$pb_inv", "$mom_12m"])
                except Exception:
                    _log_phase_a_hint("收益归因", ["$roe", "$pb_inv", "$mom_12m"])
            else:
                logger.info("  (回测报告缺少收益列，跳过归因)")
        else:
            logger.info("  (回测报告为空，跳过收益归因)")
    except ImportError:
        logger.info("  (research.attribution 不可用)")
    except Exception as e:
        logger.warning("  (收益归因失败: %s)", e)

    logger.info("  Phase A 数据就绪后，风险诊断将自动产出完整报告。")
    logger.info("  也可单独运行: python run.py risk|attribution|capacity --rid <id>")


def _log_phase_a_hint(diagnostic_name, needed_features):
    """统一的 Phase A 未就绪提示"""
    logger.info("  (Phase A 数据未就绪，%s 跳过)", diagnostic_name)
    logger.info("  → 需要特征: %s", ", ".join(needed_features[:5]))
    logger.info("  → 完成 Phase A 基本面数据接入后自动生效")


def cmd_full(args):
    logger.info("[命令] full — 启动一键跑通流程 (训练+回测+图表+选股)")
    config = load_workflow_config(args.config)
    config = apply_cli_overrides(config, args)
    _log_config_summary(config)

    # ── 数据预检 ──
    if not _check_data_availability(config):
        logger.error("数据预检未通过，流程终止。请先准备数据后再运行。")
        logger.error("  数据准备命令: python run.py data --convert")
        return

    try:
        init_qlib_env(config)
    except Exception as e:
        logger.error("Qlib 环境初始化失败: %s", e)
        logger.error("  → 请检查 workflow_config.yaml 中的 qlib.provider_uri 路径是否正确")
        logger.error("  → 请确认已运行: python run.py data --convert")
        return

    handler = config.get("dataset", {}).get("handler", "Alpha158")
    logger.info("=" * 60)
    logger.info("Qlib Pipeline 一键跑通")
    logger.info("=" * 60)

    rid = None
    ba_rid = None

    # ── 阶段 1: 训练模型 ──
    try:
        logger.info("-" * 40)
        logger.info("[阶段 1/3] 模型训练")
        logger.info("-" * 40)
        # 输出训练时间段
        segments = config.get("dataset", {}).get("segments", {})
        train_seg = segments.get("train", [])
        valid_seg = segments.get("valid", [])
        test_seg = segments.get("test", [])
        logger.info("  → 训练集: %s~%s, 验证集: %s~%s, 测试集: %s~%s",
                    train_seg[0] if len(train_seg) > 0 else "N/A",
                    train_seg[1] if len(train_seg) > 1 else "N/A",
                    valid_seg[0] if len(valid_seg) > 0 else "N/A",
                    valid_seg[1] if len(valid_seg) > 1 else "N/A",
                    test_seg[0] if len(test_seg) > 0 else "N/A",
                    test_seg[1] if len(test_seg) > 1 else "N/A")
        logger.info("  → 正在构建任务配置 ...")
        task = build_task(config)
        logger.info("  ✓ 特征处理器: %s, 模型: %s, 损失函数: %s",
                    handler, task["model"]["class"],
                    task["model"].get("kwargs", {}).get("loss", "mse"))

        logger.info("  → 正在创建数据集 (Alpha158/360 特征计算，可能需要几分钟) ...")
        dataset = init_instance_by_config(task["dataset"])
        logger.info("  ✓ 数据集创建完成")

        logger.info("  → 正在初始化模型并启动训练 ...")
        model = init_instance_by_config(task["model"])
        with R.start(experiment_name=args.experiment):
            R.log_params(**flatten_dict(task))
            model.fit(dataset)
            R.save_objects(trained_model=model)
            rid = R.get_recorder().id
        logger.info("  ✓ 训练完成, recorder_id=%s", rid)
    except Exception as e:
        logger.error("[阶段 1/3] 模型训练失败: %s", e)
        logger.error("  → 可能原因: 数据范围与时间段配置不匹配，或内存不足")
        logger.error("  → 排查建议:")
        logger.error("      1. 运行 python run.py data --check 确认数据完整性")
        logger.error("      2. 检查 workflow_config.yaml 中 data_handler.end_time 是否超出数据范围")
        logger.error("      3. 检查 dataset.segments 的 train/valid/test 是否在数据范围内")
        raise

    # ── 阶段 2: 信号预测 + 回测 ──
    try:
        logger.info("-" * 40)
        logger.info("[阶段 2/3] 信号预测与回测")
        logger.info("-" * 40)
        from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
        port_config = config.get("backtest", {})
        bt_cfg = port_config.get("backtest", {})
        logger.info("  → 回测时间范围: %s ~ %s, 初始资金: %s",
                    bt_cfg.get("start_time", "N/A"),
                    bt_cfg.get("end_time", "N/A"),
                    bt_cfg.get("account", "N/A"))

        with R.start(experiment_name=f"{args.experiment}_backtest"):
            recorder = R.get_recorder(recorder_id=rid, experiment_name=args.experiment)
            model_bt = recorder.load_object("trained_model")
            port_config = copy.deepcopy(port_config)
            s_kwargs = port_config.setdefault("strategy", {}).setdefault("kwargs", {})
            s_kwargs["model"] = model_bt
            s_kwargs["dataset"] = dataset
            recorder = R.get_recorder()
            logger.info("  → 正在生成预测信号 ...")
            sr = SignalRecord(model_bt, dataset, recorder)
            sr.generate()
            logger.info("  → 正在执行回测模拟交易 ...")
            par = PortAnaRecord(recorder, port_config, "day")
            par.generate()
            ba_rid = recorder.id
        logger.info("  ✓ 回测完成, backtest_recorder_id=%s", ba_rid)
    except Exception as e:
        logger.error("[阶段 2/3] 信号预测与回测失败: %s", e)
        if "benchmark" in str(e).lower():
            logger.error("  → 基准股票不存在，请修改 workflow_config.yaml:")
            logger.error("      backtest.backtest.benchmark 改为一个真实存在的股票代码（如 '600000'）")
        elif "index" in str(e).lower():
            logger.error("  → 回测日期超出日历范围，请修改 backtest.backtest.end_time")
        else:
            logger.error("  → 排查建议:")
            logger.error("      1. 确认训练已完成（recorder_id=%s）", rid)
            logger.error("      2. 检查 backtest 配置的时间范围和基准设置")
        logger.info("  → 训练结果仍可用，可单独重跑回测:")
        logger.info("      python run.py backtest --rid %s", rid if rid else "<recorder_id>")
        raise

    # ── 阶段 3: 生成图表 + 选股推荐 ──
    try:
        recorder = R.get_recorder(recorder_id=ba_rid, experiment_name=f"{args.experiment}_backtest")
        pred_df = recorder.load_object("pred.pkl")
        report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
        analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
        print_summary(report_normal_df, analysis_df, config=config)

        logger.info("-" * 40)
        logger.info("[阶段 3/3] 生成图表与选股推荐")
        logger.info("-" * 40)
        generate_report_charts(pred_df, report_normal_df, analysis_df, output_dir=args.output_dir)
        logger.info("  ✓ 图表已保存到: %s", args.output_dir)

        pick_topk = getattr(args, "pick_topk", 30)
        if pick_topk > 0:
            logger.info("  → 正在生成 Top-%d 选股推荐 ...", pick_topk)
            print_stock_picks(pred_df, top_k=pick_topk, output_dir="output/picks",
                           recorder_id=ba_rid, config_snapshot=config)
            logger.info("  ✓ 选股推荐已保存到 output/picks")

        # 保存买卖点记录（新增，用户要求）
        from qlib_pipeline.backtest import save_trade_records
        backtopk = config.get("backtest", {}).get("strategy", {}).get("kwargs", {}).get("topk", 50)
        backdrop = config.get("backtest", {}).get("strategy", {}).get("n_drop", 5)
        init_cash = config.get("backtest", {}).get("backtest", {}).get("account", 100000000)
        save_trade_records(pred_df, topk=backtopk, n_drop=backdrop, init_cash=init_cash,
                         report_normal_df=report_normal_df, output_dir="output/trades")
    except Exception as e:
        logger.error("[阶段 3/3] 图表/选股生成失败: %s", e)
        logger.error("  → 训练和回测结果已保存，不影响核心结论")
        logger.error("  → 可单独生成图表: python run.py backtest --rid %s", rid if rid else "<recorder_id>")
        raise

    logger.info("=" * 60)
    logger.info("Pipeline 全部完成!")
    logger.info("  训练记录: recorder_id=%s", rid)
    logger.info("  回测记录: backtest_recorder_id=%s", ba_rid)
    logger.info("=" * 60)

    # ── 阶段 4: 风险诊断（风格暴露 / 容量 / 归因）──
    _run_risk_diagnostics(config, ba_rid, pred_df, report_normal_df)

    # ── 实验追踪与协议检查（第 2.2 节 + 第 8.2 节）──
    if hasattr(args, "phase"):
        from research.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker(storage_dir="output/experiments")
        if args.phase == "confirmation":
            ok, msg = tracker.check_confirmation_protocol()
            if not ok:
                logger.error("确认阶段协议违规: %s", msg)
            else:
                logger.info("确认阶段协议检查通过")
        tracker.log_experiment(
            config=config,
            metrics={
                "recorder_id": rid,
                "backtest_recorder_id": ba_rid,
            },
            phase=args.phase,
            description=getattr(args, "experiment", "qlib_pipeline"),
            recorder_id=rid,
            test_data_range=(
                config.get("dataset", {}).get("segments", {}).get("test", [None])[0],
                config.get("dataset", {}).get("segments", {}).get("test", [None, None])[-1],
            ) if "dataset" in config else None,
        )
        logger.info(
            "探索阶段已累计 %d 组实验（包含本次），"
            "最终报告时需使用校正后的显著性阈值",
            tracker.count_exploration(),
        )
        tracker.summary()

    return rid, ba_rid


def cmd_data(args):
    logger.info("[命令] data — 数据管理")
    if args.download:
        logger.info("[步骤 1/1] 开始从 AKShare 下载历史数据 ...")
        logger.info("  → 数据目录: %s", args.csv_dir)
        if args.sample:
            logger.info("  → 采样模式: 仅下载 %d 只样本股票", args.sample)
        from data_center.download_history import download_full_history
        download_full_history(sample=args.sample)
        logger.info("[步骤 1/1] ✓ 数据下载完成")
    elif args.convert:
        logger.info("[步骤 1/1] 开始 CSV → Qlib bin 格式转换 ...")
        logger.info("  → 源 CSV 目录: %s", args.csv_dir)
        logger.info("  → 目标 bin 目录: %s", args.qlib_dir)
        logger.info("  → 数据频率: %s", args.freq)
        if args.sample:
            logger.info("  → 采样模式: 仅转换 %d 只样本股票", args.sample)
        from run_qlib_workflow import create_qlib_bin_data
        create_qlib_bin_data(Path(args.csv_dir), Path(args.qlib_dir), sample=args.sample, freq=args.freq)
        logger.info("[步骤 1/1] ✓ 数据转换完成")
    elif args.check:
        logger.info("[步骤 1/1] 检查 Qlib bin 数据完整性 ...")
        qlib_dir = Path(args.qlib_dir)
        logger.info("  → 检查目录: %s", qlib_dir)
        for f in [qlib_dir / "calendars" / "day.txt",
                  qlib_dir / "instruments" / "all.txt"]:
            if f.exists():
                with open(f) as fh:
                    logger.info("    ✓ %s: %d 行", f.name, len(fh.readlines()))
            else:
                logger.info("    ✗ %s: 缺失!", f.name)
        fd = qlib_dir / "features"
        if fd.exists():
            logger.info("    ✓ 特征目录: %d 只股票", len(list(fd.iterdir())))
        else:
            logger.info("    ✗ 特征目录缺失")
        logger.info("[步骤 1/1] ✓ 数据检查完成")
    else:
        logger.info("请指定 --download、--convert 或 --check")


# ===== Phase 1: 稳健性基础设施 =====

def cmd_rolling(args):
    """滚动 Walk-Forward 验证"""
    logger.info("[命令] rolling — 滚动 Walk-Forward 验证")
    config = load_workflow_config(args.config)
    _log_config_summary(config)

    logger.info("[参数] 滚动窗口数: %d, 训练窗口: %.1f 年, 步长: %d 个月",
                args.n_folds, args.window_years, args.step_months)
    logger.info("[步骤 1/1] 开始滚动训练与验证 ...")

    from qlib_pipeline.rolling import walk_forward_validate
    df = walk_forward_validate(
        config, n_folds=args.n_folds,
        window_years=args.window_years,
        step_months=args.step_months,
        output_dir=args.output_dir,
    )
    logger.info("[步骤 1/1] ✓ 滚动训练完成: %d 个窗口", len(df))
    logger.info("  → 结果已保存到: %s", args.output_dir)


def cmd_drift(args):
    """特征漂移 + 概念漂移检测"""
    logger.info("[命令] drift — 特征漂移检测 (PSI)")
    config = load_workflow_config(args.config)
    _log_config_summary(config)

    logger.info("[步骤 1/3] 初始化 Qlib 环境并构建数据集 ...")
    init_qlib_env(config)
    task = build_task(config)
    from qlib.utils import init_instance_by_config
    dataset = init_instance_by_config(task["dataset"])
    logger.info("[步骤 1/3] ✓ 数据集构建完成")

    try:
        logger.info("[步骤 2/3] 准备训练集与测试集特征 ...")
        train_data = dataset.prepare("train", col_set=["feature", "label"])
        test_data = dataset.prepare("test", col_set=["feature", "label"])
        train_df = train_data["feature"]
        test_df = test_data["feature"]
        logger.info("  → 训练集样本: %d, 测试集样本: %d", len(train_df), len(test_df))
        logger.info("  → 特征维度: %d", train_df.shape[1])
        logger.info("[步骤 2/3] ✓ 数据准备完成")

        logger.info("[步骤 3/3] 计算 PSI 特征漂移指标 ...")
        from qlib_pipeline.drift import compute_psi_dataframe, generate_drift_report, detect_concept_drift
        psi_df = compute_psi_dataframe(train_df, test_df)

        # 用真实模型预测计算 IC 序列，再传给概念漂移检测
        from qlib_pipeline.ic_stability import get_ic_series
        from qlib.utils import init_instance_by_config
        model = init_instance_by_config(task["model"])
        model.fit(dataset)
        ic_series_drift = get_ic_series(model, dataset)
        drift_df = detect_concept_drift(ic_series_drift)
        logger.info("  → 概念漂移检测: %d 个预警", int(drift_df["drift_warning"].sum()))

        # TODO: feature_stability() 需要跨折特征重要性列表，依赖 rolling.py 改造完成后接入
        report = generate_drift_report(psi_df, drift_df, 0.0, output_dir=args.output_dir)
        logger.info("[步骤 3/3] ✓ 漂移检测完成")
        logger.info("  → 总特征数: %d, 显著漂移特征数: %d",
                    report["psi"]["n_features"],
                    report["psi"]["n_significant_drift"])
        logger.info("  → 报告已保存到: %s", args.output_dir)
    except Exception as e:
        logger.error("漂移检测失败: %s", e)
        logger.info("提示: 漂移检测需要完整的 train/test 数据集")


def cmd_ic_stability(args):
    """IC 稳定性分析"""
    logger.info("[命令] ic-stability — IC 稳定性分析 (ICIR / IC衰减 / 分层IC)")
    config = load_workflow_config(args.config)
    _log_config_summary(config)

    logger.info("[步骤 1/4] 初始化 Qlib 环境并构建数据集 ...")
    init_qlib_env(config)
    task = build_task(config)
    from qlib.utils import init_instance_by_config
    dataset = init_instance_by_config(task["dataset"])
    model = init_instance_by_config(task["model"])
    logger.info("[步骤 1/4] ✓ 数据集与模型初始化完成")

    logger.info("[步骤 2/4] 训练模型 ...")
    model.fit(dataset)
    logger.info("[步骤 2/4] ✓ 模型训练完成")

    logger.info("[步骤 3/4] 在测试集上生成预测并计算 IC 序列 ...")
    from qlib_pipeline.ic_stability import get_ic_series
    ic_series = get_ic_series(model, dataset)
    logger.info("  → 真实逐日 RankIC 序列: %d 个交易日, IC 均值=%.4f",
                 len(ic_series), ic_series.mean() if len(ic_series) > 0 else float("nan"))
    logger.info("[步骤 3/4] ✓ IC 序列计算完成")

    logger.info("[步骤 4/4] 生成 IC 稳定性报告 (ICIR / 衰减曲线 / 分层分析) ...")
    from qlib_pipeline.ic_stability import generate_ic_stability_report
    report = generate_ic_stability_report(
        ic_series,
        output_dir=args.output_dir,
    )
    logger.info("[步骤 4/4] ✓ IC 稳定性分析完成")
    logger.info("  → ICIR: %.4f", report["icir"])
    logger.info("  → 报告已保存到: %s", args.output_dir)

    # ── 实验追踪与协议检查（第 2.2 节 + 第 8.2 节）──
    if hasattr(args, "phase"):
        from research.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker(storage_dir="output/experiments")
        if args.phase == "confirmation":
            ok, msg = tracker.check_confirmation_protocol()
            if not ok:
                logger.error("确认阶段协议违规: %s", msg)
            else:
                logger.info("确认阶段协议检查通过")
        # 记录本次实验
        nw_significant = report.get("nw_significant", False)
        nw_t_stat = report.get("nw_t_stat", None)
        tracker.log_experiment(
            config=config,
            metrics={
                "ic_mean": report.get("ic_mean"),
                "icir": report.get("icir"),
                "nw_t_stat": nw_t_stat,
                "nw_significant": nw_significant,
                "n_samples": len(ic_series) if ic_series is not None else 0,
            },
            phase=args.phase,
            description=getattr(args, "experiment", "ic_stability"),
            test_data_range=(
                config.get("dataset", {}).get("segments", {}).get("test", [None])[0],
                config.get("dataset", {}).get("segments", {}).get("test", [None, None])[-1],
            ) if "dataset" in config else None,
        )
        logger.info(
            "探索阶段已累计 %d 组实验（包含本次），"
            "最终报告时需使用校正后的显著性阈值",
            tracker.count_exploration(),
        )
        tracker.summary()


# ===== Phase 3: 鲁棒性 =====

def cmd_tscv(args):
    """Purged K-Fold TSCV"""
    logger.info("[命令] tscv — Purged K-Fold 时间序列交叉验证")
    config = load_workflow_config(args.config)
    _log_config_summary(config)

    logger.info("[参数] 折数: %d, Purge 天数: %d, Embargo 天数: %d",
                args.n_splits, args.purge_days, args.embargo_days)
    logger.info("[步骤 1/1] 开始 TSCV 训练与验证 ...")

    from qlib_pipeline.tscv import run_tscv
    df = run_tscv(
        config, n_splits=args.n_splits,
        purge_days=args.purge_days, embargo_days=args.embargo_days,
        output_dir=args.output_dir,
    )
    logger.info("[步骤 1/1] ✓ TSCV 完成: %d folds", len(df))
    logger.info("  → 结果已保存到: %s", args.output_dir)


def cmd_regime(args):
    """市场阶段稳定性分析"""
    logger.info("[命令] regime — 市场阶段稳定性分析 (牛市/熊市/震荡)")
    config = load_workflow_config(args.config)
    _log_config_summary(config)

    logger.info("[步骤 1/2] 初始化 Qlib 环境并加载市场基准数据 ...")
    from qlib_pipeline.regime import classify_market_regime, regime_analysis
    import qlib
    from qlib.data import D
    init_qlib_env(config)
    try:
        # 从 bin 文件直接读取基准数据
        qlib_dir = _ds_cfg.get("qlib_dir", "D:/trae/qlib_bin")
        benchmark_code = _regime_cfg.get("benchmark_code", "SH600000")
        benchmark_field = _regime_cfg.get("benchmark_field", "close")
        benchmark_path = Path(qlib_dir) / "features" / benchmark_code / f"{benchmark_field}.day.bin"
        if benchmark_path.exists():
            with open(benchmark_path, "rb") as f:
                benchmark_data = np.fromfile(f, dtype="<f")
            start_idx = int(benchmark_data[0])
            close_values = benchmark_data[1:]
            # 读取日历
            calendar_path = Path(qlib_dir) / "calendars" / "day.txt"
            with open(calendar_path, "r") as f:
                all_dates = [line.strip() for line in f if line.strip()]
            # 构造 price Series
            dates = all_dates[start_idx:start_idx + len(close_values)]
            price = pd.Series(close_values, index=pd.to_datetime(dates)).dropna()

            # ── 层级校验 ──
            _validate_benchmark_price(price, benchmark_code, config)
        else:
            logger.error("基准文件不存在: %s", benchmark_path)
            return
        logger.info("  → 基准数据: 600000, 数据点数: %d", len(price))
        logger.info("[步骤 1/2] ✓ 基准数据加载完成")

        logger.info("[步骤 2/2] 进行市场阶段分类与稳定性分析 ...")
        regime_df = classify_market_regime(price)
        logger.info("  → 识别到 %d 个市场阶段", regime_df["regime"].nunique())

        # 训练模型并生成真实 IC 序列
        logger.info("  → 训练模型以计算真实 IC 序列 ...")
        task = build_task(config)
        from qlib.utils import init_instance_by_config
        dataset = init_instance_by_config(task["dataset"])
        model = init_instance_by_config(task["model"])
        model.fit(dataset)
        from qlib_pipeline.ic_stability import get_ic_series
        ic_series = get_ic_series(model, dataset)
        logger.info("  → 真实 RankIC 序列: %d 个交易日, IC 均值=%.4f", len(ic_series), ic_series.mean())

        from qlib_pipeline.regime import regime_analysis
        df = regime_analysis(ic_series, regime_df, output_dir=args.output_dir)
        logger.info("[步骤 2/2] ✓ 市场阶段分析完成: %d 个阶段", len(df))
        logger.info("  → 报告已保存到: %s", args.output_dir)
    except Exception as e:
        logger.error("市场阶段分析失败: %s", e)


def cmd_sensitivity(args):
    """超参数敏感性分析"""
    logger.info("[命令] sensitivity — 超参数敏感性分析")
    config = load_workflow_config(args.config)
    _log_config_summary(config)

    from qlib_pipeline.sensitivity import hyperparameter_sensitivity, run_sensitivity_suite
    if args.param:
        values = [float(v) for v in args.values.split(",")]
        logger.info("[步骤 1/1] 单参数扫描: %s = %s", args.param, values)
        df = hyperparameter_sensitivity(config, args.param, values, output_dir=args.output_dir)
        logger.info("[步骤 1/1] ✓ 完成: %s -> %d 个扫描点", args.param, len(df))
    else:
        logger.info("[步骤 1/1] 运行敏感性分析套件 (全部关键参数) ...")
        results = run_sensitivity_suite(config, output_dir=args.output_dir)
        logger.info("[步骤 1/1] ✓ 敏感性套件完成: %d 个参数", len(results))
    logger.info("  → 结果已保存到: %s", args.output_dir)


def cmd_key_years(args):
    """关键年份独立回测（支持样本外模式）"""
    logger.info("[命令] key-years — 关键年份独立回测")
    config = load_workflow_config(args.config)
    _log_config_summary(config)

    years = args.years.split(",") if args.years else None
    if years:
        logger.info("[参数] 指定年份: %s", ", ".join(years))
    else:
        logger.info("[参数] 使用默认关键年份列表")

    if args.rid:
        logger.info("[参数] 样本外模式: 使用预训练模型 %s", args.rid)
        logger.info("  → 模型在非关键年份训练，在关键年份做纯测试（未在训练中使用的市场阶段）")

    logger.info("[步骤 1/1] 开始关键年份独立回测 ...")

    from qlib_pipeline.regime import key_year_backtest
    df = key_year_backtest(config, years=years, output_dir=args.output_dir,
                           pretrained_rid=args.rid)
    logger.info("[步骤 1/1] ✓ 关键年份回测完成: %d 个年份", len(df))
    logger.info("  → 结果已保存到: %s", args.output_dir)


def cmd_update(args):
    """每日增量更新"""
    logger.info("[命令] update — 每日增量更新")
    from data_center.daily_update import daily_update
    days = args.days
    skip_bin = args.skip_bin
    workers = args.workers if args.workers else _ds_cfg.get("max_workers", 10)
    daily_update(days=days, skip_bin=skip_bin, workers=workers)


def _normalize_to_qlib_code(stock_code: str) -> str:
    """将纯数字股票代码还原为 Qlib 格式（SH/SZ 前缀）。

    print_stock_picks() 保存 CSV 时会去掉 SH/SZ 前缀以方便人工阅读，
    但 Qlib D.features() 查询需要完整格式：SH600000 / SZ000001。
    """
    code = str(stock_code).strip().zfill(6)
    if code.startswith("6"):
        return f"SH{code}"
    else:
        return f"SZ{code}"


def _validate_benchmark_price(price: pd.Series, benchmark_code: str, config: dict) -> None:
    """校验 cmd_regime 从 bin 文件直读出的基准价格数据。

    检查项：
      1. 价格序列非空且长度合理
      2. 首尾日期在 data_handler 配置范围内
      3. 无连续多日价格不变（停牌特征）
    """
    if len(price) == 0:
        raise ValueError(f"基准 {benchmark_code} 价格序列为空，bin 文件可能损坏")

    # 检查 1: 日期范围是否在配置范围内
    dh = config.get("data_handler", {})
    dh_start = dh.get("start_time")
    dh_end = dh.get("end_time")
    if dh_start and price.index.min() < pd.Timestamp(dh_start):
        logger.warning(
            "基准 %s 价格起始日期 %s 早于 data_handler.start_time %s，可能存在日历错位",
            benchmark_code, price.index.min().strftime("%Y-%m-%d"), dh_start)
    if dh_end and price.index.max() > pd.Timestamp(dh_end):
        logger.warning(
            "基准 %s 价格结束日期 %s 晚于 data_handler.end_time %s，可能存在日历错位",
            benchmark_code, price.index.max().strftime("%Y-%m-%d"), dh_end)

    # 检查 2: 连续价格不变（停牌特征）
    price_diff = price.diff().fillna(0)
    # 连续 5 个及以上交易日价格不变 → 疑似停牌
    consecutive_flat = (price_diff == 0).astype(int).groupby(
        (price_diff != 0).astype(int).cumsum()
    ).transform("sum")
    max_flat = int(consecutive_flat.max())
    if max_flat >= 5:
        flat_dates = price.index[consecutive_flat >= 5]
        logger.warning(
            "基准 %s 存在连续 %d 天价格不变（疑似停牌），涉及日期范围: %s ~ %s",
            benchmark_code, max_flat,
            flat_dates.min().strftime("%Y-%m-%d") if len(flat_dates) > 0 else "N/A",
            flat_dates.max().strftime("%Y-%m-%d") if len(flat_dates) > 0 else "N/A")

    logger.info(
        "  → 基准 %s 校验通过: %d 个数据点, %s ~ %s",
        benchmark_code, len(price),
        price.index.min().strftime("%Y-%m-%d"),
        price.index.max().strftime("%Y-%m-%d"))


def cmd_validate_picks(args):
    """验证历史选股推荐的实际表现"""
    logger.info("[命令] validate-picks — 验证历史选股推荐")

    from pathlib import Path
    picks_dir = Path(args.picks_dir)
    if not picks_dir.exists():
        logger.error("推荐目录不存在: %s", picks_dir)
        logger.info("  → 请先运行 python run.py full 生成选股推荐")
        return

    csv_files = sorted(picks_dir.glob("stock_picks_*.csv"))
    if not csv_files:
        logger.error("推荐目录下无 CSV 文件: %s", picks_dir)
        return

    logger.info("  → 找到 %d 个历史推荐文件", len(csv_files))

    all_recommendations = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            all_recommendations.append(df)
        except Exception as e:
            logger.warning("读取 %s 失败: %s", f.name, e)

    if not all_recommendations:
        logger.error("无法读取任何推荐文件")
        return

    import qlib
    from qlib.data import D
    init_qlib_env({})  # 使用默认配置初始化 Qlib

    recs_df = pd.concat(all_recommendations, ignore_index=True)
    logger.info("  → 共 %d 条推荐记录", len(recs_df))

    # 获取推荐股票的价格数据
    lookback = args.lookback_days
    from qlib_pipeline.validate import RecommendationValidator

    validator = RecommendationValidator()
    validation_results = []
    horizons = [5, 10, 20]

    for _, rec in recs_df.iterrows():
        try:
            raw_code = rec["stock_code"]
            qlib_code = _normalize_to_qlib_code(raw_code)  # 纯数字 → SH/SZ 前缀
            date = rec["date"]
            price_data = D.features(
                [qlib_code], ["$close"],
                start_time=date,
                end_time=pd.Timestamp(date) + pd.Timedelta(days=lookback + 30)
            )
            if price_data is None or price_data.empty:
                continue

            base_price = price_data.iloc[0].values[0] if len(price_data) > 0 else None
            if base_price is None:
                continue

            result = {"date": date, "stock_code": raw_code, "rank": rec.get("rank", 0), "score": rec.get("score", 0.0)}
            for h in horizons:
                if h < len(price_data):
                    future_price = price_data.iloc[h].values[0]
                    ret = (future_price - base_price) / base_price
                    result[f"ret_{h}d"] = ret
                else:
                    result[f"ret_{h}d"] = np.nan
            validation_results.append(result)
        except Exception as e:
            continue

    if not validation_results:
        logger.error("无法完成验证（推荐日期可能超出数据范围）")
        return

    val_df = pd.DataFrame(validation_results)
    report = validator.generate_validation_report(val_df, output_dir=args.output_dir)

    logger.info("=" * 60)
    logger.info("验证报告摘要:")
    for horizon, stats in report.get("by_horizon", {}).items():
        logger.info("  %s: 命中率=%.2f%%, 平均收益=%.4f, 样本数=%d",
                     horizon,
                     stats.get("hit_rate", 0) * 100,
                     stats.get("avg_return", 0),
                     stats.get("count", 0))
    logger.info("=" * 60)
    logger.info("  → 详细报告已保存到: %s", args.output_dir)


def cmd_pick(args):
    """选股推荐：从已有回测结果中提取 Top-K 股票"""
    logger.info("[命令] pick — 选股推荐 (recorder_id=%s)", args.rid)
    config = load_workflow_config(args.config)
    init_qlib_env(config)

    logger.info("[步骤 1/2] 加载回测结果 ...")
    from qlib.workflow import R
    experiment_bt = f"{args.experiment}_backtest"
    recorder = R.get_recorder(recorder_id=args.rid, experiment_name=experiment_bt)
    pred_df = recorder.load_object("pred.pkl")
    report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
    # 输出预测日期范围
    if not pred_df.empty and "datetime" in pred_df.index.names:
        pred_dates = pred_df.index.get_level_values("datetime")
        logger.info("[步骤 1/2] ✓ 加载完成, 预测记录数: %d, 预测日期范围: %s ~ %s",
                    len(pred_df),
                    str(pred_dates.min())[:10] if len(pred_dates) > 0 else "N/A",
                    str(pred_dates.max())[:10] if len(pred_dates) > 0 else "N/A")
    else:
        logger.info("[步骤 1/2] ✓ 加载完成, 预测记录数: %d", len(pred_df))

    logger.info("[步骤 2/2] 输出回测摘要并生成 Top-%d 选股推荐 ...", args.topk)
    print_summary(report_normal_df, analysis_df, config=config)
    if args.date:
        logger.info("  → 指定日期: %s", args.date)
    picks = print_stock_picks(pred_df, top_k=args.topk, date=args.date, output_dir=args.output_dir,
                           recorder_id=args.rid, config_snapshot=config)
    logger.info("[步骤 2/2] ✓ 选股推荐已保存到: %s", args.output_dir)
    return picks


def cmd_optuna(args):
    """Optuna 超参数搜索（Phase 4 模型能力恢复）"""
    logger.info("[命令] optuna — 超参数搜索 (n_trials=%d, timeout=%ds)", args.n_trials, args.timeout)
    config = load_workflow_config(args.config)

    from tuning.optuna_search import run_optuna_search
    result = run_optuna_search(
        config=config,
        n_trials=args.n_trials,
        timeout=args.timeout,
        study_name=getattr(args, "study_name", "lgb_optimization"),
        output_dir=args.output_dir,
        phase=getattr(args, "phase", "exploration"),
    )

    if "error" in result:
        logger.error("Optuna 搜索失败: %s", result["error"])
    else:
        logger.info("Optuna 搜索完成, 最佳 IC=%.6f, 结果已保存到: %s",
                    result["best_ic"], result["output_dir"])


def cmd_explain(args):
    """SHAP 模型可解释性分析（Phase 4 模型能力恢复）"""
    logger.info("[命令] explain — SHAP 可解释性分析")
    config = load_workflow_config(args.config)
    init_qlib_env(config)

    from research.explain import generate_shap_report, get_shap_feature_importance, explain_single_prediction
    from qlib_pipeline.model import Model
    from qlib_pipeline.dataset import Dataset

    # 构建数据集
    dataset = Dataset(config).build()

    # 加载已训练模型（优先从 MLflow 加载，否则重新训练）
    experiment = getattr(args, "experiment", "qlib_pipeline")
    if getattr(args, "rid", None):
        from qlib.workflow import R
        recorder = R.get_recorder(recorder_id=args.rid, experiment_name=experiment)
        model = recorder.load_object("model.pkl")
    else:
        model = Model(config).build()
        model.fit(dataset)

    report = generate_shap_report(
        model=model,
        dataset=dataset,
        output_dir=args.output_dir,
        segment=getattr(args, "segment", "test"),
        max_samples=getattr(args, "max_samples", 2000),
    )

    if "error" in report:
        logger.error("SHAP 分析失败: %s", report["error"])
        return

    logger.info("SHAP 分析完成, 前10特征: %s", ", ".join(report.get("top_10_features", [])[:5]))

    # --full 模式：输出特征重要性排名和单样本解释
    if getattr(args, "full", False):
        logger.info("\n[完整报告] 特征重要性排名:")
        importance = get_shap_feature_importance(report.get("shap_values"), report.get("feature_names"))
        if importance:
            for rank, (feat, imp) in enumerate(importance.items()):
                logger.info("  %2d. %-30s %.4f", rank + 1, feat, imp)

        logger.info("\n[完整报告] 单样本解释 (第一个样本):")
        try:
            explanation = explain_single_prediction(
                report.get("shap_values"), report.get("feature_names"), sample_idx=0,
            )
            if explanation:
                for feat, contrib in explanation.items():
                    direction = "+" if contrib > 0 else ""
                    logger.info("  %-30s %s%.4f", feat, direction, contrib)
        except Exception as e:
            logger.info("  (单样本解释失败: %s)", e)


# ===== Phase 4: 风险诊断（风格暴露 / 收益归因 / 容量）=====

def cmd_risk(args):
    """风格暴露诊断"""
    logger.info("[命令] risk — 风格暴露诊断")
    logger.info("=" * 60)
    logger.info("⚠ 此功能依赖 Phase A 基本面数据（$roe, $pb_inv, $mom_12m 等）")
    logger.info("  数据未就绪时将优雅降级。")
    logger.info("  完成 Phase A 后，此命令将自动产出完整的风格暴露报告。")
    logger.info("=" * 60)

    config = load_workflow_config(args.config)
    init_qlib_env(config)

    from research.risk_model import (
        STYLE_FACTORS, get_style_factor_names,
        calculate_style_exposures, check_style_constraints,
        get_industry_exposures, check_industry_constraints,
    )

    logger.info("已注册风格维度: %s", ", ".join(
        f"{k}({v['name']}, feature={v['feature']})" for k, v in STYLE_FACTORS.items()
    ))

    if not args.rid:
        logger.info("未指定 --rid，仅展示诊断框架。")
        logger.info("用法: python run.py risk --rid <backtest_recorder_id>")
        return

    logger.info("尝试从回测结果 %s 加载持仓数据并诊断风险暴露 ...", args.rid)
    try:
        from qlib.workflow import R
        from qlib.data import D
        import pandas as pd

        # 从回测记录中加载持仓和预测
        experiment_bt = "qlib_pipeline_backtest"
        recorder = R.get_recorder(recorder_id=args.rid, experiment_name=experiment_bt)
        pred_df = recorder.load_object("pred.pkl")

        if pred_df is None or pred_df.empty:
            logger.warning("  → 预测数据为空，无法诊断")
            return

        logger.info("  ✓ 预测数据加载成功 (%d 行)", len(pred_df))

        # 提取最新持仓
        pick_date = pred_df.index.get_level_values("datetime")[-1] if hasattr(pred_df.index, "get_level_values") else pred_df.index[-1][0]
        if hasattr(pred_df, "loc"):
            latest_weights = pred_df.loc[pick_date].abs() / pred_df.loc[pick_date].abs().sum()
        else:
            latest_weights = pd.Series(1.0 / len(pred_df), index=pred_df.index)

        # 风格因子暴露诊断
        needed_features = [v["feature"] for v in STYLE_FACTORS.values()]
        factor_names = [v["name"] for v in STYLE_FACTORS.values()]

        # 尝试查询因子值，计算暴露
        exposures = calculate_style_exposures(latest_weights, str(pick_date)[:10])

        if exposures is None:
            logger.info("  → 风格因子数据不可用（至少 3 个因子缺失）")
            logger.info("  → 需要 Phase A 将基本面数据导入 features/ 目录")
        else:
            logger.info("  ✓ 风格暴露计算完成")
            max_active = abs(exposures["active_exposure"]).max()
            all_ok = check_style_constraints(exposures)
            if all_ok:
                logger.info("  ✓ 所有风格暴露都在阈值范围内 (|active| ≤ 0.5)")
            else:
                n_violations = (~exposures["acceptable"]).sum()
                logger.warning("  ⚠ %d 个风格因子超出约束 (%d 个可用):", n_violations, len(exposures))
                for _, row in exposures[~exposures["acceptable"]].itertuples():
                    logger.warning(
                        "    %s: active=%.3f (max=0.5)",
                        getattr(row, "factor_name"), getattr(row, "active_exposure"),
                    )

        # 行业暴露检查 — 尝试运行（数据不可用则降级）
        logger.info("\n[行业暴露检查]")
        try:
            industry_exposures = get_industry_exposures(latest_weights, str(pick_date)[:10])
            if industry_exposures is not None and not industry_exposures.empty:
                logger.info("  ✓ 行业暴露计算完成 (%d 个行业)", len(industry_exposures))
                all_ok = check_industry_constraints(industry_exposures)
                if all_ok:
                    logger.info("  ✓ 所有行业暴露都在阈值范围内 (weight ≤ 20.0%%)")
                else:
                    n_violations = (~industry_exposures["acceptable"]).sum()
                    logger.warning("  ⚠ %d 个行业超出约束:", n_violations)
                    for _, row in industry_exposures[~industry_exposures["acceptable"]].head(5).itertuples():
                        logger.warning(
                            "    %s: weight=%.1f%% (max=20.0%%)",
                            getattr(row, "industry_code"), getattr(row, "weight") * 100,
                        )
            else:
                logger.info("  → 行业分类数据不可用，跳过检查")
        except Exception as e:
            logger.info("  → 行业数据不可用，跳过检查: %s", e)

    except Exception as e:
        logger.warning("  → 加载回测记录失败: %s", e)
        logger.info("  → 请确认 recorder_id 正确且回测已完成")


def cmd_attribution(args):
    """收益归因分析"""
    logger.info("[命令] attribution — 收益归因分析")
    logger.info("=" * 60)
    logger.info("⚠ 此功能依赖 Phase A 基本面数据（风格因子值）")
    logger.info("  数据未就绪时将优雅降级。")
    logger.info("=" * 60)

    config = load_workflow_config(args.config)
    init_qlib_env(config)

    from research.attribution import decompose_excess_return, attribution_report

    if not args.rid:
        logger.info("未指定 --rid，仅展示归因分析框架。")
        logger.info("用法: python run.py attribution --rid <backtest_recorder_id>")
        return

    logger.info("尝试从回测结果 %s 加载收益数据并执行归因分析 ...", args.rid)
    try:
        from qlib.workflow import R

        experiment_bt = "qlib_pipeline_backtest"
        recorder = R.get_recorder(recorder_id=args.rid, experiment_name=experiment_bt)
        report_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")

        if report_df is None or report_df.empty:
            logger.warning("  → 回测报告为空，无法执行归因分析")
            return

        logger.info("  ✓ 回测报告加载成功 (%d 行)", len(report_df))

        # 尝试提取收益序列
        port_ret = report_df.get("return")
        bench_ret = report_df.get("benchmark_return")
        if port_ret is None or bench_ret is None:
            logger.info("  → 回测报告缺少收益列（return/benchmark_return），无法执行归因")
            logger.info("  → 需要 Phase A 风格因子数据后，方可计算完整归因")
            return

        # 尝试加载风格因子数据
        try:
            from research.risk_model import STYLE_FACTORS
            from qlib.data import D

            needed_features = [v["feature"] for v in STYLE_FACTORS.values()]
            sample_stocks = list(port_ret.index.get_level_values("instrument").unique())[:10]
            factor_values = {}
            for feat in needed_features:
                try:
                    fv = D.features(
                        list(sample_stocks), [feat],
                        start_time=str(port_ret.index.get_level_values("datetime").min())[:10],
                        end_time=str(port_ret.index.get_level_values("datetime").max())[:10],
                    )
                    if fv is not None and not fv.empty:
                        factor_values[feat] = fv
                except Exception:
                    pass

            if len(factor_values) < 3:
                logger.info(
                    "  → Phase A 风格因子数据不足（可用 %d/%d），跳过归因计算",
                    len(factor_values), len(needed_features),
                )
                logger.info("  → 需要以下因子: %s", ", ".join(needed_features))
                return

            # 实际执行归因计算
            from research.attribution import calculate_factor_returns
            decomposition = decompose_excess_return(
                port_ret, bench_ret,
                factor_exposures_df=pd.DataFrame(factor_values),
                factor_returns_df=calculate_factor_returns(
                    pd.DataFrame(factor_values), port_ret.to_frame(),
                ),
            )

            if not decomposition.empty:
                report = attribution_report(decomposition)
                logger.info("  ✓ 归因分析完成")
                print("\n" + report)
            else:
                logger.info("  → 归因分解结果为空，可能日期对齐失败")

        except Exception as e:
            logger.info("  → 风格因子数据不可用，跳过归因计算: %s", e)
            logger.info("  → 完成 Phase A 基本面数据接入后，此命令将自动产出完整归因报告")

    except Exception as e:
        logger.warning("  → 加载回测记录失败: %s", e)
        logger.info("  → 请确认 recorder_id 正确且回测已完成")


def cmd_capacity(args):
    """组合容量检查"""
    logger.info("[命令] capacity — 组合容量检查")
    logger.info("=" * 60)
    logger.info("⚠ 此功能依赖 Phase A 成交额数据（日均成交额）")
    logger.info("  数据未就绪时将优雅降级。")
    logger.info("  管理规模: %.0e 元", args.account)
    logger.info("=" * 60)

    from research.capacity import check_portfolio_capacity, check_single_stock_capacity, get_average_volume_from_data

    if not args.rid:
        logger.info("未指定 --rid，仅展示容量检查框架。")
        logger.info("用法: python run.py capacity --rid <backtest_recorder_id> --account 100000000")
        return

    logger.info("尝试从回测结果 %s 加载持仓数据并执行容量检查 ...", args.rid)
    try:
        from qlib.workflow import R
        from qlib.data import D

        experiment_bt = "qlib_pipeline_backtest"
        recorder = R.get_recorder(recorder_id=args.rid, experiment_name=experiment_bt)
        pred_df = recorder.load_object("pred.pkl")

        if pred_df is None or pred_df.empty:
            logger.warning("  → 预测数据为空，无法执行容量检查")
            return

        logger.info("  ✓ 预测数据加载成功 (%d 行)", len(pred_df))

        # 提取最新日期的持仓股票
        if hasattr(pred_df.index, "get_level_values"):
            pick_date = pred_df.index.get_level_values("datetime")[-1]
            pick_stocks = list(pred_df.loc[pick_date].index)[:50]
        else:
            pick_stocks = list(pred_df.index)[:50]

        # 尝试获取日均成交额
        try:
            # 回看60个交易日获取成交量数据
            vol_start = str(pd.Timestamp(pick_date) - pd.Timedelta(days=120))
            vol_end = str(pick_date)
            volume_data = {}
            for stock in pick_stocks[:10]:  # 采样检查
                try:
                    vol = D.features([stock], ["$volume"], start_time=vol_start, end_time=vol_end)
                    if vol is not None and not vol.empty:
                        volume_data[stock] = float(vol.mean().iloc[0]) if vol.shape[1] > 0 else 0.0
                except Exception:
                    pass

            if len(volume_data) < 3:
                logger.info(
                    "  → 日均成交额数据不足（可用 %d/%d 只），跳过容量检查",
                    len(volume_data), min(10, len(pick_stocks)),
                )
                logger.info("  → 需要 Phase A 成交额数据（$volume/日均成交额）")
                return

            # 实际执行容量检查
            volume_series = pd.Series(volume_data)
            weights = pd.Series(1.0 / len(pick_stocks), index=pick_stocks[:len(volume_series)])
            result_df, all_ok = check_portfolio_capacity(
                weights, volume_series, account_value=args.account,
            )

            logger.info("  ✓ 容量检查完成 (%d 只股票)", len(result_df))
            if all_ok:
                logger.info("  ✓ 所有股票通过流动性检查 (持仓 ≤ 5%% 日均成交额)")
            else:
                n_fail = (~result_df["acceptable"]).sum()
                logger.warning("  ⚠ %d 只股票超出流动性约束", n_fail)
                for _, row in result_df[~result_df["acceptable"]].head(5).iterrows():
                    logger.warning(
                        "    %s: 占用 %.1f%% (limit=%.1f%%)",
                        row["stock_code"], row["usage_pct"] * 100, row["max_usage_pct"] * 100,
                    )

        except Exception as e:
            logger.info("  → 成交额数据不可用，跳过容量检查: %s", e)
            logger.info("  → 完成 Phase A 成交额数据接入后，此命令将自动产出完整容量报告")

    except Exception as e:
        logger.warning("  → 加载回测记录失败: %s", e)
        logger.info("  → 请确认 recorder_id 正确且回测已完成")


def cmd_benchmark(args):
    """第 8.4 节强制性对比实验"""
    logger.info("[命令] benchmark — 第 8.4 节强制性对比实验")
    config = load_workflow_config(args.config)
    init_qlib_env(config)

    experiments = None
    if args.experiments:
        experiments = [e.strip() for e in args.experiments.split(",")]
        logger.info("[参数] 指定实验: %s", ", ".join(experiments))
    else:
        logger.info("[参数] 运行全部实验 (E1-E5)")

    from research.benchmark import run_benchmark_suite
    report = run_benchmark_suite(config, experiments=experiments, output_dir=args.output_dir)
    logger.info("实验完成，报告已保存到: %s", args.output_dir)


def main():
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    args = parse_args()
    cmd_map = {
        "train": cmd_train, "backtest": cmd_backtest, "full": cmd_full,
        "pick": cmd_pick,
        "data": cmd_data, "update": cmd_update,
        "rolling": cmd_rolling, "drift": cmd_drift, "ic-stability": cmd_ic_stability,
        "tscv": cmd_tscv, "regime": cmd_regime,
        "sensitivity": cmd_sensitivity, "key-years": cmd_key_years,
        "validate-picks": cmd_validate_picks,
        "optuna": cmd_optuna, "explain": cmd_explain,  # Phase 4 模型能力恢复
        "risk": cmd_risk, "attribution": cmd_attribution, "capacity": cmd_capacity,  # Phase 4 风险诊断
        "benchmark": cmd_benchmark,  # Phase 4 强制性对比实验
    }
    fn = cmd_map.get(args.command)
    if fn:
        fn(args)
    else:
        logger.info("未指定子命令，默认执行 full 流程")
        args.config = None; args.handler = None
        args.loss = None; args.topk = None; args.pick_topk = _strategy_cfg.get("top_k", 30)
        args.experiment = "qlib_pipeline"; args.output_dir = _output_cfg.get("charts", "output/qlib_charts")
        cmd_full(args)


if __name__ == "__main__":
    main()