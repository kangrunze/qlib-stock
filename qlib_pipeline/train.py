# -*- coding: utf-8 -*-
"""
Qlib Training Pipeline - train.py

完整训练流程（Qlib 标准 Workflow）：
  1. Init Qlib with data
  2. Create dataset (Alpha158/Alpha360)
  3. Train model (LightGBM via Qlib LGBModel)
  4. Save model + metadata
  5. (Optional) Run prediction and backtest

模型超参数配置在 workflow_config.yaml 的 qlib_lgb 段，本模块仅负责 Workflow 编排。
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

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


class CSMedianSubtract:
    """截面中位数相减处理器（自定义 Qlib Processor）。

    对每个交易日的所有股票，用该股票的 label 值减去当天全部股票 label 的中位数。
    这是 xs_ret_Nd 标签的标准实现：xs_ret = ret - 截面中位数(ret)。

    与 CSZScoreNorm 的区别：
      - CSZScoreNorm: (x - mean) / std  —— 均值代替中位数，且多做一次 std 缩放
      - CSMedianSubtract: x - median    —— 严格按标签定义，只用中位数不减 std

    作为 Qlib learn_processor 使用（在 handler 的 learn_processors 中配置），
    只作用于 label 字段组，不影响 feature。
    """

    def __init__(self, fields_group="label"):
        self.fields_group = fields_group

    def fit(self, df=None):
        """学习处理参数（截面中位数减法无需学习，空实现）。"""
        pass

    def __call__(self, df):
        from qlib.data.dataset.processor import get_group_columns
        # 获取字段组对应的列
        if not isinstance(self.fields_group, list):
            groups = [self.fields_group]
        else:
            groups = self.fields_group
        with pd.option_context("mode.chained_assignment", None):
            for g in groups:
                try:
                    cols = get_group_columns(df, g)
                except (KeyError, TypeError):
                    # 非 MultiIndex 列时退化为处理所有列
                    cols = df.columns
                # 按日期分组，每组减去该日截面中位数
                df[cols] = df[cols].groupby("datetime", group_keys=False).apply(
                    lambda x: x - x.median()
                )
        return df

    def is_for_infer(self) -> bool:
        return True

    def readonly(self) -> bool:
        return False


def _build_label_config(label_name: str) -> list:
    """
    根据 labels.primary 名称构造 Qlib label 表达式。

    支持的 label 类型:
      - ret_Nd:        Ref($close, -N)/Ref($close, -1) - 1  (未来N日绝对收益率)
      - xs_ret_Nd:     Ref($close, -N)/Ref($close, -1) - 1 - CS-Median(同表达式)  (相对中位数)
      - alpha_Nd:      Ref($close, -N)/Ref($close, -1) - 1 - 基准收益 (依赖 Phase C)
      - up_down_Nd:    If(Ref($close, -N)/Ref($close, -1) > 1, 1, 0)  (方向判断)

    Args:
        label_name: label 名称，如 "ret_20d", "xs_ret_60d"

    Returns:
        Qlib 表达式列表，如 ["Ref($close, -20)/Ref($close, -1) - 1"]
    """
    import re

    # 解析 label 名称: 前缀 + horizon
    m = re.match(r"^(ret|xs_ret|alpha|up_down)_(\d+)d$", label_name)
    if not m:
        logger.warning("未识别的 label 名称 '%s'，使用默认 2 日收益率", label_name)
        return ["Ref($close, -2)/Ref($close, -1) - 1"]

    prefix = m.group(1)
    horizon = int(m.group(2))

    # 基础收益率表达式
    base_expr = f"Ref($close, -{horizon})/Ref($close, -1) - 1"

    if prefix == "ret":
        return [base_expr]
    elif prefix == "xs_ret":
        # 相对全池中位数: xs_ret = ret - 截面中位数(ret)
        # 返回基础收益率表达式，截面中位数减法通过 learn_processors 的
        # CSMedianSubtract 实现（在 build_task 中注入），严格按标签定义
        logger.info("xs_ret_%dd: 使用基础收益率 + CSMedianSubtract 实现截面中位数减法", horizon)
        return [base_expr]
    elif prefix == "alpha":
        # 相对基准超额: 需要 Phase C 基准数据，暂时回退到基础收益率
        logger.warning("alpha_%dd 需要 Phase C 基准数据，暂时使用基础收益率", horizon)
        return [base_expr]
    elif prefix == "up_down":
        # 方向判断: 1 if 涨 else 0
        return [f"If({base_expr} > 0, 1, 0)"]
    else:
        return [base_expr]


def _get_st_stocks(config: dict) -> set:
    """获取当前 ST 股票集合（Qlib 格式，如 'sh600000'）。

    优先从 DuckDB store 读取；不可用时返回空集合并记录警告。

    已知限制：is_st 基于当前股票名称判断，非 PIT 状态。
    真正的 PIT ST 过滤需要历史股票名称数据，当前数据源不支持。
    """
    try:
        from data_center.duckdb_store import DuckDBStore
        store = DuckDBStore()
        result = store.conn.execute(
            "SELECT DISTINCT code FROM daily_data WHERE is_st = true"
        ).fetchdf()

        def _to_qlib_code(code):
            code = str(code).zfill(6)
            if code.startswith(("60", "68", "90", "11", "13")):
                return f"sh{code}"
            elif code.startswith(("00", "30", "12", "15")):
                return f"sz{code}"
            elif code.startswith(("83", "87", "43", "92", "88")):
                return f"bj{code}"
            return f"sh{code}"

        return set(_to_qlib_code(c) for c in result["code"].tolist())
    except Exception as e:
        logger.debug("无法从 DuckDB 获取 ST 股票列表: %s", e)
        return set()


def _build_stock_universe_instruments(config: dict) -> Optional[str]:
    """根据 stock_universe 配置生成过滤后的 instruments 文件。

    读取 config["stock_universe"]，对 base instruments 文件应用：
      - exclude_boards: 按板块剔除（北交所 bj 前缀）
      - exclude_st: 剔除当前 ST 股票（静态，非 PIT）
      - min_listed_days: 将每只股票的 start_date 后移 N 个交易日（PIT 过滤）
      - exclude_suspended: 由 handler 默认的 DropnaProcessor 部分覆盖
        （停牌股特征为 NaN 会被丢弃），此处不做额外处理

    将过滤后的 instruments 写入临时文件，返回 market 名称。
    若 stock_universe 配置不存在或过滤失败，返回 None。

    Args:
        config: workflow config dict

    Returns:
        market 名称（如 "filtered_all"）或 None
    """
    univ = config.get("stock_universe")
    if not univ:
        return None

    dh_cfg = config.get("data_handler", {})
    base_market = dh_cfg.get("instruments", "all")
    if not isinstance(base_market, str):
        return None  # instruments 已是自定义格式，不再过滤

    import os
    provider_uri = os.environ.get("QLIB_PROVIDER_URI") or config.get("qlib", {}).get("provider_uri")
    if not provider_uri:
        return None

    instruments_dir = Path(provider_uri) / "instruments"
    base_file = instruments_dir / f"{base_market}.txt"
    if not base_file.exists():
        logger.warning("instruments 文件不存在: %s, 跳过 stock_universe 过滤", base_file)
        return None

    # 读取 instruments 文件: symbol\tstart_date\tend_date
    df = pd.read_csv(
        base_file, sep="\t", header=None,
        names=["instrument", "start_date", "end_date"],
        dtype={"instrument": str},
    )
    original_count = len(df)

    # 1. exclude_boards: 按板块剔除
    exclude_boards = univ.get("exclude_boards", [])
    if exclude_boards:
        if "北交所" in exclude_boards:
            df = df[~df["instrument"].str.lower().str.startswith("bj")]
        logger.info("stock_universe: exclude_boards=%s", exclude_boards)

    # 2. exclude_st: 剔除当前 ST 股票（静态，非 PIT）
    if univ.get("exclude_st", False):
        st_stocks = _get_st_stocks(config)
        if st_stocks:
            before_st = len(df)
            df = df[~df["instrument"].str.lower().isin(st_stocks)]
            logger.info("stock_universe: 剔除 %d 只 ST 股票", before_st - len(df))
        else:
            logger.warning(
                "stock_universe.exclude_st=true 但无法获取 ST 股票列表，跳过 ST 过滤"
            )

    # 3. min_listed_days: 将 start_date 后移 N 个交易日
    min_days = univ.get("min_listed_days", 0)
    if min_days > 0:
        try:
            from qlib.data import D
            cal = D.calendar(
                start_time=str(df["start_date"].min()),
                end_time=str(df["end_date"].max()),
            )
            if len(cal) > min_days:
                cal_list = list(cal)
                cal_ts = [pd.Timestamp(d) for d in cal_list]

                def _adjust_start(row_start, row_end):
                    stock_start = pd.Timestamp(str(row_start))
                    # 在日历中找到 >= stock_start 的位置
                    idx = None
                    for j, d in enumerate(cal_ts):
                        if d >= stock_start:
                            idx = j
                            break
                    if idx is None:
                        return str(row_start)
                    new_idx = idx + min_days
                    if new_idx < len(cal_ts):
                        return str(cal_ts[new_idx].date())
                    return str(row_end)  # 上市太晚，后移后超过 end_date

                df["start_date"] = df.apply(
                    lambda r: _adjust_start(r["start_date"], r["end_date"]), axis=1
                )
                # 剔除 start_date > end_date 的股票（上市不足 min_listed_days）
                df = df[df["start_date"] <= df["end_date"]]
                logger.info("stock_universe: min_listed_days=%d 过滤完成", min_days)
        except Exception as e:
            logger.warning("min_listed_days 过滤失败: %s, 跳过", e)

    filtered_count = len(df)
    logger.info(
        "stock_universe 过滤: %d → %d 只股票 (排除 %d)",
        original_count, filtered_count, original_count - filtered_count,
    )

    if filtered_count == 0:
        logger.error("stock_universe 过滤后股票池为空, 回退到原始 instruments")
        return None

    # 写入过滤后的 instruments 文件
    market_name = f"filtered_{base_market}"
    filtered_file = instruments_dir / f"{market_name}.txt"
    df.to_csv(filtered_file, sep="\t", header=False, index=False)
    logger.info("过滤后 instruments 已写入: %s", filtered_file)

    return market_name


def build_task(config: dict) -> dict:
    """
    Build complete Qlib task config from workflow config.

    从统一配置构建 Qlib task dict（包含 model + dataset），
    模型超参数从 workflow_config.yaml 的 qlib_lgb 段读取。
    label 由 labels.primary 决定，自动生成 Qlib 表达式。

    xs_ret_Nd 标签会注入 CSMedianSubtract learn_processor（截面中位数减法），
    其他标签（ret_Nd / alpha_Nd / up_down_Nd）不注入额外 processor。

    2026-07-10 修复：读取 config["stock_universe"]，在 handler 层真正生效
    ST/次新股/北交所过滤，而非仅停留在下载阶段的文档意图上。

    Args:
        config: workflow config dict

    Returns:
        task dict for Qlib workflow
    """
    handler = config.get("dataset", {}).get("handler", "Alpha158")
    handler_cfg = config.get("data_handler", {}).copy()

    # stock_universe 过滤：在训练/回测阶段真正生效
    # 原实现仅 download_history.py 读取 stock_universe，qlib_pipeline 完全忽略
    filtered_market = _build_stock_universe_instruments(config)
    if filtered_market:
        handler_cfg["instruments"] = filtered_market
        logger.info("stock_universe 过滤已生效: instruments=%s", filtered_market)

    # 根据 labels.primary 构造 label 表达式
    labels_cfg = config.get("labels", {})
    primary_label = labels_cfg.get("primary", "ret_20d")
    label_expr = _build_label_config(primary_label)
    handler_cfg["label"] = label_expr
    logger.info("使用 label: %s → %s", primary_label, label_expr)

    # xs_ret_Nd 标签注入 CSMedianSubtract（截面中位数减法）
    # 仅影响 xs_ret_Nd 的 label，不影响 feature 标准化
    # 2026-07-10 修复：CSMedianSubtract 必须放入 learn_processors，
    #   确保训练标签也经过截面中位数减法（否则训练目标为绝对收益，推理目标为相对收益，
    #   二者不一致）。Alpha158 的默认 learn_processors = [DropnaLabel, CSZScoreNorm]，
    #   当用户未配置时，显式写入默认值 + CSMedianSubtract，避免覆盖丢失默认处理器。
    import re
    is_xs_ret = bool(re.match(r"^xs_ret_\d+d$", primary_label))
    if is_xs_ret:
        cs_median_cfg = {
            "class": "CSMedianSubtract",
            "module_path": "qlib_pipeline.train",
            "kwargs": {"fields_group": "label"},
        }
        existing_lp = handler_cfg.get("learn_processors")
        if existing_lp is not None:
            # 用户显式配置了 learn_processors → 追加 CSMedianSubtract
            if isinstance(existing_lp, list):
                handler_cfg["learn_processors"] = existing_lp + [cs_median_cfg]
            else:
                handler_cfg["learn_processors"] = [existing_lp, cs_median_cfg]
        else:
            # 用户未配置 learn_processors → 使用 Alpha158 默认值 + CSMedianSubtract
            # Alpha158 默认: DropnaLabel + CSZScoreNorm(label)
            handler_cfg["learn_processors"] = [
                {"class": "DropnaLabel"},
                {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
                cs_median_cfg,
            ]
        logger.info("xs_ret 标签注入 CSMedianSubtract（learn_processor，训练+推理一致）")

    # Data handler
    handler_cfg = {
        "class": handler,
        "module_path": "qlib.contrib.data.handler",
        "kwargs": handler_cfg,
    }

    # Segments
    segments = config.get("dataset", {}).get("segments", {})

    # Model — 使用 qlib_lgb 配置（Qlib 原生 LGBModel 或 RankLGBModel）
    model_cfg = config.get("qlib_lgb", create_rank_model())

    # loss="rank" 时自动路由到 RankLGBModel（绕过 LGBModel 的 loss 白名单校验）
    kwargs = model_cfg.get("kwargs", {})
    if kwargs.get("loss") == "rank":
        model_cfg = {
            "class": "RankLGBModel",
            "module_path": "qlib_pipeline.model",
            "kwargs": kwargs,
        }
        logger.info("检测到 loss=rank，使用 RankLGBModel (LambdaRank)")
    # 确保 LightGBM 随机种子生效（可复现训练）
    seed = config.get("experiment", {}).get("random_seed", 42)
    kwargs.setdefault("seed", seed)

    task = {
        "model": {
            "class": model_cfg.get("class", "LGBModel"),
            "module_path": model_cfg.get("module_path", "qlib.contrib.model.gbdt"),
            "kwargs": kwargs,
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

    # 设置 numpy 全局随机种子确保可复现
    import numpy as np
    seed = config.get("experiment", {}).get("random_seed", 42)
    np.random.seed(seed)

    provider_uri = os.environ.get("QLIB_PROVIDER_URI") or config.get("qlib", {}).get("provider_uri")
    if not provider_uri:
        raise ValueError("未设置 QLIB_PROVIDER_URI 环境变量，且配置文件中也未指定 qlib.provider_uri")
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
            recorder = R.get_recorder(recorder_id=train_rid, experiment_name=experiment_name)
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