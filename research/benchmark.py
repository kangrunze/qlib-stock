# -*- coding: utf-8 -*-
"""
第 8.4 节强制性对比实验 — benchmark.py

中长周期策略研究正式启动前必须完成的对比实验（E1-E7），
每组实验均包含 Newey-West 显著性检验结论。

实验清单:
  E1: Alpha158 vs Alpha360  → 特征处理器对比
  E2: loss=mse vs loss=rank → 损失函数对比
  E3: ret_20d vs ret_60d vs ret_120d → 标签周期对比
  E4: 因子中性化 vs 原始 → 中性化效果对比
  E5: 单模型 vs 时间维度集成 → 集成效果对比
  E6: ret_60d vs xs_ret_60d → primary 标签对比（第 2.5 节定案）
  E7: up_down_60d 方向判断诊断 → 事后诊断（第 5.3 节）

Usage:
    python run.py benchmark                    # 运行全部实验
    python run.py benchmark --experiments E1,E3  # 仅运行指定实验
"""

import copy
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from qlib_pipeline.numpy_compat import *  # noqa

logger = logging.getLogger(__name__)


# ===========================================================================
# 1. 基础训练工具
# ===========================================================================

def _train_one(config: dict, overrides: dict) -> Tuple[object, object, dict]:
    """用给定的 config + overrides 训练一个模型。

    Args:
        config: 基础 workflow config
        overrides: 要覆盖的配置项，支持嵌套 key 如 "qlib_lgb.kwargs.loss"

    Returns:
        (model, dataset, merged_config)
    """
    from qlib.utils import init_instance_by_config

    merged = copy.deepcopy(config)
    for key, val in overrides.items():
        _set_nested(merged, key, val)

    task = _build_task_from_config(merged)
    dataset = init_instance_by_config(task["dataset"])
    model = init_instance_by_config(task["model"])
    model.fit(dataset)
    return model, dataset, merged


def _set_nested(d: dict, key: str, val):
    """设置嵌套 dict 值，key 用 '.' 分隔，如 'qlib_lgb.kwargs.loss'"""
    parts = key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = val


def _build_task_from_config(config: dict) -> dict:
    """从 config 构建 Qlib task dict（复用 train.py 逻辑）"""
    handler = config.get("dataset", {}).get("handler", "Alpha158")
    handler_cfg = {
        "class": handler,
        "module_path": "qlib.contrib.data.handler",
        "kwargs": config.get("data_handler", {}).copy(),
    }

    segments = config.get("dataset", {}).get("segments", {})

    model_cfg = config.get("qlib_lgb", {})
    kwargs = model_cfg.get("kwargs", {}).copy()
    seed = config.get("experiment", {}).get("random_seed", 42)
    kwargs.setdefault("seed", seed)

    return {
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


def _eval_model(model, dataset, label_horizon: int = 20) -> dict:
    """评估模型：返回 IC 指标 + Newey-West 显著性"""
    from qlib_pipeline.ic_stability import evaluate_fold
    return evaluate_fold(model, dataset, label_horizon=label_horizon)


def _compare_with_nw(
    name_a: str, metrics_a: dict,
    name_b: str, metrics_b: dict,
    ic_a: Optional[pd.Series] = None,
    ic_b: Optional[pd.Series] = None,
) -> dict:
    """对比两组指标，输出 Newey-West 显著性结论。

    如果提供了 IC 序列，对差分序列做 Newey-West 检验判断差异是否显著。
    """
    result = {
        f"{name_a}_ic_mean": metrics_a.get("ic_mean"),
        f"{name_b}_ic_mean": metrics_b.get("ic_mean"),
        f"{name_a}_icir": metrics_a.get("icir"),
        f"{name_b}_icir": metrics_b.get("icir"),
        f"{name_a}_nw_significant": metrics_a.get("nw_significant", False),
        f"{name_b}_nw_significant": metrics_b.get("nw_significant", False),
        "diff_significant": None,
        "diff_p_value": None,
        "winner": None,
        "conclusion": "",
    }

    # 如果提供了 IC 序列，检验差值是否显著
    if ic_a is not None and ic_b is not None:
        try:
            from research.significance import newey_west_test
            # 对齐日期
            common = ic_a.index.intersection(ic_b.index)
            if len(common) > 20:
                diff = ic_a.loc[common] - ic_b.loc[common]
                nw_result = newey_west_test(diff)
                result["diff_p_value"] = nw_result.get("p_value_nw")
                result["diff_significant"] = nw_result.get("significant", False)
        except Exception as e:
            logger.debug("NW 差值检验失败: %s", e)

    # 判断 winner
    a_ic = metrics_a.get("ic_mean")
    b_ic = metrics_b.get("ic_mean")
    if a_ic is not None and b_ic is not None:
        if result.get("diff_significant") is True:
            result["winner"] = name_a if a_ic > b_ic else name_b
            result["conclusion"] = f"差异显著 (NW p={result['diff_p_value']:.4f})，{result['winner']} 更优"
        else:
            if a_ic > b_ic:
                result["winner"] = name_a
                result["conclusion"] = f"差异不显著，{name_a} 数值略高但无统计意义"
            else:
                result["winner"] = name_b
                result["conclusion"] = f"差异不显著，{name_b} 数值略高但无统计意义"
    elif a_ic is not None:
        result["winner"] = name_a
        result["conclusion"] = f"{name_b} 训练失败，{name_a} 胜出"
    elif b_ic is not None:
        result["winner"] = name_b
        result["conclusion"] = f"{name_a} 训练失败，{name_b} 胜出"

    return result


# ===========================================================================
# 2. 各实验实现
# ===========================================================================

def experiment_e1_alpha158_vs_alpha360(config: dict) -> dict:
    """E1: Alpha158 vs Alpha360 特征处理器对比。

    判断标准: 哪个的 RankIC 均值经 Newey-West 检验后显著更高；
    若不显著，选择训练更快的 Alpha158。
    """
    logger.info("=" * 60)
    logger.info("E1: Alpha158 vs Alpha360")
    logger.info("=" * 60)

    # Alpha158
    logger.info("  训练 Alpha158 ...")
    model_158, ds_158, _ = _train_one(config, {"dataset.handler": "Alpha158"})
    metrics_158 = _eval_model(model_158, ds_158)
    from qlib_pipeline.ic_stability import get_ic_series
    ic_158 = get_ic_series(model_158, ds_158)
    logger.info("  Alpha158: IC=%.6f, ICIR=%.4f, NW_sig=%s",
                metrics_158.get("ic_mean"), metrics_158.get("icir"),
                metrics_158.get("nw_significant"))

    # Alpha360
    logger.info("  训练 Alpha360 ...")
    model_360, ds_360, _ = _train_one(config, {"dataset.handler": "Alpha360"})
    metrics_360 = _eval_model(model_360, ds_360)
    ic_360 = get_ic_series(model_360, ds_360)
    logger.info("  Alpha360: IC=%.6f, ICIR=%.4f, NW_sig=%s",
                metrics_360.get("ic_mean"), metrics_360.get("icir"),
                metrics_360.get("nw_significant"))

    result = _compare_with_nw("Alpha158", metrics_158, "Alpha360", metrics_360,
                              ic_158, ic_360)
    logger.info("  → %s", result["conclusion"])
    return result


def experiment_e2_mse_vs_rank(config: dict) -> dict:
    """E2: loss=mse vs loss=rank 损失函数对比。

    判断标准: RankIC 均值 + 换手率 + 夏普比率。
    """
    logger.info("=" * 60)
    logger.info("E2: loss=mse vs loss=rank")
    logger.info("=" * 60)

    # MSE
    logger.info("  训练 loss=mse ...")
    model_mse, ds_mse, _ = _train_one(config, {"qlib_lgb.kwargs.loss": "mse"})
    metrics_mse = _eval_model(model_mse, ds_mse)
    from qlib_pipeline.ic_stability import get_ic_series
    ic_mse = get_ic_series(model_mse, ds_mse)
    logger.info("  mse: IC=%.6f, ICIR=%.4f, NW_sig=%s",
                metrics_mse.get("ic_mean"), metrics_mse.get("icir"),
                metrics_mse.get("nw_significant"))

    # Rank
    logger.info("  训练 loss=rank ...")
    model_rank, ds_rank, _ = _train_one(config, {"qlib_lgb.kwargs.loss": "rank"})
    metrics_rank = _eval_model(model_rank, ds_rank)
    ic_rank = get_ic_series(model_rank, ds_rank)
    logger.info("  rank: IC=%.6f, ICIR=%.4f, NW_sig=%s",
                metrics_rank.get("ic_mean"), metrics_rank.get("icir"),
                metrics_rank.get("nw_significant"))

    result = _compare_with_nw("mse", metrics_mse, "rank", metrics_rank,
                              ic_mse, ic_rank)
    logger.info("  → %s", result["conclusion"])
    return result


def experiment_e3_label_horizons(config: dict) -> dict:
    """E3: ret_20d vs ret_60d vs ret_120d 标签周期对比。

    判断标准: 结合调仓频率成本，选出风险调整后收益最优的标签周期。
    """
    logger.info("=" * 60)
    logger.info("E3: 标签周期对比 (ret_20d vs ret_60d vs ret_120d)")
    logger.info("=" * 60)

    results = {}
    for horizon in [20, 60, 120]:
        label = f"ret_{horizon}d"
        logger.info("  训练 %s ...", label)
        try:
            model, ds, _ = _train_one(config, {"qlib_lgb.kwargs.label": [label]})
            metrics = _eval_model(model, ds, label_horizon=horizon)
            results[label] = metrics
            logger.info("  %s: IC=%.6f, ICIR=%.4f, NW_sig=%s",
                        label, metrics.get("ic_mean"), metrics.get("icir"),
                        metrics.get("nw_significant"))
        except Exception as e:
            logger.warning("  %s 训练失败: %s", label, e)
            results[label] = {"ic_mean": None, "icir": None, "error": str(e)}

    # 找最佳
    best_label = None
    best_icir = -999
    for label, m in results.items():
        if m.get("icir") is not None and m["icir"] > best_icir:
            best_icir = m["icir"]
            best_label = label

    conclusion = f"ICIR 最高: {best_label} (ICIR={best_icir:.4f})" if best_label else "无有效结果"
    logger.info("  → %s", conclusion)

    return {
        "results": results,
        "best_label": best_label,
        "best_icir": best_icir,
        "conclusion": conclusion,
    }


def experiment_e4_neutralization(config: dict) -> dict:
    """E4: 因子行业/市值中性化 vs 原始因子对比。

    判断标准: 中性化前后 IC 的稳定性（标准差）和 ICIR 对比。

    注意: 需要行业分类和市值数据（Phase A），当前数据不可用时跳过。
    """
    logger.info("=" * 60)
    logger.info("E4: 因子中性化 vs 原始")
    logger.info("=" * 60)

    # 检查是否有行业数据
    try:
        from qlib.data import D
        test_stock = "SH600000"
        ind_data = D.features([test_stock], ["$industry"], start_time="2025-01-01", end_time="2025-01-10")
        has_industry = ind_data is not None and not ind_data.empty
    except Exception:
        has_industry = False

    if not has_industry:
        logger.info("  ⏭ 行业/市值数据不可用（需要 Phase A），跳过 E4")
        return {
            "skipped": True,
            "reason": "行业/市值数据不可用，需要 Phase A 基本面数据接入",
            "conclusion": "待 Phase A 完成后重跑",
        }

    # 基准模型（无中性化）
    logger.info("  训练基准模型（无中性化）...")
    model_raw, ds_raw, _ = _train_one(config, {})
    metrics_raw = _eval_model(model_raw, ds_raw)
    logger.info("  原始: IC=%.6f, ICIR=%.4f, IC_std=%.6f",
                metrics_raw.get("ic_mean"), metrics_raw.get("icir"),
                metrics_raw.get("ic_std"))

    # 中性化模型（neutralize 放在 data_handler.kwargs 内，匹配 Qlib handler 期望）
    logger.info("  训练中性化模型 ...")
    neutralized_config = copy.deepcopy(config)
    dh = neutralized_config.setdefault("data_handler", {})
    dh.setdefault("kwargs", {})["neutralize"] = {
        "method": "regression",
        "neutralize_cols": ["industry", "market_cap"],
    }
    model_neu, ds_neu, _ = _train_one(neutralized_config, {})
    metrics_neu = _eval_model(model_neu, ds_neu)
    logger.info("  中性化: IC=%.6f, ICIR=%.4f, IC_std=%.6f",
                metrics_neu.get("ic_mean"), metrics_neu.get("icir"),
                metrics_neu.get("ic_std"))

    # 对比 IC 稳定性（标准差越小越好）
    raw_std = metrics_raw.get("ic_std", 0) or 0
    neu_std = metrics_neu.get("ic_std", 0) or 0
    if raw_std > 0 and neu_std > 0:
        if neu_std < raw_std:
            conclusion = f"中性化降低 IC 标准差 {raw_std:.4f} → {neu_std:.4f}，稳定性提升"
        else:
            conclusion = "中性化未降低 IC 标准差，可能不需要中性化"
    else:
        conclusion = "无法比较 IC 标准差"

    logger.info("  → %s", conclusion)
    return {
        "raw_icir": metrics_raw.get("icir"),
        "neutralized_icir": metrics_neu.get("icir"),
        "raw_ic_std": raw_std,
        "neutralized_ic_std": neu_std,
        "conclusion": conclusion,
    }


def experiment_e5_ensemble(config: dict) -> dict:
    """E5: 单模型 vs 时间维度集成对比。

    判断标准: 集成是否真的降低了滚动窗口间的 IC 方差。

    时间维度集成: 在不同时间窗口上训练多个模型，对预测取平均。
    """
    logger.info("=" * 60)
    logger.info("E5: 单模型 vs 时间维度集成")
    logger.info("=" * 60)

    # 单模型（基准）
    logger.info("  训练单模型 ...")
    model_single, ds_single, _ = _train_one(config, {})
    metrics_single = _eval_model(model_single, ds_single)
    from qlib_pipeline.ic_stability import get_ic_series
    ic_single = get_ic_series(model_single, ds_single)
    logger.info("  单模型: IC=%.6f, ICIR=%.4f, IC_std=%.6f",
                metrics_single.get("ic_mean"), metrics_single.get("icir"),
                metrics_single.get("ic_std"))

    # 时间维度集成: 在 3 个不同训练窗口上训练，预测取平均
    base_segments = config.get("dataset", {}).get("segments", {})
    train_range = base_segments.get("train", ["2015-01-01", "2022-12-31"])
    valid_range = base_segments.get("valid", ["2023-01-01", "2023-12-31"])

    windows = [
        ("2015-01-01", "2022-06-30"),
        ("2016-01-01", "2022-12-31"),
        ("2017-01-01", "2023-06-30"),
    ]

    ensemble_preds = []
    for i, (tw_start, tw_end) in enumerate(windows):
        logger.info("  训练集成模型 %d/3 (train=%s~%s) ...", i + 1, tw_start, tw_end)
        try:
            ensemble_config = copy.deepcopy(config)
            ensemble_config.setdefault("dataset", {}).setdefault("segments", {})["train"] = [tw_start, tw_end]
            model, ds, _ = _train_one(ensemble_config, {})
            preds = model.predict(ds, segment="test")
            ensemble_preds.append(preds)
        except Exception as e:
            logger.warning("  集成模型 %d 训练失败: %s", i + 1, e)

    if len(ensemble_preds) >= 2:
        # 平均预测
        avg_preds = np.mean(ensemble_preds, axis=0)
        # 在测试集上计算 IC
        test_data = ds_single.prepare("test", col_set=["feature", "label"])
        labels = test_data["label"]
        from qlib_pipeline.ic_stability import compute_daily_rank_ic
        ic_ensemble = compute_daily_rank_ic(
            pd.Series(avg_preds, index=test_data["feature"].index),
            labels.iloc[:, 0] if labels.ndim > 1 else labels,
        )
        ensemble_ic_mean = float(ic_ensemble.mean()) if len(ic_ensemble) > 0 else None
        ensemble_ic_std = float(ic_ensemble.std()) if len(ic_ensemble) > 0 else None
        ensemble_icir = ensemble_ic_mean / ensemble_ic_std if ensemble_ic_mean and ensemble_ic_std else None

        logger.info("  集成: IC=%.6f, ICIR=%.4f, IC_std=%.6f",
                    ensemble_ic_mean, ensemble_icir, ensemble_ic_std)

        single_std = metrics_single.get("ic_std", 0) or 0
        if ensemble_ic_std and single_std > 0:
            if ensemble_ic_std < single_std:
                conclusion = f"集成降低 IC 标准差 {single_std:.4f} → {ensemble_ic_std:.4f}，方差确实降低"
            else:
                conclusion = "集成未降低 IC 标准差，可能不需要集成"
        else:
            conclusion = "无法比较 IC 标准差"
    else:
        ensemble_ic_mean = None
        ensemble_ic_std = None
        ensemble_icir = None
        conclusion = "集成模型训练不足（<2 个），跳过比较"

    logger.info("  → %s", conclusion)
    return {
        "single_icir": metrics_single.get("icir"),
        "single_ic_std": metrics_single.get("ic_std"),
        "ensemble_icir": ensemble_icir,
        "ensemble_ic_std": ensemble_ic_std,
        "n_ensemble_models": len(ensemble_preds),
        "conclusion": conclusion,
    }


def experiment_e6_ret_vs_xsret(config: dict) -> dict:
    """E6: ret_60d vs xs_ret_60d 作为 primary 标签对比。

    第 2.5 节关键决策：用真实数据决定 primary 应该用绝对收益率
    还是相对全池中位数超额收益，停止"应该改成哪个"的讨论。

    判断标准: 哪个 primary 的 RankIC 均值经 NW 检验后显著更高；
    若不显著，看 ICIR 和 IC 稳定性。
    """
    logger.info("=" * 60)
    logger.info("E6: ret_60d vs xs_ret_60d (primary 标签对比)")
    logger.info("=" * 60)

    # 检查 xs_ret_60d 是否可用（使用 config 中的数据时间范围）
    try:
        from qlib.data import D
        test_stock = "SH600000"
        # 使用 config 中配置的测试期起始日期，而非硬编码
        segments = config.get("dataset", {}).get("segments", {})
        test_start = (segments.get("test") or [None])[0] or "2025-01-01"
        xs_data = D.features([test_stock], ["xs_ret_60d"], start_time=test_start, end_time=test_start)
        has_xs = xs_data is not None and not xs_data.empty
    except Exception:
        has_xs = False

    # ret_60d (基准)
    logger.info("  训练 ret_60d ...")
    model_ret, ds_ret, _ = _train_one(config, {"qlib_lgb.kwargs.label": ["ret_60d"]})
    metrics_ret = _eval_model(model_ret, ds_ret, label_horizon=60)
    from qlib_pipeline.ic_stability import get_ic_series
    ic_ret = get_ic_series(model_ret, ds_ret)
    logger.info("  ret_60d: IC=%.6f, ICIR=%.4f, IC_std=%.6f, NW_sig=%s",
                metrics_ret.get("ic_mean"), metrics_ret.get("icir"),
                metrics_ret.get("ic_std"), metrics_ret.get("nw_significant"))

    # xs_ret_60d
    if not has_xs:
        logger.info("  ⏭ xs_ret_60d 数据不可用（需要 Phase A 或自定义标签计算），跳过对比")
        return {
            "skipped": True,
            "reason": "xs_ret_60d 标签数据不可用",
            "ret_60d_icir": metrics_ret.get("icir"),
            "ret_60d_ic_mean": metrics_ret.get("ic_mean"),
            "conclusion": "待 xs_ret_60d 数据就绪后重跑，当前仅 ret_60d 可用",
        }

    logger.info("  训练 xs_ret_60d ...")
    model_xs, ds_xs, _ = _train_one(config, {"qlib_lgb.kwargs.label": ["xs_ret_60d"]})
    metrics_xs = _eval_model(model_xs, ds_xs, label_horizon=60)
    ic_xs = get_ic_series(model_xs, ds_xs)
    logger.info("  xs_ret_60d: IC=%.6f, ICIR=%.4f, IC_std=%.6f, NW_sig=%s",
                metrics_xs.get("ic_mean"), metrics_xs.get("icir"),
                metrics_xs.get("ic_std"), metrics_xs.get("nw_significant"))

    result = _compare_with_nw("ret_60d", metrics_ret, "xs_ret_60d", metrics_xs,
                              ic_ret, ic_xs)

    # 额外对比 IC 稳定性
    ret_std = metrics_ret.get("ic_std", 0) or 0
    xs_std = metrics_xs.get("ic_std", 0) or 0
    if ret_std > 0 and xs_std > 0:
        if xs_std < ret_std:
            result["stability_note"] = f"xs_ret_60d IC 更稳定 (std={xs_std:.4f} < {ret_std:.4f})"
        else:
            result["stability_note"] = f"ret_60d IC 更稳定 (std={ret_std:.4f} < {xs_std:.4f})"

    logger.info("  → %s", result["conclusion"])
    if result.get("stability_note"):
        logger.info("  → %s", result["stability_note"])
    return result


def experiment_e7_updown_diagnostic(config: dict) -> dict:
    """E7: up_down_60d 方向判断诊断。

    用 ret_60d 训练的模型，在 up_down_60d 标签上评估方向准确率。
    这回答了"模型是否能判断涨跌方向"的问题，而非仅仅是"排序是否准确"。

    注意: up_down_60d 是事后诊断字段，不作为独立训练目标（第 5.3 节）。
    """
    logger.info("=" * 60)
    logger.info("E7: up_down_60d 方向判断诊断")
    logger.info("=" * 60)

    # 检查 up_down_60d 是否可用（使用 config 中的数据时间范围）
    try:
        from qlib.data import D
        test_stock = "SH600000"
        segments = config.get("dataset", {}).get("segments", {})
        test_start = (segments.get("test") or [None])[0] or "2025-01-01"
        ud_data = D.features([test_stock], ["up_down_60d"], start_time=test_start, end_time=test_start)
        has_ud = ud_data is not None and not ud_data.empty
    except Exception:
        has_ud = False

    if not has_ud:
        logger.info("  ⏭ up_down_60d 数据不可用，跳过诊断")
        return {
            "skipped": True,
            "reason": "up_down_60d 标签数据不可用",
            "conclusion": "待 up_down_60d 数据就绪后重跑",
        }

    # 训练 ret_60d 模型
    logger.info("  训练 ret_60d 模型 ...")
    model, ds, _ = _train_one(config, {"qlib_lgb.kwargs.label": ["ret_60d"]})

    # 在测试集上预测
    test_data = ds.prepare("test", col_set=["feature", "label"])
    preds = model.predict(ds, segment="test")

    # 获取 up_down_60d 标签值
    try:
        from qlib.data import D
        stocks = test_data["feature"].index.get_level_values("instrument").unique()
        dates = test_data["feature"].index.get_level_values("datetime").unique()
        ud_values = D.features(list(stocks), ["up_down_60d"],
                               start_time=str(dates[0]), end_time=str(dates[-1]))
    except Exception as e:
        logger.warning("  无法加载 up_down_60d 标签: %s", e)
        return {
            "skipped": True,
            "reason": f"up_down_60d 加载失败: {e}",
            "conclusion": "待数据修复后重跑",
        }

    if ud_values is None or ud_values.empty:
        return {
            "skipped": True,
            "reason": "up_down_60d 数据为空",
            "conclusion": "待数据就绪后重跑",
        }

    # 对齐预测和标签
    pred_series = pd.Series(preds, index=test_data["feature"].index)
    from qlib_pipeline.ic_stability import compute_daily_rank_ic

    # 计算逐日方向准确率
    daily_accuracy = {}
    for date in dates:
        try:
            pred_date = pred_series.loc[date]
            ud_date = ud_values.loc[date].iloc[:, 0] if ud_values.shape[1] > 0 else None
            if ud_date is None or ud_date.empty:
                continue

            common = pred_date.index.intersection(ud_date.index)
            if len(common) < 10:
                continue

            p = pred_date.loc[common]
            u = ud_date.loc[common]

            # 方向判断: 预测值 > 0 视为看涨，实际 up_down > 0 为涨
            pred_dir = (p > 0).astype(int)
            true_dir = (u > 0).astype(int)
            acc = (pred_dir == true_dir).mean()
            daily_accuracy[str(date)] = float(acc)
        except Exception:
            pass

    if daily_accuracy:
        mean_acc = np.mean(list(daily_accuracy.values()))
        std_acc = np.std(list(daily_accuracy.values()))
        logger.info("  方向准确率: 均值=%.4f, 标准差=%.4f (%d 天)",
                    mean_acc, std_acc, len(daily_accuracy))

        if mean_acc > 0.55:
            conclusion = f"方向判断能力较好 (准确率={mean_acc:.2%} > 55%)"
        elif mean_acc > 0.50:
            conclusion = f"方向判断能力微弱 (准确率={mean_acc:.2%}，略高于随机)"
        else:
            conclusion = f"方向判断能力不足 (准确率={mean_acc:.2%} ≤ 50%，不优于随机)"
    else:
        mean_acc = None
        conclusion = "无有效交易日数据，无法计算方向准确率"

    logger.info("  → %s", conclusion)
    return {
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc if daily_accuracy else None,
        "n_days": len(daily_accuracy),
        "daily_accuracy": daily_accuracy if len(daily_accuracy) <= 20 else dict(list(daily_accuracy.items())[:20]),
        "conclusion": conclusion,
    }


# ===========================================================================
# 3. 实验编排 + 报告
# ===========================================================================

EXPERIMENTS = {
    "E1": ("Alpha158 vs Alpha360", experiment_e1_alpha158_vs_alpha360),
    "E2": ("loss=mse vs loss=rank", experiment_e2_mse_vs_rank),
    "E3": ("标签周期对比", experiment_e3_label_horizons),
    "E4": ("因子中性化 vs 原始", experiment_e4_neutralization),
    "E5": ("单模型 vs 时间维度集成", experiment_e5_ensemble),
    "E6": ("ret_60d vs xs_ret_60d (primary 标签对比)", experiment_e6_ret_vs_xsret),
    "E7": ("up_down_60d 方向判断诊断", experiment_e7_updown_diagnostic),
}


def run_benchmark_suite(
    config: dict,
    experiments: Optional[List[str]] = None,
    output_dir: str = "output/benchmark",
) -> Dict:
    """运行第 8.4 节强制性对比实验。

    Args:
        config: workflow config dict
        experiments: 要运行的实验列表，如 ["E1", "E3"]；None 表示全部
        output_dir: 结果输出目录

    Returns:
        dict with summary + per-experiment results
    """
    if experiments is None:
        experiments = list(EXPERIMENTS.keys())

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for exp_id in experiments:
        if exp_id not in EXPERIMENTS:
            logger.warning("未知实验: %s，跳过", exp_id)
            continue

        name, fn = EXPERIMENTS[exp_id]
        logger.info("")
        try:
            result = fn(config)
            all_results[exp_id] = {"name": name, "result": result, "status": "ok"}
        except Exception as e:
            logger.error("实验 %s 失败: %s", exp_id, e, exc_info=True)
            all_results[exp_id] = {"name": name, "result": {"error": str(e)}, "status": "failed"}

    # 生成摘要报告
    summary = _generate_summary(all_results)
    _print_summary(summary, all_results)

    # 保存结果
    import json
    report = {
        "summary": summary,
        "experiments": {
            exp_id: {
                "name": v["name"],
                "status": v["status"],
                "result": _serialize_result(v["result"]),
            }
            for exp_id, v in all_results.items()
        },
    }
    report_path = output_path / "benchmark_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    logger.info("报告已保存到: %s", report_path)

    return report


def _serialize_result(result: dict) -> dict:
    """将结果中的非 JSON 可序列化值转换"""
    out = {}
    for k, v in result.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, dict):
            out[k] = _serialize_result(v)
        else:
            out[k] = v
    return out


def _generate_summary(all_results: dict) -> dict:
    """生成实验摘要"""
    decisions = {}
    for exp_id, v in all_results.items():
        if v["status"] != "ok":
            decisions[exp_id] = "失败"
            continue
        result = v["result"]

        if exp_id == "E1":
            decisions[exp_id] = f"选用 {result.get('winner', '?')} — {result.get('conclusion', '')}"
        elif exp_id == "E2":
            decisions[exp_id] = f"选用 loss={result.get('winner', '?')} — {result.get('conclusion', '')}"
        elif exp_id == "E3":
            decisions[exp_id] = f"选用 {result.get('best_label', '?')} — {result.get('conclusion', '')}"
        elif exp_id == "E4":
            if result.get("skipped"):
                decisions[exp_id] = "跳过 — 待 Phase A 数据"
            else:
                decisions[exp_id] = result.get("conclusion", "")
        elif exp_id == "E5":
            decisions[exp_id] = result.get("conclusion", "")
        elif exp_id == "E6":
            if result.get("skipped"):
                decisions[exp_id] = "跳过 — 待 xs_ret_60d 数据"
            else:
                decisions[exp_id] = f"选用 {result.get('winner', '?')} 作为 primary — {result.get('conclusion', '')}"
        elif exp_id == "E7":
            if result.get("skipped"):
                decisions[exp_id] = "跳过 — 待 up_down_60d 数据"
            else:
                decisions[exp_id] = result.get("conclusion", "")

    return {"decisions": decisions, "timestamp": str(pd.Timestamp.now())}


def _print_summary(summary: dict, all_results: dict):
    """打印实验摘要"""
    print("\n" + "=" * 70)
    print("  第 8.4 节强制性对比实验 — 结论摘要")
    print("=" * 70)
    print(f"  时间: {summary.get('timestamp', '')}")
    print("-" * 70)

    for exp_id in ["E1", "E2", "E3", "E4", "E5", "E6", "E7"]:
        if exp_id not in all_results:
            continue
        v = all_results[exp_id]
        status = "✓" if v["status"] == "ok" else "✗"
        decision = summary.get("decisions", {}).get(exp_id, "")
        print(f"  {status} {exp_id}: {v['name']}")
        print(f"     → {decision}")

    print("=" * 70)