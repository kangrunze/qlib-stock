# -*- coding: utf-8 -*-
"""
市场阶段稳定性分析 — regime.py

Phase 3 鲁棒性深化。
- 牛/熊/震荡市分类（基于 60 日均线 + 波动率）
- 分段评估（每阶段 IC + 回测指标）
- 关键年份独立回测（2015/2018/2020/2022/2024）

Usage:
    from qlib_pipeline.regime import classify_market_regime, regime_analysis
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from qlib_pipeline.numpy_compat import *  # noqa

logger = logging.getLogger(__name__)


def classify_market_regime(price_series: pd.Series,
                           ma_window: int = 60,
                           vol_window: int = 60) -> pd.DataFrame:
    """
    Classify market regime based on 60-day MA direction and volatility level.

    Regime rules:
      - Bull:  price > MA60  AND vol < median(vol)
      - Strong Bull: price > MA60 AND vol >= median(vol)
      - Bear:  price < MA60  AND vol < median(vol)
      - Strong Bear: price < MA60 AND vol >= median(vol)

    Args:
        price_series: daily close prices
        ma_window: moving average window
        vol_window: volatility window

    Returns:
        DataFrame with columns: date, price, ma, vol, regime
    """
    ma = price_series.rolling(ma_window).mean()
    returns = price_series.pct_change()
    vol = returns.rolling(vol_window).std()

    vol_median = vol.median()

    df = pd.DataFrame({
        "date": price_series.index,
        "price": price_series.values,
        "ma": ma.values,
        "vol": vol.values,
    })

    df["regime"] = "unknown"
    above_ma = df["price"] > df["ma"]
    low_vol = df["vol"] < vol_median

    df.loc[above_ma & low_vol, "regime"] = "bull"
    df.loc[above_ma & ~low_vol, "regime"] = "strong_bull"
    df.loc[~above_ma & low_vol, "regime"] = "bear"
    df.loc[~above_ma & ~low_vol, "regime"] = "strong_bear"

    df = df.dropna()
    return df


def regime_analysis(ic_series: pd.Series,
                    regime_df: pd.DataFrame,
                    output_dir: str = "output/regime") -> pd.DataFrame:
    """
    Analyze IC performance across market regimes.

    Args:
        ic_series: daily IC values indexed by date
        regime_df: regime classification DataFrame (from classify_market_regime)
        output_dir: output directory

    Returns:
        DataFrame with IC metrics per regime
    """
    # Align dates
    common_dates = ic_series.index.intersection(regime_df["date"].values)
    ic = ic_series.loc[common_dates]
    reg = regime_df.set_index("date").loc[common_dates, "regime"]

    results = []
    for r in ["bull", "strong_bull", "bear", "strong_bear"]:
        mask = reg == r
        if mask.sum() > 5:
            ic_mean = ic[mask].mean()
            ic_std = ic[mask].std()
            icir = ic_mean / ic_std if ic_std > 0 else np.nan
            results.append({
                "regime": r,
                "n_days": int(mask.sum()),
                "ic_mean": float(ic_mean),
                "ic_std": float(ic_std),
                "icir": float(icir),
                "ic_positive_ratio": float((ic[mask] > 0).mean()),
            })

    df = pd.DataFrame(results)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path / "regime_analysis.csv", index=False)

    # Generate chart
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Market Regime Classification", "IC by Regime"),
            vertical_spacing=0.15,
        )

        # Regime classification
        colors = {"bull": "green", "strong_bull": "limegreen",
                  "bear": "red", "strong_bear": "darkred"}
        for r in ["bull", "strong_bull", "bear", "strong_bear"]:
            mask = reg == r
            if mask.sum() > 0:
                years = ic[mask].index
                fig.add_trace(go.Scatter(
                    x=list(years),
                    y=[1.01 if r.startswith("bull") else 0.99] * len(years),
                    mode='markers',
                    name=r,
                    marker=dict(color=colors.get(r, "gray"), size=4),
                    opacity=0.5,
                ), row=1, col=1)

        # IC by regime bar chart
        for _, row in df.iterrows():
            fig.add_trace(go.Bar(
                x=[row["regime"]],
                y=[row["icir"]],
                name=row["regime"],
                marker_color=colors.get(row["regime"], "gray"),
            ), row=2, col=1)

        fig.update_layout(
            title="Market Regime Stability Analysis",
            template='plotly_dark',
            height=1000,
            width=1280,
            showlegend=True,
        )
        fig.write_image(str(output_path / "regime_analysis.png"), width=1280, height=1000, scale=2)
        logger.info("市场阶段分析图表已保存: regime_analysis.png")
    except Exception as e:
        logger.warning("图表生成失败: %s", e)

    logger.info("市场阶段分析完成: %d regimes", len(df))
    if not df.empty:
        for _, row in df.iterrows():
            logger.info("  %s: ICIR=%.4f, IC=%.4f, n=%d", row["regime"], row["icir"], row["ic_mean"], row["n_days"])

    return df


KEY_YEARS = {
    "2015": ("2015-01-01", "2015-12-31"),
    "2018": ("2018-01-01", "2018-12-31"),
    "2020": ("2020-01-01", "2020-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
}


def key_year_backtest(config: dict, years: Optional[List[str]] = None,
                      output_dir: str = "output/key_years",
                      pretrained_rid: Optional[str] = None) -> pd.DataFrame:
    """
    Run independent backtest for key years.

    支持两种模式：
      1. 样本外独立回测（推荐）：传入 pretrained_rid，用预训练模型在指定年份做纯测试
      2. 独立年份训练+测试（默认）：每年独立训练（用于快速诊断）

    模式 1 满足第 10.2 节要求：在一个未在训练中使用的市场阶段上进行独立回测。

    Args:
        config: workflow config dict
        years: list of years to backtest (default: KEY_YEARS)
        output_dir: output directory
        pretrained_rid: 预训练模型 recorder_id（可选，传入则使用样本外模式）

    Returns:
        DataFrame with per-year metrics
    """
    import os
    os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    import qlib
    from qlib.constant import REG_CN
    from qlib.utils import init_instance_by_config
    from qlib.workflow import R
    from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
    from qlib.utils import flatten_dict

    provider_uri = config.get("qlib", {}).get("provider_uri",
                     "D:/trae/qlib_bin")
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    from qlib.config import C
    C.joblib_backend = "threading"
    C.dataset_process_n_worker = 1

    if years is None:
        years = list(KEY_YEARS.keys())

    handler = config.get("dataset", {}).get("handler", "Alpha158")
    model_cfg = config.get("qlib_lgb", {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {"loss": "mse", "num_threads": 20},
    })

    # 样本外模式：加载预训练模型
    pretrained_model = None
    if pretrained_rid:
        logger.info("样本外独立回测模式: 使用预训练模型 %s", pretrained_rid)
        try:
            pretrained_model = R.get_recorder(recorder_id=pretrained_rid).load_object("params.pkl")
            logger.info("  → 预训练模型加载成功")
        except Exception as e:
            logger.error("  → 预训练模型加载失败: %s，回退到独立训练模式", e)
            pretrained_rid = None

    results = []
    for year in years:
        start, end = KEY_YEARS.get(year, (f"{year}-01-01", f"{year}-12-31"))

        if pretrained_model is not None:
            logger.info("关键年份样本外回测: %s (%s ~ %s)", year, start, end)
        else:
            logger.info("关键年份独立回测: %s (%s ~ %s)", year, start, end)

        try:
            handler_cfg = {
                "class": handler,
                "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "start_time": start,
                    "end_time": end,
                    "fit_start_time": start,
                    "fit_end_time": end,
                    "instruments": config.get("data_handler", {}).get("instruments", "csi300"),
                },
            }

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
                        "segments": {
                            "train": [start, start],  # 仅用于初始化，实际训练用预训练模型
                            "test": [start, end],
                        },
                    },
                },
            }

            dataset = init_instance_by_config(task["dataset"])

            with R.start(experiment_name=f"key_year_{year}"):
                if pretrained_model is not None:
                    # 样本外模式：跳过训练，直接使用预训练模型
                    model = pretrained_model
                    logger.info("  → 跳过训练（使用预训练模型），直接预测")
                else:
                    model = init_instance_by_config(task["model"])
                    model.fit(dataset)

                sr = SignalRecord(model, dataset, R.get_recorder())
                sr.generate()

                # 从配置中读取 benchmark，优先使用 backtest 配置
                bt_config = config.get("backtest", {}).get("backtest", {})
                benchmark = bt_config.get("benchmark", config.get("regime", {}).get("benchmark_code", "SH000300"))
                port_config = {
                    "strategy": {
                        "class": "TopkDropoutStrategy",
                        "module_path": "qlib.contrib.strategy.signal_strategy",
                        "kwargs": {"topk": 50, "n_drop": 5, "model": model, "dataset": dataset},
                    },
                    "backtest": {
                        "start_time": start,
                        "end_time": end,
                        "account": 100000000,
                        "benchmark": benchmark,
                        "exchange_kwargs": {"open_cost": 0.0005, "close_cost": 0.0015, "min_cost": 5},
                    },
                }

                par = PortAnaRecord(R.get_recorder(), port_config, "day")
                par.generate()
                rid = R.get_recorder().id

                logger.info("  → 回测完成, recorder_id=%s", rid)

                # 记录到 ExperimentTracker
                try:
                    from research.experiment_tracker import ExperimentTracker
                    tracker = ExperimentTracker()
                    tracker.log_experiment(
                        config=config,
                        metrics={
                            "recorder_id": rid,
                            "pretrained_rid": pretrained_rid,
                            "year": year,
                            "mode": "out_of_sample" if pretrained_model is not None else "in_sample",
                        },
                        phase="confirmation" if pretrained_model is not None else "exploration",
                        description=f"key_year_{year}_{'oos' if pretrained_model is not None else 'is'}",
                        recorder_id=rid,
                        test_data_range=(start, end),
                    )
                except Exception:
                    pass

                results.append({
                    "year": year,
                    "start": start,
                    "end": end,
                    "mode": "out_of_sample" if pretrained_model is not None else "in_sample",
                    "pretrained_rid": pretrained_rid,
                    "recorder_id": rid,
                })

        except Exception as e:
            logger.error("关键年份 %s 回测失败: %s", year, e)
            results.append({
                "year": year,
                "start": start,
                "end": end,
                "mode": "out_of_sample" if pretrained_model is not None else "in_sample",
                "error": str(e),
            })

    df = pd.DataFrame(results)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path / "key_years_backtest.csv", index=False)
    logger.info("关键年份回测结果已保存到: %s", output_path)
    return df