# -*- coding: utf-8 -*-
"""
Qlib Backtest Results Analysis - backtest.py

Generate charts and reports from backtest results using Plotly (avoids matplotlib/numpy2 compat issues).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_series(df, key):
    """Extract a numeric series from DataFrame, handling MultiIndex."""
    if df is None or df.empty:
        return None
    cols = df.columns
    target_col = None
    if isinstance(cols, pd.MultiIndex):
        matches = [c for c in cols if c[1] == key]
        if matches:
            target_col = matches[0]
    else:
        if key in cols:
            target_col = key
    if target_col is None:
        return None
    s = df[target_col]
    if hasattr(s, "ndim") and s.ndim > 1:
        s = s.iloc[:, 0] if s.shape[1] == 1 else s.mean(axis=1)
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s


def _to_list(s):
    """Safely convert series to Python list for Plotly."""
    if s is None:
        return []
    return [float(v) for v in s.values]


def _save_plotly_fig(fig, output_path: Path):
    """Save Plotly figure to PNG."""
    try:
        fig.write_image(str(output_path), width=1280, height=720, scale=2)
        return True
    except Exception as e:
        logger.warning("Plotly 保存失败: %s", e)
        return False


def generate_report_charts(pred_df, report_normal_df, analysis_df,
                          positions_df=None, output_dir: str = "output/qlib_charts"):
    """
    Generate all analysis charts from backtest results using Plotly.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.error("Plotly 未安装，无法生成图表。请运行: pip install plotly kaleido")
        return []

    charts = []
    ret = _extract_series(report_normal_df, "return")
    bench = _extract_series(report_normal_df, "bench")

    # Chart 1: Cumulative return
    try:
        if ret is not None and len(ret) > 0:
            fig = go.Figure()
            cum_ret = (1 + ret).cumprod()
            fig.add_trace(go.Scatter(
                x=list(cum_ret.index.astype(str)),
                y=_to_list(cum_ret),
                mode='lines',
                name='Strategy',
                line=dict(color='steelblue', width=2)
            ))
            if bench is not None and len(bench) > 0:
                cum_bench = (1 + bench).cumprod()
                fig.add_trace(go.Scatter(
                    x=list(cum_bench.index.astype(str)),
                    y=_to_list(cum_bench),
                    mode='lines',
                    name='Benchmark',
                    line=dict(color='orange', width=2, dash='dash')
                ))
            fig.update_layout(
                title='Cumulative Return',
                xaxis_title='Date',
                yaxis_title='Cumulative Return',
                template='plotly_dark',
                width=1280,
                height=720,
                paper_bgcolor='#0f172a',
                plot_bgcolor='#1e293b',
                font=dict(color='#f1f5f9'),
            )
            out = output_path / "qlib_01_cumulative_return.png"
            if _save_plotly_fig(fig, out):
                charts.append("qlib_01_cumulative_return.png")
                logger.info("图表生成: qlib_01_cumulative_return.png")
    except Exception as e:
        logger.error("累计收益图失败: %s", e)

    # Chart 2: Prediction distribution
    try:
        if pred_df is not None and not pred_df.empty:
            pred_values = _extract_series(pred_df, "score")
            if pred_values is None:
                for c in pred_df.columns:
                    key = c if isinstance(c, str) else c[-1] if isinstance(c, tuple) else str(c)
                    candidate = _extract_series(pred_df, key)
                    if candidate is not None and len(candidate) > 0:
                        pred_values = candidate
                        break
            if pred_values is not None and len(pred_values) > 0:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=_to_list(pred_values),
                    nbinsx=min(100, max(20, len(pred_values)//10)),
                    marker_color='steelblue',
                    opacity=0.8
                ))
                fig.update_layout(
                    title='Prediction Distribution',
                    xaxis_title='Score',
                    yaxis_title='Count',
                    template='plotly_dark',
                    width=1280,
                    height=720,
                    paper_bgcolor='#0f172a',
                    plot_bgcolor='#1e293b',
                    font=dict(color='#f1f5f9'),
                )
                out = output_path / "qlib_02_pred_distribution.png"
                if _save_plotly_fig(fig, out):
                    charts.append("qlib_02_pred_distribution.png")
                    logger.info("图表生成: qlib_02_pred_distribution.png")
    except Exception as e:
        logger.error("预测分布图失败: %s", e)

    # Chart 3: Monthly returns heatmap
    try:
        if ret is not None and len(ret) > 0:
            monthly = ret.resample('ME').apply(lambda x: (1+x).prod()-1)
            if len(monthly) > 0:
                monthly_df = monthly.to_frame(name="return")
                monthly_df["year"] = monthly_df.index.year
                monthly_df["month"] = monthly_df.index.month
                pivot = monthly_df.pivot(index="year", columns="month", values="return")
                z = pivot.values.tolist()
                x = [str(m) for m in pivot.columns]
                y = [str(y) for y in pivot.index]
                # Format annotations
                annotations = []
                for i, yy in enumerate(y):
                    for j, xx in enumerate(x):
                        val = pivot.iloc[i, j]
                        if pd.notna(val):
                            annotations.append(dict(x=xx, y=yy, text=f"{val:.2%}", showarrow=False,
                                                    font=dict(color="black" if abs(val) < 0.05 else "white", size=10)))
                fig = go.Figure(data=go.Heatmap(
                    z=z,
                    x=x,
                    y=y,
                    colorscale='RdYlGn',
                    zmid=0,
                    colorbar=dict(title='Return'),
                ))
                fig.update_layout(
                    title='Monthly Returns Heatmap',
                    template='plotly_dark',
                    width=1280,
                    height=600,
                    paper_bgcolor='#0f172a',
                    plot_bgcolor='#1e293b',
                    font=dict(color='#f1f5f9'),
                    annotations=annotations,
                )
                out = output_path / "qlib_03_monthly_heatmap.png"
                if _save_plotly_fig(fig, out):
                    charts.append("qlib_03_monthly_heatmap.png")
                    logger.info("图表生成: qlib_03_monthly_heatmap.png")
    except Exception as e:
        logger.error("月度热力图失败: %s", e)

    # Chart 4: Drawdown
    try:
        if ret is not None and len(ret) > 0:
            cum = (1 + ret).cumprod()
            running_max = cum.cummax()
            drawdown = (cum - running_max) / running_max
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(drawdown.index.astype(str)),
                y=_to_list(drawdown),
                mode='lines',
                fill='tozeroy',
                name='Drawdown',
                line=dict(color='crimson', width=1)
            ))
            fig.update_layout(
                title='Drawdown',
                xaxis_title='Date',
                yaxis_title='Drawdown',
                template='plotly_dark',
                width=1280,
                height=720,
                paper_bgcolor='#0f172a',
                plot_bgcolor='#1e293b',
                font=dict(color='#f1f5f9'),
            )
            out = output_path / "qlib_04_drawdown.png"
            if _save_plotly_fig(fig, out):
                charts.append("qlib_04_drawdown.png")
                logger.info("图表生成: qlib_04_drawdown.png")
    except Exception as e:
        logger.error("回撤图失败: %s", e)

    # Chart 5: Rolling Sharpe (90-day)
    try:
        if ret is not None and len(ret) > 60:
            rolling_sharpe = (ret.rolling(60).mean() / ret.rolling(60).std()) * np.sqrt(252)
            rolling_sharpe = rolling_sharpe.dropna()
            if len(rolling_sharpe) > 0:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(rolling_sharpe.index.astype(str)),
                    y=_to_list(rolling_sharpe),
                    mode='lines',
                    name='Rolling Sharpe (60d)',
                    line=dict(color='limegreen', width=2)
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.update_layout(
                    title='Rolling Sharpe Ratio (60-day)',
                    xaxis_title='Date',
                    yaxis_title='Sharpe',
                    template='plotly_dark',
                    width=1280,
                    height=720,
                    paper_bgcolor='#0f172a',
                    plot_bgcolor='#1e293b',
                    font=dict(color='#f1f5f9'),
                )
                out = output_path / "qlib_05_rolling_sharpe.png"
                if _save_plotly_fig(fig, out):
                    charts.append("qlib_05_rolling_sharpe.png")
                    logger.info("图表生成: qlib_05_rolling_sharpe.png")
    except Exception as e:
        logger.error("滚动夏普图失败: %s", e)

    logger.info("图表生成完成: %d 张", len(charts))
    return charts


def print_summary(report_normal_df, analysis_df):
    """Print backtest summary metrics."""
    try:
        print("\n" + "=" * 60)
        print("回测结果摘要")
        print("=" * 60)

        if report_normal_df is not None and not report_normal_df.empty:
            print("\n[每日收益统计]")
            for key in ["return", "bench", "turnover"]:
                s = _extract_series(report_normal_df, key)
                if s is not None and len(s) > 0:
                    print(f"  {key}: 日均={s.mean():.6f}, 年化={s.mean()*252:.4f}, 夏普={s.mean()/s.std()*np.sqrt(252):.4f}")

        if analysis_df is not None and not analysis_df.empty:
            print("\n[绩效指标]")
            cols = analysis_df.columns
            if isinstance(cols, pd.MultiIndex):
                for col in cols:
                    val = analysis_df[col].iloc[-1] if len(analysis_df) > 0 else None
                    if val is not None and pd.notna(val):
                        col_name = "_".join(str(c) for c in col if c)
                        print(f"  {col_name}: {val:.4f}")
            else:
                for col in cols:
                    val = analysis_df[col].iloc[-1] if len(analysis_df) > 0 else None
                    if val is not None and pd.notna(val):
                        print(f"  {col}: {val:.4f}")
    except Exception as e:
        logger.error("打印摘要失败: %s", e)
