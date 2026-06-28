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


def print_stock_picks(pred_df, top_k: int = 30, date: str = None,
                      output_dir: str = "output/picks"):
    """
    从 Qlib 预测结果中提取 Top-K 选股推荐并打印和保存。
    
    pred_df 结构（Qlib SignalRecord 输出）:
        MultiIndex: (datetime, instrument)
        Column: score
    
    Args:
        pred_df: 预测 DataFrame（来自 recorder.load_object("pred.pkl")）
        top_k: 选股数量
        date: 指定日期（None=取最新交易日）
        output_dir: CSV 输出目录
    
    Returns:
        DataFrame[stock_code, score, rank, date]
    """
    if pred_df is None or pred_df.empty:
        logger.warning("无预测数据，无法生成选股推荐")
        return None

    try:
        # 确定目标日期
        dates = pred_df.index.get_level_values(0).unique()
        if date:
            target_date = pd.Timestamp(date)
            if target_date not in dates:
                logger.warning("日期 %s 不在预测范围内，使用最新日期", date)
                target_date = dates[-1]
        else:
            target_date = dates[-1]

        # 提取当日预测
        daily_pred = pred_df.loc[target_date]
        if isinstance(daily_pred, pd.Series):
            daily_pred = daily_pred.to_frame("score")

        # 转换为 stock_code + score
        picks = daily_pred.reset_index()
        # instrument 列名可能是 'instrument' 或直接是 level 名
        inst_col = picks.columns[0]
        picks = picks.rename(columns={inst_col: "stock_code"})

        # 确保有 score 列
        if "score" not in picks.columns:
            picks["score"] = picks.iloc[:, 1] if picks.shape[1] > 1 else 0

        # 降序排序，取 Top-K
        picks = picks.dropna(subset=["score"])
        picks = picks.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
        picks["rank"] = range(1, len(picks) + 1)
        picks["date"] = target_date.strftime("%Y-%m-%d")

        # 清理股票代码格式（去掉 SH/SZ 前缀，统一为 6 位数字）
        picks["stock_code"] = picks["stock_code"].astype(str).str.replace(r'^SH|^SZ', '', regex=True)
        picks["stock_code"] = picks["stock_code"].str.zfill(6)

        # 打印选股推荐
        print("\n" + "=" * 70)
        print(f"  Top-{top_k} 选股推荐  [{target_date.strftime('%Y-%m-%d')}]")
        print("=" * 70)
        print(f"{'排名':<6}{'股票代码':<12}{'预测得分':>10}{'信号强度':>12}")
        print("-" * 70)

        for _, row in picks.iterrows():
            score = row["score"]
            # 信号强度分类
            if score > 0.03:
                strength = "★★★ 强"
            elif score > 0.01:
                strength = "★★☆ 中"
            elif score > 0:
                strength = "★☆☆ 弱"
            else:
                strength = "--- 负"
            print(f"  #{int(row['rank']):<4d} {row['stock_code']:<10} {score:>10.6f} {strength:>12}")

        print("-" * 70)
        n_strong = (picks["score"] > 0.03).sum()
        n_medium = ((picks["score"] > 0.01) & (picks["score"] <= 0.03)).sum()
        n_weak = ((picks["score"] > 0) & (picks["score"] <= 0.01)).sum()
        print(f"  强信号(>0.03): {n_strong}只 | 中信号(0.01~0.03): {n_medium}只 | 弱信号(0~0.01): {n_weak}只")
        print(f"  平均得分: {picks['score'].mean():.6f} | 最高: {picks['score'].max():.6f} | 最低: {picks['score'].min():.6f}")
        print("=" * 70)

        # 保存 CSV
        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            csv_file = out_path / f"stock_picks_{target_date.strftime('%Y%m%d')}.csv"
            picks.to_csv(csv_file, index=False, encoding="utf-8-sig")
            logger.info("选股推荐已保存: %s", csv_file)

        return picks

    except Exception as e:
        logger.error("生成选股推荐失败: %s", e)
        return None


def save_stock_picks_to_file(picks_df, output_dir: str = "output/picks", date_str: str = None):
    """保存选股结果到 CSV 文件。"""
    if picks_df is None or picks_df.empty:
        return None

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if date_str is None and "date" in picks_df.columns:
        date_str = picks_df["date"].iloc[0]
    if date_str is None:
        from datetime import datetime
        date_str = datetime.now().strftime("%Y%m%d")
    else:
        date_str = pd.Timestamp(date_str).strftime("%Y%m%d")

    csv_file = out_path / f"stock_picks_{date_str}.csv"
    picks_df.to_csv(csv_file, index=False, encoding="utf-8-sig")
    logger.info("选股推荐已保存: %s", csv_file)
    return str(csv_file)
