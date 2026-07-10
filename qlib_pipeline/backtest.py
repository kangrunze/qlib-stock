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
                          positions_df=None, output_dir: str = "output/qlib_charts",
                          model_tag: str = "", run_date: str = ""):
    """
    Generate all analysis charts from backtest results using Plotly.
    
    Args:
        model_tag: 模型标识 (如 Alpha158_LGBModel_mse_20240630)，用于文件名
        run_date: 运行日期 (如 20260707_1530)，用于日期层级目录
    """
    if run_date:
        date_dir = run_date  # YYYYMMDD_HHMM，每次运行独立目录
    else:
        from datetime import datetime
        date_dir = datetime.now().strftime("%Y%m%d_%H%M")
    
    output_path = Path(output_dir) / date_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    prefix = f"{model_tag}_" if model_tag else ""

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
                name='策略收益',
                line=dict(color='steelblue', width=2)
            ))
            if bench is not None and len(bench) > 0:
                cum_bench = (1 + bench).cumprod()
                fig.add_trace(go.Scatter(
                    x=list(cum_bench.index.astype(str)),
                    y=_to_list(cum_bench),
                    mode='lines',
                    name='基准收益',
                    line=dict(color='orange', width=2, dash='dash')
                ))
            fig.update_layout(
                title='累计收益曲线',
                xaxis_title='日期',
                yaxis_title='累计净值',
                template='plotly_dark',
                width=1280,
                height=720,
                paper_bgcolor='#0f172a',
                plot_bgcolor='#1e293b',
                font=dict(color='#f1f5f9'),
            )
            out = output_path / f"{prefix}01_cumulative_return.png"
            if _save_plotly_fig(fig, out):
                charts.append(f"{prefix}01_cumulative_return.png")
                logger.info("图表生成: %s01_cumulative_return.png", prefix)
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
                    title='预测得分分布',
                    xaxis_title='预测得分',
                    yaxis_title='频次',
                    template='plotly_dark',
                    width=1280,
                    height=720,
                    paper_bgcolor='#0f172a',
                    plot_bgcolor='#1e293b',
                    font=dict(color='#f1f5f9'),
                )
                out = output_path / f"{prefix}02_pred_distribution.png"
                if _save_plotly_fig(fig, out):
                    charts.append(f"{prefix}02_pred_distribution.png")
                    logger.info("图表生成: %s02_pred_distribution.png", prefix)
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
                    colorbar=dict(title=dict(text='收益率', side='right')),
                ))
                fig.update_layout(
                    title='月度收益热力图',
                    xaxis_title='月份',
                    yaxis_title='年份',
                    template='plotly_dark',
                    width=1280,
                    height=600,
                    paper_bgcolor='#0f172a',
                    plot_bgcolor='#1e293b',
                    font=dict(color='#f1f5f9'),
                    annotations=annotations,
                )
                out = output_path / f"{prefix}03_monthly_heatmap.png"
                if _save_plotly_fig(fig, out):
                    charts.append(f"{prefix}03_monthly_heatmap.png")
                    logger.info("图表生成: %s03_monthly_heatmap.png", prefix)
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
                name='回撤',
                line=dict(color='crimson', width=1)
            ))
            fig.update_layout(
                title='最大回撤曲线',
                xaxis_title='日期',
                yaxis_title='回撤幅度',
                template='plotly_dark',
                width=1280,
                height=720,
                paper_bgcolor='#0f172a',
                plot_bgcolor='#1e293b',
                font=dict(color='#f1f5f9'),
            )
            out = output_path / f"{prefix}04_drawdown.png"
            if _save_plotly_fig(fig, out):
                charts.append(f"{prefix}04_drawdown.png")
                logger.info("图表生成: %s04_drawdown.png", prefix)
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
                    name='滚动夏普比率(60日)',
                    line=dict(color='limegreen', width=2)
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.update_layout(
                    title='滚动夏普比率（60日窗口）',
                    xaxis_title='日期',
                    yaxis_title='夏普比率',
                    template='plotly_dark',
                    width=1280,
                    height=720,
                    paper_bgcolor='#0f172a',
                    plot_bgcolor='#1e293b',
                    font=dict(color='#f1f5f9'),
                )
                out = output_path / f"{prefix}05_rolling_sharpe.png"
                if _save_plotly_fig(fig, out):
                    charts.append(f"{prefix}05_rolling_sharpe.png")
                    logger.info("图表生成: %s05_rolling_sharpe.png", prefix)
    except Exception as e:
        logger.error("滚动夏普图失败: %s", e)

    logger.info("图表生成完成: %d 张", len(charts))
    return charts


def print_summary(report_normal_df, analysis_df, config: dict = None, picks_df = None):
    """Print backtest summary metrics with cost drag decomposition and style exposure diagnosis.

    第 10.1 节要求:
      - 换手率成本拖累: 年化换手率 × 单次交易成本 = 可验证的成本拖累
      - 风格暴露检查: 避免策略在不自知中成为单一风格因子的杠杆化押注

    Args:
        report_normal_df: 日度报告 DataFrame
        analysis_df: 分析 DataFrame
        config: workflow_config dict（用于读取交易成本参数）
        picks_df: 选股推荐 DataFrame（用于风格暴露诊断）
    """
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

            # ── 换手率成本拖累拆解（第 10.1 节）──
            turnover_s = _extract_series(report_normal_df, "turnover")
            ret_s = _extract_series(report_normal_df, "return")
            if turnover_s is not None and len(turnover_s) > 0:
                # 读取交易成本参数
                if config:
                    backtest_cfg = config.get("backtest", {}).get("backtest", {})
                    exc_cfg = backtest_cfg.get("exchange_kwargs", {})
                    open_cost = exc_cfg.get("open_cost", 0.0005)
                    close_cost = exc_cfg.get("close_cost", 0.0015)
                else:
                    open_cost, close_cost = 0.0005, 0.0015

                round_trip_cost = open_cost + close_cost
                avg_daily_turnover = float(turnover_s.mean())
                annual_turnover = avg_daily_turnover * 252
                annual_cost_drag = annual_turnover * round_trip_cost

                print("\n[换手率成本拖累拆解]")
                print(f"  日均换手率:           {avg_daily_turnover:.4%}")
                print(f"  年化换手率:           {annual_turnover:.2f}x")
                print(f"  单次往返成本:         {round_trip_cost:.4%} (open={open_cost:.4%} + close={close_cost:.4%})")
                print(f"  年化成本拖累:         {annual_cost_drag:.4%}")

                if ret_s is not None and len(ret_s) > 0:
                    annual_return = float(ret_s.mean()) * 252
                    bench_s = _extract_series(report_normal_df, "bench")
                    annual_bench = float(bench_s.mean()) * 252 if bench_s is not None else 0
                    excess_pre_cost = annual_return - annual_bench
                    excess_post_cost = excess_pre_cost - annual_cost_drag
                    print(f"  年化收益(策略):       {annual_return:.4%}")
                    print(f"  年化收益(基准):       {annual_bench:.4%}")
                    print(f"  超额收益(成本前):     {excess_pre_cost:.4%}")
                    print(f"  超额收益(成本后):     {excess_post_cost:.4%}")
                    if excess_post_cost < 0:
                        print(f"  ⚠ 成本后超额收益为负！换手率过高可能侵蚀全部 Alpha")

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

        # ── 风格暴露诊断（第 9.2 节）──
        if picks_df is not None and not picks_df.empty:
            print("\n[风格暴露诊断]")
            try:
                from research.risk_model import STYLE_FACTORS, calculate_style_exposures, check_style_constraints
                print(f"  可用风格维度: {', '.join(STYLE_FACTORS.keys())}")

                # 尝试获取因子值并进行实际计算
                exposure_calculated = False
                try:
                    from qlib.data import D
                    # 获取选股日期
                    pick_date = picks_df.index.get_level_values("datetime")[-1] if hasattr(picks_df.index, "get_level_values") else picks_df.index[-1][0]
                    pick_stocks = list(picks_df.loc[pick_date].index)[:30] if hasattr(picks_df, "loc") else list(picks_df.index.get_level_values("instrument"))[:30]
                    needed_features = [v["feature"] for v in STYLE_FACTORS.values()]

                    # 尝试加载因子值
                    factor_data = {}
                    for feat in needed_features:
                        try:
                            fv = D.features(pick_stocks, [feat], start_time=pick_date, end_time=pick_date)
                            if fv is not None and not fv.empty:
                                factor_data[feat] = fv.iloc[:, 0] if fv.shape[1] > 0 else None
                        except Exception:
                            pass

                    if len(factor_data) >= 3:  # 至少有 3 个风格因子可用
                        # 构建因子值 DataFrame
                        import pandas as pd
                        factor_df = pd.DataFrame(factor_data)
                        # 构建等权组合权重
                        weights = pd.Series(1.0 / len(pick_stocks), index=pick_stocks)
                        exposures = calculate_style_exposures(weights, factor_df)
                        if not exposures.empty:
                            print(f"  ✓ 风格暴露计算完成（%d 个维度）" % len(exposures))
                            exposures_checked, all_ok = check_style_constraints(exposures)
                            for _, row in exposures_checked.iterrows():
                                flag = " ⚠" if not row["constraint_ok"] else ""
                                print(f"    {row['name_cn']:6s}: active={row['active_exposure']:+.3f}σ (port={row['portfolio_exposure']:+.3f}σ, bench={row['benchmark_exposure']:+.3f}σ){flag}")

                            if not all_ok:
                                n_viol = (~exposures_checked["constraint_ok"]).sum()
                                print(f"  ⚠ 风格暴露超标: {n_viol} 个因子超出 |active| > 0.5σ")
                            else:
                                print(f"  ✓ 所有风格暴露在阈值内 (|active| ≤ 0.5σ)")
                            exposure_calculated = True
                except Exception as e:
                    logger.debug("风格暴露自动计算失败: %s", e)

                if not exposure_calculated:
                    print(f"  (风格因子数据不可用，需 Phase A 接入 $roe/$pb_inv/$mom_12m 等基本面字段)")
                    print(f"  → 接入后自动计算: research.risk_model.calculate_style_exposures()")
                    print(f"  → 检查阈值: |active_exposure| > 0.5 sigma")
            except ImportError:
                print("  (research.risk_model 不可用，无法进行风格暴露诊断)")

    except Exception as e:
        logger.error("打印摘要失败: %s", e)


def print_stock_picks(pred_df, top_k: int = 30, date: str = None,
                      output_dir: str = "output/picks",
                      recorder_id: str = None, config_snapshot: dict = None,
                      model_tag: str = "", run_date: str = ""):
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
        recorder_id: Qlib Recorder ID（可选，用于追溯模型版本）
        config_snapshot: 配置快照 dict（可选，用于追溯参数）
        model_tag: 模型标识，用于文件名
        run_date: 运行日期，用于日期层级目录

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
        
        # 处理 MultiIndex columns：扁平化为简单字符串
        if isinstance(picks.columns, pd.MultiIndex):
            picks.columns = [
                "_".join(str(c) for c in col if c).strip("_") 
                for col in picks.columns
            ]
        
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

        # 基于当日横截面分位数进行信号强度分级（适配 loss=mse 和 loss=rank 两种模式）
        scores = picks["score"]
        q80 = scores.quantile(0.8)
        q50 = scores.quantile(0.5)

        for _, row in picks.iterrows():
            score = row["score"]
            # 信号强度分类：基于当日分位数
            if score >= q80:
                strength = "★★★ 强"
            elif score >= q50:
                strength = "★★☆ 中"
            elif score > 0:
                strength = "★☆☆ 弱"
            else:
                strength = "--- 负"
            print(f"  #{int(row['rank']):<4d} {row['stock_code']:<10} {score:>10.6f} {strength:>12}")

        print("-" * 70)
        n_strong = (scores >= q80).sum()
        n_medium = ((scores >= q50) & (scores < q80)).sum()
        n_weak = ((scores > 0) & (scores < q50)).sum()
        print(f"  强信号(≥P80): {n_strong}只 | 中信号(P50~P80): {n_medium}只 | 弱信号(0~P50): {n_weak}只")
        print(f"  平均得分: {picks['score'].mean():.6f} | 最高: {picks['score'].max():.6f} | 最低: {picks['score'].min():.6f}")
        print("=" * 70)

        # 保存 CSV + 配置快照
        if output_dir:
            save_stock_picks_to_file(
                picks, output_dir=output_dir, date_str=target_date.strftime("%Y%m%d"),
                recorder_id=recorder_id, config_snapshot=config_snapshot,
                model_tag=model_tag, run_date=run_date,
            )

        return picks

    except Exception as e:
        logger.error("生成选股推荐失败: %s", e)
        return None


def save_stock_picks_to_file(picks_df, output_dir: str = "output/picks", date_str: str = None,
                            recorder_id: str = None, config_snapshot: dict = None,
                            model_tag: str = "", run_date: str = ""):
    """保存选股结果到 CSV 文件，同时保存配置快照用于后续追溯。

    第 11.2 节闭环设计: 推荐时的配置快照（recorder_id、配置文件版本）一并归档，
    几周后复核某期推荐表现不佳时，能准确定位到"当时用的是哪个模型、哪一组参数"。

    Args:
        picks_df: 选股结果 DataFrame
        output_dir: 输出目录
        date_str: 选股日期字符串
        recorder_id: Qlib Recorder ID（可选，用于追溯模型版本）
        config_snapshot: 配置快照 dict（可选，用于追溯参数）
        model_tag: 模型标识，用于文件名
        run_date: 运行日期，用于日期层级目录

    Returns:
        (csv_path, meta_path) 元组
    """
    if picks_df is None or picks_df.empty:
        return None, None

    import json
    from datetime import datetime as dt

    if run_date:
        date_dir = run_date  # YYYYMMDD_HHMM，每次运行独立目录
    else:
        date_dir = dt.now().strftime("%Y%m%d_%H%M")
    out_path = Path(output_dir) / date_dir
    out_path.mkdir(parents=True, exist_ok=True)

    if date_str is None and "date" in picks_df.columns:
        date_str = picks_df["date"].iloc[0]
    if date_str is None:
        date_str = dt.now().strftime("%Y%m%d")
    else:
        date_str = pd.Timestamp(date_str).strftime("%Y%m%d")

    prefix = f"{model_tag}_" if model_tag else ""
    csv_file = out_path / f"{prefix}stock_picks_{date_str}.csv"
    picks_df.to_csv(csv_file, index=False, encoding="utf-8-sig")
    logger.info("选股推荐已保存: %s", csv_file)

    meta = {
        "date": date_str,
        "saved_at": dt.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_picks": len(picks_df),
        "recorder_id": recorder_id,
        "model_tag": model_tag,
        "config_snapshot": config_snapshot or {},
    }
    meta_file = out_path / f"{prefix}stock_picks_{date_str}.meta.json"
    try:
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
        logger.info("配置快照已保存: %s", meta_file)
    except Exception as e:
        logger.warning("保存配置快照失败: %s", e)

    return str(csv_file), str(meta_file)


def save_trade_records(pred_df, topk: int = 50, n_drop: int = 5,
                       init_cash: float = 100000000, report_normal_df=None,
                       output_dir: str = "output/trades",
                       model_tag: str = "", run_date: str = ""):
    """
    从预测信号中还原每日买卖点记录并保存为 CSV。

    基于 Qlib TopkDropoutStrategy 的逻辑：
      - 每天持有 topk 只预测得分最高的股票，等权分配
      - 每期最多替换 n_drop 只（优先卖出不在新 topk 中的旧持仓）
      - 每只股票仓位 = 当日组合总资产 / topk

    输出两个文件:
      1. {output_dir}/{YYYYMMDD}/{model_tag}_trade_records.csv
      2. {output_dir}/{YYYYMMDD}/{model_tag}_daily_holdings.csv

    Args:
        pred_df: 预测 DataFrame（MultiIndex: datetime, instrument; Column: score）
        topk: 每期持有数量
        n_drop: 每期最大替换数
        init_cash: 初始资金
        report_normal_df: 日度回测报告（含 portfolio value，用于计算真实仓位金额）
        output_dir: 输出根目录
        model_tag: 模型标识，用于文件名
        run_date: 运行日期，用于日期层级目录

    Returns:
        Path to saved trade records CSV, or None on failure
    """
    if pred_df is None or pred_df.empty:
        logger.warning("无预测数据，无法生成买卖记录")
        return None

    try:
        dates = sorted(pred_df.index.get_level_values(0).unique())
        if len(dates) < 1:
            logger.warning("交易日不足，无法生成买卖记录")
            return None

        if run_date:
            date_dir = run_date  # YYYYMMDD_HHMM，每次运行独立目录
        else:
            from datetime import datetime
            date_dir = datetime.now().strftime("%Y%m%d_%H%M")
        out_path = Path(output_dir) / date_dir
        out_path.mkdir(parents=True, exist_ok=True)
        prefix = f"{model_tag}_" if model_tag else ""

        # 获取每日组合总资产（从回测报告中提取）
        portfolio_values = {}
        if report_normal_df is not None and not report_normal_df.empty:
            try:
                ret_col = _extract_series(report_normal_df, "return")
                if ret_col is not None and len(ret_col) > 0:
                    cum_ret = (1 + ret_col).cumprod()
                    # 预计算 dates 集合，避免 O(n²) 列表推导
                    pred_date_set = {pd.Timestamp(dt).strftime("%Y-%m-%d") for dt in dates}
                    for d in cum_ret.index:
                        date_str = pd.Timestamp(d).strftime("%Y-%m-%d")
                        if date_str in pred_date_set:
                            portfolio_values[date_str] = init_cash * float(cum_ret.loc[d])
            except Exception:
                pass
        if not portfolio_values:
            for d in dates:
                portfolio_values[pd.Timestamp(d).strftime("%Y-%m-%d")] = init_cash

        trade_records = []
        holding_records = []
        prev_holdings = set()

        for i, date in enumerate(dates):
            date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
            daily = pred_df.loc[date]
            if isinstance(daily, pd.Series):
                daily = daily.to_frame("score")

            topk_stocks = daily["score"].sort_values(ascending=False).head(topk)
            new_topk_set = set(topk_stocks.index)
            n_hold = min(topk, len(topk_stocks))
            pv = portfolio_values.get(date_str, init_cash)
            per_stock_value = pv / n_hold if n_hold > 0 else 0
            weight = 1.0 / n_hold if n_hold > 0 else 0

            if i == 0:
                for rank, (stock, score) in enumerate(topk_stocks.items(), 1):
                    trade_records.append({
                        "date": date_str,
                        "action": "BUY",
                        "stock_code": str(stock).replace("SH", "").replace("SZ", ""),
                        "amount": round(per_stock_value, 2),
                        "score": round(float(score), 6),
                    })
                    holding_records.append({
                        "date": date_str,
                        "rank": rank,
                        "stock_code": str(stock).replace("SH", "").replace("SZ", ""),
                        "position_value": round(per_stock_value, 2),
                        "weight": round(weight, 4),
                        "score": round(float(score), 6),
                        "is_new": 1,
                    })
                prev_holdings = set(topk_stocks.index)
            else:
                to_sell = prev_holdings - new_topk_set
                to_sell_scores = {s: daily.loc[s, "score"] if s in daily.index else -999 for s in to_sell}
                to_sell_ordered = sorted(to_sell, key=lambda s: to_sell_scores[s])[:n_drop]

                to_buy_candidates = new_topk_set - prev_holdings
                n_replace = min(len(to_sell_ordered), len(to_buy_candidates))
                to_buy = list(to_buy_candidates)[:n_replace]

                for stock in to_sell_ordered:
                    if stock in prev_holdings:
                        trade_records.append({
                            "date": date_str,
                            "action": "SELL",
                            "stock_code": str(stock).replace("SH", "").replace("SZ", ""),
                            "amount": round(per_stock_value, 2),
                            "score": round(float(daily.loc[stock, "score"]) if stock in daily.index else 0, 6),
                        })

                for stock in to_buy:
                    trade_records.append({
                        "date": date_str,
                        "action": "BUY",
                        "stock_code": str(stock).replace("SH", "").replace("SZ", ""),
                        "amount": round(per_stock_value, 2),
                        "score": round(float(daily.loc[stock, "score"]), 6),
                    })

                prev_holdings = (prev_holdings - set(to_sell_ordered)) | set(to_buy)

                for rank, (stock, score) in enumerate(topk_stocks.items(), 1):
                    is_new = 1 if stock in (set(to_buy) & new_topk_set) else 0
                    holding_records.append({
                        "date": date_str,
                        "rank": rank,
                        "stock_code": str(stock).replace("SH", "").replace("SZ", ""),
                        "position_value": round(per_stock_value, 2),
                        "weight": round(weight, 4),
                        "score": round(float(score), 6),
                        "is_new": is_new,
                    })

        if trade_records:
            trade_df = pd.DataFrame(trade_records)
            trade_df = trade_df.sort_values(["date", "action"]).reset_index(drop=True)
            csv_file = out_path / f"{prefix}trade_records.csv"
            trade_df.to_csv(csv_file, index=False, encoding="utf-8-sig")
            logger.info("买卖记录已保存: %s (%d 条)", csv_file, len(trade_df))
        else:
            logger.warning("回测期间无买卖记录")
            csv_file = None

        if holding_records:
            holding_df = pd.DataFrame(holding_records)
            holding_df = holding_df.sort_values(["date", "rank"]).reset_index(drop=True)
            holding_file = out_path / f"{prefix}daily_holdings.csv"
            holding_df.to_csv(holding_file, index=False, encoding="utf-8-sig")
            logger.info("每日持仓快照已保存: %s (%d 条)", holding_file, len(holding_df))

        if trade_records:
            buy_count = sum(1 for r in trade_records if r["action"] == "BUY")
            sell_count = sum(1 for r in trade_records if r["action"] == "SELL")
            total_buy = sum(r["amount"] for r in trade_records if r["action"] == "BUY")
            total_sell = sum(r["amount"] for r in trade_records if r["action"] == "SELL")
        else:
            buy_count = sell_count = 0
            total_buy = total_sell = 0
        holding_total = len(holding_records)

        print(f"\n[买卖交易与持仓摘要]")
        print(f"  回测交易日数: {len(dates)}")
        print(f"  买入交易: {buy_count} 次 (合计 {total_buy:,.0f} 元) | 卖出交易: {sell_count} 次 (合计 {total_sell:,.0f} 元)")
        print(f"  每日持仓快照: {holding_total} 条 (每个交易日 {topk} 只股票)")
        print(f"  已保存到目录: {out_path}")

        return str(csv_file) if csv_file else None

    except Exception as e:
        logger.error("生成买卖记录失败: %s", e)
        return None
