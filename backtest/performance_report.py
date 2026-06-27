"""
业绩报告生成模块 - 计算和展示回测业绩指标

指标包括:
    - 累计收益、年化收益、超额收益
    - 年化波动率、最大回撤、最大回撤持续时间
    - Sharpe比率、Sortino比率、Calmar比率
    - 胜率、盈亏比
    - 换手率
    - 月度收益表

使用方法:
    report = PerformanceReport()
    metrics = report.generate_report(equity_curve, trades, benchmark_equity, start, end)
    report.print_report(metrics)
"""

import logging
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class PerformanceReport:
    """业绩报告生成器。

    计算全面的回测业绩指标，支持Console输出和文件保存。

    Attributes:
        risk_free_rate: 无风险利率（年化），默认2.5%。
        trading_days_per_year: 年交易日数，默认252。
    """

    def __init__(self, risk_free_rate: float = 0.025, trading_days_per_year: int = 252):
        """初始化业绩报告生成器。

        Args:
            risk_free_rate: 无风险利率（年化）。
            trading_days_per_year: 每年交易日数。
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year
        self.rf_daily = risk_free_rate / trading_days_per_year

    def generate_report(
        self,
        equity_curve: Union[pd.Series, pd.DataFrame],
        trades: pd.DataFrame,
        benchmark_equity: Optional[Union[pd.Series, pd.DataFrame]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """生成完整的业绩报告。

        Args:
            equity_curve: 权益曲线，index为日期。
                如果是DataFrame则需要有'equity'或'total_value'列。
            trades: 交易记录DataFrame，包含:
                - date: 交易日期
                - stock_code: 股票代码
                - side: buy/sell
                - price: 成交价
                - shares: 成交数量
                - pnl: 盈亏（卖出时）
            benchmark_equity: 基准权益曲线（可选）。
            start_date: 回测起始日期。
            end_date: 回测结束日期。

        Returns:
            指标字典。
        """
        # 标准化权益曲线
        if isinstance(equity_curve, pd.DataFrame):
            if "equity" in equity_curve.columns:
                equity = equity_curve["equity"]
            elif "total_value" in equity_curve.columns:
                equity = equity_curve["total_value"]
            else:
                equity = equity_curve.iloc[:, 0]
        else:
            equity = equity_curve

        equity = equity.dropna()
        if len(equity) < 2:
            logger.warning("权益曲线数据不足，无法计算指标")
            return self._empty_report()

        # 日收益率
        daily_returns = equity.pct_change().dropna()

        report: Dict[str, Any] = {}

        # ========== 基础收益指标 ==========
        report["start_date"] = str(equity.index[0].date())
        report["end_date"] = str(equity.index[-1].date())
        report["total_days"] = len(equity)
        report["trading_days"] = len(daily_returns)

        # 累计收益
        initial_equity = equity.iloc[0]
        final_equity = equity.iloc[-1]
        report["cumulative_return"] = float(final_equity / initial_equity - 1)
        report["total_return_pct"] = round(report["cumulative_return"] * 100, 2)

        # 年化收益
        years = len(daily_returns) / self.trading_days_per_year
        if years > 0 and final_equity > 0 and initial_equity > 0:
            report["annualized_return"] = float(
                (final_equity / initial_equity) ** (1 / years) - 1
            )
        else:
            report["annualized_return"] = 0.0
        report["annualized_return_pct"] = round(report["annualized_return"] * 100, 2)

        # ========== 风险指标 ==========
        # 年化波动率
        report["annualized_volatility"] = float(
            daily_returns.std() * np.sqrt(self.trading_days_per_year)
        )
        report["annualized_volatility_pct"] = round(report["annualized_volatility"] * 100, 2)

        # 最大回撤
        max_dd, max_dd_start, max_dd_end, max_dd_duration = self._compute_max_drawdown(equity)
        report["max_drawdown"] = float(max_dd)
        report["max_drawdown_pct"] = round(max_dd * 100, 2)
        report["max_drawdown_start"] = str(max_dd_start) if max_dd_start else None
        report["max_drawdown_end"] = str(max_dd_end) if max_dd_end else None
        report["max_drawdown_duration_days"] = max_dd_duration

        # ========== 风险调整收益 ==========
        # Sharpe比率
        excess_daily = daily_returns - self.rf_daily
        if daily_returns.std() > 0:
            report["sharpe_ratio"] = float(
                excess_daily.mean() / daily_returns.std() * np.sqrt(self.trading_days_per_year)
            )
        else:
            report["sharpe_ratio"] = 0.0

        # Sortino比率（仅考虑下行波动）
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            report["sortino_ratio"] = float(
                excess_daily.mean() / downside_returns.std() * np.sqrt(self.trading_days_per_year)
            )
        else:
            report["sortino_ratio"] = report.get("sharpe_ratio", 0.0)

        # Calmar比率（年化收益/最大回撤）
        if max_dd > 0:
            report["calmar_ratio"] = float(report["annualized_return"] / max_dd)
        else:
            report["calmar_ratio"] = float("inf") if report["annualized_return"] > 0 else 0.0

        # ========== 交易统计 ==========
        if trades is not None and not trades.empty:
            trade_metrics = self._compute_trade_metrics(trades)
            report.update(trade_metrics)
        else:
            report["total_trades"] = 0
            report["win_rate"] = 0.0
            report["profit_loss_ratio"] = 0.0

        # ========== 基准对比 ==========
        if benchmark_equity is not None:
            bm_metrics = self._compute_benchmark_comparison(
                equity, benchmark_equity, daily_returns
            )
            report.update(bm_metrics)

        # ========== 月度收益 ==========
        report["monthly_returns"] = self._compute_monthly_returns(equity)

        # ========== 滚动指标 ==========
        report["rolling_1y_sharpe"] = self._compute_rolling_sharpe(daily_returns, window=252)

        return report

    def _compute_max_drawdown(
        self, equity: Union[pd.Series, pd.DataFrame]
    ) -> tuple:
        """计算最大回撤及其持续时间。

        Args:
            equity: 权益曲线。

        Returns:
            (max_drawdown, start_date, end_date, duration_days)
        """
        if isinstance(equity, pd.DataFrame):
            eq = equity.iloc[:, 0]
        else:
            eq = equity

        # 滚动峰值
        rolling_max = eq.expanding().max()
        drawdown = (eq - rolling_max) / rolling_max

        max_dd = abs(drawdown.min())
        max_dd_idx = drawdown.idxmin()

        if pd.isna(max_dd_idx):
            return 0.0, None, None, 0

        # 找到回撤开始点（峰值）
        peak_idx = eq[:max_dd_idx].idxmax()
        if pd.isna(peak_idx):
            peak_idx = max_dd_idx

        # 找到回撤结束点（恢复到峰值之后的最低点）
        try:
            after_low = eq[max_dd_idx:]
            recovery_mask = after_low >= eq[peak_idx]
            if recovery_mask.any():
                recovery_idx = after_low[recovery_mask].index[0]
            else:
                recovery_idx = eq.index[-1]
        except Exception:
            recovery_idx = eq.index[-1]

        duration_days = (recovery_idx - peak_idx).days

        return max_dd, peak_idx, recovery_idx, duration_days

    def _compute_trade_metrics(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """计算交易统计指标。

        Args:
            trades: 交易记录DataFrame。

        Returns:
            交易指标字典。
        """
        metrics = {}

        # 总交易次数
        metrics["total_trades"] = len(trades)

        # 买卖次数
        if "side" in trades.columns:
            metrics["buy_trades"] = int((trades["side"] == "buy").sum())
            metrics["sell_trades"] = int((trades["side"] == "sell").sum())
        else:
            metrics["buy_trades"] = 0
            metrics["sell_trades"] = 0

        # 盈亏统计
        if "pnl" in trades.columns:
            sell_trades = trades[trades["side"] == "sell"] if "side" in trades.columns else trades
            pnl_data = sell_trades["pnl"].dropna()

            if len(pnl_data) > 0:
                profitable = pnl_data[pnl_data > 0]
                unprofitable = pnl_data[pnl_data < 0]

                metrics["total_pnl"] = float(pnl_data.sum())
                metrics["avg_pnl"] = float(pnl_data.mean())
                metrics["max_profit"] = float(pnl_data.max())
                metrics["max_loss"] = float(pnl_data.min())

                metrics["win_rate"] = float(len(profitable) / len(pnl_data))
                metrics["win_rate_pct"] = round(metrics["win_rate"] * 100, 2)

                if len(unprofitable) > 0:
                    avg_win = profitable.mean() if len(profitable) > 0 else 0
                    avg_loss = abs(unprofitable.mean())
                    metrics["profit_loss_ratio"] = float(avg_win / avg_loss) if avg_loss > 0 else float("inf")
                else:
                    metrics["profit_loss_ratio"] = float("inf")
            else:
                metrics["total_pnl"] = 0.0
                metrics["avg_pnl"] = 0.0
                metrics["win_rate"] = 0.0
                metrics["profit_loss_ratio"] = 0.0
        else:
            metrics["total_pnl"] = 0.0
            metrics["win_rate"] = 0.0
            metrics["profit_loss_ratio"] = 0.0

        # 换手率
        if "amount" in trades.columns:
            total_buy_amount = trades[trades["side"] == "buy"]["amount"].sum() if "side" in trades.columns else 0
            metrics["total_turnover"] = float(total_buy_amount)
        else:
            metrics["total_turnover"] = 0.0

        return metrics

    def _compute_benchmark_comparison(
        self,
        strategy_equity: pd.Series,
        benchmark_equity: Union[pd.Series, pd.DataFrame],
        strategy_daily_returns: pd.Series,
    ) -> Dict[str, Any]:
        """计算与基准的对比指标。

        Args:
            strategy_equity: 策略权益。
            benchmark_equity: 基准权益。
            strategy_daily_returns: 策略日收益率。

        Returns:
            对比指标字典。
        """
        if isinstance(benchmark_equity, pd.DataFrame):
            if "equity" in benchmark_equity.columns:
                bm_eq = benchmark_equity["equity"]
            else:
                bm_eq = benchmark_equity.iloc[:, 0]
        else:
            bm_eq = benchmark_equity

        # 对齐
        common = strategy_equity.index.intersection(bm_eq.index)
        if len(common) < 2:
            return {
                "benchmark_return": 0.0,
                "excess_return": 0.0,
                "tracking_error": 0.0,
                "information_ratio": 0.0,
            }

        s_eq = strategy_equity[common]
        b_eq = bm_eq[common]

        benchmark_return = float(b_eq.iloc[-1] / b_eq.iloc[0] - 1)
        strategy_return = float(s_eq.iloc[-1] / s_eq.iloc[0] - 1)
        excess_return = strategy_return - benchmark_return

        bm_daily = b_eq.pct_change().dropna()
        aligned_s = strategy_daily_returns[strategy_daily_returns.index.isin(bm_daily.index)]
        aligned_bm = bm_daily[bm_daily.index.isin(aligned_s.index)]

        excess_daily = aligned_s - aligned_bm
        tracking_error = float(excess_daily.std() * np.sqrt(self.trading_days_per_year))

        if tracking_error > 0:
            information_ratio = float(excess_daily.mean() * self.trading_days_per_year / tracking_error)
        else:
            information_ratio = 0.0

        return {
            "benchmark_return": round(benchmark_return, 6),
            "benchmark_return_pct": round(benchmark_return * 100, 2),
            "excess_return": round(excess_return, 6),
            "excess_return_pct": round(excess_return * 100, 2),
            "tracking_error": round(tracking_error, 6),
            "tracking_error_pct": round(tracking_error * 100, 2),
            "information_ratio": round(information_ratio, 4),
        }

    def _compute_monthly_returns(self, equity: pd.Series) -> Dict[str, float]:
        """计算月度收益表。

        Args:
            equity: 权益曲线。

        Returns:
            {YYYY-MM: return_pct} 字典。
        """
        if len(equity) < 2:
            return {}

        try:
            monthly_eq = equity.resample("ME").last().dropna()
            monthly_returns = {}
            prev = None
            for date, value in monthly_eq.items():
                if prev is not None and prev > 0:
                    ret = (value - prev) / prev
                    monthly_returns[date.strftime("%Y-%m")] = round(ret, 6)
                prev = value
            return monthly_returns
        except Exception as e:
            logger.warning(f"计算月度收益失败: {e}")
            return {}

    def _compute_rolling_sharpe(
        self, daily_returns: pd.Series, window: int = 252
    ) -> List[float]:
        """计算滚动Sharpe比率。

        Args:
            daily_returns: 日收益率。
            window: 滚动窗口（天数）。

        Returns:
            滚动Sharpe比率列表。
        """
        if len(daily_returns) < window:
            return []

        rolling_sharpe = []
        for i in range(window, len(daily_returns)):
            window_returns = daily_returns.iloc[i - window : i]
            if window_returns.std() > 0:
                sr = (
                    (window_returns.mean() - self.rf_daily)
                    / window_returns.std()
                    * np.sqrt(self.trading_days_per_year)
                )
            else:
                sr = 0.0
            rolling_sharpe.append(round(sr, 4))

        return rolling_sharpe

    def _empty_report(self) -> Dict[str, Any]:
        """返回空报告。"""
        return {
            "cumulative_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "win_rate": 0.0,
            "profit_loss_ratio": 0.0,
            "total_trades": 0,
            "monthly_returns": {},
        }

    def print_report(self, report_dict: Dict[str, Any]) -> None:
        """格式化打印业绩报告到控制台。

        Args:
            report_dict: generate_report返回的指标字典。
        """
        separator = "=" * 60
        header = "回测业绩报告"

        print(f"\n{separator}")
        print(f"{header:^60}")
        print(f"{separator}")

        # 基本信息
        print(f"\n[基本信息]")
        print(f"  回测区间: {report_dict.get('start_date', 'N/A')} ~ {report_dict.get('end_date', 'N/A')}")
        print(f"  交易日数: {report_dict.get('trading_days', 0)}")

        # 收益指标
        print(f"\n[收益指标]")
        print(f"  累计收益率: {report_dict.get('total_return_pct', 0)}%")
        print(f"  年化收益率: {report_dict.get('annualized_return_pct', 0)}%")
        if "excess_return_pct" in report_dict:
            print(f"  超额收益率: {report_dict['excess_return_pct']}%")

        # 风险指标
        print(f"\n[风险指标]")
        print(f"  年化波动率: {report_dict.get('annualized_volatility_pct', 0)}%")
        print(f"  最大回撤:   {report_dict.get('max_drawdown_pct', 0)}%")
        print(f"  回撤持续:   {report_dict.get('max_drawdown_duration_days', 0)} 天")

        # 风险调整收益
        print(f"\n[风险调整收益]")
        print(f"  Sharpe比率:  {report_dict.get('sharpe_ratio', 0):.4f}")
        print(f"  Sortino比率: {report_dict.get('sortino_ratio', 0):.4f}")
        print(f"  Calmar比率:  {report_dict.get('calmar_ratio', 0):.4f}")

        # 交易统计
        print(f"\n[交易统计]")
        print(f"  总交易次数: {report_dict.get('total_trades', 0)}")
        print(f"  胜率:       {report_dict.get('win_rate_pct', 0)}%")
        plr = report_dict.get('profit_loss_ratio', 0)
        if plr == float('inf'):
            print(f"  盈亏比:     无亏损交易")
        else:
            print(f"  盈亏比:     {plr:.2f}")

        # 基准对比
        if "benchmark_return_pct" in report_dict:
            print(f"\n[基准对比]")
            print(f"  基准收益率: {report_dict['benchmark_return_pct']}%")
            print(f"  超额收益率: {report_dict.get('excess_return_pct', 0)}%")
            print(f"  跟踪误差:   {report_dict.get('tracking_error_pct', 0)}%")
            print(f"  信息比率:   {report_dict.get('information_ratio', 0):.4f}")

        # 月度收益摘要
        monthly = report_dict.get("monthly_returns", {})
        if monthly:
            print(f"\n[月度收益摘要]")
            positive_months = sum(1 for v in monthly.values() if v > 0)
            print(f"  月度胜率: {positive_months}/{len(monthly)} ({positive_months/len(monthly)*100:.1f}%)")
            best_month = max(monthly, key=monthly.get)
            worst_month = min(monthly, key=monthly.get)
            print(f"  最佳月份: {best_month} ({monthly[best_month]*100:.2f}%)")
            print(f"  最差月份: {worst_month} ({monthly[worst_month]*100:.2f}%)")

        print(f"\n{separator}\n")

    def save_report(
        self,
        report_dict: Dict[str, Any],
        output_path: str,
        format: str = "json",
    ) -> None:
        """保存业绩报告到文件。

        Args:
            report_dict: 业绩指标字典。
            output_path: 输出文件路径。
            format: 输出格式 ("json" 或 "csv")。
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if format == "json":
                # 处理不可序列化的值
                serializable = {}
                for k, v in report_dict.items():
                    if isinstance(v, dict):
                        serializable[k] = v
                    elif v == float("inf"):
                        serializable[k] = "inf"
                    elif v == float("-inf"):
                        serializable[k] = "-inf"
                    elif pd.isna(v) if isinstance(v, float) else False:
                        serializable[k] = None
                    else:
                        serializable[k] = v

                with open(path, "w", encoding="utf-8") as f:
                    json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)
                logger.info(f"报告已保存: {path}")

            elif format == "csv":
                # 扁平化报告
                flat_report = {}
                for k, v in report_dict.items():
                    if not isinstance(v, (dict, list)):
                        flat_report[k] = v

                pd.DataFrame([flat_report]).to_csv(path, index=False, encoding="utf-8-sig")
                logger.info(f"报告已保存: {path}")

            else:
                raise ValueError(f"不支持的格式: {format}")

        except Exception as e:
            logger.error(f"保存报告失败: {e}")
            raise

    def generate_summary_text(self, report_dict: Dict[str, Any]) -> str:
        """生成简短的文字摘要。

        Args:
            report_dict: 业绩指标字典。

        Returns:
            摘要文字。
        """
        lines = [
            f"回测区间: {report_dict.get('start_date', 'N/A')} ~ {report_dict.get('end_date', 'N/A')}",
            f"累计收益: {report_dict.get('total_return_pct', 0):.2f}%",
            f"年化收益: {report_dict.get('annualized_return_pct', 0):.2f}%",
            f"最大回撤: {report_dict.get('max_drawdown_pct', 0):.2f}%",
            f"Sharpe比率: {report_dict.get('sharpe_ratio', 0):.2f}",
            f"胜率: {report_dict.get('win_rate_pct', 0):.2f}%",
        ]

        if "excess_return_pct" in report_dict:
            lines.append(f"超额收益: {report_dict['excess_return_pct']:.2f}%")

        return " | ".join(lines)


def compute_returns_from_trades(
    trades: pd.DataFrame,
    initial_capital: float,
    dates: pd.DatetimeIndex,
) -> pd.Series:
    """从交易记录重构日度收益率。

    辅助函数，用于从交易明细生成权益曲线。

    Args:
        trades: 交易记录DataFrame。
        initial_capital: 初始资金。
        dates: 回测日期范围。

    Returns:
        日度权益曲线Series。
    """
    if trades.empty:
        return pd.Series(initial_capital, index=dates)

    equity = pd.Series(initial_capital, index=dates, dtype=float)

    # 按日期累加盈亏
    if "pnl" in trades.columns and "date" in trades.columns:
        daily_pnl = trades.groupby("date")["pnl"].sum()
        for date, pnl in daily_pnl.items():
            if date in equity.index:
                equity[date] += pnl

    # 前向填充
    equity = equity.cumsum()

    return equity