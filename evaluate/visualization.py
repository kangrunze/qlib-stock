"""
Visualization tools for the Stock-AI quantitative trading project.

Provides comprehensive plotting utilities for equity curves, drawdowns,
IC series, feature importance, and return distributions.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Visualizer:
    """Visualization toolkit for quantitative trading analysis.

    Provides standard plotting functions for model evaluation,
    strategy performance, and feature analysis.

    All plot methods save to file and return True on success or
    False on failure.

    Attributes:
        style: Matplotlib style to apply.
        dpi: Default DPI for saved figures.
        figsize: Default figure size tuple.
    """

    def __init__(
        self,
        style: str = "seaborn-v0_8-darkgrid",
        dpi: int = 150,
        figsize: tuple = (14, 8),
    ):
        """Initialize the visualizer.

        Args:
            style: Matplotlib style name.
            dpi: Default DPI for saved figures.
            figsize: Default figure size (width, height).
        """
        self.style = style
        self.dpi = dpi
        self.figsize = figsize

        try:
            import matplotlib.pyplot as plt

            plt.style.use(style)
        except (ImportError, Exception):
            pass  # Style might not be available

    def _get_matplotlib(self) -> Optional[Any]:
        """Import matplotlib with Agg backend for non-interactive use.

        Returns:
            matplotlib module or None if not installed.
        """
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            return plt
        except ImportError:
            logger.warning("matplotlib is not installed.")
            return None

    def _save_and_close(
        self, plt: Any, save_path: str, plot_name: str
    ) -> bool:
        """Save figure to file and close.

        Args:
            plt: Matplotlib pyplot module.
            save_path: Path to save the plot.
            plot_name: Name of the plot type for logging.

        Returns:
            True if saved successfully, False otherwise.
        """
        try:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            plt.close()
            logger.info("%s plot saved to %s", plot_name, save_path)
            return True
        except Exception as e:
            logger.error("Failed to save %s plot: %s", plot_name, e)
            plt.close("all")
            return False

    def plot_equity_curve(
        self,
        equity: Union[np.ndarray, pd.Series],
        benchmark_equity: Optional[Union[np.ndarray, pd.Series]] = None,
        save_path: Optional[str] = None,
        title: str = "Equity Curve",
        dates: Optional[Union[pd.Series, pd.DatetimeIndex]] = None,
    ) -> bool:
        """Plot equity curve with optional benchmark comparison.

        Creates a chart showing strategy equity alongside a benchmark
        equity curve for performance comparison.

        Args:
            equity: Strategy equity values.
            benchmark_equity: Benchmark equity values for comparison.
            save_path: Path to save the plot image.
            title: Plot title.
            dates: Date/time index for x-axis.

        Returns:
            True if plot was saved successfully, False otherwise.
        """
        plt = self._get_matplotlib()
        if plt is None:
            return False

        try:
            equity = np.asarray(equity, dtype=np.float64).ravel()
            equity = equity[~np.isnan(equity) & ~np.isinf(equity)]

            fig, ax = plt.subplots(figsize=self.figsize)

            # X-axis
            if dates is not None:
                dates = pd.to_datetime(np.asarray(dates))[: len(equity)]
                x_axis = dates
            else:
                x_axis = range(len(equity))

            # Normalize equity to start at 1 for comparison
            equity_norm = equity / equity[0]

            # Plot strategy equity
            ax.plot(
                x_axis,
                equity_norm,
                color="steelblue",
                linewidth=2,
                label="Strategy",
            )

            # Plot benchmark equity
            if benchmark_equity is not None:
                bench = np.asarray(benchmark_equity, dtype=np.float64).ravel()
                bench = bench[~np.isnan(bench) & ~np.isinf(bench)][: len(equity)]
                bench_norm = bench / bench[0]
                ax.plot(
                    x_axis[: len(bench_norm)],
                    bench_norm,
                    color="darkorange",
                    linewidth=2,
                    linestyle="--",
                    label="Benchmark",
                )

            # Shade outperformance/underperformance
            if benchmark_equity is not None:
                bench_interp = np.interp(
                    np.arange(len(equity_norm)),
                    np.arange(len(bench_norm)),
                    bench_norm,
                )
                ax.fill_between(
                    x_axis,
                    equity_norm,
                    bench_interp,
                    where=equity_norm >= bench_interp,
                    color="green",
                    alpha=0.1,
                    interpolate=True,
                )
                ax.fill_between(
                    x_axis,
                    equity_norm,
                    bench_interp,
                    where=equity_norm < bench_interp,
                    color="red",
                    alpha=0.1,
                    interpolate=True,
                )

            ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
            ax.set_ylabel("Normalized Equity (Start = 1.0)")
            ax.set_xlabel("Date" if dates is not None else "Trading Days")
            ax.set_title(title)
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            return self._save_and_close(plt, save_path, "Equity Curve")
        except Exception as e:
            logger.error("Failed to generate equity curve plot: %s", e)
            plt.close("all")
            return False

    def plot_drawdown(
        self,
        equity_curve: Union[np.ndarray, pd.Series],
        save_path: Optional[str] = None,
        dates: Optional[Union[pd.Series, pd.DatetimeIndex]] = None,
    ) -> bool:
        """Plot drawdown (underwater) chart for an equity curve.

        Args:
            equity_curve: Equity values.
            save_path: Path to save the plot image.
            dates: Date/time index for x-axis.

        Returns:
            True if plot was saved successfully, False otherwise.
        """
        plt = self._get_matplotlib()
        if plt is None:
            return False

        try:
            equity = np.asarray(equity_curve, dtype=np.float64).ravel()
            equity = equity[~np.isnan(equity) & ~np.isinf(equity)]

            if len(equity) < 2:
                logger.error("Need at least 2 equity data points for drawdown plot")
                return False

            # Compute drawdown
            peak = np.maximum.accumulate(equity)
            drawdown = (equity - peak) / peak * 100  # Percentage

            # X-axis
            if dates is not None:
                dates = pd.to_datetime(np.asarray(dates))[: len(equity)]
                x_axis = dates
            else:
                x_axis = range(len(equity))

            fig, ax = plt.subplots(figsize=self.figsize)

            ax.fill_between(
                x_axis, 0, drawdown, color="red", alpha=0.5
            )
            ax.plot(x_axis, drawdown, color="darkred", linewidth=1)
            ax.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
            ax.set_ylabel("Drawdown (%)")
            ax.set_xlabel("Date" if dates is not None else "Trading Days")
            ax.set_title(
                f"Drawdown Chart (Max DD: {drawdown.min():.2f}%)"
            )
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            return self._save_and_close(plt, save_path, "Drawdown")
        except Exception as e:
            logger.error("Failed to generate drawdown plot: %s", e)
            plt.close("all")
            return False

    def plot_ic_series(
        self,
        ic_df: pd.DataFrame,
        save_path: Optional[str] = None,
        window: int = 20,
    ) -> bool:
        """Plot IC and RankIC time series with rolling means.

        Args:
            ic_df: DataFrame with columns 'date', 'IC', 'RankIC'.
            save_path: Path to save the plot image.
            window: Rolling window size for computing rolling mean.

        Returns:
            True if plot was saved successfully, False otherwise.
        """
        plt = self._get_matplotlib()
        if plt is None:
            return False

        try:
            if len(ic_df) == 0:
                logger.error("Empty IC DataFrame")
                return False

            has_dates = "date" in ic_df.columns
            x_axis = (
                ic_df["date"] if has_dates else range(len(ic_df))
            )

            fig, axes = plt.subplots(2, 1, figsize=(self.figsize[0], 10),
                                     sharex=True)

            # IC plot
            ax = axes[0]
            ax.plot(x_axis, ic_df["IC"], alpha=0.3, color="steelblue",
                    label="Daily IC")
            if len(ic_df) >= window:
                rolling_ic = ic_df["IC"].rolling(window=window).mean()
                ax.plot(x_axis, rolling_ic, color="darkblue", linewidth=2,
                        label=f"IC Rolling Mean ({window}d)")
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax.axhline(y=ic_df["IC"].mean(), color="red", linestyle="--",
                       alpha=0.7, label=f"Mean IC: {ic_df['IC'].mean():.4f}")
            ax.set_ylabel("IC (Pearson)")
            ax.set_title("Information Coefficient (IC) Time Series")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)

            # RankIC plot
            ax = axes[1]
            ax.plot(x_axis, ic_df["RankIC"], alpha=0.3, color="seagreen",
                    label="Daily RankIC")
            if len(ic_df) >= window:
                rolling_rankic = ic_df["RankIC"].rolling(window=window).mean()
                ax.plot(x_axis, rolling_rankic, color="darkgreen", linewidth=2,
                        label=f"RankIC Rolling Mean ({window}d)")
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax.axhline(y=ic_df["RankIC"].mean(), color="red", linestyle="--",
                       alpha=0.7,
                       label=f"Mean RankIC: {ic_df['RankIC'].mean():.4f}")
            ax.set_ylabel("RankIC (Spearman)")
            ax.set_xlabel("Date" if has_dates else "Period")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            return self._save_and_close(plt, save_path, "IC Series")
        except Exception as e:
            logger.error("Failed to generate IC series plot: %s", e)
            plt.close("all")
            return False

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 20,
        save_path: Optional[str] = None,
        title: str = "Feature Importance",
    ) -> bool:
        """Plot feature importance as a horizontal bar chart.

        Args:
            importance_df: DataFrame with columns 'feature' and 'importance'
                (or 'shap_importance'). Sorted descending by importance.
            top_n: Number of top features to display.
            save_path: Path to save the plot image.
            title: Plot title.

        Returns:
            True if plot was saved successfully, False otherwise.
        """
        plt = self._get_matplotlib()
        if plt is None:
            return False

        try:
            if len(importance_df) == 0:
                logger.error("Empty importance DataFrame")
                return False

            # Determine importance column
            if "importance" in importance_df.columns:
                imp_col = "importance"
            elif "shap_importance" in importance_df.columns:
                imp_col = "shap_importance"
            else:
                imp_col = importance_df.columns[1]

            # Select top N
            df = importance_df.head(top_n).copy()
            df = df.sort_values(imp_col, ascending=True)

            fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.3)))

            # Color gradient
            colors = plt.cm.viridis(
                np.linspace(0.3, 0.9, len(df))
            )

            ax.barh(df["feature"], df[imp_col], color=colors, edgecolor="gray",
                    linewidth=0.5)
            ax.set_xlabel("Importance")
            ax.set_ylabel("Feature")
            ax.set_title(title)
            ax.grid(True, alpha=0.3, axis="x")

            plt.tight_layout()
            return self._save_and_close(plt, save_path, "Feature Importance")
        except Exception as e:
            logger.error("Failed to generate feature importance plot: %s", e)
            plt.close("all")
            return False

    def plot_return_distribution(
        self,
        returns: Union[np.ndarray, pd.Series],
        save_path: Optional[str] = None,
        bins: int = 100,
        title: str = "Return Distribution",
    ) -> bool:
        """Plot return distribution histogram with normal fit overlay.

        Includes key statistics (mean, std, skewness, kurtosis) in a
        text box on the plot.

        Args:
            returns: Daily return series.
            save_path: Path to save the plot image.
            bins: Number of histogram bins.
            title: Plot title.

        Returns:
            True if plot was saved successfully, False otherwise.
        """
        plt = self._get_matplotlib()
        if plt is None:
            return False

        try:
            import numpy as np

            returns = np.asarray(returns, dtype=np.float64).ravel()
            returns = returns[~np.isnan(returns) & ~np.isinf(returns)]

            if len(returns) < 2:
                logger.error("Need at least 2 return data points")
                return False

            fig, ax = plt.subplots(figsize=self.figsize)

            # Histogram
            n, bins_arr, patches = ax.hist(
                returns * 100, bins=bins, density=True, alpha=0.7,
                color="steelblue", edgecolor="white", label="Returns (%)"
            )

            # Normal distribution overlay
            mu = returns.mean() * 100
            sigma = returns.std() * 100
            x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
            y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
                -0.5 * ((x - mu) / sigma) ** 2
            )
            ax.plot(x, y, color="darkorange", linewidth=2,
                    label=f"Normal Fit (mu={mu:.2f}%, sigma={sigma:.2f}%)")

            # Compute statistics
            from scipy.stats import skew, kurtosis

            skew_val = skew(returns)
            kurt_val = kurtosis(returns)

            # Statistics text box
            stats_text = (
                f"Mean: {returns.mean() * 100:.4f}%\n"
                f"Std: {returns.std() * 100:.4f}%\n"
                f"Skewness: {skew_val:.4f}\n"
                f"Kurtosis: {kurt_val:.4f}\n"
                f"Sharpe: {returns.mean() / returns.std() * np.sqrt(252):.4f}\n"
                f"N: {len(returns)}"
            )
            ax.text(
                0.05,
                0.95,
                stats_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

            ax.set_xlabel("Return (%)")
            ax.set_ylabel("Density")
            ax.set_title(title)
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            return self._save_and_close(plt, save_path, "Return Distribution")
        except Exception as e:
            logger.error("Failed to generate return distribution plot: %s", e)
            plt.close("all")
            return False

    def plot_rolling_sharpe(
        self,
        rolling_sharpe_df: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> bool:
        """Plot rolling Sharpe ratio over time.

        Args:
            rolling_sharpe_df: DataFrame with columns 'date' and 'sharpe'.
            save_path: Path to save the plot image.

        Returns:
            True if plot was saved successfully, False otherwise.
        """
        plt = self._get_matplotlib()
        if plt is None:
            return False

        try:
            fig, ax = plt.subplots(figsize=self.figsize)

            has_dates = "date" in rolling_sharpe_df.columns
            x_axis = (
                rolling_sharpe_df["date"]
                if has_dates
                else range(len(rolling_sharpe_df))
            )

            ax.plot(x_axis, rolling_sharpe_df["sharpe"], color="steelblue",
                    linewidth=1.5, label="Rolling Sharpe")
            ax.axhline(
                y=rolling_sharpe_df["sharpe"].mean(),
                color="red",
                linestyle="--",
                linewidth=1,
                alpha=0.7,
                label=f"Mean: {rolling_sharpe_df['sharpe'].mean():.4f}",
            )
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax.set_ylabel("Sharpe Ratio")
            ax.set_xlabel("Date" if has_dates else "Period")
            ax.set_title("Rolling Sharpe Ratio")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            return self._save_and_close(plt, save_path, "Rolling Sharpe")
        except Exception as e:
            logger.error("Failed to generate rolling Sharpe plot: %s", e)
            plt.close("all")
            return False

    def plot_cumulative_returns(
        self,
        returns: Union[np.ndarray, pd.Series],
        save_path: Optional[str] = None,
        dates: Optional[Union[pd.Series, pd.DatetimeIndex]] = None,
        title: str = "Cumulative Returns",
    ) -> bool:
        """Plot cumulative return curve.

        Args:
            returns: Daily return series.
            save_path: Path to save the plot image.
            dates: Date/time index for x-axis.
            title: Plot title.

        Returns:
            True if plot was saved successfully, False otherwise.
        """
        plt = self._get_matplotlib()
        if plt is None:
            return False

        try:
            returns = np.asarray(returns, dtype=np.float64).ravel()
            returns = returns[~np.isnan(returns) & ~np.isinf(returns)]

            cum_returns = np.cumprod(1 + returns) - 1

            if dates is not None:
                dates = pd.to_datetime(np.asarray(dates))[: len(cum_returns)]
                x_axis = dates
            else:
                x_axis = range(len(cum_returns))

            fig, ax = plt.subplots(figsize=self.figsize)

            ax.plot(x_axis, cum_returns * 100, color="steelblue", linewidth=2)
            ax.fill_between(
                x_axis,
                0,
                cum_returns * 100,
                where=cum_returns >= 0,
                color="green",
                alpha=0.15,
                label="Positive",
            )
            ax.fill_between(
                x_axis,
                cum_returns * 100,
                0,
                where=cum_returns < 0,
                color="red",
                alpha=0.15,
                label="Negative",
            )
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax.set_ylabel("Cumulative Return (%)")
            ax.set_xlabel("Date" if dates is not None else "Trading Days")
            ax.set_title(
                f"{title} (Total: {cum_returns[-1] * 100:.2f}%)"
            )
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            return self._save_and_close(plt, save_path, "Cumulative Returns")
        except Exception as e:
            logger.error("Failed to generate cumulative returns plot: %s", e)
            plt.close("all")
            return False

    def plot_ic_decay(
        self,
        ic_decay_df: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> bool:
        """Plot IC decay across different forward return horizons.

        Args:
            ic_decay_df: DataFrame with columns 'horizon', 'IC', 'RankIC'.
            save_path: Path to save the plot image.

        Returns:
            True if plot was saved successfully, False otherwise.
        """
        plt = self._get_matplotlib()
        if plt is None:
            return False

        try:
            if len(ic_decay_df) == 0:
                logger.error("Empty IC decay DataFrame")
                return False

            fig, ax = plt.subplots(figsize=(10, 6))

            x = range(len(ic_decay_df))

            ax.plot(x, ic_decay_df["IC"], "o-", color="steelblue",
                    linewidth=2, markersize=8, label="IC (Pearson)")
            ax.plot(x, ic_decay_df["RankIC"], "s-", color="seagreen",
                    linewidth=2, markersize=8, label="RankIC (Spearman)")
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(ic_decay_df["horizon"], rotation=45)
            ax.set_ylabel("IC Value")
            ax.set_xlabel("Forward Return Horizon")
            ax.set_title("IC Decay Analysis: Predictive Power by Horizon")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)

            # Add value labels
            for i, (ic, rankic) in enumerate(
                zip(ic_decay_df["IC"], ic_decay_df["RankIC"])
            ):
                ax.text(i, ic, f"{ic:.4f}", ha="center", va="bottom",
                        fontsize=9)
                ax.text(i, rankic, f"{rankic:.4f}", ha="center", va="top",
                        fontsize=9)

            plt.tight_layout()
            return self._save_and_close(plt, save_path, "IC Decay")
        except Exception as e:
            logger.error("Failed to generate IC decay plot: %s", e)
            plt.close("all")
            return False