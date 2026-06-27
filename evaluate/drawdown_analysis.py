"""
Drawdown analysis for the Stock-AI quantitative trading project.

Computes maximum drawdown, drawdown duration, drawdown periods, and
provides visualization of underwater equity curves.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DrawdownAnalyzer:
    """Drawdown analyzer for quantitative trading strategies.

    Computes drawdown metrics from equity curves, including maximum
    drawdown, drawdown duration, average drawdown, and drawdown periods.

    Drawdown measures the peak-to-trough decline during a specific period,
    providing insight into worst-case losses and recovery characteristics.

    Attributes:
        dd_df: DataFrame containing drawdown analysis results.
        summary: Dictionary of drawdown summary statistics.
    """

    def __init__(self):
        """Initialize the drawdown analyzer."""
        self.dd_df = None
        self.summary = None

    def compute_drawdowns(
        self,
        equity_curve: Union[np.ndarray, pd.Series],
        dates: Optional[Union[pd.Series, pd.DatetimeIndex]] = None,
    ) -> pd.DataFrame:
        """Compute drawdown series and drawdown periods from an equity curve.

        For each point in time, calculates the drawdown as:
            drawdown = (current_equity - peak_equity) / peak_equity

        Drawdown periods are identified as contiguous regions where
        equity is below the previous peak.

        Args:
            equity_curve: Array-like of equity values over time.
            dates: Optional date/time index for the equity curve.

        Returns:
            DataFrame with columns:
            - date (if provided): Date/time index
            - equity: Equity value
            - peak: Running maximum equity
            - drawdown: Drawdown percentage (negative value)
            - in_drawdown: Boolean indicating if in a drawdown period
            - drawdown_duration: Days since current drawdown began

        Raises:
            ValueError: If equity curve is invalid.
        """
        logger.info("Computing drawdowns: equity length=%d", len(equity_curve))

        equity = np.asarray(equity_curve, dtype=np.float64).ravel()
        equity = equity[~np.isnan(equity) & ~np.isinf(equity)]

        if len(equity) < 2:
            raise ValueError("Need at least 2 equity data points")

        # Compute running maximum (peak)
        peak = np.maximum.accumulate(equity)

        # Compute drawdown as percentage
        drawdown = (equity - peak) / peak

        # Identify drawdown periods
        in_drawdown = drawdown < 0

        # Compute drawdown duration
        duration = np.zeros(len(equity), dtype=int)
        dd_start = 0
        for i in range(len(equity)):
            if in_drawdown[i]:
                if i == 0 or not in_drawdown[i - 1]:
                    dd_start = i
                duration[i] = i - dd_start + 1
            else:
                duration[i] = 0

        result_data = {
            "equity": equity,
            "peak": peak,
            "drawdown": drawdown,
            "in_drawdown": in_drawdown,
            "drawdown_duration": duration,
        }

        if dates is not None:
            dates = np.asarray(dates)[: len(equity)]
            result_data["date"] = pd.to_datetime(dates)
            self.dd_df = pd.DataFrame(result_data)
            self.dd_df = self.dd_df[["date", "equity", "peak", "drawdown",
                                     "in_drawdown", "drawdown_duration"]]
        else:
            self.dd_df = pd.DataFrame(result_data)

        logger.info(
            "Drawdowns computed: max_dd=%.4f%%, avg_dd=%.4f%%",
            drawdown.min() * 100,
            drawdown[drawdown < 0].mean() * 100 if (drawdown < 0).any() else 0,
        )

        return self.dd_df

    def compute_max_drawdown(
        self,
        equity_curve: Union[np.ndarray, pd.Series],
        dates: Optional[Union[pd.Series, pd.DatetimeIndex]] = None,
    ) -> Tuple[float, str, str, int]:
        """Compute the maximum drawdown and its characteristics.

        Args:
            equity_curve: Array-like of equity values.
            dates: Optional date index for identifying start/end dates.

        Returns:
            Tuple of (max_dd, start_date, end_date, duration_days) where:
            - max_dd: Maximum drawdown as a negative decimal
            - start_date: Date when max drawdown started (or index if no dates)
            - end_date: Date when max drawdown ended (fully recovered)
            - duration_days: Duration of the max drawdown in days

        Raises:
            ValueError: If equity curve is invalid.
        """
        logger.info("Computing maximum drawdown")

        equity = np.asarray(equity_curve, dtype=np.float64).ravel()
        equity = equity[~np.isnan(equity) & ~np.isinf(equity)]

        if len(equity) < 2:
            raise ValueError("Need at least 2 equity data points")

        # Find the maximum drawdown
        peak_idx = 0
        max_dd = 0.0
        max_dd_peak_idx = 0
        max_dd_trough_idx = 0
        max_dd_recovery_idx = len(equity) - 1

        for i in range(len(equity)):
            if equity[i] > equity[peak_idx]:
                peak_idx = i

            dd = (equity[i] - equity[peak_idx]) / equity[peak_idx]
            if dd < max_dd:
                max_dd = dd
                max_dd_peak_idx = peak_idx
                max_dd_trough_idx = i

        # Find recovery point (when equity exceeds previous peak)
        max_dd_recovery_idx = max_dd_trough_idx
        for i in range(max_dd_trough_idx + 1, len(equity)):
            if equity[i] >= equity[max_dd_peak_idx]:
                max_dd_recovery_idx = i
                break

        duration_days = max_dd_recovery_idx - max_dd_peak_idx

        # Format dates
        if dates is not None:
            dates_arr = pd.to_datetime(np.asarray(dates))
            start_date = str(dates_arr[max_dd_peak_idx].date())
            end_date = str(dates_arr[max_dd_recovery_idx].date())
        else:
            start_date = str(max_dd_peak_idx)
            end_date = str(max_dd_recovery_idx)

        logger.info(
            "Max drawdown: %.4f%% from %s to %s (duration=%d days)",
            max_dd * 100,
            start_date,
            end_date,
            duration_days,
        )

        return max_dd, start_date, end_date, duration_days

    def compute_drawdown_report(
        self,
        equity_curve: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> Dict[str, float]:
        """Compute a comprehensive drawdown summary report.

        Includes max drawdown, average drawdown, drawdown standard
        deviation, longest drawdown duration, number of drawdown periods,
        and recovery statistics.

        Args:
            equity_curve: Array-like of equity values. If None, uses
                previously computed drawdown DataFrame.

        Returns:
            Dictionary with keys:
            - max_dd: Maximum drawdown (negative decimal)
            - avg_dd: Average of all negative drawdowns
            - dd_std: Standard deviation of drawdowns
            - longest_dd_duration: Longest drawdown duration in days
            - n_drawdown_periods: Number of distinct drawdown periods
            - avg_dd_duration: Average drawdown duration in days
            - max_dd_duration: Maximum drawdown duration in days
            - current_dd: Current drawdown (if still in drawdown)
            - recovery_rate: Fraction of drawdowns that fully recovered

        Raises:
            ValueError: If no drawdown data is available.
        """
        logger.info("Computing drawdown report")

        if equity_curve is not None:
            self.compute_drawdowns(equity_curve)

        if self.dd_df is None or len(self.dd_df) == 0:
            raise ValueError(
                "No drawdown data available. Run compute_drawdowns() first."
            )

        df = self.dd_df

        # Key drawdown metrics
        max_dd = df["drawdown"].min()
        negative_dds = df[df["drawdown"] < 0]["drawdown"]
        avg_dd = negative_dds.mean() if len(negative_dds) > 0 else 0.0
        dd_std = df["drawdown"].std()

        # Drawdown periods
        # Count distinct drawdown periods (transitions from not in DD to in DD)
        if "in_drawdown" in df.columns:
            transitions = df["in_drawdown"].astype(int).diff().fillna(0)
            n_dd_periods = int((transitions == 1).sum())
        else:
            n_dd_periods = 0

        # Duration statistics
        longest_dd_duration = df["drawdown_duration"].max()
        avg_dd_duration = (
            df[df["in_drawdown"]]["drawdown_duration"].mean()
            if "in_drawdown" in df.columns and df["in_drawdown"].any()
            else 0.0
        )
        max_dd_duration = longest_dd_duration

        # Current drawdown status
        current_dd = df["drawdown"].iloc[-1]

        # Recovery rate estimation
        if "in_drawdown" in df.columns:
            # Count transitions from in DD to not in DD
            exit_transitions = (transitions == -1).sum()
            if n_dd_periods > 0:
                recovery_rate = exit_transitions / n_dd_periods
            else:
                recovery_rate = 1.0
        else:
            recovery_rate = 0.0

        self.summary = {
            "max_dd": float(max_dd),
            "avg_dd": float(avg_dd),
            "dd_std": float(dd_std),
            "longest_dd_duration": int(longest_dd_duration),
            "max_dd_duration": int(max_dd_duration),
            "avg_dd_duration": float(avg_dd_duration),
            "n_drawdown_periods": int(n_dd_periods),
            "current_dd": float(current_dd),
            "recovery_rate": float(recovery_rate),
        }

        logger.info(
            "Drawdown report: max_dd=%.4f%%, n_periods=%d, recovery_rate=%.2f%%",
            max_dd * 100,
            n_dd_periods,
            recovery_rate * 100,
        )

        return self.summary

    def plot_drawdown(
        self,
        equity_curve: Optional[Union[np.ndarray, pd.Series]] = None,
        save_path: Optional[str] = None,
    ) -> bool:
        """Plot the equity curve with drawdown overlay.

        Creates a dual-panel chart showing the equity curve on top
        and the drawdown series below.

        Args:
            equity_curve: Array-like of equity values.
                If None, uses previously computed drawdown DataFrame.
            save_path: Path to save the plot image.

        Returns:
            True if plot was successfully saved, False otherwise.
        """
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib is not installed. Cannot generate plots.")
            return False

        if equity_curve is not None:
            self.compute_drawdowns(equity_curve)

        if self.dd_df is None or len(self.dd_df) == 0:
            logger.error("No drawdown data available for plotting")
            return False

        try:
            df = self.dd_df
            has_dates = "date" in df.columns
            x_axis = df["date"] if has_dates else range(len(df))

            fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})

            # Top panel: Equity curve with underwater fill
            ax = axes[0]
            ax.plot(x_axis, df["equity"], color="steelblue", linewidth=1.5,
                    label="Equity Curve")
            ax.plot(x_axis, df["peak"], color="gray", linestyle="--", alpha=0.5,
                    label="Running Peak")
            ax.fill_between(
                x_axis,
                df["equity"],
                df["peak"],
                where=df["equity"] < df["peak"],
                color="red",
                alpha=0.15,
                label="Drawdown Region",
            )
            ax.set_ylabel("Equity")
            ax.set_title("Equity Curve with Drawdown Analysis")
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)

            # Bottom panel: Drawdown percentage
            ax = axes[1]
            ax.fill_between(
                x_axis, 0, df["drawdown"] * 100, color="red", alpha=0.5
            )
            ax.plot(x_axis, df["drawdown"] * 100, color="darkred", linewidth=1)
            ax.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
            ax.set_ylabel("Drawdown (%)")
            ax.set_xlabel("Date" if has_dates else "Index")
            ax.set_title(
                f"Drawdown (Max DD: {df['drawdown'].min() * 100:.2f}%)"
            )
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                logger.info("Drawdown plot saved to %s", save_path)

            plt.close()
            return True
        except Exception as e:
            logger.error("Failed to generate drawdown plot: %s", e)
            plt.close("all")
            return False

    def get_drawdown_periods(
        self,
        min_drawdown: float = -0.05,
    ) -> pd.DataFrame:
        """Extract distinct drawdown periods that exceed a minimum threshold.

        Args:
            min_drawdown: Minimum drawdown threshold (negative decimal).
                Only periods with drawdown below this threshold are included.

        Returns:
            DataFrame with columns:
            - start_idx: Start index of drawdown period
            - end_idx: End index of drawdown period
            - max_dd: Maximum drawdown during this period
            - duration: Duration in days
            - recovered: Whether equity recovered to previous peak
        """
        if self.dd_df is None or len(self.dd_df) == 0:
            logger.warning("No drawdown data available")
            return pd.DataFrame()

        df = self.dd_df

        periods = []
        in_period = False
        period_start = 0
        period_min_dd = 0.0

        for i in range(len(df)):
            dd = df["drawdown"].iloc[i]

            if dd < min_drawdown:
                if not in_period:
                    in_period = True
                    period_start = i
                    period_min_dd = dd
                else:
                    period_min_dd = min(period_min_dd, dd)
            elif in_period:
                # Period ended
                periods.append(
                    {
                        "start_idx": period_start,
                        "end_idx": i - 1,
                        "max_dd": period_min_dd,
                        "duration": i - period_start,
                        "recovered": True,
                    }
                )
                in_period = False

        # Handle ongoing drawdown
        if in_period:
            periods.append(
                {
                    "start_idx": period_start,
                    "end_idx": len(df) - 1,
                    "max_dd": period_min_dd,
                    "duration": len(df) - period_start,
                    "recovered": False,
                }
            )

        result = pd.DataFrame(periods)
        if len(result) > 0:
            result = result.sort_values("max_dd").reset_index(drop=True)

        logger.info("Found %d drawdown periods below threshold", len(result))
        return result