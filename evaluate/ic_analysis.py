"""
IC (Information Coefficient) analysis for the Stock-AI quantitative trading project.

Computes daily IC and RankIC metrics to evaluate model predictive power,
including ICIR, IC decay analysis, and visualization.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)


class ICAnalyzer:
    """Information Coefficient (IC) analyzer for quantitative trading models.

    Computes Pearson IC and Spearman RankIC between model predictions and
    future returns, along with derived metrics like ICIR and IC decay.

    In quantitative finance, IC measures how well predictions correlate
    with actual future returns. A higher IC indicates better predictive
    power.

    Attributes:
        ic_df: DataFrame containing daily IC and RankIC values.
        summary: Dictionary of IC summary statistics.
        decay_results: Dictionary of IC decay analysis results.
    """

    def __init__(self):
        """Initialize the IC analyzer."""
        self.ic_df = None
        self.summary = None
        self.decay_results = None

    def compute_ic(
        self,
        predictions: Union[np.ndarray, pd.Series],
        returns: Union[np.ndarray, pd.Series],
        dates: Optional[Union[pd.Series, pd.DatetimeIndex]] = None,
    ) -> pd.DataFrame:
        """Compute daily IC and RankIC between predictions and returns.

        Args:
            predictions: Model predictions.
            returns: Actual future returns.
            dates: Date index for grouping daily IC computation.
                If None, computes a single IC over all data.

        Returns:
            DataFrame with columns:
            - date: Date of IC computation
            - IC: Pearson correlation
            - RankIC: Spearman rank correlation
            - n_samples: Number of samples per date

        Raises:
            ValueError: If input data is invalid.
        """
        logger.info(
            "Computing IC: predictions shape=%s, returns shape=%s",
            np.asarray(predictions).shape,
            np.asarray(returns).shape,
        )

        # Convert to numpy arrays
        preds = np.asarray(predictions, dtype=np.float64).ravel()
        rets = np.asarray(returns, dtype=np.float64).ravel()

        if len(preds) != len(rets):
            raise ValueError(
                f"predictions and returns must have same length. "
                f"Got {len(preds)} vs {len(rets)}"
            )

        # Remove NaN/inf
        valid_mask = (
            ~np.isnan(preds)
            & ~np.isnan(rets)
            & ~np.isinf(preds)
            & ~np.isinf(rets)
        )
        preds = preds[valid_mask]
        rets = rets[valid_mask]

        if dates is not None:
            dates = np.asarray(dates)[valid_mask]

        if dates is not None and len(preds) < 10:
            logger.warning(
                "Fewer than 10 valid samples for cross-sectional IC computation"
            )
            return pd.DataFrame(columns=["date", "IC", "RankIC", "n_samples"])

        if dates is not None:
            # Compute cross-sectional IC per date
            unique_dates = np.unique(dates)
            results = []

            for date in unique_dates:
                mask = dates == date
                if mask.sum() < 5:
                    continue

                date_preds = preds[mask]
                date_rets = rets[mask]

                # Compute Pearson IC
                ic, _ = pearsonr(date_preds, date_rets)

                # Compute Spearman RankIC
                rankic, _ = spearmanr(date_preds, date_rets)

                results.append(
                    {
                        "date": date,
                        "IC": ic if not np.isnan(ic) else 0.0,
                        "RankIC": rankic if not np.isnan(rankic) else 0.0,
                        "n_samples": mask.sum(),
                    }
                )

            self.ic_df = pd.DataFrame(results)
            self.ic_df["date"] = pd.to_datetime(self.ic_df["date"])
            self.ic_df = self.ic_df.sort_values("date").reset_index(drop=True)

            logger.info(
                "Cross-sectional IC computed: %d dates, mean IC=%.6f, mean RankIC=%.6f",
                len(self.ic_df),
                self.ic_df["IC"].mean(),
                self.ic_df["RankIC"].mean(),
            )
        else:
            # Compute single IC over all data
            if len(preds) < 2:
                logger.warning("Fewer than 2 valid samples for single IC computation")
                return pd.DataFrame(columns=["date", "IC", "RankIC", "n_samples"])

            ic, _ = pearsonr(preds, rets)
            rankic, _ = spearmanr(preds, rets)

            self.ic_df = pd.DataFrame(
                [
                    {
                        "date": pd.Timestamp.now(),
                        "IC": ic if not np.isnan(ic) else 0.0,
                        "RankIC": rankic if not np.isnan(rankic) else 0.0,
                        "n_samples": len(preds),
                    }
                ]
            )

            logger.info(
                "Single IC computed: IC=%.6f, RankIC=%.6f, n=%d",
                ic,
                rankic,
                len(preds),
            )

        return self.ic_df

    def compute_ic_summary(
        self, ic_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """Compute summary statistics of IC values.

        Calculates mean IC, IC standard deviation, ICIR (Information
        Coefficient Information Ratio = mean/std), IC win rate (fraction
        of positive IC days), and mean RankIC.

        Args:
            ic_df: IC DataFrame (uses stored ic_df if None).

        Returns:
            Dictionary with keys:
            - mean_ic: Mean Pearson IC
            - std_ic: Standard deviation of IC
            - icir: IC Information Ratio (mean_ic / std_ic)
            - ic_win_rate: Fraction of days with positive IC
            - mean_rankic: Mean Spearman RankIC
            - std_rankic: Standard deviation of RankIC
            - rank_icir: RankIC Information Ratio
            - n_days: Number of IC computation days

        Raises:
            ValueError: If no IC data is available.
        """
        df = ic_df if ic_df is not None else self.ic_df

        if df is None or len(df) == 0:
            raise ValueError(
                "No IC data available. Run compute_ic() first."
            )

        mean_ic = df["IC"].mean()
        std_ic = df["IC"].std()
        icir = mean_ic / std_ic if std_ic > 0 else 0.0
        ic_win_rate = (df["IC"] > 0).mean()

        mean_rankic = df["RankIC"].mean()
        std_rankic = df["RankIC"].std()
        rank_icir = mean_rankic / std_rankic if std_rankic > 0 else 0.0

        self.summary = {
            "mean_ic": float(mean_ic),
            "std_ic": float(std_ic),
            "icir": float(icir),
            "ic_win_rate": float(ic_win_rate),
            "mean_rankic": float(mean_rankic),
            "std_rankic": float(std_rankic),
            "rank_icir": float(rank_icir),
            "n_days": len(df),
        }

        logger.info(
            "IC Summary: mean_ic=%.6f, icir=%.4f, win_rate=%.2f%%, mean_rankic=%.6f",
            mean_ic,
            icir,
            ic_win_rate * 100,
            mean_rankic,
        )

        return self.summary

    def compute_ic_decay(
        self,
        predictions: np.ndarray,
        returns_multi_horizon: Dict[str, np.ndarray],
    ) -> pd.DataFrame:
        """Compute IC decay over different forward return horizons.

        Evaluates how predictive power decays as the forward return
        horizon increases (e.g., 1-day, 5-day, 10-day, 20-day returns).

        Args:
            predictions: Model predictions.
            returns_multi_horizon: Dictionary mapping horizon labels
                (e.g., '1d', '5d', '10d') to corresponding return arrays.

        Returns:
            DataFrame with columns:
            - horizon: Horizon label
            - IC: Pearson IC at this horizon
            - RankIC: Spearman RankIC at this horizon

        Raises:
            ValueError: If input data is invalid.
        """
        logger.info(
            "Computing IC decay over %d horizons", len(returns_multi_horizon)
        )

        preds = np.asarray(predictions, dtype=np.float64).ravel()
        results = []

        for horizon_label, rets in returns_multi_horizon.items():
            rets = np.asarray(rets, dtype=np.float64).ravel()

            if len(preds) != len(rets):
                logger.warning(
                    "Length mismatch for horizon '%s': %d vs %d, skipping",
                    horizon_label,
                    len(preds),
                    len(rets),
                )
                continue

            # Clean data
            valid_mask = (
                ~np.isnan(preds)
                & ~np.isnan(rets)
                & ~np.isinf(preds)
                & ~np.isinf(rets)
            )
            preds_clean = preds[valid_mask]
            rets_clean = rets[valid_mask]

            if len(preds_clean) < 10:
                logger.warning(
                    "Fewer than 10 valid samples for horizon '%s'", horizon_label
                )
                continue

            ic, _ = pearsonr(preds_clean, rets_clean)
            rankic, _ = spearmanr(preds_clean, rets_clean)

            results.append(
                {
                    "horizon": horizon_label,
                    "IC": ic if not np.isnan(ic) else 0.0,
                    "RankIC": rankic if not np.isnan(rankic) else 0.0,
                    "n_samples": len(preds_clean),
                }
            )

        self.decay_results = pd.DataFrame(results)

        if len(self.decay_results) > 0:
            logger.info(
                "IC decay complete: best horizon=%s (IC=%.6f)",
                self.decay_results.loc[
                    self.decay_results["RankIC"].idxmax(), "horizon"
                ],
                self.decay_results["RankIC"].max(),
            )

        return self.decay_results

    def plot_ic_series(
        self,
        ic_df: Optional[pd.DataFrame] = None,
        save_path: Optional[str] = None,
        window: int = 20,
    ) -> bool:
        """Plot IC time series with rolling mean.

        Creates a chart showing daily IC and RankIC values with their
        rolling averages for trend visualization.

        Args:
            ic_df: IC DataFrame (uses stored ic_df if None).
            save_path: Path to save the plot image.
            window: Rolling window size for computing rolling mean.

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

        df = ic_df if ic_df is not None else self.ic_df

        if df is None or len(df) == 0:
            logger.error("No IC data available for plotting")
            return False

        try:
            fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

            # IC plot
            ax = axes[0]
            ax.plot(df["date"], df["IC"], alpha=0.3, color="steelblue", label="Daily IC")
            if len(df) >= window:
                rolling_ic = df["IC"].rolling(window=window).mean()
                ax.plot(
                    df["date"],
                    rolling_ic,
                    color="darkblue",
                    linewidth=2,
                    label=f"IC Rolling Mean ({window}d)",
                )
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax.axhline(
                y=df["IC"].mean(),
                color="red",
                linestyle="--",
                alpha=0.7,
                label=f"Mean IC: {df['IC'].mean():.4f}",
            )
            ax.set_ylabel("IC (Pearson)")
            ax.set_title("Information Coefficient (IC) - Time Series")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)

            # RankIC plot
            ax = axes[1]
            ax.plot(
                df["date"], df["RankIC"], alpha=0.3, color="seagreen", label="Daily RankIC"
            )
            if len(df) >= window:
                rolling_rankic = df["RankIC"].rolling(window=window).mean()
                ax.plot(
                    df["date"],
                    rolling_rankic,
                    color="darkgreen",
                    linewidth=2,
                    label=f"RankIC Rolling Mean ({window}d)",
                )
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax.axhline(
                y=df["RankIC"].mean(),
                color="red",
                linestyle="--",
                alpha=0.7,
                label=f"Mean RankIC: {df['RankIC'].mean():.4f}",
            )
            ax.set_ylabel("RankIC (Spearman)")
            ax.set_xlabel("Date")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                logger.info("IC series plot saved to %s", save_path)

            plt.close()
            return True
        except Exception as e:
            logger.error("Failed to generate IC series plot: %s", e)
            plt.close("all")
            return False

    def get_ic_stats_by_period(
        self, ic_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Compute IC statistics grouped by month or year.

        Args:
            ic_df: IC DataFrame (uses stored ic_df if None).

        Returns:
            DataFrame with monthly IC statistics.

        Raises:
            ValueError: If no IC data is available.
        """
        df = ic_df if ic_df is not None else self.ic_df

        if df is None or len(df) == 0:
            raise ValueError("No IC data available.")

        if "date" not in df.columns:
            logger.warning("No date column in IC DataFrame")
            return pd.DataFrame()

        df = df.copy()
        df["year_month"] = df["date"].dt.to_period("M")

        stats = df.groupby("year_month").agg(
            mean_ic=("IC", "mean"),
            std_ic=("IC", "std"),
            mean_rankic=("RankIC", "mean"),
            ic_win_rate=("IC", lambda x: (x > 0).mean()),
            n_days=("IC", "count"),
        ).reset_index()

        stats["year_month"] = stats["year_month"].astype(str)
        return stats