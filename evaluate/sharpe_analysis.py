"""
Risk-adjusted return analysis for the Stock-AI quantitative trading project.

Computes Sharpe Ratio, Sortino Ratio, Calmar Ratio, and rolling Sharpe
for evaluating strategy risk-adjusted performance.
"""

import logging
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SharpeAnalyzer:
    """Risk-adjusted return analyzer for quantitative trading strategies.

    Computes standard performance ratios (Sharpe, Sortino, Calmar) that
    measure returns relative to different types of risk.

    Attributes:
        annual_risk_free: Annual risk-free rate used for ratio calculations.
        trading_days: Number of trading days per year for annualization.
    """

    def __init__(self, trading_days: int = 252):
        """Initialize the Sharpe analyzer.

        Args:
            trading_days: Number of trading days per year for annualization.
                Default is 252 (typical US trading calendar).
        """
        self.trading_days = trading_days
        logger.info(
            "SharpeAnalyzer initialized: trading_days=%d", trading_days
        )

    def compute_sharpe(
        self,
        returns: Union[np.ndarray, pd.Series],
        risk_free: float = 0.03,
        annualize: bool = True,
    ) -> float:
        """Compute the annualized Sharpe ratio.

        Sharpe Ratio = (mean_return - risk_free_rate) / std(returns)

        Args:
            returns: Daily return series.
            risk_free: Annual risk-free rate (decimal, e.g., 0.03 for 3%).
            annualize: If True, returns annualized Sharpe ratio.
                If False, returns daily Sharpe ratio.

        Returns:
            Sharpe ratio as a float.

        Raises:
            ValueError: If returns data is invalid or has zero variance.
        """
        logger.info("Computing Sharpe ratio from %d returns", len(returns))

        returns = np.asarray(returns, dtype=np.float64).ravel()

        # Remove NaN/inf
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]

        if len(returns) < 2:
            raise ValueError("Need at least 2 return data points")

        if np.std(returns) == 0:
            logger.warning("Zero volatility in returns, Sharpe ratio is undefined")
            return 0.0

        if annualize:
            # Convert annual risk-free to daily
            daily_rf = (1 + risk_free) ** (1 / self.trading_days) - 1
            excess_returns = returns - daily_rf
            sharpe = np.sqrt(self.trading_days) * (
                np.mean(excess_returns) / np.std(returns)
            )
        else:
            daily_rf = risk_free / self.trading_days
            excess_returns = returns - daily_rf
            sharpe = np.mean(excess_returns) / np.std(returns)

        logger.info("Sharpe ratio computed: %.4f", sharpe)
        return float(sharpe)

    def compute_sortino(
        self,
        returns: Union[np.ndarray, pd.Series],
        risk_free: float = 0.03,
        annualize: bool = True,
        target_return: float = 0.0,
    ) -> float:
        """Compute the Sortino ratio using downside deviation.

        Sortino Ratio = (mean_return - risk_free_rate - target_return) /
                         downside_deviation

        Unlike Sharpe ratio, Sortino only penalizes downside volatility
        (returns below the target), which is more relevant for investors.

        Args:
            returns: Daily return series.
            risk_free: Annual risk-free rate (decimal).
            annualize: If True, returns annualized Sortino ratio.
            target_return: Minimum acceptable return (MAR) per period.
                Default 0.0 means any negative return is considered downside.

        Returns:
            Sortino ratio as a float.

        Raises:
            ValueError: If returns data is invalid.
        """
        logger.info("Computing Sortino ratio from %d returns", len(returns))

        returns = np.asarray(returns, dtype=np.float64).ravel()
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]

        if len(returns) < 2:
            raise ValueError("Need at least 2 return data points")

        if annualize:
            daily_rf = (1 + risk_free) ** (1 / self.trading_days) - 1
        else:
            daily_rf = risk_free / self.trading_days

        excess_returns = returns - daily_rf - target_return
        mean_excess = np.mean(excess_returns)

        # Downside deviation - only consider returns below target
        downside_returns = returns[returns < target_return]
        if len(downside_returns) < 2:
            logger.warning(
                "Fewer than 2 downside returns, Sortino ratio is undefined"
            )
            return 0.0

        downside_deviation = np.std(downside_returns)

        if downside_deviation == 0:
            logger.warning(
                "Zero downside deviation, Sortino ratio is undefined"
            )
            return 0.0

        if annualize:
            sortino = np.sqrt(self.trading_days) * (
                mean_excess / downside_deviation
            )
        else:
            sortino = mean_excess / downside_deviation

        logger.info("Sortino ratio computed: %.4f", sortino)
        return float(sortino)

    def compute_calmar(
        self,
        returns: Union[np.ndarray, pd.Series],
        max_drawdown: float,
    ) -> float:
        """Compute the Calmar ratio.

        Calmar Ratio = annualized_return / |max_drawdown|

        Measures return relative to the maximum drawdown experienced,
        with higher values indicating better risk-adjusted performance.

        Args:
            returns: Daily return series.
            max_drawdown: Maximum drawdown as a decimal (e.g., -0.25 for 25%).
                Should be a negative value.

        Returns:
            Calmar ratio as a float.

        Raises:
            ValueError: If returns data is invalid or max_drawdown is zero.
        """
        logger.info("Computing Calmar ratio")

        returns = np.asarray(returns, dtype=np.float64).ravel()
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]

        if len(returns) < 2:
            raise ValueError("Need at least 2 return data points")

        # Annualize returns
        total_return = np.prod(1 + returns)
        n_years = len(returns) / self.trading_days
        annualized_return = total_return ** (1 / n_years) - 1 if n_years > 0 else 0

        if max_drawdown == 0:
            logger.warning("Zero max drawdown, Calmar ratio is undefined")
            return 0.0

        calmar = annualized_return / abs(max_drawdown)

        logger.info(
            "Calmar ratio computed: %.4f (annual_return=%.4f%%, max_dd=%.4f%%)",
            calmar,
            annualized_return * 100,
            max_drawdown * 100,
        )
        return float(calmar)

    def compute_all_ratios(
        self,
        returns: Union[np.ndarray, pd.Series],
        max_dd: Optional[float] = None,
        risk_free: float = 0.03,
    ) -> Dict[str, float]:
        """Compute all performance ratios at once.

        Args:
            returns: Daily return series.
            max_dd: Maximum drawdown for Calmar ratio.
                If None, Calmar ratio will be None.
            risk_free: Annual risk-free rate (decimal).

        Returns:
            Dictionary with keys:
            - sharpe_ratio: Annualized Sharpe ratio
            - sortino_ratio: Annualized Sortino ratio
            - calmar_ratio: Calmar ratio (None if max_dd not provided)
            - annual_return: Annualized return
            - annual_volatility: Annualized volatility
            - n_days: Number of trading days

        Raises:
            ValueError: If returns data is invalid.
        """
        logger.info("Computing all performance ratios")

        returns = np.asarray(returns, dtype=np.float64).ravel()
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]

        if len(returns) < 2:
            raise ValueError("Need at least 2 return data points")

        # Compute annualized return and volatility
        n_days = len(returns)
        n_years = n_days / self.trading_days
        total_return = np.prod(1 + returns)
        annual_return = (
            total_return ** (1 / n_years) - 1 if n_years > 0 else 0.0
        )
        annual_volatility = np.std(returns) * np.sqrt(self.trading_days)

        results = {
            "sharpe_ratio": self.compute_sharpe(returns, risk_free),
            "sortino_ratio": self.compute_sortino(returns, risk_free),
            "annual_return": float(annual_return),
            "annual_volatility": float(annual_volatility),
            "n_days": n_days,
        }

        if max_dd is not None:
            results["calmar_ratio"] = self.compute_calmar(returns, max_dd)
        else:
            results["calmar_ratio"] = None

        logger.info(
            "All ratios computed: Sharpe=%.4f, Sortino=%.4f, Return=%.4f%%",
            results["sharpe_ratio"],
            results["sortino_ratio"],
            annual_return * 100,
        )

        return results

    def compute_rolling_sharpe(
        self,
        returns: Union[np.ndarray, pd.Series],
        window: int = 60,
        risk_free: float = 0.03,
        dates: Optional[Union[pd.Series, pd.DatetimeIndex]] = None,
    ) -> pd.DataFrame:
        """Compute rolling window Sharpe ratio.

        Args:
            returns: Daily return series.
            window: Rolling window size in days.
            risk_free: Annual risk-free rate (decimal).
            dates: Optional date index for the output DataFrame.

        Returns:
            DataFrame with columns:
            - date (if provided): Date of rolling window end
            - sharpe: Rolling Sharpe ratio
            - rolling_return: Mean return over the window
            - rolling_volatility: Standard deviation over the window

        Raises:
            ValueError: If returns data is invalid or window is too small.
        """
        logger.info(
            "Computing rolling Sharpe: window=%d, n_returns=%d",
            window,
            len(returns),
        )

        returns = np.asarray(returns, dtype=np.float64).ravel()

        if len(returns) < window:
            raise ValueError(
                f"Need at least {window} data points for rolling Sharpe, "
                f"got {len(returns)}"
            )

        if dates is not None:
            dates = pd.to_datetime(np.asarray(dates))

        daily_rf = (1 + risk_free) ** (1 / self.trading_days) - 1

        # Compute rolling statistics
        series = pd.Series(returns)
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()

        rolling_sharpe = np.sqrt(self.trading_days) * (
            (rolling_mean - daily_rf) / rolling_std
        )

        result = pd.DataFrame(
            {
                "sharpe": rolling_sharpe.values,
                "rolling_return": rolling_mean.values,
                "rolling_volatility": rolling_std.values,
            }
        )

        if dates is not None:
            result.insert(0, "date", dates.values)

        result = result.dropna().reset_index(drop=True)

        logger.info(
            "Rolling Sharpe computed: mean=%.4f, std=%.4f",
            result["sharpe"].mean(),
            result["sharpe"].std(),
        )

        return result

    def compute_information_ratio(
        self,
        strategy_returns: Union[np.ndarray, pd.Series],
        benchmark_returns: Union[np.ndarray, pd.Series],
        annualize: bool = True,
    ) -> float:
        """Compute the Information Ratio relative to a benchmark.

        Information Ratio = mean(tracking_error) / std(tracking_error)
        where tracking_error = strategy_returns - benchmark_returns

        Measures excess return per unit of tracking error risk.

        Args:
            strategy_returns: Strategy daily return series.
            benchmark_returns: Benchmark daily return series.
            annualize: If True, returns annualized Information Ratio.

        Returns:
            Information Ratio as a float.

        Raises:
            ValueError: If data is invalid or zero tracking error volatility.
        """
        strategy = np.asarray(strategy_returns, dtype=np.float64).ravel()
        benchmark = np.asarray(benchmark_returns, dtype=np.float64).ravel()

        if len(strategy) != len(benchmark):
            raise ValueError(
                f"Length mismatch: strategy({len(strategy)}) vs "
                f"benchmark({len(benchmark)})"
            )

        # Remove NaN/inf
        valid = (
            ~np.isnan(strategy)
            & ~np.isnan(benchmark)
            & ~np.isinf(strategy)
            & ~np.isinf(benchmark)
        )
        strategy = strategy[valid]
        benchmark = benchmark[valid]

        if len(strategy) < 2:
            raise ValueError("Need at least 2 valid data points")

        tracking_error = strategy - benchmark
        mean_te = np.mean(tracking_error)
        std_te = np.std(tracking_error)

        if std_te == 0:
            logger.warning(
                "Zero tracking error volatility, Information Ratio is undefined"
            )
            return 0.0

        if annualize:
            ir = np.sqrt(self.trading_days) * (mean_te / std_te)
        else:
            ir = mean_te / std_te

        logger.info("Information Ratio computed: %.4f", ir)
        return float(ir)