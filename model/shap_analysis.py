"""
SHAP-based feature importance analysis for the Stock-AI quantitative trading project.

Provides SHAP value computation, visualization, and feature ranking for
interpreting model predictions.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Check SHAP availability
_SHAP_AVAILABLE = False
try:
    import shap

    _SHAP_AVAILABLE = True
except ImportError:
    logger.warning(
        "SHAP is not installed. SHAP-based analysis will not be available. "
        "Install with: pip install shap"
    )


class SHAPAnalyzer:
    """SHAP (SHapley Additive exPlanations) feature importance analyzer.

    Computes SHAP values for trained models to explain feature contributions
    to predictions. Provides visualization and feature ranking utilities.

    Attributes:
        shap_values: Computed SHAP values array.
        feature_names: List of feature names.
        expected_value: Base value (expected model output).
    """

    def __init__(self):
        """Initialize SHAP analyzer."""
        if not _SHAP_AVAILABLE:
            logger.warning(
                "SHAP is not available. Analysis methods will return empty results."
            )

        self.shap_values = None
        self.feature_names = None
        self.expected_value = None

    def analyze(
        self,
        model: Any,
        X_sample: Union[np.ndarray, pd.DataFrame],
        max_samples: int = 1000,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Compute SHAP values and feature importance ranking.

        For tree-based models (LightGBM, XGBoost, CatBoost), uses the
        efficient TreeExplainer. For other models, falls back to
        KernelExplainer with a subset of samples.

        Args:
            model: Trained model object (must have predict method).
            X_sample: Sample of features for SHAP computation.
            max_samples: Maximum number of samples to use for SHAP
                computation (to manage computation time).

        Returns:
            Tuple of (shap_values, importance_df) where:
            - shap_values: SHAP values array of shape (n_samples, n_features)
            - importance_df: DataFrame with columns 'feature' and
              'shap_importance' sorted by mean absolute SHAP value.

        Raises:
            RuntimeError: If SHAP is not installed.
            ValueError: If input data is invalid.
        """
        if not _SHAP_AVAILABLE:
            raise RuntimeError(
                "SHAP is not installed. Install with: pip install shap"
            )

        logger.info(
            "Starting SHAP analysis: X_sample shape=%s, max_samples=%d",
            X_sample.shape,
            max_samples,
        )

        # Convert to numpy if needed
        if isinstance(X_sample, pd.DataFrame):
            self.feature_names = list(X_sample.columns)
            X = X_sample.values
        else:
            X = np.asarray(X_sample, dtype=np.float64)
            self.feature_names = [
                f"feature_{i}" for i in range(X.shape[1])
            ]

        # Subsample if needed
        if len(X) > max_samples:
            rng = np.random.RandomState(42)
            indices = rng.choice(len(X), max_samples, replace=False)
            X = X[indices]
            logger.info("Subsampled to %d samples for SHAP computation", max_samples)

        # Handle NaN/inf
        X = np.where(np.isinf(X), np.nan, X)
        col_means = np.nanmean(X, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        X = np.where(np.isnan(X), col_means, X)

        # Try TreeExplainer first (for tree-based models)
        try:
            explainer = shap.TreeExplainer(model)
            self.shap_values = explainer.shap_values(X)
            self.expected_value = explainer.expected_value
            logger.info("Using TreeExplainer for SHAP computation")
        except Exception as e:
            logger.warning(
                "TreeExplainer failed: %s. Falling back to KernelExplainer", e
            )
            try:
                # Use a small background dataset
                background = X[: min(100, len(X))]
                explainer = shap.KernelExplainer(
                    model.predict, background
                )
                self.shap_values = explainer.shap_values(
                    X[: min(200, len(X))]
                )
                self.expected_value = explainer.expected_value
                logger.info("Using KernelExplainer for SHAP computation")
            except Exception as e2:
                logger.error("KernelExplainer also failed: %s", e2)
                raise RuntimeError(
                    f"SHAP analysis failed for this model: {e2}"
                ) from e2

        # Compute feature importance ranking
        if isinstance(self.shap_values, list):
            # Multi-output case - use the first output
            sv = self.shap_values[0] if self.shap_values else np.array([])
        else:
            sv = self.shap_values

        if sv is None or len(sv) == 0:
            logger.warning("SHAP values are empty")
            return np.array([]), pd.DataFrame(
                columns=["feature", "shap_importance"]
            )

        mean_abs_shap = np.abs(sv).mean(axis=0)

        # Create importance DataFrame
        importance_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "shap_importance": mean_abs_shap,
            }
        )
        importance_df = importance_df.sort_values(
            "shap_importance", ascending=False
        ).reset_index(drop=True)

        logger.info(
            "SHAP analysis complete: top feature=%s, importance=%f",
            importance_df.iloc[0]["feature"]
            if len(importance_df) > 0
            else "N/A",
            importance_df.iloc[0]["shap_importance"]
            if len(importance_df) > 0
            else 0.0,
        )

        return sv, importance_df

    def plot_summary(
        self,
        shap_values: Optional[np.ndarray] = None,
        X_sample: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        save_path: Optional[str] = None,
        plot_type: str = "dot",
    ) -> bool:
        """Generate and save a SHAP summary plot.

        Creates a beeswarm (dot) summary plot showing feature importance
        and impact direction.

        Args:
            shap_values: SHAP values (uses stored values if None).
            X_sample: Feature data for plot (uses stored data if None).
            save_path: Path to save the plot image.
            plot_type: Type of summary plot - 'dot' (beeswarm), 'bar',
                or 'violin'.

        Returns:
            True if plot was successfully saved, False otherwise.
        """
        if not _SHAP_AVAILABLE:
            logger.warning("SHAP is not installed. Cannot generate plots.")
            return False

        sv = shap_values if shap_values is not None else self.shap_values
        if sv is None:
            logger.error("No SHAP values available for plotting")
            return False

        if X_sample is None and self.feature_names is None:
            logger.error("No feature data available for plotting")
            return False

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # Prepare feature data
            if X_sample is not None:
                if isinstance(X_sample, pd.DataFrame):
                    X_display = X_sample.values[: len(sv)]
                    feature_names = list(X_sample.columns)
                else:
                    X_display = np.asarray(X_sample)[: len(sv)]
                    feature_names = self.feature_names
            else:
                X_display = np.zeros((len(sv), len(self.feature_names)))
                feature_names = self.feature_names

            plt.figure(figsize=(12, 8))

            if plot_type == "bar":
                shap.summary_plot(
                    sv,
                    X_display,
                    feature_names=feature_names,
                    plot_type="bar",
                    show=False,
                    max_display=20,
                )
            elif plot_type == "violin":
                shap.summary_plot(
                    sv,
                    X_display,
                    feature_names=feature_names,
                    plot_type="violin",
                    show=False,
                    max_display=20,
                )
            else:
                shap.summary_plot(
                    sv,
                    X_display,
                    feature_names=feature_names,
                    show=False,
                    max_display=20,
                )

            if save_path:
                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close()
                logger.info("SHAP summary plot saved to %s", save_path)

            plt.close()
            return True
        except Exception as e:
            logger.error("Failed to generate SHAP summary plot: %s", e)
            return False

    def get_top_features(
        self,
        shap_values: Optional[np.ndarray] = None,
        X_sample: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        n: int = 20,
    ) -> pd.DataFrame:
        """Get top N features ranked by SHAP importance.

        Args:
            shap_values: SHAP values (uses stored values if None).
            X_sample: Feature data for feature names.
            n: Number of top features to return.

        Returns:
            DataFrame with columns 'rank', 'feature', 'shap_importance',
            and 'impact_direction' (+1 or -1 indicating average impact
            direction), sorted by importance.

        Raises:
            ValueError: If no SHAP values are available.
        """
        sv = shap_values if shap_values is not None else self.shap_values
        if sv is None:
            raise ValueError(
                "No SHAP values available. Run analyze() first or provide shap_values."
            )

        if isinstance(sv, list):
            sv = sv[0] if sv else np.array([])

        mean_abs_shap = np.abs(sv).mean(axis=0)
        mean_shap = sv.mean(axis=0)  # Direction of impact

        # Get feature names
        if X_sample is not None and isinstance(X_sample, pd.DataFrame):
            feature_names = list(X_sample.columns)
        else:
            feature_names = self.feature_names or [
                f"feature_{i}" for i in range(len(mean_abs_shap))
            ]

        df = pd.DataFrame(
            {
                "feature": feature_names,
                "shap_importance": mean_abs_shap,
                "impact_direction": np.sign(mean_shap),
            }
        )

        df = df.sort_values("shap_importance", ascending=False).reset_index(
            drop=True
        )
        df = df.head(n)
        df.insert(0, "rank", range(1, len(df) + 1))

        logger.info(
            "Top %d features extracted: #1=%s (importance=%f)",
            n,
            df.iloc[0]["feature"] if len(df) > 0 else "N/A",
            df.iloc[0]["shap_importance"] if len(df) > 0 else 0.0,
        )

        return df

    def plot_dependence(
        self,
        feature: str,
        shap_values: Optional[np.ndarray] = None,
        X_sample: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        interaction_feature: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> bool:
        """Generate and save a SHAP dependence plot for a specific feature.

        Args:
            feature: Feature name to plot.
            shap_values: SHAP values (uses stored values if None).
            X_sample: Feature data for plotting.
            interaction_feature: Optional feature for color interaction.
            save_path: Path to save the plot image.

        Returns:
            True if plot was successfully saved, False otherwise.
        """
        if not _SHAP_AVAILABLE:
            logger.warning("SHAP is not installed.")
            return False

        sv = shap_values if shap_values is not None else self.shap_values
        if sv is None or X_sample is None:
            logger.error("SHAP values and X_sample are required")
            return False

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            if isinstance(X_sample, pd.DataFrame):
                X_display = X_sample.values[: len(sv)]
                feature_names = list(X_sample.columns)
            else:
                X_display = np.asarray(X_sample)[: len(sv)]
                feature_names = self.feature_names

            feature_idx = None
            if feature in feature_names:
                feature_idx = feature_names.index(feature)

            if feature_idx is None:
                logger.error("Feature '%s' not found", feature)
                return False

            interaction_idx = None
            if interaction_feature and interaction_feature in feature_names:
                interaction_idx = feature_names.index(interaction_feature)

            plt.figure(figsize=(10, 6))
            shap.dependence_plot(
                feature_idx,
                sv,
                X_display,
                feature_names=feature_names,
                interaction_index=interaction_idx,
                show=False,
            )

            if save_path:
                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close()
                logger.info(
                    "SHAP dependence plot for '%s' saved to %s",
                    feature,
                    save_path,
                )

            plt.close()
            return True
        except Exception as e:
            logger.error(
                "Failed to generate SHAP dependence plot for '%s': %s",
                feature,
                e,
            )
            return False

    def get_feature_interactions(
        self,
        shap_values: Optional[np.ndarray] = None,
        X_sample: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    ) -> Optional[pd.DataFrame]:
        """Compute SHAP interaction values and feature interaction strengths.

        Args:
            shap_values: SHAP values (uses stored values if None).
            X_sample: Feature data.

        Returns:
            DataFrame with interaction strengths between feature pairs,
            or None if SHAP interaction values are not available.
        """
        if not _SHAP_AVAILABLE:
            logger.warning("SHAP is not installed.")
            return None

        sv = shap_values if shap_values is not None else self.shap_values
        if sv is None:
            logger.error("No SHAP values available")
            return None

        try:
            # Get SHAP interaction values if available
            if hasattr(sv, "shape") and len(sv.shape) == 3:
                # SHAP interaction values have shape (n_samples, n_features, n_features)
                interaction_values = np.abs(sv).mean(axis=0)
                np.fill_diagonal(interaction_values, 0)

                feature_names = self.feature_names or [
                    f"feature_{i}" for i in range(interaction_values.shape[0])
                ]

                # Find top interactions
                interactions = []
                n_features = len(feature_names)
                for i in range(n_features):
                    for j in range(i + 1, n_features):
                        interactions.append(
                            {
                                "feature_1": feature_names[i],
                                "feature_2": feature_names[j],
                                "interaction_strength": interaction_values[i, j],
                            }
                        )

                df = pd.DataFrame(interactions)
                df = df.sort_values(
                    "interaction_strength", ascending=False
                ).reset_index(drop=True)
                return df
            else:
                logger.info(
                    "SHAP interaction values not available for this model"
                )
                return None
        except Exception as e:
            logger.error("Failed to compute feature interactions: %s", e)
            return None