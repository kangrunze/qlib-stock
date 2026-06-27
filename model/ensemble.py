"""
Multi-model ensemble for the Stock-AI quantitative trading project.

Combines LightGBM, XGBoost, and CatBoost models with weighted averaging
to produce robust stock return predictions.
"""

import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

from .base_model import BaseModel
from .lgb_model import LightGBMModel
from .xgb_model import XGBoostModel
from .cat_model import CatBoostModel

logger = logging.getLogger(__name__)

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
_DEFAULT_CONFIG_PATH = os.path.join(_CONFIG_DIR, "model_config.yaml")


class EnsembleModel(BaseModel):
    """Multi-model ensemble combining LightGBM, XGBoost, and CatBoost.

    Trains all three gradient boosting models independently and combines
    their predictions using configurable weighted averaging.

    Attributes:
        lgb_model: LightGBM model instance.
        xgb_model: XGBoost model instance.
        cat_model: CatBoost model instance.
        weights: List of weights [lgb_weight, xgb_weight, cat_weight] for
            combining predictions.
        is_fitted: Whether all models have been trained.
    """

    def __init__(
        self,
        weights: Optional[List[float]] = None,
        lgb_params: Optional[Dict[str, Any]] = None,
        xgb_params: Optional[Dict[str, Any]] = None,
        cat_params: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
    ):
        """Initialize the ensemble model.

        Args:
            weights: List of 3 weights [lgb, xgb, cat] for weighted averaging.
                Defaults to config values or [0.5, 0.3, 0.2].
            lgb_params: Custom LightGBM parameters.
            xgb_params: Custom XGBoost parameters.
            cat_params: Custom CatBoost parameters.
            config_path: Path to model_config.yaml.
        """
        super().__init__()
        self.weights = weights or self._load_weights(config_path)
        self._validate_weights()

        self.lgb_model = LightGBMModel(params=lgb_params, config_path=config_path)
        self.xgb_model = XGBoostModel(params=xgb_params, config_path=config_path)
        self.cat_model = CatBoostModel(params=cat_params, config_path=config_path)

        self.individual_predictions_ = {}

    def _load_weights(self, config_path: Optional[str] = None) -> List[float]:
        """Load ensemble weights from configuration file.

        Args:
            config_path: Path to configuration file.

        Returns:
            List of 3 weight values.
        """
        path = config_path or _DEFAULT_CONFIG_PATH
        try:
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            weights = config.get("ensemble", {}).get("weights", [0.5, 0.3, 0.2])
            logger.info("Loaded ensemble weights from config: %s", weights)
            return list(weights)
        except (FileNotFoundError, Exception):
            logger.info("Using default ensemble weights: [0.5, 0.3, 0.2]")
            return [0.5, 0.3, 0.2]

    def _validate_weights(self) -> None:
        """Validate that weights are properly configured.

        Raises:
            ValueError: If weights are invalid.
        """
        if len(self.weights) != 3:
            raise ValueError(
                f"Expected 3 weights for [lgb, xgb, cat], got {len(self.weights)}"
            )

        if not all(w >= 0 for w in self.weights):
            raise ValueError(f"Weights must be non-negative, got {self.weights}")

        total = sum(self.weights)
        if total <= 0:
            raise ValueError(f"Weights must sum to a positive value, got sum={total}")

        # Normalize weights to sum to 1
        self.weights = [w / total for w in self.weights]
        logger.info("Normalized ensemble weights: %s", self.weights)

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        X_valid: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_valid: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> "EnsembleModel":
        """Train all three models in the ensemble.

        Args:
            X: Training features.
            y: Training targets.
            X_valid: Validation features for early stopping.
            y_valid: Validation targets for early stopping.

        Returns:
            self: The fitted ensemble model instance.

        Raises:
            ValueError: If input data is invalid.
            RuntimeError: If any model fails to train.
        """
        logger.info(
            "Fitting ensemble model with weights=%s: X shape=%s, y shape=%s",
            self.weights,
            X.shape,
            y.shape,
        )

        self._set_feature_names(X)

        # Train LightGBM
        logger.info("Training LightGBM model...")
        try:
            self.lgb_model.fit(X, y, X_valid=X_valid, y_valid=y_valid)
            logger.info("LightGBM model trained successfully")
        except Exception as e:
            logger.error("LightGBM training failed: %s", e)
            raise RuntimeError(f"LightGBM training failed: {e}") from e

        # Train XGBoost
        logger.info("Training XGBoost model...")
        try:
            self.xgb_model.fit(X, y, X_valid=X_valid, y_valid=y_valid)
            logger.info("XGBoost model trained successfully")
        except Exception as e:
            logger.error("XGBoost training failed: %s", e)
            raise RuntimeError(f"XGBoost training failed: {e}") from e

        # Train CatBoost
        logger.info("Training CatBoost model...")
        try:
            self.cat_model.fit(X, y, X_valid=X_valid, y_valid=y_valid)
            logger.info("CatBoost model trained successfully")
        except Exception as e:
            logger.error("CatBoost training failed: %s", e)
            raise RuntimeError(f"CatBoost training failed: {e}") from e

        self.is_fitted = True
        logger.info("Ensemble model fitting complete")
        return self

    def predict(
        self, X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """Generate weighted average predictions from all models.

        Args:
            X: Input features.

        Returns:
            Weighted average predictions array of shape (n_samples,).

        Raises:
            RuntimeError: If models are not fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble models must be fitted before prediction")

        logger.info(
            "Generating ensemble predictions: %d samples, weights=%s",
            len(X),
            self.weights,
        )

        # Get individual predictions
        preds = self.get_individual_predictions(X)

        # Compute weighted average
        ensemble_pred = np.zeros(len(X))
        for pred, weight in zip(
            [preds["lgb"], preds["xgb"], preds["cat"]], self.weights
        ):
            ensemble_pred += weight * np.asarray(pred, dtype=np.float64)

        logger.info(
            "Ensemble prediction complete: mean=%f, std=%f",
            ensemble_pred.mean(),
            ensemble_pred.std(),
        )

        return ensemble_pred

    def get_individual_predictions(
        self, X: Union[np.ndarray, pd.DataFrame]
    ) -> Dict[str, np.ndarray]:
        """Get predictions from each individual model.

        Args:
            X: Input features.

        Returns:
            Dictionary with keys 'lgb', 'xgb', 'cat' mapping to
            individual model prediction arrays.

        Raises:
            RuntimeError: If models are not fitted.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Ensemble models must be fitted before getting individual predictions"
            )

        predictions = {
            "lgb": self.lgb_model.predict(X),
            "xgb": self.xgb_model.predict(X),
            "cat": self.cat_model.predict(X),
        }
        self.individual_predictions_ = predictions
        return predictions

    def save(self, path: str) -> None:
        """Save the ensemble model configuration to disk.

        Note: Use save_models() to save individual models separately.

        Args:
            path: File path to save the ensemble config.

        Raises:
            IOError: If file cannot be written.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        try:
            with open(path, "wb") as f:
                pickle.dump(
                    {
                        "weights": self.weights,
                        "is_fitted": self.is_fitted,
                        "feature_names_": self.feature_names_,
                    },
                    f,
                )
            logger.info("Ensemble config saved to %s", path)
        except Exception as e:
            logger.error("Failed to save ensemble config: %s", e)
            raise IOError(f"Failed to save ensemble config: {e}") from e

    def load(self, path: str) -> "EnsembleModel":
        """Load the ensemble model configuration from disk.

        Note: Use load_models() to load individual models separately.

        Args:
            path: File path to load the ensemble config from.

        Returns:
            self: The loaded ensemble instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            IOError: If file cannot be read.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Ensemble config file not found: {path}")

        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            self.weights = data["weights"]
            self.is_fitted = data.get("is_fitted", False)
            self.feature_names_ = data.get("feature_names_")
            logger.info("Ensemble config loaded from %s", path)
            return self
        except Exception as e:
            logger.error("Failed to load ensemble config: %s", e)
            raise IOError(f"Failed to load ensemble config: {e}") from e

    def save_models(self, directory: str) -> None:
        """Save all individual models to a directory.

        Args:
            directory: Directory path to save models into.
                Creates lgb_model.pkl, xgb_model.pkl, cat_model.pkl,
                and ensemble_config.pkl.

        Raises:
            RuntimeError: If models are not fitted.
            IOError: If files cannot be written.
        """
        if not self.is_fitted:
            raise RuntimeError("Models must be fitted before saving")

        os.makedirs(directory, exist_ok=True)

        logger.info("Saving ensemble models to directory: %s", directory)

        # Save individual models
        self.lgb_model.save(os.path.join(directory, "lgb_model.pkl"))
        self.xgb_model.save(os.path.join(directory, "xgb_model.pkl"))
        self.cat_model.save(os.path.join(directory, "cat_model.pkl"))

        # Save ensemble config
        self.save(os.path.join(directory, "ensemble_config.pkl"))

        logger.info("All ensemble models saved successfully")

    def load_models(self, directory: str) -> "EnsembleModel":
        """Load all individual models from a directory.

        Args:
            directory: Directory path containing the saved model files.

        Returns:
            self: The loaded ensemble model instance.

        Raises:
            FileNotFoundError: If model files are not found.
            IOError: If files cannot be read.
        """
        logger.info("Loading ensemble models from directory: %s", directory)

        # Load individual models
        self.lgb_model.load(os.path.join(directory, "lgb_model.pkl"))
        self.xgb_model.load(os.path.join(directory, "xgb_model.pkl"))
        self.cat_model.load(os.path.join(directory, "cat_model.pkl"))

        # Load ensemble config
        self.load(os.path.join(directory, "ensemble_config.pkl"))

        self.is_fitted = True
        logger.info("All ensemble models loaded successfully")
        return self

    def get_feature_importance(self) -> pd.DataFrame:
        """Get aggregated feature importance from all models.

        Returns the weighted average of feature importances across
        LightGBM, XGBoost, and CatBoost.

        Returns:
            DataFrame with columns 'feature', 'importance', and
            individual model importance columns, sorted by combined
            importance in descending order.

        Raises:
            RuntimeError: If models are not fitted.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Models must be fitted before getting feature importance"
            )

        logger.info("Computing aggregated feature importance from ensemble")

        # Get individual importances
        lgb_imp = self.lgb_model.get_feature_importance()
        xgb_imp = self.xgb_model.get_feature_importance()
        cat_imp = self.cat_model.get_feature_importance()

        # Merge on feature name
        df = lgb_imp.copy()
        df.columns = ["feature", "lgb_importance"]

        if len(xgb_imp) > 0:
            xgb_imp.columns = ["feature", "xgb_importance"]
            df = df.merge(xgb_imp, on="feature", how="outer")

        if len(cat_imp) > 0:
            cat_imp.columns = ["feature", "cat_importance"]
            df = df.merge(cat_imp, on="feature", how="outer")

        # Fill NaN with 0
        for col in ["lgb_importance", "xgb_importance", "cat_importance"]:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # Normalize each column to [0, 1] for fair comparison
        for col in df.columns:
            if col != "feature":
                col_max = df[col].max()
                if col_max > 0:
                    df[col] = df[col] / col_max

        # Compute weighted importance
        w_lgb, w_xgb, w_cat = self.weights
        df["importance"] = (
            w_lgb * df.get("lgb_importance", 0)
            + w_xgb * df.get("xgb_importance", 0)
            + w_cat * df.get("cat_importance", 0)
        )

        df = df.sort_values("importance", ascending=False).reset_index(drop=True)

        logger.info(
            "Aggregated importance computed: top feature=%s, score=%f",
            df.iloc[0]["feature"] if len(df) > 0 else "N/A",
            df.iloc[0]["importance"] if len(df) > 0 else 0.0,
        )

        return df

    def set_weights(self, weights: List[float]) -> None:
        """Update ensemble weights.

        Args:
            weights: New list of 3 weights [lgb, xgb, cat].

        Raises:
            ValueError: If weights are invalid.
        """
        self.weights = list(weights)
        self._validate_weights()
        logger.info("Ensemble weights updated to: %s", self.weights)

    def get_weights(self) -> List[float]:
        """Get current ensemble weights.

        Returns:
            List of 3 normalized weights.
        """
        return list(self.weights)

    def get_metrics(self) -> Dict[str, Any]:
        """Get training metrics for the ensemble model.

        Returns:
            Dictionary with training metrics for each sub-model.
        """
        metrics = {
            "ensemble_weights": self.get_weights(),
            "n_features": len(self.feature_names_) if self.feature_names_ else 0,
        }

        # Collect best scores from sub-models if available
        for model_name, model in [("lgb", self.lgb_model),
                                   ("xgb", self.xgb_model),
                                   ("cat", self.cat_model)]:
            if hasattr(model, "best_score_"):
                metrics[f"{model_name}_best_score"] = model.best_score_
            if hasattr(model, "best_iteration_"):
                metrics[f"{model_name}_best_iter"] = model.best_iteration_

        return metrics