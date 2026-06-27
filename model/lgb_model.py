"""
LightGBM model implementation for the Stock-AI quantitative trading project.

Wraps lightgbm.LGBMRegressor with configuration loading, early stopping,
and model persistence support.
"""

import logging
import os
import pickle
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

from .base_model import BaseModel

logger = logging.getLogger(__name__)

# Path to model configuration file
_CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
_DEFAULT_CONFIG_PATH = os.path.join(_CONFIG_DIR, "model_config.yaml")


class LightGBMModel(BaseModel):
    """LightGBM gradient boosting model wrapper.

    Wraps lightgbm.LGBMRegressor with support for:
    - Configuration loading from YAML config file
    - Early stopping via validation set
    - NaN/inf value handling in input data
    - Model persistence (save/load)
    - Feature importance extraction

    Attributes:
        model: The underlying lightgbm.LGBMRegressor instance.
        params: Model parameters dictionary.
        best_iteration_: Best iteration count after early stopping.
        best_score_: Best validation score after early stopping.
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
    ):
        """Initialize LightGBM model.

        Args:
            params: Model parameters. Overrides config file defaults.
            config_path: Path to model_config.yaml. Uses default if not provided.
        """
        super().__init__()
        self.params = self._load_config(config_path)
        if params:
            self.params.update(params)
        self.best_iteration_ = None
        self.best_score_ = None

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load LightGBM parameters from YAML configuration file.

        Args:
            config_path: Path to configuration file.

        Returns:
            Dictionary of LightGBM parameters.
        """
        path = config_path or _DEFAULT_CONFIG_PATH
        try:
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            params = config.get("lightgbm", {})
            logger.info("Loaded LightGBM config from %s: %s", path, params)
            return dict(params)
        except FileNotFoundError:
            logger.warning(
                "Config file not found at %s, using default parameters", path
            )
            return self._get_default_params()
        except Exception as e:
            logger.error("Error loading config: %s, using defaults", e)
            return self._get_default_params()

    @staticmethod
    def _get_default_params() -> Dict[str, Any]:
        """Get hardcoded default parameters when config file is unavailable.

        Returns:
            Default parameter dictionary.
        """
        return {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 128,
            "max_depth": 10,
            "learning_rate": 0.05,
            "n_estimators": 1000,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "min_child_samples": 20,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        }

    def _extract_early_stopping(self) -> int:
        """Extract early_stopping_rounds from params.

        Returns:
            Number of early stopping rounds.
        """
        return self.params.pop("early_stopping_rounds", 50)

    def _extract_eval_metric(self) -> str:
        """Extract evaluation metric from params.

        Returns:
            Evaluation metric string.
        """
        return self.params.get("metric", "rmse")

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        X_valid: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_valid: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> "LightGBMModel":
        """Fit LightGBM model on training data with optional early stopping.

        Args:
            X: Training features.
            y: Training targets.
            X_valid: Validation features for early stopping.
            y_valid: Validation targets for early stopping.

        Returns:
            self: The fitted model instance.

        Raises:
            ImportError: If lightgbm is not installed.
            ValueError: If input data is invalid.
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError(
                "lightgbm is required. Install with: pip install lightgbm"
            )

        logger.info(
            "Fitting LightGBM model: X shape=%s, y shape=%s", X.shape, y.shape
        )

        # Store feature names
        self._set_feature_names(X)

        # Validate and clean input data
        X_clean, y_clean = self._validate_input(X, y)

        # Extract early stopping rounds before passing to LGBMRegressor
        early_stopping_rounds = self._extract_early_stopping()
        eval_metric = self._extract_eval_metric()

        # Create model
        self.model = lgb.LGBMRegressor(**self.params)

        # Prepare fit kwargs
        fit_kwargs = {
            "eval_set": None,
            "eval_metric": eval_metric,
        }

        if X_valid is not None and y_valid is not None:
            X_valid_clean, y_valid_clean = self._validate_input(X_valid, y_valid)
            fit_kwargs["eval_set"] = [(X_valid_clean, y_valid_clean)]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(early_stopping_rounds),
                lgb.log_evaluation(period=100),
            ]
            logger.info(
                "Early stopping enabled: rounds=%d, eval_metric=%s",
                early_stopping_rounds,
                eval_metric,
            )
        else:
            logger.info("No validation set provided, training without early stopping")

        # Fit the model
        self.model.fit(X_clean, y_clean, **fit_kwargs)

        # Store best iteration info
        if hasattr(self.model, "best_iteration_"):
            self.best_iteration_ = self.model.best_iteration_
            logger.info("Best iteration: %d", self.best_iteration_)

        if hasattr(self.model, "best_score_"):
            self.best_score_ = self.model.best_score_
            if isinstance(self.best_score_, dict):
                logger.info("Best validation score: %s", str(self.best_score_))
            else:
                logger.info("Best validation score: %f", self.best_score_)

        self.is_fitted = True
        logger.info("LightGBM model fitting complete")
        return self

    def predict(
        self, X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """Generate predictions using the fitted model.

        Args:
            X: Input features.

        Returns:
            Predictions array of shape (n_samples,).

        Raises:
            RuntimeError: If model is not fitted.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before prediction")

        X_clean, _ = self._validate_input(X)
        predictions = self.model.predict(X_clean)

        logger.debug("Generated predictions: shape=%s, mean=%f", predictions.shape, predictions.mean())
        return predictions

    def save(self, path: str) -> None:
        """Save the trained model to disk using pickle.

        Args:
            path: File path to save the model.

        Raises:
            RuntimeError: If model is not fitted.
            IOError: If file cannot be written.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before saving")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        try:
            with open(path, "wb") as f:
                pickle.dump(
                    {
                        "model": self.model,
                        "params": self.params,
                        "best_iteration_": self.best_iteration_,
                        "best_score_": self.best_score_,
                        "feature_names_": self.feature_names_,
                    },
                    f,
                )
            logger.info("LightGBM model saved to %s", path)
        except Exception as e:
            logger.error("Failed to save model to %s: %s", path, e)
            raise IOError(f"Failed to save model: {e}") from e

    def load(self, path: str) -> "LightGBMModel":
        """Load a trained model from disk.

        Args:
            path: File path to load the model from.

        Returns:
            self: The loaded model instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            IOError: If file cannot be read.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            self.model = data["model"]
            self.params = data.get("params", {})
            self.best_iteration_ = data.get("best_iteration_")
            self.best_score_ = data.get("best_score_")
            self.feature_names_ = data.get("feature_names_")
            self.is_fitted = True

            logger.info("LightGBM model loaded from %s", path)
            return self
        except Exception as e:
            logger.error("Failed to load model from %s: %s", path, e)
            raise IOError(f"Failed to load model: {e}") from e

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores from the trained model.

        Returns:
            DataFrame with columns 'feature' and 'importance', sorted
            by importance in descending order.

        Raises:
            RuntimeError: If model is not fitted.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before getting feature importance")

        try:
            importances = self.model.feature_importances_
        except AttributeError:
            logger.warning("Model does not have feature_importances_ attribute")
            return pd.DataFrame(columns=["feature", "importance"])

        if self.feature_names_ is None:
            self.feature_names_ = [f"feature_{i}" for i in range(len(importances))]

        df = pd.DataFrame(
            {"feature": self.feature_names_, "importance": importances}
        )
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)

        logger.info(
            "Feature importance extracted: top feature=%s, score=%f",
            df.iloc[0]["feature"] if len(df) > 0 else "N/A",
            df.iloc[0]["importance"] if len(df) > 0 else 0.0,
        )

        return df

    def get_params(self) -> Dict[str, Any]:
        """Get current model parameters.

        Returns:
            Dictionary of model parameters.
        """
        return dict(self.params) if self.params else {}