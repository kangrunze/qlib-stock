"""
Base model class for the Stock-AI quantitative trading project.

Defines the abstract interface that all model implementations must follow.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all trading models.

    All model implementations (LightGBM, XGBoost, CatBoost, ensembles) must
    inherit from this class and implement all abstract methods.

    Attributes:
        model: The underlying model object (set by subclasses).
        is_fitted: Whether the model has been trained.
    """

    def __init__(self):
        self.model = None
        self.is_fitted = False
        self.feature_names_ = None

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[list] = None,
    ) -> "BaseModel":
        """Fit the model on training data.

        Args:
            X: Training features, shape (n_samples, n_features).
            y: Training targets, shape (n_samples,).
            eval_set: Optional validation data for early stopping.
                List of (X_valid, y_valid) tuples.

        Returns:
            self: The fitted model instance.

        Raises:
            NotImplementedError: Subclass must implement this method.
        """
        raise NotImplementedError("Subclass must implement fit()")

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for input features.

        Args:
            X: Input features, shape (n_samples, n_features).

        Returns:
            Predictions array of shape (n_samples,).

        Raises:
            NotImplementedError: Subclass must implement this method.
        """
        raise NotImplementedError("Subclass must implement predict()")

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the trained model to disk.

        Args:
            path: File path to save the model to.

        Raises:
            NotImplementedError: Subclass must implement this method.
        """
        raise NotImplementedError("Subclass must implement save()")

    @abstractmethod
    def load(self, path: str) -> "BaseModel":
        """Load a trained model from disk.

        Args:
            path: File path to load the model from.

        Returns:
            self: The loaded model instance.

        Raises:
            NotImplementedError: Subclass must implement this method.
        """
        raise NotImplementedError("Subclass must implement load()")

    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores.

        Returns:
            DataFrame with columns 'feature' and 'importance', sorted
            by importance in descending order.

        Raises:
            NotImplementedError: Subclass must implement this method.
        """
        raise NotImplementedError("Subclass must implement get_feature_importance()")

    def _validate_input(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> tuple:
        """Validate input data and handle NaN/inf values.

        Args:
            X: Input feature array.
            y: Optional target array.

        Returns:
            Tuple of (X_clean, y_clean) with NaN/inf handled.

        Raises:
            ValueError: If input data is invalid.
        """
        logger.info("Validating input data: shape=%s", X.shape)

        if X is None or len(X) == 0:
            raise ValueError("Input X is empty or None")

        # Convert to float64 for consistency
        X = np.asarray(X, dtype=np.float64)

        # Replace inf with NaN
        X = np.where(np.isinf(X), np.nan, X)

        # Check NaN ratio
        nan_ratio = np.isnan(X).mean()
        if nan_ratio > 0.5:
            logger.warning("High NaN ratio in input: %.2f%%", nan_ratio * 100)

        # Fill NaN with column mean (or 0 if entire column is NaN)
        col_means = np.nanmean(X, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        nan_mask = np.isnan(X)
        X = np.where(nan_mask, col_means, X)

        logger.info(
            "Input validation complete: cleaned %d NaN positions",
            nan_mask.sum(),
        )

        if y is not None:
            y = np.asarray(y, dtype=np.float64)
            y = np.where(np.isinf(y), np.nan, y)
            y = np.where(np.isnan(y), np.nanmean(y), y)

        return X, y

    def _set_feature_names(self, X) -> None:
        """Extract and store feature names from input data.

        Args:
            X: Input data (numpy array or pandas DataFrame).
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
        elif hasattr(X, "shape"):
            n_features = X.shape[1]
            self.feature_names_ = [f"feature_{i}" for i in range(n_features)]