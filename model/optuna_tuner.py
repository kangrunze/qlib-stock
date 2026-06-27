"""
Optuna-based hyperparameter optimizer for the Stock-AI quantitative trading project.

Provides automated hyperparameter tuning for LightGBM, XGBoost, and CatBoost models
using RankIC (Spearman correlation) as the optimization target metric.
"""

import logging
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def _compute_rankic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute RankIC (Spearman rank correlation) between true and predicted values.

    Args:
        y_true: True target values.
        y_pred: Predicted values.

    Returns:
        RankIC value (float between -1 and 1).
    """
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 10:
        logger.warning("Fewer than 10 valid samples for RankIC computation")
        return 0.0

    ic, _ = spearmanr(y_true[mask], y_pred[mask])
    return ic if not np.isnan(ic) else 0.0


class OptunaTuner:
    """Optuna-based hyperparameter tuner for trading models.

    Uses RankIC (Spearman correlation) as the optimization target,
    which is the standard metric in quantitative finance for evaluating
    factor/model predictive power.

    Attributes:
        study_name: Name prefix for Optuna studies.
        storage: Optuna storage backend (None for in-memory).
        best_params: Dictionary of best parameters per model type.
    """

    def __init__(
        self,
        study_name: str = "stock_model_tuning",
        storage: Optional[str] = None,
    ):
        """Initialize the Optuna tuner.

        Args:
            study_name: Name prefix for Optuna study.
            storage: Optuna storage URL (e.g., 'sqlite:///optuna.db').
                None for in-memory storage.
        """
        self.study_name = study_name
        self.storage = storage
        self.best_params = {"lgb": {}, "xgb": {}, "cat": {}}

    def _create_study(self, model_type: str) -> "optuna.Study":
        """Create an Optuna study for the given model type.

        Args:
            model_type: One of 'lgb', 'xgb', 'cat'.

        Returns:
            Optuna Study object.
        """
        try:
            import optuna
        except ImportError:
            raise ImportError(
                "optuna is required for hyperparameter tuning. "
                "Install with: pip install optuna"
            )

        study = optuna.create_study(
            study_name=f"{self.study_name}_{model_type}",
            storage=self.storage,
            direction="maximize",
            load_if_exists=True,
        )
        return study

    def _get_time_series_splits(
        self, n_samples: int, n_splits: int = 5
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate time series cross-validation splits.

        For time series data, training data must come before validation data.

        Args:
            n_samples: Total number of samples.
            n_splits: Number of cross-validation folds.

        Returns:
            List of (train_idx, valid_idx) tuples.
        """
        splits = []
        indices = np.arange(n_samples)

        for i in range(n_splits):
            split_point = int(n_samples * (0.6 + 0.08 * i))
            train_idx = indices[:split_point]
            valid_idx = indices[split_point:]
            splits.append((train_idx, valid_idx))

        return splits

    def tune_lgb(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
        n_trials: int = 50,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Tune LightGBM hyperparameters using Optuna.

        Search space:
        - learning_rate: [0.01, 0.2] log-uniform
        - max_depth: [5, 15] integer
        - num_leaves: [31, 255] integer
        - feature_fraction: [0.5, 1.0] uniform
        - bagging_fraction: [0.5, 1.0] uniform
        - lambda_l1: [0, 1] uniform
        - lambda_l2: [0, 1] uniform

        Args:
            X_train: Training features.
            y_train: Training targets.
            X_valid: Validation features.
            y_valid: Validation targets.
            n_trials: Number of Optuna trials.
            timeout: Timeout in seconds for the study.

        Returns:
            Dictionary of best hyperparameters.
        """
        try:
            import optuna
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError(f"Required package not installed: {e}")

        logger.info(
            "Starting LightGBM tuning: n_trials=%d, X_train shape=%s",
            n_trials,
            X_train.shape,
        )

        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64)
        X_valid = np.asarray(X_valid, dtype=np.float64)
        y_valid = np.asarray(y_valid, dtype=np.float64)

        def objective(trial: optuna.Trial) -> float:
            """Optuna objective function for LightGBM."""
            params = {
                "objective": "regression",
                "metric": "rmse",
                "boosting_type": "gbdt",
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.2, log=True
                ),
                "max_depth": trial.suggest_int("max_depth", 5, 15),
                "num_leaves": trial.suggest_int("num_leaves", 31, 255),
                "feature_fraction": trial.suggest_float(
                    "feature_fraction", 0.5, 1.0
                ),
                "bagging_fraction": trial.suggest_float(
                    "bagging_fraction", 0.5, 1.0
                ),
                "lambda_l1": trial.suggest_float("lambda_l1", 0, 1),
                "lambda_l2": trial.suggest_float("lambda_l2", 0, 1),
                "n_estimators": 500,
                "random_state": 42,
                "n_jobs": -1,
                "verbosity": -1,
            }

            # TimeSeriesSplit CV
            cv_scores = []
            splits = self._get_time_series_splits(len(X_train), n_splits=3)

            for train_idx, valid_idx in splits:
                X_tr, y_tr = X_train[train_idx], y_train[train_idx]
                X_val, y_val = X_train[valid_idx], y_train[valid_idx]

                model = lgb.LGBMRegressor(**params)
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[
                        lgb.early_stopping(30),
                        lgb.log_evaluation(0),
                    ],
                )

                preds = model.predict(X_val)
                rankic = _compute_rankic(y_val, preds)
                cv_scores.append(rankic)

            return np.mean(cv_scores)

        # Run on validation set only (held-out evaluation)
        def objective_heldout(trial: optuna.Trial) -> float:
            """Optuna objective function using held-out validation set."""
            params = {
                "objective": "regression",
                "metric": "rmse",
                "boosting_type": "gbdt",
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.2, log=True
                ),
                "max_depth": trial.suggest_int("max_depth", 5, 15),
                "num_leaves": trial.suggest_int("num_leaves", 31, 255),
                "feature_fraction": trial.suggest_float(
                    "feature_fraction", 0.5, 1.0
                ),
                "bagging_fraction": trial.suggest_float(
                    "bagging_fraction", 0.5, 1.0
                ),
                "lambda_l1": trial.suggest_float("lambda_l1", 0, 1),
                "lambda_l2": trial.suggest_float("lambda_l2", 0, 1),
                "n_estimators": 1000,
                "random_state": 42,
                "n_jobs": -1,
                "verbosity": -1,
            }

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                callbacks=[
                    lgb.early_stopping(30),
                    lgb.log_evaluation(0),
                ],
            )

            preds = model.predict(X_valid)
            rankic = _compute_rankic(y_valid, preds)
            return rankic

        study = self._create_study("lgb")
        study.optimize(
            objective_heldout,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )

        self.best_params["lgb"] = study.best_params
        logger.info(
            "LightGBM tuning complete: best RankIC=%f, best params=%s",
            study.best_value,
            study.best_params,
        )

        return dict(study.best_params)

    def tune_xgb(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
        n_trials: int = 50,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Tune XGBoost hyperparameters using Optuna.

        Search space:
        - learning_rate: [0.01, 0.2] log-uniform
        - max_depth: [5, 15] integer
        - subsample: [0.5, 1.0] uniform
        - colsample_bytree: [0.5, 1.0] uniform
        - reg_alpha: [0, 1] uniform
        - reg_lambda: [0, 1] uniform

        Args:
            X_train: Training features.
            y_train: Training targets.
            X_valid: Validation features.
            y_valid: Validation targets.
            n_trials: Number of Optuna trials.
            timeout: Timeout in seconds for the study.

        Returns:
            Dictionary of best hyperparameters.
        """
        try:
            import optuna
            import xgboost as xgb
        except ImportError as e:
            raise ImportError(f"Required package not installed: {e}")

        logger.info(
            "Starting XGBoost tuning: n_trials=%d, X_train shape=%s",
            n_trials,
            X_train.shape,
        )

        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64)
        X_valid = np.asarray(X_valid, dtype=np.float64)
        y_valid = np.asarray(y_valid, dtype=np.float64)

        def objective(trial: optuna.Trial) -> float:
            """Optuna objective function for XGBoost."""
            params = {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.2, log=True
                ),
                "max_depth": trial.suggest_int("max_depth", 5, 15),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.5, 1.0
                ),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
                "n_estimators": 1000,
                "random_state": 42,
                "n_jobs": -1,
                "verbosity": 0,
            }

            model = xgb.XGBRegressor(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=False,
            )

            preds = model.predict(X_valid)
            rankic = _compute_rankic(y_valid, preds)
            return rankic

        study = self._create_study("xgb")
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )

        self.best_params["xgb"] = study.best_params
        logger.info(
            "XGBoost tuning complete: best RankIC=%f, best params=%s",
            study.best_value,
            study.best_params,
        )

        return dict(study.best_params)

    def tune_cat(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
        n_trials: int = 50,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Tune CatBoost hyperparameters using Optuna.

        Search space:
        - learning_rate: [0.01, 0.2] log-uniform
        - depth: [5, 12] integer
        - l2_leaf_reg: [1, 10] uniform
        - subsample: [0.5, 1.0] uniform

        Args:
            X_train: Training features.
            y_train: Training targets.
            X_valid: Validation features.
            y_valid: Validation targets.
            n_trials: Number of Optuna trials.
            timeout: Timeout in seconds for the study.

        Returns:
            Dictionary of best hyperparameters.
        """
        try:
            import optuna
            from catboost import CatBoostRegressor
        except ImportError as e:
            raise ImportError(f"Required package not installed: {e}")

        logger.info(
            "Starting CatBoost tuning: n_trials=%d, X_train shape=%s",
            n_trials,
            X_train.shape,
        )

        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64)
        X_valid = np.asarray(X_valid, dtype=np.float64)
        y_valid = np.asarray(y_valid, dtype=np.float64)

        def objective(trial: optuna.Trial) -> float:
            """Optuna objective function for CatBoost."""
            params = {
                "loss_function": "RMSE",
                "eval_metric": "RMSE",
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.2, log=True
                ),
                "depth": trial.suggest_int("depth", 5, 12),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "iterations": 1000,
                "random_seed": 42,
                "thread_count": -1,
                "verbose": 0,
            }

            model = CatBoostRegressor(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=(X_valid, y_valid),
            )

            preds = model.predict(X_valid)
            rankic = _compute_rankic(y_valid, preds)
            return rankic

        study = self._create_study("cat")
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )

        self.best_params["cat"] = study.best_params
        logger.info(
            "CatBoost tuning complete: best RankIC=%f, best params=%s",
            study.best_value,
            study.best_params,
        )

        return dict(study.best_params)

    def tune_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
        n_trials: int = 50,
        timeout: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Tune all three model types (LightGBM, XGBoost, CatBoost).

        Args:
            X_train: Training features.
            y_train: Training targets.
            X_valid: Validation features.
            y_valid: Validation targets.
            n_trials: Number of Optuna trials per model.
            timeout: Timeout in seconds per model.

        Returns:
            Dictionary with keys 'lgb', 'xgb', 'cat' mapping to
            best parameter dictionaries.
        """
        logger.info(
            "Starting full hyperparameter tuning for all models: n_trials=%d each",
            n_trials,
        )

        results = {}

        # Tune LightGBM
        try:
            logger.info("Tuning LightGBM...")
            results["lgb"] = self.tune_lgb(
                X_train, y_train, X_valid, y_valid,
                n_trials=n_trials, timeout=timeout,
            )
        except Exception as e:
            logger.error("LightGBM tuning failed: %s", e)
            results["lgb"] = {"error": str(e)}

        # Tune XGBoost
        try:
            logger.info("Tuning XGBoost...")
            results["xgb"] = self.tune_xgb(
                X_train, y_train, X_valid, y_valid,
                n_trials=n_trials, timeout=timeout,
            )
        except Exception as e:
            logger.error("XGBoost tuning failed: %s", e)
            results["xgb"] = {"error": str(e)}

        # Tune CatBoost
        try:
            logger.info("Tuning CatBoost...")
            results["cat"] = self.tune_cat(
                X_train, y_train, X_valid, y_valid,
                n_trials=n_trials, timeout=timeout,
            )
        except Exception as e:
            logger.error("CatBoost tuning failed: %s", e)
            results["cat"] = {"error": str(e)}

        logger.info("Full hyperparameter tuning complete")
        return results

    def get_best_params(self, model_type: str) -> Dict[str, Any]:
        """Get best parameters for a specific model type.

        Args:
            model_type: One of 'lgb', 'xgb', 'cat'.

        Returns:
            Dictionary of best hyperparameters.

        Raises:
            ValueError: If model_type is invalid or tuning hasn't been run.
        """
        if model_type not in self.best_params:
            raise ValueError(
                f"Unknown model type: {model_type}. Use 'lgb', 'xgb', or 'cat'."
            )

        if not self.best_params[model_type]:
            raise ValueError(
                f"No best params for {model_type}. Run tune_{model_type}() first."
            )

        return dict(self.best_params[model_type])

    def get_all_best_params(self) -> Dict[str, Dict[str, Any]]:
        """Get best parameters for all model types.

        Returns:
            Dictionary mapping model type to best parameter dictionary.
        """
        return {
            k: dict(v) for k, v in self.best_params.items() if v
        }