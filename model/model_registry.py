"""
Model registry for the Stock-AI quantitative trading project.

Provides versioned model management with save/load/rollback capabilities,
metadata tracking, and best model selection based on validation metrics.
"""

import json
import logging
import os
import pickle
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default storage path relative to project root
_DEFAULT_STORAGE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models"
)


class ModelRegistry:
    """Versioned model storage and management system.

    Manages model versions with metadata tracking (timestamps, parameters,
    validation metrics) and provides model discovery, loading, and
    best-model selection.

    Storage structure::

        models/{model_name}/{version}/
            model.pkl       # Serialized model file
            meta.json       # Metadata (params, metrics, timestamp)

    Attributes:
        storage_path: Root directory for model storage.
        metadata_filename: Name of metadata JSON file.
        model_filename: Name of serialized model file.
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        metadata_filename: str = "meta.json",
        model_filename: str = "model.pkl",
    ):
        """Initialize the model registry.

        Args:
            storage_path: Root directory for model storage.
                Defaults to project/models/ directory.
            metadata_filename: Name of metadata JSON file per version.
            model_filename: Name of serialized model file per version.
        """
        self.storage_path = storage_path or _DEFAULT_STORAGE_PATH
        self.metadata_filename = metadata_filename
        self.model_filename = model_filename

        # Ensure storage directory exists
        os.makedirs(self.storage_path, exist_ok=True)
        logger.info(
            "ModelRegistry initialized: storage_path=%s", self.storage_path
        )

    def _get_model_dir(self, name: str) -> str:
        """Get the directory path for a model name.

        Args:
            name: Model name/identifier.

        Returns:
            Directory path string.
        """
        return os.path.join(self.storage_path, name)

    def _get_version_dir(self, name: str, version: str) -> str:
        """Get the directory path for a specific model version.

        Args:
            name: Model name/identifier.
            version: Version string.

        Returns:
            Version directory path string.
        """
        return os.path.join(self._get_model_dir(name), version)

    def _generate_version(self) -> str:
        """Generate a timestamp-based version string.

        Returns:
            Version string in format 'v_YYYYMMDD_HHMMSS'.
        """
        return datetime.now().strftime("v_%Y%m%d_%H%M%S")

    def _build_metadata(
        self,
        model: Any,
        name: str,
        version: str,
        val_ic: Optional[float] = None,
        val_rankic: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build metadata dictionary for a model version.

        Args:
            model: The model instance.
            name: Model name.
            version: Version string.
            val_ic: Validation IC (Pearson correlation).
            val_rankic: Validation RankIC (Spearman correlation).
            extra: Additional metadata fields.

        Returns:
            Metadata dictionary.
        """
        metadata = {
            "name": name,
            "version": version,
            "created_at": datetime.now().isoformat(),
            "model_type": type(model).__name__,
        }

        # Add validation metrics if available
        if val_ic is not None:
            metadata["val_ic"] = val_ic
        if val_rankic is not None:
            metadata["val_rankic"] = val_rankic

        # Add model-specific info
        if hasattr(model, "params"):
            metadata["params"] = model.params
        elif hasattr(model, "get_params") and callable(model.get_params):
            try:
                metadata["params"] = model.get_params()
            except Exception:
                pass

        if hasattr(model, "best_iteration_"):
            metadata["best_iteration"] = model.best_iteration_

        if hasattr(model, "best_score_"):
            metadata["best_score"] = model.best_score_

        if hasattr(model, "feature_names_"):
            metadata["n_features"] = (
                len(model.feature_names_)
                if model.feature_names_
                else None
            )

        # Add extra metadata
        if extra:
            metadata.update(extra)

        return metadata

    def save_model(
        self,
        model: Any,
        name: str,
        version: Optional[str] = None,
        val_ic: Optional[float] = None,
        val_rankic: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save a model with versioning and metadata.

        Args:
            model: Model instance to save (must have save/load methods).
            name: Model name/identifier (e.g., 'lgb_v1', 'ensemble').
            version: Version string. Auto-generated if None.
            val_ic: Validation IC for tracking.
            val_rankic: Validation RankIC for tracking.
            extra: Additional metadata to store.

        Returns:
            Version string of the saved model.

        Raises:
            ValueError: If model cannot be serialized.
            IOError: If files cannot be written.
        """
        if version is None:
            version = self._generate_version()

        version_dir = self._get_version_dir(name, version)
        os.makedirs(version_dir, exist_ok=True)

        logger.info(
            "Saving model '%s' version '%s' to %s", name, version, version_dir
        )

        # Build and save metadata
        metadata = self._build_metadata(
            model, name, version, val_ic, val_rankic, extra
        )

        metadata_path = os.path.join(version_dir, self.metadata_filename)
        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info("Metadata saved to %s", metadata_path)
        except Exception as e:
            logger.error("Failed to save metadata: %s", e)
            raise IOError(f"Failed to save metadata: {e}") from e

        # Save model
        model_path = os.path.join(version_dir, self.model_filename)
        try:
            # Use model's save method if available, otherwise pickle
            if hasattr(model, "save") and callable(model.save):
                model.save(model_path)
            else:
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
            logger.info("Model saved to %s", model_path)
        except Exception as e:
            # Clean up metadata on failure
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            logger.error("Failed to save model: %s", e)
            raise IOError(f"Failed to save model: {e}") from e

        logger.info(
            "Model '%s' version '%s' saved successfully. val_ic=%s, val_rankic=%s",
            name,
            version,
            val_ic,
            val_rankic,
        )
        return version

    def load_model(
        self,
        name: str,
        version: str = "latest",
        model_cls: Optional[type] = None,
    ) -> Any:
        """Load a specific version of a model.

        Args:
            name: Model name/identifier.
            version: Version string or 'latest' for the most recent version.
            model_cls: Optional model class for reconstruction.

        Returns:
            Loaded model instance.

        Raises:
            FileNotFoundError: If model or version does not exist.
            ValueError: If no versions exist for this model.
            IOError: If files cannot be read.
        """
        model_dir = self._get_model_dir(name)

        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model '{name}' not found in registry")

        if version == "latest":
            version = self._get_latest_version(name)
            if version is None:
                raise ValueError(f"No versions found for model '{name}'")

        version_dir = self._get_version_dir(name, version)

        if not os.path.exists(version_dir):
            raise FileNotFoundError(
                f"Version '{version}' not found for model '{name}'"
            )

        model_path = os.path.join(version_dir, self.model_filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info("Loading model '%s' version '%s'", name, version)

        try:
            with open(model_path, "rb") as f:
                data = pickle.load(f)

            # If data is a dict with model stored inside
            if isinstance(data, dict) and "model" in data:
                model_obj = data["model"]
            else:
                model_obj = data

            logger.info("Model loaded successfully")
            return model_obj
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            raise IOError(f"Failed to load model from {model_path}: {e}") from e

    def load_metadata(
        self, name: str, version: str = "latest"
    ) -> Dict[str, Any]:
        """Load metadata for a specific model version.

        Args:
            name: Model name/identifier.
            version: Version string or 'latest'.

        Returns:
            Metadata dictionary.

        Raises:
            FileNotFoundError: If model or version does not exist.
        """
        model_dir = self._get_model_dir(name)
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model '{name}' not found in registry")

        if version == "latest":
            version = self._get_latest_version(name)
            if version is None:
                raise ValueError(f"No versions found for model '{name}'")

        metadata_path = os.path.join(
            self._get_version_dir(name, version), self.metadata_filename
        )

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Metadata not found for model '{name}' version '{version}'"
            )

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            logger.error("Failed to load metadata: %s", e)
            raise IOError(f"Failed to load metadata: {e}") from e

    def _get_latest_version(self, name: str) -> Optional[str]:
        """Get the latest version string for a model.

        Args:
            name: Model name/identifier.

        Returns:
            Latest version string or None if no versions exist.
        """
        model_dir = self._get_model_dir(name)
        if not os.path.exists(model_dir):
            return None

        versions = os.listdir(model_dir)
        if not versions:
            return None

        # Sort by modification time, newest first
        versions.sort(
            key=lambda v: os.path.getmtime(
                os.path.join(model_dir, v)
            ),
            reverse=True,
        )
        return versions[0]

    def _get_all_versions(self, name: str) -> List[str]:
        """Get all version strings for a model, sorted by recency.

        Args:
            name: Model name/identifier.

        Returns:
            List of version strings.
        """
        model_dir = self._get_model_dir(name)
        if not os.path.exists(model_dir):
            return []

        versions = os.listdir(model_dir)
        versions.sort(
            key=lambda v: os.path.getmtime(
                os.path.join(model_dir, v)
            ),
            reverse=True,
        )
        return versions

    def list_models(self) -> pd.DataFrame:
        """List all saved models with their versions and metadata.

        Returns:
            DataFrame with columns:
            - name: Model name
            - version: Version string
            - created_at: Creation timestamp
            - model_type: Model class name
            - val_ic: Validation IC (if available)
            - val_rankic: Validation RankIC (if available)
            - n_features: Number of features (if available)
        """
        records = []

        if not os.path.exists(self.storage_path):
            logger.warning(
                "Storage path does not exist: %s", self.storage_path
            )
            return pd.DataFrame()

        for model_name in os.listdir(self.storage_path):
            model_dir = self._get_model_dir(model_name)
            if not os.path.isdir(model_dir):
                continue

            for version in os.listdir(model_dir):
                version_dir = os.path.join(model_dir, version)
                if not os.path.isdir(version_dir):
                    continue

                metadata_path = os.path.join(
                    version_dir, self.metadata_filename
                )
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, "r", encoding="utf-8") as f:
                            meta = json.load(f)

                        records.append(
                            {
                                "name": meta.get("name", model_name),
                                "version": meta.get("version", version),
                                "created_at": meta.get("created_at", ""),
                                "model_type": meta.get("model_type", ""),
                                "val_ic": meta.get("val_ic"),
                                "val_rankic": meta.get("val_rankic"),
                                "n_features": meta.get("n_features"),
                            }
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to read metadata for %s/%s: %s",
                            model_name,
                            version,
                            e,
                        )

        if not records:
            logger.info("No models found in registry")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df = df.sort_values("created_at", ascending=False).reset_index(
            drop=True
        )
        logger.info("Listed %d model versions", len(df))
        return df

    def delete_model(self, name: str, version: Optional[str] = None) -> bool:
        """Delete a model version or entire model from registry.

        Args:
            name: Model name/identifier.
            version: Specific version to delete. If None, deletes all
                versions of this model.

        Returns:
            True if deletion was successful.

        Raises:
            FileNotFoundError: If model does not exist.
        """
        if version is not None:
            version_dir = self._get_version_dir(name, version)
            if not os.path.exists(version_dir):
                raise FileNotFoundError(
                    f"Version '{version}' not found for model '{name}'"
                )

            shutil.rmtree(version_dir)
            logger.info(
                "Deleted model '%s' version '%s'", name, version
            )
            return True
        else:
            model_dir = self._get_model_dir(name)
            if not os.path.exists(model_dir):
                raise FileNotFoundError(
                    f"Model '{name}' not found in registry"
                )

            shutil.rmtree(model_dir)
            logger.info("Deleted all versions of model '%s'", name)
            return True

    def get_best_model(self, name: str, metric: str = "val_rankic") -> Any:
        """Load the best model version based on validation metric.

        Compares all versions of a model and returns the one with the
        highest validation metric.

        Args:
            name: Model name/identifier.
            metric: Metric to compare ('val_rankic' or 'val_ic').

        Returns:
            The best model instance.

        Raises:
            ValueError: If no versions with the metric are found.
            FileNotFoundError: If model does not exist.
        """
        model_dir = self._get_model_dir(name)
        if not os.path.exists(model_dir):
            raise FileNotFoundError(
                f"Model '{name}' not found in registry"
            )

        best_version = None
        best_metric = -float("inf")

        for version in os.listdir(model_dir):
            metadata_path = os.path.join(
                model_dir, version, self.metadata_filename
            )
            if not os.path.exists(metadata_path):
                continue

            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                metric_val = meta.get(metric)
                if metric_val is not None and metric_val > best_metric:
                    best_metric = metric_val
                    best_version = version
            except Exception as e:
                logger.warning(
                    "Skipping version %s due to error: %s", version, e
                )

        if best_version is None:
            raise ValueError(
                f"No versions with metric '{metric}' found for model '{name}'"
            )

        logger.info(
            "Best model '%s': version=%s, %s=%f",
            name,
            best_version,
            metric,
            best_metric,
        )

        return self.load_model(name, version=best_version)

    def get_best_model_info(
        self, name: str, metric: str = "val_rankic"
    ) -> Dict[str, Any]:
        """Get metadata for the best model version.

        Args:
            name: Model name/identifier.
            metric: Metric to compare.

        Returns:
            Metadata dictionary for the best version.

        Raises:
            ValueError: If no versions with the metric are found.
        """
        model_dir = self._get_model_dir(name)
        if not os.path.exists(model_dir):
            raise FileNotFoundError(
                f"Model '{name}' not found in registry"
            )

        best_version = None
        best_metric = -float("inf")
        best_meta = {}

        for version in os.listdir(model_dir):
            metadata_path = os.path.join(
                model_dir, version, self.metadata_filename
            )
            if not os.path.exists(metadata_path):
                continue

            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                metric_val = meta.get(metric)
                if metric_val is not None and metric_val > best_metric:
                    best_metric = metric_val
                    best_version = version
                    best_meta = meta
            except Exception as e:
                logger.warning(
                    "Skipping version %s due to error: %s", version, e
                )

        if best_version is None:
            raise ValueError(
                f"No versions with metric '{metric}' found for model '{name}'"
            )

        logger.info(
            "Best model info for '%s': version=%s, %s=%f",
            name,
            best_version,
            metric,
            best_metric,
        )

        return best_meta

    def rollback(self, name: str, target_version: str) -> str:
        """Rollback to a specific version by creating a new version as a copy.

        Creates a new version that is a copy of the target version,
        preserving the version history.

        Args:
            name: Model name/identifier.
            target_version: Version to rollback to.

        Returns:
            New version string created from the rollback.

        Raises:
            FileNotFoundError: If target version does not exist.
        """
        target_dir = self._get_version_dir(name, target_version)
        if not os.path.exists(target_dir):
            raise FileNotFoundError(
                f"Target version '{target_version}' not found for model '{name}'"
            )

        new_version = self._generate_version()
        new_dir = self._get_version_dir(name, new_version)
        os.makedirs(new_dir, exist_ok=True)

        # Copy all files from target version
        for filename in os.listdir(target_dir):
            src = os.path.join(target_dir, filename)
            dst = os.path.join(new_dir, filename)
            if os.path.isfile(src):
                shutil.copy2(src, dst)

        # Update metadata with rollback info
        meta_path = os.path.join(new_dir, self.metadata_filename)
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["version"] = new_version
            meta["created_at"] = datetime.now().isoformat()
            meta["rolled_back_from"] = target_version
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info(
            "Rolled back model '%s' from version '%s' to new version '%s'",
            name,
            target_version,
            new_version,
        )

        return new_version

    def get_version_history(self, name: str) -> List[Dict[str, Any]]:
        """Get version history for a model with metadata.

        Args:
            name: Model name/identifier.

        Returns:
            List of metadata dictionaries sorted by creation time
            (newest first).

        Raises:
            FileNotFoundError: If model does not exist.
        """
        model_dir = self._get_model_dir(name)
        if not os.path.exists(model_dir):
            raise FileNotFoundError(
                f"Model '{name}' not found in registry"
            )

        history = []
        for version in os.listdir(model_dir):
            meta_path = os.path.join(
                model_dir, version, self.metadata_filename
            )
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    history.append(meta)
                except Exception as e:
                    logger.warning(
                        "Failed to read metadata for version %s: %s",
                        version,
                        e,
                    )

        history.sort(
            key=lambda x: x.get("created_at", ""), reverse=True
        )
        return history

    def model_exists(self, name: str, version: Optional[str] = None) -> bool:
        """Check if a model or specific version exists in the registry.

        Args:
            name: Model name/identifier.
            version: Optional specific version to check.

        Returns:
            True if the model/version exists.
        """
        if version:
            return os.path.exists(
                os.path.join(
                    self._get_version_dir(name, version),
                    self.model_filename,
                )
            )
        return os.path.exists(self._get_model_dir(name))