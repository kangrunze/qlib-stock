"""
配置管理器 - 读取和管理 settings.yaml 配置文件

提供统一的配置访问接口，支持路径、数据源、因子、模型等配置项的获取。
"""

import os
import logging

import yaml

logger = logging.getLogger(__name__)


class SettingsManager:
    """配置管理器，从 settings.yaml 加载配置并提供便捷访问方法。

    使用单例模式，确保全局共享同一份配置。

    Attributes:
        _instance: 单例实例
        _config: 原始配置字典
        _settings_path: settings.yaml 文件路径
    """

    _instance = None
    _config = None

    def __new__(cls, settings_path: str = None):
        """单例模式：确保全局只有一个配置实例。"""
        if cls._instance is None:
            cls._instance = super(SettingsManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, settings_path: str = None):
        """初始化配置管理器。

        Args:
            settings_path: settings.yaml 文件的路径。如果为 None，自动查找。
        """
        if self._initialized:
            return

        if settings_path is None:
            # 自动查找 config/settings.yaml
            current_dir = os.path.dirname(os.path.abspath(__file__))
            settings_path = os.path.join(current_dir, "settings.yaml")

        self._settings_path = settings_path
        self._load_config()
        self._initialized = True

    def _load_config(self):
        """从 YAML 文件加载配置。"""
        try:
            with open(self._settings_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f)
            if self._config is None:
                self._config = {}
            logger.info("配置加载成功: %s", self._settings_path)
        except FileNotFoundError:
            logger.warning("配置文件未找到: %s，使用默认配置", self._settings_path)
            self._config = {}
        except yaml.YAMLError as e:
            logger.error("YAML 解析错误: %s", e)
            self._config = {}

    def reload(self):
        """重新加载配置文件。"""
        self._load_config()
        logger.info("配置已重新加载")

    # ===== 路径配置 =====

    def get_path(self, key: str, default: str = None) -> str:
        """获取路径配置项。

        Args:
            key: 路径键名，如 'base_dir', 'data_dir' 等
            default: 默认值

        Returns:
            路径字符串
        """
        value = self._config.get("paths", {}).get(key, default)
        if value:
            value = os.path.normpath(value)
        return value

    @property
    def base_dir(self) -> str:
        return self.get_path("base_dir", ".")

    @property
    def data_dir(self) -> str:
        return self.get_path("data_dir", os.path.join(self.base_dir, "data"))

    @property
    def model_dir(self) -> str:
        return self.get_path("model_dir", os.path.join(self.base_dir, "models"))

    @property
    def output_dir(self) -> str:
        return self.get_path("output_dir", os.path.join(self.base_dir, "output"))

    @property
    def log_dir(self) -> str:
        return self.get_path("log_dir", os.path.join(self.output_dir, "logs"))

    @property
    def parquet_dir(self) -> str:
        return self.get_path("parquet_dir", os.path.join(self.data_dir, "daily"))

    @property
    def features_dir(self) -> str:
        return self.get_path("features_dir", os.path.join(self.data_dir, "features"))

    @property
    def index_dir(self) -> str:
        return self.get_path("index_dir", os.path.join(self.data_dir, "index"))

    @property
    def industry_dir(self) -> str:
        return self.get_path("industry_dir", os.path.join(self.data_dir, "industry"))

    @property
    def financial_dir(self) -> str:
        return self.get_path("financial_dir", os.path.join(self.data_dir, "financial"))

    # ===== 数据源配置 =====

    @property
    def primary_source(self) -> str:
        return self._config.get("data_source", {}).get("primary", "akshare")

    @property
    def backup_source(self) -> str:
        return self._config.get("data_source", {}).get("backup", "baostock")

    @property
    def start_date(self) -> str:
        return self._config.get("data_source", {}).get("start_date", "2015-01-01")

    @property
    def end_date(self) -> str:
        return self._config.get("data_source", {}).get("end_date", "2025-12-31")

    # ===== 因子配置 =====

    def get_factor_config(self, key: str, default=None):
        """获取因子配置项。"""
        return self._config.get("factors", {}).get(key, default)

    @property
    def turn_periods(self) -> list:
        return self.get_factor_config("turn_periods", [5, 20])

    @property
    def volume_ratio_periods(self) -> list:
        return self.get_factor_config("volume_ratio_periods", [5, 20])

    @property
    def momentum_periods(self) -> list:
        return self.get_factor_config("momentum_periods", [10, 20])

    @property
    def volatility_periods(self) -> list:
        return self.get_factor_config("volatility_periods", [20, 60])

    @property
    def rsi_periods(self) -> list:
        return self.get_factor_config("rsi_periods", [6, 14, 24])

    # ===== 模型配置 =====

    def get_model_config(self, key: str, default=None):
        """获取模型配置项。"""
        return self._config.get("model", {}).get(key, default)

    @property
    def default_model(self) -> str:
        return self.get_model_config("default_model", "lightgbm")

    @property
    def cv_folds(self) -> int:
        return self.get_model_config("cv_folds", 5)

    @property
    def random_state(self) -> int:
        return self.get_model_config("random_state", 42)

    # ===== 因子存储配置 =====

    def get_store_config(self, key: str, default=None):
        """获取因子存储配置项。"""
        return self._config.get("factor_store", {}).get(key, default)

    @property
    def store_compression(self) -> str:
        return self.get_store_config("compression", "snappy")

    @property
    def default_version(self) -> str:
        return self.get_store_config("default_version", "v1.0")

    # ===== 日志配置 =====

    def get_logging_config(self, key: str, default=None):
        """获取日志配置项。"""
        return self._config.get("logging", {}).get(key, default)

    @property
    def log_level(self) -> str:
        return self.get_logging_config("level", "INFO")

    @property
    def log_format(self) -> str:
        return self.get_logging_config(
            "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    # ===== 便捷方法 =====

    def get(self, key_path: str, default=None):
        """通过点分隔路径获取配置项。

        Args:
            key_path: 配置路径，如 'paths.base_dir', 'factors.rsi_periods'
            default: 默认值

        Returns:
            配置值

        Example:
            >>> sm = SettingsManager()
            >>> sm.get('paths.data_dir')
            'd:/project/qlib-stock/stock-ai/data'
        """
        keys = key_path.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value if value is not None else default

    def ensure_dirs(self):
        """确保所有配置中的目录存在。"""
        paths_to_check = [
            self.base_dir,
            self.data_dir,
            self.model_dir,
            self.output_dir,
            self.log_dir,
            self.parquet_dir,
            self.features_dir,
            self.index_dir,
            self.industry_dir,
            self.financial_dir,
        ]
        for p in paths_to_check:
            if p and not os.path.exists(p):
                os.makedirs(p, exist_ok=True)
                logger.info("创建目录: %s", p)

    def __repr__(self) -> str:
        return f"SettingsManager(path='{self._settings_path}', items={len(self._config)})"


# 全局配置实例（方便直接导入使用）
settings = SettingsManager()