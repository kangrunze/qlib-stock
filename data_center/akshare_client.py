"""
AKShare API 封装客户端

提供多线程股票数据下载、重试机制、速率限制、中文列名映射、数据保存等功能。
"""

import os
import time
import random
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import yaml

logger = logging.getLogger(__name__)

# ---- 加载配置 ----
_CONFIG_PATH = Path(__file__).parent.parent / "config" / "settings.yaml"
_config = {}
if _CONFIG_PATH.exists():
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        _config = yaml.safe_load(f)
else:
    logger.warning("未找到 settings.yaml，使用默认配置")

_paths = _config.get("paths", {})
_ds_conf = _config.get("data_source", {})

RETRY_MAX = _ds_conf.get("retry_max", 3)
RETRY_DELAY = _ds_conf.get("retry_delay", 2)
MAX_WORKERS = _ds_conf.get("max_workers", 10)
START_DATE = _ds_conf.get("start_date", "2005-01-01")
PARQUET_DIR = _paths.get("parquet_dir", "data/daily")
INDEX_DIR = _paths.get("index_dir", "data/index")
INDUSTRY_DIR = _paths.get("industry_dir", "data/industry")
REFERENCE_DIR = _paths.get("reference_dir", "data/reference")

# ---- 中英文列名映射 ----
CN_TO_EN_COL_MAP = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "涨跌幅": "pct_chg",
    "换手率": "turnover_rate",
    "振幅": "amplitude",
    "股票代码": "code",
    "股票名称": "name",
    "前收盘": "pre_close",
}


def _retry(func: Callable, max_retries: int = RETRY_MAX, delay: float = RETRY_DELAY) -> Callable:
    """带重试逻辑的装饰器"""
    def wrapper(*args, **kwargs):
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exc = e
                if attempt < max_retries:
                    wait = delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
                    logger.warning(
                        "第 %d/%d 次尝试失败 [%s]: %s，%.1f秒后重试...",
                        attempt, max_retries, func.__name__, str(e)[:120], wait
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        "第 %d/%d 次尝试失败 [%s]: %s，已达最大重试次数",
                        attempt, max_retries, func.__name__, str(e)[:120]
                    )
        raise last_exc
    return wrapper


def _auto_retry_methods(cls):
    """自动对类中以 download_ 开头的方法添加重试装饰器"""
    for attr_name in dir(cls):
        if attr_name.startswith("download_") or attr_name in ("get_stock_list",):
            attr = getattr(cls, attr_name)
            if callable(attr):
                setattr(cls, attr_name, _retry(attr))
    return cls


def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """将DataFrame的中文列名映射为英文列名"""
    rename_map = {}
    for col in df.columns:
        if col in CN_TO_EN_COL_MAP:
            rename_map[col] = CN_TO_EN_COL_MAP[col]
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _ensure_dir(path: str) -> str:
    """确保目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def _rate_limit(min_interval: float = 0.3):
    """请求间速率限制"""
    if not hasattr(_rate_limit, "_last_call"):
        _rate_limit._last_call = 0.0
    now = time.time()
    elapsed = now - _rate_limit._last_call
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _rate_limit._last_call = time.time()


class AKShareClient:
    """
    AKShare API 封装客户端

    提供多线程数据下载、中文列名映射、ST标记、停牌标记、Parquet保存等功能。

    Attributes:
        retry_max: 最大重试次数
        retry_delay: 重试间隔（秒）
        max_workers: 并发下载线程数
        parquet_dir: 日线数据Parquet保存目录
    """

    def __init__(
        self,
        retry_max: int = RETRY_MAX,
        retry_delay: float = RETRY_DELAY,
        max_workers: int = MAX_WORKERS,
        parquet_dir: Optional[str] = None,
    ):
        self.retry_max = retry_max
        self.retry_delay = retry_delay
        self.max_workers = max_workers
        self.parquet_dir = Path(parquet_dir or PARQUET_DIR)
        _ensure_dir(str(self.parquet_dir))
        self._ak = None
        logger.info(
            "AKShareClient 初始化完成: workers=%d, retry_max=%d, parquet_dir=%s",
            max_workers, retry_max, self.parquet_dir
        )

    @property
    def ak(self):
        """延迟加载 akshare"""
        if self._ak is None:
            import akshare as ak
            self._ak = ak
        return self._ak

    def _do_with_retry(self, func: Callable, *args, **kwargs):
        """执行带重试的函数调用"""
        last_exc = None
        for attempt in range(1, self.retry_max + 1):
            try:
                _rate_limit()
                return func(*args, **kwargs)
            except Exception as e:
                last_exc = e
                if attempt < self.retry_max:
                    wait = self.retry_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
                    logger.warning(
                        "第 %d/%d 次调用失败: %s，%.1f秒后重试...",
                        attempt, self.retry_max, str(e)[:100], wait
                    )
                    time.sleep(wait)
                else:
                    logger.error("已达最大重试次数 %d: %s", self.retry_max, str(e)[:100])
        raise last_exc

    # ==================== 股票列表 ====================

    def get_stock_list(self) -> pd.DataFrame:
        """
        获取A股全量股票列表

        Returns:
            DataFrame, 列: code, name
        """
        logger.info("正在获取A股全量股票列表...")
        df = self._do_with_retry(self.ak.stock_info_a_code_name)
        df = _map_columns(df)

        # 统一code格式: 6位数字
        if "code" in df.columns:
            df["code"] = df["code"].astype(str).str.zfill(6)

        # 标记ST
        if "name" in df.columns:
            df["is_st"] = df["name"].str.contains("ST", na=False)

        logger.info("获取到 %d 只股票", len(df))
        return df

    # ==================== 日线数据下载 ====================

    def download_daily_data(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """
        下载单只股票日线数据（前复权）

        Args:
            symbol: 股票代码，如 "000001"
            start: 起始日期 "YYYYMMDD" 或 "YYYY-MM-DD"
            end: 结束日期 "YYYYMMDD" 或 "YYYY-MM-DD"
            adjust: 复权类型，"qfq"=前复权, "hfq"=后复权, ""=不复权

        Returns:
            DataFrame with standardized English columns
        """
        if start is None:
            start = START_DATE.replace("-", "")
        else:
            start = start.replace("-", "")
        if end is None:
            end = datetime.now().strftime("%Y%m%d")
        else:
            end = end.replace("-", "")

        logger.debug("下载日线数据: symbol=%s, start=%s, end=%s, adjust=%s", symbol, start, end, adjust)

        try:
            df = self._do_with_retry(
                self.ak.stock_zh_a_hist,
                symbol=symbol,
                period="daily",
                start_date=start,
                end_date=end,
                adjust=adjust,
            )
        except Exception:
            logger.debug("stock_zh_a_hist 失败，尝试 stock_zh_a_daily...")
            df = self._do_with_retry(
                self.ak.stock_zh_a_daily,
                symbol=f"sh{symbol}" if symbol.startswith(("6", "9")) else f"sz{symbol}",
                start_date=start,
                end_date=end,
                adjust=adjust,
            )

        if df is None or len(df) == 0:
            logger.warning("symbol=%s 未返回数据", symbol)
            return pd.DataFrame()

        df = _map_columns(df)

        # 添加派生列
        df["code"] = symbol
        if "name" in df.columns:
            df["is_st"] = df["name"].str.contains("ST", na=False)
        else:
            df["is_st"] = False

        # 标记停牌: volume==0 或 close没有变化
        if "volume" in df.columns and "close" in df.columns:
            df["is_suspended"] = (df["volume"] == 0) | (
                df["close"].diff() == 0
            )
        else:
            df["is_suspended"] = False

        # 确保date列是datetime类型
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        return df

    def batch_download(
        self,
        codes: List[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
        save: bool = True,
        show_progress: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        多线程批量下载日线数据

        Args:
            codes: 股票代码列表
            start: 起始日期
            end: 结束日期
            save: 是否保存为Parquet文件
            show_progress: 是否显示进度条

        Returns:
            Dict[str, DataFrame]: code->DataFrame映射
        """
        from tqdm import tqdm

        results: Dict[str, pd.DataFrame] = {}
        failed: List[str] = []

        logger.info("开始批量下载 %d 只股票日线数据...", len(codes))

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.download_daily_data, code, start, end, "qfq"): code
                for code in codes
            }

            pbar = tqdm(
                total=len(codes),
                desc="下载日线数据",
                disable=not show_progress,
                ncols=100,
            )
            for future in as_completed(futures):
                code = futures[future]
                try:
                    df = future.result()
                    if df is not None and len(df) > 0:
                        results[code] = df
                        if save:
                            self._save_parquet(code, df)
                    else:
                        failed.append(code)
                except Exception as e:
                    logger.error("下载失败 code=%s: %s", code, str(e)[:120])
                    failed.append(code)
                pbar.update(1)
            pbar.close()

        logger.info(
            "批量下载完成: 成功=%d, 失败=%d [%s]",
            len(results), len(failed), ",".join(failed[:10]) if failed else "无"
        )
        return results

    def _save_parquet(self, code: str, df: pd.DataFrame) -> str:
        """保存单只股票数据为Parquet"""
        filepath = self.parquet_dir / f"{code}.parquet"
        df.to_parquet(str(filepath), index=False)
        return str(filepath)

    # ==================== 指数数据 ====================

    def download_index_data(self, index_code: str = "000001") -> pd.DataFrame:
        """
        下载指数日线数据

        Args:
            index_code: 指数代码，如 "000001"=上证指数, "000300"=沪深300, "000905"=中证500

        Returns:
            DataFrame with standardized columns
        """
        logger.info("下载指数数据: %s", index_code)
        df = self._do_with_retry(
            self.ak.index_zh_a_hist,
            symbol=index_code,
            period="daily",
            start_date=START_DATE.replace("-", ""),
            end_date=datetime.now().strftime("%Y%m%d"),
        )
        df = _map_columns(df)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        df["code"] = index_code

        _ensure_dir(INDEX_DIR)
        save_path = Path(INDEX_DIR) / f"{index_code}.parquet"
        df.to_parquet(str(save_path), index=False)
        logger.info("指数 %s 已保存到 %s", index_code, save_path)
        return df

    # ==================== 行业数据 ====================

    def download_industry_data(self) -> pd.DataFrame:
        """
        下载行业分类数据

        Returns:
            DataFrame with columns: code, industry, industry_code
        """
        logger.info("正在下载行业分类数据...")
        try:
            df = self._do_with_retry(self.ak.stock_board_industry_name_em)
            df = _map_columns(df)
            _ensure_dir(INDUSTRY_DIR)
            save_path = Path(INDUSTRY_DIR) / "industry_classification.parquet"
            df.to_parquet(str(save_path), index=False)
            logger.info("行业分类数据已保存到 %s", save_path)
            return df
        except Exception as e:
            logger.warning("下载行业分类数据失败: %s", str(e)[:120])

            # fallback: 从东方财富获取
            try:
                df = self._do_with_retry(self.ak.stock_sector_detail)
                df = _map_columns(df)
                _ensure_dir(INDUSTRY_DIR)
                save_path = Path(INDUSTRY_DIR) / "industry_classification.parquet"
                df.to_parquet(str(save_path), index=False)
                logger.info("行业分类数据(fallback)已保存到 %s", save_path)
                return df
            except Exception as e2:
                logger.error("fallback下载行业分类数据也失败: %s", str(e2)[:120])
                return pd.DataFrame()

    # ==================== 交易日历 ====================

    def download_trading_calendar(self) -> pd.DataFrame:
        """
        下载交易日历

        Returns:
            DataFrame with column: trade_date
        """
        logger.info("正在下载交易日历...")
        try:
            df = self._do_with_retry(self.ak.tool_trade_date_hist_sina)
            if "trade_date" in df.columns:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
            _ensure_dir(REFERENCE_DIR)
            save_path = Path(REFERENCE_DIR) / "trade_calendar.parquet"
            df.to_parquet(str(save_path), index=False)
            logger.info("交易日历已保存到 %s, 共%d个交易日", save_path, len(df))
            return df
        except Exception as e:
            logger.error("下载交易日历失败: %s", str(e)[:120])
            return pd.DataFrame()


# 模块级便捷函数
def get_stock_list() -> pd.DataFrame:
    """便捷获取股票列表"""
    client = AKShareClient()
    return client.get_stock_list()


def batch_download(codes: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """便捷批量下载"""
    client = AKShareClient()
    return client.batch_download(codes, start, end)