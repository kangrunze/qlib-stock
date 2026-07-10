"""
AKShare API 封装客户端

提供多线程股票数据下载、重试机制、速率限制、中文列名映射、数据保存等功能。
"""

import os
import time
import random
import logging
import threading
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


# 模块级限流锁（跨线程共享，确保 _rate_limit 在多线程下真正串行）
_rate_limit_lock = threading.Lock()


def _rate_limit(min_interval: float = 0.3):
    """请求间速率限制（线程安全）。

    使用模块级 _rate_limit_lock 保护 _last_call 状态，
    确保多线程并发调用时真正按 min_interval 间隔串行执行，
    而非多个线程同时读写 _last_call 导致限流失效。
    """
    with _rate_limit_lock:
        now = time.time()
        elapsed = now - getattr(_rate_limit, "_last_call", 0.0)
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

        ⚠ 已知限制（生存者偏差）:
            本函数通过 ak.stock_info_a_code_name 获取的是「当前在市」的股票列表，
            历史上已退市的股票（如 600001 邯郸钢铁、000003 PT金田A 等）永远不会出现。
            这导致长周期（5年以上）回测会系统性高估收益，尤其是熊市尾部的退市股缺失，
            在解读回测结果时必须考虑此偏差。
            如需退市股列表，请调用 get_delisted_stock_list()。

        Returns:
            DataFrame, 列: code, name, is_st
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

    def get_delisted_stock_list(self) -> pd.DataFrame:
        """
        获取A股历史退市股票列表。

        通过 AKShare 的 stock_info_sh_delist（上交所退市）和 stock_info_sz_delist（深交所退市）
        两个接口合并获取。返回包含退市日期的 DataFrame，可用于补全生存者偏差缺失的数据。

        Returns:
            DataFrame, 列: code, name, list_date, delist_date, exchange
            任一接口失败时跳过该交易所并继续；两个接口都失败时返回空 DataFrame。
        """
        logger.info("正在获取A股退市股票列表...")
        frames = []

        # 上交所退市股
        try:
            df_sh = self._do_with_retry(self.ak.stock_info_sh_delist)
            if df_sh is not None and len(df_sh) > 0:
                df_sh = df_sh.rename(columns={
                    "公司代码": "code",
                    "公司简称": "name",
                    "上市日期": "list_date",
                    "暂停上市日期": "delist_date",
                })
                df_sh["code"] = df_sh["code"].astype(str).str.zfill(6)
                df_sh["exchange"] = "SSE"
                frames.append(df_sh)
                logger.info("上交所退市股: %d 只", len(df_sh))
        except Exception as e:
            logger.warning("获取上交所退市股失败: %s", str(e)[:120])

        # 深交所退市股
        try:
            df_sz = self._do_with_retry(self.ak.stock_info_sz_delist)
            if df_sz is not None and len(df_sz) > 0:
                df_sz = df_sz.rename(columns={
                    "证券代码": "code",
                    "证券简称": "name",
                    "上市日期": "list_date",
                    "终止上市日期": "delist_date",
                })
                df_sz["code"] = df_sz["code"].astype(str).str.zfill(6)
                df_sz["exchange"] = "SZSE"
                frames.append(df_sz)
                logger.info("深交所退市股: %d 只", len(df_sz))
        except Exception as e:
            logger.warning("获取深交所退市股失败: %s", str(e)[:120])

        if not frames:
            logger.warning(
                "当前数据源无法获取退市股列表（上交所和深交所接口均失败），"
                "此功能待接入其他数据源（如东方财富/同花顺）后实现"
            )
            return pd.DataFrame(columns=["code", "name", "list_date", "delist_date", "exchange"])

        result = pd.concat(frames, ignore_index=True)
        logger.info("退市股列表合计: %d 只", len(result))
        return result

    # ==================== 日线数据下载 ====================

    def download_daily_data(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        adjust: str = "hfq",
    ) -> pd.DataFrame:
        """
        下载单只股票日线数据（后复权）

        ⚠ 复权类型说明（2026-07-09 修复未来函数）:
            原默认 qfq（前复权）会以最新价格为锚点重写全部历史价格，
            每次有新的送股/分红事件，所有历史 K 线都会被改写，
            构成教科书级的未来函数（Look-ahead Bias）。
            现改为 hfq（后复权），以最早价格为锚向后调整，
            历史价格一经确定不再变化，消除未来函数。

        ⚠ is_st 标记的已知限制:
            is_st 基于股票当前名称判断（akshare 返回的 name 是当前最新名称），
            应用于整段历史数据。对于曾经ST后来摘帽（或反之）的股票，
            历史区间的 is_st 标记不准确。这是已知限制，
            待接入历史ST状态表后修正。

        Args:
            symbol: 股票代码，如 "000001"
            start: 起始日期 "YYYYMMDD" 或 "YYYY-MM-DD"
            end: 结束日期 "YYYYMMDD" 或 "YYYY-MM-DD"
            adjust: 复权类型，"hfq"=后复权（默认，无未来函数）, "qfq"=前复权（有未来函数，不推荐）, ""=不复权

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
            # is_st 基于当前名称判断整段历史（见 docstring 中的已知限制说明）
            df["is_st"] = df["name"].str.contains("ST", na=False)
        else:
            # 无 name 列时置 NaN，不硬编码 False
            df["is_st"] = pd.NA

        # 停牌判定：同时满足 volume==0 且 amount==0 才判定为停牌
        #   正常交易日即使收盘价和前一天一样，成交量和成交额几乎不可能同时为0
        #   退化方案：若无 amount 列，退回只用 volume==0
        if "volume" in df.columns:
            vol_zero = (df["volume"] == 0) | df["volume"].isna()
            if "amount" in df.columns:
                amt_zero = (df["amount"] == 0) | df["amount"].isna()
                df["is_suspended"] = vol_zero & amt_zero
            else:
                # 退化方案：无 amount 列，仅用 volume==0（可能误判平收日）
                df["is_suspended"] = vol_zero
        else:
            df["is_suspended"] = pd.NA

        # 涨跌停标记（2026-07-09 新增，供回测时过滤不可成交订单）
        #   判定逻辑：当日 (high == low == close) 且 close 相对前一日涨跌幅
        #   接近涨跌停阈值 → 一字板，买不到也卖不出
        #   注意：此处仅做粗略标记，精确判定需在回测引擎中按板块阈值检查
        if all(c in df.columns for c in ["open", "high", "low", "close", "preclose"]):
            pass  # preclose 可能在下游计算，此处跳过
        if all(c in df.columns for c in ["high", "low", "close"]):
            # 一字板：最高=最低=收盘，且当日有成交量（非停牌）
            is_one_price = (df["high"] == df["low"]) & (df["high"] == df["close"])
            df["is_limit"] = is_one_price & ~df["is_suspended"].fillna(True)
        else:
            df["is_limit"] = pd.NA

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
        save_raw: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        多线程批量下载日线数据

        同时下载后复权(hfq)和不复权(raw)两份数据：
          - hfq 用于因子计算（保存为 {code}.parquet，无未来函数）
          - raw  用于人工核查和后续复权因子计算（保存为 {code}_raw.parquet）

        Args:
            codes: 股票代码列表
            start: 起始日期
            end: 结束日期
            save: 是否保存为Parquet文件
            show_progress: 是否显示进度条
            save_raw: 是否同时下载并保存不复权数据（默认 True）

        Returns:
            Dict[str, DataFrame]: code->DataFrame映射（后复权数据）
        """
        from tqdm import tqdm

        results: Dict[str, pd.DataFrame] = {}
        failed: List[str] = []

        logger.info("开始批量下载 %d 只股票日线数据 (hfq%s)...",
                    len(codes), "+raw" if save_raw else "")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.download_daily_data, code, start, end, "hfq"): code
                for code in codes
            }
            # 若需要 raw，额外提交一批不复权下载任务
            raw_futures = {}
            if save_raw:
                raw_futures = {
                    executor.submit(self.download_daily_data, code, start, end, ""): code
                    for code in codes
                }

            pbar = tqdm(
                total=len(codes) + (len(codes) if save_raw else 0),
                desc="下载日线数据",
                disable=not show_progress,
                ncols=100,
            )
            # 收集 hfq 结果
            for future in as_completed(futures):
                code = futures[future]
                try:
                    df = future.result()
                    if df is not None and len(df) > 0:
                        results[code] = df
                        if save:
                            self._save_parquet(code, df, adjust="hfq")
                    else:
                        failed.append(code)
                except Exception as e:
                    logger.error("下载失败 code=%s (hfq): %s", code, str(e)[:120])
                    failed.append(code)
                pbar.update(1)
            # 收集 raw 结果
            for future in as_completed(raw_futures):
                code = raw_futures[future]
                try:
                    df = future.result()
                    if df is not None and len(df) > 0 and save:
                        self._save_parquet(code, df, adjust="")
                except Exception as e:
                    # raw 下载失败不影响主流程，仅记录 debug
                    logger.debug("下载不复权数据失败 code=%s: %s", code, str(e)[:120])
                pbar.update(1)
            pbar.close()

        logger.info(
            "批量下载完成: 成功=%d, 失败=%d [%s]",
            len(results), len(failed), ",".join(failed[:10]) if failed else "无"
        )
        return results

    def _save_parquet(self, code: str, df: pd.DataFrame, adjust: str = "hfq") -> str:
        """保存单只股票数据为Parquet。

        Args:
            code: 股票代码
            df: 日线 DataFrame
            adjust: 复权类型，"hfq"=后复权（默认，无未来函数）, ""=不复权。
                    hfq 保存为 {code}.parquet（用于因子计算）；
                    不复权保存为 {code}_raw.parquet（用于人工核查和后续复权因子计算）。
        """
        suffix = "" if adjust == "hfq" else "_raw"
        filepath = self.parquet_dir / f"{code}{suffix}.parquet"
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


def get_delisted_stock_list() -> pd.DataFrame:
    """便捷获取退市股票列表"""
    client = AKShareClient()
    return client.get_delisted_stock_list()


def batch_download(codes: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """便捷批量下载"""
    client = AKShareClient()
    return client.batch_download(codes, start, end)