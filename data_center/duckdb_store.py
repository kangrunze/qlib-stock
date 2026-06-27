"""
DuckDB 查询接口

基于 DuckDB 连接 Parquet 文件，提供高效的跨股票、跨日期查询能力。
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict

import pandas as pd
import yaml

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ---- 加载配置 ----
_CONFIG_PATH = _PROJECT_ROOT / "config" / "settings.yaml"
with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
    _config = yaml.safe_load(f)

_paths = _config.get("paths", {})
PARQUET_DIR = Path(_paths.get("parquet_dir", "data/daily"))
INDEX_DIR = Path(_paths.get("index_dir", "data/index"))
REFERENCE_DIR = Path(_paths.get("reference_dir", "data/reference"))


class DuckDBStore:
    """
    DuckDB 数据查询引擎

    通过 DuckDB 连接 Parquet 文件目录，提供 SQL 查询、行情快照、
    股票列表查询、交易日历查询等接口。

    优势: DuckDB 可以直接读取 Parquet 文件，无需导入内存，
         支持 SQL 查询和自动优化，非常适合此场景。

    Attributes:
        conn: DuckDB 连接对象
        parquet_dir: 日线数据Parquet目录
        index_dir: 指数数据目录
        reference_dir: 参考数据目录
    """

    def __init__(
        self,
        parquet_dir: Optional[str] = None,
        index_dir: Optional[str] = None,
        reference_dir: Optional[str] = None,
        db_path: Optional[str] = None,  # None = 内存模式
    ):
        self.parquet_dir = Path(parquet_dir) if parquet_dir else PARQUET_DIR
        self.index_dir = Path(index_dir) if index_dir else INDEX_DIR
        self.reference_dir = Path(reference_dir) if reference_dir else REFERENCE_DIR

        import duckdb

        self.db_path = db_path
        self.conn = duckdb.connect(database=db_path if db_path else ":memory:")
        self._views_registered = False

        logger.info(
            "DuckDBStore 初始化: db=%s, parquet_dir=%s",
            db_path or ":memory:", self.parquet_dir
        )

    def register_views(self) -> None:
        """
        注册 Parquet 视图

        将所有 Parquet 文件注册为 DuckDB 视图，包括:
        - daily_data: 所有日线数据的联合视图
        - stock_list_view: 股票列表视图
        """
        if self._views_registered:
            logger.debug("视图已注册，跳过")
            return

        # 注册日线视图 - 扫描所有parquet文件
        pattern = str(self.parquet_dir / "*.parquet")
        try:
            self.conn.execute(f"""
                CREATE OR REPLACE VIEW daily_data AS
                SELECT * FROM read_parquet('{pattern}', union_by_name=true)
            """)
            logger.info("已注册 daily_data 视图 (pattern=%s)", pattern)
        except Exception as e:
            logger.warning("注册 daily_data 视图失败: %s，可能无数据文件", str(e)[:100])
            # 尝试用 glob 方式
            parquet_files = sorted(self.parquet_dir.glob("*.parquet"))
            if parquet_files:
                file_list = [str(f).replace("\\", "/") for f in parquet_files]
                self.conn.execute(f"""
                    CREATE OR REPLACE VIEW daily_data AS
                    SELECT * FROM read_parquet({file_list}, union_by_name=true)
                """)
                logger.info("已注册 daily_data 视图 (%d 文件)", len(parquet_files))

        # 注册股票列表视图
        stock_list_path = self.reference_dir / "stock_list.parquet"
        if stock_list_path.exists():
            sp = str(stock_list_path).replace("\\", "/")
            self.conn.execute(f"""
                CREATE OR REPLACE VIEW stock_list_view AS
                SELECT * FROM read_parquet('{sp}')
            """)
            logger.info("已注册 stock_list_view 视图")

        # 注册指数视图
        index_files = sorted(self.index_dir.glob("*.parquet"))
        if index_files:
            ip = str(self.index_dir / "*.parquet").replace("\\", "/")
            try:
                self.conn.execute(f"""
                    CREATE OR REPLACE VIEW index_data AS
                    SELECT * FROM read_parquet('{ip}', union_by_name=true)
                """)
                logger.info("已注册 index_data 视图")
            except Exception as e:
                logger.warning("注册 index_data 视图失败: %s", str(e)[:100])

        self._views_registered = True

    def query(self, sql: str) -> pd.DataFrame:
        """
        直接执行SQL查询

        Args:
            sql: SQL查询语句

        Returns:
            DataFrame结果
        """
        self.register_views()
        result = self.conn.execute(sql).fetchdf()
        return result

    def get_stock_price(
        self,
        code: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        获取单只股票价格数据

        Args:
            code: 股票代码
            start: 起始日期 (YYYY-MM-DD)
            end: 结束日期 (YYYY-MM-DD)

        Returns:
            DataFrame with columns: date, open, high, low, close, volume, amount, etc.
        """
        self.register_views()

        conditions = [f"code = '{code}'"]
        if start:
            conditions.append(f"date >= '{start}'")
        if end:
            conditions.append(f"date <= '{end}'")

        where_clause = " AND ".join(conditions)
        sql = f"""
            SELECT *
            FROM daily_data
            WHERE {where_clause}
            ORDER BY date ASC
        """
        return self.conn.execute(sql).fetchdf()

    def get_market_snapshot(
        self,
        date: str,
        top_n: int = 50,
        sort_by: str = "pct_chg",
        order: str = "DESC",
    ) -> pd.DataFrame:
        """
        获取某日全市场快照

        Args:
            date: 日期 (YYYY-MM-DD)
            top_n: 返回前N只股票
            sort_by: 排序字段
            order: DESC/ASC

        Returns:
            当日行情快照DataFrame
        """
        self.register_views()

        valid_cols = {"pct_chg", "amount", "volume", "turnover_rate", "amplitude"}
        sort_col = sort_by if sort_by in valid_cols else "pct_chg"
        order_clause = order if order.upper() in ("ASC", "DESC") else "DESC"

        sql = f"""
            SELECT *
            FROM daily_data
            WHERE date = '{date}'
                AND is_suspended = false
                AND is_st = false
            ORDER BY {sort_col} {order_clause}
            LIMIT {top_n}
        """
        return self.conn.execute(sql).fetchdf()

    def get_stock_list(self) -> pd.DataFrame:
        """
        获取股票列表

        Returns:
            DataFrame with columns: code, name, is_st
        """
        reference_path = self.reference_dir / "stock_list.parquet"
        if reference_path.exists():
            return pd.read_parquet(str(reference_path))

        # 回退: 从daily_data中获取去重code列表
        self.register_views()
        sql = """
            SELECT DISTINCT code
            FROM daily_data
            ORDER BY code
        """
        return self.conn.execute(sql).fetchdf()

    def get_trade_calendar(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        获取交易日历

        Args:
            start: 起始日期 (YYYY-MM-DD)
            end: 结束日期 (YYYY-MM-DD)

        Returns:
            DataFrame with date column
        """
        # 优先从文件获取
        cal_path = self.reference_dir / "trade_calendar.parquet"
        if cal_path.exists():
            df = pd.read_parquet(str(cal_path))
            if "trade_date" in df.columns:
                df = df.rename(columns={"trade_date": "date"})
            elif "date" not in df.columns and len(df.columns) > 0:
                df = df.rename(columns={df.columns[0]: "date"})
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                if start:
                    df = df[df["date"] >= start]
                if end:
                    df = df[df["date"] <= end]
            return df

        # 回退: 从daily_data中获取去重日期
        self.register_views()
        conditions = []
        if start:
            conditions.append(f"date >= '{start}'")
        if end:
            conditions.append(f"date <= '{end}'")
        where = " AND ".join(conditions) if conditions else "1=1"
        sql = f"""
            SELECT DISTINCT date
            FROM daily_data
            WHERE {where}
            ORDER BY date
        """
        return self.conn.execute(sql).fetchdf()

    def get_cross_section(
        self,
        date: str,
        columns: Optional[List[str]] = None,
        exclude_st: bool = True,
        exclude_suspended: bool = True,
    ) -> pd.DataFrame:
        """
        获取某日期全市场截面数据

        Args:
            date: 日期 (YYYY-MM-DD)
            columns: 需要的列列表, None表示全部
            exclude_st: 是否排除ST
            exclude_suspended: 是否排除停牌

        Returns:
            截面DataFrame
        """
        self.register_views()

        select_cols = "*" if columns is None else ", ".join(columns)
        conditions = [f"date = '{date}'"]
        if exclude_st:
            conditions.append("is_st = false")
        if exclude_suspended:
            conditions.append("is_suspended = false")

        where_clause = " AND ".join(conditions)
        sql = f"""
            SELECT {select_cols}
            FROM daily_data
            WHERE {where_clause}
            ORDER BY code
        """
        return self.conn.execute(sql).fetchdf()

    def get_price_history_multi(
        self,
        codes: List[str],
        start: str,
        end: str,
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        批量获取多只股票历史价格

        Args:
            codes: 股票代码列表
            start: 起始日期
            end: 结束日期
            fields: 需要的字段列表

        Returns:
            Multi-index (code, date) DataFrame
        """
        self.register_views()

        select_cols = ", ".join(fields) if fields else "*"
        code_list = ", ".join([f"'{c}'" for c in codes])
        sql = f"""
            SELECT {select_cols}
            FROM daily_data
            WHERE code IN ({code_list})
                AND date >= '{start}'
                AND date <= '{end}'
            ORDER BY code, date
        """
        return self.conn.execute(sql).fetchdf()

    def get_statistics(
        self,
        start: str,
        end: str,
        code: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        获取统计摘要

        Args:
            start: 起始日期
            end: 结束日期
            code: 指定股票代码, None表示全市场

        Returns:
            统计DataFrame
        """
        self.register_views()

        code_filter = f"AND code = '{code}'" if code else ""
        sql = f"""
            SELECT
                code,
                COUNT(*) as trading_days,
                MIN(date) as first_date,
                MAX(date) as last_date,
                AVG(close) as avg_close,
                STDDEV(close) as std_close,
                SUM(volume) as total_volume,
                AVG(turnover_rate) as avg_turnover,
                AVG(pct_chg) as avg_pct_chg
            FROM daily_data
            WHERE date >= '{start}'
                AND date <= '{end}'
                {code_filter}
            GROUP BY code
            ORDER BY code
        """
        return self.conn.execute(sql).fetchdf()

    def close(self):
        """关闭DuckDB连接"""
        if self.conn:
            self.conn.close()
            logger.info("DuckDB 连接已关闭")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# 模块级便捷函数
def query(sql: str) -> pd.DataFrame:
    """便捷执行SQL查询"""
    with DuckDBStore() as store:
        return store.query(sql)


def get_market_snapshot(date: str, top_n: int = 50) -> pd.DataFrame:
    """便捷获取市场快照"""
    with DuckDBStore() as store:
        return store.get_market_snapshot(date, top_n)