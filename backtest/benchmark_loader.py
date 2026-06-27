"""
基准数据加载模块 - 加载指数数据作为回测基准

支持:
    - CSI 300 (000300) - 沪深300
    - CSI 500 (000905) - 中证500
    - 通用指数加载和对比

使用方法:
    loader = BenchmarkLoader(data_dir="data/index")
    benchmark_equity = loader.load_benchmark("000905", "2024-01-01", "2025-06-30")
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple, List

logger = logging.getLogger(__name__)

# 常用基准指数代码
BENCHMARK_CODES = {
    "CSI300": "000300",
    "CSI500": "000905",
    "CSI1000": "000852",
    "SSE50": "000016",
    "SZ50": "399550",
}

# 基准指数名称
BENCHMARK_NAMES = {
    "000300": "沪深300",
    "000905": "中证500",
    "000852": "中证1000",
    "000016": "上证50",
    "399550": "深证50",
}


class BenchmarkLoader:
    """基准指数数据加载器。

    从本地Parquet/CSV文件加载指数日线数据，计算净值曲线。

    Attributes:
        data_dir: 指数数据存储目录。
    """

    def __init__(self, data_dir: str = "data/index"):
        """初始化基准加载器。

        Args:
            data_dir: 指数数据目录路径。
        """
        self.data_dir = Path(data_dir)

    def load_benchmark(
        self,
        benchmark_code: str,
        start_date: str,
        end_date: str,
        data_dir: Optional[str] = None,
    ) -> pd.DataFrame:
        """加载基准指数数据并计算净值曲线。

        首先尝试从本地文件加载，如果文件不存在则返回模拟数据。

        Args:
            benchmark_code: 基准指数代码（如 "000905" 或 "CSI500"）。
            start_date: 起始日期 (YYYY-MM-DD)。
            end_date: 结束日期 (YYYY-MM-DD)。
            data_dir: 数据目录（覆盖实例默认值）。

        Returns:
            DataFrame，包含以下列:
            - date (index): 日期
            - close: 收盘价
            - equity: 净值（以start_date价格为1.0归一化）
            - daily_return: 日收益率
        """
        # 解析代码别名
        code = BENCHMARK_CODES.get(benchmark_code, benchmark_code)

        search_dir = Path(data_dir) if data_dir else self.data_dir

        logger.info(f"加载基准指数 {code} ({BENCHMARK_NAMES.get(code, '')}): {start_date} ~ {end_date}")

        # 尝试多种文件命名模式加载
        df = self._try_load_file(search_dir, code, start_date, end_date)

        if df is not None:
            return self._compute_equity_curve(df, start_date)

        # 尝试通过akshare在线获取
        df = self._try_akshare_load(code, start_date, end_date)

        if df is not None:
            return self._compute_equity_curve(df, start_date)

        # 最后回退：生成模拟基准数据
        logger.warning(f"无法加载指数 {code} 数据，使用模拟数据")
        return self._generate_fallback_benchmark(code, start_date, end_date)

    def _try_load_file(
        self, data_dir: Path, code: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """尝试从本地文件加载指数数据。

        Args:
            data_dir: 数据目录。
            code: 指数代码。
            start_date: 起始日期。
            end_date: 结束日期。

        Returns:
            加载的DataFrame或None。
        """
        # 尝试不同的文件模式
        patterns = [
            f"{code}.parquet",
            f"index_{code}.parquet",
            f"{code}.csv",
            f"index_{code}.csv",
            f"{code}_daily.parquet",
        ]

        for pattern in patterns:
            file_path = data_dir / pattern
            if file_path.exists():
                try:
                    if pattern.endswith(".parquet"):
                        df = pd.read_parquet(file_path)
                    else:
                        df = pd.read_csv(file_path)

                    # 标准化列名
                    df.columns = [c.lower() for c in df.columns]

                    # 确保有date列
                    date_col_candidates = ["date", "trade_date", "日期", "datetime"]
                    date_col = None
                    for cand in date_col_candidates:
                        if cand in df.columns:
                            date_col = cand
                            break
                        elif cand.lower() in df.columns:
                            date_col = cand.lower()
                            break

                    if date_col is None and df.index.name in date_col_candidates:
                        df = df.reset_index()
                        date_col = df.index.name

                    if date_col is None:
                        # 尝试第一列
                        date_col = df.columns[0]

                    df[date_col] = pd.to_datetime(df[date_col])

                    # 确保有close列
                    close_col_candidates = ["close", "收盘价", "adj_close", "close_price"]
                    close_col = None
                    for cand in close_col_candidates:
                        if cand in df.columns:
                            close_col = cand
                            break

                    if close_col is None:
                        # 取数值列作为收盘价
                        for col in df.columns:
                            if col != date_col and pd.api.types.is_numeric_dtype(df[col]):
                                close_col = col
                                break

                    if close_col is None:
                        logger.warning(f"文件 {file_path} 无可用收盘价列")
                        continue

                    df = df[[date_col, close_col]].copy()
                    df.columns = ["date", "close"]
                    df = df.set_index("date")
                    df = df.sort_index()

                    mask = (df.index >= start_date) & (df.index <= end_date)
                    df = df[mask]

                    if not df.empty:
                        logger.info(f"从文件加载: {file_path}, {len(df)}条记录")
                        return df
                except Exception as e:
                    logger.warning(f"读取文件 {file_path} 失败: {e}")

        return None

    def _try_akshare_load(
        self, code: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """尝试通过akshare在线加载指数数据。

        Args:
            code: 指数代码。
            start_date: 起始日期。
            end_date: 结束日期。

        Returns:
            加载的DataFrame或None。
        """
        try:
            import akshare as ak

            # 尝试获取指数日线
            index_code_map = {
                "000300": "sh000300",
                "000905": "sh000905",
                "000852": "sh000852",
                "000016": "sh000016",
                "399550": "sz399550",
            }

            symbol = index_code_map.get(code, f"sh{code}")

            df = ak.stock_zh_index_daily(symbol=symbol)
            df["date"] = pd.to_datetime(df["date"])
            df = df.rename(columns={"close": "close"})
            df = df.set_index("date").sort_index()

            mask = (df.index >= start_date) & (df.index <= end_date)
            df = df[mask]

            if not df.empty:
                logger.info(f"通过akshare在线加载: {code}, {len(df)}条记录")
                return df
        except Exception as e:
            logger.warning(f"akshare加载指数 {code} 失败: {e}")

        return None

    def _compute_equity_curve(self, df: pd.DataFrame, start_date: str) -> pd.DataFrame:
        """计算归一化的净值曲线。

        Args:
            df: 包含date(index)和close列的DataFrame。
            start_date: 基准日期。

        Returns:
            添加了equity和daily_return列的DataFrame。
        """
        df = df.sort_index().copy()

        # 设置起始基准价
        try:
            start_mask = df.index >= pd.Timestamp(start_date)
            if start_mask.any():
                first_close = df.loc[start_mask, "close"].iloc[0]
            elif len(df) > 0:
                first_close = df["close"].iloc[0]
            else:
                first_close = 1.0
        except Exception:
            first_close = 1.0

        if first_close and first_close > 0:
            df["equity"] = df["close"] / first_close
        else:
            df["equity"] = 1.0

        df["daily_return"] = df["close"].pct_change()
        df.iloc[0, df.columns.get_loc("daily_return")] = 0.0

        return df

    def _generate_fallback_benchmark(
        self, code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """当无法获取真实数据时，生成模拟基准净值曲线。

        使用年化8%的增长率加上随机波动模拟指数表现。
        仅在数据缺失时作为最后手段，回测结果仅供参考。

        Args:
            code: 指数代码。
            start_date: 起始日期。
            end_date: 结束日期。

        Returns:
            模拟的基准DataFrame。
        """
        logger.info(f"生成模拟基准数据: {code}")

        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        dates = pd.date_range(start_dt, end_dt, freq="B")

        n_days = len(dates)
        if n_days == 0:
            n_days = 100

        # 年化8%收益，日波动1.5%
        daily_mu = 0.08 / 252
        daily_sigma = 0.015
        np.random.seed(42)
        daily_returns = np.random.normal(daily_mu, daily_sigma, n_days)

        equity = np.cumprod(1 + daily_returns)
        close = 1000 * equity

        df = pd.DataFrame(
            {
                "date": dates,
                "close": close,
                "equity": equity,
                "daily_return": daily_returns,
            }
        )
        df = df.set_index("date")

        logger.warning(
            f"模拟基准数据已生成: {len(df)}条记录，"
            f"年化收益约为 {daily_mu * 252:.1%}。"
            f"请尽快下载真实指数数据以获得准确回测结果。"
        )
        return df


def compare_to_benchmark(
    strategy_equity: pd.Series,
    benchmark_equity: pd.Series,
) -> Dict[str, float]:
    """将策略净值与基准净值进行对比分析。

    Args:
        strategy_equity: 策略净值序列（index为日期）。
        benchmark_equity: 基准净值序列（index为日期）。

    Returns:
        包含对比指标的字典:
        - excess_return: 超额收益
        - tracking_error: 跟踪误差
        - information_ratio: 信息比率
        - strategy_return: 策略总收益
        - benchmark_return: 基准总收益
    """
    # 对齐日期
    common_dates = strategy_equity.index.intersection(benchmark_equity.index)

    if len(common_dates) < 2:
        logger.warning("策略与基准无有效重叠日期")
        return {
            "excess_return": 0.0,
            "tracking_error": 0.0,
            "information_ratio": 0.0,
            "strategy_return": 0.0,
            "benchmark_return": 0.0,
        }

    s_equity = strategy_equity[common_dates]
    b_equity = benchmark_equity[common_dates]

    # 策略收益
    strategy_return = s_equity.iloc[-1] / s_equity.iloc[0] - 1

    # 基准收益
    benchmark_return = b_equity.iloc[-1] / b_equity.iloc[0] - 1

    # 超额收益
    excess_return = strategy_return - benchmark_return

    # 日收益率
    s_daily = s_equity.pct_change().dropna()
    b_daily = b_equity.pct_change().dropna()

    # 对齐日收益率
    common_dates_daily = s_daily.index.intersection(b_daily.index)
    s_daily = s_daily[common_dates_daily]
    b_daily = b_daily[common_dates_daily]

    # 超额日收益
    excess_daily = s_daily - b_daily

    # 跟踪误差（年化）
    tracking_error = excess_daily.std() * np.sqrt(252) if len(excess_daily) > 0 else 0.0

    # 信息比率
    if tracking_error > 0:
        information_ratio = excess_daily.mean() * 252 / tracking_error
    else:
        information_ratio = 0.0

    result = {
        "strategy_return": round(strategy_return, 6),
        "benchmark_return": round(benchmark_return, 6),
        "excess_return": round(excess_return, 6),
        "tracking_error": round(tracking_error, 6),
        "information_ratio": round(information_ratio, 4),
    }

    logger.info(
        f"基准对比: 策略收益={strategy_return:.2%}, 基准收益={benchmark_return:.2%}, "
        f"超额={excess_return:.2%}, 信息比率={information_ratio:.2f}"
    )

    return result