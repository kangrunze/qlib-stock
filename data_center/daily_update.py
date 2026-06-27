"""
日度增量更新脚本

- 下载最近N天数据覆盖已有Parquet文件（应对复权因子修正）
- 更新股票列表状态（新上市、ST变更、退市）
- 支持定时调度和手动触发
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import pandas as pd
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data_center.akshare_client import AKShareClient

# ---- 日志配置 ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("daily_update")

# ---- 加载配置 ----
_CONFIG_PATH = _PROJECT_ROOT / "config" / "settings.yaml"
with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
    _config = yaml.safe_load(f)

_paths = _config.get("paths", {})
_ds_conf = _config.get("data_source", {})

PARQUET_DIR = Path(_paths.get("parquet_dir", "data/daily"))
REFERENCE_DIR = Path(_paths.get("reference_dir", "data/reference"))
INCREMENTAL_WINDOW = _ds_conf.get("incremental_window", 30)
MAX_WORKERS = _ds_conf.get("max_workers", 10)
RETRY_MAX = _ds_conf.get("retry_max", 3)
RETRY_DELAY = _ds_conf.get("retry_delay", 2)


class DailyUpdater:
    """
    日度增量更新器

    负责每日自动更新股票行情数据，包括：
    - 获取最新N天的行情数据（默认30天，应对复权修正）
    - 合并已有历史数据，更新Parquet文件
    - 更新股票列表状态

    Attributes:
        parquet_dir: 日线数据Parquet目录
        reference_dir: 参考数据目录
        incremental_window: 增量更新窗口（天数）
        max_workers: 并发下载线程数
    """

    def __init__(
        self,
        parquet_dir: Optional[str] = None,
        incremental_window: int = INCREMENTAL_WINDOW,
        max_workers: int = MAX_WORKERS,
    ):
        self.parquet_dir = Path(parquet_dir) if parquet_dir else PARQUET_DIR
        self.reference_dir = REFERENCE_DIR
        self.incremental_window = incremental_window
        self.max_workers = max_workers
        self.client = AKShareClient(max_workers=max_workers)

        # 确保目录存在
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        self.reference_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "DailyUpdater 初始化: window=%d天, workers=%d, parquet_dir=%s",
            incremental_window, max_workers, self.parquet_dir
        )

    def update_stock_list(self) -> pd.DataFrame:
        """
        更新股票列表状态

        Returns:
            更新后的股票列表 DataFrame
        """
        logger.info("正在更新股票列表...")

        new_list = self.client.get_stock_list()
        reference_path = self.reference_dir / "stock_list.parquet"

        if reference_path.exists():
            old_list = pd.read_parquet(str(reference_path))
            old_codes = set(old_list["code"].tolist())
            new_codes = set(new_list["code"].tolist())

            added = new_codes - old_codes
            removed = old_codes - new_codes

            if added:
                logger.info("  新上市股票: %d 只: %s", len(added), ", ".join(sorted(added)[:20]))
            if removed:
                logger.info("  可能退市股票: %d 只: %s", len(removed), ", ".join(sorted(removed)[:20]))
            if not added and not removed:
                logger.info("  股票列表无变化")
        else:
            logger.info("  首次创建股票列表")

        # 保存最新列表
        new_list.to_parquet(str(reference_path), index=False)
        logger.info("  股票列表已保存到 %s (%d只)", reference_path, len(new_list))
        return new_list

    def update_daily_data(self) -> Dict[str, int]:
        """
        更新所有股票的日线数据

        下载最近 incremental_window 天的数据，与已有历史数据合并后覆盖保存。

        Returns:
            Dict[str, int]: {"success": N, "failed": N, "skipped": N}
        """
        # 读取现有股票列表
        reference_path = self.reference_dir / "stock_list.parquet"
        if not reference_path.exists():
            logger.warning("股票列表不存在，先进行更新")
            self.update_stock_list()

        stock_list = pd.read_parquet(str(reference_path))
        codes = stock_list["code"].tolist()
        logger.info("准备更新 %d 只股票的日线数据 (增量窗口=%d天)", len(codes), self.incremental_window)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.incremental_window + 10)
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")

        results = {"success": 0, "failed": 0, "updated_rows": 0}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._update_single_stock, code, start_str, end_str
                ): code for code in codes
            }
            with tqdm(total=len(codes), desc="增量更新日线", ncols=100) as pbar:
                for future in as_completed(futures):
                    code = futures[future]
                    try:
                        status, rows = future.result()
                        if status == "success":
                            results["success"] += 1
                            results["updated_rows"] += rows
                        elif status == "failed":
                            results["failed"] += 1
                        else:
                            pass  # skipped
                    except Exception as e:
                        logger.error("code=%s 更新异常: %s", code, str(e)[:120])
                        results["failed"] += 1
                    pbar.update(1)

        logger.info(
            "日线更新完成: 成功=%d, 失败=%d, 新增行数=%d",
            results["success"], results["failed"], results["updated_rows"]
        )
        return results

    def _update_single_stock(
        self, code: str, start: str, end: str
    ) -> tuple:
        """
        更新单只股票的日线数据

        下载最新数据，与已有历史合并去重后保存。

        Args:
            code: 股票代码
            start: 起始日期 (YYYYMMDD)
            end: 结束日期 (YYYYMMDD)

        Returns:
            (status, new_rows): status in ("success", "failed", "skipped", "no_new_data")
        """
        parquet_path = self.parquet_dir / f"{code}.parquet"

        try:
            new_df = self.client.download_daily_data(code, start, end, "qfq")
            if new_df is None or len(new_df) == 0:
                return ("no_new_data", 0)

            if parquet_path.exists():
                old_df = pd.read_parquet(str(parquet_path))
                # 合并去重: 以date为基准，新的覆盖旧的
                old_df["_src"] = "old"
                new_df["_src"] = "new"
                combined = pd.concat([old_df, new_df], ignore_index=True)

                if "date" not in combined.columns:
                    return ("failed", 0)

                # 按date+code去重，保留new的
                combined = combined.sort_values(["_src", "date"], ascending=[False, True])
                combined = combined.drop_duplicates(subset=["code", "date"], keep="first")
                combined = combined.drop(columns=["_src"])

                old_rows = len(old_df) - 1  # 减去_src
                new_rows = len(combined) - old_rows
            else:
                combined = new_df
                new_rows = len(combined)

            # 排序后保存
            if "date" in combined.columns:
                combined = combined.sort_values("date")
            combined.to_parquet(str(parquet_path), index=False)

            return ("success", max(0, new_rows))
        except Exception as e:
            logger.error("code=%s 更新失败: %s", code, str(e)[:100])
            return ("failed", 0)

    def run_full_update(self) -> None:
        """执行完整日度更新流程"""
        start_time = time.time()

        logger.info("=" * 60)
        logger.info("开始日度增量更新, 日期: %s", datetime.now().strftime("%Y-%m-%d"))
        logger.info("=" * 60)

        # 1. 更新股票列表
        logger.info("Phase 1/2: 更新股票列表")
        self.update_stock_list()

        # 2. 更新日线数据
        logger.info("Phase 2/2: 更新日线数据")
        daily_result = self.update_daily_data()

        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info("日度更新完成!")
        logger.info("  耗时: %.1f 秒", elapsed)
        logger.info("  日线更新: 成功=%d, 失败=%d", daily_result["success"], daily_result["failed"])
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="日度增量数据更新")
    parser.add_argument(
        "--window", "-w",
        type=int,
        default=INCREMENTAL_WINDOW,
        help=f"增量更新窗口天数（默认: {INCREMENTAL_WINDOW}）"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help=f"并发线程数（默认: {MAX_WORKERS}）"
    )
    parser.add_argument(
        "--skip-stock-list",
        action="store_true",
        help="跳过股票列表更新"
    )
    args = parser.parse_args()

    updater = DailyUpdater(
        incremental_window=args.window,
        max_workers=args.workers,
    )

    if args.skip_stock_list:
        updater.update_daily_data()
    else:
        updater.run_full_update()


if __name__ == "__main__":
    main()