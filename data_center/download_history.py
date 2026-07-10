#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全量A股历史数据初始化下载脚本

使用 akshare_client.py 进行多线程批量下载。
- 排除北交所（代码以8/4/920开头）、ST股票、上市不足250个交易日的股票
- 并行下载日线数据，保存为 {code}.parquet
- 下载指数数据、行业分类数据、股票列表
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 确保项目根目录在sys.path中
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
logger = logging.getLogger("download_history")

# ---- 加载配置 ----
_CONFIG_PATH = _PROJECT_ROOT / "config" / "settings.yaml"
with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
    _config = yaml.safe_load(f)

_paths = _config.get("paths", {})
_ds_conf = _config.get("data_source", {})
_univ_conf = _config.get("stock_universe", {})

PARQUET_DIR = Path(_paths.get("parquet_dir", "data/daily"))
INDEX_DIR = Path(_paths.get("index_dir", "data/index"))
INDUSTRY_DIR = Path(_paths.get("industry_dir", "data/industry"))
REFERENCE_DIR = Path(_paths.get("reference_dir", "data/reference"))
START_DATE = _ds_conf.get("start_date", "2005-01-01")
MAX_WORKERS = _ds_conf.get("max_workers", 10)
MIN_LISTED_DAYS = _univ_conf.get("min_listed_days", 250)
EXCLUDE_ST = _univ_conf.get("exclude_st", True)
EXCLUDE_BOARDS = _univ_conf.get("exclude_boards", ["北交所"])

# 北交所代码前缀
_BEJIAO_PREFIXES = ("8", "4", "920", "83", "87", "43")


def _is_beijiao(code: str) -> bool:
    """判断是否为北交所股票"""
    code = str(code).zfill(6)
    return any(code.startswith(p) for p in _BEJIAO_PREFIXES)


def _filter_stock_list(df: pd.DataFrame) -> pd.DataFrame:
    """
    过滤股票列表:
    1. 排除北交所
    2. ST 股票不再在下载阶段永久剔除，而是保留数据，在回测时按 PIT 状态动态过滤
       （2026-07-09 修复幸存者偏差：用当前 ST 状态过滤历史样本会偏向幸存者）
    3. 排除上市不足250个交易日的（通过快速检查每个code的最小日期）

    Args:
        df: 全量股票列表，至少含 code 列

    Returns:
        过滤后的DataFrame
    """
    logger.info("开始过滤股票列表，原始数量: %d", len(df))

    # 1. 排除北交所
    if EXCLUDE_BOARDS and "北交所" in EXCLUDE_BOARDS:
        mask_bj = df["code"].apply(_is_beijiao)
        n_bj = mask_bj.sum()
        df = df[~mask_bj]
        logger.info("排除北交所: -%d, 剩余: %d", n_bj, len(df))

    # 2. ST 处理（2026-07-09 修复幸存者偏差）
    #    原逻辑：在下载阶段用当前 ST 状态永久剔除 → 幸存者偏差
    #    新逻辑：保留 ST 股票的历史数据，仅在 is_st 列标记当前状态
    #    回测时由 DataHandler 按每日 PIT 状态动态过滤
    if EXCLUDE_ST:
        if "is_st" in df.columns:
            n_st_current = int(df["is_st"].sum())
            logger.info(
                "当前 ST 股票: %d 只（保留历史数据，回测时按 PIT 状态动态过滤）",
                n_st_current,
            )
        elif "name" in df.columns:
            n_st_current = int(df["name"].str.contains("ST", na=False).sum())
            df["is_st"] = df["name"].str.contains("ST", na=False)
            logger.info(
                "当前 ST 股票: %d 只（保留历史数据，回测时按 PIT 状态动态过滤）",
                n_st_current,
            )

    return df.reset_index(drop=True)


def _check_listed_days(client: AKShareClient, codes: List[str]) -> List[str]:
    """
    快速检查每个代码是否有足够的历史数据（>= MIN_LISTED_DAYS 条）
    通过简单下载头部数据而非全量数据来判断

    Args:
        client: AKShareClient 实例
        codes: 候选代码列表

    Returns:
        满足条件的代码列表
    """
    logger.info("正在检查上市天数 (min=%d天)...", MIN_LISTED_DAYS)

    valid_codes = []
    invalid_count = 0
    # 用较少线程做检查
    check_workers = min(5, MAX_WORKERS)

    with ThreadPoolExecutor(max_workers=check_workers) as executor:
        futures = {
            executor.submit(client.download_daily_data, code, START_DATE, None, "hfq"): code
            for code in codes
        }
        with tqdm(total=len(codes), desc="检查上市天数", ncols=100) as pbar:
            for future in as_completed(futures):
                code = futures[future]
                try:
                    df = future.result()
                    if df is not None and len(df) >= MIN_LISTED_DAYS:
                        valid_codes.append(code)
                    else:
                        invalid_count += 1
                except Exception:
                    invalid_count += 1
                pbar.update(1)

    logger.info("上市天数检查完成: 通过=%d, 不通过=%d", len(valid_codes), invalid_count)
    return valid_codes


def download_full_history(
    sample: Optional[int] = None,
    skip_index: bool = False,
    skip_industry: bool = False,
    workers: Optional[int] = None,
) -> None:
    """
    执行全量历史数据下载

    Args:
        sample: 仅下载前N只股票（用于测试）
        skip_index: 跳过指数数据下载
        skip_industry: 跳过行业数据下载
        workers: 并发线程数, 默认使用配置值
    """
    _workers = workers if workers is not None else MAX_WORKERS
    start_time = time.time()
    total_start = time.time()

    # ---- 确保目录存在 ----
    for d in [PARQUET_DIR, INDEX_DIR, INDUSTRY_DIR, REFERENCE_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    client = AKShareClient(max_workers=_workers)

    # ============ Step 1: 获取股票列表 ============
    logger.info("=" * 60)
    logger.info("Step 1/5: 获取股票列表")
    logger.info("=" * 60)

    stock_list = client.get_stock_list()
    logger.info("全量股票: %d 只", len(stock_list))

    # 过滤
    stock_list = _filter_stock_list(stock_list)
    codes = stock_list["code"].tolist()
    logger.info("过滤后候选股票: %d 只", len(codes))

    # 保存股票列表
    reference_path = REFERENCE_DIR / "stock_list.parquet"
    stock_list.to_parquet(str(reference_path), index=False)
    logger.info("股票列表已保存到 %s", reference_path)

    # Sample 模式
    if sample and sample > 0:
        codes = codes[:sample]
        logger.info("Sample模式: 仅下载前 %d 只股票", len(codes))

    # ============ Step 2: 上市天数检查 ============
    logger.info("=" * 60)
    logger.info("Step 2/5: 检查上市天数 (>=%d天)", MIN_LISTED_DAYS)
    logger.info("=" * 60)

    valid_codes = _check_listed_days(client, codes)
    logger.info("最终下载股票数: %d 只", len(valid_codes))

    # ============ Step 3: 批量下载日线数据 ============
    logger.info("=" * 60)
    logger.info("Step 3/5: 批量下载日线数据 (workers=%d)", _workers)
    logger.info("=" * 60)

    step_start = time.time()
    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=_workers) as executor:
        futures = {
            executor.submit(client.download_daily_data, code, START_DATE, None, "hfq"): code
            for code in valid_codes
        }
        with tqdm(total=len(valid_codes), desc="下载日线", ncols=100) as pbar:
            for future in as_completed(futures):
                code = futures[future]
                try:
                    df = future.result()
                    if df is not None and len(df) > 0:
                        parquet_path = PARQUET_DIR / f"{code}.parquet"
                        df.to_parquet(str(parquet_path), index=False)
                        success_count += 1
                    else:
                        fail_count += 1
                        logger.warning("code=%s 返回空数据", code)
                except Exception as e:
                    fail_count += 1
                    logger.error("code=%s 下载失败: %s", code, str(e)[:120])
                pbar.update(1)

    elapsed = time.time() - step_start
    logger.info("日线数据下载完成: 成功=%d, 失败=%d, 耗时=%.1f秒", success_count, fail_count, elapsed)

    # ============ Step 3b: 下载退市股历史数据（修复幸存者偏差） ============
    # 退市股必须保留其退市前的历史数据进入回测样本，
    # 否则组合在退市股上的损失会被完全掩盖
    logger.info("=" * 60)
    logger.info("Step 3b: 下载退市股历史数据（修复幸存者偏差）")
    logger.info("=" * 60)
    try:
        delist_df = client.get_delisted_stock_list()
        if not delist_df.empty:
            delist_codes = delist_df["code"].tolist()
            logger.info("退市股: %d 只，开始下载历史数据...", len(delist_codes))
            delist_success = 0
            delist_fail = 0
            with ThreadPoolExecutor(max_workers=_workers) as executor:
                delist_futures = {
                    executor.submit(
                        client.download_daily_data, code, START_DATE, None, "hfq"
                    ): code
                    for code in delist_codes
                    if not _is_beijiao(code)
                }
                with tqdm(total=len(delist_futures), desc="退市股下载", ncols=100) as pbar:
                    for future in as_completed(delist_futures):
                        code = delist_futures[future]
                        try:
                            df_delist = future.result()
                            if df_delist is not None and len(df_delist) > 0:
                                save_path = PARQUET_DIR / f"{code}.parquet"
                                df_delist.to_parquet(str(save_path), index=False)
                                delist_success += 1
                            else:
                                delist_fail += 1
                        except Exception as e:
                            delist_fail += 1
                            logger.debug("退市股 %s 下载失败: %s", code, str(e)[:80])
                        pbar.update(1)
            logger.info("退市股下载完成: 成功=%d, 失败=%d", delist_success, delist_fail)
        else:
            logger.warning("未获取到退市股列表，跳过退市股下载")
    except Exception as e:
        logger.warning("退市股下载流程异常: %s", str(e)[:120])

    # ============ Step 4: 下载指数数据 ============
    if not skip_index:
        logger.info("=" * 60)
        logger.info("Step 4/5: 下载指数数据")
        logger.info("=" * 60)

        index_codes = ["000001", "000300", "000905", "000016", "399001", "399006", "000852"]
        index_names = ["上证指数", "沪深300", "中证500", "上证50", "深证成指", "创业板指", "中证1000"]
        for idx_code, idx_name in zip(index_codes, index_names):
            try:
                client.download_index_data(idx_code)
                logger.info("  [OK] %s (%s)", idx_name, idx_code)
            except Exception as e:
                logger.error("  [FAIL] %s (%s): %s", idx_name, idx_code, str(e)[:100])

    # ============ Step 5: 下载行业数据 ============
    if not skip_industry:
        logger.info("=" * 60)
        logger.info("Step 5/5: 下载行业分类数据")
        logger.info("=" * 60)
        try:
            client.download_industry_data()
            logger.info("行业分类数据下载完成")
        except Exception as e:
            logger.error("行业分类数据下载失败: %s", str(e)[:120])

    # ============ 完成 ============
    total_elapsed = time.time() - total_start
    logger.info("=" * 60)
    logger.info("全量历史数据下载完成!")
    logger.info("  股票数量: %d", success_count)
    logger.info("  总耗时: %.1f 秒 (%.1f 分钟)", total_elapsed, total_elapsed / 60)
    logger.info("  数据目录: %s", PARQUET_DIR)
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="A股全量历史数据初始化下载")
    parser.add_argument(
        "--sample", "-s",
        type=int,
        default=None,
        help="仅下载前N只股票（用于测试），默认下载全部"
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="跳过指数数据下载"
    )
    parser.add_argument(
        "--skip-industry",
        action="store_true",
        help="跳过行业数据下载"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        help=f"并发线程数（默认: {MAX_WORKERS}）"
    )
    args = parser.parse_args()

    # 按参数调整worker数
    workers = args.workers if args.workers is not None else MAX_WORKERS

    download_full_history(
        sample=args.sample,
        skip_index=args.skip_index,
        skip_industry=args.skip_industry,
        workers=workers,
    )


if __name__ == "__main__":
    main()


    