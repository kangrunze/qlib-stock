#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
每日增量更新脚本

功能：
1. 增量下载最近 N 天的新数据（默认 3 天）
2. 增量更新 CSV 文件
3. 增量更新 Qlib bin 文件（追加新日期）
4. 更新 calendars 和 instruments 文件
5. 更新 workflow_config.yaml 的 end_time

用法：
  python run.py update                    # 更新最近 3 天
  python run.py update --days 7           # 更新最近 7 天
  python run.py update --skip-bin         # 只更新 CSV，不更新 bin
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Set, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import yaml
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
logger = logging.getLogger("daily_update")

# ---- 加载配置 ----
_CONFIG_PATH = _PROJECT_ROOT / "config" / "settings.yaml"
with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
    _config = yaml.safe_load(f)

_paths = _config.get("paths", {})
_ds_conf = _config.get("data_source", {})

CSV_DIR = Path(_ds_conf.get("csv_dir", "D:/data"))
QLIB_DIR = Path("D:/trae/qlib_bin")  # 固定为实际数据路径
MAX_WORKERS = _ds_conf.get("max_workers", 10)

# 北交所代码前缀
_BEIJIAO_PREFIXES = ("8", "4", "920", "83", "87", "43")


def _is_beijiao(code: str) -> bool:
    """判断是否为北交所股票"""
    code = str(code).zfill(6)
    return any(code.startswith(p) for p in _BEIJIAO_PREFIXES)


def get_existing_csv_files() -> Dict[str, Path]:
    """获取所有已存在的 CSV 文件"""
    if not CSV_DIR.exists():
        return {}
    
    files = {}
    for f in CSV_DIR.glob("*.csv"):
        code = f.stem
        files[code] = f
    
    return files


def get_existing_qlib_stocks_from_features() -> Set[str]:
    """从 features 目录发现股票列表"""
    features_dir = QLIB_DIR / "features"
    if not features_dir.exists():
        return set()
    
    stocks = set()
    for stock_dir in features_dir.iterdir():
        if stock_dir.is_dir() and (stock_dir / "close.day.bin").exists():
            stocks.add(stock_dir.name)
    
    return stocks


def get_existing_qlib_dates() -> Set[str]:
    """获取 Qlib calendars 中已存在的日期"""
    calendar_path = QLIB_DIR / "calendars" / "day.txt"
    if not calendar_path.exists():
        return set()
    
    with open(calendar_path, "r", encoding="utf-8") as f:
        dates = set(line.strip() for line in f if line.strip())
    
    return dates


def get_existing_qlib_stocks() -> Dict[str, Dict[str, str]]:
    """获取 Qlib instruments 中已存在的股票信息"""
    instruments_path = QLIB_DIR / "instruments" / "all.txt"
    if not instruments_path.exists():
        return {}
    
    stocks = {}
    with open(instruments_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                code, start_date, end_date = parts[0], parts[1], parts[2]
                stocks[code] = {"start": start_date, "end": end_date}
    
    return stocks


def download_incremental_data(
    stock_codes: Set[str],
    days: int = 3,
    workers: int = MAX_WORKERS,
) -> Dict[str, pd.DataFrame]:
    """
    增量下载最近 N 天的新数据
    
    Args:
        stock_codes: 股票代码集合
        days: 下载最近多少天的数据
        workers: 并发线程数
    
    Returns:
        新下载的 DataFrame 字典 {code: df}
    """
    client = AKShareClient(max_workers=workers)
    
    # 计算起始日期
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    start_date_str = start_date.strftime("%Y-%m-%d")
    
    logger.info("=" * 60)
    logger.info("增量下载数据: %s ~ %s", start_date_str, end_date.strftime("%Y-%m-%d"))
    logger.info("=" * 60)
    
    new_data = {}
    fail_count = 0
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                client.download_daily_data, code, start_date_str, None, "qfq"
            ): code
            for code in stock_codes
            if not _is_beijiao(code)
        }
        
        with tqdm(total=len(futures), desc="增量下载", ncols=100) as pbar:
            for future in as_completed(futures):
                code = futures[future]
                try:
                    df = future.result()
                    if df is not None and len(df) > 0:
                        new_data[code] = df
                except Exception as e:
                    fail_count += 1
                    logger.error("code=%s 下载失败: %s", code, str(e)[:120])
                pbar.update(1)
    
    logger.info("增量下载完成: 成功=%d, 失败=%d", len(new_data), fail_count)
    return new_data


def update_csv_files(
    existing_files: Dict[str, Path],
    new_data: Dict[str, pd.DataFrame],
) -> int:
    """
    增量更新 CSV 文件
    
    Args:
        existing_files: 已存在的 CSV 文件映射
        new_data: 新下载的 DataFrame
    
    Returns:
        更新的股票数量
    """
    logger.info("=" * 60)
    logger.info("增量更新 CSV 文件")
    logger.info("=" * 60)
    
    updated_count = 0
    
    for code, new_df in tqdm(new_data.items(), desc="更新 CSV", ncols=100):
        if code not in existing_files:
            continue
        
        csv_path = existing_files[code]
        
        try:
            # 读取现有数据
            if csv_path.exists():
                old_df = pd.read_csv(csv_path)
                # 合并数据（去重）
                combined_df = pd.concat([old_df, new_df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=["date"], keep="last")
                combined_df = combined_df.sort_values("date").reset_index(drop=True)
            else:
                combined_df = new_df
            
            # 保存
            combined_df.to_csv(csv_path, index=False)
            updated_count += 1
            
        except Exception as e:
            logger.error("更新 %s 失败: %s", code, str(e)[:120])
    
    logger.info("CSV 更新完成: %d 只股票", updated_count)
    return updated_count


def update_bin_file(
    bin_path: Path,
    date_value_pairs: List[tuple],
    new_date_index: Dict[str, int],
) -> None:
    """
    更新 bin 文件：扩展日历长度并写入新数据
    
    bin 文件格式: [start_date_idx, val_0, val_1, ..., val_{N-1}]
    其中 start_date_idx 是该股票第一个数据日在全局日历中的索引,
    val_i 对应全局日历中第 (start_date_idx + i) 个日期的值。
    
    Args:
        bin_path: bin 文件路径
        date_value_pairs: [(date_str, value), ...] 需要写入的新数据
        new_date_index: {date_str: global_index} 新的全局日历索引
    """
    if not bin_path.exists():
        return
    
    # 读取现有数据
    with open(bin_path, "rb") as f:
        old_data = np.fromfile(f, dtype="<f")
    
    if len(old_data) == 0:
        return
    
    # 第一个值是起始日期索引
    start_idx = int(old_data[0])
    old_values = old_data[1:]
    old_len = len(old_values)
    
    # 计算新的总长度（基于新日历中最大日期索引 + 1）
    max_idx = old_len  # 至少保持原长度
    for dt_str, _ in date_value_pairs:
        if dt_str in new_date_index:
            global_idx = new_date_index[dt_str]
            local_idx = global_idx - start_idx
            if local_idx >= 0:
                max_idx = max(max_idx, local_idx + 1)
    
    # 创建新数组，复制旧数据
    new_full_values = np.full(max_idx, np.nan, dtype=np.float32)
    new_full_values[:old_len] = old_values
    
    # 写入新数据
    for dt_str, val in date_value_pairs:
        if dt_str in new_date_index:
            global_idx = new_date_index[dt_str]
            local_idx = global_idx - start_idx
            if 0 <= local_idx < max_idx:
                new_full_values[local_idx] = val
    
    # 写入文件
    bin_data = np.hstack([np.float32(start_idx), new_full_values]).astype("<f")
    with open(bin_path, "wb") as f:
        bin_data.tofile(f)


def update_qlib_bin_files(
    new_data: Dict[str, pd.DataFrame],
    old_dates: Set[str],
    new_dates: List[str],
) -> int:
    """
    增量更新 Qlib bin 文件
    
    Args:
        new_data: 新下载的 DataFrame
        old_dates: 旧的日历日期集合
        new_dates: 新的日历日期列表
    
    Returns:
        更新的股票数量
    """
    logger.info("=" * 60)
    logger.info("增量更新 Qlib bin 文件")
    logger.info("=" * 60)
    
    features_dir = QLIB_DIR / "features"
    if not features_dir.exists():
        logger.error("Qlib features 目录不存在: %s", features_dir)
        return 0
    
    # 计算新增日期数量
    new_only_dates = [d for d in new_dates if d not in old_dates]
    if not new_only_dates:
        logger.info("没有新增日期，跳过 bin 更新")
        return 0
    
    logger.info("新增日期: %d 天 (%s ~ %s)", 
                len(new_only_dates), new_only_dates[0], new_only_dates[-1])
    
    # 构建新的全局日期索引
    new_date_index = {d: i for i, d in enumerate(new_dates)}
    
    updated_count = 0
    
    # 字段映射
    field_mapping = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "amount": "amount",
    }
    
    for code, df in tqdm(new_data.items(), desc="更新 bin", ncols=100):
        stock_dir = features_dir / code
        if not stock_dir.exists():
            continue
        
        # 准备日期-值映射
        date_value_map = {}
        for _, row in df.iterrows():
            dt_str = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
            date_value_map[dt_str] = row
        
        # 更新每个字段的 bin 文件
        for parquet_field, bin_field in field_mapping.items():
            if parquet_field not in df.columns:
                continue
            
            bin_path = stock_dir / f"{bin_field}.day.bin"
            if not bin_path.exists():
                continue
            
            # 提取需要写入的日期-值对
            date_value_pairs = []
            for dt in new_only_dates:
                if dt in date_value_map:
                    val = date_value_map[dt][parquet_field]
                    if pd.notna(val):
                        date_value_pairs.append((dt, float(val)))
                    else:
                        date_value_pairs.append((dt, np.nan))
            
            # 更新 bin 文件
            try:
                update_bin_file(bin_path, date_value_pairs, new_date_index)
            except Exception as e:
                logger.error("更新 %s/%s 失败: %s", code, bin_field, str(e)[:120])
        
        updated_count += 1
    
    logger.info("bin 更新完成: %d 只股票", updated_count)
    return updated_count


def update_calendars(new_dates: List[str]) -> None:
    """更新 calendars 文件"""
    logger.info("=" * 60)
    logger.info("更新 calendars 文件")
    logger.info("=" * 60)
    
    calendar_path = QLIB_DIR / "calendars" / "day.txt"
    calendar_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(calendar_path, "w", encoding="utf-8") as f:
        for d in new_dates:
            f.write(f"{d}\n")
    
    logger.info("calendars 已更新: %d 个日期", len(new_dates))


def update_instruments(
    existing_stocks: Dict[str, Dict[str, str]],
    new_data: Dict[str, pd.DataFrame],
    new_dates: List[str],
) -> None:
    """更新 instruments 文件"""
    logger.info("=" * 60)
    logger.info("更新 instruments 文件")
    logger.info("=" * 60)
    
    instruments_dir = QLIB_DIR / "instruments"
    instruments_dir.mkdir(parents=True, exist_ok=True)
    
    # 更新股票信息
    for code, df in new_data.items():
        if len(df) == 0:
            continue
        
        latest_date = pd.Timestamp(df["date"].max()).strftime("%Y-%m-%d")
        
        if code in existing_stocks:
            # 更新结束日期
            existing_stocks[code]["end"] = latest_date
        else:
            # 新增股票
            earliest_date = pd.Timestamp(df["date"].min()).strftime("%Y-%m-%d")
            existing_stocks[code] = {"start": earliest_date, "end": latest_date}
    
    # 写入 all.txt
    all_path = instruments_dir / "all.txt"
    with open(all_path, "w", encoding="utf-8") as f:
        for code in sorted(existing_stocks.keys()):
            info = existing_stocks[code]
            f.write(f"{code}\t{info['start']}\t{info['end']}\n")
    
    # 写入 csi300.txt（简化处理，使用所有股票）
    csi300_path = instruments_dir / "csi300.txt"
    with open(csi300_path, "w", encoding="utf-8") as f:
        for code in sorted(existing_stocks.keys()):
            info = existing_stocks[code]
            f.write(f"{code}\t{info['start']}\t{info['end']}\n")
    
    logger.info("instruments 已更新: %d 只股票", len(existing_stocks))


def update_workflow_config(new_end_date: str) -> None:
    """更新 workflow_config.yaml 的 end_time"""
    logger.info("=" * 60)
    logger.info("更新 workflow_config.yaml")
    logger.info("=" * 60)
    
    config_path = _PROJECT_ROOT / "qlib_pipeline" / "workflow_config.yaml"
    if not config_path.exists():
        logger.warning("workflow_config.yaml 不存在，跳过更新")
        return
    
    with open(config_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # 查找并更新 data_handler.end_time
    updated = False
    for i, line in enumerate(lines):
        if "end_time:" in line and "data_handler" in "".join(lines[max(0, i-10):i]):
            # 使用正则替换 end_time 值
            import re
            new_line = re.sub(
                r'(end_time:\s*")[^"]*(")',
                f'\\g<1>{new_end_date}\\g<2>',
                line
            )
            lines[i] = new_line
            updated = True
            break
    
    if updated:
        with open(config_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        logger.info("workflow_config.yaml 已更新: end_time = %s", new_end_date)
    else:
        logger.warning("未找到 data_handler.end_time 配置项")


def daily_update(days: int = 3, skip_bin: bool = False, workers: int = MAX_WORKERS) -> None:
    """
    执行每日增量更新
    
    Args:
        days: 下载最近多少天的数据
        skip_bin: 是否跳过 bin 文件更新
        workers: 并发线程数
    """
    start_time = time.time()
    
    logger.info("=" * 60)
    logger.info("开始每日增量更新")
    logger.info("=" * 60)
    
    # 1. 获取现有数据信息
    existing_dates = get_existing_qlib_dates()
    existing_stocks = get_existing_qlib_stocks()
    stock_codes = get_existing_qlib_stocks_from_features()
    
    logger.info("现有数据: %d 只股票 (features), %d 个日期 (Qlib)", 
                len(stock_codes), len(existing_dates))
    
    if not stock_codes:
        logger.error("没有现有的 Qlib 数据，请先运行全量下载")
        return
    
    # 2. 增量下载新数据
    new_data = download_incremental_data(stock_codes, days, workers)
    if not new_data:
        logger.info("没有新数据需要更新")
        return
    
    # 3. 收集所有新日期
    all_new_dates = set()
    for df in new_data.values():
        for dt in df["date"]:
            all_new_dates.add(pd.Timestamp(dt).strftime("%Y-%m-%d"))
    
    # 合并新旧日期
    combined_dates = sorted(existing_dates | all_new_dates)
    
    # 4. 更新 bin 文件
    if not skip_bin:
        update_qlib_bin_files(new_data, existing_dates, combined_dates)
    
    # 5. 更新 calendars
    update_calendars(combined_dates)
    
    # 6. 更新 instruments
    update_instruments(existing_stocks, new_data, combined_dates)
    
    # 7. 更新 workflow_config.yaml
    new_end_date = combined_dates[-1]
    update_workflow_config(new_end_date)
    
    # 完成
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("每日增量更新完成!")
    logger.info("  更新股票数: %d", len(new_data))
    logger.info("  新增日期数: %d", len(all_new_dates - existing_dates))
    logger.info("  最新日期: %s", new_end_date)
    logger.info("  总耗时: %.1f 秒", elapsed)
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="每日增量更新")
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=3,
        help="下载最近多少天的数据（默认: 3）"
    )
    parser.add_argument(
        "--skip-bin",
        action="store_true",
        help="跳过 bin 文件更新（只更新 CSV）"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=MAX_WORKERS,
        help=f"并发线程数（默认: {MAX_WORKERS}）"
    )
    args = parser.parse_args()
    
    daily_update(days=args.days, skip_bin=args.skip_bin, workers=args.workers)


if __name__ == "__main__":
    main()
