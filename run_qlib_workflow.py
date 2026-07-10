# -*- coding: utf-8 -*-
"""
Qlib 完整工作流脚本
====================
功能:
  1. 将 D:\data 目录下的 CSV 数据（sh.XXXXXX.csv / sz.XXXXXX.csv）转换为 Qlib bin 格式
     支持日线(day)、周线(week)、月线(month) 三种频率
  2. 初始化 Qlib 并运行完整的量化工作流:
     - 使用 Alpha158 特征 + LightGBM 模型
     - 在 CSI300 数据上训练（2008-2020）
     - 预测和回测
     - 生成分析报告图表

用法:
  python run_qlib_workflow.py                           # 转换全部股票（日线）
  python run_qlib_workflow.py --sample 100              # 仅转换前100只股票（快速测试）
  python run_qlib_workflow.py --freq week --sample 100  # 转换为周线
  python run_qlib_workflow.py --freq month              # 转换为月线
  python run_qlib_workflow.py --skip-convert            # 跳过数据转换，直接运行工作流
"""

import argparse
import logging
import os
import sys
import struct
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("qlib_workflow")

# ============================================================
# 路径常量（支持环境变量覆盖，默认使用相对路径）
# ============================================================
_PROJECT_ROOT = Path(__file__).parent

CSV_DATA_DIR = Path(os.environ.get("QLIB_CSV_DIR", "D:/data"))
QLIB_DATA_DIR = Path(os.environ.get("QLIB_DATA_DIR", str(_PROJECT_ROOT / "qlib_data" / "cn_data")))
CHART_OUTPUT_DIR = Path(os.environ.get("QLIB_CHART_DIR", str(_PROJECT_ROOT / "output" / "qlib_charts")))

# Qlib bin 格式需要的字段
# 注意: Qlib FeatureD.feature() 会将 "$close" 转换为 "close" (去掉第一个 $)
# 所以 bin 文件名应为 close.day.bin 而不是 $close.day.bin
INCLUDE_FIELDS = ["open", "close", "high", "low", "volume", "amount"]

# ============================================================
# 命令行参数
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Qlib 完整工作流")
    parser.add_argument("--sample", type=int, default=None,
                        help="仅处理前 N 只股票（快速测试）")
    parser.add_argument("--skip-convert", action="store_true",
                        help="跳过数据转换，直接运行工作流")
    parser.add_argument("--freq", type=str, default="day",
                        choices=["day", "week", "month"],
                        help="数据频率: day(日线) | week(周线) | month(月线)")
    return parser.parse_args()


# ============================================================
# 第一步: CSV 数据转换为 Qlib bin 格式
# ============================================================
def _fetch_csi300_constituents():
    """通过 AKShare 获取当前时点沪深300真实成分股代码列表。

    返回的是当前最新快照（非历史时点），用于替代原来 csi300.txt 复制全市场的错误实现。
    历史时点成分股的动态接入留作后续任务。

    Returns:
        set[str]: 6 位数字股票代码集合；AKShare 不可用或调用失败时返回空集合（调用方需处理 fallback）。
    """
    try:
        import akshare as ak
    except ImportError:
        logger.warning("akshare 未安装，无法获取沪深300真实成分股列表")
        return set()
    try:
        df = ak.index_stock_cons_csindex(symbol="000300")
        if df is None or len(df) == 0:
            logger.warning("AKShare index_stock_cons_csindex 返回空数据")
            return set()
        # 成分券代码列可能叫 '成分券代码'，统一转为 6 位字符串
        code_col = "成分券代码" if "成分券代码" in df.columns else df.columns[4]
        codes = set(df[code_col].astype(str).str.zfill(6).tolist())
        logger.info("AKShare 获取沪深300成分股: %d 只", len(codes))
        return codes
    except Exception as e:
        logger.warning("获取沪深300成分股失败: %s", str(e)[:150])
        return set()


def discover_csv_files(csv_dir: Path, sample: int = None):
    """
    发现并收集 CSV 文件

    Args:
        csv_dir: CSV 文件所在目录
        sample: 限制读取的股票数量

    Returns:
        排序后的 CSV 文件路径列表
    """
    csv_files = sorted(csv_dir.glob("*.csv"))
    logger.info("发现 %d 个 CSV 文件于 %s", len(csv_files), csv_dir)

    if sample is not None and sample > 0:
        csv_files = csv_files[:sample]
        logger.info("--sample=%d, 仅处理前 %d 个文件", sample, len(csv_files))

    return csv_files


def extract_stock_code(filename_stem: str) -> str:
    """
    从文件名中提取股票代码

    文件名格式: sh.600000, sz.000001
    返回6位数字代码: 600000, 000001

    Args:
        filename_stem: 不含后缀的文件名

    Returns:
        6位股票代码字符串
    """
    # 移除 sh. / sz. 前缀
    if "." in filename_stem:
        code = filename_stem.split(".")[-1]
    else:
        code = filename_stem
    return code.strip()


def read_and_normalize_csv(csv_path: Path) -> pd.DataFrame:
    """
    读取单个 CSV 文件并规范化列名

    Args:
        csv_path: CSV 文件路径

    Returns:
        规范化后的 DataFrame, 额外包含 '$' 列（股票代码）
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # 规范化列名
    df.columns = df.columns.str.strip().str.lower()

    # 重命名 date -> datetime（Qlib 标准格式）
    if "date" in df.columns:
        df.rename(columns={"date": "datetime"}, inplace=True)

    # 转换日期格式
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    return df


def resample_daily_to_freq(df: pd.DataFrame, freq: str = "week") -> pd.DataFrame:
    """
    将日线 OHLCV 数据重采样为周线/月线。

    重采样规则：
      - open:  周期内第一个交易日的开盘价
      - high:  周期内最高价
      - low:   周期内最低价
      - close: 周期内最后一个交易日的收盘价
      - volume: 周期内成交量之和
      - amount: 周期内成交额之和

    Args:
        df: 日线数据，必须包含 datetime, open, high, low, close, volume 列
        freq: "week" 或 "month"

    Returns:
        重采样后的 DataFrame
    """
    if freq not in ("week", "month"):
        return df

    if "datetime" not in df.columns:
        return df

    df = df.copy()
    df = df.set_index("datetime").sort_index()

    # 确定重采样锚点
    resample_rule = "W-FRI" if freq == "week" else "ME"

    resampled = df.resample(resample_rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "amount": "sum",
    })

    # 也保留 code 列（如果有的话）
    if "code" in df.columns:
        code_val = df["code"].iloc[0]
        resampled["code"] = code_val

    # 重置索引，恢复 datetime 列
    resampled = resampled.reset_index()

    # 移除含有 NaN 的行（首尾可能不完整）
    resampled = resampled.dropna()

    return resampled


def create_qlib_bin_data(csv_dir: Path, qlib_dir: Path, sample: int = None, freq: str = "day"):
    """
    将 CSV 数据转换为 Qlib bin 格式（支持日线/周线/月线）。

    Qlib bin 格式目录结构:
      qlib_dir/
        calendars/
          {freq}.txt            -- 交易日历（每行一个日期 YYYY-MM-DD）
        instruments/
          all.txt               -- 所有股票代码列表
          csi300.txt            -- CSI300 成分股
        features/
          <code>/
            <field>.{freq}.bin  -- 每个特征一个文件, float32 格式

    Args:
        csv_dir: CSV 数据源目录（日线数据）
        qlib_dir: Qlib 数据输出目录
        sample: 限制股票数量
        freq: 数据频率: "day" | "week" | "month"
    """
    logger.info("=" * 60)
    logger.info("第一步: CSV 数据转换为 Qlib bin 格式 (频率: %s)", freq)
    logger.info("=" * 60)

    # 确保目录结构
    features_dir = qlib_dir / "features"
    calendars_dir = qlib_dir / "calendars"
    instruments_dir = qlib_dir / "instruments"
    for d in [features_dir, calendars_dir, instruments_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 发现 CSV 文件
    csv_files = discover_csv_files(csv_dir, sample)
    if not csv_files:
        logger.error("未找到任何 CSV 文件，请检查 %s", csv_dir)
        sys.exit(1)

    # 收集所有数据
    all_data = {}  # code -> DataFrame
    failed_files = []
    total_date_set = set()

    # 周线/月线的最小数据量要求（日线 500 → 周线 100 → 月线 24）
    min_rows = {"day": 500, "week": 100, "month": 24}.get(freq, 500)

    for csv_path in csv_files:
        code = extract_stock_code(csv_path.stem)

        if len(code) != 6 or not code.isdigit():
            logger.debug("跳过无效文件名: %s (代码=%s)", csv_path.name, code)
            continue

        try:
            df = read_and_normalize_csv(csv_path)

            required = ["datetime"] + INCLUDE_FIELDS
            missing = [c for c in required if c not in df.columns]
            if missing:
                logger.warning("文件 %s 缺少列: %s, 跳过", csv_path.name, missing)
                failed_files.append(csv_path.name)
                continue

            df = df.dropna(subset=["datetime"]).copy()
            df = df.sort_values("datetime").reset_index(drop=True)

            # 周线/月线: 先重采样
            if freq != "day":
                df = resample_daily_to_freq(df, freq)
                if df is None or len(df) == 0:
                    logger.debug("跳过 %s: 重采样后无数据", code)
                    continue

            # 过滤数据太少的股票
            if len(df) < min_rows:
                logger.debug("跳过 %s: 仅有 %d 行数据 (< %d)", code, len(df), min_rows)
                continue

            for dt in df["datetime"]:
                total_date_set.add(pd.Timestamp(dt).strftime("%Y-%m-%d"))

            all_data[code] = df

        except Exception as e:
            logger.error("读取文件 %s 失败: %s", csv_path.name, e)
            failed_files.append(csv_path.name)

    logger.info("成功读取 %d 只股票数据, 失败 %d 个文件", len(all_data), len(failed_files))
    if not all_data:
        logger.error("没有可用数据，退出")
        sys.exit(1)

    # 构建交易日历
    all_dates = sorted(total_date_set)
    date_index = {d: i for i, d in enumerate(all_dates)}

    logger.info("交易日范围: %s ~ %s, 共 %d 个 %s 周期",
                all_dates[0], all_dates[-1], len(all_dates),
                {"day": "交易日", "week": "周", "month": "月"}.get(freq, "周期"))

    # 清空旧 features 目录（Windows 兼容处理）
    if features_dir.exists():
        import shutil
        try:
            shutil.rmtree(features_dir)
        except Exception as e:
            logger.warning("shutil.rmtree 失败 (%s), 尝试手动删除...", e)
            # 回退: 逐个删除文件再删除目录
            for root, dirs, files in os.walk(str(features_dir), topdown=False):
                for name in files:
                    try:
                        os.remove(os.path.join(root, name))
                    except OSError:
                        pass
                for name in dirs:
                    try:
                        os.rmdir(os.path.join(root, name))
                    except OSError:
                        pass
            try:
                os.rmdir(str(features_dir))
            except OSError:
                pass
        logger.info("已清空旧 features 目录: %s", features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)

    # 写入 bin 文件
    logger.info("开始写入 bin 文件...")
    for code, df in all_data.items():
        stock_dir = features_dir / code
        stock_dir.mkdir(parents=True, exist_ok=True)

        date_value_map = {}
        for _, row in df.iterrows():
            dt_str = pd.Timestamp(row["datetime"]).strftime("%Y-%m-%d")
            date_value_map[dt_str] = row

        sorted_data_dates = sorted(
            dt_str for dt_str in date_value_map if dt_str in date_index
        )
        if not sorted_data_dates:
            logger.warning("股票 %s 没有有效日期数据, 跳过", code)
            continue
        first_date_idx = date_index[sorted_data_dates[0]]

        for field in INCLUDE_FIELDS:
            bin_path = stock_dir / f"{field}.{freq}.bin"

            values = np.full(len(all_dates), np.nan, dtype=np.float32)
            for dt_str, idx in date_index.items():
                if dt_str in date_value_map:
                    val = date_value_map[dt_str][field]
                    if pd.notna(val):
                        values[idx] = float(val)

            bin_data = np.hstack([first_date_idx, values]).astype("<f")
            with open(bin_path, "wb") as f:
                bin_data.tofile(f)

    logger.info("bin 文件写入完成: %d 只股票, %d 个特征", len(all_data), len(INCLUDE_FIELDS))

    # 写入交易日历
    calendar_path = calendars_dir / f"{freq}.txt"
    with open(calendar_path, "w", encoding="utf-8") as f:
        for d in all_dates:
            f.write(f"{d}\n")
    logger.info("交易日历已写入: %s (%d 个周期)", calendar_path, len(all_dates))

    # 写入 instruments 文件
    codes_list = sorted(all_data.keys())

    # all.txt: 包含全部股票 (Qlib 格式: instrument\tstart_datetime\tend_datetime)
    all_instruments_path = instruments_dir / "all.txt"
    with open(all_instruments_path, "w", encoding="utf-8") as f:
        for code in codes_list:
            start_date = all_dates[0]
            end_date = all_dates[-1]
            if code in all_data:
                earliest = all_data[code]["datetime"].min()
                latest = all_data[code]["datetime"].max()
                start_date = pd.Timestamp(earliest).strftime("%Y-%m-%d")
                end_date = pd.Timestamp(latest).strftime("%Y-%m-%d")
            f.write(f"{code}\t{start_date}\t{end_date}\n")
    logger.info("全量股票列表已写入: %s (%d 只)", all_instruments_path, len(codes_list))

    # csi300.txt: 接入 AKShare 获取当前时点真实沪深300成分股列表
    # 仅当本地 CSV 数据中存在该成分股代码时才写入 csi300.txt
    # TODO(后续任务): 接入历史时点（按月/季度快照）成分股数据，改为按交易日动态判断
    csi300_path = instruments_dir / "csi300.txt"
    csi300_codes = _fetch_csi300_constituents()
    if csi300_codes:
        # 取本地数据与真实成分股的交集
        csi300_in_local = [c for c in codes_list if c in csi300_codes]
        missing_in_local = [c for c in csi300_codes if c not in all_data]
        with open(csi300_path, "w", encoding="utf-8") as f:
            for code in csi300_in_local:
                earliest = all_data[code]["datetime"].min()
                latest = all_data[code]["datetime"].max()
                start_date = pd.Timestamp(earliest).strftime("%Y-%m-%d")
                end_date = pd.Timestamp(latest).strftime("%Y-%m-%d")
                f.write(f"{code}\t{start_date}\t{end_date}\n")
        logger.warning(
            "⚠ 当前 csi300.txt 使用的是静态的当前时点成分股列表（AKShare %s 快照），"
            "不是历史动态时点成分股。早期年份的回测可能包含'未来才纳入指数'的股票，"
            "存在一定的成分股穿越风险，待接入历史时点成分股数据后修正。",
            datetime.now().strftime("%Y-%m-%d")
        )
        logger.info(
            "CSI300 instruments 已写入: %s (%d 只真实成分股, 本地缺失 %d 只)",
            csi300_path, len(csi300_in_local), len(missing_in_local)
        )
    else:
        # AKShare 不可用时退化为全市场（保持原有行为，但明确警告）
        logger.warning(
            "⚠ AKShare 不可用或获取沪深300成分股失败，csi300.txt 退化为全市场股票列表。"
            "这会导致 instruments:'csi300' 的配置实际跑全市场，请尽快修复数据源。"
        )
        with open(csi300_path, "w", encoding="utf-8") as f:
            for code in codes_list:
                start_date = all_dates[0]
                end_date = all_dates[-1]
                if code in all_data:
                    earliest = all_data[code]["datetime"].min()
                    latest = all_data[code]["datetime"].max()
                    start_date = pd.Timestamp(earliest).strftime("%Y-%m-%d")
                    end_date = pd.Timestamp(latest).strftime("%Y-%m-%d")
                f.write(f"{code}\t{start_date}\t{end_date}\n")
        logger.info("CSI300 instruments 已写入(fallback 全市场): %s (%d 只)", csi300_path, len(codes_list))

    # 写入 st marking（用于标记 ST 股票，这里全部标记为正常）
    st_path = instruments_dir / "st.txt"
    with open(st_path, "w", encoding="utf-8") as f:
        pass  # 空文件，没有 ST 标记
    logger.info("ST 标记文件已写入: %s", st_path)

    # 汇总输出 Qlib 目录结构信息
    logger.info("Qlib 数据目录结构:")
    for root, dirs, files in os.walk(str(qlib_dir)):
        level = root.replace(str(qlib_dir), "").count(os.sep)
        indent = "  " * level
        file_count = len(files)
        dir_count = len(dirs)
        if level <= 1:
            logger.info("%s%s/ (%d 个子目录, %d 个文件)",
                        indent, os.path.basename(root), dir_count, file_count)

    logger.info("CSV -> Qlib bin 转换完成!")

    # 生存者偏差警告（Task 4）
    logger.warning(
        "⚠ 当前股票池不包含历史退市股票，长周期（5年以上）回测收益可能被系统性高估，"
        "这一点在解读回测结果时必须考虑"
    )


# ============================================================
# 第二步: 初始化 Qlib 并运行工作流
# ============================================================
def run_qlib_workflow(provider_uri: str, chart_dir: Path, freq: str = "day"):
    """
    运行完整的 Qlib 工作流:
      1. 初始化 Qlib
      2. 定义任务（Alpha158 + LightGBM）
      3. 训练模型
      4. 回测
      5. 生成分析报告图表

    Args:
        provider_uri: Qlib 数据目录路径
        chart_dir: 图表输出目录
    """
    logger.info("=" * 60)
    logger.info("第二步: 运行 Qlib 工作流")
    logger.info("=" * 60)

    # ---- 延迟导入 Qlib 相关模块 ----
    try:
        import qlib
        from qlib.config import REG_CN
        from qlib.data import D
        from qlib.utils import exists_qlib_data, flatten_dict
        from qlib.workflow import R
        from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
        from qlib.contrib.model.gbdt import LGBModel
        from qlib.contrib.data.handler import Alpha158
        from qlib.data.dataset import DatasetH
        from qlib.contrib.evaluate import risk_analysis
        from qlib.utils import init_instance_by_config
    except ImportError as e:
        logger.error("无法导入 Qlib 模块: %s", e)
        logger.error("请确保已安装 qlib: pip install pyqlib")
        sys.exit(1)

    # ---- 检查数据是否已转换 ----
    # Skip exists_qlib_data check due to pandas compatibility issue in qlib 0.9.7
    from pathlib import Path
    qlib_check = Path(provider_uri).expanduser()
    if not (qlib_check / "calendars" / f"{freq}.txt").exists():
        logger.error("Qlib 数据不存在于 %s (calendars/%s.txt)，请先运行数据转换", provider_uri, freq)
        sys.exit(1)
    logger.info("Qlib 数据检查通过")

    # ---- 初始化 Qlib ----
    logger.info("初始化 Qlib, 数据路径: %s", provider_uri)
    # 修复 numpy 2.x 兼容性：设置 qlib 为单线程，避免 joblib 子进程中 np.isclose 报错
    os.environ["NUMEXPR_MAX_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    # 强制 qlib 使用单线程
    import qlib
    from qlib.config import C
    C.joblib_backend = "threading"
    C.maxtasksperchild = None
    logger.info("Qlib 初始化成功 (单线程模式)")

    # ---- 工作流配置 ----
    # 从 qlib_pipeline/workflow_config.yaml 读取配置，替代硬编码值（保留原值作为 fallback）
    import yaml
    from pathlib import Path
    config_path = Path(__file__).parent / "qlib_pipeline" / "workflow_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_handler_cfg = cfg.get("data_handler", {})
    market = data_handler_cfg.get("instruments", "csi300")
    benchmark = None  # 不使用基准指数对比（避免 index code 不存在的问题）

    data_handler_config = {
        "start_time": data_handler_cfg.get("start_time", "2008-01-01"),
        "end_time": data_handler_cfg.get("end_time", "2024-12-31"),
        "fit_start_time": data_handler_cfg.get("fit_start_time", "2008-01-01"),
        "fit_end_time": data_handler_cfg.get("fit_end_time", "2020-12-31"),
        "instruments": market,
        "freq": freq,  # freq 仍由函数参数传入，不读 yaml
    }

    # 模型超参：默认硬编码值，再用 yaml 中的 kwargs 覆盖
    qlib_lgb_cfg = cfg.get("qlib_lgb", {})
    default_model_kwargs = {
        "loss": "mse",
        "colsample_bytree": 0.8879,
        "learning_rate": 0.0421,
        "subsample": 0.8789,
        "lambda_l1": 205.6999,
        "lambda_l2": 580.9768,
        "max_depth": 8,
        "num_leaves": 210,
        "num_threads": 20,
    }
    model_kwargs = {**default_model_kwargs, **qlib_lgb_cfg.get("kwargs", {})}

    # loss=rank 时切换为 RankLGBModel（qlib_pipeline.model），否则用 yaml 中指定的类
    if model_kwargs.get("loss") == "rank":
        model_class = "RankLGBModel"
        model_module_path = "qlib_pipeline.model"
    else:
        model_class = qlib_lgb_cfg.get("class", "LGBModel")
        model_module_path = qlib_lgb_cfg.get("module_path", "qlib.contrib.model.gbdt")

    task = {
        "model": {
            "class": model_class,
            "module_path": model_module_path,
            "kwargs": model_kwargs,
        },
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "Alpha158",
                    "module_path": "qlib.contrib.data.handler",
                    "kwargs": data_handler_config,
                },
                "segments": cfg.get("dataset", {}).get("segments", {
                    "train": ("2008-01-01", "2020-12-31"),
                    "valid": ("2021-01-01", "2022-12-31"),
                    "test": ("2023-01-01", "2024-12-31"),
                }),
            },
        },
    }

    # ---- 创建模型和数据集 ----
    logger.info("创建 LightGBM 模型和 Alpha158 数据集...")
    model = init_instance_by_config(task["model"])
    dataset = init_instance_by_config(task["dataset"])
    logger.info("数据集创建完成")

    # ---- 训练模型 ----
    logger.info("开始训练模型...")
    with R.start(experiment_name="train_model"):
        R.log_params(**flatten_dict(task))
        model.fit(dataset)
        R.save_objects(trained_model=model)
        rid = R.get_recorder().id
        logger.info("模型训练完成, recorder_id=%s", rid)

    # ---- 回测配置 ----
    backtest_cfg = cfg.get("backtest", {})
    strategy_kwargs_cfg = backtest_cfg.get("strategy", {}).get("kwargs", {})
    backtest_inner_cfg = backtest_cfg.get("backtest", {})

    port_analysis_config = {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
            },
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "model": model,
                "dataset": dataset,
                "topk": strategy_kwargs_cfg.get("topk", 50),
                "n_drop": strategy_kwargs_cfg.get("n_drop", 5),
            },
        },
        "backtest": {
            "start_time": backtest_inner_cfg.get("start_time", "2023-01-01"),
            "end_time": backtest_inner_cfg.get("end_time", "2024-12-31"),
            "account": backtest_inner_cfg.get("account", 100000000),
            "benchmark": benchmark,
            "exchange_kwargs": backtest_inner_cfg.get("exchange_kwargs", {
                "freq": "day",
                "limit_threshold": 0.099,
                "deal_price": "open",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            }),
        },
    }

    # ---- 运行回测 ----
    logger.info("开始回测分析...")
    with R.start(experiment_name="backtest_analysis"):
        recorder = R.get_recorder(recorder_id=rid, experiment_name="train_model")
        model_bt = recorder.load_object("trained_model")
        recorder = R.get_recorder()
        ba_rid = recorder.id

        sr = SignalRecord(model_bt, dataset, recorder)
        sr.generate()

        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()
        logger.info("回测分析完成, recorder_id=%s", ba_rid)

    # ---- 生成分析图表 ----
    logger.info("开始生成分析图表...")
    try:
        generate_analysis_charts(R, dataset, ba_rid, chart_dir)
    except Exception as e:
        logger.error("生成图表时出错: %s", e)
        raise


# ============================================================
# 第三步: 生成分析图表并保存为 PNG
# ============================================================
def generate_analysis_charts(R, dataset, ba_rid, chart_dir: Path):
    """
    从回测结果中提取数据，使用 Qlib 内置分析模块生成图表并保存为 PNG

    Args:
        R: Qlib workflow R 对象
        dataset: 训练好的数据集
        ba_rid: 回测 recorder ID
        chart_dir: 图表输出目录
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    chart_dir.mkdir(parents=True, exist_ok=True)

    # 获取回测 recorder
    recorder = R.get_recorder(recorder_id=ba_rid, experiment_name="backtest_analysis")

    # 加载回测结果
    try:
        pred_df = recorder.load_object("pred.pkl")
    except Exception as e:
        logger.warning("无法加载 pred.pkl: %s", e)
        pred_df = None

    try:
        report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    except Exception as e:
        logger.warning("无法加载 report_normal_1day.pkl: %s", e)
        report_normal_df = None

    try:
        positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
    except Exception as e:
        logger.warning("无法加载 positions_normal_1day.pkl: %s", e)
        positions = None

    try:
        analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
    except Exception as e:
        logger.warning("无法加载 port_analysis_1day.pkl: %s", e)
        analysis_df = None

    saved_files = []

    # ============================================================
    # 图表 1: 收益曲线图 (report_graph)
    # ============================================================
    if report_normal_df is not None:
        try:
            from qlib.contrib.report import analysis_position

            fig = analysis_position.report_graph(report_normal_df)
            save_path = chart_dir / "01_report.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved_files.append(save_path)
            logger.info("图表已保存: %s", save_path)
        except Exception as e:
            logger.error("生成 report_graph 失败: %s", e)

    # ============================================================
    # 图表 2: 风险分析图 (risk_analysis_graph)
    # ============================================================
    if analysis_df is not None and report_normal_df is not None:
        try:
            from qlib.contrib.report import analysis_position

            fig = analysis_position.risk_analysis_graph(analysis_df, report_normal_df)
            save_path = chart_dir / "02_risk_analysis.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved_files.append(save_path)
            logger.info("图表已保存: %s", save_path)
        except Exception as e:
            logger.error("生成 risk_analysis_graph 失败: %s", e)

    # ============================================================
    # 图表 3: Score IC 图
    # ============================================================
    if pred_df is not None:
        try:
            from qlib.contrib.report import analysis_position

            label_df = dataset.prepare("test", col_set="label")
            label_df.columns = ["label"]
            pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)

            fig = analysis_position.score_ic_graph(pred_label)
            save_path = chart_dir / "03_score_ic.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved_files.append(save_path)
            logger.info("图表已保存: %s", save_path)
        except Exception as e:
            logger.error("生成 score_ic_graph 失败: %s", e)

    # ============================================================
    # 图表 4: 模型性能图 (model_performance_graph)
    # ============================================================
    if pred_df is not None:
        try:
            from qlib.contrib.report import analysis_model

            label_df = dataset.prepare("test", col_set="label")
            label_df.columns = ["label"]
            pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)

            fig = analysis_model.model_performance_graph(pred_label)
            save_path = chart_dir / "04_model_performance.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved_files.append(save_path)
            logger.info("图表已保存: %s", save_path)
        except Exception as e:
            logger.error("生成 model_performance_graph 失败: %s", e)

    # ============================================================
    # 图表 5: 预测值分布直方图
    # ============================================================
    if pred_df is not None:
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            pred_values = pred_df.iloc[:, 0].dropna()
            ax.hist(pred_values, bins=100, color="steelblue", edgecolor="white", alpha=0.8)
            ax.set_title("Prediction Value Distribution", fontsize=14)
            ax.set_xlabel("Prediction", fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            save_path = chart_dir / "05_pred_distribution.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved_files.append(save_path)
            logger.info("图表已保存: %s", save_path)
        except Exception as e:
            logger.error("生成预测分布图失败: %s", e)

    # ============================================================
    # 图表 6: 累计收益曲线图（手动绘制）
    # ============================================================
    if report_normal_df is not None:
        try:
            fig, ax = plt.subplots(figsize=(14, 7))

            if "return" in report_normal_df.columns:
                cum_ret = (1 + report_normal_df["return"]).cumprod()
                ax.plot(cum_ret.index, cum_ret.values, label="Portfolio", color="steelblue", linewidth=1.5)

            if "bench" in report_normal_df.columns:
                cum_bench = (1 + report_normal_df["bench"]).cumprod()
                ax.plot(cum_bench.index, cum_bench.values, label="Benchmark", color="orange",
                        linewidth=1.5, linestyle="--")

            ax.set_title("Cumulative Return: Portfolio vs Benchmark", fontsize=14)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Cumulative Return", fontsize=12)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            fig.autofmt_xdate()
            fig.tight_layout()
            save_path = chart_dir / "06_cumulative_return.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved_files.append(save_path)
            logger.info("图表已保存: %s", save_path)
        except Exception as e:
            logger.error("生成累计收益图失败: %s", e)

    # ============================================================
    # 汇总
    # ============================================================
    logger.info("=" * 60)
    logger.info("图表生成完成! 共保存 %d 张图表到:", len(saved_files))
    for f in saved_files:
        logger.info("  - %s", f.name)
    logger.info("=" * 60)


# ============================================================
# 主入口
# ============================================================
def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Qlib 完整工作流启动")
    logger.info("  CSV 数据目录: %s", CSV_DATA_DIR)
    logger.info("  Qlib 数据目录: %s", QLIB_DATA_DIR)
    logger.info("  图表输出目录: %s", CHART_OUTPUT_DIR)
    if args.sample:
        logger.info("  样本模式: 限制 %d 只股票", args.sample)
    if args.skip_convert:
        logger.info("  跳过数据转换")
    logger.info("=" * 60)

    # 第一步: 数据转换
    if not args.skip_convert:
        try:
            create_qlib_bin_data(CSV_DATA_DIR, QLIB_DATA_DIR, sample=args.sample, freq=args.freq)
        except Exception as e:
            logger.error("数据转换失败: %s", e, exc_info=True)
            sys.exit(1)
    else:
        logger.info("跳过数据转换步骤，使用已有数据: %s", QLIB_DATA_DIR)

    # 检查数据目录是否存在
    if not QLIB_DATA_DIR.exists():
        logger.error("Qlib 数据目录不存在: %s", QLIB_DATA_DIR)
        logger.error("请先不使用 --skip-convert 运行一次以生成数据")
        sys.exit(1)

    # 第二步: 运行 Qlib 工作流
    try:
        run_qlib_workflow(
            provider_uri=str(QLIB_DATA_DIR),
            chart_dir=CHART_OUTPUT_DIR,
            freq=args.freq,
        )
    except Exception as e:
        logger.error("Qlib 工作流执行失败: %s", e, exc_info=True)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("全部任务完成!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
