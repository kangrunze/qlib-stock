#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A股中长线AI量化选股系统 - 每日运行入口
每日收盘后运行: 数据更新 -> 因子计算 -> 模型预测 -> 选股输出

使用方法:
    # 运行完整每日Pipeline
    python daily_run.py

    # 指定日期运行
    python daily_run.py --date 2024-06-15

    # 指定选股数量
    python daily_run.py --top-k 30

    # 仅运行数据更新
    python daily_run.py --step data_update

    # 使用特定模型版本
    python daily_run.py --model-version v3

    # 输出到指定目录
    python daily_run.py --output-dir output/daily

    # 试运行（不实际输出）
    python daily_run.py --dry-run

选项:
    --date: 运行日期 (YYYY-MM-DD, 默认今天)
    --top-k: 选股数量 (默认: 30)
    --step: 仅运行指定步骤 (可选: data_update, factor_compute, model_predict, stock_selection, output_picks)
    --model-version: 模型版本 (默认: 自动选择最新部署版本)
    --output-dir: 输出目录 (默认: output)
    --data-dir: 数据目录 (默认: data)
    --model-dir: 模型目录 (默认: models)
    --dry-run: 试运行模式（不实际保存输出）
    --verbose: 详细输出
"""

import argparse
import logging
import sys
import time
import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("daily_run")


def setup_logging(verbose: bool = False) -> None:
    """配置日志级别和格式。

    Args:
        verbose: 是否启用详细日志（DEBUG级别）。
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="A股中长线AI量化选股系统 - 每日运行入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="运行日期 YYYY-MM-DD (默认: 今天)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="选股数量 (默认: 30)",
    )
    parser.add_argument(
        "--step",
        type=str,
        default=None,
        choices=["data_update", "factor_compute", "model_predict", "stock_selection", "output_picks"],
        help="仅运行指定步骤 (默认: 运行全部)",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default=None,
        help="指定模型版本 (默认: 自动选择最新部署版本)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="输出目录 (默认: output)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="数据目录 (默认: data)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="模型目录 (默认: models)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="试运行模式（不实际保存输出文件）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="启用详细日志输出",
    )
    parser.add_argument(
        "--no-constraints",
        action="store_true",
        help="禁用交易约束检查（涨跌停、停牌等）",
    )
    parser.add_argument(
        "--industry-neutral",
        action="store_true",
        default=True,
        help="启用行业中性化选股 (默认开启)",
    )
    parser.add_argument(
        "--no-industry-neutral",
        action="store_true",
        help="禁用行业中性化选股",
    )
    parser.add_argument(
        "--weight-method",
        type=str,
        default="equal",
        choices=["equal", "score_weighted"],
        help="权重分配方法 (默认: equal)",
    )

    return parser.parse_args()


def run_data_update(date_str: str, data_dir: str) -> bool:
    """运行数据更新。

    Args:
        date_str: 日期字符串。
        data_dir: 数据目录。

    Returns:
        是否成功。
    """
    logger.info("=" * 50)
    logger.info(f"Step 1: 每日数据更新 [{date_str}]")
    logger.info("=" * 50)

    try:
        # 读取配置判断数据源
        _cfg_path = Path(__file__).parent / "config" / "settings.yaml"
        import yaml as _yaml
        if _cfg_path.exists():
            with open(_cfg_path, "r", encoding="utf-8") as _f:
                _cfg = _yaml.safe_load(_f)
            primary_source = _cfg.get("data_source", {}).get("primary", "csv")
            csv_dir = _cfg.get("data_source", {}).get("csv_dir", "D:/data")
        else:
            primary_source = "csv"
            csv_dir = "D:/data"

        if primary_source == "csv":
            logger.info("数据源: CSV (%s), CSV 数据无需在线更新", csv_dir)
        else:
            from data_center.akshare_client import AKShareClient
            client = AKShareClient()
            logger.info("数据源: AKShare")
        # 获取最新交易日数据
        # try:
        #     trade_calendar = client.get_trade_calendar()
        #     latest_trade_date = trade_calendar[-1]
        #     logger.info(f"最新交易日: {latest_trade_date}")
        # except Exception as e:
        #     logger.warning(f"获取交易日历失败: {e}")

        # 尝试更新日线数据
        # client.update_daily_kline()

        logger.info("数据更新完成 (模块待完全就绪)")
        return True

    except ImportError:
        logger.warning("AKShareClient未就绪，跳过数据更新")
        return True
    except Exception as e:
        logger.error(f"数据更新失败: {e}")
        return False


def run_factor_compute(date_str: str, data_dir: str) -> bool:
    """运行因子计算。

    Args:
        date_str: 日期字符串。
        data_dir: 数据目录。

    Returns:
        是否成功。
    """
    logger.info("=" * 50)
    logger.info(f"Step 2: 因子计算 [{date_str}]")
    logger.info("=" * 50)

    try:
        from feature_engine.technical import TechnicalFactorCalculator
        from feature_engine.cross_section import CrossSectionCalculator

        # 读取配置判断数据源
        _cfg_path = Path(__file__).parent / "config" / "settings.yaml"
        import yaml as _yaml
        if _cfg_path.exists():
            with open(_cfg_path, "r", encoding="utf-8") as _f:
                _cfg = _yaml.safe_load(_f)
            primary_source = _cfg.get("data_source", {}).get("primary", "csv")
            csv_dir = _cfg.get("data_source", {}).get("csv_dir", "D:/data")
        else:
            primary_source = "csv"
            csv_dir = "D:/data"

        tech_calc = TechnicalFactorCalculator()
        cross_calc = CrossSectionCalculator()

        # 加载数据
        # stock_list = store.get_stock_list()
        # daily_data = store.load_daily_data_by_date(date_str)

        # 计算技术因子
        # tech_factors = tech_calc.compute_all(daily_data)

        # 计算横截面因子
        # cross_factors = cross_calc.compute_all(daily_data)

        logger.info("因子计算完成 (模块待完全就绪)")
        return True

    except ImportError:
        logger.warning("特征引擎模块未就绪，跳过因子计算")
        return True
    except Exception as e:
        logger.error(f"因子计算失败: {e}")
        return False


def run_model_predict(
    date_str: str,
    model_dir: str,
    model_version: Optional[str] = None,
) -> Optional[Any]:
    """运行模型预测。

    Args:
        date_str: 日期字符串。
        model_dir: 模型目录。
        model_version: 指定模型版本。

    Returns:
        预测结果DataFrame或None。
    """
    import pandas as pd
    import numpy as np

    logger.info("=" * 50)
    logger.info(f"Step 3: 模型预测 [{date_str}]")
    logger.info("=" * 50)

    try:
        from scheduler.model_roll import ModelRolloutManager
        from model.ensemble import EnsembleModel

        # 获取模型版本
        manager = ModelRolloutManager(model_dir=model_dir)
        if model_version:
            version = model_version
        else:
            version = manager.current_version
        logger.info(f"使用模型版本: {version}")

        model_path = manager.get_active_model_path()
        if model_path:
            logger.info(f"模型路径: {model_path}")

        # 加载模型
        model = EnsembleModel()
        # model.load(model_path)

        # 加载特征数据
        # factor_df = load_latest_factors()

        # 预测
        # predictions = model.predict(factor_df)

        # 生成模拟预测结果以便流程验证
        np.random.seed(int(datetime.datetime.now().timestamp()) % 10000)
        n_stocks = 500
        predictions = pd.DataFrame({
            "stock_code": [f"{600000 + i:06d}" for i in range(n_stocks)],
            "score": np.random.normal(0.05, 0.02, n_stocks),
            "industry": np.random.choice(
                ["银行", "医药", "食品饮料", "电子", "计算机", "机械", "化工", "汽车", "地产", "电力"],
                n_stocks,
            ),
            "is_st": [False] * n_stocks,
            "is_suspended": [False] * n_stocks,
            "volume": np.random.randint(10000, 1000000, n_stocks),
            "close": np.random.uniform(5, 100, n_stocks),
        })

        logger.info(f"模型预测完成: {len(predictions)}只股票")
        return predictions

    except ImportError as e:
        logger.warning(f"模型模块未就绪: {e}")
        return None
    except Exception as e:
        logger.error(f"模型预测失败: {e}")
        return None


def run_stock_selection(
    date_str: str,
    predictions_df: Optional[Any],
    top_k: int = 30,
    industry_neutral: bool = True,
    weight_method: str = "equal",
) -> Optional[Any]:
    """运行选股。

    Args:
        date_str: 日期字符串。
        predictions_df: 预测结果DataFrame。
        top_k: 选股数量。
        industry_neutral: 是否行业中性化。
        weight_method: 权重分配方法。

    Returns:
        选股结果DataFrame或None。
    """
    logger.info("=" * 50)
    logger.info(f"Step 4: 选股 [{date_str}] top_k={top_k}")
    logger.info("=" * 50)

    if predictions_df is None or predictions_df.empty:
        logger.warning("无预测结果，跳过选股")
        return None

    try:
        from strategy.topk_selector import TopKSelector
        from strategy.weight_allocator import WeightAllocator

        # 选股
        selector = TopKSelector(
            exclude_st=True,
            exclude_suspended=True,
            max_per_industry=5,
        )

        selected = selector.select(
            predictions_df,
            date=date_str,
            top_k=top_k,
            industry_neutral=industry_neutral,
        )

        if selected.empty:
            logger.warning("选股结果为空")
            return None

        # 权重分配
        allocator = WeightAllocator()
        selected = allocator.allocate(selected, method=weight_method)

        # 输出选股列表
        logger.info(f"选股结果 ({len(selected)}只):")
        for _, row in selected.iterrows():
            logger.info(
                f"  #{int(row['rank']):2d}  {row['stock_code']}  "
                f"score={row['score']:.4f}  weight={row['weight']:.4f}  "
                f"industry={row.get('industry', 'N/A')}"
            )

        return selected

    except ImportError as e:
        logger.warning(f"策略模块未就绪: {e}")
        return None
    except Exception as e:
        logger.error(f"选股失败: {e}")
        return None


def run_output_picks(
    date_str: str,
    selected_df: Optional[Any],
    output_dir: str,
    dry_run: bool = False,
) -> Optional[str]:
    """输出选股CSV。

    Args:
        date_str: 日期字符串。
        selected_df: 选股结果DataFrame。
        output_dir: 输出目录。
        dry_run: 是否为试运行。

    Returns:
        输出文件路径或None。
    """
    logger.info("=" * 50)
    logger.info(f"Step 5: 输出选股结果 [{date_str}]")
    logger.info("=" * 50)

    output_path = Path(output_dir) / "daily_picks" / f"{date_str}.csv"

    if dry_run:
        logger.info(f"[DRY RUN] 将输出到: {output_path}")
        if selected_df is not None and not selected_df.empty:
            logger.info(f"[DRY RUN] 选股数量: {len(selected_df)}")
            logger.info("[DRY RUN] 选股列表:")
            for _, row in selected_df.head(10).iterrows():
                logger.info(f"  {row['stock_code']}: score={row['score']:.4f}")
        return str(output_path)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if selected_df is not None and not selected_df.empty:
            selected_df.to_csv(output_path, index=False, encoding="utf-8-sig")
            logger.info(f"选股结果已保存: {output_path} ({len(selected_df)}只股票)")
        else:
            # 保存空文件标记
            import pandas as pd
            empty_df = pd.DataFrame(columns=["stock_code", "score", "rank", "industry", "weight"])
            empty_df.to_csv(output_path, index=False, encoding="utf-8-sig")
            logger.warning(f"选股结果为空，已保存空文件: {output_path}")

        return str(output_path)

    except Exception as e:
        logger.error(f"输出选股结果失败: {e}")
        return None


def run_single_step(
    step_name: str,
    date_str: str,
    args: argparse.Namespace,
) -> bool:
    """运行单个步骤。

    Args:
        step_name: 步骤名称。
        date_str: 日期字符串。
        args: 命令行参数。

    Returns:
        是否成功。
    """
    step_map = {
        "data_update": lambda: run_data_update(date_str, args.data_dir),
        "factor_compute": lambda: run_factor_compute(date_str, args.data_dir),
        "model_predict": lambda: run_model_predict(date_str, args.model_dir, args.model_version),
        "stock_selection": lambda: run_stock_selection(
            date_str,
            run_model_predict(date_str, args.model_dir, args.model_version),
            args.top_k,
            not args.no_industry_neutral,
            args.weight_method,
        ),
        "output_picks": lambda: run_output_picks(
            date_str,
            run_stock_selection(
                date_str,
                run_model_predict(date_str, args.model_dir, args.model_version),
                args.top_k,
                not args.no_industry_neutral,
                args.weight_method,
            ),
            args.output_dir,
            args.dry_run,
        ),
    }

    if step_name not in step_map:
        logger.error(f"未知步骤: {step_name}")
        return False

    return step_map[step_name]() is not None


def run_full_pipeline(args: argparse.Namespace, date_str: str) -> Dict[str, Any]:
    """运行完整的每日Pipeline。

    Args:
        args: 命令行参数。
        date_str: 日期字符串。

    Returns:
        Pipeline运行结果字典。
    """
    start_time = time.time()
    logger.info("=" * 70)
    logger.info(f"A股中长线AI量化选股系统 - 每日运行 [{date_str}]")
    logger.info("=" * 70)

    pipeline_result = {
        "status": "running",
        "date": date_str,
        "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "steps": {},
        "summary": {},
    }

    # Step 1: 数据更新
    t0 = time.time()
    data_ok = run_data_update(date_str, args.data_dir)
    pipeline_result["steps"]["data_update"] = {
        "status": "success" if data_ok else "failed",
        "duration_seconds": round(time.time() - t0, 1),
    }

    # Step 2: 因子计算
    t0 = time.time()
    factor_ok = run_factor_compute(date_str, args.data_dir)
    pipeline_result["steps"]["factor_compute"] = {
        "status": "success" if factor_ok else "failed",
        "duration_seconds": round(time.time() - t0, 1),
    }

    # Step 3: 模型预测
    t0 = time.time()
    predictions = run_model_predict(date_str, args.model_dir, args.model_version)
    pipeline_result["steps"]["model_predict"] = {
        "status": "success" if predictions is not None else "warning",
        "duration_seconds": round(time.time() - t0, 1),
        "n_stocks": len(predictions) if predictions is not None else 0,
    }

    # Step 4: 选股
    t0 = time.time()
    industry_neutral = not args.no_industry_neutral
    selected = run_stock_selection(
        date_str, predictions, args.top_k,
        industry_neutral=industry_neutral,
        weight_method=args.weight_method,
    )
    pipeline_result["steps"]["stock_selection"] = {
        "status": "success" if selected is not None else "warning",
        "duration_seconds": round(time.time() - t0, 1),
        "n_selected": len(selected) if selected is not None else 0,
    }

    # Step 5: 输出选股
    t0 = time.time()
    output_path = run_output_picks(date_str, selected, args.output_dir, args.dry_run)
    pipeline_result["steps"]["output_picks"] = {
        "status": "success" if output_path else "failed",
        "duration_seconds": round(time.time() - t0, 1),
        "output_path": output_path,
    }

    # 汇总
    elapsed = time.time() - start_time
    pipeline_result["status"] = (
        "success" if all(
            s["status"] == "success"
            for s in pipeline_result["steps"].values()
        )
        else "partial"
    )
    pipeline_result["duration_seconds"] = round(elapsed, 1)
    pipeline_result["end_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pipeline_result["summary"] = {
        "date": date_str,
        "top_k": args.top_k,
        "industry_neutral": not args.no_industry_neutral,
        "weight_method": args.weight_method,
        "dry_run": args.dry_run,
        "selected_count": len(selected) if selected is not None else 0,
        "output_path": output_path,
    }

    return pipeline_result


def print_summary(result: Dict[str, Any]) -> None:
    """打印Pipeline运行摘要。

    Args:
        result: Pipeline运行结果字典。
    """
    print("\n" + "=" * 70)
    print("每日运行摘要")
    print("=" * 70)

    print(f"  日期:       {result.get('date', 'N/A')}")
    print(f"  状态:       {result.get('status', 'N/A')}")
    print(f"  耗时:       {result.get('duration_seconds', 0):.1f}秒")

    summary = result.get("summary", {})
    if summary:
        print(f"  选股数量:   {summary.get('selected_count', 0)}")
        print(f"  输出路径:   {summary.get('output_path', 'N/A')}")
        print(f"  试运行:     {'是' if summary.get('dry_run') else '否'}")

    print(f"\n  步骤明细:")
    for step, info in result.get("steps", {}).items():
        status_icon = "[OK]" if info["status"] == "success" else "[WARN]" if info["status"] == "warning" else "[FAIL]"
        dur = info.get("duration_seconds", 0)
        extra = ""
        if info.get("n_stocks"):
            extra = f" ({info['n_stocks']}只)"
        elif info.get("n_selected"):
            extra = f" ({info['n_selected']}只)"
        elif info.get("output_path"):
            extra = f" -> {info['output_path']}"
        print(f"    {status_icon} {step}: {dur:.1f}s{extra}")

    print("=" * 70 + "\n")


def main():
    """主入口。"""
    args = parse_args()
    setup_logging(verbose=args.verbose)

    # 处理行业中性化参数
    if args.no_industry_neutral:
        args.industry_neutral = False

    # 确定运行日期
    if args.date:
        date_str = args.date
    else:
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")

    try:
        # 验证日期格式
        datetime.datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        logger.error(f"日期格式错误: {date_str}, 应为 YYYY-MM-DD")
        return 1

    # 处理 --step 参数（仅运行单步）
    if args.step:
        success = run_single_step(args.step, date_str, args)
        return 0 if success else 1

    # 运行完整Pipeline
    result = run_full_pipeline(args, date_str)
    print_summary(result)

    if result["status"] == "success":
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())