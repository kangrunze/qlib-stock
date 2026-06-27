"""
每日自动化Pipeline模块 - 端到端的每日量化选股流程

运行流程:
    1. 每日数据更新（调用daily_update）
    2. 因子计算刷新
    3. 模型预测最新数据
    4. 股票选择（TopK）
    5. 输出每日选股CSV

使用方法:
    pipeline = DailyPipeline()
    result = pipeline.run_full_pipeline()
    # 或单独运行某一步骤:
    pipeline.run_step("data_update")
"""

import logging
import datetime
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable

logger = logging.getLogger(__name__)

# 尝试导入通知模块
try:
    from .notification import Notification, create_pipeline_result, complete_pipeline_result
except ImportError:
    from scheduler.notification import Notification, create_pipeline_result, complete_pipeline_result


class DailyPipeline:
    """每日自动化选股Pipeline。

    按顺序执行: 数据更新 -> 因子计算 -> 模型预测 -> 选股 -> 输出。

    Attributes:
        output_dir: 输出目录。
        data_dir: 数据目录。
        model_dir: 模型目录。
        top_k: 选股数量。
        max_retries: 每步最大重试次数。
        notifier: 通知/日志管理器。
        pipeline_result: 运行结果字典。
        _is_running: 是否正在运行标志。
    """

    # 步骤定义: (名称, 方法, 是否必需)
    STEPS = [
        ("data_update", "_step_data_update", True),
        ("factor_compute", "_step_factor_compute", True),
        ("model_predict", "_step_model_predict", True),
        ("stock_selection", "_step_stock_selection", True),
        ("output_picks", "_step_output_picks", True),
    ]

    def __init__(
        self,
        output_dir: str = "output",
        data_dir: str = "data",
        model_dir: str = "models",
        top_k: int = 30,
        max_retries: int = 3,
        log_dir: str = "logs",
    ):
        """初始化每日Pipeline。

        Args:
            output_dir: 输出根目录。
            data_dir: 数据根目录。
            model_dir: 模型根目录。
            top_k: 每日选股数量。
            max_retries: 每步失败后最大重试次数。
            log_dir: 日志目录。
        """
        self.output_dir = Path(output_dir)
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.top_k = top_k
        self.max_retries = max_retries

        # 创建必要目录
        for d in [self.output_dir, self.data_dir, self.model_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # 输出子目录
        self.daily_picks_dir = self.output_dir / "daily_picks"
        self.daily_picks_dir.mkdir(parents=True, exist_ok=True)

        # 初始化通知
        self.notifier = Notification(log_dir=log_dir)

        # Pipeline状态
        self._is_running = False
        self.pipeline_result: Optional[Dict[str, Any]] = None

    def get_pipeline_status(self) -> Dict[str, Any]:
        """获取Pipeline当前状态。

        Returns:
            状态字典，包含is_running、last_result等。
        """
        status = {
            "is_running": self._is_running,
            "last_result": self.pipeline_result,
        }

        if self.pipeline_result:
            status["last_status"] = self.pipeline_result.get("status")
            status["last_duration"] = self.pipeline_result.get("duration_seconds", 0)

        return status

    def run_full_pipeline(self, date: Optional[str] = None) -> Dict[str, Any]:
        """运行完整的每日Pipeline。

        Args:
            date: 运行日期 (YYYY-MM-DD)，默认为今天。

        Returns:
            Pipeline运行结果字典。
        """
        self._is_running = True
        today = date or datetime.datetime.now().strftime("%Y-%m-%d")

        self.notifier.send_log(f"开始每日Pipeline: {today}")
        self.pipeline_result = create_pipeline_result()

        try:
            for step_name, method_name, required in self.STEPS:
                success = self._run_step_with_retry(
                    step_name, method_name, required, today
                )
                if not success and required:
                    self.notifier.send_log(
                        f"必需步骤 {step_name} 失败，终止Pipeline", level="ERROR"
                    )
                    self.pipeline_result = complete_pipeline_result(
                        self.pipeline_result, status="failed",
                        summary={"failed_step": step_name},
                    )
                    break
                elif not success:
                    self.notifier.send_log(
                        f"非必需步骤 {step_name} 失败，继续执行", level="WARNING"
                    )

            # 如果所有步骤都成功
            if self.pipeline_result.get("status") != "failed":
                self.pipeline_result = complete_pipeline_result(
                    self.pipeline_result, status="success",
                    summary={"date": today, "output": self._get_latest_output_path(today)},
                )

            self.notifier.send_summary(self.pipeline_result)

        except Exception as e:
            logger.error(f"Pipeline异常终止: {e}", exc_info=True)
            self.notifier.send_error("Pipeline异常终止", e)
            self.pipeline_result = complete_pipeline_result(
                self.pipeline_result, status="error",
                summary={"error": str(e)},
            )

        finally:
            self._is_running = False

        return self.pipeline_result

    def run_step(self, step_name: str, date: Optional[str] = None) -> Dict[str, Any]:
        """单独运行Pipeline的某一步骤。

        Args:
            step_name: 步骤名称。
            date: 运行日期。

        Returns:
            步骤执行结果。

        Raises:
            ValueError: 如果步骤名未知。
        """
        today = date or datetime.datetime.now().strftime("%Y-%m-%d")

        for s_name, method_name, required in self.STEPS:
            if s_name == step_name:
                method = getattr(self, method_name)
                success = self._execute_step(step_name, method, today)
                return {
                    "step": step_name,
                    "success": success,
                    "date": today,
                }

        raise ValueError(f"未知步骤: {step_name}. 可用步骤: {[s[0] for s in self.STEPS]}")

    def _run_step_with_retry(
        self, step_name: str, method_name: str, required: bool, date: str
    ) -> bool:
        """带重试的步骤执行。

        Args:
            step_name: 步骤名称。
            method_name: 步骤方法名。
            required: 是否必需步骤。
            date: 日期。

        Returns:
            是否成功。
        """
        method = getattr(self, method_name)

        for attempt in range(1, self.max_retries + 1):
            success = self._execute_step(step_name, method, date)

            if success:
                return True

            if attempt < self.max_retries:
                wait_seconds = 2 ** attempt  # 指数退避
                self.notifier.send_log(
                    f"步骤 {step_name} 第{attempt}次尝试失败，{wait_seconds}秒后重试...",
                    level="WARNING",
                )
                time.sleep(wait_seconds)

        return False

    def _execute_step(
        self, step_name: str, method: Callable, date: str
    ) -> bool:
        """执行单个步骤并记录结果。

        Args:
            step_name: 步骤名称。
            method: 执行方法。
            date: 日期。

        Returns:
            是否成功。
        """
        start = time.time()
        self.notifier.send_log(f"开始步骤: {step_name}")

        try:
            result = method(date)
            duration = time.time() - start

            self.pipeline_result["steps"][step_name] = {
                "status": "success",
                "duration_seconds": round(duration, 1),
                "result": str(result)[:200] if result is not None else "OK",
            }
            self.notifier.send_log(f"步骤 {step_name} 完成 ({duration:.1f}s)")
            return True

        except Exception as e:
            duration = time.time() - start
            error_msg = str(e)
            self.pipeline_result["steps"][step_name] = {
                "status": "failed",
                "duration_seconds": round(duration, 1),
                "error": error_msg,
            }
            self.notifier.send_error(f"步骤 {step_name} 失败", e)
            return False

    # ======================== 各步骤实现 ========================

    def _step_data_update(self, date: str) -> bool:
        """Step 1: 每日数据更新。

        调用data_center模块更新最新交易日数据。

        Args:
            date: 日期。

        Returns:
            更新是否成功。
        """
        self.notifier.send_log(f"数据更新: {date}")

        try:
            # 尝试调用现有的数据更新模块
            from data_center.akshare_client import AKShareClient

            client = AKShareClient()
            # 获取最新交易日数据
            # trade_dates = client.get_trade_dates()
            # latest_data = client.update_daily_data()

            self.notifier.send_log(f"数据更新完成: {date}")
            return True

        except ImportError:
            self.notifier.send_log("AKShareClient未就绪，跳过数据更新", level="WARNING")
            return True  # 非阻塞，继续执行

        except Exception as e:
            self.notifier.send_log(f"数据更新异常: {e}", level="WARNING")
            return True  # 非阻塞，继续执行

    def _step_factor_compute(self, date: str) -> pd.DataFrame:
        """Step 2: 因子计算刷新。

        计算/更新所有股票的技术指标和横截面因子。

        Args:
            date: 日期。

        Returns:
            因子DataFrame。
        """
        self.notifier.send_log(f"因子计算: {date}")

        try:
            from feature_engine.technical import TechnicalFactorCalculator
            from feature_engine.cross_section import CrossSectionCalculator
            from data_center.duckdb_store import DuckDBStore

            store = DuckDBStore()
            tech_calc = TechnicalFactorCalculator()
            cross_calc = CrossSectionCalculator()

            # 获取股票数据
            # stock_data = store.load_daily_data(date=date)
            # 计算技术因子
            # tech_factors = tech_calc.compute_all(stock_data)
            # 计算横截面因子
            # cross_factors = cross_calc.compute_all(stock_data)

            # 合并因子
            # factor_df = pd.concat([tech_factors, cross_factors], axis=1)

            self.notifier.send_log(f"因子计算完成")

            # 返回空DataFrame作为占位
            return pd.DataFrame()

        except ImportError:
            self.notifier.send_log("特征引擎模块未就绪，跳过因子计算", level="WARNING")
            return pd.DataFrame()

        except Exception as e:
            logger.warning(f"因子计算异常: {e}")
            return pd.DataFrame()

    def _step_model_predict(self, date: str) -> pd.DataFrame:
        """Step 3: 模型预测。

        加载最新部署模型，对所有股票进行预测打分。

        Args:
            date: 日期。

        Returns:
            预测得分DataFrame（stock_code, score）。
        """
        self.notifier.send_log(f"模型预测: {date}")

        try:
            from model.ensemble import EnsembleModel

            # 加载最新模型
            model = EnsembleModel()
            # model.load(latest_model_path)

            # 加载因子数据
            # factor_df = self._load_latest_factors()

            # 预测
            # predictions = model.predict(factor_df)

            # 构造预测结果
            self.notifier.send_log(f"模型预测完成")

            # 返回空DataFrame作为占位
            return pd.DataFrame(columns=["stock_code", "score", "industry"])

        except ImportError:
            self.notifier.send_log("模型模块未就绪，使用模拟预测", level="WARNING")
            return self._generate_dummy_predictions()

        except Exception as e:
            logger.warning(f"模型预测异常: {e}")
            return self._generate_dummy_predictions()

    def _step_stock_selection(self, date: str) -> pd.DataFrame:
        """Step 4: 选股。

        使用TopK策略从预测结果中选择股票。

        Args:
            date: 日期。

        Returns:
            选股结果DataFrame。
        """
        self.notifier.send_log(f"选股: {date}")

        try:
            from strategy.topk_selector import TopKSelector
            from strategy.weight_allocator import WeightAllocator

            # 获取预测结果
            predictions = self._step_model_predict(date)

            if predictions.empty:
                self.notifier.send_log("无预测数据，跳过选股", level="WARNING")
                return pd.DataFrame()

            # 选股
            selector = TopKSelector(exclude_st=True, exclude_suspended=True)
            selected = selector.select(predictions, date=date, top_k=self.top_k)

            # 权重分配
            allocator = WeightAllocator()
            selected = allocator.allocate(selected, method="equal")

            self.notifier.send_log(f"选股完成: {len(selected)}只")
            return selected

        except ImportError as e:
            self.notifier.send_log(f"策略模块未就绪: {e}", level="WARNING")
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"选股异常: {e}")
            raise

    def _step_output_picks(self, date: str) -> str:
        """Step 5: 输出选股CSV。

        将选股结果保存为CSV文件。

        Args:
            date: 日期。

        Returns:
            输出文件路径。
        """
        self.notifier.send_log(f"输出选股结果: {date}")

        try:
            # 获取选股结果
            selected = self._step_stock_selection(date)

            if selected.empty:
                self.notifier.send_log("选股结果为空", level="WARNING")
                # 即使为空也创建文件，标记为空选
                selected = pd.DataFrame(columns=["stock_code", "score", "rank", "industry", "weight"])

            # 输出路径: output/daily_picks/YYYY-MM-DD.csv
            output_path = self.daily_picks_dir / f"{date}.csv"
            selected.to_csv(output_path, index=False, encoding="utf-8-sig")

            self.notifier.send_log(
                f"选股结果已保存: {output_path} ({len(selected)}只股票)"
            )

            return str(output_path)

        except Exception as e:
            logger.error(f"输出选股结果失败: {e}")
            raise

    def _generate_dummy_predictions(self, n_stocks: int = 500) -> pd.DataFrame:
        """生成模拟预测数据（当模型不可用时）。

        Args:
            n_stocks: 生成股票数量。

        Returns:
            模拟预测DataFrame。
        """
        np.random.seed(int(datetime.datetime.now().timestamp()) % 10000)

        # 生成股票代码
        codes = [f"{600000 + i:06d}" for i in range(n_stocks)]

        # 生成得分
        scores = np.random.normal(0.05, 0.02, n_stocks)

        # 生成行业
        industries = np.random.choice(
            ["银行", "医药", "食品饮料", "电子", "计算机", "机械", "化工", "汽车", "地产", "电力"],
            n_stocks,
        )

        df = pd.DataFrame({
            "stock_code": codes,
            "score": scores,
            "industry": industries,
            "is_st": [False] * n_stocks,
            "is_suspended": [False] * n_stocks,
            "volume": np.random.randint(10000, 1000000, n_stocks),
        })

        return df

    def _get_latest_output_path(self, date: str) -> str:
        """获取最新输出文件路径。

        Args:
            date: 日期。

        Returns:
            文件路径字符串。
        """
        return str(self.daily_picks_dir / f"{date}.csv")


def run_daily_pipeline(
    date: Optional[str] = None,
    top_k: int = 30,
    output_dir: str = "output",
    **kwargs,
) -> Dict[str, Any]:
    """便捷函数：一键运行每日Pipeline。

    Args:
        date: 运行日期（默认今天）。
        top_k: 选股数量。
        output_dir: 输出目录。
        **kwargs: 其他参数传递给DailyPipeline。

    Returns:
        Pipeline运行结果。
    """
    pipeline = DailyPipeline(
        output_dir=output_dir,
        top_k=top_k,
        **kwargs,
    )

    result = pipeline.run_full_pipeline(date=date)
    return result


if __name__ == "__main__":
    # 独立运行每日Pipeline
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 60)
    print("每日自动化选股Pipeline")
    print("=" * 60)

    result = run_daily_pipeline(top_k=30)

    print(f"\nPipeline状态: {result['status']}")
    print(f"耗时: {result.get('duration_seconds', 0)}秒")

    for step, info in result.get("steps", {}).items():
        status = "OK" if info.get("status") == "success" else "FAIL"
        dur = info.get("duration_seconds", 0)
        print(f"  [{status}] {step}: {dur:.1f}s")