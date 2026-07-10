# -*- coding: utf-8 -*-
"""
Qlib Pipeline Model 配置

提供 LGBModel 配置的兜底创建函数 + RankLGBModel 排序学习子类。

背景:
  Qlib 0.9.7 的 LGBModel.__init__ 对 loss 参数做了白名单校验:
    if loss not in {"mse", "binary"}: raise NotImplementedError
  这意味着 loss="rank" 从未真正跑通过。要使用 LambdaRank 排序目标,
  需要绕过这个校验, 直接设置 LightGBM 原生的 objective="lambdarank"。

  本模块通过 RankLGBModel 子类绕过校验:
    - loss 参数接受 "rank" 但不传给父类（避免触发 NotImplementedError）
    - 在 params 中直接设置 objective="lambdarank"
    - LambdaRank 需要 group 参数（按日期分组排序）, 在 _prepare_data 中自动构造
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RankLGBModel:
    """LambdaRank 排序学习模型（绕过 Qlib LGBModel 的 loss 白名单校验）。

    Qlib 0.9.7 的 LGBModel 只支持 loss="mse"/"binary"，不支持排序目标。
    本类通过继承并覆写 __init__ 和 _prepare_data，直接使用 LightGBM 原生的
    objective="lambdarank" 实现选股排序学习。

    与原生 LGBModel 的区别:
      1. loss="rank" 不会触发 NotImplementedError
      2. objective 强制设为 "lambdarank"
      3. _prepare_data 自动按日期分组构造 group 供 LambdaRank 使用
      4. 标签会被转为 rank（LambdaRank 要求非负整数标签）

    注意: LambdaRank 优化的是排序质量（NDCG），而非绝对值，
          适合"选股排序"场景，预测值的绝对大小无意义，只有相对排序有意义。
    """

    def __init__(self, loss="rank", early_stopping_rounds=50, num_boost_round=1000, **kwargs):
        from qlib.contrib.model.gbdt import LGBModel

        # 不调用 LGBModel.__init__（它会校验 loss），直接复制其初始化逻辑
        # 但 objective 用 lambdarank 而非 loss
        self.params = {"objective": "lambdarank", "verbosity": -1}
        # 参数名转换：sklearn API 名 → LightGBM 原生名
        # lgb.train() 不识别 subsample/colsample_bytree，必须转为 bagging_fraction/feature_fraction
        # 否则这两个参数会被静默忽略，导致 Optuna 搜索 subsample/colsample 时所有 trial 等效
        param_map = {
            "subsample": "bagging_fraction",
            "colsample_bytree": "feature_fraction",
            "subsample_freq": "bagging_freq",
        }
        converted = {}
        for k, v in kwargs.items():
            if k in param_map:
                converted[param_map[k]] = v
            else:
                converted[k] = v
        # bagging_freq 默认设为 1，使 bagging_fraction 生效（否则 LightGBM 默认 freq=0 不抽样）
        if "bagging_fraction" in converted and "bagging_freq" not in converted:
            converted["bagging_freq"] = 1
        self.params.update(converted)
        self.early_stopping_rounds = early_stopping_rounds
        self.num_boost_round = num_boost_round
        self.model = None
        logger.info("RankLGBModel: 使用 LambdaRank 排序目标 (objective=lambdarank)")

    def _prepare_data(self, dataset, reweighter=None):
        """构造 LambdaRank 需要的数据：特征 + 标签 + 按日期分组的 group。"""
        from qlib.data.dataset import DatasetH
        from qlib.data.dataset.handler import DataHandlerLP

        ds_l = []
        assert "train" in dataset.segments
        for key in ["train", "valid"]:
            if key in dataset.segments:
                df = dataset.prepare(key, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
                if df.empty:
                    raise ValueError("Empty data from dataset, please check your dataset config.")
                x, y = df["feature"], df["label"]

                if y.values.ndim == 2 and y.values.shape[1] == 1:
                    y = np.squeeze(y.values)
                else:
                    raise ValueError("LightGBM doesn't support multi-label training")

                import lightgbm as lgb

                # LambdaRank 需要按日期分组：每个交易日内的股票构成一个 group
                # index 是 MultiIndex (datetime, instrument)，level=0 是 datetime
                if isinstance(x.index, pd.MultiIndex):
                    dates = x.index.get_level_values(0)
                    # 计算每个日期组的样本数
                    group_sizes = pd.Series(dates).groupby(dates).size().values
                else:
                    # 非多索引，整段作为一个 group
                    dates = None
                    group_sizes = np.array([len(x)])

                # LambdaRank 要求标签为非负整数（代表相关性等级）
                # 将连续收益率标签按日期内分位数转为 0-4 的整数等级
                y_ranked = self._labels_to_rank(y, dates)

                lgb_ds = lgb.Dataset(x.values, label=y_ranked, group=group_sizes)
                ds_l.append((lgb_ds, key))
        return ds_l

    def _labels_to_rank(self, y, dates):
        """将连续标签转为 LambdaRank 需要的非负整数等级（按日期内分位数）。"""
        if dates is None:
            # 无日期分组，全局分位数
            ranks = pd.qcut(y, q=5, labels=False, duplicates="drop")
            return np.asarray(ranks, dtype=int).clip(0, 4)
        # 按日期分组，每日内分 5 档
        y_series = pd.Series(y, index=dates)
        ranked = np.zeros(len(y), dtype=int)
        for dt in y_series.index.unique():
            mask = y_series.index == dt
            vals = y_series[mask].values
            if len(vals) < 5:
                ranked[mask] = 2  # 样本太少，统一给中间等级
            else:
                ranks = pd.qcut(vals, q=5, labels=False, duplicates="drop")
                ranked[mask] = np.asarray(ranks, dtype=int).clip(0, 4)
        return ranked

    def fit(self, dataset, num_boost_round=None, **kwargs):
        """训练 LambdaRank 模型。"""
        import lightgbm as lgb

        ds_l = self._prepare_data(dataset)
        num_boost_round = num_boost_round or self.num_boost_round

        train_ds = ds_l[0][0]
        valid_ds_list = [(ds, name) for ds, name in ds_l[1:]]

        callbacks = []
        if self.early_stopping_rounds and valid_ds_list:
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds))

        self.model = lgb.train(
            self.params,
            train_ds,
            num_boost_round=num_boost_round,
            valid_sets=[d for d, _ in valid_ds_list],
            valid_names=[n for _, n in valid_ds_list],
            callbacks=callbacks,
        )
        logger.info("RankLGBModel 训练完成, best_iteration=%d", self.model.best_iteration)
        return self

    def predict(self, dataset, segment="test"):
        """预测：返回每个样本的排序分数（绝对值无意义，只有相对排序有意义）。"""
        from qlib.data.dataset.handler import DataHandlerLP

        if self.model is None:
            raise ValueError("model not fitted")
        df = dataset.prepare(segment, col_set=["feature"], data_key=DataHandlerLP.DK_I)
        return pd.Series(self.model.predict(df["feature"].values), index=df["feature"].index)

    def feature_importance(self, **kwargs):
        """返回特征重要性。"""
        if self.model is None:
            raise ValueError("model not fitted")
        return pd.Series(
            self.model.feature_importance(importance_type="gain"),
            index=self.model.feature_name(),
        ).sort_values(ascending=False)


def create_rank_model(loss: str = "mse", **overrides) -> dict:
    """创建 LightGBM 模型配置字典（兜底，不含完整超参）。

    正常路径：workflow_config.yaml → qlib_lgb.kwargs → build_task()
    兜底路径：配置文件缺失 qlib_lgb 段时调用本函数

    当 loss="rank" 时返回 RankLGBModel 配置（绕过 LGBModel 的 loss 白名单校验）。
    """
    if loss == "rank":
        model_cfg = {
            "class": "RankLGBModel",
            "module_path": "qlib_pipeline.model",
            "kwargs": {
                "loss": "rank",
                "num_threads": 4,
            },
        }
    else:
        model_cfg = {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": loss,
                "num_threads": 4,
            },
        }
    model_cfg["kwargs"].update(overrides)
    return model_cfg