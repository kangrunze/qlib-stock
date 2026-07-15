Top 5 配置排名总览
排名	配置文件	超额(扣后)	IR(扣后)	换手率	成本拖累	best_iter
1	rank1_csi500_optuna22_ndrop1	+15.19%	+1.074	17.30x	3.46%	1
2	rank2_csi500_optuna22_ndrop2	-0.33%	-0.025	32.31x	6.46%	1
3	rank3_csi500_optuna22_ndrop3	-6.01%	-0.086	47.23x	9.45%	1
4	rank4_csi500_optuna37_ndrop3	-7.60%	-0.599	49.72x	9.45%	45
5	rank5_csi500_optuna37_ndrop5	-11.73%	-0.946	78.90x	7.41%	45
参数解读：两组 Optuna 参数对比
参数	Optuna#37 (csi300调参)	Optuna#22 (csi500调参)	差异解读
learning_rate	0.0428	0.0275	csi500 学习率更低，因数据量大需更保守收敛
max_depth	3	4	csi500 更深树，数据量大可承受更复杂结构
num_leaves	31	55	csi500 更多叶子节点，配合更深的树捕获更细模式
subsample	0.783	0.999	csi500 几乎不采样，数据量大无需降采样防过拟合
colsample_bytree	0.743	0.697	相近，csi500 略低
lambda_l1	0.139	~0 (1.2e-7)	csi500 几乎不需 L1 正则化
lambda_l2	77.54	~0 (1.3e-8)	csi500 几乎不需 L2 正则化
min_child_samples	68	76	相近，csi500 略大
核心差异：csi300 股票池小（~300 只），模型容易过拟合，需要强 L2 正则化（77.5）+ 浅树（depth=3）防止过拟合。csi500 股票池大（~800 只），数据量充足，模型可承受更复杂结构（depth=4, leaves=55）+ 极弱正则化，让模型自由学习。

n_drop 参数解读
n_drop	换手率	成本拖累	超额(扣前)	超额(扣后)	解读
5	78.90x	7.41%	-4.32%	-11.73%	每期换 5 只，高频换仓，成本侵蚀大
3	47.23x	9.45%	+3.45%	-6.01%	每期换 3 只，中等换手
2	32.31x	6.46%	+2.68%	-0.33%	每期换 2 只，低换手，接近盈亏平衡
1	17.30x	3.46%	+18.65%	+15.19%	每期换 1 只，极低换手，Alpha 充分保留
核心逻辑：ret_60d 标签预测的是 60 日未来收益，选股信号在较长时间内有效。n_drop=1 每期仅替换 1 只股票（30 只中换 1 只 = 3.3% 替换率），最大程度保留了已有持仓的信号价值，同时将交易成本降至 3.46%。

各配置文件详解
Rank 1: csi500_optuna22_ndrop1（最优配置）
查看配置文件

股票池：csi500（~800 只中证500成分股）
标签：ret_60d（60 日未来收益率）
损失函数：rank（LambdaRank 排序优化）
参数来源：csi500 专属 Optuna 50 轮搜索 trial #22
n_drop：1（每期仅替换 1 只，最低换手）
绩效：扣后超额 +15.19%，IR=1.074
适用场景：中低频量化策略，持仓周期约 60 日
Rank 2: csi500_optuna22_ndrop2
查看配置文件

与 Rank 1 仅 n_drop 不同（2 vs 1）
绩效：扣后超额 -0.33%，几乎盈亏平衡
适用场景：需要在信号新鲜度和成本之间折中
Rank 3: csi500_optuna22_ndrop3
查看配置文件

与 Rank 1 仅 n_drop 不同（3 vs 1）
绩效：扣后超额 -6.01%，扣前正超额 +3.45%
适用场景：中频策略，接受一定换手成本换取更多选股机会
Rank 4: csi500_optuna37_ndrop3
查看配置文件

参数来源：csi300 Optuna#37 直接用于 csi500（未针对 csi500 调优）
绩效：扣后超额 -7.60%，best_iter=45 说明模型有效但参数非最优
适用场景：对照实验，验证 csi500 专属调参的必要性
Rank 5: csi500_optuna37_ndrop5
查看配置文件

参数来源：csi300 Optuna#37 + 高换手 n_drop=5
绩效：扣后超额 -11.73%，高换手导致成本侵蚀
适用场景：基线对照，展示未优化前的表现
使用方式
每个配置文件均为独立的完整 workflow_config.yaml，可直接通过 --config 参数使用：


PowerShell

# 完整训练+回测
python run.py full --config configs/top5/rank1_csi500_optuna22_ndrop1.yaml

# 复用已训练模型仅回测（需指定 recorder_id）
python run.py backtest --rid 8a6ad9fa248445a7b1288339b8a01edc --experiment qlib_pipeline --config configs/top5/rank1_csi500_optuna22_ndrop1.yaml
当前主配置文件 qlib_pipeline/workflow_config.yaml 已设为 Rank 1 最优配置（csi500 + Optuna#22 + n_drop=1）。