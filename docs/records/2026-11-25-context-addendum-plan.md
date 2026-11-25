# 任务三 补充方案（评分前提交）：剩余情绪与校准

日期：2026-11-25。交接文档“任务三”验收要求包含“剩余情绪模型在训练内拟合”和报告校准；第一轮 context_v1 遗漏了这两项（已看过 context_v1 结果，诚实记录）。本补充只执行交接文档事先要求的检查，不新增其它候选、不改窗口和指数。配置 configs/context_v1b.json；context_v1 的模型与结果保持不变，新运行使用新的 run-id context_v1b。

## 剩余情绪 R
- 对 4 个情绪内容变量（net_sentiment、equal_weight_sentiment、sentiment_change_5、post_heat_20）分别用 OLS（含截距）对 M 的 6 个行情变量 + G 的 3 个大盘变量回归，取残差作为“无法被行情解释的情绪”。
- 输入先用拟合集中位数填补。选 C 时残差模型只在 train 上拟合并用于 validation；最终模型在 refit 上拟合残差模型，评价期只应用、不重新估计。
- 覆盖类 3 列（company_coverage、slot_coverage、sentiment_missing）保持原值一并加入，与 S 组列数一致。
- 新模型：MGR、MGBR。比较：R|MG、R|MGB、MGR 相对 MGS。判定规则与 context_v1 相同。

## 校准
对所有模型报告 2025 合并：平均预测概率、实际上涨比例、10 分箱 ECE、校准截距与斜率（y 对 logit(p) 的无正则逻辑回归）。只作描述，不用于选择模型。

## 变量字典
新增变量写入 docs/data_dictionary.md。
