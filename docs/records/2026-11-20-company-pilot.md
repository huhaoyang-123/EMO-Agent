# 阶段6开发记录：公司层面有限试点

日期2026-11-20，Issue #11，分支feat/11-company-pilot，前置PR #10核实合并。从origin/main开始；本轮PR #12已于2026-11-21创建，待审未合并；发布状态以PROGRESS.md及次日收尾记录为准。

用户授权有限个股试点，覆盖历史成分，保留指数参照。方案提交e2e3b01在评分之前；实现及两次正式运行提交ce43b49dbcbb0481dc995d86dc1d95bd902cf940。未看到结果后变更特征、参数、时间切分、门槛或阈值。

新增company.py、两个CLI命令、19项针对性测试、独立verify_company_pilot.py；更新真实审计、新数据清单与范围文档。原股票日线及字段字典直接移动，核验记录2026-11-20-stock-migration.json；没有复制多余的源工作簿。

输入版本为data_manifest.json v2、既有cutoff_v1；派生data/derived/company_pilot_v1；正式输出outputs/company_pilot_v1。数据/模型不上Git。SHA-256及环境版本记录于metadata.json/run.json。

实际执行：uv sync --locked通过；pytest 61 passed/6既有警告；audit-data 17文件通过；git diff --check通过（最终文档后还会复查）。公司面板构建、固定72模型运行、保存回读概率一致。独立验证：{"ok": true, "label_rows_checked": 64456, "turnover_rows_checked": 32228, "source_ids_checked": 83653, "independent_metric_groups": 32, "independent_gain_groups": 42, "reproduced_artifacts": 83, "run_code_commit": "ce43b49dbcbb0481dc995d86dc1d95bd902cf940", "checks": ["raw_price_labels", "global_calendar", "turnover_units", "source_trace_and_counts", "mature_splits", "prediction_outcomes", "date_equal_metrics", "paired_gains", "artifact_hashes"]}。

完整复跑run-id=company_pilot_v1_repro；83个模型/预测/指标/配置产物哈希一致，run.json的run_id自然不同。删除该复跑目录被自动审批拒绝（工具仅返回blocked by policy），不绕过；暂保留该目录，正式产物正常。

首次合成测试2失败/17通过：GroupBy.observed为同名布尔属性，显式grouped['observed']访问修复。新合成时段使用Python datetime规避依赖弃用警告；旧测试的6警告保留。本次失败发生于真实构建/评分前，未放宽断言或更改实验口径。

结果：2025公司方向增益两个期限为负，2026的5日微弱正值不能支撑继续门槛；完整数值、公司与指数对照、数学公式见docs/results/company_pilot_v1.md。暂未证明转向个股产生更强的正预测信息。固定共享斜率与正则强度、历史统计标签/真实可得时间未核实、复权与盘后口径、已接触历史期间都是解释边界。

未执行：新源采集、投资者人数、事件、更多模型/公司特定斜率搜索、Agent前端、最新部署模型、前瞻收益评估、自动合并。下一步是审阅并冻结本轮；任何新的信息假设另立范围。
