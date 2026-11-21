# 本地研究数据

真实数据保存在本目录并被 Git 忽略。GitHub 克隆后不会自动获得 CNRDS 数据。

授权成员通过团队已有授权渠道获取同一份数据包，按 `docs/data_manifest.json` 的 path 放置：`raw/guba/` 为四个原始 ZIP；`input/market/` 为四个行情/成员/日历快照；`input/sentiment/` 为六个 CSV 和 summary.json。勿将研究数据上传 Issues、PR 或公开附件。

从项目根运行 `python -m uv run --locked emo-agent audit-data` 检查全部文件与哈希。报告中任何 missing/hash_mismatch 都必须先解决；基础 pytest 不依赖真实数据，可先验证环境。需要更新数据时保留新版本目录，重新生成有审阅记录的清单，不静默覆盖当前 input。

未来派生表放 `data/derived/`；模型、预测与评估报告放 `outputs/`。上游行情原始工作簿未包含在当前资料中，所以成员可复现的起点是已提供的行情 CSV 快照；情绪可从原始 ZIP 重建。

阶段6新增两份只读输入：input/market/stock_daily.csv 与 input/market/stock_daily_dictionary.txt，均已列入清单。原目录直接移动，不保留重复副本。迁移记录 docs/records/2026-11-20-stock-migration.json；不要把数据上传GitHub。公司面板从既有cutoff_v1与这些快照构建，详见个股报告。
