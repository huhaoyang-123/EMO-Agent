# 本地研究数据

真实数据保存在本目录并被 Git 忽略。GitHub 克隆后不会自动获得 CNRDS 数据。

授权成员通过团队已有授权渠道获取同一份数据包，按 `docs/data_manifest.json` 的 path 放置：`raw/guba/` 为四个原始 ZIP；`input/market/` 为四个行情/成员/日历快照；`input/sentiment/` 为六个 CSV 和 summary.json。勿将研究数据上传 Issues、PR 或公开附件。

从项目根运行 `python -m uv run --locked emo-agent audit-data` 检查全部文件与哈希。报告中任何 missing/hash_mismatch 都必须先解决；基础 pytest 不依赖真实数据，可先验证环境。需要更新数据时保留新版本目录，重新生成有审阅记录的清单，不静默覆盖当前 input。

未来派生表放 `data/derived/`；模型、预测与评估报告放 `outputs/`。上游行情原始工作簿未包含在当前资料中，所以成员可复现的起点是已提供的行情 CSV 快照；情绪可从原始 ZIP 重建。

阶段6新增两份只读输入：input/market/stock_daily.csv 与 input/market/stock_daily_dictionary.txt，均已列入清单。原目录直接移动，不保留重复副本。迁移记录 docs/records/2026-11-20-stock-migration.json；不要把数据上传GitHub。公司面板从既有cutoff_v1与这些快照构建，详见个股报告。

## 任务三与任务六交接（2026-11-26）

团队通过授权渠道交接 ZIP，按包内路径还原。压缩包在本地 `data/raw/handoffs/2026-11-26/` 归档。

| 交接包 | 本地路径 | 校验依据 |
|---|---|---|
| `task3_data_and_outputs_v2.zip` | `data/input/context/`、`data/derived/context_v1/`、`outputs/context_v1/` 和 `outputs/context_v1b/` | 派生数据的 `metadata.json` 和两个运行目录的 `run.json` |
| `task6_shadow_outputs.zip` | `outputs/shadow_v1/` | `freeze.json` 中记录的输入、配置和模型哈希，以及 `shadow_log.csv` 的逐行引用 |

`audit-data` 的清单仍覆盖原 17 份基础输入。两项交接另按各自元数据校验。中证全指行情截至 2026-10-09，个股日收益截至 2026-10-19，情绪统计截至 2026-06-30。准备 Agent 时按数据源分别展示可用日期。[开发规范](../docs/agent_development.md) · [交接记录](../docs/records/2026-11-26-agent-spec-handoff.md)
