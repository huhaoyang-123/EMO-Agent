# 2026-11-26 接收交接数据与确立开发规范

对应 [Issue #18](https://github.com/huhaoyang-123/EMO-Agent/issues/18)，分支 `chore/18-adopt-analysis-agent-spec`，起点为主线 `c32fbc5`。本次交付为本地数据就位和规范落库。发布状态见 Issue 关联的 PR。

## 开发规范

用户提供的《EMO-Agent：金融市场分析智能体开发文档》v1.0 完整保存在 `docs/agent_development.md`，与原件逐字节一致，SHA-256 为 `0d1ec6a678a65a00b2d61c9dc1c4e6147ec041e56a73b8f8ccc5debc86a2335e`。README、AGENTS 和 PROGRESS 均指向该规范。确定性 Python 工具计算金融指标，单 Agent 负责调用与解释，基于现有行情、股吧统计和研究报告提供历史分析。规范列出的配置、数据加载器、工具调用和 Streamlit 界面属于后续实施任务。

## 本地数据迁入

两份原始 ZIP 从桌面移动到 `data/raw/handoffs/2026-11-26/`。迁移前后哈希一致，原位置已不存在这两个 ZIP。按包内路径展开 313 个文件，检查路径范围和冲突后排他创建，并逐个核对解包前后字节哈希。

| ZIP | 解包文件数 | ZIP SHA-256 |
|---|---:|---|
| `task3_data_and_outputs_v2.zip` | 301 | `fa2fa6b4233ae0b3eb4925e514f68b6e58e0a9c7174877b2e81f2a0cd123db83` |
| `task6_shadow_outputs.zip` | 12 | `8aaefbfe09a84c39971b84fa89282d5628aa033f49c6b2fd70cc51d52573bafb` |

- task3 的两份 CSMAR 输入位于 `data/input/context/`。中证全指为 970 行，2022-10-10 至 2026-10-09；成分股日收益为 59,972 行，2022-12-01 至 2026-10-19。派生表和元数据共 2 份，位于 `data/derived/context_v1/`，覆盖 645 个日期，其中 625 个符合建模条件。
- task3 的两次正式运行位于 `outputs/context_v1/` 和 `outputs/context_v1b/`。两目录各含一份 `run.json`，以及由其完整列出的 127 和 168 个产物，包含 120 和 160 个模型文件。输入、产物、配置及现有 `cutoff_v1` 的引用哈希逐项匹配。
- task6 位于 `outputs/shadow_v1/`，冻结记录哈希为 `8f4189297818a1bdd8e65aad1a4b521cf208ba5e49d5c7eff96754ba3453db9c`。包内实物为 6 个拟合模型文件；两个先验概率直接写在冻结记录中。旧执行记录所写的“7 个 joblib”与包内实物及哈希清单不符。影子日志含 73 日期、584 行，均为 `retro_batch`，评估窗口使用其中 67 日期。日志引用的输入与模型哈希吻合。
- 解包后的真实数据、模型、预测和压缩包均由 Git 忽略。基础输入清单维持原 17 项；交接文件使用上表与包内清单追踪。

## 验证及下一步

`uv sync --locked`、`pytest`（69 项通过、6 条既有日期算术弃用警告）、CLI 帮助及差异检查通过。规范原文 8 处 Markdown 硬换行使用行尾双空格，单独检查该文件时保留原样，其余文件使用默认差异检查。以 `audit-data` 的 `audit_project` 审计函数只读检查 17 个基础输入及对账，保留现有审计输出文件。另对三份派生数据（`cutoff_v1`、`company_pilot_v1`、`context_v1`）的 8 个元数据输出哈希做了检查，均通过。未反序列化收到的模型，未重训或重跑历史实验。

下一阶段按开发规范第 9 节启动 Agent 配置、数据可用性接口与分析工具，并通过独立 Issue 和 PR 验收。
