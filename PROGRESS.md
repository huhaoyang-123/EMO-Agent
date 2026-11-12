# 项目进度

更新日期：2026-11-12（Asia/Shanghai）

## 当前阶段

阶段 2 已完成本地构建与验证：625 个可用预测日期，27 项测试通过。阶段 1 已验证，PR #4 尚未合并；阶段 2 从已验证基础分支继续，使用依赖 PR。

[Issue #1](https://github.com/huhaoyang-123/EMO-Agent/issues/1) · [PR #4](https://github.com/huhaoyang-123/EMO-Agent/pull/4) · 分支 `chore/1-project-foundation`。

## 已完成

- 19 个必要数据/来源说明/复现文件已直接迁入，逐个 SHA-256 核验，原路径不保留副本。
- 明确 931723、行情＋情绪、1 日/5 日上涨概率；事件与 ETF 暂不加入。
- README、AGENTS、研究计划、数据字典、数据清单、Issue/PR 模板与基础实现已建立。

## 验证与发布状态

- Python 3.13.7 + uv 0.13.7，`python -m uv sync --locked` 通过，uv.lock 锁定依赖与哈希。
- `python -m uv run --locked pytest`：21 passed。
- 本地真实数据审计通过：15 文件哈希一致，2,376 行行情、83,653 行情绪、3,044,795 篇帖子；日度合计一致。
- CLI 帮助、语法检查通过。远程 CI 和 PR 状态在 Issue #1 与关联 PR 中查看；尚未合并。


## 下一步

1. [Issue #2](https://github.com/huhaoyang-123/EMO-Agent/issues/2)：完成按 16:00 截止的分时情绪聚合和每日共同样本；不能直接使用现有行业统计日表作为预测输入。
2. [Issue #3](https://github.com/huhaoyang-123/EMO-Agent/issues/3)：冻结样本、特征和时间切分，训练行情/行情＋情绪 × 1 日/5 日四个逻辑回归模型。
3. 输出逐日预测与准确率、概率误差、分月表现，保留最终测试独立性。

## 已知限制

- 现有共同范围约 645 个交易日；有效样本数量尚未确定。
- 历史情绪计数何时对外可见及是否修订未核实。
- 统计日表包含星期五晚间时段，直接用于当天 16:00 预测会泄漏。
- 行情有发布前回算记录；日历 `split` 是旧方案，不可沿用到新实验。
- 阶段 2 样本已保存 data/derived/cutoff_v1/，详见 docs/records/2026-11-12-dataset.md。
- 下一步进入阶段 3。尚未训练模型，准确率未知。无最新行情抓取或实时 Agent。
