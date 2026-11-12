# EMO Agent

研究对象为中证光通信主题指数 **931723**。使用历史行情与成分股股吧情绪，分别估计下一交易日、未来第 5 个交易日收盘价高于当日收盘价的概率。第一版采用逻辑回归，比较「行情」与「行情＋情绪」。暂不加入事件、ETF 或自动交易。

## 当前阶段

阶段 1：协作与可复现基础。数据已迁入本地 `data/`；尚未清洗成最终模型样本，尚未训练模型，准确率未知。进度以 [PROGRESS.md](PROGRESS.md) 为准。

后续：截止时点安全的共同样本 → 四个逻辑回归模型 → 冻结测试与结果解释 → 每日运行 Agent。

## 启动

需要 Python 3.13、Git 和 uv 0.13.7。Windows/macOS/Linux 均从仓库根目录执行；示例不依赖本机绝对路径。

```sh
python -m pip install uv==0.13.7
python -m uv sync --locked
python -m uv run --locked pytest
python -m uv run --locked emo-agent --help
```

克隆仓库只包含代码、文档、合成测试数据和哈希清单。真实数据未上传 GitHub：授权成员按 [data/README.md](data/README.md) 放置同一批数据，然后运行：

```sh
python -m uv run --locked emo-agent audit-data
```

报告默认写入 `outputs/audits/data_audit.json`。没有数据时审计命令会明确报错；基础测试仍可独立运行。

可选重建历史情绪快照（扫描约千万条记录，需要足够内存和时间）：

```sh
python -m uv run --locked python scripts/rebuild_sentiment.py
```

重建结果写入 `data/derived/sentiment_rebuilt/`，原始包和迁入快照不被覆盖。该脚本复现的是既有统计日口径，**不是** 16:00 可用的最终预测特征。当前没有 `train` 命令。

## 项目目录

- `src/emo_agent/`：基础数据检查、标签规则和命令行。
- `tests/`：不依赖真实研究数据的合成样例测试。
- `scripts/`：历史情绪重建工具。
- `docs/`：研究计划、字段字典、协作规范和开发记录。
- `data/`：本地原始数据、迁入快照及后续派生数据，Git 忽略。
- `outputs/`：运行报告、预测、模型及日志，Git 忽略。
- `.github/`：Issue/PR 模板和持续集成。

## 协作

每项任务先建 Issue，使用 `类型/编号-简述` 分支，通过 PR 合并到 `main`。PR 需写验收证据，并使用 `Closes #编号` 关联任务。开发结束更新 `PROGRESS.md` 和 `docs/records/`。AI 工作规则见 [AGENTS.md](AGENTS.md)。

研究路线见 [docs/research_plan.md](docs/research_plan.md)，数据说明见 [docs/data_dictionary.md](docs/data_dictionary.md)。文件完整与预测有效是两回事；情绪统计的历史实际可得时间仍未核实。
