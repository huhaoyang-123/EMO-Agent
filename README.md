# EMO Agent

EMO-Agent 是面向**中证光通信主题指数（931723）及其历史成分公司**的本地金融市场与投资者情绪分析项目。通过自然语言调用 Python 工具，分析历史行情、市场广度、成分股表现和股吧统计情绪，并给出可追溯的数据依据。

## 当前阶段

后续开发统一以 [金融市场分析智能体开发文档 v1.0](docs/agent_development.md) 为规范。首版采用单 Agent，计划支持市场概览、成分股比较、情绪与行情关系、异常候选和研究证据查询。确定性 Python 工具计算数值，LLM 负责调用和解释。MVP 处于开发准备阶段，当前可运行的是已有的研究 CLI。进度见 [PROGRESS.md](PROGRESS.md)。

既有短期预测研究作为证据保留。多轮实验未发现稳定的情绪增量，冻结模型在影子检验中未优于训练期上涨比例参照。[个股有限试点](docs/results/company_pilot_v1.md) · [大盘背景和广度](docs/results/context_v1.md) · [影子预测](docs/results/shadow_v1.md)

## 安装与现有研究 CLI

需要 Python 3.13、Git 和 uv 0.13.7。Windows/macOS/Linux 均从仓库根目录执行；示例不依赖本机绝对路径。

```sh
python -m pip install uv==0.13.7
python -m uv sync --locked
python -m uv run --locked pytest
python -m uv run --locked emo-agent --help
```

克隆仓库后，授权成员按 [data/README.md](data/README.md) 准备本地真实数据和交接包，再检查基础输入：

```sh
python -m uv run --locked emo-agent audit-data
```

报告默认写入 `outputs/audits/data_audit.json`。任务三、任务六的交接资产按各自元数据另行核验。基础测试可以在没有真实数据时独立运行。

## 历史研究复现

可选重建历史情绪快照（扫描约千万条记录，需要足够内存和时间）：

```sh
python -m uv run --locked python scripts/rebuild_sentiment.py
```

重建结果写入 `data/derived/sentiment_rebuilt/`，原始包和迁入快照不被覆盖。该脚本复现的是既有统计日口径，**不是** 16:00 可用的最终预测特征。构建共同样本：`python -m uv run --locked emo-agent build-dataset`。训练并评估：`python -m uv run --locked emo-agent train-baseline --run-id your_run_id`。实验要求工作区干净；run-id 必须是新名称，避免覆盖已有结果。

已完成的正式实验在 `outputs/baseline_v1/`，四个模型位于 `models/`，逐日概率位于 `test_predictions.csv`。输入数据和模型未上传 GitHub，授权成员取得相同数据后可按以上两条命令复现。

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

历史研究路线见 [docs/research_plan.md](docs/research_plan.md)，数据说明见 [docs/data_dictionary.md](docs/data_dictionary.md)。情绪统计的历史实际可得时间仍未核实。

模型公式、四模型实际系数与下一步路线见 [docs/model_math.md](docs/model_math.md)。

稳定性实验：`python -m uv run --locked emo-agent analyze-stability --run-id your_new_stability_run`。需要相同的 cutoff_v1 与 baseline_v1，本地输出包含逐季指标、八组模型和贡献分解；详见增益诊断报告。

阶段5已完成情绪表达、时滞、自变量选择及浅层树对照，见[变量选择报告](docs/results/features_v1.md)。命令：`python -m uv run --locked emo-agent analyze-features --run-id your_new_features_run`。当前尚未找到稳定情绪变量或复杂度升级依据。

个股试点：先 `python -m uv run --locked emo-agent build-company-panel`，然后 `python -m uv run --locked emo-agent analyze-company --run-id your_new_company_run`。需要已有cutoff_v1和迁入的stock_daily.csv，数据缺失时先按data/README.md准备并运行build-dataset。已有公司面板目录不覆盖。独立核验：`python -m uv run --locked python scripts/verify_company_pilot.py --run-id your_new_company_run`。
