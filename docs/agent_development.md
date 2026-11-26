# EMO-Agent：金融市场分析智能体开发文档

> 版本：v1.0  
> 制定日期：2026-11-26  
> 仓库：<https://github.com/huhaoyang-123/EMO-Agent>  
> 审查基线：`main` 最新合并提交 `c32fbc5d6920c4316875a17b73b0809655e50856`（PR #17，2026-11-26 合并）  
> 开发周期：约 7—10 个工作日  
> 文档性质：基于当前仓库真实代码、数据字典和实验报告制定；本文件是开发方案，不代表下述 Agent 功能已经实现。

---

## 1. 项目决策

### 1.1 产品定位

将 EMO-Agent 从“尝试预测光通信指数及成分股涨跌的研究框架”，收敛为：

**面向中证光通信主题指数（931723）及其历史成分公司的金融市场与投资者情绪分析智能体。**

Agent 通过自然语言理解用户的问题，调用本地 Python 分析工具，完成行情概览、市场广度分析、成分股比较、情绪—行情关系分析和异常变化识别，最后生成有数据依据的解释。

产品核心不是“预测明天涨跌”，而是帮助用户回答：

1. 最近一段时间，光通信板块的行情、波动和市场广度发生了什么变化？
2. 投资者讨论热度和情绪方向发生了什么变化？
3. 哪些历史成分公司在讨论热度、情绪方向或股价表现上变化明显？
4. 情绪与价格在同一时期是否出现背离？背离发生在哪些日期？
5. 当前研究已经证明了什么、没有证明什么？模型的历史预测表现究竟如何？

### 1.2 为什么不再把预测作为产品核心

仓库已经完成了多轮相对完整的研究，而不是只训练过一个基线模型：

- 指数基线：1 日预测在 2026H1 中加入情绪后准确率仅提高 0.86 个百分点；5 日准确率没有变化，模型明显偏向预测上涨。
- 跨季度稳定性：2025 年的情绪增量没有稳定复现；加入情绪后整体 Brier 表现没有改善。
- 情绪表达与时滞实验：25 组候选表达未形成稳定增益，受限浅层树模型也没有总体优势。
- 个股有限试点：31,001 行合格公司日、64 个历史成分代码；公司层面的情绪方向也没有通过预先设定的继续门槛。
- 大盘背景与广度实验：没有发现大盘背景或情绪的稳定增量；广度变量在 5 日期限上只有较弱迹象，区间仍包含零。
- 影子预测：2026-07-01 至 2026-10-09 的 67 个日期上，冻结模型的 Brier 均差于训练期上涨比例参照，且预测上涨概率明显高于实际上涨比例。

这些结果说明：当前数据和模型设定没有提供足够证据支持一个可靠的短期涨跌预测产品。它们不等于证明情绪信息在任何场景下都没有价值，但足以支持停止在同一批历史数据上继续扩大模型搜索。

因此，本项目不再以“提高预测准确率”为开发目标。历史预测实验作为研究证据保留，Agent 不得将其包装为已验证有效的预测系统。

### 1.3 产品边界

**本期要做：**

- 单一 Agent，不做多 Agent 协作。
- 只分析已有的 931723 指数、历史成分股、股吧统计情绪、宽基市场和市场广度数据。
- 所有金融指标由 Python 确定性计算，LLM 只负责理解问题、选择工具和解释结果。
- 支持历史回顾、横向比较、异常识别和研究证据查询。
- 本地运行，数据只读，不改动原始研究数据和实验产物。

**本期不做：**

- 不继续训练或调优涨跌预测模型。
- 不接入新的行情供应商，不做自动爬虫或实时行情采集。
- 不做自动交易、买卖建议、目标价或收益承诺。
- 不做评论正文的情绪识别、主题提取或投资者观点总结——当前数据没有逐条评论正文。
- 不做投资者人数分析——当前数据没有用户 ID，帖子数不能替代独立投资者人数。
- 不做 RAG 知识库、多智能体、复杂工作流、模型微调或知识图谱。
- 不让 LLM 直接读任意文件、执行 Python 代码或生成 SQL。

---

## 2. 最新仓库审查结论

### 2.1 当前代码结构

最新 `main` 已包含以下主要模块：

| 文件 | 当前职责 | Agent 开发中的处理 |
|---|---|---|
| `src/emo_agent/audit.py` | 输入数据审计、路径安全与文件校验 | 复用数据检查思路 |
| `src/emo_agent/dataset.py` | 16:00 截止的指数行情/情绪样本构建 | 复用字段与时间口径，不重写历史样本 |
| `src/emo_agent/labels.py` | 1 日/5 日标签构建 | 仅用于理解研究口径；Agent 默认不预测 |
| `src/emo_agent/baseline.py` | 逻辑回归基线训练与评分 | 保留研究用途，不作为本期核心产品 |
| `src/emo_agent/stability.py` | 滚动稳定性实验 | 保留研究用途 |
| `src/emo_agent/features.py` | 情绪表达、时滞和受限树模型实验 | 保留研究用途 |
| `src/emo_agent/company.py` | 成分股情绪/行情面板、公司层面模型试点 | 重点复用数据聚合与字段口径；不复用模型训练流程作为 Agent 工具 |
| `src/emo_agent/context.py` | 宽基市场、市场广度、剩余情绪和校准 | 复用 `market_features`、`breadth_features` 等计算定义 |
| `src/emo_agent/shadow.py` | 冻结模型、影子预测与事后评分 | 保留为研究审计能力，不包装成实时预测 |
| `src/emo_agent/cli.py` | 现有命令行入口 | 在此基础上增加 Agent 启动入口，避免破坏已有命令 |

当前 CLI 已包含：

`audit-data`、`build-dataset`、`train-baseline`、`analyze-stability`、`analyze-features`、`build-context`、`analyze-context`、`build-company-panel`、`analyze-company`、`shadow-freeze`、`shadow-predict`、`shadow-evaluate`。

当前 `pyproject.toml` 的核心依赖为 NumPy、Pandas、scikit-learn、lxml、joblib；尚未加入 LLM SDK 或 Web UI 框架。Agent 与前端因此属于本期新增模块。

### 2.2 当前数据资产

| 数据资产 | 仓库记录的规模/范围 | 可支持的分析 |
|---|---|---|
| 股吧成分股分时统计 | 83,653 条记录，约 3,044,795 篇帖子，2023-10-31 至 2026-06-30 | 帖子量、正负中性数量、净情绪、覆盖情况 |
| 931723 指数行情 | 2,376 行，2016-12-30 至 2026-10-19；早期部分为回算数据 | 指数收盘、收益、成交额、波动 |
| 931723 历史成分区间 | 64 个历史代码 | 按历史有效区间识别成分公司 |
| 个股日行情 | 59,972 行、65 个代码，2022-12-01 至 2026-10-19 | 成分股收益、波动、成交量热度、换手率等 |
| 中证全指 000985 行情 | CSMAR 指数数据；当前记录覆盖至 2026-10-09 | 大盘收益、波动、相对表现 |
| 成分股日收益 | CSMAR 个股日行情 Dretnd | 成分股上涨比例、收益离散度 |
| `cutoff_v1` | 645 个日期，其中 625 个 eligible；2023-11-28 至 2026-06-30 | 指数级行情与情绪联合分析 |
| `context_v1` | 625 个可用日期 | 宽基市场和成分股广度 |
| `company_pilot_v1` | 32,228 个预期公司日；31,001 行合格记录；625 个日期；64 个历史代码 | 公司级情绪与行情比较 |
| `shadow_v1` | 冻结模型与 2026-07 至 10 月历史影子预测报告 | 展示既有研究结论和模型校准问题 |

**重要：真实数据不在 GitHub 仓库中。** 仓库克隆后只有代码、文档、合成测试数据和数据清单。Agent 运行前必须确认本地已经按 `data/README.md` 放置授权数据，并通过 `emo-agent audit-data`。

### 2.3 数据字段与解释边界

指数级 `daily_samples.csv` 主要包含：

- 行情：`return_1`、`return_5`、`return_20`、`volatility_5`、`volatility_20`、`amount_relative_20`。
- 情绪：`net_sentiment`、`equal_weight_sentiment`、`sentiment_change_5`、`post_heat_20`。
- 数据质量：`company_coverage`、`slot_coverage`、`sentiment_missing`。
- 标签和未来收益：`target_1`、`target_5`、`return_forward_1`、`return_forward_5`。这些是研究评价字段，**绝不能作为当前日期分析的输入特征或解释变量**。

`context_features.csv` 主要包含：

- `market_return_1`
- `market_volatility_20`
- `relative_return_5`
- `breadth_up`
- `return_dispersion`
- `breadth_n`（检查字段，不是模型特征）

`company_samples.csv` 主要包含：

- 公司代码 `code`、日期 `date`。
- 个股收益：`stock_return_1`、`stock_return_5`、`stock_return_20`。
- 个股波动：`stock_volatility_5`、`stock_volatility_20`。
- 交易活跃度：`volume_heat_20`、`log_turnover`、`log_float_value`。
- 公司情绪：`net_sentiment`、`sentiment_change_5`。
- 讨论热度：`log_post_rate`、`post_heat_20`。
- 数据质量：`slot_coverage`、`hour_coverage`、`window_hours`、`sentiment_missing`、`heat_history_missing`、`direction_history_missing`。
- 资格字段：`eligible`、`exclusion_reason`。

`index_samples.csv` 是公司面板构建时同步生成的指数级行业情绪样本。公司级来源追溯保存在 `source_trace.csv`。

**必须遵守的限制：**

1. 股吧数据没有逐条评论正文，因此不能总结评论内容或提取主题。
2. 没有用户 ID，因此不能计算独立投资者人数。
3. `Readnum`、`Commentnum` 的累计可见时点未核实，不应作为核心指标或实时输入。
4. 情绪统计的真实发布时间和历史修订时间未核实。当前研究是按统计区间 `interval_end` 对齐，不代表该信息在历史预测时点一定已经公开可得。
5. 情绪数据截至 2026-06-30。综合情绪分析的默认可用区间不得越过该日期。
6. 931723 指数行情可延伸至 2026-10-19；000985 宽基行情当前记录只到 2026-10-09。不同数据源的结束日期必须分别展示，不能假装它们同步更新。
7. `equal_weight_sentiment` 是已观测公司的等权情绪均值，不是按指数正式权重计算的指数情绪。
8. `breadth_up` 使用成分股 Dretnd，数据字典说明其不含现金红利再投资。
9. `Adjprcnd` 是可比收盘价，但不含现金红利再投资；不能把它解释为总回报。
10. 公司派生表当前主要以六位证券代码作为标识。若要显示公司中文名称，应先建立并验证代码—名称映射；映射完成前只展示代码，不让 LLM 猜名称。

---

## 3. 最小可交付产品（MVP）

### 3.1 功能清单

MVP 只实现五个只读分析工具：

| 工具 | 用户问题 | 核心输出 | 优先级 |
|---|---|---|---|
| `get_market_overview` | 最近一段时间板块发生了什么？ | 指数收益、波动、成交、情绪、市场广度摘要 | P0 |
| `compare_constituents` | 哪些成分公司变化最大？ | 公司指标排名、有效样本数、覆盖率 | P0 |
| `analyze_sentiment_market_relation` | 情绪和行情是否一致？ | 同期相关、同向/背离日期、样本量 | P0 |
| `detect_anomalies` | 哪些日期出现异常变化？ | 异常日期、当前值、历史基线、偏离程度 | P1 |
| `get_research_evidence` | 情绪到底有没有预测价值？ | 研究报告中的已验证结果与限制 | P0 |

不需要让 LLM 自由发挥生成新的金融计算方法。每项计算必须有确定的 Python 函数、明确的参数和固定的返回结构。

### 3.2 工具一：`get_market_overview`

**用途：** 给出用户指定区间内 931723 指数、投资者情绪和市场广度的概览。

参数：

```json
{
  "start_date": "2025-01-01",
  "end_date": "2025-03-31",
  "include_sentiment": true
}
```

约束：

- 日期必须为 `YYYY-MM-DD`，且 `start_date <= end_date`。
- 默认只查询本地已有数据，不联网。
- `include_sentiment=true` 时，实际区间不能超过情绪数据的可用日期。
- 如果用户请求的日期超出情绪数据范围，必须明确返回情绪不可用的区间；可以继续提供有数据的纯行情结果，但不能把情绪缺失解释为中性。
- 报告实际样本日期数、缺失数和覆盖率。

建议返回：

```json
{
  "tool": "get_market_overview",
  "requested_period": ["2025-01-01", "2025-03-31"],
  "actual_period": ["2025-01-02", "2025-03-31"],
  "observations": 58,
  "index_metrics": {
    "period_return": 0.0,
    "mean_daily_return": 0.0,
    "volatility_20_latest": 0.0,
    "mean_amount_relative_20": 0.0
  },
  "sentiment_metrics": {
    "mean_net_sentiment": 0.0,
    "mean_equal_weight_sentiment": 0.0,
    "mean_post_heat_20": 0.0,
    "mean_company_coverage": 0.0,
    "mean_slot_coverage": 0.0
  },
  "breadth_metrics": {
    "mean_breadth_up": 0.0,
    "mean_return_dispersion": 0.0
  },
  "data_quality": {
    "sentiment_missing_days": 0,
    "context_missing_days": 0
  },
  "warnings": []
}
```

上面数值仅为结构示例，不是实际实验结果。

### 3.3 工具二：`compare_constituents`

**用途：** 在指定区间内比较历史成分公司的情绪、讨论热度和行情表现。

参数：

```json
{
  "start_date": "2025-01-01",
  "end_date": "2025-06-30",
  "metric": "post_heat_20",
  "direction": "highest",
  "top_n": 10
}
```

允许的 `metric`：

- `net_sentiment`
- `sentiment_change_5`
- `log_post_rate`
- `post_heat_20`
- `stock_return_1`
- `stock_return_5`
- `stock_volatility_20`
- `volume_heat_20`

约束：

- `direction` 只能是 `highest` 或 `lowest`。
- `top_n` 限制在 1—20。
- 仅使用 `eligible=true` 的记录；缺失值不填成 0。
- 每家公司同时返回有效天数、指标均值/中位数、可用覆盖比例。
- 公司排序必须在同一个日期范围和同一个指标口径下完成。
- 默认综合分析区间不超过 2026-06-30。
- 不得把“帖子更多”解释为“投资者更多”。

建议返回：

```json
{
  "tool": "compare_constituents",
  "period": ["2025-01-01", "2025-06-30"],
  "metric": "post_heat_20",
  "direction": "highest",
  "results": [
    {
      "code": "000000",
      "valid_days": 100,
      "coverage": 0.95,
      "mean": 0.0,
      "median": 0.0
    }
  ],
  "warnings": []
}
```

### 3.4 工具三：`analyze_sentiment_market_relation`

**用途：** 描述情绪指标与同期市场表现之间的关系，识别情绪与价格方向不一致的日期。

参数：

```json
{
  "start_date": "2025-01-01",
  "end_date": "2025-06-30",
  "return_window": 1
}
```

`return_window` 只允许 1 或 5，分别使用 `return_1` 或 `return_5`。

建议计算：

- 净情绪与同期收益的 Pearson 相关系数。
- 净情绪为正且收益为负、净情绪为负且收益为正的日期。
- 同向日期数、背离日期数、有效样本数。
- 可选展示 20 日滚动相关，但必须说明滚动窗口长度和有效样本数。

这里的收益与情绪都是同期/历史窗口指标，不是未来收益。该工具只描述共变关系，不能得出情绪导致涨跌、情绪可以预测未来或存在可交易反转信号等结论。

建议返回：

```json
{
  "tool": "analyze_sentiment_market_relation",
  "period": ["2025-01-01", "2025-06-30"],
  "return_window": 1,
  "n": 110,
  "pearson_correlation": 0.0,
  "same_direction_days": 0,
  "divergence_days": 0,
  "divergence_records": [],
  "interpretation_limit": "同期描述性关系，不代表因果或预测能力"
}
```

### 3.5 工具四：`detect_anomalies`

**用途：** 识别指定指标相对过去一段时间的异常偏离。

参数：

```json
{
  "start_date": "2025-01-01",
  "end_date": "2025-06-30",
  "metric": "net_sentiment",
  "lookback": 20,
  "threshold": 2.5
}
```

允许指标：

- `net_sentiment`
- `post_heat_20`
- `breadth_up`
- `return_1`
- `return_5`
- `return_dispersion`

计算规则：

1. 按交易日排序。
2. 对每个日期，只使用它之前的 `lookback` 个有效观测建立基线，不把当前值放入自身基线。
3. 至少需要 15 个有效历史观测；不足时不判定异常。
4. 首版使用滚动均值和标准差计算 Z-score。
5. `abs(z_score) >= threshold` 时标记为“统计异常候选”。
6. 输出当前值、历史均值、历史标准差、Z-score 和基线样本数。

**注意：** 2.5 是工程筛查阈值，不是经过多重检验修正的统计显著性结论。界面应写“异常候选/值得检查”，不能写“显著事件”或自动推断原因。

### 3.6 工具五：`get_research_evidence`

**用途：** 回答项目研究结论、模型效果和数据限制问题。

参数：

```json
{
  "topic": "all"
}
```

`topic` 可选：

- `index_baseline`
- `sentiment_stability`
- `company_pilot`
- `market_context`
- `shadow_test`
- `data_limitations`
- `all`

建议由 `docs/agent_evidence.json` 保存经核对的结构化结论，来源链接指向仓库内正式报告。该 JSON 由开发者维护，不允许 LLM 自行修改。

工具返回的研究证据必须保留以下结论：

- 当前情绪指标没有显示稳定的短期涨跌预测增量。
- 公司层面试点没有通过预先设定的继续门槛。
- 市场广度 5 日结果有弱迹象，但区间包含零，不能宣称有效。
- 影子预测阶段所有模型 Brier 均差于训练期上涨比例参照。
- 所有历史分析结果均不能等同于可执行策略收益。

---

## 4. 系统架构

### 4.1 总体结构

```text
用户
 │
 ▼
Streamlit 聊天界面
 │
 ▼
Agent Orchestrator（单 Agent）
 │
 ├── LLM Client（OpenAI 兼容工具调用接口）
 │
 └── Tool Registry（仅注册白名单工具）
       │
       ├── get_market_overview
       ├── compare_constituents
       ├── analyze_sentiment_market_relation
       ├── detect_anomalies
       └── get_research_evidence
              │
              ▼
       Python Analysis Layer
              │
              ▼
       Read-only Data Loader
              │
              ├── data/derived/cutoff_v1/
              ├── data/derived/context_v1/
              ├── data/derived/company_pilot_v1/
              ├── docs/results/
              └── outputs/shadow_v1/（可选，仅用于研究审计）
```

数据分析层与 LLM 分离。LLM 不直接读取 CSV，不直接运行 pandas，不生成任意 SQL。它只能调用工具注册表中的固定函数。

### 4.2 目录规划

在现有仓库中新增以下结构，不覆盖现有研究模块：

```text
src/emo_agent/
├── agent/
│   ├── __init__.py
│   ├── client.py              # LLM 客户端与环境变量配置
│   ├── orchestrator.py       # 工具调用循环
│   ├── prompts.py            # 系统提示词
│   ├── schemas.py            # 工具参数 JSON Schema
│   └── registry.py           # 工具白名单与执行分发
│
├── analysis/
│   ├── __init__.py
│   ├── loader.py             # 本地数据加载、元数据与字段校验
│   ├── overview.py           # 指数/市场概览
│   ├── constituents.py       # 成分公司比较
│   ├── relation.py           # 情绪—行情同期关系
│   ├── anomalies.py          # 异常候选识别
│   └── evidence.py           # 研究证据读取
│
└── app.py                    # Streamlit 主界面

configs/
└── agent_v1.json             # 数据路径、可用区间、工具阈值等

docs/
└── agent_evidence.json       # 结构化研究证据及来源

tests/
├── test_agent_tools.py
├── test_analysis_overview.py
├── test_analysis_constituents.py
├── test_analysis_relation.py
├── test_analysis_anomalies.py
└── test_agent_orchestrator.py
```

如果现有代码中某些函数可以直接复用，应优先复用，不要为了符合目录图而搬动或重命名旧模块。

### 4.3 各模块函数接口

#### `analysis/loader.py`

```python
def load_analysis_data(root: Path) -> AnalysisData:
    """加载并校验 Agent 所需的本地派生数据。"""

def get_data_availability(data: AnalysisData) -> dict:
    """返回各数据源的起止日期、行数、缺失和更新时间信息。"""

def validate_date_range(start_date: str, end_date: str, available_range: tuple) -> None:
    """校验日期格式、顺序及数据源覆盖范围。"""
```

要求：

- 使用仓库根目录推导路径；可支持 `EMO_AGENT_DATA_ROOT` 指定根目录。
- 启动时检查文件是否存在、必需字段是否齐全。
- 优先校验已有 `metadata.json` 中的 SHA-256。
- 对真实数据只读；不得自动重建、覆盖或清洗原始数据。
- 数据缺失时返回清楚的缺失文件名和准备方式，不尝试猜测替代文件。
- 对不同数据源分别记录可用日期，不设置一个虚假的统一“最新日期”。

建议 `AnalysisData` 至少包含：

```python
@dataclass
class AnalysisData:
    index_daily: pd.DataFrame
    context_daily: pd.DataFrame
    company_daily: pd.DataFrame
    index_company_daily: pd.DataFrame
    availability: dict
```

#### `analysis/overview.py`

```python
def get_market_overview(
    data: AnalysisData,
    start_date: str,
    end_date: str,
    include_sentiment: bool = True,
) -> dict:
    ...
```

计算应来自 `daily_samples.csv` 和 `context_features.csv`，不得使用未来标签列。期间收益用区间首尾有效收盘价计算；波动率、成交活跃度和情绪指标明确写出计算口径。

#### `analysis/constituents.py`

```python
def compare_constituents(
    data: AnalysisData,
    start_date: str,
    end_date: str,
    metric: str,
    direction: str = "highest",
    top_n: int = 10,
) -> dict:
    ...
```

使用 `company_samples.csv`，过滤 `eligible=true`。输出每家公司有效天数和覆盖比例，避免只按极少数观测排序。

#### `analysis/relation.py`

```python
def analyze_sentiment_market_relation(
    data: AnalysisData,
    start_date: str,
    end_date: str,
    return_window: int = 1,
) -> dict:
    ...
```

使用 `index_samples.csv` 或 `daily_samples.csv` 的同期情绪与收益字段。不要使用 `return_forward_1`、`return_forward_5` 或未来标签。

#### `analysis/anomalies.py`

```python
def detect_anomalies(
    data: AnalysisData,
    start_date: str,
    end_date: str,
    metric: str,
    lookback: int = 20,
    threshold: float = 2.5,
) -> dict:
    ...
```

滚动基线必须只使用过去观测；缺失值不填 0。所有异常结果都返回基线窗口、有效样本数和阈值。

#### `analysis/evidence.py`

```python
def get_research_evidence(topic: str = "all") -> dict:
    """读取 docs/agent_evidence.json，不重新拟合模型。"""
```

不得在用户提问时自动重跑研究实验。研究报告是静态证据，修改必须走代码审查和版本记录。

---

## 5. Agent 与 LLM 实现

### 5.1 技术选型

| 层 | 技术 | 说明 |
|---|---|---|
| 语言 | Python 3.13 | 与当前仓库一致 |
| 数据分析 | Pandas、NumPy | 已有依赖 |
| 统计/模型研究 | scikit-learn | 已有依赖，MVP 不新增训练任务 |
| Agent | OpenAI Python SDK 的工具调用接口 | 新增依赖；可通过 `base_url` 适配兼容接口 |
| Web UI | Streamlit | 新增依赖，快速完成单页交互 |
| 图表 | Streamlit 原生 `st.line_chart` / `st.bar_chart` | 暂不引入复杂可视化库 |
| 配置 | JSON + 环境变量 | API Key 不写入仓库 |
| 测试 | pytest | 当前已有 |

建议在仓库根目录执行：

```bash
python -m uv add openai streamlit
```

然后检查依赖变更并提交 `pyproject.toml` 与 `uv.lock`。不要手动编辑锁文件。

### 5.2 环境变量

至少支持：

```text
OPENAI_API_KEY=你的API密钥
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=支持工具调用的模型名称
EMO_AGENT_DATA_ROOT=仓库根目录（可选）
```

`OPENAI_BASE_URL` 可选；若使用其他兼容服务，按服务商要求设置。`.env`、密钥和个人配置必须加入 `.gitignore`，不得提交到 GitHub。

### 5.3 工具调用循环

`agent/orchestrator.py` 的核心逻辑：

1. 接收用户问题和当前会话消息。
2. 将系统提示词、用户消息和五个工具 Schema 发送给 LLM。
3. 如果模型返回工具调用，检查工具名是否在白名单中。
4. 校验参数类型、日期范围、枚举值和数量上限。
5. 调用对应 Python 函数，获取结构化 JSON。
6. 将工具结果返回给 LLM。
7. 模型生成最终解释；如果仍需要工具调用，继续下一轮。
8. 最终回答附上数据范围、样本量、缺失情况和必要限制。

建议限制：

- 单轮最多 4 次工具调用。
- 单轮最多 3 轮工具循环。
- 单个工具执行超时 15 秒；整体回答超时按模型服务情况设置。
- 工具参数必须经过本地校验，不能信任 LLM 自己生成的日期、路径或指标名。
- 工具异常以结构化错误返回，不把 Python Traceback 或 API Key 传给模型。
- 工具执行全部只读；没有文件写入、模型训练、删除和网络爬取工具。

### 5.4 系统提示词约束

系统提示词至少包含以下规则：

1. 你是光通信板块历史市场分析助手，不是投资顾问。
2. 任何具体数字必须来自工具返回结果，不允许自行估算或编造。
3. 如果用户要求的日期超出数据覆盖范围，明确说明缺失，不得外推。
4. 如果缺少情绪数据，不得把缺失解释为中性情绪。
5. 区分同期相关、历史描述、预测表现和因果关系。
6. 不得把情绪与收益同期相关描述成情绪导致价格变化。
7. 不得把历史模型预测包装成未来预测，不得生成买入、卖出、目标价或收益保证。
8. 用户问“情绪能否预测涨跌”时，必须调用 `get_research_evidence`，如实呈现负面或不确定结果。
9. 每次分析至少说明实际日期区间、有效观测数和重要的数据限制。
10. 工具没有返回的数据就明确说无法判断，不使用常识补齐。

### 5.5 最终回答格式

Agent 面向用户的自然语言回答建议按以下顺序组织：

- **结论：** 先回答用户的问题。
- **数据依据：** 日期区间、样本数、核心指标。
- **具体发现：** 通过工具返回的表格或趋势说明变化。
- **限制：** 缺失覆盖、数据截止日期、同期关系非因果等。
- **进一步分析：** 只有在确实有必要时，建议用户追问一个具体方向。

回答不需要每次都生成长报告。简单问题直接回答；复杂问题再生成分段分析。

---

## 6. Streamlit 界面

### 6.1 页面结构

MVP 只做一个页面，不拆分复杂多级导航。

**侧边栏：**

- 项目名称 EMO-Agent。
- 数据审计状态：正常 / 缺失文件 / 哈希不一致。
- 各数据源最新日期。
- 当前支持的情绪分析区间。
- 模型 API 连接状态（不得显示密钥）。

**主页面：**

- 标题：光通信板块市场与情绪分析。
- 一段简短的产品说明和数据范围。
- `st.chat_message` 展示用户与 Agent 对话。
- `st.chat_input` 接收问题。
- 当工具返回时间序列时，用 `st.line_chart` 展示。
- 当工具返回成分公司排序时，用 `st.dataframe` 展示。
- 用 `st.expander("本次分析的数据依据")` 展示数据范围、样本数、数据源和限制。

不要一开始就做复杂 Dashboard、账户体系、登录系统、数据库持久化或多页面交互。

### 6.2 启动命令

在仓库根目录执行：

```bash
python -m uv sync --locked
python -m uv run --locked pytest
python -m uv run --locked streamlit run src/emo_agent/app.py
```

Windows PowerShell 可先设置环境变量：

```powershell
$env:OPENAI_API_KEY="你的API密钥"
$env:OPENAI_MODEL="支持工具调用的模型名称"
python -m uv run --locked streamlit run src/emo_agent/app.py
```

如果使用兼容服务，再设置 `OPENAI_BASE_URL`。

---

## 7. 数据读取与口径

### 7.1 默认数据路径

```text
data/
├── input/
│   ├── market/
│   │   ├── index_931723_full.csv
│   │   ├── index_931723_members.csv
│   │   └── stock_daily.csv
│   └── context/
│       ├── market_index_000985.csv
│       └── member_stock_returns.csv
└── derived/
    ├── cutoff_v1/
    │   ├── daily_samples.csv
    │   ├── interval_assignments.csv
    │   └── metadata.json
    ├── context_v1/
    │   ├── context_features.csv
    │   └── metadata.json
    └── company_pilot_v1/
        ├── company_samples.csv
        ├── index_samples.csv
        ├── source_trace.csv
        └── metadata.json
```

实际文件是否存在，以本机 `audit-data` 和元数据检查为准。以上是代码和数据字典记录的路径，不代表 GitHub 已托管真实 CSV。

### 7.2 可用日期的处理

启动时为每份数据分别计算：

- `min_date`
- `max_date`
- `row_count`
- `eligible_count`
- `missing_count`
- `metadata_hash_valid`

建议 MVP 采用以下策略：

| 查询类型 | 默认数据范围 |
|---|---|
| 指数 + 情绪联合分析 | 2023-11-28 至 2026-06-30 的实际交集 |
| 成分股 + 情绪联合分析 | 2023-11-28 至 2026-06-30 的实际交集 |
| 指数纯行情分析 | 以本地 931723 行情文件实际范围为准，当前记录到 2026-10-19 |
| 宽基/广度分析 | 以 000985 和个股日收益文件实际交集为准，当前 000985 记录到 2026-10-09 |
| 历史预测证据 | 读取 `docs/results/` 中已提交的报告；本地影子日志如不存在则不展示逐日预测 |

用户请求 2026-08 的情绪分析时，Agent 应回答情绪数据截至 2026-06-30，不能提供 8 月情绪。若有 8 月纯行情数据，可以另行提供纯行情概览，并明确不含情绪。

### 7.3 防止未来信息泄漏

虽然 MVP 不做预测，但仍须避免把未来信息混入历史解释：

- 所有描述性指标只使用用户请求日期内可用的当前/历史字段。
- `target_1`、`target_5`、`return_forward_1`、`return_forward_5` 只属于研究评估数据，不得作为分析工具输入。
- 计算某日期的滚动异常时，基线只允许使用该日期之前的数据。
- 计算情绪与行情关系时，明确使用同期 `return_1` 或 `return_5`，不要误称为未来收益。
- 不能将 2026-07—10 月影子预测称为当日实时预测：这 67 日预测是在 2026-11-25 一次性批量生成（`retro_batch`），只使用各日期 15:00 及以前的行情。

---

## 8. 测试与验收

### 8.1 单元测试

每个分析工具必须使用合成 DataFrame 测试，不依赖真实数据和 API Key。

| 测试模块 | 必须覆盖 |
|---|---|
| Loader | 文件缺失、字段缺失、哈希不一致、日期范围不一致 |
| Overview | 区间收益、均值、波动率、缺失覆盖计算正确 |
| Constituents | 只保留 eligible 行、缺失不填零、排序正确、有效天数正确 |
| Relation | 同期相关、同向/背离日期数、零方差与空样本处理 |
| Anomalies | 当前值不进入自身基线、历史样本不足时不报异常、标准差为零时安全处理 |
| Agent Registry | 未注册工具不能调用、非法参数被拒绝 |
| Orchestrator | 多轮工具调用、工具错误处理、最大调用次数限制 |
| Prompt | 不能把同期相关解释为因果、不能生成预测或投资建议 |

### 8.2 集成测试

在本地真实数据环境中：

1. `audit-data` 全部通过。
2. Agent 数据加载器读取三份派生数据成功。
3. 五个工具都能返回合法 JSON。
4. 同一问题重复运行，Python 计算结果一致。
5. API Key 缺失时，界面给出清楚提示，而不是启动崩溃。
6. 情绪数据范围以外的日期请求能够正确拒绝或降级为纯行情分析。
7. 工具返回缺失值时，LLM 不得补造数字。
8. 不改动 `data/input/`、`data/derived/`、`outputs/` 中任何原有文件。

### 8.3 必须通过的演示案例

**案例 A：板块整体分析**

用户：“分析 2025 年第一季度光通信板块的行情、市场广度和股吧情绪。”

预期：Agent 调用市场概览工具，返回区间收益、波动、净情绪、讨论热度、广度和有效日期数；图表与文字中的数字一致。

**案例 B：成分公司比较**

用户：“2025 年上半年，哪些历史成分公司的讨论热度上升最明显？给我前五名。”

预期：Agent 调用成分股比较工具，按 `post_heat_20` 排序，同时展示有效天数和覆盖率；只展示代码，不编造公司名称。

**案例 C：情绪与价格背离**

用户：“2025 年上半年有没有情绪偏正面、但指数下跌的日期？”

预期：Agent 调用情绪—行情关系工具，给出背离日期、样本量和同期描述性解释，不把背离称为反转信号。

**案例 D：研究结论核查**

用户：“我们之前的实验到底证明股吧情绪能不能预测指数涨跌？”

预期：Agent 调用研究证据工具，综合指数基线、稳定性、公司试点、市场背景和影子预测结果，明确回答目前没有稳定增量预测证据。

**案例 E：超出情绪数据范围**

用户：“分析 2026 年 8 月光通信板块的股吧情绪。”

预期：Agent 明确说明情绪数据截至 2026-06-30；不得生成 8 月情绪值。可以询问用户是否需要该月份纯行情分析，但不能悄悄改成行情回答。

### 8.4 最终验收标准

- [ ] Windows 本地可按 README 命令安装并启动。
- [ ] `pytest` 全部通过。
- [ ] `audit-data` 通过，派生数据元数据校验通过。
- [ ] 五个工具均可独立运行并返回结构化结果。
- [ ] Agent 至少完成三轮不同类型的自然语言分析。
- [ ] 每个数值都能追溯到工具结果和数据源。
- [ ] 日期超范围、字段缺失和工具异常均有明确提示。
- [ ] API Key 未进入 Git 历史。
- [ ] 不改动历史实验结果，不新增预测模型训练。
- [ ] README、PROGRESS、AGENTS 和开发记录已更新。
- [ ] 演示过程能够清楚说明产品做什么、不做什么。

---

## 9. 7—10 天开发计划

### 第 1 天：冻结范围与本地数据检查

**交付物：** `configs/agent_v1.json`、数据可用性审计记录。

- 拉取最新 `main`，确认工作区干净。
- 阅读 `README.md`、`AGENTS.md`、`PROGRESS.md`、`docs/data_dictionary.md`。
- 运行 `audit-data`。
- 检查 `cutoff_v1`、`context_v1`、`company_pilot_v1` 三份派生数据是否存在，元数据哈希是否正确。
- 确认情绪数据和宽基数据的真实截止日期。
- 不重建历史数据，不重新跑模型实验。

### 第 2 天：数据加载层

**交付物：** `analysis/loader.py` 和 `tests/test_analysis_loader.py`。

- 实现统一数据加载对象。
- 实现文件、字段、哈希和日期范围检查。
- 统一日期字段格式。
- 明确区分指数级、公司级和宽基级数据。
- 完成合成数据测试。

### 第 3 天：市场概览与公司比较

**交付物：** `analysis/overview.py`、`analysis/constituents.py`。

- 完成 `get_market_overview`。
- 完成 `compare_constituents`。
- 输出样本数、覆盖率、缺失情况。
- 用真实数据手工核对 2—3 个日期区间。

### 第 4 天：关系分析、异常检测和研究证据

**交付物：** `analysis/relation.py`、`analysis/anomalies.py`、`analysis/evidence.py`、`docs/agent_evidence.json`。

- 完成同期情绪—行情关系。
- 完成滚动异常候选识别。
- 把已有研究报告中的结论整理为结构化证据。
- 为所有指标增加边界测试。

### 第 5 天：LLM 工具调用

**交付物：** `agent/client.py`、`agent/registry.py`、`agent/orchestrator.py`、`agent/prompts.py`。

- 添加 OpenAI SDK。
- 实现工具 Schema 和白名单。
- 实现工具调用循环。
- 限制调用次数和参数范围。
- 用 Mock LLM 完成自动化测试，不依赖真实 API。

### 第 6 天：Streamlit 交互界面

**交付物：** `app.py`。

- 添加 Streamlit。
- 实现聊天输入、历史消息和工具结果展示。
- 加入数据状态和日期范围提示。
- 展示必要图表和表格。
- 不做复杂 Dashboard。

### 第 7 天：端到端联调

**交付物：** 三个可运行的演示案例。

- 使用真实 API 测试工具调用。
- 验证模型是否会误读工具结果。
- 测试日期越界、缺失情绪、无数据、工具异常。
- 核对界面数字与 Python 计算结果。
- 修复影响演示的 bug。

### 第 8—10 天：收尾与答辩准备

**交付物：** 可运行版本、README、PROGRESS、演示脚本和项目报告素材。

- 更新依赖锁文件。
- 在 Windows 环境完整启动一次。
- 固化三到五个演示问题。
- 更新 `README.md`、`PROGRESS.md`、`AGENTS.md`。
- 新增 `docs/records/YYYY-MM-DD-agent-mvp.md`。
- 准备项目架构图、工具调用流程图和研究结果页。

如果实际只剩 7 天，优先保证数据加载、三个核心工具、LLM 工具调用和 Streamlit 界面。异常检测可降为可选功能；不要砍掉数据校验和防幻觉测试。

---

## 10. 开发纪律与风险控制

### 10.1 必须保留的工程纪律

沿用仓库已有的协作规范：

- 每个任务先建 Issue，独立分支开发，通过 PR 合并。
- 不直接在 `main` 开发。
- 每次改动后更新 `PROGRESS.md` 和 `docs/records/`。
- 原始输入和历史派生数据只读。
- 不提交真实数据、API Key、模型文件和大体积输出。
- 不覆盖旧实验 run-id。
- 真实数据审计与合成测试分别记录，不能把 CI 通过等同于真实数据可用。
- 记录实际执行的测试，不得把未执行的验证写成通过。

### 10.2 必须更新的现有文档

当前文档有部分内容仍停留在研究阶段：

- `README.md`：目前仍以短期涨跌预测研究作为项目简介；应改成“历史市场与情绪分析 Agent”，并保留研究背景。
- `PROGRESS.md`：文件顶部更新日期仍为 2026-11-21，但正文已经追加 11-25 的工作；本期完成后更新顶部日期和当前阶段。
- `AGENTS.md`：当前范围主要描述预测研究；需要增加 Agent 产品边界、工具白名单、数据只读、LLM 不得自行计算等规则。
- `pyproject.toml` / `uv.lock`：新增依赖后必须同步锁定。
- `docs/data_dictionary.md`：若 Agent 使用字段口径有新增解释，应补充，而不是修改既有字段定义。

### 10.3 风险清单

| 风险 | 影响 | 处理方式 |
|---|---|---|
| 本地缺少真实数据 | Agent 无法加载分析数据 | 启动前审计，提示按 `data/README.md` 准备 |
| 情绪数据发布时间未核实 | 不能声称历史实时可得 | 产品定位为历史回顾分析 |
| 情绪数据截止 2026-06-30 | 之后无法做情绪分析 | 按数据源分开显示可用日期 |
| 000985 截止 2026-10-09 | 后续宽基和广度分析不完整 | 不外推，不把缺失补成 0 |
| 公司名称映射不完整 | 可能错认公司 | 先显示证券代码，名称映射另行核验 |
| LLM 编造数字 | 金融分析可信度受损 | 数值全部由 Python 工具返回 |
| LLM 误解相关关系 | 可能形成错误因果叙述 | Prompt 与工具输出双重限制 |
| 模型预测历史表现差 | 产品容易被误认为预测工具 | MVP 不提供未来涨跌预测 |
| 开发时间不足 | 功能无法完整交付 | 严格执行 P0/P1，砍掉非必要 UI 和高级功能 |

---

## 11. 项目答辩叙事

建议将项目讲成三个阶段：

**第一阶段：研究问题。**  
项目最初希望利用股吧投资者情绪识别光通信板块的短期预测信号。团队整理了 931723 历史成分公司、行情与股吧统计数据，构建了按 16:00 截止的历史样本，并完成逻辑回归基线、跨季度稳定性、情绪时滞与表达、个股有限试点、大盘背景和市场广度等实验。

**第二阶段：研究发现。**  
多轮实验没有找到情绪变量稳定的增量预测价值；在 2026 年 7—10 月的冻结模型首轮检验中，模型表现也没有优于训练期上涨比例参照。团队因此停止在同一批历史样本上继续扩大模型搜索，不把弱结果包装成有效预测。

**第三阶段：产品转向。**  
团队重新定位投资者情绪数据的用途：它未必能稳定预测未来收益，但可以作为观察市场参与者讨论热度、情绪方向和市场变化的描述性信息。于是将项目转向一个金融分析 Agent，让用户通过自然语言调用本地数据分析工具，完成市场概览、成分公司比较、情绪—行情关系和异常变化识别。

这条叙事保留了研究过程的真实性，也能解释为什么最后的 Agent 不是一个简单的聊天机器人：它能根据用户问题调用确定性金融分析工具，并将结构化计算结果组织成可追溯的分析报告。

---

## 12. 参考文件与代码入口

以下均为本开发文档制定时审查过的仓库文件：

- [README.md](https://github.com/huhaoyang-123/EMO-Agent/blob/main/README.md)
- [AGENTS.md](https://github.com/huhaoyang-123/EMO-Agent/blob/main/AGENTS.md)
- [PROGRESS.md](https://github.com/huhaoyang-123/EMO-Agent/blob/main/PROGRESS.md)
- [数据字典](https://github.com/huhaoyang-123/EMO-Agent/blob/main/docs/data_dictionary.md)
- [数据准备说明](https://github.com/huhaoyang-123/EMO-Agent/blob/main/data/README.md)
- [CLI 入口](https://github.com/huhaoyang-123/EMO-Agent/blob/main/src/emo_agent/cli.py)
- [公司层面面板与试点](https://github.com/huhaoyang-123/EMO-Agent/blob/main/src/emo_agent/company.py)
- [市场背景与广度计算](https://github.com/huhaoyang-123/EMO-Agent/blob/main/src/emo_agent/context.py)
- [影子预测流程](https://github.com/huhaoyang-123/EMO-Agent/blob/main/src/emo_agent/shadow.py)
- [个股有限试点报告](https://github.com/huhaoyang-123/EMO-Agent/blob/main/docs/results/company_pilot_v1.md)
- [大盘背景与广度报告](https://github.com/huhaoyang-123/EMO-Agent/blob/main/docs/results/context_v1.md)
- [影子预测报告](https://github.com/huhaoyang-123/EMO-Agent/blob/main/docs/results/shadow_v1.md)
- [最新合并 PR #17](https://github.com/huhaoyang-123/EMO-Agent/pull/17)

---

## 13. 最终开发原则

本期只追求一个结果：

**做出一个可以在本地稳定运行、能正确调用金融分析工具、能对已有数据给出可追溯解释的单 Agent。**

不以预测准确率作为本期验收指标，不再开新的模型研究分支，不再寻找新的数据源。每个功能都必须有明确的输入、输出和测试；每个结论都必须能回到数据或正式研究报告。

如果功能不能在剩余时间内稳定交付，就删减功能，而不是增加技术复杂度。
