# 数据字典与已收集数据

## 目录与版本

`docs/data_manifest.json` 是迁入数据的相对路径、字节数与 SHA-256 清单；`docs/records/2026-11-12-migration.json` 记录 19 个文件的迁移时状态。说明和脚本迁入后可修改，数据快照只读。真实数据未进入 Git，成员按 `data/README.md` 准备。

| 文件/组 | 内容 | 当前用途及限制 |
|---|---|---|
| data/raw/guba/*.zip | 4 个年度包，24 个 XLSX，约 1,040 万条全市场统计 | 重建与源行追溯；没有逐帖正文 |
| data/input/market/index_931723_full.csv | 2,376 行，2016-12-30—2026-10-19 | 收盘与成交额；2023-10-31 前为回算 |
| index_931723_members.csv | 历史成分有效区间 | 动态成员匹配，64 个历史代码；不是固定 64 只同时在指数内 |
| index_931723_changes.csv | 调入调出记录 | 核查成分表来源；源日期含不同业务含义 |
| trading_calendar.csv | 973 行，2022-10-10—2026-10-14 | 旧日历快照；需对照行情，split/has_931723 不作为新实验规则 |
| data/input/sentiment/931723成分股分时明细.csv | 83,653 行，2023-10-31—2026-06-30 | 预测情绪的重聚合输入；3,044,795 篇已统计帖子 |
| 931723行业统计日.csv | 974 个统计日期 | 描述和对账；含盘后时段，不能直接用于 16:00 预测 |
| 931723行业分时.csv | 1,809 个行业时段 | 时间窗口检查与行业对账 |
| 931723成分股覆盖.csv | 64 行 | 逐证券覆盖 |
| 931723缺失时段.csv | 6,702 行 | 缺失未知原因，不等于零帖 |
| 931723源文件清单.csv | 24 行 | ZIP 哈希与内部 XLSX 行数 |
| summary.json | 历史提取摘要 | 645 个交易日，473 日全部公司至少有记录，117 日全部预期时段齐全 |

以上为迁入快照信息，不代表模型样本已准备完成。行情原始上游工作簿未在旧目录找到；当前以已提供的 clean CSV 作为有哈希的输入快照，不声称能从上游重新抓取行情。

## 行情字段

- `date`：交易日期，ISO YYYY-MM-DD；键唯一并升序。
- `close`：指数收盘点位，大于零。
- `pct_chg`：源日涨跌幅，百分数单位；建模收益由 close 重算，不把 1 当 1%。
- `amount_100m`：成交金额，亿元。
- `source`：csindex_official 或 csindex_backfill(回算)。
- 成分表 `stkcd`：六位文本代码；`in_date` 包含、`out_date` 不包含，空值为尚未结束。

## 情绪字段

- 主键 `Scode + Date + Time`；Scode 按文本读取，保留前导零；Coname 是源简称。
- `Date`：供应商统计日期标签，不是公开时间。
- `Time`、`interval_start/end`：源时段及按说明推定的起止，时间基准 Asia/Shanghai；没有逐帖精确时间。
- `Tpostnum`、`Pospostnum`、`Negpostnum`、`Neupostnum`：总/正/负/中性帖子数量，非负整数，后三者之和等于总数。
- `Readnum`、`Commentnum`：源互动计数；统计截至时点未核实，第一版不用。
- `net_sentiment`：(正−负)/全部帖；无有效分母留空。
- `directional_sentiment`：(正−负)/(正＋负)；无有效分母留空。
- `equal_weight_sentiment`：已观测公司的净情绪均值，非正式指数权重。
- `company_coverage`、`slot_coverage`：观测公司/预期公司、观测时段/预期时段。统计日覆盖不等于预测窗口覆盖。
- `source_zip/source_xlsx/source_excel_row/source_record_id`：逐行追溯字段。
- `in_date/out_date/membership_date_basis`：以源 Date 匹配历史成员；跨调整日的统计时段需披露近似。
- `count_snapshot_verified=0`：历史实际可得时间和修订情况未验证。
- `Tradetm_dum` 为源时段标记；`is_trade_date` 来自旧日历，两个字段不等价。

## 后续生成产物

共同样本和标签已生成于 data/derived/cutoff_v1/；四模型、预测概率和评分已生成于 outputs/baseline_v1/，结果见 docs/results/baseline_v1.md。原 ETF、事件、个股日线及重复浏览版工作簿未迁入本阶段项目。

## cutoff_v1 共同样本（阶段 2）

文件 `data/derived/cutoff_v1/daily_samples.csv`：645 个预测日期，其中 625 个 eligible。2023-11-28—2026-06-30；训练/验证/测试候选数 266/243/116，模型阶段仍须剔除边界标签未成熟行。

- cutoff_at：Asia/Shanghai，每交易日 16:00。
- interval_assignments.csv：90,355 个预期公司时段（含 6,702 缺失），按 interval_end 归到第一个不早于它的预测时点；观测源 ID 只出现一次。该文件可逐行追溯时间归属。
- expected_companies 是窗口内源 Date 对应动态历史成员的并集；跨调整窗口可能多于当日 50 家。company/slot_coverage 均按这个窗口重算，不直接沿用统计日覆盖。
- target_1/target_5：终点收盘严格高于起点为 1；label_end_date_1/5 是标签成熟日期；return_forward_1/5 仅用于评估，绝不能作为特征。
- return_1/5/20：当日收盘相对过去交易日收盘的收益，小数单位；volatility_5/20：日收益样本标准差；amount_relative_20：当日成交额相对前 20 日均值减 1。
- net_sentiment/equal_weight_sentiment：对齐窗口的帖子加权/公司等权情绪；sentiment_change_5：当前净情绪减此前 5 个窗口的均值。
- post_heat_20：log(1+当前每观测公司时段帖子量) 减 log(1+此前 20 窗口该强度均值)，降低周末窗口更长造成的机械影响。
- sentiment_missing：情绪缺失为 1；缺失计数/情绪保留空值。模型只使用训练集统计量填补；不额外筛掉低覆盖日期。
- eligible/exclusion_reason：行情历史窗口是否齐全及不合格原因；period 为初始日期分组，不能代替成熟标签过滤。
- availability_assumption：仅 interval_end 对齐，历史真实发布可得时间未验证。
- metadata.json：源数据哈希、特征白名单、输出哈希与构建提交；excluded_dates.csv：20 个历史窗口不足日期。

指标从发布后行情起算，不使用发布前回算作为第一版特征历史。2026-06-30 的 5 日标签终点为 2026-07-07，可使用行情表中已有未来价格做标签，不用于输入。

## context_v1 新增变量（任务三，Issue #14）

来源：data/input/context/market_index_000985.csv（CSMAR 国内指数日行情，中证全指收盘）、member_stock_returns.csv（CSMAR 个股日行情 Dretnd，不含现金红利再投资），成员区间 data/input/market/index_931723_members.csv。全部使用预测日 15:00 收盘及以前的数据，早于 16:00 截止。生成：`emo-agent build-context` → data/derived/context_v1/context_features.csv（metadata.json 记录输入与输出哈希）。

| 字段 | 定义 |
|---|---|
| market_return_1 | 000985 当日收盘 / 前一交易日收盘 − 1 |
| market_volatility_20 | 000985 过去 20 个日收益标准差（ddof=1） |
| relative_return_5 | 931723 return_5 − 000985 5 日收益 |
| breadth_up | 当日 931723 历史成员（in_date ≤ 日期 < out_date）中有收益记录的股票里，日收益 > 0 的比例 |
| return_dispersion | 上述成员股当日收益横截面标准差（ddof=1） |
| breadth_n | 当日参与计算的成员股数（只作检查，不入模） |
| *_resid（运行时生成，不落盘） | net_sentiment、equal_weight_sentiment、sentiment_change_5、post_heat_20 对 6 个行情变量 + 3 个大盘变量的 OLS 残差；只在每一步的拟合集上估计，之后的行只应用不重估 |
