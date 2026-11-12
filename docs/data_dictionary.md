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

## 尚待生成

最终共同样本、1/5 日标签、cutoff_at、label_end_date、特征与处理日志、预测概率、模型评分。目前没有准确率结果。原 ETF、事件、个股日线及重复浏览版工作簿未迁入本阶段项目。
