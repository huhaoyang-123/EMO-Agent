# 阶段4：稳定性与来源诊断

日期2026-11-17；Issue [#7](https://github.com/huhaoyang-123/EMO-Agent/issues/7)；分支feat/7-stability-attribution。基于已合并的main 02daa0d；前置PR #4/#5/#6已合并。本轮本地完成，PR及CI待发布后更新，不自动合并。

## 目标与修改

回答此前情绪增益能否跨期延续、哪些输入组贡献改善。新增冻结配置、stability.py和CLI、7项边界/归因测试；新增报告并更新README、PROGRESS和公式文档进展。细节与未执行下一步见docs/results/stability_v1.md。

初次评分前提交0c7c781b5f131e27b7e493b87d8c2886dec33466；固定C、组定义、季度边界、选参和bootstrap事先冻结。初次concat产生pandas FutureWarning（比例参照C=None形成全空类型列），以float/NaN明确类型修复，提交3e5d487bfacec90273e63536a4cee1e2fc030c96。该修复不改变模型、参数或结果；未隐藏警告。没有失败测试或放宽断言。

## 输入与输出

输入data/derived/cutoff_v1/daily_samples.csv与outputs/baseline_v1/，以及docs/data_manifest.json列出的只读输入。run.json记录全部输入与配置SHA-256、代码提交和依赖版本，原基线与输入哈希均核验。

输出outputs/stability_v1/：144个重训模型，10,404行逐日概率（不是独立日期数），划分、选参、逐折与合并指标、配对差、贡献、run.json和verification.json。2025主评价1日239天、5日223天；2026回顾各116天。临时outputs/stability_v1_reprocheck完整复跑、哈希核验后已删除，保留正式产物与核验记录。

## 实际验证

- python -m uv sync --locked：通过。
- python -m uv run --locked pytest：38 passed。
- python -m uv run --locked emo-agent audit-data：ok=true，15个文件。
- analyze-stability --run-id stability_v1：成功；2026旧基线逐日概率复现容差1e-12。
- analyze-stability --run-id stability_v1_reprocheck：成功；156个产物SHA-256全部相同，包括144个模型。
- 根据逐日预测独立计算42项配对总体差值及18项平均边际贡献（用子集权重公式，不复用归因函数），60项误差均小于1e-12。
- git diff --check：通过（发布前再次检查文档改动）。

## 发现与限制

1日主分析准确率50.63%→42.26%，Brier增益-0.000150，改善2/4季度；5日60.54%→59.64%，Brier增益-0.006530，改善1/4季度。固定C时也未获得稳定改善。2026贡献分解不等于稳定未来预测力或因果归因。

2025/2026均已用于历史开发或查看；本轮不是全新独立测试。四季度样本短、5日区间重叠、实际历史到达时间和修订未核实；未做交易回测。未执行进一步特征搜索、模型升级或每日Agent。

## 下一步

先做覆盖/时间窗口/分布诊断，预先限制滞后和平滑实验数量，避免追逐历史结果；后续留新日期作前瞻影子验证。审阅PR后才能标记发布闭环完成。
