"""Compare eligible historical constituents over the same requested dates."""
from .loader import COMPANY_METRICS, coverage_warning, period, records, restrict


def compare_constituents(data, start_date, end_date, metric, direction="highest", top_n=10):
    if metric not in COMPANY_METRICS or direction not in ("highest", "lowest") or type(top_n) is not int or not 1 <= top_n <= 20:
        raise ValueError("无效的公司指标、排序方向或数量。")
    requested = [start_date, end_date]
    f = restrict(data.company_daily, *requested)
    valid = f.loc[f.eligible].dropna(subset=[metric])
    ranks = valid.groupby("code")[metric].agg(valid_days="count", mean="mean", median="median")
    ranks["coverage"] = ranks.valid_days / f.groupby("code").size().reindex(ranks.index)
    ranks["mean_slot_coverage"] = valid.groupby("code").slot_coverage.mean()
    ranks = ranks.reset_index().sort_values(["mean", "code"], ascending=[direction == "lowest", True]).head(top_n)
    warnings = coverage_warning(requested, period(data.company_daily), "成分股分析面板")
    if valid.empty:
        warnings.append("该区间没有该指标的合格观测。")
    return {
        "tool": "compare_constituents", "period": requested, "actual_period": period(valid),
        "metric": metric, "direction": direction, "observations": len(valid),
        "candidate_companies": int(valid.code.nunique()), "excluded_or_missing_rows": len(f) - len(valid),
        "method": "按合格公司日的指标均值排序。覆盖率为有效天数除以公司在所选区间的面板天数。",
        "results": records(ranks), "sources": [data.sources["company_daily"]], "warnings": warnings,
    }
