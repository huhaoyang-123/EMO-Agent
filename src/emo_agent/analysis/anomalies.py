"""Screen deviations against prior valid observations."""
import math

from .loader import coverage_warning, period, records, restrict

CONTEXT_METRICS = ("breadth_up", "return_dispersion")
INDEX_METRICS = ("net_sentiment", "post_heat_20", "return_1", "return_5")
ANOMALY_METRICS = (*INDEX_METRICS, *CONTEXT_METRICS)


def detect_anomalies(data, start_date, end_date, metric, lookback=20, threshold=2.5):
    if metric not in ANOMALY_METRICS or type(lookback) is not int or not 15 <= lookback <= 252:
        raise ValueError("指标或历史窗口无效，历史窗口须为 15 至 252。")
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0.5 <= threshold <= 10:
        raise ValueError("阈值须为 0.5 至 10。")
    source = "context_daily" if metric in CONTEXT_METRICS else "index_daily"
    f = getattr(data, source)[["date", metric]].dropna().sort_values("date").copy()
    history = f[metric].shift(1).rolling(lookback, min_periods=15)
    f["baseline_mean"], f["baseline_std"] = history.mean(), history.std()
    f["baseline_n"] = f[metric].shift(1).rolling(lookback, min_periods=1).count().fillna(0).astype(int)
    f["z_score"] = (f[metric] - f.baseline_mean) / f.baseline_std.where(f.baseline_std.gt(0))
    requested = [start_date, end_date]
    filtered = restrict(f, *requested)
    candidates = filtered.loc[filtered.z_score.abs().ge(threshold)]
    warnings = coverage_warning(requested, period(f), metric) if not f.empty else []
    return {
        "tool": "detect_anomalies", "period": requested, "actual_period": period(filtered),
        "metric": metric, "observations": len(filtered), "scored_observations": int(filtered.z_score.notna().sum()),
        "lookback": lookback, "threshold": threshold, "count": len(candidates),
        "results": records(candidates.rename(columns={metric: "value"})),
        "interpretation_limit": "以此前有效观测计算均值和样本标准差。异常候选是工程筛查结果。",
        "sources": [data.sources[source]], "warnings": warnings,
    }
