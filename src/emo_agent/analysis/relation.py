"""Describe co-movement with contemporaneous trailing returns."""
from .loader import coverage_warning, number, period, records, restrict


def analyze_sentiment_market_relation(data, start_date, end_date, return_window=1):
    if type(return_window) is not int or return_window not in (1, 5):
        raise ValueError("收益窗口只支持 1 或 5 个交易日。")
    requested = [start_date, end_date]
    col = f"return_{return_window}"
    f = restrict(data.index_daily, *requested)
    valid = f.dropna(subset=["net_sentiment", col]).copy()
    product = valid.net_sentiment * valid[col]
    diverging = valid.loc[product.lt(0), ["date", "net_sentiment", col]]
    can_correlate = len(valid) > 1 and valid.net_sentiment.nunique() > 1 and valid[col].nunique() > 1
    available = data.index_daily.dropna(subset=["net_sentiment", col])
    warnings = coverage_warning(requested, period(available), "情绪与行情的共同区间") if len(available) else []
    if not can_correlate:
        warnings.append("有效观测不足或指标为常数，Pearson 相关系数不可用。")
    return {
        "tool": "analyze_sentiment_market_relation", "period": requested, "actual_period": period(valid),
        "return_window": return_window, "n": len(valid), "missing_days": len(f) - len(valid),
        "pearson_correlation": number(valid.net_sentiment.corr(valid[col])) if can_correlate else None,
        "same_direction_days": int(product.gt(0).sum()), "divergence_days": len(diverging),
        "zero_direction_days": int(product.eq(0).sum()),
        "positive_sentiment_negative_return_days": int(diverging.net_sentiment.gt(0).sum()),
        "negative_sentiment_positive_return_days": int(diverging.net_sentiment.lt(0).sum()),
        "divergence_records": records(diverging),
        "interpretation_limit": "同期描述性关系仅用于历史回顾，不能用来判定因果、未来方向或交易信号。",
        "sources": [data.sources["index_daily"]], "warnings": warnings,
    }
