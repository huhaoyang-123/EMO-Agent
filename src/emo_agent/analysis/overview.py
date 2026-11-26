"""Index, sentiment and breadth summaries."""
import numpy as np

from .loader import SENTIMENT_METRICS, coverage_warning, number, period, records, restrict


def get_market_overview(data, start_date, end_date, include_sentiment=True):
    requested = [start_date, end_date]
    prices = data.index_prices.copy()
    prices["return_1"] = prices.close.pct_change(fill_method=None)
    prices["volatility_20"] = prices.return_1.rolling(20, min_periods=20).std()
    prices["amount_relative_20"] = prices.amount_100m / prices.amount_100m.shift(1).rolling(20).mean().replace(0, np.nan) - 1
    idx = restrict(prices, *requested)
    market = data.market_prices.copy()
    market["market_return_1"] = market.close.pct_change(fill_method=None)
    market["market_volatility_20"] = market.market_return_1.rolling(20, min_periods=20).std()
    broad = restrict(market, *requested)
    ctx = restrict(data.context_daily, *requested)
    valid_ctx = ctx.dropna(subset=["breadth_up", "return_dispersion"])
    breadth_range = period(data.context_daily.dropna(subset=["breadth_up", "return_dispersion"]))
    warnings = coverage_warning(requested, period(prices), "931723 行情")
    warnings += coverage_warning(requested, period(market), "中证全指行情")
    if breadth_range:
        warnings += coverage_warning(requested, breadth_range, "成分股广度派生表")
    if "csindex_backfill" in idx.source.values:
        warnings.append("所选区间包含指数发布前的回算行情。")
    if idx.empty:
        warnings.append("请求区间没有指数交易日。")

    sentiment, sent_range, valid_sent = None, None, 0
    series = idx[["date", "close", "return_1"]].copy()
    if include_sentiment:
        available = data.index_daily.dropna(subset=["net_sentiment"])
        sent = restrict(available, *requested)
        sent_range, valid_sent = period(sent), len(sent)
        warnings += coverage_warning(requested, period(available), "股吧统计情绪")
        if not sent.empty:
            sentiment = {key: number(sent[key].mean()) for key in SENTIMENT_METRICS}
            series = series.merge(sent[["date", "net_sentiment", "post_heat_20"]], on="date", how="left")
            warnings.append("情绪按统计时段对齐，历史真实发布和修订时间未核实。")

    return {
        "tool": "get_market_overview", "requested_period": requested, "actual_period": period(idx),
        "observations": len(idx), "sentiment_period": sent_range, "sentiment_observations": valid_sent,
        "breadth_period": period(valid_ctx), "breadth_observations": len(valid_ctx),
        "broad_market_period": period(broad), "broad_market_observations": len(broad),
        "index_metrics": {
            "period_return": number(idx.close.iloc[-1] / idx.close.iloc[0] - 1) if len(idx) > 1 else None,
            "mean_daily_return": number(idx.return_1.mean()),
            "volatility_20_latest": number(idx.volatility_20.iloc[-1]) if len(idx) else None,
            "mean_amount_relative_20": number(idx.amount_relative_20.mean()),
        },
        "sentiment_metrics": sentiment,
        "breadth_metrics": {
            "mean_breadth_up": number(valid_ctx.breadth_up.mean()),
            "mean_return_dispersion": number(valid_ctx.return_dispersion.mean()),
            "mean_constituents_observed": number(valid_ctx.breadth_n.mean()),
        },
        "broad_market_metrics": {
            "period_return": number(broad.close.iloc[-1] / broad.close.iloc[0] - 1) if len(broad) > 1 else None,
            "mean_daily_return": number(broad.market_return_1.mean()),
        },
        "data_quality": {"sentiment_missing_days": len(idx) - valid_sent if include_sentiment else None,
                         "context_missing_days": len(idx) - len(valid_ctx)},
        "method": "区间收益为区间最后一个收盘价除以第一个收盘价减一。收益和波动以小数表示。广度来自历史成分的 Dretnd。",
        "sources": [data.sources[key] for key in ("index_prices", "market_prices", "context_daily")]
                   + ([data.sources["index_daily"]] if include_sentiment else []),
        "warnings": warnings, "series": records(series),
    }
