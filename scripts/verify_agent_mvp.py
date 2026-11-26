"""Verify the MVP against local snapshots and Streamlit without an API key."""
import json
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from emo_agent.agent.registry import ToolRegistry
from emo_agent.analysis.loader import load_analysis_data
from emo_agent.audit import audit_project


def verify(root):
    assert audit_project(root)["ok"]
    data = load_analysis_data(root)
    registry = ToolRegistry(data, root)

    def run(name, **args):
        result = registry.execute(name, json.dumps(args))
        json.dumps(result, allow_nan=False)
        return result

    q1 = {"start_date": "2025-01-01", "end_date": "2025-03-31"}
    h1 = {"start_date": q1["start_date"], "end_date": "2025-06-30"}
    overview = run("get_market_overview", **q1)
    raw_prices = data.index_prices.set_index("date").loc[q1["start_date"]:q1["end_date"], "close"]
    assert overview["observations"] == overview["sentiment_observations"] == overview["breadth_observations"] == len(raw_prices)
    expected_return = raw_prices.iloc[-1] / raw_prices.iloc[0] - 1
    assert abs(overview["index_metrics"]["period_return"] - expected_return) < 1e-12

    top = run("compare_constituents", **h1, metric="post_heat_20", top_n=5)
    company = data.company_daily
    eligible = company.loc[company.eligible & company.date.between(h1["start_date"], h1["end_date"])]
    expected = eligible.groupby("code").post_heat_20.mean().dropna().sort_values(ascending=False).head(5)
    assert [row["code"] for row in top["results"]] == list(expected.index)
    relation = run("analyze_sentiment_market_relation", **h1)
    frame = data.index_daily.set_index("date").loc[h1["start_date"]:h1["end_date"]]
    assert relation["positive_sentiment_negative_return_days"] == int(((frame.net_sentiment > 0) & (frame.return_1 < 0)).sum())
    anomalies = run("detect_anomalies", **h1, metric="net_sentiment")
    assert all(abs(row["z_score"]) >= 2.5 for row in anomalies["results"])
    evidence = run("get_research_evidence", topic="all")
    assert len(evidence["results"]) == 6
    august = run("get_market_overview", start_date="2026-08-01", end_date="2026-08-31")
    assert august["observations"] > 0 and august["sentiment_metrics"] is None

    with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "", "EMO_AGENT_DATA_ROOT": str(root)}):
        app = AppTest.from_file(str(root / "src/emo_agent/app.py"), default_timeout=45).run()
        assert not app.exception and app.chat_input[0].disabled
        app.radio[0].set_value("工具直查").run()
        assert not app.exception
        app.button[0].click().run()
        assert not app.exception and len(app.metric) == 3
        assert app.metric[0].value == f"{expected_return:.2%}"

    return {
        "q1_index_return": expected_return, "q1_observations": len(raw_prices),
        "top_five_codes": list(expected.index), "relation_pairs": relation["n"],
        "positive_sentiment_negative_return_days": relation["positive_sentiment_negative_return_days"],
        "anomaly_candidates": anomalies["count"], "evidence_topics": len(evidence["results"]),
        "august_index_days_without_sentiment": august["observations"],
        "streamlit_no_key_and_offline_result_cards": "passed",
    }


if __name__ == "__main__":
    print(json.dumps(verify(Path.cwd()), indent=2))
