import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pandas as pd
import pytest
from openai import OpenAI

from emo_agent.agent.orchestrator import ask
from emo_agent.agent.registry import ToolRegistry
from emo_agent.analysis.anomalies import detect_anomalies
from emo_agent.analysis.constituents import compare_constituents
from emo_agent.analysis.loader import COLUMNS, DataError, date_pair, load_analysis_data
from emo_agent.analysis.overview import get_market_overview
from emo_agent.analysis.relation import analyze_sentiment_market_relation

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/agent_v1.json").read_text(encoding="utf-8"))
SOURCE_NAMES = ["index_daily", "context_daily", "company_daily", "index_company_daily", "index_prices", "market_prices"]
APRIL = ("2025-04-01", "2025-04-30")


@pytest.fixture
def data():
    dates = pd.date_range("2025-04-01", periods=20)
    index = pd.DataFrame({name: np.arange(20) / 100 for name in COLUMNS["index_daily"] if name != "date"})
    index["date"] = dates
    index["close"] = 100 + np.arange(20)
    index["return_1"] = [-0.01, 0.01] * 10
    index["return_5"] = 0
    index["net_sentiment"] = [0.01, 0.01, -0.1, None] + [0.1] * 16
    index["eligible"] = True
    context = pd.DataFrame({name: 0.4 for name in COLUMNS["context_daily"] if name != "date"}, index=range(20))
    context["date"] = dates
    company = pd.DataFrame({name: [0.2, 100, 0.1, None] for name in COLUMNS["company_daily"] if name not in ("date", "code", "eligible")})
    company["date"] = [dates[0], dates[1], dates[0], dates[1]]
    company["code"] = ["000001", "000001", "000002", "000002"]
    company["eligible"] = [True, False, True, True]
    prices = index[["date", "close"]].assign(amount_100m=5, source="csindex_official")
    return SimpleNamespace(
        index_daily=index, context_daily=context, company_daily=company,
        index_company_daily=index, index_prices=prices, market_prices=prices,
        sources={name: name for name in SOURCE_NAMES},
    )


@pytest.fixture
def workspace(tmp_path, data):
    sources = CONFIG["datasets"]

    def put(relative, content):
        p = tmp_path / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return hashlib.sha256(p.read_bytes()).hexdigest()

    for name in SOURCE_NAMES:
        f = getattr(data, name)[list(COLUMNS[name])].copy()
        f["date"] = f.date.dt.strftime("%Y-%m-%d")
        f["target_1"] = 999
        put(sources[name], f.to_csv(index=False))
    digest = lambda key: hashlib.sha256((tmp_path / sources[key]).read_bytes()).hexdigest()
    put("configs/agent_v1.json", json.dumps(CONFIG))
    put("docs/data_manifest.json", json.dumps({"files": [{"path": sources["index_prices"], "sha256": digest("index_prices")}]}))
    groups = {"cutoff_v1": ["index_daily"], "context_v1": ["context_daily"],
              "company_pilot_v1": ["company_daily", "index_company_daily"]}
    for group, keys in groups.items():
        meta = {"outputs": {Path(sources[key]).name: digest(key) for key in keys}}
        if group == "context_v1":
            meta["inputs"] = {"market_index": {"path": sources["market_prices"], "sha256": digest("market_prices")}}
            meta["source_sample_sha256"] = digest("index_daily")
        put(f"data/derived/{group}/metadata.json", json.dumps(meta))
    return tmp_path


def test_loader_checks_hashes_and_removes_outcome_fields(workspace):
    data = load_analysis_data(workspace)
    assert data.company_daily.code.tolist()[0] == "000001"
    assert "target_1" not in data.index_daily and "target_1" not in data.company_daily
    assert data.availability["company_daily"]["eligible_count"] == 3
    path = workspace / CONFIG["datasets"]["index_daily"]
    original = path.read_bytes()
    path.write_bytes(original + b"\n")
    with pytest.raises(DataError, match="SHA-256"):
        load_analysis_data(workspace)
    path.unlink()
    with pytest.raises(DataError, match="缺少"):
        load_analysis_data(workspace)


def test_loader_rejects_valid_hash_with_absent_field_or_date_mismatch(workspace):
    meta_path = workspace / "data/derived/context_v1/metadata.json"
    path = workspace / CONFIG["datasets"]["context_daily"]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    original = pd.read_csv(path)
    for altered, msg in [
        (original.drop(columns="breadth_up"), "字段或记录无效"),
        (original.iloc[1:], "日期范围不一致"),
    ]:
        altered.to_csv(path, index=False)
        meta["outputs"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(DataError, match=msg):
            load_analysis_data(workspace)


@pytest.mark.parametrize("start,end", [
    ("20250401", "2025-04-30"), ("2025-02-30", "2025-04-30"),
    ("2025-05-01", "2025-04-30"), ("2025-04-01T00:00", "2025-04-30"),
])
def test_invalid_dates(start, end):
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        date_pair(start, end)


def test_overview_uses_separate_source_ranges_and_first_last_close(data):
    data.index_daily = data.index_daily.iloc[:4]
    result = get_market_overview(data, *APRIL)
    assert result["observations"] == 20
    assert result["index_metrics"]["period_return"] == pytest.approx(0.19)
    assert result["index_metrics"]["volatility_20_latest"] is None
    assert result["breadth_metrics"]["mean_breadth_up"] == pytest.approx(0.4)
    assert result["sentiment_observations"] == 3
    assert result["data_quality"]["sentiment_missing_days"] == 17
    beyond = get_market_overview(data, "2025-04-10", "2025-04-30")
    assert beyond["sentiment_metrics"] is None and beyond["observations"] == 11
    assert any("股吧统计情绪可用日期" in line for line in beyond["warnings"])
    assert get_market_overview(data, *APRIL, False)["data_quality"]["sentiment_missing_days"] is None


def test_constituent_ranking_excludes_ineligible_and_preserves_missing(data):
    result = compare_constituents(data, *APRIL, "net_sentiment")
    assert [r["code"] for r in result["results"]] == ["000001", "000002"]
    assert result["results"][0]["mean"] == pytest.approx(0.2)
    assert result["results"][0]["coverage"] == 0.5
    assert result["results"][1]["valid_days"] == 1
    assert result["excluded_or_missing_rows"] == 2


def test_relation_counts_only_paired_rows_and_handles_constant(data):
    result = analyze_sentiment_market_relation(data, *APRIL, 1)
    assert result["n"] == 19
    assert result["divergence_days"] == 9
    assert result["same_direction_days"] == 10
    assert result["pearson_correlation"] is not None
    assert analyze_sentiment_market_relation(data, *APRIL, 5)["pearson_correlation"] is None
    assert analyze_sentiment_market_relation(data, "2026-01-01", "2026-01-10")["n"] == 0


def test_anomaly_uses_prior_valid_history_including_before_requested_period(data):
    y = list(range(17)) + [None, 50, 1]
    data.index_daily["net_sentiment"] = y
    result = detect_anomalies(data, "2025-04-19", "2025-04-19", "net_sentiment", 15, 2.5)
    out = result["results"][0]
    assert out["baseline_n"] == 15 and out["baseline_mean"] == pytest.approx(9)
    assert out["z_score"] == pytest.approx((50 - 9) / np.std(list(range(2, 17)), ddof=1))
    assert detect_anomalies(data, "2025-04-01", "2025-04-15", "net_sentiment")["count"] == 0
    data.index_daily["net_sentiment"] = [1] * 19 + [20]
    assert detect_anomalies(data, *APRIL, "net_sentiment")["count"] == 0


@pytest.mark.parametrize("name,arguments", [
    ("run_python", "{}"), ("get_market_overview", '{"path":"/etc/passwd"}'),
    ("get_research_evidence", "[]"), ("get_research_evidence", '{"topic":"all","topic":"shadow_test"}'),
    ("detect_anomalies", '{"start_date":"2025-04-01","end_date":"2025-04-30","metric":"net_sentiment","threshold":NaN}'),
    ("compare_constituents", '{"start_date":"2025-04-01","end_date":"2025-04-30","metric":"net_sentiment","top_n":true}'),
])
def test_tool_registry_blocks_unlisted_or_invalid_requests(data, name, arguments):
    with pytest.raises(ValueError):
        ToolRegistry(data, ROOT).execute(name, arguments)


def _native_reply(request_id, *, calls=None, content=None):
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = [{"id": f"{request_id}-{i}", "type": "function",
                                  "function": {"name": name, "arguments": json.dumps(args)}}
                                 for i, (name, args) in enumerate(calls)]
    return httpx.Response(200, json={
        "id": request_id, "object": "chat.completion", "created": 0,
        "model": CONFIG["model"], "choices": [
            {"index": 0, "finish_reason": "tool_calls" if calls else "stop", "message": message},
        ],
    })


@pytest.mark.parametrize("question,name,args", [
    ("分析四月行情", "get_market_overview", {"start_date": APRIL[0], "end_date": APRIL[1]}),
    ("四月情绪和行情哪些日背离", "analyze_sentiment_market_relation", {"start_date": APRIL[0], "end_date": APRIL[1]}),
    ("情绪有预测价值吗", "get_research_evidence", {"topic": "all"}),
])
def test_deepseek_tool_round_trip_over_http(data, question, name, args):
    requests = []

    def respond(request):
        assert request.url.path == "/chat/completions"
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            assert body["thinking"] == {"type": "disabled"}
            assert body["model"] == "deepseek-flash"
            assert len(body["tools"]) == 5 and all(t["type"] == "function" for t in body["tools"])
            return _native_reply("ask", calls=[(name, args)])
        messages = body["messages"]
        assert messages[-2]["tool_calls"][0]["id"] == messages[-1]["tool_call_id"] == "ask-0"
        assert messages[-1]["role"] == "tool"
        assert "series" not in json.loads(messages[-1]["content"])
        return _native_reply("answer", content="工具结果给出了对应的观察。")

    with OpenAI(base_url="https://api.deepseek.com", api_key="test-key", max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
        answer = ask(question, ToolRegistry(data, ROOT), client, CONFIG)
    assert len(requests) == 2 and answer.results[0]["tool"] == name
    assert answer.text == "工具结果给出了对应的观察。"
    json.dumps(answer.results, allow_nan=False)


def test_multiround_errors_bounds_and_fabricated_numbers(data):
    requests = []
    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            return _native_reply("invalid", calls=[("run_hidden_code", {})])
        if len(requests) == 2:
            assert "error" in json.loads(body["messages"][-1]["content"])
            return _native_reply("ok", calls=[("get_research_evidence", {"topic": "shadow_test"})])
        assert body["tool_choice"] == "none"
        return _native_reply("finish", content="模型准确率为 999%。")

    with OpenAI(base_url="https://api.deepseek.com", api_key="test-key", max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
        answer = ask("查研究结论", ToolRegistry(data, ROOT), client, {**CONFIG, "max_tool_calls": 2})
    assert len(requests) == 3 and len(answer.results) == 2
    assert answer.results[0]["error"] and "999" not in answer.text


def test_api_failure_does_not_expose_provider_error_body(data):
    fake = httpx.MockTransport(lambda _: httpx.Response(401, json={"error": {"message": "sensitive fake detail"}}))
    with OpenAI(base_url="https://api.deepseek.com", api_key="test-key", max_retries=0,
                http_client=httpx.Client(transport=fake)) as client:
        answer = ask("查看行情", ToolRegistry(data, ROOT), client, CONFIG)
    assert "密钥" in answer.text and "sensitive" not in answer.text
