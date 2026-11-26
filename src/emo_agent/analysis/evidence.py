"""Serve curated evidence with repository sources."""
from pathlib import Path

from emo_agent.audit import safe_path
from .loader import read_json

TOPICS = ("index_baseline", "sentiment_stability", "company_pilot", "market_context",
          "shadow_test", "data_limitations", "all")


def get_research_evidence(topic="all", *, root=None):
    if topic not in TOPICS:
        raise ValueError("未知研究主题。")
    root = Path.cwd() if root is None else root
    evidence = read_json(root / "docs/agent_evidence.json")
    selected = TOPICS[:-1] if topic == "all" else (topic,)
    results = [{"topic": key, **evidence[key]} for key in selected]
    sources = list(dict.fromkeys(row[k] for row in results for k in ("source", "related") if k in row))
    if any(not safe_path(root, path).is_file() for path in sources):
        raise ValueError("研究证据的来源报告缺失。")
    return {"tool": "get_research_evidence", "topic": topic, "results": results,
            "sources": sources}
