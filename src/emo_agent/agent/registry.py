"""Validate requested parameters before dispatching a fixed local function."""
from functools import partial
import json
import math
import re

from emo_agent.analysis.anomalies import detect_anomalies
from emo_agent.analysis.constituents import compare_constituents
from emo_agent.analysis.evidence import get_research_evidence
from emo_agent.analysis.overview import get_market_overview
from emo_agent.analysis.relation import analyze_sentiment_market_relation

from .schemas import TOOLS

SCHEMAS = {entry["function"]["name"]: entry["function"]["parameters"] for entry in TOOLS}
VALUE_TYPES = {"string": (str,), "integer": (int,), "number": (int, float), "boolean": (bool,)}


def _unique_pairs(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("参数键重复。")
    return result


def _parse(name, raw):
    if name not in SCHEMAS:
        raise ValueError("请求了未注册的分析工具。")
    if not isinstance(raw, str) or len(raw) > 4096:
        raise ValueError("工具参数过长或格式错误。")
    try:
        args = json.loads(raw, object_pairs_hook=_unique_pairs)
    except (ValueError, RecursionError) as exc:
        raise ValueError("工具参数不是有效的 JSON 对象。") from exc
    schema = SCHEMAS[name]
    props = schema["properties"]
    if not isinstance(args, dict) or set(args) - props.keys() or set(schema["required"]) - args.keys():
        raise ValueError("工具参数有缺失或包含未定义字段。")
    for key, value in args.items():
        rule = props[key]
        if type(value) not in VALUE_TYPES[rule["type"]]:
            raise ValueError(f"工具参数 {key} 的类型错误。")
        if "enum" in rule and value not in rule["enum"]:
            raise ValueError(f"工具参数 {key} 不在允许范围。")
        if "pattern" in rule and not re.fullmatch(rule["pattern"], value):
            raise ValueError(f"工具参数 {key} 须使用 YYYY-MM-DD。")
        if "minimum" in rule and (not rule["minimum"] <= value <= rule["maximum"] or not math.isfinite(value)):
            raise ValueError(f"工具参数 {key} 不在允许范围。")
    return args


class ToolRegistry:
    def __init__(self, data, root):
        self._tools = {fn.__name__: partial(fn, data) for fn in (
            get_market_overview, compare_constituents, analyze_sentiment_market_relation, detect_anomalies
        )}
        self._tools["get_research_evidence"] = partial(get_research_evidence, root=root)

    def execute(self, name, raw):
        args = _parse(name, raw)
        return self._tools[name](**args)
