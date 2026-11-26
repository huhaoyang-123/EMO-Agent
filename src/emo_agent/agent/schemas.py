"""Provider tool schemas and local parameter constraints."""
from emo_agent.analysis.anomalies import ANOMALY_METRICS
from emo_agent.analysis.evidence import TOPICS
from emo_agent.analysis.loader import COMPANY_METRICS

DATE = {"type": "string", "pattern": r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$", "description": "YYYY-MM-DD"}
DATES = {"start_date": DATE, "end_date": DATE}


def choice(items, value_type="string", default=None):
    schema = {"type": value_type, "enum": list(items)}
    return schema if default is None else {**schema, "default": default}


def bounded(kind, lower, upper, default):
    return {"type": kind, "minimum": lower, "maximum": upper, "default": default}


def tool(name, description, properties):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties,
                       "required": [key for key, schema in properties.items() if "default" not in schema],
                       "additionalProperties": False},
    }}


TOOLS = [
    tool("get_market_overview", "查询光通信指数市场概览、宽基、广度和股吧情绪。用户询问情绪时保留 include_sentiment=true。",
         {**DATES, "include_sentiment": {"type": "boolean", "default": True}}),
    tool("compare_constituents", "按指定指标比较历史成分股。讨论热度上升使用 post_heat_20。",
         {**DATES, "metric": choice(COMPANY_METRICS), "direction": choice(("highest", "lowest"), default="highest"),
          "top_n": bounded("integer", 1, 20, 10)}),
    tool("analyze_sentiment_market_relation", "分析净情绪与同期收益相关性，查找正情绪下跌或负情绪上涨的日期。",
         {**DATES, "return_window": choice((1, 5), "integer", 1)}),
    tool("detect_anomalies", "按过去有效样本的 Z 分数筛查异常候选。",
         {**DATES, "metric": choice(ANOMALY_METRICS), "lookback": bounded("integer", 15, 252, 20),
          "threshold": bounded("number", 0.5, 10, 2.5)}),
    tool("get_research_evidence", "查询正式研究结论。回答情绪预测有效性、历史模型评价和数据限制时使用。",
         {"topic": choice(TOPICS, default="all")}),
]
