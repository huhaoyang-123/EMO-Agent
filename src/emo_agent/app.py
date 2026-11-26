"""Local Streamlit interface for the 931723 analysis Agent."""
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from emo_agent.agent.client import create_client
from emo_agent.agent.orchestrator import MAX_QUESTION_CHARS, ask
from emo_agent.agent.registry import ToolRegistry
from emo_agent.agent.schemas import TOOLS
from emo_agent.analysis.loader import DataError, config_at, load_analysis_data
from emo_agent.audit import audit_project

TITLES = {
    "get_market_overview": "市场概览",
    "compare_constituents": "成分股比较",
    "analyze_sentiment_market_relation": "情绪与行情",
    "detect_anomalies": "异常候选",
    "get_research_evidence": "研究证据",
}
SOURCE_NAMES = {
    "index_prices": "光通信指数行情", "sentiment": "股吧统计情绪",
    "market_prices": "中证全指行情", "context_daily": "成分股广度表",
    "company_daily": "公司分析面板",
}
EXAMPLES = (
    "分析 2025 年第一季度光通信板块的行情、市场广度和股吧情绪。",
    "2025 年上半年哪些历史成分股的讨论热度上升最明显？列出前五名。",
    "2025 年上半年哪些交易日情绪偏正面，但指数下跌？",
    "此前的实验是否支持用股吧情绪预测指数涨跌？",
)
PARAM_LABELS = {"metric": "指标", "direction": "排序方向", "top_n": "展示公司数",
                "return_window": "收益窗口（交易日）", "lookback": "历史观测窗口",
                "threshold": "Z 分数阈值", "topic": "研究主题"}
GITHUB = "https://github.com/huhaoyang-123/EMO-Agent/blob/main/"


@st.cache_resource(show_spinner="正在核对本地数据与元数据…")
def _load(root, version):
    if not audit_project(root)["ok"]:
        raise DataError("基础输入缺失或审计失败，请按 data/README.md 检查。")
    return load_analysis_data(root)


def _version(root, config):
    # Hashing occurs in the loader. File timestamps invalidate the cached snapshots.
    selected = {root / value for value in config["datasets"].values()}
    selected.update({path.parent / "metadata.json" for path in selected if "derived" in path.parts})
    selected.update(root / value for value in (
        "configs/agent_v1.json", "docs/data_manifest.json",
        "data/input/context/member_stock_returns.csv",
    ))
    selected.update(root / item["path"] for item in
                    json.loads((root / "docs/data_manifest.json").read_text(encoding="utf-8-sig"))["files"])
    return tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) if path.is_file()
                 else (str(path), None, None) for path in sorted(selected))


def _percent(value):
    return "暂无数据" if value is None else f"{value:.2%}"


def _date_span(span):
    return "暂无数据" if span is None else " 至 ".join(span)


def _table(rows, **kwargs):
    if rows:
        frame = pd.DataFrame(rows)
        if "date" in frame:
            frame.date = frame.date.str[:10]
        st.dataframe(frame, width="stretch", hide_index=True, **kwargs)


def _overview(result):
    columns = st.columns(3)
    columns[0].metric("区间指数收益", _percent(result["index_metrics"]["period_return"]))
    columns[1].metric("日均成分股上涨比例", _percent(result["breadth_metrics"]["mean_breadth_up"]))
    net = (result["sentiment_metrics"] or {}).get("net_sentiment")
    columns[2].metric("平均净情绪", _percent(net))
    index = result["index_metrics"]
    heat = (result["sentiment_metrics"] or {}).get("post_heat_20")
    st.caption(f"最新二十日波动率 {_percent(index['volatility_20_latest'])} ｜ "
               f"日均相对成交额 {_percent(index['mean_amount_relative_20'])} ｜ "
               f"平均讨论热度 {'暂无数据' if heat is None else f'{heat:.3f}'}")
    st.caption(f"指数 {_date_span(result['actual_period'])} · {result['observations']} 个交易日 ｜ "
               f"股吧 {_date_span(result['sentiment_period'])} · {result['sentiment_observations']} 个交易日 ｜ "
               f"广度 {result['breadth_observations']} 个交易日")
    if rows := result["series"]:
        frame = pd.DataFrame(rows)
        frame.date = pd.to_datetime(frame.date)
        st.line_chart(frame, x="date", y="close", x_label="日期", y_label="指数收盘点位")
        if "net_sentiment" in frame and frame.net_sentiment.notna().any():
            st.line_chart(frame, x="date", y="net_sentiment", x_label="日期", y_label="净情绪（小数）")


def _show(result, widget_key="direct"):
    name = result["tool"]
    st.markdown(f"**{TITLES.get(name, '分析结果')}**")
    if error := result.get("error"):
        st.warning(error)
        return
    if name == "get_market_overview":
        _overview(result)
    elif name == "compare_constituents":
        st.caption(f"{_date_span(result['actual_period'])} · 共 {result['observations']} 个有效公司日")
        _table(result["results"], column_config={
            "code": st.column_config.TextColumn("证券代码"),
            "valid_days": st.column_config.NumberColumn("有效天数"),
            "coverage": st.column_config.NumberColumn("有效覆盖", format="percent"),
            "mean": st.column_config.NumberColumn("均值"),
            "median": st.column_config.NumberColumn("中位数"),
            "mean_slot_coverage": st.column_config.NumberColumn("平均时段覆盖", format="percent"),
        })
    elif name == "analyze_sentiment_market_relation":
        columns = st.columns(3)
        corr = result["pearson_correlation"]
        columns[0].metric("Pearson 相关", "暂无数据" if corr is None else f"{corr:.3f}")
        columns[1].metric("背离日", result["divergence_days"])
        columns[2].metric("有效观测", result["n"])
        selected = st.radio("背离方向", ["全部", "正情绪、收益为负", "负情绪、收益为正"],
                            horizontal=True, key=f"divergence_{widget_key}")
        rows = result["divergence_records"]
        if selected != "全部":
            sign = 1 if selected == "正情绪、收益为负" else -1
            rows = [row for row in rows if sign * row["net_sentiment"] > 0]
        _table(rows)
    elif name == "detect_anomalies":
        st.metric("异常候选", result["count"])
        _table(result["results"])
    elif name == "get_research_evidence":
        for item in result["results"]:
            st.markdown(f"**{item['title']}** · {item['finding']} {item['limit']} [来源]({GITHUB}{item['source']})")
    for warning in result.get("warnings", []):
        st.info(warning)
    with st.expander("本次分析的数据依据"):
        st.json({k: v for k, v in result.items() if k not in ("series", "divergence_records", "results")})


def _direct(registry, availability):
    labels = {title: name for name, title in TITLES.items()}
    name = labels[st.selectbox("分析工具", labels)]
    properties = next(item["function"]["parameters"]["properties"] for item in TOOLS
                      if item["function"]["name"] == name)
    with st.form(key="direct_" + name):
        params = {}
        if "start_date" in properties:
            dates = st.columns(2)
            params["start_date"] = dates[0].date_input("开始日期", value=pd.Timestamp("2025-01-01")).isoformat()
            params["end_date"] = dates[1].date_input("结束日期", value=pd.Timestamp("2025-03-31")).isoformat()
            st.caption(f"股吧统计截至 {availability['sentiment']['max_date']}。")
        for key, rule in properties.items():
            if key in params:
                continue
            default = rule.get("default")
            if "enum" in rule:
                options = rule["enum"]
                params[key] = st.selectbox(PARAM_LABELS[key], options, index=options.index(default) if default in options else 0)
            elif rule["type"] == "boolean":
                params[key] = st.checkbox("同时查看股吧情绪", default)
            else:
                params[key] = st.number_input(PARAM_LABELS[key], min_value=rule["minimum"], max_value=rule["maximum"], value=default)
        if st.form_submit_button("运行分析", type="primary"):
            try:
                st.session_state.direct_result = registry.execute(name, json.dumps(params))
            except ValueError as exc:
                st.session_state.direct_result = {"tool": name, "error": str(exc)}
    previous = st.session_state.get("direct_result")
    if previous and previous["tool"] == name:
        _show(previous)


def _chat(registry, config):
    with st.sidebar:
        key = st.text_input("DeepSeek API 密钥", type="password",
                            help="可留空并从 DEEPSEEK_API_KEY 读取。本地会话仅在请求时使用。")
        model = st.text_input("DeepSeek 模型", value=os.getenv("DEEPSEEK_MODEL") or config["model"])
        configured = bool(key.strip() or os.getenv("DEEPSEEK_API_KEY", "").strip())
        st.caption("接口已配置，连接将在提问时验证" if configured else "接口未配置，可使用工具直查")
        st.caption("对话模式会将问题、最近会话及精简的工具结果发送给 DeepSeek。")
    if configured:
        if example := st.selectbox("示例问题", EXAMPLES, index=None, placeholder="选择一个问题"):
            use_example = st.button("运行示例")
        else:
            use_example = False
    else:
        example, use_example = None, False
        st.info("使用 DeepSeek 对话前请在本机配置密钥。五项分析工具可在「工具直查」中直接运行。")
    history = st.session_state.setdefault("chat", [])
    shown = 0
    for entry in history:
        with st.chat_message(entry["role"]):
            st.text(entry["content"])
            for result in entry.get("results", []):
                _show(result, f"chat_{shown}")
                shown += 1
    typed = st.chat_input("输入历史分析问题", disabled=not configured, max_chars=MAX_QUESTION_CHARS)
    question = typed or (example if use_example else None)
    if question:
        with st.chat_message("user"):
            st.text(question)
        with st.chat_message("assistant"):
            with st.spinner("正在调用分析工具…"):
                with create_client(config, api_key=key) as client:
                    answer = ask(question, registry, client, config, model=model.strip(), history=history)
            st.text(answer.text)
            for result in answer.results:
                _show(result, f"chat_{shown}")
                shown += 1
        history.extend([{"role": "user", "content": question},
                        {"role": "assistant", "content": answer.text, "results": answer.results}])
        del history[:-20]


def main():
    st.set_page_config(page_title="EMO-Agent · 光通信分析", page_icon="📈", layout="wide")
    st.title("光通信板块市场与情绪分析")
    st.caption("中证光通信主题指数 931723 · 历史行情、成分股广度与股吧统计")
    root = Path(os.getenv("EMO_AGENT_DATA_ROOT", str(Path.cwd()))).resolve()
    try:
        config = config_at(root)
        data = _load(root, _version(root, config))
    except (DataError, KeyError, OSError, ValueError) as exc:
        with st.sidebar:
            st.error("数据审计失败")
        st.error(str(exc) if isinstance(exc, DataError) else "数据或元数据不可用，请检查项目根目录和数据准备说明。")
        return
    with st.sidebar:
        st.header("数据状态")
        st.success("输入审计与派生文件校验通过")
        for name, title in SOURCE_NAMES.items():
            info = data.availability[name]
            st.caption(f"{title} · {info['min_date']} 至 {info['max_date']}")
    registry = ToolRegistry(data, root)
    mode = st.radio("分析方式", ["DeepSeek 对话", "工具直查"], horizontal=True)
    if mode == "DeepSeek 对话":
        _chat(registry, config)
    else:
        _direct(registry, data.availability)


if __name__ == "__main__":
    main()
