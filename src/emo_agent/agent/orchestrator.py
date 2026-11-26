"""Bounded multi-round tool calls and structured evidence for the UI."""
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
import json
import re

from openai import APIConnectionError, APIStatusError, APITimeoutError

from .prompts import SYSTEM_PROMPT
from .schemas import TOOLS

MAX_QUESTION_CHARS = 2000


@dataclass
class Answer:
    text: str
    results: list[dict]


def _json(data):
    return json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _for_model(result):
    # The full series and detailed tables remain local for deterministic rendering.
    condensed = {k: v for k, v in result.items() if k not in ("series", "divergence_records")}
    if "results" in condensed:
        condensed["results"] = condensed["results"][:20]
    return _json(condensed)


def _safe_summary(text, results):
    valid = [item for item in results if "error" not in item]
    if not valid:
        return "本轮没有获得可用的分析依据。请核对日期、指标或数据状态。" if results else (
            "请说明要查询的历史区间和指标，或指定想核查的研究结论。本工具支持历史分析。"
        )
    if all("observations" in item and item["observations"] == 0 or "n" in item and item["n"] == 0
           for item in valid):
        return "所选区间没有匹配的有效观测。可用区间和原因见数据依据。"
    # Counts, dates and percentages are displayed directly from Python results.
    unsafe = r"\d|[一二两三四五六七八九十百千点]+\s*(?:个|只|日|天|月|年|成|倍|%|％|百分点)|[零一二两三四五六七八九]点[零一二两三四五六七八九]|(?:建议|应该)(?:买入|卖出|加仓|减仓)|因果成立|保证收益"
    if not text or re.search(unsafe, text) or re.search(r"https?://|<[^>]+>", text):
        return "本次分析使用了下方工具结果。可查看对应指标、实际样本区间和限制。"
    return text.strip()


def _perform(registry, name, args, seconds):
    pool = ThreadPoolExecutor(max_workers=1)
    job = pool.submit(registry.execute, name, args)
    try:
        return job.result(timeout=seconds)
    except TimeoutError:
        return {"tool": name, "error": "分析工具超时。"}
    except ValueError as exc:
        return {"tool": name, "error": str(exc)}
    except Exception:
        return {"tool": name, "error": "分析工具执行失败，请检查本地数据。"}
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def ask(question, registry, client, config, *, model=None, history=()):
    question = question.strip()
    if not question or len(question) > MAX_QUESTION_CHARS:
        return Answer("请输入长度合适的分析问题。", [])
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in list(history)[-6:]:
        if item["role"] in ("user", "assistant"):
            messages.append({"role": item["role"], "content": item["content"][:MAX_QUESTION_CHARS]})
    messages.append({"role": "user", "content": question})
    results = []
    limit = config["max_tool_calls"]
    rounds = config["max_tool_rounds"]
    try:
        for turn in range(rounds + 1):
            allow_calls = turn < rounds and len(results) < limit
            completion = client.chat.completions.create(
                model=model or config["model"], messages=messages, tools=TOOLS,
                tool_choice="auto" if allow_calls else "none",
                extra_body={"thinking": {"type": "disabled"}}, max_tokens=1400,
            )
            if not completion.choices:
                break
            choice = completion.choices[0]
            message = choice.message
            calls = message.tool_calls or []
            if not calls:
                text = message.content if choice.finish_reason == "stop" else None
                return Answer(_safe_summary(text, results), results)
            if not allow_calls or len(calls) > limit - len(results):
                return Answer("本轮已达到工具调用上限。请缩小问题范围。", results)
            ids = [call.id for call in calls]
            if len(set(ids)) != len(ids) or any(call.type != "function" or not call.id for call in calls):
                return Answer("DeepSeek 返回了无法识别的工具请求。", results)
            messages.append({"role": "assistant", "content": message.content,
                             "tool_calls": [call.model_dump(include={"id", "type", "function"}) for call in calls]})
            for call in calls:
                value = _perform(registry, call.function.name, call.function.arguments, config["tool_timeout_seconds"])
                results.append(value)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": _for_model(value)})
                if value.get("error") == "分析工具超时。":
                    return Answer("本轮分析超时。已完成的结果仍可查看。", results)
    except APITimeoutError:
        return Answer("连接 DeepSeek 超时。", results)
    except APIConnectionError:
        return Answer("无法连接 DeepSeek，请检查网络。", results)
    except APIStatusError as exc:
        hints = {401: "请核对或更新 DeepSeek 密钥。", 402: "请检查 DeepSeek 账户余额。",
                 429: "请求过于频繁，请稍后重试。"}
        return Answer(f"DeepSeek 请求失败。{hints.get(exc.status_code, '请核对模型设置或稍后重试。')}", results)
    return Answer(_safe_summary(None, results), results)
