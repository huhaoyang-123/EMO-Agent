"""Verify local snapshots and expose only fields available to analysis."""
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd

from emo_agent.audit import safe_path
from emo_agent.dataset import sha256

COMPANY_METRICS = (
    "net_sentiment", "sentiment_change_5", "log_post_rate", "post_heat_20",
    "stock_return_1", "stock_return_5", "stock_volatility_20", "volume_heat_20",
)
SENTIMENT_METRICS = (
    "net_sentiment", "equal_weight_sentiment", "post_heat_20",
    "company_coverage", "slot_coverage",
)
COLUMNS = {
    "index_daily": ("date", "close", "return_1", "return_5", "volatility_20",
                    "amount_relative_20", "sentiment_missing", "eligible", *SENTIMENT_METRICS),
    "context_daily": ("date", "market_return_1", "market_volatility_20",
                      "relative_return_5", "breadth_up", "return_dispersion", "breadth_n"),
    "company_daily": ("date", "code", "eligible", "slot_coverage", *COMPANY_METRICS),
    "index_company_daily": ("date", "return_1", "return_5", "net_sentiment", "eligible"),
    "index_prices": ("date", "close", "amount_100m", "source"),
    "market_prices": ("date", "close"),
}


class DataError(ValueError):
    """A data preparation problem safe to report without a traceback."""


@dataclass
class AnalysisData:
    index_daily: pd.DataFrame
    context_daily: pd.DataFrame
    company_daily: pd.DataFrame
    index_company_daily: pd.DataFrame
    index_prices: pd.DataFrame
    market_prices: pd.DataFrame
    availability: dict
    sources: dict


def config_at(root: Path) -> dict:
    return read_json(root / "configs/agent_v1.json")


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError
        return value
    except (OSError, ValueError) as exc:
        raise DataError(f"无法读取 {path.name}，请按 data/README.md 准备文件。") from exc


def _require_hash(path, expected, relative):
    if not path.is_file():
        raise DataError(f"缺少 {relative}，请按 data/README.md 准备文件。")
    if not expected or sha256(path) != expected:
        raise DataError(f"{relative} 的 SHA-256 与记录不一致。")


def _read_frame(path, columns, relative):
    try:
        frame = pd.read_csv(path, usecols=list(columns), dtype={"code": "string", "date": "string"})
        if frame.empty or not frame.date.str.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}").fillna(False).all():
            raise ValueError("日期格式错误或无记录")
        frame["date"] = pd.to_datetime(frame.date, format="%Y-%m-%d")
        keys = ["date", "code"] if "code" in frame else ["date"]
        if frame.duplicated(keys).any():
            raise ValueError("日期或公司日键重复")
        if "code" in frame and not frame.code.str.fullmatch(r"[0-9]{6}").fillna(False).all():
            raise ValueError("证券代码格式错误")
        if "eligible" in frame:
            if not frame.eligible.isin([True, False]).all():
                raise ValueError("eligible 字段无效")
            frame["eligible"] = frame.eligible.astype(bool)
        for column in set(columns) - {"date", "code", "eligible", "source"}:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
            if np.isinf(frame[column]).any():
                raise ValueError(f"{column} 有无穷值")
        if "close" in frame and (frame.close.isna() | frame.close.le(0)).any():
            raise ValueError("收盘价必须为正、非空")
        return frame.sort_values(keys).reset_index(drop=True)
    except (OSError, ValueError) as exc:
        raise DataError(f"{relative} 的字段或记录无效。") from exc


def _describe(frame, source, value_columns=None):
    eligible = frame[frame.eligible] if "eligible" in frame else frame
    fields = value_columns or list(frame.select_dtypes(include="number").columns)
    return {
        "path": source, "min_date": frame.date.min().date().isoformat(),
        "max_date": frame.date.max().date().isoformat(), "row_count": len(frame),
        "eligible_count": len(eligible), "missing_count": int(frame[fields].isna().any(axis=1).sum()),
        "metadata_hash_valid": True,
    }


def load_analysis_data(root: Path) -> AnalysisData:
    root = root.resolve()
    sources = config_at(root)["datasets"]
    if set(sources) != set(COLUMNS):
        raise DataError("configs/agent_v1.json 的数据源配置无效。")
    base = read_json(root / "docs/data_manifest.json")
    base_hashes = {entry["path"]: entry["sha256"] for entry in base["files"]}
    context_path = safe_path(root, sources["context_daily"])
    context_meta = read_json(context_path.parent / "metadata.json")
    for item in context_meta["inputs"].values():
        _require_hash(safe_path(root, item["path"]), item["sha256"], item["path"])
    frames = {}
    for name, columns in COLUMNS.items():
        relative = sources[name]
        path = safe_path(root, relative)
        if name == "index_prices":
            expected = base_hashes.get(relative)
        elif name == "market_prices":
            item = context_meta["inputs"]["market_index"]
            expected = item["sha256"] if item["path"] == relative else None
        else:
            meta = context_meta if name == "context_daily" else read_json(path.parent / "metadata.json")
            expected = meta["outputs"].get(path.name)
        _require_hash(path, expected, relative)
        frames[name] = _read_frame(path, columns, relative)
    _require_hash(safe_path(root, sources["index_daily"]), context_meta["source_sample_sha256"], sources["index_daily"])
    index_dates = set(frames["index_daily"].date)
    if (set(frames["context_daily"].date) != index_dates or not set(frames["company_daily"].date) <= index_dates
            or not index_dates <= set(frames["index_prices"].date)):
        raise DataError("指数、广度与公司派生表的日期范围不一致。")
    if set(frames["index_company_daily"].date) != index_dates:
        raise DataError("行业情绪与指数派生表的日期范围不一致。")
    availability = {name: _describe(frame, sources[name]) for name, frame in frames.items()}
    sentiment = frames["index_daily"].dropna(subset=["net_sentiment"])
    if sentiment.empty:
        raise DataError("指数样本没有可用的情绪统计。")
    availability["sentiment"] = _describe(sentiment, sources["index_daily"], list(SENTIMENT_METRICS))
    return AnalysisData(**frames, availability=availability, sources=sources)


def get_data_availability(data: AnalysisData) -> dict:
    return data.availability


def date_pair(start_date: str, end_date: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    try:
        start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
        if start.isoformat() != start_date or end.isoformat() != end_date or start > end:
            raise ValueError
        return pd.Timestamp(start), pd.Timestamp(end)
    except (ValueError, TypeError) as exc:
        raise ValueError("日期须为 YYYY-MM-DD，且开始日期不得晚于结束日期。") from exc


def validate_date_range(start_date: str, end_date: str, available_range: tuple) -> None:
    start, end = date_pair(start_date, end_date)
    low, high = map(pd.Timestamp, available_range)
    if end < low or start > high:
        raise ValueError("请求日期与数据可用区间没有交集。")


def restrict(frame: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    start, end = date_pair(start_date, end_date)
    return frame.loc[frame.date.between(start, end)].copy()


def period(frame: pd.DataFrame) -> list[str] | None:
    if frame.empty:
        return None
    return [frame.date.min().date().isoformat(), frame.date.max().date().isoformat()]


def number(value) -> float | None:
    return float(value) if pd.notna(value) and np.isfinite(value) else None


def records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", date_format="iso", date_unit="s", double_precision=15))


def coverage_warning(requested, available, label):
    lo, hi = available
    if requested[0] < lo or requested[1] > hi:
        return [f"{label}可用日期为 {lo} 至 {hi}。超出部分没有该数据。"]
    return []
