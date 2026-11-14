"""Build cutoff-safe exploratory samples from immutable input snapshots."""
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
import pandas as pd
from emo_agent.audit import audit_project, validate_sentiment
from emo_agent.labels import future_labels

TZ = "Asia/Shanghai"
KEY = ["Scode", "Date", "Time"]
COUNTS = ["Tpostnum", "Pospostnum", "Negpostnum", "Neupostnum"]
MARKET = ["return_1", "return_5", "return_20", "volatility_5", "volatility_20", "amount_relative_20"]
SENTIMENT = ["net_sentiment", "equal_weight_sentiment", "sentiment_change_5", "post_heat_20", "company_coverage", "slot_coverage", "sentiment_missing"]
SLOTS = {"0:00-24:00": (0, 0, 1, 0), "0:00-9:00": (0, 0, 0, 9), "15:00-9:00": (-1, 15, 0, 9), "9:00-15:00": (0, 9, 0, 15), "15:00-24:00": (0, 15, 1, 0)}


def slot_bounds(dates, slots):
    naive = pd.to_datetime(dates)
    dates = naive.dt.tz_localize(TZ)
    start, end = dates.copy(), dates.copy()
    if not slots.isin(SLOTS).all():
        raise ValueError("unknown provider time slot")
    for slot, (sd, sh, ed, eh) in SLOTS.items():
        mask = slots.eq(slot)
        start.loc[mask] = pd.to_datetime(naive.loc[mask].astype("int64") + (sd * 24 + sh) * 3_600_000_000_000).dt.tz_localize(TZ)
        end.loc[mask] = pd.to_datetime(naive.loc[mask].astype("int64") + (ed * 24 + eh) * 3_600_000_000_000).dt.tz_localize(TZ)
    return start, end


def assign_cutoffs(ends, cutoffs):
    """Each interval belongs to its first cutoff >= end; out-of-range -> NaT."""
    ends = pd.DatetimeIndex(ends)
    cutoffs = pd.DatetimeIndex(cutoffs)
    if not len(cutoffs) or cutoffs.tz is None or ends.tz is None:
        raise ValueError("timezone-aware nonempty cutoffs required")
    if cutoffs.has_duplicates or not cutoffs.is_monotonic_increasing or ends.hasnans:
        raise ValueError("invalid cutoff or interval end")
    positions = cutoffs.searchsorted(ends, side="left")
    values = [cutoffs[i] if i < len(cutoffs) else pd.NaT for i in positions]
    return pd.Series(values, dtype=f"datetime64[ns, {TZ}]")


def expected_grid(members, start, end):
    rows = []
    for day in pd.date_range(start, end):
        date = day.strftime("%Y-%m-%d")
        active = members[(members.in_date <= date) & (members.out_date.isna() | (members.out_date > date))]
        w = day.weekday()
        slots = ["0:00-24:00"] if w >= 5 else (["0:00-9:00", "9:00-15:00"] if w == 0 else (["15:00-9:00", "9:00-15:00", "15:00-24:00"] if w == 4 else ["15:00-9:00", "9:00-15:00"]))
        rows.extend((code, date, slot) for code in active.stkcd for slot in slots)
    result = pd.DataFrame(rows, columns=KEY)
    if result.duplicated(KEY).any():
        raise ValueError("overlapping membership periods")
    result["interval_start"], result["interval_end"] = slot_bounds(result.Date, result.Time)
    return result


def align_sentiment(detail, expected, cutoffs):
    validate_sentiment(detail)
    if not detail.source_record_id.is_unique:
        raise ValueError("duplicate source id")
    extra = detail[KEY].merge(expected[KEY], on=KEY, how="left", indicator=True)
    if extra._merge.ne("both").any():
        raise ValueError("source row outside historical membership/expected slots")
    frame = expected.merge(detail[KEY + COUNTS + ["source_record_id"]], on=KEY, how="left", validate="one_to_one")
    frame["cutoff_at"] = assign_cutoffs(frame.interval_end, cutoffs).array
    frame["observed"] = frame.source_record_id.notna()
    frame["assignment_status"] = np.where(frame.cutoff_at.isna(), "after_final_cutoff", "assigned")
    valid = frame[frame.cutoff_at.notna()].copy()
    if not (valid.interval_end <= valid.cutoff_at).all():
        raise ValueError("future interval leaked")
    observed = valid[valid.observed]
    totals = observed.groupby("cutoff_at")[COUNTS].sum(min_count=1)
    daily = pd.DataFrame(index=pd.DatetimeIndex(cutoffs, name="cutoff_at"))
    daily = daily.join(totals)
    daily["expected_slots"] = valid.groupby("cutoff_at").size().reindex(daily.index, fill_value=0)
    daily["observed_slots"] = observed.groupby("cutoff_at").size().reindex(daily.index, fill_value=0)
    daily["expected_companies"] = valid.groupby("cutoff_at").Scode.nunique().reindex(daily.index, fill_value=0)
    daily["observed_companies"] = observed.groupby("cutoff_at").Scode.nunique().reindex(daily.index, fill_value=0)
    daily["slot_coverage"] = daily.observed_slots / daily.expected_slots.replace(0, np.nan)
    daily["company_coverage"] = daily.observed_companies / daily.expected_companies.replace(0, np.nan)
    daily["net_sentiment"] = (daily.Pospostnum - daily.Negpostnum) / daily.Tpostnum.replace(0, np.nan)
    company = observed.groupby(["cutoff_at", "Scode"])[COUNTS].sum()
    company["sentiment"] = (company.Pospostnum - company.Negpostnum) / company.Tpostnum.replace(0, np.nan)
    daily["equal_weight_sentiment"] = company.groupby("cutoff_at").sentiment.mean()
    daily["sentiment_missing"] = daily.net_sentiment.isna().astype(int)
    daily["max_interval_end"] = observed.groupby("cutoff_at").interval_end.max()
    daily["window_first_interval_start"] = valid.groupby("cutoff_at").interval_start.min()
    return daily.reset_index(), frame


def add_features(prices, sentiment):
    result = prices.copy()
    returns = result.close.pct_change(fill_method=None)
    for n in [1, 5, 20]:
        result[f"return_{n}"] = result.close.pct_change(n, fill_method=None)
    for n in [5, 20]:
        result[f"volatility_{n}"] = returns.rolling(n, min_periods=n).std(ddof=1)
    result["amount_relative_20"] = result.amount_100m / result.amount_100m.shift(1).rolling(20, min_periods=20).mean().replace(0, np.nan) - 1
    result["cutoff_at"] = pd.to_datetime(result.date + " 16:00:00").dt.tz_localize(TZ)
    result = result.merge(sentiment, on="cutoff_at", validate="one_to_one")
    result["sentiment_change_5"] = result.net_sentiment - result.net_sentiment.shift(1).rolling(5, min_periods=5).mean()
    intensity = result.Tpostnum / result.observed_slots.replace(0, np.nan)
    result["post_heat_20"] = np.log1p(intensity) - np.log1p(intensity.shift(1).rolling(20, min_periods=20).mean())
    result[MARKET + SENTIMENT] = result[MARKET + SENTIMENT].replace([np.inf, -np.inf], np.nan)
    result["eligible"] = result[MARKET].notna().all(axis=1)
    result["exclusion_reason"] = np.where(result.eligible, "", "market_warmup_or_missing")
    result["period"] = np.select([result.date <= "2024-12-31", result.date <= "2025-12-31"], ["train", "validation"], default="test")
    result["availability_assumption"] = "interval_end_only_not_verified_publication"
    return result


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def git_revision(root):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def build_dataset(root: Path):
    audit = audit_project(root)
    if not audit["ok"]:
        raise ValueError("input audit failed; run audit-data")
    market = root / "data/input/market"
    folder = root / "data/input/sentiment"
    prices_all = pd.read_csv(market / "index_931723_full.csv")
    members = pd.read_csv(market / "index_931723_members.csv", dtype={"stkcd": str})
    detail = pd.read_csv(folder / "931723成分股分时明细.csv", dtype={"Scode": str})
    start, end = detail.Date.min(), detail.Date.max()
    if start < "2023-10-31":
        raise ValueError("pre-release sentiment is not supported")
    prices = prices_all[prices_all.date.between(start, end)].copy()
    calendar = pd.read_csv(market / "trading_calendar.csv")
    reference = set(calendar.loc[calendar.date.between(start, end), "date"])
    if set(prices.date) != reference:
        raise ValueError("price/calendar mismatch in modelling range")
    cutoffs = pd.DatetimeIndex(pd.to_datetime(prices.date + " 16:00:00")).tz_localize(TZ)
    expected = expected_grid(members, start, end)
    # Ensure computed source intervals agree with the immutable snapshot.
    lo, hi = slot_bounds(detail.Date, detail.Time)
    if not lo.dt.tz_localize(None).eq(pd.to_datetime(detail.interval_start)).all() or not hi.dt.tz_localize(None).eq(pd.to_datetime(detail.interval_end)).all():
        raise ValueError("source interval reconstruction mismatch")
    aligned, mapping = align_sentiment(detail, expected, cutoffs)
    source_missing = pd.read_csv(folder / "931723缺失时段.csv", dtype={"Scode": str})
    if set(map(tuple, mapping.loc[~mapping.observed, KEY].values)) != set(map(tuple, source_missing[KEY].values)):
        raise ValueError("missing-slot snapshot mismatch")
    sample = add_features(prices, aligned)
    for h in [1, 5]:
        labels = future_labels(prices_all, h)
        labels["date"] = labels.date.dt.strftime("%Y-%m-%d")
        labels = labels.rename(columns={"target": f"target_{h}", "label_end_date": f"label_end_date_{h}"})
        sample = sample.merge(labels, on="date", validate="one_to_one")
        sample[f"return_forward_{h}"] = sample.date.map(pd.Series(prices_all.close.shift(-h).values / prices_all.close.values - 1, index=prices_all.date))
    destination = root / "data/derived/cutoff_v1"
    destination.mkdir(parents=True, exist_ok=True)
    sample.to_csv(destination / "daily_samples.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    mapping.to_csv(destination / "interval_assignments.csv", index=False, encoding="utf-8-sig")
    removed = sample[~sample.eligible][["date", "exclusion_reason"]]
    removed.to_csv(destination / "excluded_dates.csv", index=False, encoding="utf-8-sig")
    checks = {
        "input_rows": len(detail), "assigned_observed_rows": int((mapping.observed & mapping.cutoff_at.notna()).sum()),
        "unassigned_observed_rows": int((mapping.observed & mapping.cutoff_at.isna()).sum()),
        "missing_expected_slots": int((~mapping.observed).sum()),
        "future_interval_violations": 0,
        "source_posts": int(detail.Tpostnum.sum()), "assigned_posts": int(aligned.Tpostnum.sum()),
        "prediction_dates": len(sample), "eligible_dates": int(sample.eligible.sum()), "excluded_dates": len(removed),
        "eligible_by_period": sample[sample.eligible].groupby("period").size().to_dict(),
        "min_slot_coverage": float(aligned.slot_coverage.min()),
        "zero_observation_windows": int(aligned.observed_slots.eq(0).sum()),
        "end_date": end, "start_date": start,
    }
    manifest = json.loads((root / "docs/data_manifest.json").read_text(encoding="utf-8-sig"))
    metadata = {"schema_version": 1, "code_commit": git_revision(root), "cutoff": "16:00 Asia/Shanghai", "market_features": MARKET, "sentiment_features": SENTIMENT, "checks": checks, "inputs": manifest["files"], "outputs": {p.name: sha256(p) for p in destination.glob("*.csv")}}
    (destination / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return checks
