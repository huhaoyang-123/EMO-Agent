"""Extract CNRDS GUBA aggregates for the historical members of CSI 931723.

The four source ZIPs are read only. A row belongs to the index universe when
its source statistical Date is within the member's [in_date, out_date) interval.
The resulting date is still a provider label, not a verified publication time.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import io
import json
import re
import zipfile

import numpy as np
import pandas as pd
from lxml import etree as ET


# Paths are relative to the repository; migrated snapshots remain read only.
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "guba"
OUT = ROOT / "data" / "derived" / "sentiment_rebuilt"
MEMBERS = ROOT / "data" / "input" / "market" / "index_931723_members.csv"
CALENDAR = ROOT / "data" / "input" / "market" / "trading_calendar.csv"
FIELDS = [
    "Scode", "Coname", "Date", "Time", "Weektime", "Tradetm_dum",
    "Tpostnum", "Pospostnum", "Negpostnum", "Neupostnum", "Readnum", "Commentnum",
]
COUNT_FIELDS = ["Tpostnum", "Pospostnum", "Negpostnum", "Neupostnum", "Readnum", "Commentnum"]
ROW = re.compile(rb"<row\b.*?</row>", re.S)
FIRST = re.compile(rb'<c\b[^>]*\br="A\d+"[^>]*>.*?</c>', re.S)
VAL = re.compile(rb"<v>(.*?)</v>", re.S)
RNUM = re.compile(rb'<row\b[^>]*\br="(\d+)"')


def stream_rows(stream):
    tail = b""
    while True:
        block = stream.read(4 * 1024 * 1024)
        if not block:
            if b"<row " in tail:
                raise ValueError("XLSX has an unterminated row")
            break
        data = tail + block
        end = data.rfind(b"</row>")
        if end < 0:
            tail = data
            continue
        end += 6
        yield from ROW.finditer(data[:end])
        tail = data[end:]


def decode_row(raw, strings):
    row = ET.fromstring(raw)
    values = [None] * len(FIELDS)
    for cell in row:
        ref = cell.get("r")
        letter = re.match(r"[A-Z]+", ref)[0]
        if len(letter) != 1 or letter > "L":
            raise ValueError(f"Unexpected source column: {ref}")
        value = cell.find("v")
        if cell.get("t") == "s":
            decoded = strings[int(value.text)] if value is not None else None
        elif cell.get("t") == "inlineStr":
            decoded = "".join(cell.itertext())
        else:
            decoded = value.text if value is not None else None
        values[ord(letter) - 65] = decoded
    return values


def zip_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_slots(date):
    weekday = date.weekday()
    if weekday >= 5:
        return ["0:00-24:00"]
    if weekday == 0:
        return ["0:00-9:00", "9:00-15:00"]
    if weekday == 4:
        return ["15:00-9:00", "9:00-15:00", "15:00-24:00"]
    return ["15:00-9:00", "9:00-15:00"]


def add_sentiment(frame):
    total = frame.Tpostnum.replace(0, np.nan)
    directional = (frame.Pospostnum + frame.Negpostnum).replace(0, np.nan)
    frame["net_sentiment"] = (frame.Pospostnum - frame.Negpostnum) / total
    frame["directional_sentiment"] = (frame.Pospostnum - frame.Negpostnum) / directional
    return frame


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    members = pd.read_csv(MEMBERS, dtype={"stkcd": str, "in_date": str, "out_date": str})
    assert not members.duplicated(["stkcd", "in_date"]).any()
    codes = set(members.stkcd)
    selected = []
    manifest = []

    for archive in sorted(SOURCE.glob("*.zip")):
        archive_sha256 = zip_hash(archive)
        with zipfile.ZipFile(archive) as outer:
            for info in outer.infolist():
                if not info.filename.endswith(".xlsx"):
                    continue
                source_rows = selected_rows = 0
                with zipfile.ZipFile(io.BytesIO(outer.read(info))) as book:
                    shared = ET.fromstring(book.read("xl/sharedStrings.xml"))
                    strings = ["".join(si.itertext()) for si in shared]
                    del shared
                    code_ids = {str(i).encode() for i, value in enumerate(strings) if value in codes}
                    with book.open("xl/worksheets/sheet1.xml") as stream:
                        for match in stream_rows(stream):
                            raw = match.group()
                            row_num = int(RNUM.search(raw)[1])
                            if row_num <= 2:
                                if row_num == 1 and decode_row(raw, strings) != FIELDS:
                                    raise ValueError(f"Changed source schema: {archive.name} / {info.filename}")
                                continue
                            source_rows += 1
                            cell = FIRST.search(raw)
                            if cell is None:
                                raise ValueError("Source row lacks a security code")
                            value = VAL.search(cell.group())
                            if value is None:
                                raise ValueError("Source row has an empty security code")
                            if b't="s"' in cell.group():
                                keep = value[1] in code_ids
                            else:
                                keep = value[1].decode().zfill(6) in codes
                            if keep:
                                values = decode_row(raw, strings)
                                if values[0] not in codes:
                                    raise ValueError("Security code failed a decode check")
                                selected.append(values + [archive.name, info.filename, row_num])
                                selected_rows += 1
                manifest.append({
                    "source_zip": archive.name,
                    "zip_sha256": archive_sha256,
                    "source_xlsx": info.filename,
                    "source_rows": source_rows,
                    "historical_member_code_rows": selected_rows,
                })
                print(json.dumps(manifest[-1], ensure_ascii=False), flush=True)

    raw = pd.DataFrame(selected, columns=FIELDS + ["source_zip", "source_xlsx", "source_excel_row"])
    assert raw["Scode"].nunique() == len(codes)
    assert not raw[FIELDS].isna().any().any()
    assert not raw.duplicated(["Scode", "Date", "Time"]).any()
    for field in COUNT_FIELDS:
        raw[field] = pd.to_numeric(raw[field], errors="raise").astype("int64")
        assert (raw[field] >= 0).all()
    assert (raw.Tpostnum == raw.Pospostnum + raw.Negpostnum + raw.Neupostnum).all()
    raw["source_record_id"] = (
        raw.source_zip + "::" + raw.source_xlsx + "::" + raw.source_excel_row.astype(str)
    )
    assert raw.source_record_id.is_unique

    detail = raw.merge(members, left_on="Scode", right_on="stkcd", validate="many_to_many")
    detail = detail[(detail.Date >= detail.in_date) & (detail.out_date.isna() | (detail.Date < detail.out_date))].copy()
    assert detail.source_record_id.is_unique
    assert detail.Scode.nunique() == len(codes)
    detail = detail.drop(columns="stkcd")
    detail.insert(0, "index_code", "931723")
    detail["membership_date_basis"] = "source_statistical_Date"
    date = pd.to_datetime(detail.Date)
    detail["interval_start"] = date
    detail["interval_end"] = date
    for slot, start_day, start_hour, end_day, end_hour in [
        ("0:00-24:00", 0, 0, 1, 0),
        ("0:00-9:00", 0, 0, 0, 9),
        ("15:00-9:00", -1, 15, 0, 9),
        ("9:00-15:00", 0, 9, 0, 15),
        ("15:00-24:00", 0, 15, 1, 0),
    ]:
        mask = detail.Time.eq(slot)
        detail.loc[mask, "interval_start"] = date[mask] + pd.Timedelta(days=start_day, hours=start_hour)
        detail.loc[mask, "interval_end"] = date[mask] + pd.Timedelta(days=end_day, hours=end_hour)
    detail["interval_start"] = detail.interval_start.dt.strftime("%Y-%m-%d %H:%M:%S")
    detail["interval_end"] = detail.interval_end.dt.strftime("%Y-%m-%d %H:%M:%S")
    detail["interval_basis"] = "按源说明推定；非逐帖时间戳"
    detail["count_snapshot_verified"] = 0
    add_sentiment(detail)

    calendar = pd.read_csv(CALENDAR, dtype={"date": str})
    trade_dates = set(calendar.date)
    detail["is_trade_date"] = detail.Date.isin(trade_dates).astype(int)
    weekday_names = {0: "星期一", 1: "星期二", 2: "星期三", 3: "星期四", 4: "星期五", 5: "星期六", 6: "星期日"}
    assert (date.dt.weekday.map(weekday_names) == detail.Weektime).all()
    assert ((detail.Time == "9:00-15:00").astype(int) == detail.Tradetm_dum.astype(int)).all()
    assert (pd.to_datetime(detail.interval_end) > pd.to_datetime(detail.interval_start)).all()
    detail = detail.sort_values(["Date", "Scode", "interval_start"])

    start_date = members.in_date.min()
    end_date = min(raw.Date.max(), "2026-06-30")
    all_dates = pd.date_range(start_date, end_date)
    expected_rows = []
    for day in all_dates:
        label = day.strftime("%Y-%m-%d")
        active = members[(members.in_date <= label) & (members.out_date.isna() | (members.out_date > label))]
        for code in active.stkcd:
            expected_rows.extend((code, label, slot) for slot in expected_slots(day))
    expected = pd.DataFrame(expected_rows, columns=["Scode", "Date", "Time"])
    assert not expected.duplicated(["Scode", "Date", "Time"]).any()
    missing = expected.merge(detail[["Scode", "Date", "Time"]], on=["Scode", "Date", "Time"], how="left", indicator=True)
    missing = missing[missing._merge == "left_only"].drop(columns="_merge")
    missing["reason"] = "源文件未见记录；零发帖或未覆盖原因未确认"
    missing["is_trade_date"] = missing.Date.isin(trade_dates).astype(int)

    company = detail.groupby(["Date", "Scode"], sort=True)[COUNT_FIELDS].sum().reset_index()
    company["observed_slots"] = detail.groupby(["Date", "Scode"]).size().values
    company["expected_slots"] = pd.to_datetime(company.Date).map(lambda d: len(expected_slots(d)))
    company["complete_slots"] = (company.observed_slots == company.expected_slots).astype(int)
    add_sentiment(company)
    industry = company.groupby("Date", sort=True)[COUNT_FIELDS].sum()
    group = company.groupby("Date", sort=True)
    industry["observed_companies"] = group.Scode.nunique()
    industry["complete_companies"] = group.complete_slots.sum()
    industry["observed_slots"] = group.observed_slots.sum()
    industry["equal_weight_sentiment"] = group.net_sentiment.mean()
    industry["max_company_post_share"] = group.Tpostnum.max() / industry.Tpostnum
    industry = industry.reindex([d.strftime("%Y-%m-%d") for d in all_dates])
    expected_by_day = expected.groupby("Date").agg(expected_companies=("Scode", "nunique"), expected_slots=("Time", "size"))
    industry = industry.join(expected_by_day)
    industry["company_coverage"] = industry.observed_companies / industry.expected_companies
    industry["slot_coverage"] = industry.observed_slots / industry.expected_slots
    industry["complete_pool_date"] = (industry.observed_slots == industry.expected_slots).astype(int)
    industry["is_trade_date"] = industry.index.isin(trade_dates).astype(int)
    add_sentiment(industry)
    industry = industry.reset_index(names="Date")

    sg = detail.groupby(["Date", "Time", "interval_start", "interval_end"], sort=True)
    slots = sg[COUNT_FIELDS].sum()
    slots["observed_companies"] = sg.Scode.nunique()
    slots["equal_weight_sentiment"] = sg.net_sentiment.mean()
    slots = slots.reset_index()
    slots["expected_companies"] = slots.Date.map(expected_by_day.expected_companies)
    slots["company_coverage"] = slots.observed_companies / slots.expected_companies
    slots["is_trade_date"] = slots.Date.isin(trade_dates).astype(int)
    add_sentiment(slots)

    coverage = members.groupby("stkcd").agg(first_member_date=("in_date", "min"), member_intervals=("in_date", "size")).reset_index().rename(columns={"stkcd": "Scode"})
    observed = detail.groupby("Scode").agg(observed_rows=("Date", "size"), observed_dates=("Date", "nunique"), first_observed_date=("Date", "min"), last_observed_date=("Date", "max"), posts=("Tpostnum", "sum"))
    exp = expected.groupby("Scode").agg(expected_slots=("Time", "size"), expected_dates=("Date", "nunique"))
    coverage = coverage.join(observed, on="Scode").join(exp, on="Scode")
    coverage["missing_slots"] = coverage.expected_slots - coverage.observed_rows
    coverage["slot_coverage"] = coverage.observed_rows / coverage.expected_slots

    outputs = {
        "931723成分股分时明细.csv": detail,
        "931723行业统计日.csv": industry,
        "931723行业分时.csv": slots,
        "931723成分股覆盖.csv": coverage,
        "931723缺失时段.csv": missing,
    }
    for filename, frame in outputs.items():
        frame.to_csv(OUT / filename, index=False, encoding="utf-8-sig", float_format="%.10g")
    pd.DataFrame(manifest).to_csv(OUT / "931723源文件清单.csv", index=False, encoding="utf-8-sig")

    trade = industry[industry.is_trade_date == 1]
    summary = {
        "index_code": "931723",
        "source_xlsx_count": len(manifest),
        "source_rows": int(sum(item["source_rows"] for item in manifest)),
        "all_historical_code_rows": len(raw),
        "historical_member_codes": len(codes),
        "dynamic_member_rows": len(detail),
        "dynamic_member_posts": int(detail.Tpostnum.sum()),
        "date_first": start_date,
        "date_last": end_date,
        "trade_dates": len(trade),
        "all_member_present_trade_dates": int((trade.observed_companies == trade.expected_companies).sum()),
        "all_slots_complete_trade_dates": int(trade.complete_pool_date.sum()),
        "missing_expected_slots": len(missing),
        "missing_expected_company_dates": int((industry.expected_companies - industry.observed_companies).sum()),
        "duplicate_keys": int(detail.duplicated(["Scode", "Date", "Time"]).sum()),
        "count_balance_errors": int((detail.Tpostnum != detail.Pospostnum + detail.Negpostnum + detail.Neupostnum).sum()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
