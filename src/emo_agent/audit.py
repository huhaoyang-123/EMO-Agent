"""Read-only snapshot integrity and basic semantic checks."""
from pathlib import Path
import hashlib
import json
import pandas as pd


def safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError("manifest path escapes project root")
    return path


def verify_manifest(root: Path, manifest: dict) -> list[dict]:
    results = []
    for item in manifest["files"]:
        path = safe_path(root, item["path"])
        state = "missing"
        if path.is_file():
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            state = "ok" if digest == item["sha256"] and path.stat().st_size == item["bytes"] else "hash_mismatch"
        results.append({"path": item["path"], "status": state})
    return results


def validate_sentiment(frame: pd.DataFrame) -> None:
    if frame.duplicated(["Scode", "Date", "Time"]).any():
        raise ValueError("duplicate sentiment key")
    if not frame["Scode"].astype(str).str.fullmatch(r"[0-9]{6}").all():
        raise ValueError("Scode must be six-digit text")
    counts = frame[["Tpostnum", "Pospostnum", "Negpostnum", "Neupostnum"]].apply(pd.to_numeric, errors="raise")
    if counts.isna().any().any() or (counts < 0).any().any() or (counts % 1 != 0).any().any():
        raise ValueError("invalid post count")
    if not counts.Tpostnum.eq(counts.Pospostnum + counts.Negpostnum + counts.Neupostnum).all():
        raise ValueError("sentiment counts do not reconcile")


def audit_project(root: Path) -> dict:
    from emo_agent.labels import future_labels
    manifest = json.loads((root / "docs/data_manifest.json").read_text(encoding="utf-8-sig"))
    files = verify_manifest(root, manifest)
    report = {"ok": all(x["status"] == "ok" for x in files), "files": files, "checks": {}}
    if not report["ok"]:
        return report
    try:
        prices = pd.read_csv(root / "data/input/market/index_931723_full.csv")
        future_labels(prices, 1)
        folder = root / "data/input/sentiment"
        detail = pd.read_csv(folder / "931723成分股分时明细.csv", dtype={"Scode": str})
        validate_sentiment(detail)
        daily = pd.read_csv(folder / "931723行业统计日.csv")
        if daily.Date.duplicated().any():
            raise ValueError("duplicate daily sentiment date")
        observed = detail.groupby("Date").Tpostnum.sum().sort_index()
        exported = daily.set_index("Date").Tpostnum.sort_index()
        if not observed.index.equals(exported.index) or not observed.eq(exported).all():
            raise ValueError("daily totals do not reconcile")
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        if len(detail) != summary["dynamic_member_rows"] or int(detail.Tpostnum.sum()) != summary["dynamic_member_posts"]:
            raise ValueError("summary does not reconcile")
        report["checks"] = {"price_rows": len(prices), "sentiment_rows": len(detail), "posts": int(detail.Tpostnum.sum()), "daily_reconciliation": "passed"}
        if any(x["path"] == "data/input/market/stock_daily.csv" for x in manifest["files"]):
            from emo_agent.company import validate_stocks
            stocks = pd.read_csv(root / "data/input/market/stock_daily.csv", dtype={"Stkcd": str})
            validate_stocks(stocks)
            report["checks"].update(stock_rows=len(stocks), stock_companies=int(stocks.Stkcd.nunique()))
    except (ValueError, KeyError) as exc:
        report["ok"] = False
        report["error"] = str(exc)
    return report
