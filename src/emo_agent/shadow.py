"""Task 6 (Issue #16): frozen market-only models and new-date shadow predictions.

Plan: docs/records/2026-11-25-shadow-plan.md, configs/shadow_v1.json.
Features for a date use only closes at or before 15:00 that day (16:00 cutoff).
"""
from datetime import datetime, timezone, timedelta
from pathlib import Path
import importlib.metadata
import json
import joblib
import numpy as np
import pandas as pd
from emo_agent.audit import safe_path
from emo_agent.baseline import fit_model, score
from emo_agent.context import market_features, breadth_features, calibration, INPUTS as CONTEXT_INPUTS
from emo_agent.dataset import MARKET, sha256, git_revision
from emo_agent.labels import future_labels
from emo_agent.stability import bootstrap_indices

TZ = timezone(timedelta(hours=8))
INDEX_FILE = "data/input/market/index_931723_full.csv"
OUT = "outputs/shadow_v1"


def model_features(model, config):
    if model not in config["models"]:
        raise ValueError("unknown model")
    cols = list(MARKET)
    if "G" in model: cols += config["groups"]["G"]
    if "B" in model: cols += config["groups"]["B"]
    return cols


def build_frame(root: Path, config: dict) -> pd.DataFrame:
    """All dates with market, context features and labels (labels may be immature)."""
    prices = pd.read_csv(safe_path(root, INDEX_FILE)).sort_values("date").reset_index(drop=True)
    p = prices.copy()
    r = p.close.pct_change(fill_method=None)
    for n in [1, 5, 20]:
        p[f"return_{n}"] = p.close.pct_change(n, fill_method=None)
    for n in [5, 20]:
        p[f"volatility_{n}"] = r.rolling(n, min_periods=n).std(ddof=1)
    p["amount_relative_20"] = p.amount_100m / p.amount_100m.shift(1).rolling(20, min_periods=20).mean().replace(0, np.nan) - 1
    idx = pd.read_csv(safe_path(root, CONTEXT_INPUTS["market_index"]), dtype={"date": str})
    rets = pd.read_csv(safe_path(root, CONTEXT_INPUTS["member_returns"]), dtype={"stkcd": str, "date": str})
    mem = pd.read_csv(safe_path(root, CONTEXT_INPUTS["members"]), dtype=str, encoding="utf-8-sig")[["stkcd", "in_date", "out_date"]]
    p = p.merge(market_features(idx), on="date", how="left").merge(breadth_features(rets, mem), on="date", how="left")
    p["relative_return_5"] = p.return_5 - p.market_return_5
    for h in config["horizons"]:
        lab = future_labels(prices, h)
        p[f"target_{h}"] = lab.target.values
        p[f"label_end_date_{h}"] = lab.label_end_date.dt.strftime("%Y-%m-%d").values
    p = p[p.date >= config["freeze"]["first_sample_date"]].reset_index(drop=True)
    p[MARKET] = p[MARKET].replace([np.inf, -np.inf], np.nan)
    p["market_complete"] = p[MARKET].notna().all(axis=1)
    return p


def _mature(frame, h, start, end):
    d = frame[(frame.date > start) & (frame.date <= end) & frame.market_complete & frame[f"target_{h}"].notna()]
    return d[d[f"label_end_date_{h}"] <= end]


def freeze_models(root: Path) -> dict:
    root = Path(root)
    config = json.loads((root / "configs/shadow_v1.json").read_text(encoding="utf-8"))
    out = root / OUT
    if (out / "freeze.json").exists():
        raise ValueError("models already frozen; never re-freeze over an existing run")
    frame = build_frame(root, config)
    f = config["freeze"]
    (out / "models").mkdir(parents=True, exist_ok=True)
    record = {"frozen_at": datetime.now(TZ).isoformat(timespec="seconds"), "code_commit": git_revision(root),
              "config_sha256": sha256(root / "configs/shadow_v1.json"),
              "inputs": {k: sha256(safe_path(root, v)) for k, v in {"index_931723": INDEX_FILE, **CONTEXT_INPUTS}.items()},
              "models": {}}
    for h in config["horizons"]:
        train = _mature(frame, h, "0000", f["train_end"])
        valid = _mature(frame, h, f["train_end"], f["validation_end"])
        refit = _mature(frame, h, "0000", f["refit_end"])
        record["models"][f"prior_{h}d"] = {"probability": float(refit[f"target_{h}"].mean()), "n_refit": len(refit)}
        for m in config["models"]:
            feats = model_features(m, config)
            trials = []
            for C in config["C_grid"]:
                prob = fit_model(train, feats, h, C, config["random_seed"]).predict_proba(valid[feats])[:, 1]
                trials.append((score(valid[f"target_{h}"], prob, config["threshold"])["brier"], C))
            brier, C = min(trials)
            model = fit_model(refit, feats, h, C, config["random_seed"])
            path = out / "models" / f"{m}_{h}d.joblib"
            joblib.dump(model, path)
            record["models"][f"{m}_{h}d"] = {"C": C, "validation_brier": brier, "features": feats, "n_train": len(train),
                                            "n_validation": len(valid), "n_refit": len(refit), "sha256": sha256(path)}
    (out / "freeze.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {k: {"C": v.get("C"), "n_refit": v["n_refit"]} for k, v in record["models"].items()}


def _load_frozen(root: Path):
    out = root / OUT
    record = json.loads((out / "freeze.json").read_text(encoding="utf-8"))
    models = {}
    for key, info in record["models"].items():
        if key.startswith("prior"):
            continue
        path = out / "models" / f"{key}.joblib"
        if sha256(path) != info["sha256"]:
            raise ValueError(f"frozen model changed: {key}")
        models[key] = (joblib.load(path), info["features"])
    return record, models


def shadow_predict(root: Path, date: str, mode: str = "daily") -> list:
    """Append one row per model/horizon for `date`. Refuses dates inside the training period or already logged."""
    root = Path(root)
    config = json.loads((root / "configs/shadow_v1.json").read_text(encoding="utf-8"))
    record, models = _load_frozen(root)
    if date <= config["freeze"]["refit_end"]:
        raise ValueError("date is inside the frozen training period")
    log_path = root / OUT / "shadow_log.csv"
    if log_path.exists() and date in set(pd.read_csv(log_path, dtype={"date": str}).date):
        raise ValueError("date already predicted; the log is append-only")
    frame = build_frame(root, config)
    row = frame[frame.date == date]
    if row.empty:
        raise ValueError("no 931723 close for this date yet; update data/input first")
    stamp = datetime.now(TZ).isoformat(timespec="seconds")
    inputs = {k: sha256(safe_path(root, v)) for k, v in {"index_931723": INDEX_FILE, **CONTEXT_INPUTS}.items()}
    rows = []
    for h in config["horizons"]:
        rows.append({"date": date, "horizon": h, "model": "prior", "probability_up": record["models"][f"prior_{h}d"]["probability"],
                     "missing_features": ""})
        for m in config["models"]:
            model, feats = models[f"{m}_{h}d"]
            missing = [c for c in feats if pd.isna(row.iloc[0][c])]
            prob = np.nan if missing else float(model.predict_proba(row[feats])[:, 1][0])  # never impute a live date
            rows.append({"date": date, "horizon": h, "model": m, "probability_up": prob, "missing_features": ";".join(missing)})
    log = pd.DataFrame(rows)
    log["generated_at"], log["mode"] = stamp, mode
    log["model_sha256"] = [record["models"].get(f"{m}_{h}d", {}).get("sha256", "") for m, h in zip(log.model, log.horizon)]
    log["inputs_sha256"] = json.dumps(inputs, sort_keys=True)
    log["code_commit"] = git_revision(root)
    log.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
    return log[["date", "horizon", "model", "probability_up", "missing_features"]].to_dict("records")


def shadow_evaluate(root: Path, start: str = None, end: str = None) -> dict:
    root = Path(root)
    config = json.loads((root / "configs/shadow_v1.json").read_text(encoding="utf-8"))
    start, end = start or config["evaluation"]["start"], end or config["evaluation"]["end"]
    log = pd.read_csv(root / OUT / "shadow_log.csv", dtype={"date": str})
    frame = build_frame(root, config)
    last_close = frame.date.max()
    results, comps, calib = [], [], []
    for h in config["horizons"]:
        lab = frame[["date", f"target_{h}", f"label_end_date_{h}"]].rename(columns={f"target_{h}": "actual", f"label_end_date_{h}": "label_end"})
        part = log[(log.horizon == h) & log.date.between(start, end)].merge(lab, on="date")
        part = part[part.actual.notna() & (part.label_end <= last_close)]
        table = part.pivot_table(index="date", columns="model", values="probability_up", aggfunc="first").dropna()
        y = part.groupby("date").actual.first().reindex(table.index).to_numpy(int)
        for m in table.columns:
            results.append({"horizon": h, "model": m, **score(y, table[m].to_numpy(), config["threshold"])})
            calib.append({"horizon": h, "model": m, **calibration(y, table[m].to_numpy(), config["calibration_bins"])})
        idx = bootstrap_indices(np.zeros(len(y)), config["bootstrap_repetitions"], config["bootstrap_block_length"], config["random_seed"])
        for a, b, label in config["comparisons"]:
            d = (table[b].to_numpy() - y) ** 2 - (table[a].to_numpy() - y) ** 2
            draws = d[idx].mean(axis=1)
            comps.append({"horizon": h, "comparison": label, "n": len(y), "brier_delta": float(d.mean()),
                          "ci95_low": float(np.quantile(draws, 0.025)), "ci95_high": float(np.quantile(draws, 0.975))})
    out = root / OUT
    pd.DataFrame(results).to_csv(out / "evaluation_metrics.csv", index=False)
    pd.DataFrame(comps).to_csv(out / "evaluation_comparisons.csv", index=False)
    pd.DataFrame(calib).to_csv(out / "evaluation_calibration.csv", index=False)
    meta = {"evaluated_at": datetime.now(TZ).isoformat(timespec="seconds"), "window": [start, end], "last_close_in_data": last_close,
            "dependencies": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "joblib"]}}
    (out / "evaluation.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"metrics": results, "comparisons": comps}
