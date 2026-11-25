"""Task 3 (Issue #14): broad-market context (G) and constituent breadth (B).

Frozen plan: docs/records/2026-11-25-context-plan.md, configs/context_v1.json.
Every new feature uses only closes at or before 15:00 on the prediction date,
i.e. before the 16:00 cutoff of cutoff_v1. 2025/2026 are already-seen periods,
so results are development diagnostics, not a new holdout.
"""
from pathlib import Path
import importlib.metadata
import json
import numpy as np
import pandas as pd
from emo_agent.audit import safe_path
from emo_agent.baseline import score
from emo_agent.dataset import MARKET, SENTIMENT, sha256, git_revision
from emo_agent.stability import fold_parts, choose_c, fit_predict_save, bootstrap_indices

CONTEXT = ["market_return_1", "market_volatility_20", "relative_return_5", "breadth_up", "return_dispersion"]
INPUTS = {"market_index": "data/input/context/market_index_000985.csv",
          "member_returns": "data/input/context/member_stock_returns.csv",
          "members": "data/input/market/index_931723_members.csv"}


def market_features(index_prices: pd.DataFrame) -> pd.DataFrame:
    """index_prices: date, close of the broad index (one row per trading day)."""
    p = index_prices[["date", "close"]].dropna().sort_values("date")
    if p.date.duplicated().any():
        raise ValueError("duplicate market index dates")
    r = p.close.pct_change(fill_method=None)
    return pd.DataFrame({"date": p.date.values,
                         "market_return_1": r.values,
                         "market_return_5": p.close.pct_change(5, fill_method=None).values,
                         "market_volatility_20": r.rolling(20, min_periods=20).std(ddof=1).values})


def breadth_features(returns: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """returns: stkcd, date, ret. members: stkcd, in_date, out_date ([in, out), out may be empty)."""
    r = returns.dropna(subset=["ret"]).copy()
    m = members.copy()
    m["out_date"] = m.out_date.fillna("9999-12-31").replace("", "9999-12-31")
    joined = r.merge(m, on="stkcd", how="inner")
    joined = joined[(joined.date >= joined.in_date) & (joined.date < joined.out_date)]
    if joined.duplicated(["stkcd", "date"]).any():
        raise ValueError("overlapping membership intervals")
    g = joined.groupby("date").ret
    out = pd.DataFrame({"breadth_up": g.apply(lambda x: float((x > 0).mean())),
                        "return_dispersion": g.std(ddof=1),
                        "breadth_n": g.size()}).reset_index()
    return out


def build_context(root: Path) -> dict:
    root = Path(root)
    config = json.loads((root / "configs/context_v1.json").read_text(encoding="utf-8"))
    source = safe_path(root, config["source"])
    sample = pd.read_csv(source / "daily_samples.csv", encoding="utf-8-sig")
    idx = pd.read_csv(safe_path(root, INPUTS["market_index"]), dtype={"date": str})
    ret = pd.read_csv(safe_path(root, INPUTS["member_returns"]), dtype={"stkcd": str, "date": str})
    mem = pd.read_csv(safe_path(root, INPUTS["members"]), dtype=str, encoding="utf-8-sig")
    mem = mem.rename(columns={"stkcd": "stkcd"})[["stkcd", "in_date", "out_date"]]
    mk = market_features(idx)
    br = breadth_features(ret, mem)
    ctx = sample[["date", "return_5"]].merge(mk, on="date", how="left").merge(br, on="date", how="left")
    ctx["relative_return_5"] = ctx.return_5 - ctx.market_return_5
    ctx = ctx[["date"] + CONTEXT + ["breadth_n"]]
    eligible = sample.loc[sample.eligible, "date"]
    checks = {"rows": len(ctx), "eligible_dates": int(len(eligible)),
              "eligible_missing": {c: int(ctx.loc[ctx.date.isin(eligible), c].isna().sum()) for c in CONTEXT},
              "breadth_n_min_eligible": int(ctx.loc[ctx.date.isin(eligible), "breadth_n"].min()),
              "breadth_n_max_eligible": int(ctx.loc[ctx.date.isin(eligible), "breadth_n"].max())}
    dest = safe_path(root, config["context_dataset"])
    dest.mkdir(parents=True, exist_ok=True)
    ctx.to_csv(dest / "context_features.csv", index=False, encoding="utf-8")
    meta = {"schema_version": 1, "code_commit": git_revision(root), "cutoff": "16:00 Asia/Shanghai; all inputs are 15:00 closes",
            "market_index": config["market_index"], "features": CONTEXT, "checks": checks,
            "inputs": {k: {"path": v, "sha256": sha256(safe_path(root, v))} for k, v in INPUTS.items()},
            "source_sample_sha256": sha256(source / "daily_samples.csv"),
            "outputs": {"context_features.csv": sha256(dest / "context_features.csv")}}
    (dest / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return checks


def features_for(model: str, config: dict) -> list:
    if model[0] != "M" or len(set(model)) != len(model) or not set(model[1:]) <= {"G", "B", "S"}:
        raise ValueError("invalid model name")
    cols = list(MARKET)
    if "G" in model: cols += config["groups"]["G"]
    if "B" in model: cols += config["groups"]["B"]
    if "S" in model: cols += SENTIMENT
    return cols


def load_frame(root: Path, config: dict) -> pd.DataFrame:
    sample = pd.read_csv(safe_path(root, config["source"]) / "daily_samples.csv", encoding="utf-8-sig")
    dest = safe_path(root, config["context_dataset"])
    meta = json.loads((dest / "metadata.json").read_text(encoding="utf-8"))
    if sha256(dest / "context_features.csv") != meta["outputs"]["context_features.csv"]:
        raise ValueError("context dataset changed")
    ctx = pd.read_csv(dest / "context_features.csv")
    frame = sample.merge(ctx.drop(columns=["breadth_n"]), on="date", how="left", validate="one_to_one")
    if len(frame) != len(sample) or not frame.date.is_monotonic_increasing:
        raise ValueError("misaligned context merge")
    return frame


def _rows(data, prob, h, arm, fold, model, C, thr):
    out = data[["date", f"label_end_date_{h}", f"target_{h}"]].rename(
        columns={f"label_end_date_{h}": "label_end_date", f"target_{h}": "actual"}).copy()
    out["arm"], out["fold"], out["horizon"], out["model"] = arm, fold, h, model
    out["C"] = float(C) if C is not None else np.nan
    out["probability_up"] = prob
    out["predicted"] = (np.asarray(prob) >= thr).astype(int)
    return out


def run_context(root: Path, run_id: str) -> dict:
    root = Path(root)
    cfg_path = root / "configs/context_v1.json"
    config = json.loads(cfg_path.read_text(encoding="utf-8"))
    frame = load_frame(root, config)
    out = root / "outputs" / run_id
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    preds, trials = [], []
    schedule = [(f["name"], f, "fold") for f in config["folds"]]
    r = config["retrospective"]
    schedule.append((r["name"], {"train_end": "2025-09-30", "validation_end": r["refit_end"], "evaluation_end": r["evaluation_end"]}, "retro"))
    for h in config["horizons"]:
        for name, fold, kind in schedule:
            parts = fold_parts(frame, h, fold)
            ev = parts["evaluation"]
            prior = float(parts["refit"][f"target_{h}"].mean())
            for arm in ["selected_C", "fixed_C"]:
                preds.append(_rows(ev, np.full(len(ev), prior), h, arm, name, "prior", None, config["threshold"]))
            for model in config["models"]:
                feats = features_for(model, config)
                C_sel, tr = choose_c(parts["train"], parts["validation"], feats, h, config)
                trials += [{"horizon": h, "fold": name, "model": model, **t} for t in tr]
                for arm, C in [("selected_C", C_sel), ("fixed_C", config["fixed_C"])]:
                    path = out / "models" / arm / name / f"{model}_{h}d.joblib"
                    prob = fit_predict_save(parts["refit"], ev, h, feats, C, config, path)
                    preds.append(_rows(ev, prob, h, arm, name, model, C, config["threshold"]))
    P = pd.concat(preds, ignore_index=True)
    P.to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(trials).to_csv(out / "validation_trials.csv", index=False)
    summary = summarize(P, config)
    for k, v in summary.items():
        v.to_csv(out / f"{k}.csv", index=False)
    run = {"run_id": run_id, "code_commit": git_revision(root), "config_sha256": sha256(cfg_path),
           "context_metadata_sha256": sha256(safe_path(root, config["context_dataset"]) / "metadata.json"),
           "interpretation": "development diagnostics on already-seen 2025/2026H1; not a new holdout",
           "dependencies": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "joblib"]},
           "artifacts": {p.relative_to(out).as_posix(): sha256(p) for p in sorted(out.rglob("*")) if p.is_file()}}
    (out / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"run_id": run_id, "prediction_rows": len(P), "models": len(list((out / "models").rglob("*.joblib")))}


def summarize(P: pd.DataFrame, config: dict) -> dict:
    thr = config["threshold"]
    metrics = []
    for (arm, h, fold, model), g in P.groupby(["arm", "horizon", "fold", "model"]):
        metrics.append({"arm": arm, "horizon": h, "fold": fold, "model": model, **score(g.actual, g.probability_up, thr)})
    pooled = []
    dev = P[P.fold.str.startswith("2025")]
    for (arm, h, model), g in dev.groupby(["arm", "horizon", "model"]):
        pooled.append({"arm": arm, "horizon": h, "fold": "2025_pooled", "model": model, **score(g.actual, g.probability_up, thr)})
    comps = []
    for (arm, h), part in P.groupby(["arm", "horizon"]):
        table = part.pivot_table(index=["fold", "date"], columns="model", values="probability_up", aggfunc="first")
        y = part.groupby(["fold", "date"]).actual.first().reindex(table.index).to_numpy(int)
        if table.isna().any().any():
            raise ValueError("models do not share evaluation dates")
        folds = table.index.get_level_values("fold").to_numpy()
        is_dev = np.char.startswith(folds.astype(str), "2025")
        for a, b, label in config["comparisons"]:
            delta = (table[b].to_numpy() - y) ** 2 - (table[a].to_numpy() - y) ** 2
            row = {"arm": arm, "horizon": h, "comparison": label, "base": a, "augmented": b}
            for f in pd.unique(folds):
                row[f"brier_delta_{f}"] = float(delta[folds == f].mean())
            d = delta[is_dev]
            idx = bootstrap_indices(folds[is_dev], config["bootstrap_repetitions"], config["bootstrap_block_length"], config["random_seed"])
            draws = d[idx].mean(axis=1)
            q = [row[f"brier_delta_{f['name']}"] for f in config["folds"]]
            row.update({"brier_delta_2025_pooled": float(d.mean()),
                        "ci95_low": float(np.quantile(draws, 0.025)), "ci95_high": float(np.quantile(draws, 0.975)),
                        "quarters_improved": int(sum(x < 0 for x in q))})
            comps.append(row)
    C = pd.DataFrame(comps)
    verdict = []
    for (h, label), g in C.groupby(["horizon", "comparison"]):
        s = g.set_index("arm")
        sel, fix = s.loc["selected_C"], s.loc["fixed_C"]
        helps = bool(sel.brier_delta_2025_pooled < 0 and sel.quarters_improved >= 3 and fix.brier_delta_2025_pooled <= 0)
        verdict.append({"horizon": h, "comparison": label, "selected_pooled_delta": sel.brier_delta_2025_pooled,
                        "selected_quarters_improved": int(sel.quarters_improved), "fixed_pooled_delta": fix.brier_delta_2025_pooled,
                        "retro_2026H1_selected_delta": sel.get("brier_delta_2026H1", np.nan), "helps_by_frozen_rule": helps})
    return {"fold_metrics": pd.DataFrame(metrics), "pooled_metrics": pd.DataFrame(pooled),
            "comparisons": C, "verdict": pd.DataFrame(verdict)}
