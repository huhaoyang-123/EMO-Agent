"""Frozen company panel pilot: availability, attention, then sentiment direction.

All historical evaluations are developmental. No real-time or causal claim.
"""
from pathlib import Path
import importlib.metadata
import json
import subprocess
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from emo_agent.audit import audit_project, safe_path
from emo_agent.dataset import COUNTS, sha256, git_revision
from emo_agent.stability import bootstrap_indices

ARMS = ("A_market", "B_attention", "C_direction")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def validate_stocks(frame):
    if not frame.Stkcd.str.fullmatch(r"[0-9]{6}").all():
        raise ValueError("stock codes must be six-digit text")
    if frame.duplicated(["Stkcd", "Trddt"]).any():
        raise ValueError("duplicate stock/date")
    if pd.to_datetime(frame.Trddt, errors="raise").isna().any():
        raise ValueError("missing stock date")
    for column in ["Clsprc", "Adjprcnd", "Dnshrtrd", "Dnvaltrd", "Dsmvosd"]:
        x = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(x).all() or (x < 0).any():
            raise ValueError("invalid stock numeric field: " + column)
        if column in ["Clsprc", "Adjprcnd", "Dsmvosd"] and x.eq(0).any():
            raise ValueError("nonpositive price/value")
    if not frame.Markettype.isin([1, 4, 16, 32, 64]).all():
        raise ValueError("turnover conversion requires A-share CNY units")


def aggregate_intervals(assignments, group_company=True):
    """Preserve missing counts and traceability; rate denominator is observed hours."""
    f = assignments.loc[assignments.cutoff_at.notna()].copy()
    for column in ["interval_start", "interval_end", "cutoff_at"]:
        f[column] = pd.to_datetime(f[column])
        if f[column].dt.tz is None or f[column].isna().any():
            raise ValueError("timezone-aware interval times required")
    if not f.observed.isin([True, False]).all():
        raise ValueError("observed must be boolean")
    if (f.interval_end > f.cutoff_at).any():
        raise ValueError("future interval in features")
    if f.duplicated(["Scode", "Date", "Time"]).any():
        raise ValueError("duplicate expected interval")
    observed = f.observed.astype(bool)
    if f.loc[observed, "source_record_id"].isna().any() or f.loc[observed, "source_record_id"].duplicated().any():
        raise ValueError("invalid source trace")
    counts = f.loc[observed, COUNTS].apply(pd.to_numeric, errors="raise")
    if counts.isna().any().any() or (counts < 0).any().any() or (counts % 1 != 0).any().any():
        raise ValueError("invalid observed count")
    if not counts.Tpostnum.eq(counts.Pospostnum + counts.Negpostnum + counts.Neupostnum).all():
        raise ValueError("unbalanced counts")
    if f.loc[~observed, COUNTS].notna().any().any():
        raise ValueError("missing interval cannot have known counts")
    f["hours"] = (f.interval_end - f.interval_start).dt.total_seconds() / 3600
    if (f.hours <= 0).any():
        raise ValueError("invalid interval duration")
    f["observed_hours"] = f.hours.where(observed, 0)
    f["date"] = f.cutoff_at.dt.strftime("%Y-%m-%d")
    f["code"] = f.Scode if group_company else "INDEX"
    grouped = f.groupby(["code", "date"], sort=True)
    result = grouped[COUNTS].sum(min_count=1)
    result["expected_slots"] = grouped.size()
    result["observed_slots"] = grouped["observed"].sum()
    result["expected_hours"] = grouped.hours.sum()
    result["observed_hours"] = grouped.observed_hours.sum()
    result["cutoff_at"] = grouped.cutoff_at.first()
    result["window_start"] = grouped.interval_start.min()
    result["source_records"] = grouped.source_record_id.agg(lambda x: "|".join(x.dropna().astype(str)))
    result["slot_coverage"] = result.observed_slots / result.expected_slots
    result["hour_coverage"] = result.observed_hours / result.expected_hours
    result["window_hours"] = (result.cutoff_at - result.window_start).dt.total_seconds() / 3600
    result["net_sentiment"] = (result.Pospostnum - result.Negpostnum) / result.Tpostnum.replace(0, np.nan)
    result["sentiment_missing"] = result.net_sentiment.isna().astype(int)
    result["post_rate"] = result.Tpostnum / result.observed_hours.replace(0, np.nan)
    return result.reset_index()


def sentiment_history(daily, calendar):
    pieces = []
    for code, rows in daily.groupby("code", sort=True):
        f = rows.set_index("date").reindex(calendar).copy()
        f["code"] = code
        f["log_post_rate"] = np.log1p(f.post_rate)
        previous_rate = f.post_rate.shift(1).rolling(20, min_periods=20).mean()
        previous_net = f.net_sentiment.shift(1).rolling(5, min_periods=5).mean()
        f["post_heat_20"] = f.log_post_rate - np.log1p(previous_rate)
        f["sentiment_change_5"] = f.net_sentiment - previous_net
        f["heat_history_missing"] = f.post_heat_20.isna().astype(int)
        f["direction_history_missing"] = f.sentiment_change_5.isna().astype(int)
        # Only expected company dates are candidates; reindexing still breaks gaps.
        pieces.append(f.loc[f.expected_slots.notna()].rename_axis("date").reset_index())
    return pd.concat(pieces, ignore_index=True)


def stock_history(stocks, calendar, horizons):
    """Shift on the market calendar, never on a stock's observed-row calendar."""
    validate_stocks(stocks)
    if len(set(calendar)) != len(calendar) or list(calendar) != sorted(calendar):
        raise ValueError("invalid market calendar")
    if not stocks.Trddt.isin(calendar).all():
        raise ValueError("stock date absent from market calendar")
    pieces = []
    for code, rows in stocks.groupby("Stkcd", sort=True):
        f = rows.set_index("Trddt").reindex(calendar).copy()
        f["code"] = code
        f["stock_present"] = f.Stkcd.notna()
        f["stock_traded"] = f.Dnshrtrd.gt(0)
        price = f.Adjprcnd.where(f.stock_traded)
        for n in [1, 5, 20]:
            f[f"stock_return_{n}"] = price / price.shift(n) - 1
        for n in [5, 20]:
            f[f"stock_volatility_{n}"] = f.stock_return_1.rolling(n, min_periods=n).std(ddof=1)
        volume_mean = f.Dnshrtrd.shift(1).rolling(20, min_periods=20).mean()
        f["volume_heat_20"] = np.log1p(f.Dnshrtrd) - np.log1p(volume_mean)
        f["turnover_pct"] = 100 * f.Dnshrtrd * f.Clsprc / (1000 * f.Dsmvosd)
        f["log_turnover"] = np.log1p(f.turnover_pct)
        f["log_float_value"] = np.log(1000 * f.Dsmvosd)
        dates = pd.Series(calendar, index=f.index)
        for h in horizons:
            future = price.shift(-h)
            f[f"return_forward_{h}"] = future / price - 1
            f[f"target_{h}"] = (future > price).astype(float).where(future.notna() & price.notna())
            f[f"label_end_date_{h}"] = dates.shift(-h)
        pieces.append(f.rename_axis("date").reset_index())
    return pd.concat(pieces, ignore_index=True)


def membership_mask(frame, members):
    active = pd.Series(False, index=frame.index)
    for row in members.itertuples():
        active |= frame.code.eq(row.stkcd) & frame.date.ge(row.in_date) & (frame.date.lt(row.out_date) if pd.notna(row.out_date) else True)
    return active


def source_inputs(root, config):
    if not audit_project(root)["ok"]:
        raise ValueError("input audit failed")
    source = safe_path(root, config["source"])
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    for name, digest in metadata["outputs"].items():
        if sha256(safe_path(source, name)) != digest:
            raise ValueError("cutoff source changed")
    return source, metadata


def require_clean(root):
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("commit code/config before a recorded build or scoring")


def build(root):
    require_clean(root)
    config_path = root / "configs/company_pilot_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source, _ = source_inputs(root, config)
    assignments = pd.read_csv(source / "interval_assignments.csv", dtype={"Scode": str})
    index = pd.read_csv(source / "daily_samples.csv")
    prices = pd.read_csv(root / "data/input/market/index_931723_full.csv")
    members = pd.read_csv(root / "data/input/market/index_931723_members.csv", dtype={"stkcd": str})
    stocks = pd.read_csv(safe_path(root, config["stock_source"]), dtype={"Stkcd": str})
    calendar = prices.date.tolist()
    sentiment_calendar = index.date.tolist()
    company_sentiment = sentiment_history(aggregate_intervals(assignments), sentiment_calendar)
    industry_sentiment = sentiment_history(aggregate_intervals(assignments, False), sentiment_calendar)
    market = stock_history(stocks, calendar, config["horizons"])
    # Sentiment is built separately from target prices; no target enters a feature.
    panel = company_sentiment.merge(market, on=["code", "date"], how="left", validate="one_to_one")
    panel = panel.merge(index[["date", *config["index_market"], "eligible"]].rename(columns={"eligible": "index_eligible"}), on="date", how="left", validate="many_to_one")
    panel["member_at_cutoff"] = membership_mask(panel, members)
    market_ok = panel[config["company_market"]].notna().all(axis=1)
    panel["exclusion_reason"] = np.select(
        [~panel.member_at_cutoff, ~panel.stock_present.fillna(False), ~panel.stock_traded.fillna(False), ~market_ok, ~panel.index_eligible],
        ["not_member_at_cutoff", "missing_stock_row", "no_current_trade", "market_history_missing", "index_warmup"], default="")
    panel["eligible"] = panel.exclusion_reason.eq("")
    panel = panel.sort_values(["date", "code"]).reset_index(drop=True)
    index_keep = ["date", *config["index_market"], "eligible"] + [x for h in config["horizons"] for x in (f"target_{h}", f"label_end_date_{h}", f"return_forward_{h}")]
    industry = index[index_keep].merge(industry_sentiment, on="date", validate="one_to_one").sort_values("date")
    if not np.isclose(company_sentiment.Tpostnum.sum(), assignments.Tpostnum.sum()):
        raise ValueError("company aggregate does not reconcile")
    out = safe_path(root, config["dataset"])
    out.mkdir(parents=True, exist_ok=False)
    # Full source trace in a separate file keeps the model table compact.
    company_sentiment[["code", "date", "source_records"]].to_csv(out / "source_trace.csv", index=False)
    panel = panel.drop(columns="source_records")
    industry = industry.drop(columns="source_records")
    panel.to_csv(out / "company_samples.csv", index=False)
    industry.to_csv(out / "index_samples.csv", index=False)
    panel.loc[~panel.eligible, ["code", "date", "exclusion_reason"]].to_csv(out / "excluded_rows.csv", index=False)
    checks = {
        "stock_source_rows": len(stocks), "stock_source_companies": int(stocks.Stkcd.nunique()),
        "stock_source_start": stocks.Trddt.min(), "stock_source_end": stocks.Trddt.max(),
        "expected_company_days": len(panel), "companies": int(panel.code.nunique()),
        "dates": int(panel.date.nunique()), "eligible_rows": int(panel.eligible.sum()),
        "eligible_dates": int(panel.loc[panel.eligible, "date"].nunique()),
        "eligible_companies": int(panel.loc[panel.eligible, "code"].nunique()),
        "exclusions": panel.loc[~panel.eligible, "exclusion_reason"].value_counts().to_dict(),
        "no_observed_intervals": int(panel.observed_slots.eq(0).sum()),
        "posts_reconciled": int(company_sentiment.Tpostnum.sum()),
        "label_missing_eligible": {str(h): int((panel.eligible & panel[f"target_{h}"].isna()).sum()) for h in config["horizons"]},
    }
    inputs = [config_path, root / "docs/data_manifest.json", source / "metadata.json", source / "daily_samples.csv", source / "interval_assignments.csv", safe_path(root, config["stock_source"])]
    write_json(out / "metadata.json", {"code_commit": git_revision(root), "checks": checks, "availability": "interval_end_only_publication_unverified", "inputs": {p.relative_to(root).as_posix(): sha256(p) for p in inputs}, "outputs": {p.name: sha256(p) for p in sorted(out.glob("*.csv"))}})
    return checks


def daily_weights(frame):
    return (1 / frame.groupby("date").date.transform("size")).to_numpy()


class PanelLogit:
    """Train-only numeric preprocessing and regularized company intercepts."""
    def __init__(self, features, company, C=0.01, seed=42):
        self.features = list(features)
        self.company = company
        self.C = C
        self.seed = seed

    def fit(self, frame, target):
        y = np.asarray(target, dtype=int)
        if set(y) != {0, 1}:
            raise ValueError("training requires both classes")
        weights = daily_weights(frame)
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        numeric = self.imputer.fit_transform(frame[self.features])
        self.scaler = StandardScaler().fit(numeric, sample_weight=weights)
        x = self.scaler.transform(numeric)
        if self.company:
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            x = np.column_stack([x, self.encoder.fit_transform(frame[["code"]])])
        self.estimator = LogisticRegression(C=self.C, l1_ratio=0.0, solver="lbfgs", max_iter=3000, tol=1e-7, random_state=self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            self.estimator.fit(x, y, sample_weight=weights)
        self.training_weight_sum = float(weights.sum())
        return self

    def predict_proba(self, frame):
        x = self.scaler.transform(self.imputer.transform(frame[self.features]))
        if self.company:
            x = np.column_stack([x, self.encoder.transform(frame[["code"]])])
        return self.estimator.predict_proba(x)

    def coefficients(self):
        names = self.features + (list(self.encoder.get_feature_names_out(["code"])) if self.company else [])
        return [{"feature": "intercept", "coefficient": float(self.estimator.intercept_[0])}] + [
            {"feature": n, "coefficient": float(c)} for n, c in zip(names, self.estimator.coef_[0], strict=True)]


def split_panel(frame, horizon, fold):
    f = frame.loc[frame.eligible & frame[f"target_{horizon}"].notna()].copy()
    end = f[f"label_end_date_{horizon}"]
    if end.isna().any() or (end <= f.date).any():
        raise ValueError("invalid label maturity")
    train = f[(f.date <= fold["train_end"]) & (end <= fold["train_end"])]
    test = f[(f.date > fold["train_end"]) & (f.date <= fold["evaluation_end"]) & (end <= fold["evaluation_end"])]
    return train, test


def features_for(config, universe, arm):
    if arm not in ARMS or universe not in ("company", "index"):
        raise ValueError("unknown experiment arm")
    cols = config[universe + "_market"] + config["controls"]
    if arm in ARMS[1:]:
        cols += config["attention"]
    if arm == ARMS[2]:
        cols += config["direction"]
    if len(cols) != len(set(cols)):
        raise ValueError("duplicate feature")
    return cols


def metrics(frame):
    w = daily_weights(frame)
    y, p = frame.actual.to_numpy(int), frame.probability_up.to_numpy(float)
    if not len(y) or not np.isfinite(p).all() or not ((0 <= p) & (p <= 1)).all():
        raise ValueError("invalid prediction")
    correct = (p >= 0.5) == y
    recalls = [float(np.average(correct[y == value], weights=w[y == value])) for value in (0, 1) if (y == value).any()]
    return {"n": len(frame), "dates": int(frame.date.nunique()), "companies": int(frame.code.nunique()),
            "accuracy": float(np.average(correct, weights=w)), "row_accuracy": float(correct.mean()),
            "balanced_accuracy": float(np.mean(recalls)) if len(recalls) == 2 else None,
            "brier": float(np.average((p-y)**2, weights=w)),
            "actual_up_rate": float(np.average(y, weights=w)), "predicted_up_rate": float(np.average(p >= 0.5, weights=w))}


def daily_losses(predictions):
    keys = ["universe", "horizon", "fold", "period", "date", "code"]
    if predictions.duplicated(keys + ["model"]).any():
        raise ValueError("duplicate prediction key")
    if predictions.groupby(keys).actual.nunique().ne(1).any():
        raise ValueError("unpaired outcomes")
    p = predictions.pivot(index=keys, columns="model", values="probability_up")
    if set(p.columns) != set(["prior", *ARMS]) or p.isna().any().any():
        raise ValueError("unpaired model rows")
    f = predictions.copy()
    f["brier"] = (f.probability_up - f.actual)**2
    f["accuracy"] = ((f.probability_up >= 0.5) == f.actual).astype(float)
    return f.groupby(keys[:-1] + ["model"], as_index=False).agg(brier=("brier", "mean"), accuracy=("accuracy", "mean"), companies=("code", "size"))


def contrasts(losses, config):
    summaries, folds = [], []
    for h in config["horizons"]:
        for period in ["primary_2025", "diagnostic_2026H1", "all"]:
            f = losses[losses.horizon.eq(h)]
            if period != "all":
                f = f[f.period.eq(period)]
            p = f.pivot(index=["fold", "date"], columns=["universe", "model"], values="brier").sort_index()
            if p.isna().any().any():
                raise ValueError("index and company dates differ")
            draws = bootstrap_indices(p.index.get_level_values("fold"), config["bootstrap_repetitions"], config["bootstrap_block_length"], config["seed"] + h)
            gains = {}
            for universe in ["company", "index"]:
                for base, added in [(ARMS[0], ARMS[1]), (ARMS[1], ARMS[2]), (ARMS[0], ARMS[2])]:
                    gains[(universe, base, added)] = (p[(universe, base)] - p[(universe, added)]).to_numpy()
            gains[("company_minus_index", ARMS[1], ARMS[2])] = gains[("company", ARMS[1], ARMS[2])] - gains[("index", ARMS[1], ARMS[2])]
            for (universe, base, added), gain in gains.items():
                lo, hi = np.quantile(gain[draws].mean(axis=1), config["interval_quantiles"])
                per_fold = pd.Series(gain, index=p.index).groupby(level="fold").mean()
                summaries.append({"horizon": h, "period": period, "universe": universe, "base": base, "added": added, "gain": float(gain.mean()), "low": float(lo), "high": float(hi), "dates": len(p), "positive_folds": int((per_fold > 0).sum()), "folds": len(per_fold)})
                if period == "all":
                    folds.extend({"horizon": h, "fold": name, "universe": universe, "base": base, "added": added, "gain": float(value)} for name, value in per_fold.items())
    return pd.DataFrame(summaries), pd.DataFrame(folds)


def continuation_gates(comparisons, config):
    gates = []
    for h in config["horizons"]:
        subset = comparisons[(comparisons.horizon == h) & (comparisons.base == ARMS[1]) & (comparisons.added == ARMS[2])]
        primary = subset[(subset.universe == "company") & (subset.period == "primary_2025")].iloc[0]
        extension = subset[(subset.universe == "company") & (subset.period == "diagnostic_2026H1")].iloc[0]
        versus_index = subset[(subset.universe == "company_minus_index") & (subset.period == "primary_2025")].iloc[0]
        checks = {"primary_gain_at_least_minimum": bool(primary.gain >= config["minimum_brier_gain"]), "primary_interval_positive": bool(primary.low > 0), "primary_positive_3_of_4": bool(primary.positive_folds >= 3), "extension_gain_positive": bool(extension.gain > 0), "extension_positive_2_of_2": bool(extension.positive_folds == 2), "beats_index_interval_positive": bool(versus_index.low > 0)}
        gates.append({"horizon": h, "passed": all(checks.values()), "checks": checks})
    return gates


def run(root, run_id):
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id):
        raise ValueError("invalid run id")
    require_clean(root)
    config_path = root / "configs/company_pilot_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_inputs(root, config)
    dataset = safe_path(root, config["dataset"])
    meta_path = dataset / "metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    for name, expected in metadata["outputs"].items():
        if sha256(safe_path(dataset, name)) != expected:
            raise ValueError("panel dataset changed")
    for name, expected in metadata["inputs"].items():
        if sha256(safe_path(root, name)) != expected:
            raise ValueError("panel source/config changed")
    frames = {u: pd.read_csv(dataset / (u + "_samples.csv"), dtype={"code": str}) for u in ("company", "index")}
    out = root / "outputs" / run_id
    out.mkdir(parents=True, exist_ok=False)
    (out / "models").mkdir()
    write_json(out / "config.json", config)
    predictions, splits, coefficients = [], [], []
    for fold in config["folds"]:
        for h in config["horizons"]:
            parts = {u: split_panel(frame, h, fold) for u, frame in frames.items()}
            common_dates = sorted(set(parts["company"][1].date) & set(parts["index"][1].date))
            if not common_dates:
                raise ValueError("no common evaluation dates")
            for universe, (train, test) in parts.items():
                test = test[test.date.isin(common_dates)].sort_values(["date", "code"])
                for stage, f in [("train", train), ("evaluation", test)]:
                    splits.append({"universe": universe, "horizon": h, "fold": fold["name"], "stage": stage, "n": len(f), "dates": f.date.nunique(), "companies": f.code.nunique(), "first_date": f.date.min(), "last_date": f.date.max(), "latest_label_end": f[f"label_end_date_{h}"].max(), "unseen_company_rows": int((~f.code.isin(train.code)).sum()) if stage == "evaluation" else 0})
                weights = daily_weights(train)
                prior = np.average(train[f"target_{h}"], weights=weights)
                for arm in ["prior", *ARMS]:
                    if arm == "prior":
                        p = np.full(len(test), prior)
                    else:
                        cols = features_for(config, universe, arm)
                        model = PanelLogit(cols, universe == "company", config["C"], config["seed"]).fit(train, train[f"target_{h}"])
                        p = model.predict_proba(test)[:, 1]
                        path = out / "models" / f"{universe}_{fold['name']}_{h}d_{arm}.joblib"
                        joblib.dump(model, path)
                        if not np.allclose(p, joblib.load(path).predict_proba(test)[:, 1], atol=1e-12, rtol=0):
                            raise ValueError("saved model probability mismatch")
                        coefficients.extend({"universe": universe, "fold": fold["name"], "horizon": h, "model": arm, **c} for c in model.coefficients())
                    result = test[["date", "code", f"target_{h}", f"label_end_date_{h}"]].rename(columns={f"target_{h}": "actual", f"label_end_date_{h}": "label_end_date"}).copy()
                    result["universe"], result["fold"], result["period"] = universe, fold["name"], fold["period"]
                    result["horizon"], result["model"], result["probability_up"] = h, arm, p
                    predictions.append(result)
            print(f"{fold['name']} h={h}: paired dates={len(common_dates)}", flush=True)
    pred = pd.concat(predictions, ignore_index=True)
    pred.to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(splits).to_csv(out / "splits.csv", index=False)
    pd.DataFrame(coefficients).to_csv(out / "coefficients.csv", index=False)
    for filename, grouping in [("metrics", ["universe", "period", "horizon", "model"]), ("fold_metrics", ["universe", "fold", "horizon", "model"]), ("company_metrics", ["universe", "period", "code", "horizon", "model"])]:
        result = [{**dict(zip(grouping, key, strict=True)), **metrics(f)} for key, f in pred.groupby(grouping, sort=True)]
        pd.DataFrame(result).to_csv(out / (filename + ".csv"), index=False)
    losses = daily_losses(pred)
    losses.to_csv(out / "daily_losses.csv", index=False)
    comparisons, fold_comparisons = contrasts(losses, config)
    comparisons.to_csv(out / "contrasts.csv", index=False)
    fold_comparisons.to_csv(out / "fold_contrasts.csv", index=False)
    gates = continuation_gates(comparisons, config)
    write_json(out / "continuation_gates.json", gates)
    manifest = {"run_id": run_id, "code_commit": git_revision(root), "git_clean_at_start": True, "dataset_metadata_sha256": sha256(meta_path), "config_sha256": sha256(config_path), "assessment": "retrospective_development_only", "dependencies": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "scipy", "joblib"]}, "artifacts": {p.relative_to(out).as_posix(): sha256(p) for p in sorted(out.rglob("*")) if p.is_file()}}
    write_json(out / "run.json", manifest)
    return {"run_id": run_id, "models": len(list((out / "models").glob("*"))), "prediction_rows": len(pred), "continuation_gates": gates}
