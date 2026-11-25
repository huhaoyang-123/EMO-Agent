"""Frozen time-split logistic baselines. Test outcomes never select settings."""
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from emo_agent.dataset import MARKET, SENTIMENT, sha256, git_revision
from emo_agent.audit import audit_project, safe_path


def split_data(frame, horizon, config):
    data = frame.loc[frame.eligible & frame[f"target_{horizon}"].notna()].copy()
    end = pd.to_datetime(data[f"label_end_date_{horizon}"])
    train_end, val_end = pd.Timestamp(config["train_end"]), pd.Timestamp(config["validation_end"])
    date = pd.to_datetime(data.date)
    if (end < date).any():
        raise ValueError("label precedes prediction")
    return {
        "train": data[(date <= train_end) & (end <= train_end)],
        "validation": data[(date > train_end) & (date <= val_end) & (end <= val_end)],
        "refit": data[(date <= val_end) & (end <= val_end)],
        "test": data[(date > val_end) & (date <= pd.Timestamp(config["test_end"]))],
    }


def make_model(C, seed):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(C=C, l1_ratio=0.0, solver="lbfgs", max_iter=3000, tol=1e-7, random_state=seed)),
    ])


def fit_model(frame, features, horizon, C, seed):
    if frame[f"target_{horizon}"].nunique() != 2:
        raise ValueError("training needs both outcome classes")
    model = make_model(C, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(frame[features], frame[f"target_{horizon}"].astype(int))
    return model


def score(y, probability, threshold=0.5):
    y, probability = np.asarray(y, dtype=int), np.asarray(probability, dtype=float)
    if not len(y) or not np.isfinite(probability).all() or not ((0 <= probability) & (probability <= 1)).all():
        raise ValueError("invalid scores")
    predicted = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    balanced = balanced_accuracy_score(y, predicted) if len(np.unique(y)) == 2 else None
    return {"n": len(y), "correct": int((y == predicted).sum()), "accuracy": float(accuracy_score(y, predicted)), "balanced_accuracy": float(balanced) if balanced is not None else None, "brier": float(brier_score_loss(y, probability)), "actual_up_rate": float(y.mean()), "predicted_up_rate": float(predicted.mean()), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def select_settings(frame, config):
    """Only train and validation data are accessed for fitting/scoring here."""
    settings, trials, validation_predictions = [], [], []
    for h in config["horizons"]:
        parts = split_data(frame, h, config)
        train, valid = parts["train"], parts["validation"]
        for name, features in [("market", MARKET), ("market_sentiment", MARKET + SENTIMENT)]:
            candidates = []
            for C in config["C_grid"]:
                model = fit_model(train, features, h, C, config["random_seed"])
                probability = model.predict_proba(valid[features])[:, 1]
                metrics = score(valid[f"target_{h}"], probability, config["threshold"])
                trials.append({"horizon": h, "model": name, "C": C, **metrics})
                candidates.append((metrics["brier"], C, probability))
            best = min(candidates, key=lambda x: (x[0], x[1]))
            settings.append({"horizon": h, "model": name, "features": features, "C": best[1], "validation_brier": best[0]})
            validation_predictions.append(prediction_frame(valid, best[2], h, name, config["threshold"], "validation"))
        prior = float(train[f"target_{h}"].mean())
        validation_predictions.append(prediction_frame(valid, np.full(len(valid), prior), h, "prior", config["threshold"], "validation"))
    return settings, pd.DataFrame(trials), pd.concat(validation_predictions, ignore_index=True)


def prediction_frame(data, probability, horizon, name, threshold, period):
    result = data[["date", "cutoff_at", f"label_end_date_{horizon}", f"target_{horizon}"]].copy()
    result = result.rename(columns={f"label_end_date_{horizon}": "label_end_date", f"target_{horizon}": "actual"})
    result["horizon"] = horizon
    result["model"] = name
    result["period"] = period
    result["probability_up"] = probability
    result["predicted"] = (np.asarray(probability) >= threshold).astype(int)
    result["correct"] = result.predicted.eq(result.actual)
    return result


def paired_block_intervals(predictions, config):
    """Descriptive paired circular-block bootstrap; not an independence claim."""
    rng = np.random.default_rng(config["random_seed"])
    results = []
    for h in config["horizons"]:
        part = predictions[predictions.horizon.eq(h)]
        market = part[part.model.eq("market")].sort_values("date")
        aug = part[part.model.eq("market_sentiment")].sort_values("date")
        if market.date.tolist() != aug.date.tolist():
            raise ValueError("unpaired test dates")
        y = market.actual.to_numpy(int)
        p, q = market.probability_up.to_numpy(), aug.probability_up.to_numpy()
        delta_acc = (q >= config["threshold"]).astype(int) == y
        delta_acc = delta_acc.astype(float) - ((p >= config["threshold"]).astype(int) == y).astype(float)
        delta_brier = (q-y)**2 - (p-y)**2
        n, length = len(y), config["bootstrap_block_length"]
        draws = []
        for _ in range(config["bootstrap_repetitions"]):
            starts = rng.integers(0, n, size=int(np.ceil(n/length)))
            index = ((starts[:, None] + np.arange(length)) % n).ravel()[:n]
            draws.append([delta_acc[index].mean(), delta_brier[index].mean()])
        bounds = np.quantile(draws, [0.025, 0.975], axis=0)
        results.append({"horizon": h, "accuracy_delta": float(delta_acc.mean()), "accuracy_delta_95pct_low": float(bounds[0,0]), "accuracy_delta_95pct_high": float(bounds[1,0]), "brier_delta": float(delta_brier.mean()), "brier_delta_95pct_low": float(bounds[0,1]), "brier_delta_95pct_high": float(bounds[1,1]), "block_length": length})
    return results


def run_baseline(root: Path, run_id: str):
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id):
        raise ValueError("run_id must be letters, digits, dash or underscore")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("commit code/config before a recorded experiment")
    if not audit_project(root)["ok"]:
        raise ValueError("immutable inputs changed")
    config_path = root / "configs/baseline_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset = safe_path(root, config["dataset"])
    metadata = json.loads((dataset / "metadata.json").read_text(encoding="utf-8"))
    for name, expected in metadata["outputs"].items():
        if sha256(safe_path(dataset, name)) != expected:
            raise ValueError("dataset output hash mismatch")
    if metadata["market_features"] != MARKET or metadata["sentiment_features"] != SENTIMENT:
        raise ValueError("feature whitelist mismatch")
    frame = pd.read_csv(dataset / "daily_samples.csv")
    if frame.date.duplicated().any() or not frame.date.is_monotonic_increasing:
        raise ValueError("invalid sample dates")
    out = root / "outputs" / run_id
    out.mkdir(parents=True, exist_ok=False)
    (out / "models").mkdir()
    (out / "config.json").write_text(json.dumps(config, indent=2)+"\n", encoding="utf-8")
    # Freeze all choices using validation before any test scoring.
    settings, trials, validation_predictions = select_settings(frame, config)
    (out / "selected_settings.json").write_text(json.dumps(settings, indent=2)+"\n", encoding="utf-8")
    trials.to_csv(out / "validation_trials.csv", index=False)
    validation_predictions.to_csv(out / "validation_predictions.csv", index=False)
    predictions, metrics, coefficients, splits = [], [], [], []
    for h in config["horizons"]:
        parts = split_data(frame, h, config)
        for stage, data in parts.items():
            splits.append({"horizon": h, "stage": stage, "n": len(data), "first_date": data.date.min(), "last_date": data.date.max(), "latest_label_end": data[f"label_end_date_{h}"].max(), "up_count": int(data[f"target_{h}"].sum())})
        refit, test = parts["refit"], parts["test"]
        prior = float(refit[f"target_{h}"].mean())
        p = np.full(len(test), prior)
        predictions.append(prediction_frame(test, p, h, "prior", config["threshold"], "test"))
        metrics.append({"horizon": h, "model": "prior", "C": None, "training_up_rate": prior, **score(test[f"target_{h}"], p)})
        for selected in [x for x in settings if x["horizon"] == h]:
            features, name = selected["features"], selected["model"]
            model = fit_model(refit, features, h, selected["C"], config["random_seed"])
            p = model.predict_proba(test[features])[:,1]
            path = out / "models" / f"{name}_{h}d.joblib"
            joblib.dump(model, path)
            if not np.allclose(joblib.load(path).predict_proba(test[features])[:,1], p, atol=1e-12, rtol=0):
                raise ValueError("serialized model prediction mismatch")
            predictions.append(prediction_frame(test, p, h, name, config["threshold"], "test"))
            metrics.append({"horizon": h, "model": name, "C": selected["C"], "training_up_rate": prior, **score(test[f"target_{h}"], p, config["threshold"])})
            for feature, coef in zip(model.named_steps["impute"].get_feature_names_out(), model.named_steps["logit"].coef_[0]):
                coefficients.append({"horizon": h, "model": name, "feature": feature, "standardized_coefficient": float(coef)})
    predictions = pd.concat(predictions, ignore_index=True)
    predictions.to_csv(out / "test_predictions.csv", index=False, encoding="utf-8-sig")
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(out / "test_metrics.csv", index=False)
    pd.DataFrame(splits).to_csv(out / "split_audit.csv", index=False)
    pd.DataFrame(coefficients).to_csv(out / "coefficients.csv", index=False)
    monthly = []
    for (h, name, month), part in predictions.groupby(["horizon", "model", predictions.date.str[:7]]):
        monthly.append({"horizon": int(h), "model": name, "month": month, **score(part.actual, part.probability_up, config["threshold"])})
    pd.DataFrame(monthly).to_csv(out / "monthly_metrics.csv", index=False)
    intervals = paired_block_intervals(predictions, config)
    pd.DataFrame(intervals).to_csv(out / "sentiment_delta_intervals.csv", index=False)
    run = {"run_id": run_id, "code_commit": git_revision(root), "git_clean_at_start": True, "config_sha256": sha256(config_path), "dataset_metadata_sha256": sha256(dataset / "metadata.json"), "dataset_sample_sha256": sha256(dataset / "daily_samples.csv"), "dataset_code_commit": metadata["code_commit"], "dependencies": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "scipy", "joblib"]}, "settings": settings, "paired_block_intervals": intervals, "availability_assumption": "interval_end_only_not_verified_publication", "artifacts": {p.relative_to(out).as_posix(): sha256(p) for p in sorted(out.rglob("*")) if p.is_file()}}
    (out / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return metrics.astype(object).where(pd.notna(metrics), None).to_dict("records")
