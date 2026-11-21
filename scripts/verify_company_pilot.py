"""Independent arithmetic checks against raw prices and saved predictions.

Run from project root: python -m uv run --locked python scripts/verify_company_pilot.py
Optional --reproduction compares the frozen run's artifact manifests.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from emo_agent.dataset import sha256


def verify(root, run_id, reproduction=None):
    out = root / "outputs" / run_id
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    dataset = root / cfg["dataset"]
    panel = pd.read_csv(dataset / "company_samples.csv", dtype={"code": str})
    metadata = json.loads((dataset / "metadata.json").read_text(encoding="utf-8"))
    for name, digest in run["artifacts"].items():
        assert sha256(out / name) == digest, name
    for name, digest in metadata["outputs"].items():
        assert sha256(dataset / name) == digest, name
    source = pd.read_csv(root / cfg["stock_source"], dtype={"Stkcd": str}).set_index(["Stkcd", "Trddt"])
    calendar = pd.read_csv(root / "data/input/market/index_931723_full.csv").date.tolist()
    positions = {date: i for i, date in enumerate(calendar)}
    current_keys = list(zip(panel.code, panel.date))
    current = source.reindex(current_keys).reset_index(drop=True)
    assert np.allclose(panel.turnover_pct, 100*current.Dnshrtrd*current.Clsprc/(1000*current.Dsmvosd), equal_nan=True)
    label_checks = 0
    for h in cfg["horizons"]:
        end_dates = [calendar[positions[date]+h] if positions[date]+h < len(calendar) else None for date in panel.date]
        future = source.reindex(list(zip(panel.code, end_dates))).reset_index(drop=True)
        valid = current.Dnshrtrd.gt(0) & future.Dnshrtrd.gt(0) & current.Adjprcnd.notna() & future.Adjprcnd.notna()
        expected = (future.Adjprcnd > current.Adjprcnd).astype(float).where(valid)
        np.testing.assert_allclose(panel[f"target_{h}"], expected, equal_nan=True, rtol=0, atol=0)
        assert panel[f"label_end_date_{h}"].tolist() == end_dates
        label_checks += len(panel)
    trace = pd.read_csv(dataset / "source_trace.csv", dtype={"code": str})
    assignments = pd.read_csv(root / cfg["source"] / "interval_assignments.csv", dtype={"Scode": str})
    traced_ids = [item for row in trace.source_records.dropna() for item in row.split("|")]
    source_ids = assignments.loc[assignments.observed & assignments.cutoff_at.notna(), "source_record_id"].tolist()
    assert len(traced_ids) == len(set(traced_ids)) and set(traced_ids) == set(source_ids)
    assert panel.Tpostnum.sum() == assignments.Tpostnum.sum()
    pred = pd.read_csv(out / "predictions.csv", dtype={"code": str})
    metric_checks = 0
    for row in pd.read_csv(out / "metrics.csv").itertuples():
        f = pred[(pred.universe == row.universe) & (pred.period == row.period) & (pred.horizon == row.horizon) & (pred.model == row.model)]
        daily = []
        weighted_confusion = np.zeros((2, 2))
        for date, day in f.groupby("date"):
            y, p = day.actual.to_numpy(int), day.probability_up.to_numpy(float)
            daily.append([((p-y)**2).mean(), ((p >= .5) == y).mean()])
            for actual, guessed in zip(y, (p >= .5).astype(int)):
                weighted_confusion[actual, guessed] += 1/len(day)
        means = np.mean(daily, axis=0)
        balanced = np.mean(np.diag(weighted_confusion) / weighted_confusion.sum(axis=1))
        np.testing.assert_allclose([row.brier, row.accuracy, row.balanced_accuracy], [*means, balanced], atol=1e-12, rtol=0)
        metric_checks += 1
    # Each prediction label must come from the immutable reconstructed table.
    for universe in ["company", "index"]:
        table = pd.read_csv(dataset / (universe + "_samples.csv"), dtype={"code": str}).set_index(["code", "date"])
        for h in cfg["horizons"]:
            f = pred[(pred.universe == universe) & (pred.horizon == h)]
            truth = table.reindex(list(zip(f.code, f.date)))
            np.testing.assert_array_equal(f.actual.to_numpy(), truth[f"target_{h}"].to_numpy())
    splits = pd.read_csv(out / "splits.csv")
    for fold in cfg["folds"]:
        f = splits[splits.fold.eq(fold["name"])]
        assert f.loc[f.stage.eq("train"), "latest_label_end"].le(fold["train_end"]).all()
        assert f.loc[f.stage.eq("evaluation"), "first_date"].gt(fold["train_end"]).all()
        assert f.loc[f.stage.eq("evaluation"), "latest_label_end"].le(fold["evaluation_end"]).all()
    # Directly reconstruct each paired gain from raw saved probabilities.
    gain_checks = 0
    for row in pd.read_csv(out / "contrasts.csv").itertuples():
        f = pred[pred.horizon.eq(row.horizon)]
        if row.period != "all": f = f[f.period.eq(row.period)]
        gains = {}
        for universe in (["company", "index"] if row.universe == "company_minus_index" else [row.universe]):
            g = f[f.universe.eq(universe)]
            pair = g[g.model.isin([row.base, row.added])].pivot(index=["date", "code", "actual"], columns="model", values="probability_up")
            assert pair.notna().all().all()
            y = pair.index.get_level_values("actual").to_numpy()
            diff = (pair[row.base]-y)**2 - (pair[row.added]-y)**2
            gains[universe] = diff.groupby(level="date").mean()
        delta = gains["company"]-gains["index"] if row.universe == "company_minus_index" else gains[row.universe]
        assert delta.notna().all()
        assert np.isclose(delta.mean(), row.gain, atol=1e-12, rtol=0)
        gain_checks += 1
    identical = None
    if reproduction:
        other = root / "outputs" / reproduction
        repeated = json.loads((other / "run.json").read_text(encoding="utf-8"))
        assert run["code_commit"] == repeated["code_commit"]
        assert run["artifacts"] == repeated["artifacts"]
        for name, digest in repeated["artifacts"].items():
            assert sha256(other/name) == digest
        identical = len(run["artifacts"])
    result = {"ok": True, "label_rows_checked": label_checks, "turnover_rows_checked": len(panel), "source_ids_checked": len(traced_ids), "independent_metric_groups": metric_checks, "independent_gain_groups": gain_checks, "reproduced_artifacts": identical, "run_code_commit": run["code_commit"], "checks": ["raw_price_labels", "global_calendar", "turnover_units", "source_trace_and_counts", "mature_splits", "prediction_outcomes", "date_equal_metrics", "paired_gains", "artifact_hashes"]}
    (out / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="company_pilot_v1")
    parser.add_argument("--reproduction")
    args = parser.parse_args()
    print(json.dumps(verify(Path.cwd(), args.run_id, args.reproduction)))
