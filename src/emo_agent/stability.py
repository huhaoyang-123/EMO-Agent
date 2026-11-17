"""Predeclared rolling diagnosis and feature-group attribution, not a new holdout."""
from itertools import permutations
from pathlib import Path
import importlib.metadata
import json
import subprocess
import joblib
import numpy as np
import pandas as pd
from emo_agent.audit import audit_project, safe_path
from emo_agent.baseline import fit_model, score
from emo_agent.dataset import MARKET, SENTIMENT, sha256, git_revision

GROUPS = ("quality", "direction", "heat")
COMPARISONS = [(0,1,"quality"), (1,7,"content_given_quality"), (1,3,"direction_given_quality"), (1,5,"heat_given_quality"), (5,7,"direction_given_quality_heat"), (3,7,"heat_given_quality_direction"), (0,7,"all_added")]


def model_name(mask):
    if mask == -1:
        return "prior"
    return "M" + "".join(letter for i, letter in enumerate("QDH") if mask & (1 << i))


def features_for(mask, groups):
    flat = [x for group in GROUPS for x in groups[group]]
    if len(flat) != len(set(flat)) or set(flat) != set(SENTIMENT):
        raise ValueError("groups must partition the frozen sentiment/quality features")
    if mask not in range(8):
        raise ValueError("invalid group mask")
    selected = {x for i, group in enumerate(GROUPS) if mask & (1 << i) for x in groups[group]}
    return MARKET + [x for x in SENTIMENT if x in selected]


def fold_parts(frame, horizon, fold):
    valid = frame.eligible & frame[f"target_{horizon}"].notna()
    data = frame.loc[valid].copy()
    dates = pd.to_datetime(data.date)
    end = pd.to_datetime(data[f"label_end_date_{horizon}"])
    t, v, e = (pd.Timestamp(fold[x]) for x in ["train_end", "validation_end", "evaluation_end"])
    if not t < v < e or (end < dates).any() or end.isna().any():
        raise ValueError("invalid boundaries or label availability")
    return {
        "train": data[(dates <= t) & (end <= t)],
        "validation": data[(dates > t) & (dates <= v) & (end <= v)],
        "refit": data[(dates <= v) & (end <= v)],
        "evaluation": data[(dates > v) & (dates <= e) & (end <= e)],
        "evaluation_unmatured": data[(dates > v) & (dates <= e) & (end > e)],
    }


def choose_c(train, validation, features, horizon, config):
    if len(validation) == 0:
        raise ValueError("empty inner validation")
    trials = []
    for C in config["C_grid"]:
        model = fit_model(train, features, horizon, C, config["random_seed"])
        probability = model.predict_proba(validation[features])[:,1]
        trials.append({"C": C, **score(validation[f"target_{horizon}"], probability, config["threshold"])})
    best = min(trials, key=lambda x: (x["brier"], x["C"]))
    return best["C"], trials


def prediction_rows(data, probability, horizon, scenario, fold, mask, C, threshold):
    result = data[["date", f"label_end_date_{horizon}", f"target_{horizon}"]].rename(columns={f"label_end_date_{horizon}":"label_end_date", f"target_{horizon}":"actual"}).copy()
    result["scenario"], result["fold"], result["horizon"] = scenario, fold, horizon
    result["mask"], result["model"], result["C"] = mask, model_name(mask), C
    result["probability_up"] = probability
    result["predicted"] = (np.asarray(probability) >= threshold).astype(int)
    return result


def fit_predict_save(refit, evaluation, h, features, C, config, model_path):
    model = fit_model(refit, features, h, C, config["random_seed"])
    probability = model.predict_proba(evaluation[features])[:,1]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    restored = joblib.load(model_path).predict_proba(evaluation[features])[:,1]
    if not np.allclose(probability, restored, atol=1e-12, rtol=0):
        raise ValueError("serialized prediction differs")
    return probability


def bootstrap_indices(folds, repetitions, block_length, seed):
    """Paired circular blocks within each fold, never across a fold boundary."""
    if repetitions < 1 or block_length < 1 or not len(folds):
        raise ValueError("invalid bootstrap settings")
    folds = np.asarray(folds)
    rng = np.random.default_rng(seed)
    pieces = []
    for fold in pd.unique(folds):
        positions = np.flatnonzero(folds == fold)
        n = len(positions)
        starts = rng.integers(0,n,size=(repetitions,int(np.ceil(n/block_length))))
        index = ((starts[:,:,None] + np.arange(block_length)) % n).reshape(repetitions,-1)[:,:n]
        pieces.append(positions[index])
    return np.concatenate(pieces, axis=1)


def mean_marginal_contributions(losses):
    """Average retrained-group loss improvements over the six addition orders."""
    if set(losses) != set(range(8)):
        raise ValueError("all eight subsets required")
    arrays = {m:np.asarray(x, dtype=float) for m,x in losses.items()}
    if len({x.shape for x in arrays.values()}) != 1:
        raise ValueError("unaligned loss arrays")
    contribution = np.zeros((len(arrays[0]),3))
    for order in permutations(range(3)):
        mask = 0
        for group in order:
            added = mask | (1 << group)
            contribution[:,group] += (arrays[mask]-arrays[added])/6
            mask = added
    if not np.allclose(contribution.sum(axis=1), arrays[0]-arrays[7], atol=1e-12,rtol=0):
        raise ValueError("attribution efficiency failed")
    return contribution


def aligned_predictions(part):
    part = part[part["mask"] >= 0]
    key = ["fold","date"]
    if part.duplicated(key+["mask"]).any():
        raise ValueError("duplicate paired prediction")
    if part.groupby(key).actual.nunique().ne(1).any():
        raise ValueError("targets disagree across groups")
    table = part.pivot(index=key,columns="mask",values="probability_up").sort_index()
    if set(table.columns) != set(range(8)) or table.isna().any().any():
        raise ValueError("groups do not share evaluation dates")
    target = part.groupby(key).actual.first().reindex(table.index).to_numpy(int)
    return table, target


def summarize(predictions, config):
    fold_metrics, pooled = [], []
    for (scenario,h,fold,mask), rows in predictions.groupby(["scenario","horizon","fold","mask"]):
        fold_metrics.append({"scenario":scenario,"horizon":h,"fold":fold,"mask":mask,"model":model_name(mask),**score(rows.actual,rows.probability_up,config["threshold"])})
    for (scenario,h,mask), rows in predictions.groupby(["scenario","horizon","mask"]):
        pooled.append({"scenario":scenario,"horizon":h,"mask":mask,"model":model_name(mask),**score(rows.actual,rows.probability_up,config["threshold"])})
    contrasts, contrast_folds, attributions, attribution_folds = [], [], [], []
    for (scenario,h), part in predictions.groupby(["scenario","horizon"]):
        table, y = aligned_predictions(part)
        folds = table.index.get_level_values("fold").to_numpy()
        indices = bootstrap_indices(folds,config["bootstrap_repetitions"],config["bootstrap_block_length"],config["random_seed"]+int(h))
        losses = {m:(table[m].to_numpy()-y)**2 for m in range(8)}
        correct = {m:((table[m].to_numpy() >= config["threshold"]).astype(int)==y).astype(float) for m in range(8)}
        for base,added,name in COMPARISONS:
            gain = losses[base]-losses[added]
            accuracy_delta = correct[added]-correct[base]
            bg = np.quantile(gain[indices].mean(axis=1),[0.025,0.975])
            ba = np.quantile(accuracy_delta[indices].mean(axis=1),[0.025,0.975])
            wins = 0
            for fold in pd.unique(folds):
                mask = folds==fold
                value = float(gain[mask].mean())
                wins += int(value > 0)
                contrast_folds.append({"scenario":scenario,"horizon":h,"fold":fold,"comparison":name,"n":int(mask.sum()),"brier_gain":value,"accuracy_delta":float(accuracy_delta[mask].mean())})
            contrasts.append({"scenario":scenario,"horizon":h,"comparison":name,"base":model_name(base),"added":model_name(added),"n":len(y),"brier_gain":float(gain.mean()),"brier_gain_low":float(bg[0]),"brier_gain_high":float(bg[1]),"accuracy_delta":float(accuracy_delta.mean()),"accuracy_delta_low":float(ba[0]),"accuracy_delta_high":float(ba[1]),"improved_folds":wins,"total_folds":len(pd.unique(folds))})
        values = mean_marginal_contributions(losses)
        for i,group in enumerate(GROUPS):
            interval = np.quantile(values[indices,i].mean(axis=1),[0.025,0.975])
            attributions.append({"scenario":scenario,"horizon":h,"group":group,"n":len(y),"brier_contribution":float(values[:,i].mean()),"low":float(interval[0]),"high":float(interval[1])})
            for fold in pd.unique(folds):
                mask = folds==fold
                attribution_folds.append({"scenario":scenario,"horizon":h,"fold":fold,"group":group,"n":int(mask.sum()),"brier_contribution":float(values[mask,i].mean())})
    return {"fold_metrics.csv":fold_metrics,"pooled_metrics.csv":pooled,"contrasts.csv":contrasts,"fold_contrasts.csv":contrast_folds,"attribution.csv":attributions,"fold_attribution.csv":attribution_folds}


def run_stability(root: Path, run_id: str):
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id):
        raise ValueError("invalid run id")
    if subprocess.check_output(["git","status","--porcelain"],cwd=root,text=True).strip():
        raise ValueError("commit frozen code and config before scoring")
    if not audit_project(root)["ok"]:
        raise ValueError("input audit failed")
    config_path = root/"configs/stability_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset = safe_path(root,config["dataset"])
    metadata = json.loads((dataset/"metadata.json").read_text(encoding="utf-8"))
    for name,expected in metadata["outputs"].items():
        if sha256(safe_path(dataset,name)) != expected:
            raise ValueError("dataset changed")
    baseline = safe_path(root,config["baseline_run"])
    old_run = json.loads((baseline/"run.json").read_text(encoding="utf-8"))
    for name,expected in old_run["artifacts"].items():
        if sha256(safe_path(baseline,name)) != expected:
            raise ValueError("old baseline changed")
    frame = pd.read_csv(dataset/"daily_samples.csv")
    if frame.date.duplicated().any() or not frame.date.is_monotonic_increasing:
        raise ValueError("sample dates not sorted and unique")
    out = root/"outputs"/run_id
    out.mkdir(parents=True,exist_ok=False)
    (out/"config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    predictions, splits, trials, settings, split_dates = [], [], [], [], []
    for fold in config["folds"]:
        for h in config["horizons"]:
            parts = fold_parts(frame,h,fold)
            for stage,data in parts.items():
                splits.append({"fold":fold["name"],"horizon":h,"stage":stage,"n":len(data),"first_date":data.date.min(),"last_date":data.date.max(),"latest_label_end":data[f"label_end_date_{h}"].max()})
                for row in data[["date",f"label_end_date_{h}"]].to_dict("records"):
                    split_dates.append({"fold":fold["name"],"horizon":h,"stage":stage,**row})
            for mask in range(8):
                features = features_for(mask,config["groups"])
                chosen, candidate_trials = choose_c(parts["train"],parts["validation"],features,h,config)
                trials.extend({"fold":fold["name"],"horizon":h,"mask":mask,"model":model_name(mask),**r} for r in candidate_trials)
                for scenario,C in [("rolling_nested",chosen),("rolling_fixed",config["fixed_C"])]:
                    model_path = out/"models"/scenario/fold["name"]/f"{model_name(mask)}_{h}d.joblib"
                    probability = fit_predict_save(parts["refit"],parts["evaluation"],h,features,C,config,model_path)
                    predictions.append(prediction_rows(parts["evaluation"],probability,h,scenario,fold["name"],mask,C,config["threshold"]))
                    settings.append({"scenario":scenario,"fold":fold["name"],"horizon":h,"mask":mask,"C":C,"features":features})
            prior = float(parts["refit"][f"target_{h}"].mean())
            for scenario in ["rolling_nested","rolling_fixed"]:
                predictions.append(prediction_rows(parts["evaluation"],np.full(len(parts["evaluation"]),prior),h,scenario,fold["name"],-1,None,config["threshold"]))
            print(f"{fold['name']} h={h}: refit={len(parts['refit'])}, evaluation={len(parts['evaluation'])}, tail excluded={len(parts['evaluation_unmatured'])}",flush=True)
    retro = config["retrospective"]
    old_predictions = pd.read_csv(baseline/"test_predictions.csv")
    for h in config["horizons"]:
        eligible = frame[frame.eligible & frame[f"target_{h}"].notna()]
        refit = eligible[(eligible.date <= retro["refit_end"]) & (eligible[f"label_end_date_{h}"] <= retro["refit_end"])]
        evaluation = eligible[(eligible.date > retro["refit_end"]) & (eligible.date <= retro["evaluation_end"])]
        splits.append({"fold":retro["name"],"horizon":h,"stage":"retrospective_refit","n":len(refit),"first_date":refit.date.min(),"last_date":refit.date.max(),"latest_label_end":refit[f"label_end_date_{h}"].max()})
        splits.append({"fold":retro["name"],"horizon":h,"stage":"retrospective_evaluation","n":len(evaluation),"first_date":evaluation.date.min(),"last_date":evaluation.date.max(),"latest_label_end":evaluation[f"label_end_date_{h}"].max()})
        for mask in range(8):
            features = features_for(mask,config["groups"])
            path = out/"models"/"retrospective_fixed"/retro["name"]/f"{model_name(mask)}_{h}d.joblib"
            probability = fit_predict_save(refit,evaluation,h,features,config["fixed_C"],config,path)
            if mask in (0,7):
                name = "market" if mask==0 else "market_sentiment"
                original = old_predictions[(old_predictions.horizon==h)&(old_predictions.model==name)]
                if original.date.tolist()!=evaluation.date.tolist() or not np.allclose(probability,original.probability_up,atol=1e-12,rtol=0):
                    raise ValueError("retrospective does not reproduce original baseline")
            predictions.append(prediction_rows(evaluation,probability,h,"retrospective_fixed",retro["name"],mask,config["fixed_C"],config["threshold"]))
        prior = float(refit[f"target_{h}"].mean())
        predictions.append(prediction_rows(evaluation,np.full(len(evaluation),prior),h,"retrospective_fixed",retro["name"],-1,None,config["threshold"]))
    predictions = pd.concat(predictions,ignore_index=True)
    predictions.to_csv(out/"predictions.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(splits).to_csv(out/"split_audit.csv",index=False)
    pd.DataFrame(split_dates).to_csv(out/"split_membership.csv",index=False)
    pd.DataFrame(trials).to_csv(out/"inner_trials.csv",index=False)
    (out/"selected_settings.json").write_text(json.dumps(settings,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    for filename,rows in summarize(predictions,config).items():
        pd.DataFrame(rows).to_csv(out/filename,index=False)
    run = {"code_commit":git_revision(root),"run_id":run_id,"config_sha256":sha256(config_path),"sample_sha256":sha256(dataset/"daily_samples.csv"),"dataset_metadata_sha256":sha256(dataset/"metadata.json"),"baseline_run_sha256":sha256(baseline/"run.json"),"groups":config["groups"],"original_predictions_reproduced":True,"interpretation":"historical development diagnostics and retrospective decomposition; no new independent holdout or causal attribution", "dependencies":{p:importlib.metadata.version(p) for p in ["numpy","pandas","scikit-learn","scipy","joblib"]},"artifacts":{p.relative_to(out).as_posix():sha256(p) for p in sorted(out.rglob("*")) if p.is_file()}}
    (out/"run.json").write_text(json.dumps(run,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return {"run_id":run_id,"prediction_rows":len(predictions),"models":len(list((out/"models").rglob("*.joblib"))),"original_predictions_reproduced":True}
