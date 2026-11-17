import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from emo_agent.dataset import MARKET, SENTIMENT
from emo_agent.stability import features_for, fold_parts, choose_c, bootstrap_indices, mean_marginal_contributions, aligned_predictions, summarize


def config():
    return json.loads((Path(__file__).resolve().parents[1]/"configs/stability_v1.json").read_text())


def sample():
    dates = pd.bdate_range("2023-11-28","2026-06-30")
    rng = np.random.default_rng(41)
    frame = pd.DataFrame(rng.normal(size=(len(dates),len(MARKET+SENTIMENT))),columns=MARKET+SENTIMENT)
    frame["date"] = dates.strftime("%Y-%m-%d")
    frame["eligible"] = True
    for h in [1,5]:
        frame[f"target_{h}"] = rng.integers(0,2,len(dates))
        frame[f"label_end_date_{h}"] = (dates+pd.offsets.BDay(h)).strftime("%Y-%m-%d")
    return frame


def test_feature_groups_partition_and_preserve_original_order():
    groups = config()["groups"]
    assert features_for(0,groups)==MARKET
    assert features_for(7,groups)==MARKET+SENTIMENT
    assert features_for(1,groups)==MARKET+groups["quality"]
    assert features_for(4,groups)==MARKET+["post_heat_20"]
    groups["quality"].append("target_1")
    with pytest.raises(ValueError): features_for(1,groups)


def test_all_fold_boundaries_require_mature_labels_and_do_not_overlap():
    frame=sample();all_dates=[]
    for fold in config()["folds"]:
        parts=fold_parts(frame,5,fold)
        for name,cutoff in [("train",fold["train_end"]),("validation",fold["validation_end"]),("refit",fold["validation_end"]),("evaluation",fold["evaluation_end"])]:
            assert parts[name].label_end_date_5.max()<=cutoff
        assert parts["evaluation"].date.min()>fold["validation_end"]
        assert (parts["evaluation_unmatured"].label_end_date_5>fold["evaluation_end"]).all()
        assert set(parts["refit"].date).isdisjoint(parts["evaluation"].date)
        all_dates+=parts["evaluation"].date.tolist()
    assert len(all_dates)==len(set(all_dates))
    assert max(all_dates)<="2025-12-31"


def test_future_and_outer_evaluation_mutations_cannot_change_selection_or_refit_inputs():
    frame=sample();cfg=config();fold=cfg["folds"][0]
    before=fold_parts(frame,1,fold)
    first,trials=choose_c(before["train"],before["validation"],MARKET+SENTIMENT,1,cfg)
    mask=frame.date>fold["validation_end"]
    frame.loc[mask,MARKET+SENTIMENT]=1e6
    frame.loc[mask,"target_1"]=1-frame.loc[mask,"target_1"]
    after=fold_parts(frame,1,fold)
    second,new_trials=choose_c(after["train"],after["validation"],MARKET+SENTIMENT,1,cfg)
    assert first==second and trials==new_trials
    pd.testing.assert_frame_equal(before["refit"],after["refit"])


def test_order_averaged_contributions_allocate_interaction_symmetrically():
    individual=np.array([0.01,0.02,-0.005])
    losses={}
    for mask in range(8):
        gain=sum(individual[i] for i in range(3) if mask&(1<<i))
        if mask&1 and mask&2:gain+=0.03
        losses[mask]=np.full(12,0.25-gain)
    values=mean_marginal_contributions(losses)
    np.testing.assert_allclose(values, np.tile([0.025,0.035,-0.005],(12,1)),atol=1e-12)
    np.testing.assert_allclose(values.sum(axis=1),losses[0]-losses[7],atol=1e-12)
    with pytest.raises(ValueError):mean_marginal_contributions({0:losses[0]})


def test_paired_bootstrap_keeps_fold_counts_and_never_crosses_fold_boundary():
    folds=np.array(["Q1"]*7+["Q2"]*5)
    index=bootstrap_indices(folds,100,3,42)
    assert index.shape==(100,12)
    assert (folds[index[:,:7]]=="Q1").all()
    assert (folds[index[:,7:]]=="Q2").all()
    np.testing.assert_array_equal(index,bootstrap_indices(folds,100,3,42))


def predictions():
    rows=[]
    for date in pd.date_range("2025-01-01",periods=24):
        for mask in range(8):
            rows.append({"fold":"Q1" if date.day<=12 else "Q2","date":date.strftime("%Y-%m-%d"),"mask":mask,"actual":date.day%2,"probability_up":0.5+0.005*mask,"scenario":"synthetic","horizon":1})
    return pd.DataFrame(rows)


def test_unpaired_dates_and_disagreeing_labels_fail():
    frame=predictions()
    with pytest.raises(ValueError):aligned_predictions(frame.iloc[1:])
    frame.loc[0,"actual"]=1-frame.loc[0,"actual"]
    with pytest.raises(ValueError):aligned_predictions(frame)


def test_pooled_attribution_equals_total_improvement_and_same_dates():
    cfg=config();cfg["bootstrap_repetitions"]=40
    output=summarize(predictions(),cfg)
    total=next(x for x in output["contrasts.csv"] if x["comparison"]=="all_added")
    contribution=sum(x["brier_contribution"] for x in output["attribution.csv"])
    assert abs(total["brier_gain"]-contribution)<1e-12
    assert total["n"]==24 and total["total_folds"]==2
    assert all(x["n"]==24 for x in output["pooled_metrics.csv"])
