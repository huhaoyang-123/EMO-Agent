import joblib
import numpy as np
import pandas as pd
from emo_agent.baseline import make_model, select_settings, split_data, score
from emo_agent.dataset import MARKET, SENTIMENT


def config():
    return {"train_end":"2024-12-31", "validation_end":"2025-12-31", "test_end":"2026-06-30", "horizons":[1,5], "C_grid":[0.01,0.1], "random_seed":42, "threshold":0.5}


def data():
    dates = pd.bdate_range("2024-01-01", "2026-06-30")
    rng = np.random.default_rng(3)
    f = pd.DataFrame(rng.normal(size=(len(dates),len(MARKET+SENTIMENT))),columns=MARKET+SENTIMENT)
    f["date"] = dates.strftime("%Y-%m-%d")
    f["cutoff_at"] = f.date + " 16:00:00+08:00"
    f["eligible"] = True
    for h in [1,5]:
        f[f"label_end_date_{h}"] = (dates+pd.offsets.BDay(h)).strftime("%Y-%m-%d")
        f[f"target_{h}"] = rng.integers(0,2,len(dates))
    return f


def test_split_requires_mature_train_and_validation_labels():
    parts = split_data(data(),5,config())
    assert parts["train"].label_end_date_5.max() <= "2024-12-31"
    assert parts["validation"].label_end_date_5.max() <= "2025-12-31"
    assert parts["refit"].label_end_date_5.max() <= "2025-12-31"
    assert parts["test"].date.min() > "2025-12-31"
    assert set(parts["test"].date).isdisjoint(parts["refit"].date)


def test_selection_is_unchanged_by_test_labels_and_features():
    frame = data()
    original, original_trials, _ = select_settings(frame,config())
    mask = frame.date > "2025-12-31"
    frame.loc[mask,MARKET+SENTIMENT] = 1e6
    for h in [1,5]: frame.loc[mask,f"target_{h}"] = 1-frame.loc[mask,f"target_{h}"]
    changed, changed_trials, _ = select_settings(frame,config())
    assert original == changed
    pd.testing.assert_frame_equal(original_trials,changed_trials)


def test_imputation_scaling_and_serialization_use_training_only(tmp_path):
    train = pd.DataFrame({"x":[0.,2.,np.nan,4.],"z":[1.,0.,1.,0.]})
    model = make_model(0.1,42).fit(train,[0,1,0,1])
    assert model.named_steps["impute"].statistics_[0] == 2
    assert model.named_steps["scale"].mean_[0] == 2
    test = pd.DataFrame({"x":[1e6,np.nan],"z":[0.,1.]})
    probability=model.predict_proba(test)
    assert model.named_steps["scale"].mean_[0] == 2
    path=tmp_path/'test.joblib';joblib.dump(model,path)
    np.testing.assert_allclose(joblib.load(path).predict_proba(test),probability,atol=1e-12)


def test_accuracy_and_threshold_boundary():
    result=score([0,1,1],[0.1,0.5,0.4])
    assert result["correct"] == 2
    assert result["accuracy"] == 2/3
    assert (result["tn"],result["fp"],result["fn"],result["tp"]) == (1,0,1,1)
