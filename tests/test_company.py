import json
from pathlib import Path
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
import pytest

from emo_agent.company import (ARMS, PanelLogit, aggregate_intervals, contrasts,
    daily_losses, daily_weights, features_for, membership_mask, metrics,
    sentiment_history, split_panel, stock_history, validate_stocks)


def config():
    return json.loads((Path(__file__).resolve().parents[1] / "configs/company_pilot_v1.json").read_text(encoding="utf-8"))


def stocks_fixture():
    calendar = pd.bdate_range("2024-01-01", periods=35).strftime("%Y-%m-%d").tolist()
    rows = []
    for code, scale in [("000001", 1), ("000002", 3)]:
        for i, date in enumerate(calendar):
            rows.append(dict(Stkcd=code, Trddt=date, Clsprc=10 * scale, Adjprcnd=(10+i/10)*scale,
                             Dnshrtrd=1000+i, Dnvaltrd=10000, Dsmvosd=1000, Markettype=4))
    return pd.DataFrame(rows), calendar


def intervals_fixture():
    rows = []
    for code in ["000001", "000002"]:
        for i, date in enumerate(pd.bdate_range("2024-01-01", periods=30)):
            cutoff = pd.Timestamp(f"{date:%Y-%m-%d} 16:00:00", tz="Asia/Shanghai")
            rows.append(dict(Scode=code, Date=date.strftime("%Y-%m-%d"), Time="9:00-15:00",
                interval_start=shift_time(cutoff, -7), interval_end=shift_time(cutoff, -1),
                cutoff_at=cutoff, observed=True, source_record_id=f"{code}-{i}",
                Tpostnum=10+i, Pospostnum=5+i, Negpostnum=2, Neupostnum=3))
    return pd.DataFrame(rows)


def shift_time(value, hours):
    return pd.Timestamp(value.to_pydatetime() + timedelta(hours=hours))


def test_stock_labels_follow_global_dates_with_missing_and_zero_volume():
    stocks, calendar = stocks_fixture()
    stocks = stocks[~(stocks.Stkcd.eq("000001") & stocks.Trddt.eq(calendar[1]))].copy()
    stocks.loc[stocks.Stkcd.eq("000001") & stocks.Trddt.eq(calendar[5]), "Dnshrtrd"] = 0
    f = stock_history(stocks, calendar, [1, 5]).set_index(["code", "date"])
    assert pd.isna(f.loc[("000001", calendar[0]), "target_1"])
    assert f.loc[("000001", calendar[0]), "label_end_date_1"] == calendar[1]
    assert pd.isna(f.loc[("000001", calendar[0]), "target_5"])
    assert f.loc[("000002", calendar[0]), "target_5"] == 1
    assert pd.isna(f.loc[("000002", calendar[-1]), "target_1"])


def test_flat_target_turnover_and_past_only_company_features():
    stocks, calendar = stocks_fixture()
    stocks.loc[stocks.Trddt.eq(calendar[1]), "Adjprcnd"] = stocks.loc[stocks.Trddt.eq(calendar[0]), "Adjprcnd"].to_numpy()
    before = stock_history(stocks, calendar, [1, 5])
    a = before[before.code.eq("000001")]
    assert a.iloc[0].target_1 == 0
    assert a.iloc[0].turnover_pct == 1  # 1000 shares / 100000 tradable shares.
    stocks.loc[stocks.Stkcd.eq("000002"), "Adjprcnd"] *= 123
    stocks.loc[stocks.Trddt.gt(calendar[25]), "Dnshrtrd"] *= 10000
    after = stock_history(stocks, calendar, [1, 5])
    cols = ["stock_return_1", "stock_return_5", "stock_return_20", "volume_heat_20", "log_turnover"]
    pd.testing.assert_frame_equal(before.loc[before.code.eq("000001") & before.date.le(calendar[25]), cols],
                                  after.loc[after.code.eq("000001") & after.date.le(calendar[25]), cols])


@pytest.mark.parametrize("mutation", ["duplicate", "code", "units", "negative"])
def test_stock_validation(mutation):
    stocks, _ = stocks_fixture()
    if mutation == "duplicate": stocks = pd.concat([stocks, stocks.iloc[:1]])
    if mutation == "code": stocks.loc[0, "Stkcd"] = "1"
    if mutation == "units": stocks.loc[0, "Markettype"] = 2
    if mutation == "negative": stocks.loc[0, "Dsmvosd"] = -1
    with pytest.raises(ValueError): validate_stocks(stocks)


def test_sentiment_missing_is_not_zero_and_hours_are_observed():
    f = intervals_fixture().iloc[:2].copy()
    f.loc[1, "Date"] = f.loc[0, "Date"]
    f.loc[1, "Time"] = "0:00-9:00"
    f.loc[1, "cutoff_at"] = f.loc[0, "cutoff_at"]
    f.loc[1, "interval_start"] = shift_time(f.loc[0, "cutoff_at"], -16)
    f.loc[1, "interval_end"] = f.loc[0, "interval_start"]
    f.loc[1, "observed"] = False
    f.loc[1, ["Tpostnum", "Pospostnum", "Negpostnum", "Neupostnum", "source_record_id"]] = np.nan
    row = aggregate_intervals(f).iloc[0]
    assert row.observed_hours == 6 and row.expected_hours == 15
    assert row.slot_coverage == 0.5 and row.hour_coverage == 0.4
    assert row.post_rate == 10/6 and row.Tpostnum == 10
    f.loc[:, "observed"] = False
    f.loc[:, ["Tpostnum", "Pospostnum", "Negpostnum", "Neupostnum", "source_record_id"]] = np.nan
    row = aggregate_intervals(f).iloc[0]
    assert pd.isna(row.Tpostnum) and pd.isna(row.net_sentiment) and row.sentiment_missing == 1


@pytest.mark.parametrize("mutation", ["future", "source", "counts", "missing_counts"])
def test_interval_corruption_is_rejected(mutation):
    f = intervals_fixture()
    if mutation == "future": f.loc[0, "interval_end"] = shift_time(f.loc[0, "cutoff_at"], 1)
    if mutation == "source": f.loc[1, "source_record_id"] = f.loc[0, "source_record_id"]
    if mutation == "counts": f.loc[0, "Pospostnum"] = 999
    if mutation == "missing_counts": f.loc[0, "observed"] = False
    with pytest.raises(ValueError): aggregate_intervals(f)


def test_sentiment_history_has_no_future_or_other_company_leakage():
    f = intervals_fixture()
    daily = aggregate_intervals(f)
    calendar = sorted(daily.date.unique())
    before = sentiment_history(daily, calendar)
    after_source = f.copy()
    mask = after_source.Scode.eq("000002") | after_source.Date.gt(calendar[25])
    after_source.loc[mask, "Tpostnum"] += 100000
    after_source.loc[mask, "Pospostnum"] += 100000
    after = sentiment_history(aggregate_intervals(after_source), calendar)
    cols = ["net_sentiment", "sentiment_change_5", "post_heat_20", "log_post_rate"]
    keep = before.code.eq("000001") & before.date.le(calendar[25])
    pd.testing.assert_frame_equal(before.loc[keep, cols], after.loc[keep, cols])
    a = before[before.code.eq("000001")].reset_index(drop=True)
    assert np.isclose(a.loc[20, "sentiment_change_5"], a.loc[20, "net_sentiment"]-a.loc[15:19, "net_sentiment"].mean())


def test_membership_is_prediction_date_known():
    members = pd.DataFrame([dict(stkcd="000001", in_date="2024-01-02", out_date="2024-01-04")])
    f = pd.DataFrame({"code": ["000001"]*4, "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]})
    assert membership_mask(f, members).tolist() == [False, True, True, False]


def model_frame():
    rows = []
    for i, date in enumerate(pd.bdate_range("2024-01-01", periods=30)):
        for code in ["000001", "000002"]:
            rows.append(dict(date=date.strftime("%Y-%m-%d"), code=code, x=float(i % 7), eligible=True,
                             target_1=i % 2, label_end_date_1=(date+pd.offsets.BDay()).strftime("%Y-%m-%d")))
    return pd.DataFrame(rows)


def test_shared_date_split_purges_unmatured_labels():
    f = model_frame()
    fold = dict(train_end="2024-01-15", evaluation_end="2024-01-31")
    train, test = split_panel(f, 1, fold)
    assert (train.label_end_date_1 <= fold["train_end"]).all()
    assert (test.label_end_date_1 <= fold["evaluation_end"]).all()
    assert not set(train.date) & set(test.date)
    assert train.groupby("date").code.nunique().eq(2).all()
    assert test.groupby("date").code.nunique().eq(2).all()


def test_preprocessing_train_only_unknown_company_and_serialization(tmp_path):
    f = model_frame()
    train, test = f.iloc[:40].copy(), f.iloc[40:].copy()
    train.loc[0, "x"] = np.nan
    model = PanelLogit(["x"], True).fit(train, train.target_1)
    assert model.imputer.statistics_[0] == train.x.median()
    assert np.isclose(model.training_weight_sum, train.date.nunique())
    before = model.scaler.mean_.copy()
    test["x"] = 1000000
    test["code"] = "999999"
    p = model.predict_proba(test)
    assert np.isfinite(p).all()
    np.testing.assert_array_equal(before, model.scaler.mean_)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    np.testing.assert_array_equal(p, joblib.load(path).predict_proba(test))


def test_daily_equal_weights_and_metrics_do_not_overweight_busy_dates():
    f = pd.DataFrame(dict(date=["2025-01-02"] + ["2025-01-03"]*9,
                          code=[str(i) for i in range(10)], actual=[1]*10, probability_up=[1]+[0]*9))
    assert np.isclose(daily_weights(f).sum(), 2)
    m = metrics(f)
    assert np.isclose(m["accuracy"], 0.5) and m["row_accuracy"] == 0.1
    assert np.isclose(m["brier"], 0.5)


def test_feature_arms_nested_and_no_labels_in_inputs():
    cfg = config()
    for universe in ["company", "index"]:
        a, b, c = [features_for(cfg, universe, arm) for arm in ARMS]
        assert b == a + cfg["attention"] and c == b + cfg["direction"]
        assert all(not x.startswith(("target", "return_forward", "label_end")) for x in c)


def prediction_fixture():
    rows = []
    for universe, codes in [("company", ["000001", "000002"]), ("index", ["INDEX"])]:
        for fold, period, prefix in [("2025Q1", "primary_2025", "2025-01"), ("2026Q1", "diagnostic_2026H1", "2026-01")]:
            for i in range(10):
                for code in codes:
                    for arm, p in [("prior", .5), (ARMS[0], .5), (ARMS[1], .6), (ARMS[2], .7 if universe == "company" else .6)]:
                        rows.append(dict(universe=universe, code=code, fold=fold, period=period, date=f"{prefix}-{i+1:02d}", horizon=1, model=arm, actual=1, probability_up=p))
    return pd.DataFrame(rows)


def test_paired_daily_gains_and_between_universe_difference():
    f = prediction_fixture()
    cfg = config(); cfg.update(horizons=[1], bootstrap_repetitions=50)
    summary, _ = contrasts(daily_losses(f), cfg)
    selected = summary[(summary.base == ARMS[1]) & (summary.added == ARMS[2])]
    assert np.allclose(selected[selected.universe == "company"].gain, .07)
    assert np.allclose(selected[selected.universe == "index"].gain, 0)
    assert np.allclose(selected[selected.universe == "company_minus_index"].low, .07)
    with pytest.raises(ValueError, match="unpaired model"):
        daily_losses(f.iloc[1:])


def test_all_companies_on_day_stay_together_in_bootstrap_loss():
    f = prediction_fixture()
    before = daily_losses(f)
    more = f[f.universe.eq("company")].copy(); more["code"] += "copy"
    after = daily_losses(pd.concat([f, more]))
    pd.testing.assert_series_equal(before.brier, after.brier)
