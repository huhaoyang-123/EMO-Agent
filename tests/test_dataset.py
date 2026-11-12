import numpy as np
import pandas as pd
import pytest
from emo_agent.dataset import TZ, MARKET, SENTIMENT, assign_cutoffs, slot_bounds, expected_grid, align_sentiment, add_features


def test_friday_night_weekend_and_holiday_wait_for_next_prediction():
    ends = pd.to_datetime(["2025-01-03 15:00", "2025-01-04 00:00", "2025-01-05 00:00", "2025-01-06 00:00", "2025-01-06 15:00"]).tz_localize(TZ)
    cutoffs = pd.to_datetime(["2025-01-03 16:00", "2025-01-06 16:00"]).tz_localize(TZ)
    got = assign_cutoffs(ends, cutoffs)
    assert got.iloc[0] == cutoffs[0]
    assert got.iloc[1:].eq(cutoffs[1]).all()
    holiday = assign_cutoffs(pd.to_datetime(["2025-01-07 15:00"]).tz_localize(TZ), pd.to_datetime(["2025-01-06 16:00", "2025-01-08 16:00"]).tz_localize(TZ))
    assert holiday.iloc[0].day == 8


def test_exact_cutoff_included_and_after_final_is_not_dropped_silently():
    cutoff = pd.to_datetime(["2025-01-06 16:00"]).tz_localize(TZ)
    got = assign_cutoffs(pd.to_datetime(["2025-01-06 16:00", "2025-01-06 16:01"]).tz_localize(TZ), cutoff)
    assert got.iloc[0] == cutoff[0]
    assert pd.isna(got.iloc[1])


def test_membership_is_half_open():
    members = pd.DataFrame({"stkcd": ["000001", "000002"], "in_date": ["2025-01-03", "2025-01-06"], "out_date": ["2025-01-06", None]})
    grid = expected_grid(members, "2025-01-03", "2025-01-06")
    assert set(grid.loc[grid.Date.eq("2025-01-06"), "Scode"]) == {"000002"}
    assert set(grid.loc[grid.Date.eq("2025-01-05"), "Scode"]) == {"000001"}


def test_unknown_time_slot_fails():
    with pytest.raises(ValueError):
        slot_bounds(pd.Series(["2025-01-06"]), pd.Series(["unknown"]))


def test_missing_is_not_neutral_and_post_counts_reconcile():
    members = pd.DataFrame({"stkcd": ["000001", "000002"], "in_date": ["2025-01-06"]*2, "out_date": [None]*2})
    grid = expected_grid(members, "2025-01-06", "2025-01-07")
    detail = grid.iloc[:1][["Scode", "Date", "Time"]].copy()
    for name, value in {"Tpostnum": 2, "Pospostnum": 2, "Negpostnum": 0, "Neupostnum": 0, "source_record_id": "synthetic-1"}.items(): detail[name] = value
    cutoffs = pd.to_datetime(["2025-01-06 16:00", "2025-01-07 16:00"]).tz_localize(TZ)
    daily, mapping = align_sentiment(detail, grid, cutoffs)
    assert daily.net_sentiment.iloc[0] == 1
    assert daily.slot_coverage.iloc[0] == 0.25
    assert pd.isna(daily.net_sentiment.iloc[1])
    assert pd.isna(daily.Tpostnum.iloc[1])
    assert mapping.loc[mapping.observed, "source_record_id"].is_unique
    assert daily.Tpostnum.sum() == detail.Tpostnum.sum()


def test_features_do_not_change_when_future_inputs_change():
    dates = pd.bdate_range("2025-01-01", periods=45)
    prices = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "close": np.arange(45)+100., "amount_100m": np.arange(45)+50.})
    sentiment = pd.DataFrame({"cutoff_at": pd.to_datetime(dates.strftime("%Y-%m-%d") + " 16:00:00").tz_localize(TZ), "net_sentiment": np.linspace(-0.4,0.4,45), "equal_weight_sentiment": 0.1, "Tpostnum": np.arange(45)+10., "observed_slots": 100, "slot_coverage": 1., "company_coverage": 1., "sentiment_missing": 0})
    original = add_features(prices, sentiment)
    prices.loc[35:, "close"] *= 9
    sentiment.loc[35:, "net_sentiment"] = -1
    sentiment.loc[35:, "Tpostnum"] *= 10
    mutated = add_features(prices, sentiment)
    pd.testing.assert_frame_equal(original.loc[:34, MARKET+SENTIMENT], mutated.loc[:34, MARKET+SENTIMENT])
