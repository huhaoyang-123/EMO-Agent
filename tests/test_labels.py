import pandas as pd
import pytest
from emo_agent.labels import future_labels, mature_labels


def prices():
    return pd.DataFrame({"date": ["2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09", "2025-01-10", "2025-01-13"], "close": [100, 101, 101, 99, 102, 103, 100]})


def test_one_day_uses_trading_rows_and_keeps_missing_tail():
    result = future_labels(prices(), 1)
    assert result.target.iloc[:6].tolist() == [1, 0, 0, 1, 1, 0]
    assert pd.isna(result.target.iloc[-1])
    assert result.label_end_date.iloc[0] == pd.Timestamp("2025-01-06")


def test_five_day_endpoint_and_unknown_tail():
    result = future_labels(prices(), 5)
    assert result.target.iloc[:2].tolist() == [1, 0]
    assert result.target.iloc[2:].isna().all()
    assert result.label_end_date.iloc[0] == pd.Timestamp("2025-01-10")


def test_training_boundary_excludes_unobserved_future():
    result = mature_labels(future_labels(prices(), 5), "2025-01-10")
    assert len(result) == 1
    assert result.date.iloc[0] == pd.Timestamp("2025-01-03")


@pytest.mark.parametrize("horizon", [0, -1, 1.5, True])
def test_invalid_horizon(horizon):
    with pytest.raises(ValueError):
        future_labels(prices(), horizon)


@pytest.mark.parametrize("kind", ["duplicate", "unsorted", "zero", "missing", "infinite"])
def test_invalid_prices(kind):
    frame = prices()
    if kind == "duplicate": frame.loc[1, "date"] = frame.loc[0, "date"]
    elif kind == "unsorted": frame = frame.iloc[::-1]
    elif kind == "zero": frame.loc[0, "close"] = 0
    elif kind == "missing": frame.loc[0, "close"] = float("nan")
    elif kind == "infinite": frame = frame.astype({"close": float}); frame.loc[0, "close"] = float("inf")
    with pytest.raises(ValueError):
        future_labels(frame, 1)
