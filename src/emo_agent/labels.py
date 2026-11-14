"""Trading-row targets with explicit label availability."""
import pandas as pd


def future_labels(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    dates = pd.to_datetime(prices["date"], errors="raise")
    close = pd.to_numeric(prices["close"], errors="raise")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("dates must be unique, present and sorted")
    if close.isna().any() or not close.between(0, float("inf"), inclusive="neither").all():
        raise ValueError("close must be finite and positive")
    future = close.shift(-horizon)
    label = (future > close).astype("Int64").mask(future.isna())
    return pd.DataFrame({"date": dates, "label_end_date": dates.shift(-horizon), "target": label})


def mature_labels(labels: pd.DataFrame, as_of: str) -> pd.DataFrame:
    """Training at the close of as_of may only use outcomes already observed."""
    mask = labels["target"].notna() & (labels["label_end_date"] <= pd.Timestamp(as_of))
    return labels.loc[mask].copy()
