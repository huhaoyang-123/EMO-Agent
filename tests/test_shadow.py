import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from emo_agent.dataset import MARKET
from emo_agent.shadow import model_features, _mature


def config():
    return json.loads((Path(__file__).resolve().parents[1] / "configs/shadow_v1.json").read_text())


def test_shadow_models_are_market_only_and_known():
    c = config()
    assert model_features("M", c) == MARKET
    assert model_features("MGB", c) == MARKET + c["groups"]["G"] + c["groups"]["B"]
    with pytest.raises(ValueError):
        model_features("MGS", c)  # sentiment models are not frozen for new dates


def test_mature_rows_never_include_labels_ending_after_boundary():
    dates = pd.bdate_range("2025-12-20", "2026-01-10").strftime("%Y-%m-%d")
    f = pd.DataFrame({"date": dates, "market_complete": True, "target_5": 1})
    f["label_end_date_5"] = pd.Series(dates).shift(-5).fillna("2026-02-01").values
    part = _mature(f, 5, "0000", "2025-12-31")
    assert (part.label_end_date_5 <= "2025-12-31").all() and (part.date <= "2025-12-31").all()
    assert len(part) == len(f[(f.date <= "2025-12-31") & (f.label_end_date_5 <= "2025-12-31")])
