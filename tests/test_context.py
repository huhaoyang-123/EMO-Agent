import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from emo_agent.dataset import MARKET, SENTIMENT
from emo_agent.context import market_features, breadth_features, features_for, summarize


def config():
    return json.loads((Path(__file__).resolve().parents[1] / "configs/context_v1.json").read_text())


def test_market_features_use_only_past_and_current_closes():
    dates = pd.bdate_range("2024-01-01", periods=30).strftime("%Y-%m-%d")
    close = pd.Series(np.arange(100.0, 130.0))
    f = market_features(pd.DataFrame({"date": dates, "close": close}))
    assert np.isnan(f.market_return_1.iloc[0])
    assert f.market_return_1.iloc[1] == pytest.approx(101 / 100 - 1)
    assert f.market_volatility_20.iloc[:20].isna().all() and f.market_volatility_20.iloc[20:].notna().all()
    changed = close.copy(); changed.iloc[-1] = 999
    g = market_features(pd.DataFrame({"date": dates, "close": changed}))
    pd.testing.assert_frame_equal(f.iloc[:-1], g.iloc[:-1])  # a future close never changes earlier rows


def test_breadth_respects_half_open_membership_and_skips_missing():
    members = pd.DataFrame({"stkcd": ["A", "B", "C"], "in_date": ["2024-01-01", "2024-01-01", "2024-01-03"],
                            "out_date": ["2024-01-03", None, None]})
    rets = pd.DataFrame({"stkcd": ["A", "B", "C", "A", "B", "C", "B"],
                         "date": ["2024-01-02"] * 3 + ["2024-01-03"] * 3 + ["2024-01-04"],
                         "ret": [0.01, -0.02, 0.05, 0.03, 0.01, -0.01, np.nan]})
    b = breadth_features(rets, members).set_index("date")
    assert b.loc["2024-01-02", "breadth_n"] == 2 and b.loc["2024-01-02", "breadth_up"] == 0.5  # C not yet a member
    assert b.loc["2024-01-03", "breadth_n"] == 2  # A left on its out_date
    assert "2024-01-04" not in b.index  # missing return is not treated as zero


def test_feature_sets_and_invalid_names():
    c = config()
    assert features_for("M", c) == MARKET
    assert features_for("MGBS", c) == MARKET + c["groups"]["G"] + c["groups"]["B"] + SENTIMENT
    for bad in ["G", "MGG", "MX"]:
        with pytest.raises(ValueError):
            features_for(bad, c)


def test_summarize_pairs_models_on_same_dates():
    c = config(); c["bootstrap_repetitions"] = 50
    rng = np.random.default_rng(0); rows = []
    folds = [f["name"] for f in c["folds"]] + ["2026H1"]
    for arm in ["selected_C", "fixed_C"]:
        for h in [1, 5]:
            for fold in folds:
                dates = [f"{fold}-{i}" for i in range(20)]; y = rng.integers(0, 2, 20)
                for m in ["prior"] + c["models"]:
                    p = np.full(20, 0.5) if m in ("prior", "M") else np.clip(0.5 + 0.3 * (y - 0.5), 0, 1)
                    rows.append(pd.DataFrame({"arm": arm, "horizon": h, "fold": fold, "model": m, "date": dates,
                                              "actual": y, "probability_up": p}))
    s = summarize(pd.concat(rows), c)
    g = s["comparisons"].query("comparison=='G_given_M' and arm=='selected_C' and horizon==1").iloc[0]
    assert g.brier_delta_2025_pooled < 0 and g.quarters_improved == 4
    assert s["verdict"].query("comparison=='G_given_M'").helps_by_frozen_rule.all()


def _addendum_config():
    return json.loads((Path(__file__).resolve().parents[1] / "configs/context_v1b.json").read_text())


def test_residuals_fit_only_on_fitting_rows_and_remove_linear_market_part():
    from emo_agent.context import add_residuals
    c = _addendum_config(); rng = np.random.default_rng(3)
    cols = MARKET + c["groups"]["G"] + SENTIMENT
    fit = pd.DataFrame(rng.normal(size=(200, len(cols))), columns=cols)
    fit["net_sentiment"] = 2 * fit["return_1"] + 0.1 * rng.normal(size=200)
    later = pd.DataFrame(rng.normal(size=(50, len(cols))), columns=cols)
    a_fit, a_later = add_residuals(fit, [fit, later], c)
    assert abs(np.corrcoef(a_fit.net_sentiment_resid, fit.return_1)[0, 1]) < 1e-8
    shocked = later.copy(); shocked["net_sentiment"] += 100  # changing later rows must not change fitted coefficients
    _, b_later = add_residuals(fit, [fit, shocked], c)
    np.testing.assert_allclose(b_later.net_sentiment_resid - a_later.net_sentiment_resid, 100)
    assert features_for("MGR", c)[-7:] == [f"{t}_resid" for t in c["residual"]["targets"]] + c["residual"]["coverage_columns_kept_raw"]
    with pytest.raises(ValueError):
        features_for("MGSR", c)


def test_calibration_perfect_and_overconfident():
    from emo_agent.context import calibration
    rng = np.random.default_rng(5); p = rng.uniform(0.05, 0.95, 20000); y = (rng.uniform(size=20000) < p).astype(int)
    good = calibration(y, p)
    assert good["ece_10bin"] < 0.02 and abs(good["calibration_slope"] - 1) < 0.1
    over = calibration(y, np.clip(0.5 + 2 * (p - 0.5), 0.01, 0.99))
    assert over["calibration_slope"] < 0.8
