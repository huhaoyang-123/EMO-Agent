import hashlib
import pandas as pd
import pytest
from emo_agent.audit import verify_manifest, validate_sentiment


def test_manifest_detects_missing_and_modified(tmp_path):
    content = b"synthetic input"
    path = tmp_path / "sample.csv"
    manifest = {"files": [{"path": "sample.csv", "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}]}
    assert verify_manifest(tmp_path, manifest)[0]["status"] == "missing"
    path.write_bytes(content)
    assert verify_manifest(tmp_path, manifest)[0]["status"] == "ok"
    path.write_bytes(b"changed")
    assert verify_manifest(tmp_path, manifest)[0]["status"] == "hash_mismatch"


def test_manifest_rejects_path_escape(tmp_path):
    with pytest.raises(ValueError):
        verify_manifest(tmp_path, {"files": [{"path": "../outside"}]})


def sample():
    return pd.DataFrame([{"Scode": "000070", "Date": "2025-01-06", "Time": "9:00-15:00", "Tpostnum": 3, "Pospostnum": 1, "Negpostnum": 1, "Neupostnum": 1}])


def test_valid_counts_and_leading_zero():
    validate_sentiment(sample())


@pytest.mark.parametrize("kind", ["duplicate", "code", "balance", "negative", "fractional"])
def test_sentiment_rejects_corruption(kind):
    frame = sample()
    if kind == "duplicate": frame = pd.concat([frame, frame])
    elif kind == "code": frame.loc[0, "Scode"] = "70"
    elif kind == "balance": frame.loc[0, "Tpostnum"] = 4
    elif kind == "negative": frame.loc[0, "Pospostnum"] = -1
    elif kind == "fractional": frame = frame.astype({"Tpostnum": float}); frame.loc[0, "Tpostnum"] = 3.5
    with pytest.raises(ValueError):
        validate_sentiment(frame)
