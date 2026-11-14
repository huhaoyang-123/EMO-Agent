import json
import sys
from emo_agent.cli import main


def test_missing_data_reports_failure(tmp_path, monkeypatch):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "data_manifest.json").write_text(json.dumps({"files": [{"path": "data/missing.csv", "sha256": "0" * 64, "bytes": 1}]}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["emo-agent", "audit-data", "--root", str(tmp_path)])
    assert main() == 1
    report = json.loads((tmp_path / "outputs/audits/data_audit.json").read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert report["files"][0]["status"] == "missing"
