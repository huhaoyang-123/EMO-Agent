"""Project CLI. Model training is a later issue."""
import argparse
import json
from pathlib import Path
from emo_agent.audit import audit_project, safe_path


def main() -> int:
    parser = argparse.ArgumentParser(description="EMO Agent research foundations")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit-data", help="verify local input hashes and counts")
    audit.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    report = audit_project(root)
    destination = safe_path(root, "outputs/audits/data_audit.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "report": str(destination), "files": len(report["files"])}, ensure_ascii=False))
    return 0 if report["ok"] else 1
