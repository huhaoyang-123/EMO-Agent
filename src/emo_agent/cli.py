"""Project CLI: audit snapshots, build cutoff samples, evaluate baselines."""
import argparse
import json
from pathlib import Path
from emo_agent.audit import audit_project, safe_path


def main() -> int:
    parser = argparse.ArgumentParser(description="EMO Agent research foundations")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit-data", help="verify local input hashes and counts")
    audit.add_argument("--root", type=Path, default=Path.cwd())
    build = commands.add_parser("build-dataset", help="align intervals and construct daily labels/features")
    build.add_argument("--root", type=Path, default=Path.cwd())
    train = commands.add_parser("train-baseline", help="fit frozen logistic baselines and score held-out dates")
    train.add_argument("--root", type=Path, default=Path.cwd())
    train.add_argument("--run-id", required=True)
    stability = commands.add_parser("analyze-stability", help="rolling group ablation and retrospective attribution")
    stability.add_argument("--root", type=Path, default=Path.cwd())
    stability.add_argument("--run-id", required=True)
    features = commands.add_parser("analyze-features", help="past-only feature selection and constrained nonlinear comparison")
    features.add_argument("--root", type=Path, default=Path.cwd())
    features.add_argument("--run-id", required=True)
    company_build = commands.add_parser("build-company-panel", help="build cutoff-aligned historical company panel")
    company_build.add_argument("--root", type=Path, default=Path.cwd())
    company_run = commands.add_parser("analyze-company", help="frozen company and index attention/direction comparison")
    company_run.add_argument("--root", type=Path, default=Path.cwd())
    company_run.add_argument("--run-id", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "build-company-panel":
        from emo_agent.company import build
        print(json.dumps(build(root), ensure_ascii=False))
        return 0
    if args.command == "analyze-company":
        from emo_agent.company import run
        print(json.dumps(run(root, args.run_id), ensure_ascii=False))
        return 0
    if args.command == "build-dataset":
        from emo_agent.dataset import build_dataset
        print(json.dumps(build_dataset(root), ensure_ascii=False))
        return 0
    if args.command == "train-baseline":
        from emo_agent.baseline import run_baseline
        print(json.dumps(run_baseline(root, args.run_id), ensure_ascii=False))
        return 0
    if args.command == "analyze-stability":
        from emo_agent.stability import run_stability
        print(json.dumps(run_stability(root, args.run_id), ensure_ascii=False))
        return 0
    if args.command == "analyze-features":
        from emo_agent.features import run
        print(json.dumps(run(root, args.run_id), ensure_ascii=False))
        return 0
    report = audit_project(root)
    destination = safe_path(root, "outputs/audits/data_audit.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "report": str(destination), "files": len(report["files"])}, ensure_ascii=False))
    return 0 if report["ok"] else 1
