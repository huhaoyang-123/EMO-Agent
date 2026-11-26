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
    bctx = commands.add_parser("build-context", help="task 3: broad-market context and constituent breadth features")
    bctx.add_argument("--root", type=Path, default=Path.cwd())
    actx = commands.add_parser("analyze-context", help="task 3: M/MG/MS/MGS/MGB/MGBS rolling comparison")
    actx.add_argument("--root", type=Path, default=Path.cwd())
    actx.add_argument("--run-id", required=True)
    actx.add_argument("--config", default="configs/context_v1.json")
    company_build = commands.add_parser("build-company-panel", help="build cutoff-aligned historical company panel")
    company_build.add_argument("--root", type=Path, default=Path.cwd())
    company_run = commands.add_parser("analyze-company", help="frozen company and index attention/direction comparison")
    company_run.add_argument("--root", type=Path, default=Path.cwd())
    company_run.add_argument("--run-id", required=True)
    commands.add_parser("shadow-freeze", help="task 6: fit and freeze market-only models on data up to 2026-06-30").add_argument("--root", type=Path, default=Path.cwd())
    sp = commands.add_parser("shadow-predict", help="task 6: append frozen-model probabilities for one new date")
    sp.add_argument("--root", type=Path, default=Path.cwd())
    sp.add_argument("--date", required=True)
    sp.add_argument("--mode", default="daily", choices=["daily", "retro_batch"])
    se = commands.add_parser("shadow-evaluate", help="task 6: score logged predictions whose labels have matured")
    se.add_argument("--root", type=Path, default=Path.cwd())
    se.add_argument("--start")
    se.add_argument("--end")
    agent = commands.add_parser("agent", help="ask DeepSeek to analyze the local historical data")
    agent.add_argument("question")
    direct = commands.add_parser("agent-tool", help="run a read-only analysis tool without an API key")
    direct.add_argument("tool")
    direct.add_argument("arguments", help="tool parameters as a JSON object")
    check = commands.add_parser("agent-check", help="verify local inputs and display analysis coverage")
    for command in (agent, direct, check):
        command.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()
    if args.command in {"agent", "agent-tool", "agent-check"}:
        import os
        from emo_agent.analysis.loader import config_at, load_analysis_data

        root = (args.root or Path(os.getenv("EMO_AGENT_DATA_ROOT", Path.cwd()))).resolve()
        try:
            if not audit_project(root)["ok"]:
                raise ValueError("基础输入缺失或审计失败，请按 data/README.md 检查。")
            data = load_analysis_data(root)
            if args.command == "agent-check":
                print(json.dumps(data.availability, ensure_ascii=False, indent=2, allow_nan=False))
                return 0
            from emo_agent.agent.registry import ToolRegistry

            registry = ToolRegistry(data, root)
            if args.command == "agent-tool":
                result = registry.execute(args.tool, args.arguments)
            else:
                from emo_agent.agent.client import create_client
                from emo_agent.agent.orchestrator import ask

                config = config_at(root)
                with create_client(config) as client:
                    answer = ask(args.question, registry, client, config, model=os.getenv("DEEPSEEK_MODEL"))
                result = {"answer": answer.text, "results": answer.results}
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
            return 0
        except (ValueError, OSError, KeyError) as exc:
            print(json.dumps({"error": str(exc) if isinstance(exc, ValueError)
                              else "数据或配置不可用。"}, ensure_ascii=False))
            return 1
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
    if args.command == "build-context":
        from emo_agent.context import build_context
        print(json.dumps(build_context(root), ensure_ascii=False))
        return 0
    if args.command == "analyze-context":
        from emo_agent.context import run_context
        print(json.dumps(run_context(root, args.run_id, args.config), ensure_ascii=False))
        return 0
    if args.command == "shadow-freeze":
        from emo_agent.shadow import freeze_models
        print(json.dumps(freeze_models(root), ensure_ascii=False))
        return 0
    if args.command == "shadow-predict":
        from emo_agent.shadow import shadow_predict
        print(json.dumps(shadow_predict(root, args.date, args.mode), ensure_ascii=False))
        return 0
    if args.command == "shadow-evaluate":
        from emo_agent.shadow import shadow_evaluate
        print(json.dumps(shadow_evaluate(root, args.start, args.end), ensure_ascii=False, default=float))
        return 0
    report = audit_project(root)
    destination = safe_path(root, "outputs/audits/data_audit.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "report": str(destination), "files": len(report["files"])}, ensure_ascii=False))
    return 0 if report["ok"] else 1
