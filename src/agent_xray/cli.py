from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from .analyzer import analyze_task, load_adapted_tasks, load_tasks, resolve_task
from .capture import capture_task
from .comparison import compare_model_runs, format_model_comparison
from .flywheel import run_flywheel
from .grader import grade_tasks, load_rules
from .replay import format_replay_text, replay_fixture
from .root_cause import classify_failures
from .schema import AgentTask
from .surface import (
    diff_tasks,
    format_reasoning_text,
    format_surface_text,
    format_tree_text,
    reasoning_for_task,
    surface_for_task,
    tree_for_tasks,
)

DEFAULT_LOG_DIR = os.environ.get("AGENT_XRAY_LOG_DIR", ".")
FORMAT_CHOICES = ["auto", "generic", "openai", "langchain", "anthropic", "crewai", "otel"]


def _dump(data: object) -> None:
    print(json.dumps(data, indent=2))


def _load(args: argparse.Namespace) -> list[AgentTask]:
    log_dir = (
        getattr(args, "log_dir", None) or getattr(args, "log_dir_opt", None) or DEFAULT_LOG_DIR
    )
    return _load_tasks_with_format(
        log_dir,
        days=getattr(args, "days", None),
        format_name=getattr(args, "format", "auto"),
    )


def _load_tasks_with_format(
    log_dir: str | Path,
    *,
    days: int | None = None,
    format_name: str = "auto",
) -> list[AgentTask]:
    if format_name != "auto":
        return load_adapted_tasks(log_dir, format=format_name, days=days)
    tasks = load_tasks(log_dir, days=days)
    if tasks:
        return tasks
    return load_adapted_tasks(log_dir, format="auto", days=days)


def cmd_analyze(args: argparse.Namespace) -> int:
    tasks = _load_tasks_with_format(args.log_dir, days=args.days, format_name=args.format)
    rules = load_rules(args.rules) if args.rules else load_rules()
    grades = grade_tasks(tasks, rules)
    grade_distribution = {
        grade: sum(1 for item in grades if item.grade == grade)
        for grade in ["GOLDEN", "GOOD", "OK", "WEAK", "BROKEN"]
    }
    summary = {
        "tasks": len(tasks),
        "rules": rules.name,
        "grade_distribution": grade_distribution,
    }
    payload: dict[str, Any] = {
        "summary": summary,
        "tasks": [
            {
                "task_id": grade.task_id,
                "grade": grade.grade,
                "score": grade.score,
                "site": analyze_task(resolve_task(tasks, grade.task_id)).site_name,
            }
            for grade in grades
        ],
    }
    if args.json:
        _dump(payload)
    else:
        print(f"Analyzed {summary['tasks']} task(s) with rules={rules.name}")
        print(grade_distribution)
    return 0


def cmd_surface(args: argparse.Namespace) -> int:
    tasks = _load(args)
    data = surface_for_task(resolve_task(tasks, args.task_id))
    _dump(data) if args.json else print(format_surface_text(data))
    return 0


def cmd_reasoning(args: argparse.Namespace) -> int:
    tasks = _load(args)
    data = reasoning_for_task(resolve_task(tasks, args.task_id))
    _dump(data) if args.json else print(format_reasoning_text(data))
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    tasks = _load(args)
    data = diff_tasks(resolve_task(tasks, args.task_id_1), resolve_task(tasks, args.task_id_2))
    _dump(data) if args.json else print(json.dumps(data, indent=2))
    return 0


def cmd_grade(args: argparse.Namespace) -> int:
    tasks = _load_tasks_with_format(args.log_dir, days=args.days, format_name=args.format)
    rules = load_rules(args.rules)
    grades = grade_tasks(tasks, rules)
    failures = classify_failures(tasks, grades)
    distribution = {
        grade: sum(1 for item in grades if item.grade == grade)
        for grade in ["GOLDEN", "GOOD", "OK", "WEAK", "BROKEN"]
    }
    summary = {
        "tasks": len(tasks),
        "rules": rules.name,
        "distribution": distribution,
    }
    payload: dict[str, Any] = {
        "summary": summary,
        "tasks": [
            {
                "task_id": grade.task_id,
                "grade": grade.grade,
                "score": grade.score,
                "reasons": grade.reasons,
            }
            for grade in grades
        ],
        "root_causes": [asdict(result) for result in failures],
    }
    if args.json:
        _dump(payload)
    else:
        print(f"Graded {len(tasks)} task(s) with rules={rules.name}")
        print(distribution)
    return 0


def cmd_tree(args: argparse.Namespace) -> int:
    data = tree_for_tasks(_load(args))
    _dump(data) if args.json else print(format_tree_text(data))
    return 0


def cmd_capture(args: argparse.Namespace) -> int:
    tasks = _load(args)
    output = Path(args.out) if args.out else Path.cwd() / "captured" / f"{args.task_id}.json"
    path = capture_task(tasks, args.task_id, output, sanitize=not args.no_sanitize)
    _dump({"fixture": str(path)}) if args.json else print(path)
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    result = replay_fixture(args.fixture, _load(args))
    _dump(result) if args.json else print(format_replay_text(result))
    return 0


def cmd_flywheel(args: argparse.Namespace) -> int:
    result = run_flywheel(
        args.log_dir,
        rules_path=args.rules,
        fixture_dir=args.fixture_dir,
        baseline_path=args.baseline,
        output_path=args.out,
    )
    _dump(result.to_dict()) if args.json else print(json.dumps(result.to_dict(), indent=2))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    result = compare_model_runs(
        args.left_log_dir,
        args.right_log_dir,
        rules_path=args.rules,
    )
    _dump(result.to_dict()) if args.json else print(format_model_comparison(result))
    return 0


def cmd_tui(args: argparse.Namespace) -> int:
    try:
        from agent_xray.tui.app import AgentXrayApp
    except ImportError:
        print("TUI requires textual. Install with: pip install agent-xray[tui]")
        return 1

    app = AgentXrayApp(log_dir=args.log_dir, task_id=args.task_id)
    app.run()
    return 0


def _add_format_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format",
        choices=FORMAT_CHOICES,
        default="auto",
        help="Trace log format (default: auto-detect)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-xray",
        description="Analyze and replay agent step logs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze", help="Analyze a log directory")
    p_analyze.add_argument("log_dir")
    p_analyze.add_argument("--days", type=int)
    p_analyze.add_argument("--rules")
    _add_format_option(p_analyze)
    p_analyze.add_argument("--json", action="store_true")
    p_analyze.set_defaults(func=cmd_analyze)

    for name, handler in (("surface", cmd_surface), ("reasoning", cmd_reasoning)):
        parser_ = sub.add_parser(name, help=f"{name.title()} output for a task")
        parser_.add_argument("task_id")
        parser_.add_argument("--log-dir", dest="log_dir_opt")
        parser_.add_argument("--days", type=int)
        _add_format_option(parser_)
        parser_.add_argument("--json", action="store_true")
        parser_.set_defaults(func=handler)

    p_diff = sub.add_parser("diff", help="Compare two tasks")
    p_diff.add_argument("task_id_1")
    p_diff.add_argument("task_id_2")
    p_diff.add_argument("--log-dir", dest="log_dir_opt")
    p_diff.add_argument("--days", type=int)
    _add_format_option(p_diff)
    p_diff.add_argument("--json", action="store_true")
    p_diff.set_defaults(func=cmd_diff)

    p_grade = sub.add_parser("grade", help="Grade a log directory")
    p_grade.add_argument("log_dir")
    p_grade.add_argument(
        "--rules",
        default=str(Path(__file__).resolve().parent / "rules" / "default.json"),
    )
    p_grade.add_argument("--days", type=int)
    _add_format_option(p_grade)
    p_grade.add_argument("--json", action="store_true")
    p_grade.set_defaults(func=cmd_grade)

    p_tree = sub.add_parser("tree", help="Show a day/site/task tree")
    p_tree.add_argument("--log-dir", dest="log_dir_opt")
    p_tree.add_argument("--days", type=int)
    _add_format_option(p_tree)
    p_tree.add_argument("--json", action="store_true")
    p_tree.set_defaults(func=cmd_tree)

    p_capture = sub.add_parser("capture", help="Capture a task as a sanitized fixture")
    p_capture.add_argument("task_id")
    p_capture.add_argument("--log-dir", dest="log_dir_opt")
    p_capture.add_argument("--days", type=int)
    _add_format_option(p_capture)
    p_capture.add_argument("--out")
    p_capture.add_argument("--no-sanitize", action="store_true")
    p_capture.add_argument("--json", action="store_true")
    p_capture.set_defaults(func=cmd_capture)

    p_replay = sub.add_parser("replay", help="Compare a fixture to current logs")
    p_replay.add_argument("fixture")
    p_replay.add_argument("--log-dir", dest="log_dir_opt")
    p_replay.add_argument("--days", type=int)
    _add_format_option(p_replay)
    p_replay.add_argument("--json", action="store_true")
    p_replay.set_defaults(func=cmd_replay)

    p_flywheel = sub.add_parser(
        "flywheel",
        help="Run end-to-end grading, root-cause analysis, and baseline comparison",
    )
    p_flywheel.add_argument("log_dir")
    p_flywheel.add_argument("--rules")
    p_flywheel.add_argument("--fixture-dir")
    p_flywheel.add_argument(
        "--baseline",
        help="Previous flywheel JSON output used for grade delta and regression comparison",
    )
    p_flywheel.add_argument("--out")
    p_flywheel.add_argument("--json", action="store_true")
    p_flywheel.set_defaults(func=cmd_flywheel)

    p_compare = sub.add_parser("compare", help="Compare two model run directories")
    p_compare.add_argument("left_log_dir")
    p_compare.add_argument("right_log_dir")
    p_compare.add_argument("--rules")
    p_compare.add_argument("--json", action="store_true")
    p_compare.set_defaults(func=cmd_compare)

    p_tui = sub.add_parser("tui", help="Open the interactive decision-surface inspector")
    p_tui.add_argument("log_dir", help="Trace log directory or jsonl file to inspect")
    p_tui.add_argument("--task-id", help="Specific task id to open. Defaults to the latest task.")
    p_tui.set_defaults(func=cmd_tui)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    handler = getattr(args, "func", None)
    if not callable(handler):
        raise ValueError("parser did not assign a command handler")
    command_handler = cast(Callable[[argparse.Namespace], int], handler)
    return command_handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
