from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from .analyzer import analyze_task, load_tasks, resolve_task
from .capture import capture_task
from .grader import grade_tasks, load_rules
from .replay import format_replay_text, replay_fixture
from .root_cause import classify_failures
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


def _dump(data: object) -> None:
    print(json.dumps(data, indent=2))


def _load(args: argparse.Namespace) -> list:
    log_dir = (
        getattr(args, "log_dir", None) or getattr(args, "log_dir_opt", None) or DEFAULT_LOG_DIR
    )
    return load_tasks(log_dir, days=getattr(args, "days", None))


def cmd_analyze(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.log_dir, days=args.days)
    rules = load_rules(args.rules) if args.rules else load_rules()
    grades = grade_tasks(tasks, rules)
    payload = {
        "summary": {
            "tasks": len(tasks),
            "rules": rules.name,
            "grade_distribution": {
                grade: sum(1 for item in grades if item.grade == grade)
                for grade in ["GOLDEN", "GOOD", "OK", "WEAK", "BROKEN"]
            },
        },
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
        print(f"Analyzed {payload['summary']['tasks']} task(s) with rules={rules.name}")
        print(payload["summary"]["grade_distribution"])
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
    tasks = load_tasks(args.log_dir, days=args.days)
    rules = load_rules(args.rules)
    grades = grade_tasks(tasks, rules)
    failures = classify_failures(tasks, grades)
    payload = {
        "summary": {
            "tasks": len(tasks),
            "rules": rules.name,
            "distribution": {
                grade: sum(1 for item in grades if item.grade == grade)
                for grade in ["GOLDEN", "GOOD", "OK", "WEAK", "BROKEN"]
            },
        },
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
        print(payload["summary"]["distribution"])
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-xray", description="Analyze and replay agent step logs."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze", help="Analyze a log directory")
    p_analyze.add_argument("log_dir")
    p_analyze.add_argument("--days", type=int)
    p_analyze.add_argument("--rules")
    p_analyze.add_argument("--json", action="store_true")
    p_analyze.set_defaults(func=cmd_analyze)

    for name, handler in (("surface", cmd_surface), ("reasoning", cmd_reasoning)):
        parser_ = sub.add_parser(name, help=f"{name.title()} output for a task")
        parser_.add_argument("task_id")
        parser_.add_argument("--log-dir", dest="log_dir_opt")
        parser_.add_argument("--days", type=int)
        parser_.add_argument("--json", action="store_true")
        parser_.set_defaults(func=handler)

    p_diff = sub.add_parser("diff", help="Compare two tasks")
    p_diff.add_argument("task_id_1")
    p_diff.add_argument("task_id_2")
    p_diff.add_argument("--log-dir", dest="log_dir_opt")
    p_diff.add_argument("--days", type=int)
    p_diff.add_argument("--json", action="store_true")
    p_diff.set_defaults(func=cmd_diff)

    p_grade = sub.add_parser("grade", help="Grade a log directory")
    p_grade.add_argument("log_dir")
    p_grade.add_argument(
        "--rules", default=str(Path(__file__).resolve().parent / "rules" / "default.json")
    )
    p_grade.add_argument("--days", type=int)
    p_grade.add_argument("--json", action="store_true")
    p_grade.set_defaults(func=cmd_grade)

    p_tree = sub.add_parser("tree", help="Show a day/site/task tree")
    p_tree.add_argument("--log-dir", dest="log_dir_opt")
    p_tree.add_argument("--days", type=int)
    p_tree.add_argument("--json", action="store_true")
    p_tree.set_defaults(func=cmd_tree)

    p_capture = sub.add_parser("capture", help="Capture a task as a sanitized fixture")
    p_capture.add_argument("task_id")
    p_capture.add_argument("--log-dir", dest="log_dir_opt")
    p_capture.add_argument("--days", type=int)
    p_capture.add_argument("--out")
    p_capture.add_argument("--no-sanitize", action="store_true")
    p_capture.add_argument("--json", action="store_true")
    p_capture.set_defaults(func=cmd_capture)

    p_replay = sub.add_parser("replay", help="Compare a fixture to current logs")
    p_replay.add_argument("fixture")
    p_replay.add_argument("--log-dir", dest="log_dir_opt")
    p_replay.add_argument("--days", type=int)
    p_replay.add_argument("--json", action="store_true")
    p_replay.set_defaults(func=cmd_replay)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
