from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from . import __version__
from .analyzer import analyze_task, analyze_tasks, load_adapted_tasks, load_tasks, resolve_task
from .capture import capture_task
from .comparison import compare_model_runs, format_model_comparison
from .flywheel import run_flywheel
from .grader import grade_tasks, load_rules
from .replay import format_replay_text, replay_fixture
from .reports import (
    report_actions,
    report_actions_data,
    report_actions_markdown,
    report_broken,
    report_broken_data,
    report_broken_markdown,
    report_coding,
    report_coding_data,
    report_coding_markdown,
    report_compare_days,
    report_compare_days_data,
    report_compare_days_markdown,
    report_cost,
    report_cost_data,
    report_cost_markdown,
    report_fixes,
    report_fixes_data,
    report_fixes_markdown,
    report_flows,
    report_flows_data,
    report_flows_markdown,
    report_golden,
    report_golden_data,
    report_golden_markdown,
    report_health,
    report_health_data,
    report_health_markdown,
    report_outcomes,
    report_outcomes_data,
    report_outcomes_markdown,
    report_research,
    report_research_data,
    report_research_markdown,
    report_tools,
    report_tools_data,
    report_tools_markdown,
)
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
DEFAULT_RULES_PATH = Path(__file__).resolve().parent / "rules" / "default.json"
FORMAT_CHOICES = [
    "auto",
    "generic",
    "openai",
    "openai_chat",
    "langchain",
    "anthropic",
    "crewai",
    "otel",
]
GRADE_LABELS = ["GOLDEN", "GOOD", "OK", "WEAK", "BROKEN"]
SUPPORTED_FORMATS_TEXT = (
    "generic JSONL step logs, OpenAI SDK JSONL, LangChain JSONL, "
    "Anthropic JSONL, CrewAI JSONL, OpenTelemetry JSONL"
)
ANSI_RESET = "\033[0m"
ANSI_COLORS = {
    "header": "\033[36m",
    "path": "\033[36m",
    "GOLDEN": "\033[32m",
    "GOOD": "\033[36m",
    "OK": "\033[37m",
    "WEAK": "\033[33m",
    "BROKEN": "\033[31m",
}


class CliError(Exception):
    """User-facing CLI error."""


@dataclass(slots=True)
class CliSettings:
    verbose: bool = False
    quiet: bool = False
    color: bool = True


def _settings(args: argparse.Namespace | CliSettings | None) -> CliSettings:
    if isinstance(args, CliSettings):
        return args
    if args is None:
        return CliSettings(color=os.getenv("NO_COLOR") is None)
    return CliSettings(
        verbose=bool(getattr(args, "verbose", False)),
        quiet=bool(getattr(args, "quiet", False)),
        color=not bool(getattr(args, "no_color", False)) and os.getenv("NO_COLOR") is None,
    )


def _dump(data: object) -> None:
    print(json.dumps(data, indent=2))


def _emit(
    message: str, args: argparse.Namespace | CliSettings | None, *, final: bool = False
) -> None:
    ui = _settings(args)
    if ui.quiet and not final:
        return
    print(message)


def _emit_verbose(message: str, args: argparse.Namespace | CliSettings | None) -> None:
    ui = _settings(args)
    if ui.verbose and not ui.quiet:
        print(message)


def _paint(text: str, color_key: str, args: argparse.Namespace | CliSettings | None) -> str:
    ui = _settings(args)
    if not ui.color:
        return text
    color = ANSI_COLORS.get(color_key)
    if color is None:
        return text
    return f"{color}{text}{ANSI_RESET}"


def _format_elapsed(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def _grade_distribution(grades: list[Any]) -> dict[str, int]:
    counts = Counter(grade.grade for grade in grades)
    return {label: counts.get(label, 0) for label in GRADE_LABELS}


def _format_grade_summary(
    tasks: list[AgentTask],
    rules_name: str,
    grades: list[Any],
    args: argparse.Namespace | CliSettings | None,
) -> str:
    distribution = _grade_distribution(grades)
    lines = [
        _paint("GRADE SUMMARY", "header", args),
        f"Tasks: {len(tasks)}",
        f"Rules: {rules_name}",
        "",
    ]
    for label in GRADE_LABELS:
        lines.append(f"  {_paint(label, label, args)}: {distribution[label]}")
    return "\n".join(lines)


def _colorize_report_headers(text: str, args: argparse.Namespace | CliSettings | None) -> str:
    lines = text.splitlines()
    if not lines:
        return text
    lines[0] = _paint(lines[0], "header", args)
    if len(lines) > 1 and lines[1].strip() and set(lines[1].strip()) <= {"=", "-"}:
        lines[1] = _paint(lines[1], "header", args)
    return "\n".join(lines)


def _run_command(args: argparse.Namespace, action: Callable[[], int]) -> int:
    try:
        return action()
    except CliError as exc:
        _emit(str(exc), args, final=True)
        return 1
    except KeyError as exc:
        message = str(exc.args[0]) if exc.args else str(exc)
        _emit(message, args, final=True)
        return 1


def _copy_traversable_tree(source: Any, destination: Path) -> int:
    copied = 0
    destination.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        if child.name == "__pycache__":
            continue
        if child.is_dir():
            copied += _copy_traversable_tree(child, destination / child.name)
            continue
        if child.name.endswith(".py"):
            continue
        target = destination / child.name
        target.write_bytes(child.read_bytes())
        copied += 1
    return copied


def _synthetic_example_lines() -> list[str]:
    return [
        json.dumps(
            {
                "task_id": "golden-task",
                "step": 1,
                "tool_name": "browser_navigate",
                "tool_input": {"url": "https://shop.example.test"},
                "tool_result": "Homepage loaded.",
                "duration_ms": 900,
                "timestamp": "2026-03-26T12:00:00Z",
                "browser": {"page_url": "https://shop.example.test/"},
                "user_text": "Buy the wireless headset and complete checkout on shop.example.test.",
                "task_category": "commerce",
            },
            sort_keys=True,
        ),
        json.dumps(
            {
                "task_id": "golden-task",
                "step": 2,
                "tool_name": "browser_click",
                "tool_input": {"ref": "product-wireless-headset"},
                "tool_result": "Product detail page loaded.",
                "duration_ms": 450,
                "timestamp": "2026-03-26T12:01:00Z",
                "browser": {"page_url": "https://shop.example.test/products/wireless-headset"},
            },
            sort_keys=True,
        ),
        json.dumps(
            {
                "task_id": "golden-task",
                "step": 3,
                "tool_name": "browser_fill_ref",
                "tool_input": {
                    "ref": "shipping-form",
                    "fields": ["address", "zip"],
                    "text": "123 Main St 60601",
                },
                "tool_result": "Shipping form accepted.",
                "duration_ms": 600,
                "timestamp": "2026-03-26T12:02:00Z",
                "browser": {"page_url": "https://shop.example.test/checkout"},
            },
            sort_keys=True,
        ),
        json.dumps(
            {
                "task_id": "golden-task",
                "step": 4,
                "tool_name": "browser_snapshot",
                "tool_input": {},
                "tool_result": "Payment page visible.",
                "duration_ms": 200,
                "timestamp": "2026-03-26T12:03:00Z",
                "browser": {"page_url": "https://shop.example.test/payment"},
            },
            sort_keys=True,
        ),
        json.dumps(
            {
                "event": "task_complete",
                "task_id": "golden-task",
                "status": "success",
                "final_answer": "Checkout completed.",
                "total_steps": 4,
                "total_duration_s": 2.15,
                "timestamp": "2026-03-26T12:03:30Z",
            },
            sort_keys=True,
        ),
        json.dumps(
            {
                "task_id": "broken-task",
                "step": 1,
                "tool_name": "browser_snapshot",
                "tool_input": {},
                "error": "Timed out waiting for checkout.",
                "timestamp": "2026-03-26T13:00:00Z",
                "browser": {"page_url": "https://shop.example.test/checkout"},
                "user_text": "Recover the stuck checkout flow on shop.example.test.",
                "task_category": "commerce",
            },
            sort_keys=True,
        ),
        json.dumps(
            {
                "task_id": "broken-task",
                "step": 2,
                "tool_name": "browser_snapshot",
                "tool_input": {},
                "error": "Timed out waiting for checkout.",
                "timestamp": "2026-03-26T13:01:00Z",
                "browser": {"page_url": "https://shop.example.test/checkout"},
            },
            sort_keys=True,
        ),
        json.dumps(
            {
                "task_id": "broken-task",
                "step": 3,
                "tool_name": "browser_snapshot",
                "tool_input": {},
                "error": "Timed out waiting for checkout.",
                "timestamp": "2026-03-26T13:02:00Z",
                "browser": {"page_url": "https://shop.example.test/checkout"},
            },
            sort_keys=True,
        ),
        json.dumps(
            {
                "event": "task_complete",
                "task_id": "broken-task",
                "status": "failed",
                "total_steps": 3,
                "total_duration_s": 1.8,
                "timestamp": "2026-03-26T13:02:30Z",
            },
            sort_keys=True,
        ),
        json.dumps(
            {
                "task_id": "research-task",
                "step": 1,
                "tool_name": "web_search",
                "tool_input": {"query": "best wireless headset battery life"},
                "tool_result": "Found buying guides and reviews.",
                "timestamp": "2026-03-26T14:00:00Z",
                "user_text": "Research wireless headset battery life options.",
                "task_category": "research",
            },
            sort_keys=True,
        ),
        json.dumps(
            {
                "task_id": "research-task",
                "step": 2,
                "tool_name": "web_open",
                "tool_input": {"url": "https://reviews.example.test/headsets"},
                "tool_result": "Opened review roundup.",
                "timestamp": "2026-03-26T14:01:00Z",
            },
            sort_keys=True,
        ),
        json.dumps(
            {
                "task_id": "research-task",
                "step": 3,
                "tool_name": "write_summary",
                "tool_input": {"format": "bullets"},
                "tool_result": "Prepared short summary with citations.",
                "timestamp": "2026-03-26T14:02:00Z",
            },
            sort_keys=True,
        ),
        json.dumps(
            {
                "task_id": "research-task",
                "step": 4,
                "tool_name": "write_summary",
                "tool_input": {"format": "final"},
                "tool_result": "Delivered recommendation.",
                "timestamp": "2026-03-26T14:03:00Z",
            },
            sort_keys=True,
        ),
        json.dumps(
            {
                "event": "task_complete",
                "task_id": "research-task",
                "status": "success",
                "final_answer": "Compared several headsets.",
                "total_steps": 4,
                "total_duration_s": 2.0,
                "timestamp": "2026-03-26T14:03:30Z",
            },
            sort_keys=True,
        ),
    ]


def _populate_quickstart_dir(
    destination: Path, args: argparse.Namespace | CliSettings | None
) -> None:
    copied = 0
    try:
        copied = _copy_traversable_tree(files("agent_xray.examples"), destination)
    except ModuleNotFoundError:
        copied = 0
    if copied > 0:
        _emit_verbose(f"Copied {copied} bundled example trace file(s) into {destination}", args)
        return
    fallback_path = destination / "demo_20260326.jsonl"
    fallback_path.write_text("\n".join(_synthetic_example_lines()) + "\n", encoding="utf-8")
    _emit_verbose(f"No bundled example traces found; wrote synthetic demo to {fallback_path}", args)


def _load(args: argparse.Namespace) -> list[AgentTask]:
    log_dir = (
        getattr(args, "log_dir", None) or getattr(args, "log_dir_opt", None) or DEFAULT_LOG_DIR
    )
    return _load_tasks_with_format(
        log_dir,
        days=getattr(args, "days", None),
        format_name=getattr(args, "format", "auto"),
        settings=args,
    )


def _load_tasks_with_format(
    log_dir: str | Path,
    *,
    days: int | None = None,
    format_name: str = "auto",
    settings: argparse.Namespace | CliSettings | None = None,
) -> list[AgentTask]:
    ui = _settings(settings)
    path = Path(log_dir)
    if not path.exists():
        raise CliError(f"Directory not found: {path}. Run agent-xray quickstart for a demo.")
    _emit_verbose(
        f"Loading traces from {path} (format={format_name}, days={days if days is not None else 'all'})",
        ui,
    )
    started = perf_counter()
    if format_name != "auto":
        tasks = load_adapted_tasks(path, format=format_name, days=days)
    else:
        tasks = load_tasks(path, days=days)
        if not tasks:
            _emit_verbose("Native trace loader found no tasks; trying adapters.", ui)
            tasks = load_adapted_tasks(path, format="auto", days=days)
    if not tasks:
        raise CliError(
            f"No agent traces found in {path}. Supported formats: {SUPPORTED_FORMATS_TEXT}"
        )
    total_steps = sum(len(task.steps) for task in tasks)
    _emit_verbose(
        f"Loaded {len(tasks)} task(s), {total_steps} step(s) in {_format_elapsed(perf_counter() - started)}",
        ui,
    )
    return tasks


def cmd_analyze(args: argparse.Namespace) -> int:
    def _action() -> int:
        tasks = _load_tasks_with_format(
            args.log_dir, days=args.days, format_name=args.format, settings=args
        )
        started = perf_counter()
        rules = load_rules(args.rules) if args.rules else load_rules()
        grades = grade_tasks(tasks, rules)
        grade_distribution = _grade_distribution(grades)
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
        _emit_verbose(
            f"Analyzed {len(tasks)} task(s) in {_format_elapsed(perf_counter() - started)}",
            args,
        )
        if args.json:
            _dump(payload)
        else:
            total_steps = sum(len(t.steps) for t in tasks)
            total = summary["tasks"]
            lines = [f"Analyzed {total} task(s) across {total_steps} step(s)", ""]
            for label in GRADE_LABELS:
                count = grade_distribution[label]
                pct = (count / total * 100) if total else 0.0
                lines.append(f"  {label + ':':10s} {count:>3d}  ({pct:4.1f}%)")
            lines.append("")
            lines.append("Run 'agent-xray grade <dir> --json' for per-task details.")
            _emit("\n".join(lines), args, final=True)
        return 0

    return _run_command(args, _action)


def cmd_surface(args: argparse.Namespace) -> int:
    def _action() -> int:
        tasks = _load(args)
        data = surface_for_task(resolve_task(tasks, args.task_id))
        if args.json:
            _dump(data)
        else:
            _emit(format_surface_text(data), args, final=True)
        return 0

    return _run_command(args, _action)


def cmd_reasoning(args: argparse.Namespace) -> int:
    def _action() -> int:
        tasks = _load(args)
        data = reasoning_for_task(resolve_task(tasks, args.task_id))
        if args.json:
            _dump(data)
        else:
            _emit(format_reasoning_text(data), args, final=True)
        return 0

    return _run_command(args, _action)


def cmd_diff(args: argparse.Namespace) -> int:
    def _action() -> int:
        tasks = _load(args)
        data = diff_tasks(resolve_task(tasks, args.task_id_1), resolve_task(tasks, args.task_id_2))
        if args.json:
            _dump(data)
        else:
            lines: list[str] = []
            for key, section in data.items():
                lines.append(f"\n{key}:")
                if isinstance(section, dict):
                    for sub_key, value in section.items():
                        lines.append(f"  {sub_key}: {value}")
                else:
                    lines.append(f"  {section}")
            _emit("\n".join(lines), args, final=True)
        return 0

    return _run_command(args, _action)


def cmd_grade(args: argparse.Namespace) -> int:
    def _action() -> int:
        tasks = _load_tasks_with_format(
            args.log_dir, days=args.days, format_name=args.format, settings=args
        )
        started = perf_counter()
        rules = load_rules(args.rules)
        if not args.json:
            print(f"Grading {len(tasks)} task(s)...", file=sys.stderr, flush=True)
        grades = grade_tasks(tasks, rules)
        if not args.json:
            print("Done.", file=sys.stderr, flush=True)
        failures = classify_failures(tasks, grades)
        distribution = _grade_distribution(grades)
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
        _emit_verbose(
            f"Graded {len(tasks)} task(s) in {_format_elapsed(perf_counter() - started)}",
            args,
        )
        if args.json:
            _dump(payload)
        else:
            grade_text = _format_grade_summary(tasks, rules.name, grades, args)
            hint = "\nHint: Use 'agent-xray surface <task-id> --log-dir <dir>' to inspect a specific task."
            _emit(grade_text + hint, args, final=True)
        return 0

    return _run_command(args, _action)


def cmd_tree(args: argparse.Namespace) -> int:
    def _action() -> int:
        data = tree_for_tasks(_load(args))
        if args.json:
            _dump(data)
        else:
            _emit(format_tree_text(data), args, final=True)
        return 0

    return _run_command(args, _action)


def cmd_capture(args: argparse.Namespace) -> int:
    def _action() -> int:
        tasks = _load(args)
        output = Path(args.out) if args.out else Path.cwd() / "captured" / f"{args.task_id}.json"
        path = capture_task(tasks, args.task_id, output, sanitize=not args.no_sanitize)
        if args.json:
            _dump({"fixture": str(path)})
        else:
            _emit(str(path), args, final=True)
        return 0

    return _run_command(args, _action)


def cmd_replay(args: argparse.Namespace) -> int:
    def _action() -> int:
        result = replay_fixture(args.fixture, _load(args))
        if args.json:
            _dump(result)
        else:
            _emit(format_replay_text(result), args, final=True)
        return 0

    return _run_command(args, _action)


def cmd_flywheel(args: argparse.Namespace) -> int:
    def _action() -> int:
        result = run_flywheel(
            args.log_dir,
            rules_path=args.rules,
            fixture_dir=args.fixture_dir,
            baseline_path=args.baseline,
            output_path=args.out,
        )
        if args.json:
            _dump(result.to_dict())
        else:
            _emit(json.dumps(result.to_dict(), indent=2), args, final=True)
        return 0

    return _run_command(args, _action)


def cmd_compare(args: argparse.Namespace) -> int:
    def _action() -> int:
        result = compare_model_runs(
            args.left_log_dir,
            args.right_log_dir,
            rules_path=args.rules,
        )
        if args.json:
            _dump(result.to_dict())
        else:
            _emit(format_model_comparison(result), args, final=True)
        return 0

    return _run_command(args, _action)


def cmd_tui(args: argparse.Namespace) -> int:
    def _action() -> int:
        try:
            from agent_xray.tui.app import AgentXrayApp
        except ImportError:
            _emit(
                "TUI requires textual. Install with: pip install agent-xray[tui]", args, final=True
            )
            return 1

        app = AgentXrayApp(log_dir=args.log_dir, task_id=args.task_id)
        app.run()
        return 0

    return _run_command(args, _action)


def cmd_quickstart(args: argparse.Namespace) -> int:
    def _action() -> int:
        demo_dir = Path(tempfile.mkdtemp(prefix="agent_xray_quickstart_"))
        _populate_quickstart_dir(demo_dir, args)
        _emit(f"Quickstart traces: {_paint(str(demo_dir), 'path', args)}", args)

        tasks = _load_tasks_with_format(demo_dir, format_name="auto", settings=args)
        rules = load_rules(DEFAULT_RULES_PATH)
        grade_started = perf_counter()
        grades = grade_tasks(tasks, rules)
        _emit_verbose(
            f"Quickstart grade step completed in {_format_elapsed(perf_counter() - grade_started)}",
            args,
        )
        target_grade = next((grade for grade in grades if grade.grade == "BROKEN"), None)
        if target_grade is None:
            target_grade = min(grades, key=lambda item: item.score)
        surface_started = perf_counter()
        surface_text = format_surface_text(
            surface_for_task(resolve_task(tasks, target_grade.task_id))
        )
        _emit_verbose(
            f"Quickstart surface step completed in {_format_elapsed(perf_counter() - surface_started)}",
            args,
        )
        report_started = perf_counter()
        analyses = analyze_tasks(tasks)
        health_text = _colorize_report_headers(report_health(tasks, grades, analyses), args)
        _emit_verbose(
            f"Quickstart report step completed in {_format_elapsed(perf_counter() - report_started)}",
            args,
        )

        if _settings(args).quiet:
            _emit(health_text, args, final=True)
            return 0

        sections = [
            _paint("QUICKSTART", "header", args),
            f"Workflow: grade -> surface {target_grade.task_id} -> report health",
            "",
            _paint("[1/3] grade", "header", args),
            _format_grade_summary(tasks, rules.name, grades, args),
            "",
            _paint(f"[2/3] surface {target_grade.task_id}", "header", args),
            surface_text,
            "",
            _paint("[3/3] report health", "header", args),
            health_text,
        ]
        _emit("\n".join(sections), args, final=True)
        return 0

    return _run_command(args, _action)


def _grade_and_analyze(
    args: argparse.Namespace,
) -> tuple[
    list[AgentTask],
    list[Any],
    dict[str, Any],
]:
    started = perf_counter()
    tasks = _load_tasks_with_format(
        args.log_dir, days=args.days, format_name=args.format, settings=args
    )
    rules = load_rules(args.rules) if args.rules else load_rules()
    _emit_verbose(f"Grading {len(tasks)} task(s) with rules={rules.name}", args)
    grades = grade_tasks(tasks, rules)
    _emit_verbose(f"Analyzing {len(tasks)} task(s) for report generation", args)
    analyses = analyze_tasks(tasks)
    _emit_verbose(
        f"Prepared report inputs in {_format_elapsed(perf_counter() - started)}",
        args,
    )
    return tasks, grades, analyses


def cmd_report(args: argparse.Namespace) -> int:
    def _action() -> int:
        tasks, grades, analyses = _grade_and_analyze(args)
        report_type = args.report_type
        use_json = getattr(args, "json", False)
        use_markdown = getattr(args, "markdown", False)

        if use_json and use_markdown:
            _emit("--json and --markdown are mutually exclusive", args, final=True)
            return 1

        if report_type == "compare":
            if not args.day1 or not args.day2:
                _emit("--day1 and --day2 are required for the compare report", args, final=True)
                return 1
            if use_json:
                _dump(report_compare_days_data(tasks, grades, analyses, args.day1, args.day2))
            elif use_markdown:
                _emit(
                    report_compare_days_markdown(tasks, grades, analyses, args.day1, args.day2),
                    args,
                    final=True,
                )
            else:
                _emit(
                    _colorize_report_headers(
                        report_compare_days(tasks, grades, analyses, args.day1, args.day2),
                        args,
                    ),
                    args,
                    final=True,
                )
            return 0

        text_funcs = {
            "health": lambda: report_health(tasks, grades, analyses),
            "golden": lambda: report_golden(tasks, grades, analyses),
            "broken": lambda: report_broken(tasks, grades, analyses),
            "tools": lambda: report_tools(tasks, analyses),
            "flows": lambda: report_flows(tasks, analyses),
            "outcomes": lambda: report_outcomes(tasks, grades, analyses),
            "actions": lambda: report_actions(tasks, grades, analyses),
            "coding": lambda: report_coding(tasks, analyses),
            "research": lambda: report_research(tasks, analyses),
            "cost": lambda: report_cost(tasks, analyses),
            "fixes": lambda: report_fixes(tasks, grades, analyses),
        }
        data_funcs: dict[str, Any] = {
            "health": lambda: report_health_data(tasks, grades, analyses),
            "golden": lambda: report_golden_data(tasks, grades, analyses),
            "broken": lambda: report_broken_data(tasks, grades, analyses),
            "tools": lambda: report_tools_data(tasks, analyses),
            "flows": lambda: report_flows_data(tasks, analyses),
            "outcomes": lambda: report_outcomes_data(tasks, grades, analyses),
            "actions": lambda: report_actions_data(tasks, grades, analyses),
            "coding": lambda: report_coding_data(tasks, analyses),
            "research": lambda: report_research_data(tasks, analyses),
            "cost": lambda: report_cost_data(tasks, analyses),
            "fixes": lambda: report_fixes_data(tasks, grades, analyses),
        }
        markdown_funcs: dict[str, Any] = {
            "health": lambda: report_health_markdown(tasks, grades, analyses),
            "golden": lambda: report_golden_markdown(tasks, grades, analyses),
            "broken": lambda: report_broken_markdown(tasks, grades, analyses),
            "tools": lambda: report_tools_markdown(tasks, analyses),
            "flows": lambda: report_flows_markdown(tasks, analyses),
            "outcomes": lambda: report_outcomes_markdown(tasks, grades, analyses),
            "actions": lambda: report_actions_markdown(tasks, grades, analyses),
            "coding": lambda: report_coding_markdown(tasks, analyses),
            "research": lambda: report_research_markdown(tasks, analyses),
            "cost": lambda: report_cost_markdown(tasks, analyses),
            "fixes": lambda: report_fixes_markdown(tasks, grades, analyses),
        }

        if report_type not in text_funcs:
            _emit(f"Unknown report type: {report_type}", args, final=True)
            return 1

        if use_json:
            _dump(data_funcs[report_type]())
        elif use_markdown:
            _emit(markdown_funcs[report_type](), args, final=True)
        else:
            _emit(_colorize_report_headers(text_funcs[report_type](), args), args, final=True)
        return 0

    return _run_command(args, _action)


def cmd_record(args: argparse.Namespace) -> int:
    """Run a subprocess and capture tool calls from its stdout.

    The subprocess is expected to print JSON lines to stdout following a simple
    protocol.  Each line must be a JSON object with at least ``tool_name`` and
    ``tool_input`` fields.  Optional fields: ``tool_result``, ``error``,
    ``duration_ms``, ``model_name``, ``task_id``.

    Lines that are not valid JSON or do not contain ``tool_name`` are passed
    through to the terminal unchanged.
    """

    def _action() -> int:
        command = args.command
        if not command:
            _emit(
                "No command specified. Usage: agent-xray record -- python my_agent.py",
                args,
                final=True,
            )
            return 1
        # Strip leading -- if present
        if command and command[0] == "--":
            command = command[1:]
        if not command:
            _emit("No command specified after --", args, final=True)
            return 1

        from .instrument.base import StepRecorder

        task_id = args.task_id or f"record-{os.getpid()}"
        recorder = StepRecorder(args.output_dir, task_id=task_id)
        recorder.start_task(task_id)

        try:
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=sys.stderr,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError:
            _emit(f"Command not found: {command[0]}", args, final=True)
            recorder.close()
            return 1

        step_count = 0
        try:
            for raw_line in proc.stdout or []:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    print(raw_line, end="")
                    continue
                if not isinstance(payload, dict) or "tool_name" not in payload:
                    print(raw_line, end="")
                    continue
                step_count += 1
                tool_input = payload.get("tool_input")
                if not isinstance(tool_input, dict):
                    tool_input = {"value": tool_input} if tool_input is not None else {}
                recorder.record_step(
                    task_id=str(payload.get("task_id", task_id)),
                    tool_name=str(payload["tool_name"]),
                    tool_input=tool_input,
                    tool_result=payload.get("tool_result"),
                    error=payload.get("error"),
                    duration_ms=payload.get("duration_ms"),
                    model_name=payload.get("model_name"),
                )
        finally:
            proc.wait()

        recorder.end_task(task_id, "success" if proc.returncode == 0 else "failed")
        recorder.close()

        _emit(
            f"Recorded {step_count} step(s) to {recorder.output_dir}",
            args,
            final=True,
        )
        return proc.returncode or 0

    return _run_command(args, _action)


def _add_format_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format",
        choices=FORMAT_CHOICES,
        default="auto",
        help="Trace log format (default: auto-detect)",
    )


def _add_subparser(
    subcommands: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    *,
    help_text: str,
    example: str,
) -> argparse.ArgumentParser:
    return subcommands.add_parser(
        name,
        help=help_text,
        epilog=f"Example: {example}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-xray",
        description="Analyze and replay agent step logs.",
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("--verbose", action="store_true", help="Show progress and timing")
    verbosity.add_argument("--quiet", action="store_true", help="Suppress progress output")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI color output")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = _add_subparser(
        sub,
        "analyze",
        help_text="Analyze a log directory",
        example="agent-xray analyze ./traces --rules browser_flow --json",
    )
    p_analyze.add_argument("log_dir", help="Directory containing .jsonl trace files")
    p_analyze.add_argument("--days", type=int, help="Include only the N most recent days of traces")
    p_analyze.add_argument(
        "--rules",
        help="Ruleset name (default, browser_flow, coding_agent, research_agent) or path to JSON",
    )
    _add_format_option(p_analyze)
    p_analyze.add_argument("--json", action="store_true", help="Output results as JSON")
    p_analyze.set_defaults(func=cmd_analyze)

    for name, handler, example in (
        ("surface", cmd_surface, "agent-xray surface task-123 --log-dir ./traces"),
        ("reasoning", cmd_reasoning, "agent-xray reasoning task-123 --log-dir ./traces --json"),
    ):
        parser_ = _add_subparser(
            sub,
            name,
            help_text=f"{name.title()} output for a task",
            example=example,
        )
        parser_.add_argument("task_id", help="Task ID or prefix to search for")
        parser_.add_argument(
            "--log-dir", dest="log_dir_opt", help="Directory containing .jsonl trace files"
        )
        parser_.add_argument(
            "--days", type=int, help="Include only the N most recent days of traces"
        )
        _add_format_option(parser_)
        parser_.add_argument("--json", action="store_true", help="Output results as JSON")
        parser_.set_defaults(func=handler)

    p_diff = _add_subparser(
        sub,
        "diff",
        help_text="Compare two tasks",
        example="agent-xray diff task-123 task-124 --log-dir ./traces",
    )
    p_diff.add_argument("task_id_1", help="First task ID to compare")
    p_diff.add_argument("task_id_2", help="Second task ID to compare")
    p_diff.add_argument(
        "--log-dir", dest="log_dir_opt", help="Directory containing .jsonl trace files"
    )
    p_diff.add_argument("--days", type=int, help="Include only the N most recent days of traces")
    _add_format_option(p_diff)
    p_diff.add_argument("--json", action="store_true", help="Output results as JSON")
    p_diff.set_defaults(func=cmd_diff)

    p_grade = _add_subparser(
        sub,
        "grade",
        help_text="Grade a log directory",
        example="agent-xray grade ./traces --rules browser_flow",
    )
    p_grade.add_argument("log_dir", help="Directory containing .jsonl trace files")
    p_grade.add_argument(
        "--rules",
        default=str(DEFAULT_RULES_PATH),
        help="Ruleset name (default, browser_flow, coding_agent, research_agent) or path to JSON",
    )
    p_grade.add_argument("--days", type=int, help="Include only the N most recent days of traces")
    _add_format_option(p_grade)
    p_grade.add_argument("--json", action="store_true", help="Output results as JSON")
    p_grade.set_defaults(func=cmd_grade)

    p_tree = _add_subparser(
        sub,
        "tree",
        help_text="Show a day/site/task tree",
        example="agent-xray tree --log-dir ./traces",
    )
    p_tree.add_argument(
        "--log-dir", dest="log_dir_opt", help="Directory containing .jsonl trace files"
    )
    p_tree.add_argument("--days", type=int, help="Include only the N most recent days of traces")
    _add_format_option(p_tree)
    p_tree.add_argument("--json", action="store_true", help="Output results as JSON")
    p_tree.set_defaults(func=cmd_tree)

    p_capture = _add_subparser(
        sub,
        "capture",
        help_text="Capture a task as a sanitized fixture",
        example="agent-xray capture task-123 --log-dir ./traces --out ./fixtures/task-123.json",
    )
    p_capture.add_argument("task_id", help="Task ID or prefix to search for")
    p_capture.add_argument(
        "--log-dir", dest="log_dir_opt", help="Directory containing .jsonl trace files"
    )
    p_capture.add_argument("--days", type=int, help="Include only the N most recent days of traces")
    _add_format_option(p_capture)
    p_capture.add_argument("--out", help="Output file path for captured fixture")
    p_capture.add_argument(
        "--no-sanitize", action="store_true", help="Disable PII sanitization in captured fixtures"
    )
    p_capture.add_argument("--json", action="store_true", help="Output results as JSON")
    p_capture.set_defaults(func=cmd_capture)

    p_replay = _add_subparser(
        sub,
        "replay",
        help_text="Compare a fixture to current logs",
        example="agent-xray replay ./fixtures/task-123.json --log-dir ./traces",
    )
    p_replay.add_argument("fixture", help="Path to a captured fixture JSON file")
    p_replay.add_argument(
        "--log-dir", dest="log_dir_opt", help="Directory containing .jsonl trace files"
    )
    p_replay.add_argument("--days", type=int, help="Include only the N most recent days of traces")
    _add_format_option(p_replay)
    p_replay.add_argument("--json", action="store_true", help="Output results as JSON")
    p_replay.set_defaults(func=cmd_replay)

    p_flywheel = _add_subparser(
        sub,
        "flywheel",
        help_text="Run end-to-end grading, root-cause analysis, and baseline comparison",
        example="agent-xray flywheel ./traces --baseline ./baseline.json --json",
    )
    p_flywheel.add_argument("log_dir", help="Directory containing .jsonl trace files")
    p_flywheel.add_argument(
        "--rules",
        help="Ruleset name (default, browser_flow, coding_agent, research_agent) or path to JSON",
    )
    p_flywheel.add_argument("--fixture-dir", help="Directory containing golden fixture files")
    p_flywheel.add_argument(
        "--baseline",
        help="Previous flywheel JSON output used for grade delta and regression comparison",
    )
    p_flywheel.add_argument("--out", help="Output file path for flywheel results")
    p_flywheel.add_argument("--json", action="store_true", help="Output results as JSON")
    p_flywheel.set_defaults(func=cmd_flywheel)

    p_compare = _add_subparser(
        sub,
        "compare",
        help_text="Compare two model run directories",
        example="agent-xray compare ./runs/model-a ./runs/model-b --rules browser_flow",
    )
    p_compare.add_argument("left_log_dir", help="First trace directory to compare")
    p_compare.add_argument("right_log_dir", help="Second trace directory to compare")
    p_compare.add_argument(
        "--rules",
        help="Ruleset name (default, browser_flow, coding_agent, research_agent) or path to JSON",
    )
    p_compare.add_argument("--json", action="store_true", help="Output results as JSON")
    p_compare.set_defaults(func=cmd_compare)

    p_report = _add_subparser(
        sub,
        "report",
        help_text="Generate a report (health, golden, broken, tools, flows, outcomes, actions, coding, research, cost, fixes, compare)",
        example="agent-xray report ./traces health",
    )
    p_report.add_argument("log_dir", help="Directory containing .jsonl trace files")
    p_report.add_argument(
        "report_type",
        choices=[
            "health",
            "golden",
            "broken",
            "tools",
            "flows",
            "outcomes",
            "actions",
            "coding",
            "research",
            "cost",
            "fixes",
            "compare",
        ],
        help="Type of report to generate",
    )
    p_report.add_argument(
        "--rules",
        help="Ruleset name (default, browser_flow, coding_agent, research_agent) or path to JSON",
    )
    p_report.add_argument("--days", type=int, help="Include only the N most recent days of traces")
    _add_format_option(p_report)
    p_report.add_argument("--day1", help="First day for compare report (YYYYMMDD)")
    p_report.add_argument("--day2", help="Second day for compare report (YYYYMMDD)")
    p_report.add_argument("--json", action="store_true", help="Output results as JSON")
    p_report.add_argument("--markdown", action="store_true", help="Output results as Markdown")
    p_report.set_defaults(func=cmd_report)

    p_tui = _add_subparser(
        sub,
        "tui",
        help_text="Open the interactive decision-surface inspector",
        example="agent-xray tui ./traces --task-id task-123",
    )
    p_tui.add_argument("log_dir", help="Trace log directory or jsonl file to inspect")
    p_tui.add_argument("--task-id", help="Specific task id to open. Defaults to the latest task.")
    p_tui.set_defaults(func=cmd_tui)

    p_quickstart = _add_subparser(
        sub,
        "quickstart",
        help_text="Create a demo trace directory and run a full walkthrough",
        example="agent-xray quickstart",
    )
    p_quickstart.set_defaults(func=cmd_quickstart)

    p_record = _add_subparser(
        sub,
        "record",
        help_text="Run a subprocess and capture tool calls from its stdout as JSONL steps",
        example="agent-xray record --output-dir ./traces -- python my_agent.py",
    )
    p_record.add_argument(
        "--output-dir",
        default="./traces",
        help="Directory for JSONL output (default: ./traces)",
    )
    p_record.add_argument(
        "--task-id",
        default=None,
        help="Task identifier (default: auto-generated)",
    )
    p_record.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run (everything after --)",
    )
    p_record.set_defaults(func=cmd_record)

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
