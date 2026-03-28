from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from importlib.resources import files
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from . import __version__
from .analyzer import analyze_task, analyze_tasks, load_adapted_tasks, load_tasks, resolve_task
from .capture import capture_task
from .comparison import compare_model_runs, format_model_comparison
from .flywheel import run_flywheel
from .grader import GradeResult, grade_tasks, load_rules
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
    report_spins,
    report_spins_data,
    report_spins_markdown,
    report_timeline,
    report_timeline_data,
    report_timeline_markdown,
    report_tools,
    report_tools_data,
    report_tools_markdown,
)
from .root_cause import classify_failures
from .schema import AgentTask
from .surface import (
    diff_tasks,
    enriched_tree_for_tasks,
    format_diff_summary,
    format_enriched_tree_text,
    format_prompt_diff,
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


def _safe_print(text: str) -> None:
    """Print with graceful fallback for terminals that can't encode all Unicode."""
    try:
        print(text)
    except UnicodeEncodeError:
        import sys
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(text.encode(enc, errors="replace").decode(enc, errors="replace"))


def _dump(data: object) -> None:
    _safe_print(json.dumps(data, indent=2))


def _emit(
    message: str, args: argparse.Namespace | CliSettings | None, *, final: bool = False
) -> None:
    ui = _settings(args)
    if ui.quiet and not final:
        return
    _safe_print(message)


def _emit_verbose(message: str, args: argparse.Namespace | CliSettings | None) -> None:
    ui = _settings(args)
    if ui.verbose and not ui.quiet:
        _safe_print(message)


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


def _parse_bucket_arg(value: str | None) -> int:
    """Parse a bucket argument like '15m' or '1h' into minutes."""
    if not value:
        return 60
    value = value.strip().lower()
    if value.endswith("m"):
        try:
            return max(1, int(value[:-1]))
        except ValueError:
            return 60
    if value.endswith("h"):
        try:
            return max(1, int(value[:-1]) * 60)
        except ValueError:
            return 60
    try:
        return max(1, int(value))
    except ValueError:
        return 60


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
    except FileNotFoundError as exc:
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


def _resolve_log_dir(args: argparse.Namespace) -> str:
    return (
        getattr(args, "log_dir", None)
        or getattr(args, "log_dir_pos", None)
        or getattr(args, "log_dir_opt", None)
        or DEFAULT_LOG_DIR
    )


def _load(args: argparse.Namespace) -> list[AgentTask]:
    return _load_tasks_with_format(
        _resolve_log_dir(args),
        days=getattr(args, "days", None),
        format_name=getattr(args, "format", "auto"),
        pattern=getattr(args, "pattern", None),
        settings=args,
    )


def _load_tasks_with_format(
    log_dir: str | Path,
    *,
    days: int | None = None,
    format_name: str = "auto",
    pattern: str | None = None,
    settings: argparse.Namespace | CliSettings | None = None,
) -> list[AgentTask]:
    ui = _settings(settings)
    path = Path(log_dir)
    if not path.exists():
        raise CliError(
            f"Path not found: {path}\n"
            f"Set AGENT_XRAY_LOG_DIR or pass a valid path.\n"
            f"Run 'agent-xray quickstart' for a demo."
        )
    _emit_verbose(
        f"Loading traces from {path} (format={format_name}, days={days if days is not None else 'all'}"
        + (f", pattern={pattern}" if pattern else "")
        + ")",
        ui,
    )
    started = perf_counter()
    if format_name != "auto":
        tasks = load_adapted_tasks(path, format=format_name, days=days)
    else:
        tasks = load_tasks(path, days=days, pattern=pattern)
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
            args.log_dir, days=args.days, format_name=args.format,
            pattern=getattr(args, "pattern", None), settings=args,
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
            try:
                rel = os.path.relpath(args.log_dir)
                short_path = rel if len(rel) < len(str(args.log_dir)) else str(args.log_dir)
            except ValueError:
                short_path = str(args.log_dir)
            lines.append(f"Run 'agent-xray grade {short_path} --json' for per-task details.")
            # Suggest domain-specific rules if most tasks are web/browser
            web_count = sum(
                1 for t in tasks if t.task_category == "web"
                or any(s.tool_name.startswith("browser_") for s in t.steps)
            )
            using_default_rules = rules.name == "default"
            if web_count > len(tasks) * 0.4 and using_default_rules:
                lines.append(
                    f"\nTip: {web_count}/{len(tasks)} tasks are browser tasks."
                    " Use '--rules browser_flow' for domain-specific grading."
                )
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
        elif getattr(args, "summary", False):
            output = format_diff_summary(data)
            # Append prompt diff if prompts differ
            prompt_diff_lines = data.get("prompt_diff") or []
            if prompt_diff_lines:
                output += "\n\n" + format_prompt_diff(data)
            _emit(output, args, final=True)
        else:
            lines: list[str] = []
            for key, section in data.items():
                if key == "prompt_diff":
                    # Use the formatted prompt diff instead of raw list
                    prompt_diff_lines = data.get("prompt_diff") or []
                    if prompt_diff_lines:
                        lines.append("")
                        lines.append(format_prompt_diff(data))
                    continue
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
            args.log_dir, days=args.days, format_name=args.format,
            pattern=getattr(args, "pattern", None), settings=args,
        )
        tasks = _apply_filters(args, tasks)
        if not tasks:
            _emit("No tasks remain after applying filters.", args, final=True)
            return 0
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
            log_path = args.log_dir
            sections: list[str] = [grade_text]

            # Root cause summary (the actionable headline)
            if failures:
                cause_counts = Counter(f.root_cause for f in failures)
                top_causes = ", ".join(
                    f"{cause} ({n})" for cause, n in cause_counts.most_common(5)
                )
                sections.append(f"\nTop issues: {top_causes}")

            # Broken tools warning
            tool_errors: dict[str, list[int]] = {}
            for task in tasks:
                for step in task.steps:
                    name = step.tool_name
                    if name:
                        tool_errors.setdefault(name, [0, 0])
                        tool_errors[name][1] += 1
                        if step.error:
                            tool_errors[name][0] += 1
            broken_tools = [
                (name, errs, total)
                for name, (errs, total) in tool_errors.items()
                if total >= 5 and errs / total >= 0.8
            ]
            if broken_tools:
                broken_tools.sort(key=lambda x: -x[1] / x[2])
                parts = [f"{n} ({e}/{t}={e*100//t}%)" for n, e, t in broken_tools[:5]]
                sections.append(f"Broken tools: {', '.join(parts)}")

            # Data completeness analysis
            from .completeness import check_completeness
            completeness = check_completeness(tasks)
            if completeness.warnings:
                sections.append(f"\n{completeness.format_text()}")

            # Hints — use relative path when shorter
            try:
                rel = os.path.relpath(log_path)
                short_path = rel if len(rel) < len(str(log_path)) else str(log_path)
            except ValueError:
                short_path = str(log_path)
            sections.append(
                f"\nNext: agent-xray report {short_path} actions  # what to fix first"
            )
            sections.append(
                f"      agent-xray surface <task-id> {short_path}  # inspect a task"
            )

            # Suggest domain-specific rules if most tasks are web/browser
            web_count = sum(
                1 for t in tasks if t.task_category == "web"
                or any(s.tool_name.startswith("browser_") for s in t.steps)
            )
            using_default_rules = rules.name == "default"
            if web_count > len(tasks) * 0.4 and using_default_rules:
                sections.append(
                    f"Tip: {web_count}/{len(tasks)} tasks are browser tasks."
                    " Use '--rules browser_flow' for domain-specific grading."
                )
            _emit("\n".join(sections), args, final=True)
        return 0

    return _run_command(args, _action)


def cmd_tree(args: argparse.Namespace) -> int:
    def _action() -> int:
        tasks = _apply_filters(args, _load(args))
        if not tasks:
            _emit("No tasks remain after applying filters.", args, final=True)
            return 0
        rules_path = getattr(args, "rules", None)
        if rules_path:
            rules = load_rules(rules_path)
            grades = grade_tasks(tasks, rules)
            enriched = enriched_tree_for_tasks(tasks, grades)
            if args.json:
                _dump(enriched)
            else:
                _emit(format_enriched_tree_text(enriched), args, final=True)
        else:
            enriched = enriched_tree_for_tasks(tasks)
            if args.json:
                _dump(enriched)
            else:
                _emit(format_enriched_tree_text(enriched), args, final=True)
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
    tasks = _apply_filters(args, tasks)
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

        if report_type == "overhead":
            from .baseline import (
                format_overhead_report,
                group_by_prompt_hash,
                load_baselines,
                measure_all_overhead,
                overhead_report_data,
            )

            baselines_dir = getattr(args, "baselines", None)
            if not baselines_dir:
                _emit("--baselines is required for the overhead report", args, final=True)
                return 1
            baselines = load_baselines(baselines_dir)
            if not baselines:
                _emit(f"No baselines found in {baselines_dir}", args, final=True)
                return 1
            grade_map = {g.task_id: g.grade for g in grades}
            results = measure_all_overhead(tasks, grade_map, baselines)
            hash_groups = group_by_prompt_hash(tasks, analyses, grade_map, baselines)
            if use_json:
                _dump(overhead_report_data(results, hash_groups))
            else:
                _emit(
                    _colorize_report_headers(
                        format_overhead_report(results, hash_groups), args
                    ),
                    args,
                    final=True,
                )
            return 0

        if report_type == "prompt-impact":
            from .baseline import (
                format_prompt_impact_report,
                group_by_prompt_hash,
                load_baselines,
                prompt_impact_data,
            )

            baselines_dir = getattr(args, "baselines", None)
            baselines = load_baselines(baselines_dir) if baselines_dir else None
            grade_map = {g.task_id: g.grade for g in grades}
            hash_groups = group_by_prompt_hash(tasks, analyses, grade_map, baselines)
            if use_json:
                _dump(prompt_impact_data(hash_groups))
            else:
                _emit(
                    _colorize_report_headers(
                        format_prompt_impact_report(hash_groups), args
                    ),
                    args,
                    final=True,
                )
            return 0

        bucket_minutes = _parse_bucket_arg(getattr(args, "bucket", "1h"))

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
            "timeline": lambda: report_timeline(tasks, grades, analyses, bucket_minutes),
            "spins": lambda: report_spins(tasks, analyses),
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
            "timeline": lambda: report_timeline_data(tasks, grades, analyses, bucket_minutes),
            "spins": lambda: report_spins_data(tasks, analyses),
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
            "timeline": lambda: report_timeline_markdown(tasks, grades, analyses, bucket_minutes),
            "spins": lambda: report_spins_markdown(tasks, analyses),
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


def cmd_completeness(args: argparse.Namespace) -> int:
    """Check data completeness of agent traces."""
    def _action() -> int:
        from .completeness import check_completeness
        tasks = _load_tasks_with_format(
            args.log_dir, days=args.days, format_name=args.format,
            pattern=getattr(args, "pattern", None), settings=args,
        )
        report = check_completeness(tasks)
        if args.json:
            _dump({
                "score_pct": report.score_pct,
                "dimensions_checked": report.dimensions_checked,
                "dimensions_ok": report.dimensions_ok,
                "warnings": [
                    {
                        "dimension": w.dimension,
                        "severity": w.severity,
                        "message": w.message,
                        "affected_pct": w.affected_pct,
                        "fix_hint": w.fix_hint,
                    }
                    for w in report.warnings
                ],
            })
        else:
            _emit(report.format_text(), args, final=True)
        return 0
    return _run_command(args, _action)


def cmd_diagnose(args: argparse.Namespace) -> int:
    """Classify failures and build a prioritized fix plan."""
    def _action() -> int:
        from .diagnose import build_fix_plan, format_fix_plan_text, validate_fix_targets
        from .root_cause import classify_failures
        tasks = _load_tasks_with_format(
            args.log_dir, days=args.days, format_name=args.format,
            pattern=getattr(args, "pattern", None), settings=args,
        )
        tasks = _apply_filters(args, tasks)
        if not tasks:
            _emit("No tasks remain after applying filters.", args, final=True)
            return 0
        rules = load_rules(getattr(args, "rules", "default"))
        grades = grade_tasks(tasks, rules)
        classifications = classify_failures(tasks, grades)
        plan = build_fix_plan(classifications)
        project_root = getattr(args, "project_root", None) or os.environ.get(
            "AGENT_XRAY_PROJECT_ROOT"
        )
        if project_root:
            validate_fix_targets(plan, project_root)
        if args.json:
            _dump([entry.to_dict() for entry in plan])
        else:
            _emit(format_fix_plan_text(plan), args, final=True)
        return 0
    return _run_command(args, _action)


def cmd_validate_targets(args: argparse.Namespace) -> int:
    """Validate that fix-plan target paths resolve to existing files."""
    from .diagnose import (
        CODE_EXTENSIONS,
        get_target_resolver,
        list_all_targets,
    )

    project_root = getattr(args, "project_root", None) or os.environ.get(
        "AGENT_XRAY_PROJECT_ROOT"
    )
    if not project_root:
        print("Error: --project-root is required (or set AGENT_XRAY_PROJECT_ROOT).")
        return 1

    root = Path(project_root)
    if not root.is_dir():
        print(f"Error: project root not found: {project_root}")
        return 1

    resolver_name = getattr(args, "resolver", None)
    resolver = get_target_resolver(resolver_name)
    all_targets = list_all_targets(resolver)

    total = 0
    valid = 0
    stale_details: list[str] = []
    lines = [f"TARGET VALIDATION (project root: {root})", "=" * 60]

    for cause in sorted(all_targets):
        targets = all_targets[cause]
        lines.append(f"  {cause}:")
        for target in targets:
            is_path = "/" in target and Path(target).suffix in CODE_EXTENSIONS
            if not is_path:
                # Non-path targets are always considered valid (descriptions)
                lines.append(f"    [OK]   {target}")
                total += 1
                valid += 1
                continue
            total += 1
            full_path = root / target
            if full_path.exists():
                lines.append(f"    [OK]   {target}")
                valid += 1
            else:
                lines.append(f"    [STALE] {target}  \u2190 NOT FOUND")
                stale_details.append(f"{cause}: {target}")

    stale = total - valid
    lines.append("")
    lines.append(f"Summary: {valid}/{total} targets valid, {stale} stale")
    if stale:
        lines.append("  Stale targets need updating in your TargetResolver.")

    print("\n".join(lines))
    return 1 if stale else 0


def cmd_watch(args: argparse.Namespace) -> int:
    """Watch a JSONL log file and print grades as tasks complete."""
    def _action() -> int:
        from .watch import watch_file
        watch_file(
            args.file,
            rules_path=getattr(args, "rules", None),
            poll_interval=getattr(args, "poll", 2.0),
            json_output=getattr(args, "json", False),
            color=not getattr(args, "no_color", False) and os.getenv("NO_COLOR") is None,
        )
        return 0
    return _run_command(args, _action)


def cmd_rules_list(args: argparse.Namespace) -> int:
    """List available built-in rulesets."""
    from importlib.resources import files as pkg_files
    rules_dir = Path(str(pkg_files("agent_xray.rules")))
    lines = ["Available rulesets:", ""]
    for path in sorted(rules_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            name = data.get("name", path.stem)
            desc = data.get("description", "")
            lines.append(f"  {name:20s} {desc}")
        except Exception:
            continue
    lines.extend([
        "",
        "Use: agent-xray grade <dir> --rules <name>",
        "Create custom: agent-xray rules init --base default > my_rules.json",
    ])
    print("\n".join(lines))
    return 0


def cmd_rules_show(args: argparse.Namespace) -> int:
    """Show a ruleset's full JSON."""
    rules = load_rules(args.name)
    data = {
        "name": rules.name,
        "description": rules.description,
        "signals": rules.signals,
        "grade_thresholds": rules.grade_thresholds,
        "golden_requirements": rules.golden_requirements,
    }
    print(json.dumps(data, indent=2))
    return 0


def cmd_rules_init(args: argparse.Namespace) -> int:
    """Scaffold a custom ruleset."""
    base_name = args.base or "default"
    scaffold = {
        "name": "my_custom_rules",
        "description": "Custom grading rules. Edit signals, thresholds, and requirements to match your agent.",
        "extends": base_name,
        "signals": [
            {
                "name": "example_custom_signal",
                "metric": "unique_tools",
                "gte": 5,
                "points": 1,
                "reason": "+1 used 5+ unique tools (customize this)"
            }
        ],
        "grade_thresholds": {},
        "golden_requirements": []
    }
    print(json.dumps(scaffold, indent=2))
    return 0


def _parse_since(value: str) -> datetime:
    """Parse a --since value into a datetime.

    Accepts relative durations like ``2h``, ``30m``, ``1d`` or ISO timestamps.
    """
    match = re.fullmatch(r"(\d+)\s*([smhd])", value.strip())
    if match:
        amount = int(match.group(1))
        unit = match.group(2)
        delta = {
            "s": timedelta(seconds=amount),
            "m": timedelta(minutes=amount),
            "h": timedelta(hours=amount),
            "d": timedelta(days=amount),
        }[unit]
        return datetime.now(timezone.utc) - delta
    # Try ISO timestamp
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        raise CliError(
            f"Invalid --since value: {value!r}\n"
            f"Use a relative duration (e.g. 2h, 30m, 1d) or an ISO timestamp."
        ) from None


def _task_timestamp(task: AgentTask) -> datetime | None:
    """Extract the earliest timestamp from a task."""
    for step in task.sorted_steps:
        ts = step.timestamp
        if ts:
            try:
                dt = datetime.fromisoformat(ts)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue
    if task.outcome and task.outcome.timestamp:
        try:
            dt = datetime.fromisoformat(task.outcome.timestamp)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            pass
    return None


def _filter_tasks(
    tasks: list[AgentTask],
    *,
    grade_filter: str | None = None,
    site_filter: str | None = None,
    outcome_filter: str | None = None,
    since_filter: str | None = None,
    grades: list[GradeResult] | None = None,
) -> list[AgentTask]:
    """Apply CLI filter flags to a task list.

    When ``grade_filter`` is used and ``grades`` is not provided, tasks are
    graded on the fly with the default ruleset.
    """
    filtered = list(tasks)

    if since_filter:
        cutoff = _parse_since(since_filter)
        kept: list[AgentTask] = []
        for task in filtered:
            ts = _task_timestamp(task)
            if ts is None or ts >= cutoff:
                kept.append(task)
        filtered = kept

    if site_filter:
        site_lower = site_filter.lower()
        filtered = [
            task for task in filtered
            if site_lower in analyze_task(task).site_name.lower()
        ]

    if outcome_filter:
        outcome_lower = outcome_filter.lower()
        filtered = [
            task for task in filtered
            if task.outcome is not None and outcome_lower in task.outcome.status.lower()
        ]

    if grade_filter:
        allowed = {g.strip().upper() for g in grade_filter.split(",")}
        if grades is None:
            rules = load_rules()
            grades = grade_tasks(filtered, rules)
        grade_map = {g.task_id: g.grade for g in grades}
        filtered = [
            task for task in filtered
            if grade_map.get(task.task_id, "") in allowed
        ]

    return filtered


def _apply_filters(
    args: argparse.Namespace,
    tasks: list[AgentTask],
    grades: list[GradeResult] | None = None,
) -> list[AgentTask]:
    """Convenience wrapper: read filter flags from args and apply them."""
    return _filter_tasks(
        tasks,
        grade_filter=getattr(args, "grade_filter", None),
        site_filter=getattr(args, "site_filter", None),
        outcome_filter=getattr(args, "outcome_filter", None),
        since_filter=getattr(args, "since_filter", None),
        grades=grades,
    )


def cmd_search(args: argparse.Namespace) -> int:
    """Search tasks by user_text substring."""

    def _action() -> int:
        tasks = _load(args)
        query = args.query.lower()
        matches: list[dict[str, Any]] = []
        grade_map: dict[str, str] = {}
        grade_filter = getattr(args, "grade_filter", None)

        if grade_filter:
            rules = load_rules()
            grades = grade_tasks(tasks, rules)
            grade_map = {g.task_id: g.grade for g in grades}
            allowed = {g.strip().upper() for g in grade_filter.split(",")}
        else:
            allowed = None

        for task in tasks:
            text = task.task_text or ""
            if query not in text.lower():
                continue
            if allowed and grade_map.get(task.task_id, "") not in allowed:
                continue
            analysis = analyze_task(task)
            entry: dict[str, Any] = {
                "task_id": task.task_id,
                "grade": grade_map.get(task.task_id, ""),
                "outcome": task.outcome.status if task.outcome else "",
                "step_count": len(task.steps),
                "site": analysis.site_name,
                "user_text": text[:80],
            }
            matches.append(entry)

        if not matches:
            _emit(f"No tasks matching {args.query!r}.", args, final=True)
            return 0

        if args.json:
            _dump(matches)
        else:
            lines: list[str] = [f"Found {len(matches)} task(s) matching {args.query!r}:", ""]
            for entry in matches:
                grade_str = f" [{entry['grade']}]" if entry["grade"] else ""
                outcome_str = f" ({entry['outcome']})" if entry["outcome"] else ""
                lines.append(
                    f"  {entry['task_id']}{grade_str}{outcome_str}"
                    f"  steps={entry['step_count']}  site={entry['site']}"
                )
                lines.append(f"    {entry['user_text']}")
            _emit("\n".join(lines), args, final=True)
        return 0

    return _run_command(args, _action)


def cmd_golden(args: argparse.Namespace) -> int:
    """Golden exemplar ranking subcommands."""
    from .golden import (
        OPTIMIZATION_PROFILES,
        capture_exemplar,
        find_exemplars,
        format_golden_ranking,
        rank_golden_runs,
    )
    from .replay import load_fixture

    subcmd = getattr(args, "golden_command", None)

    if subcmd == "rank":
        def _action() -> int:
            tasks = _load(args)
            rules = load_rules(getattr(args, "rules", None))
            optimize = getattr(args, "optimize", "balanced")
            site_filter = getattr(args, "site_filter", None)
            rankings = rank_golden_runs(tasks, rules=rules, optimize=optimize)
            if site_filter:
                site_lower = site_filter.lower()
                rankings = {
                    s: r for s, r in rankings.items() if site_lower in s.lower()
                }
            if args.json:
                _dump({s: [r.to_dict() for r in rs] for s, rs in rankings.items()})
            else:
                _emit(format_golden_ranking(rankings, optimize), args, final=True)
            return 0

        return _run_command(args, _action)

    if subcmd == "best":
        def _action() -> int:
            tasks = _load(args)
            rules = load_rules(getattr(args, "rules", None))
            optimize = getattr(args, "optimize", "balanced")
            exemplars = find_exemplars(tasks, rules=rules, optimize=optimize)
            if args.json:
                _dump([e.to_dict() for e in exemplars])
            else:
                if not exemplars:
                    _emit("No golden/good runs found.", args, final=True)
                else:
                    lines: list[str] = ["EXEMPLARS (best run per site)", "=" * 40]
                    for e in exemplars:
                        lines.append(
                            f"  {e.site_name}: {e.task_id[:12]}  eff={e.efficiency:.2f}  "
                            f"{e.step_count} steps  {e.duration_s:.0f}s  "
                            f"${e.cost_usd:.2f}  {e.flow_summary}"
                        )
                    _emit("\n".join(lines), args, final=True)
            return 0

        return _run_command(args, _action)

    if subcmd == "capture":
        def _action() -> int:
            tasks = _load(args)
            rules = load_rules(getattr(args, "rules", None))
            optimize = getattr(args, "optimize", "balanced")
            site = getattr(args, "site_name", None)
            output = getattr(args, "out", None)
            path = capture_exemplar(
                tasks, rules=rules, site=site, optimize=optimize, output_path=output,
            )
            if args.json:
                _dump({"fixture": str(path)})
            else:
                _emit(str(path), args, final=True)
            return 0

        return _run_command(args, _action)

    if subcmd == "compare":
        def _action() -> int:
            tasks = _load(args)
            rules = load_rules(getattr(args, "rules", None))
            fixtures_dir = Path(args.fixtures)
            if not fixtures_dir.exists():
                raise CliError(f"Fixtures directory not found: {fixtures_dir}")
            optimize = getattr(args, "optimize", "balanced")
            rankings = rank_golden_runs(tasks, rules=rules, optimize=optimize)
            results: list[dict[str, Any]] = []
            for fixture_path in sorted(fixtures_dir.glob("*.json")):
                try:
                    fixture = load_fixture(fixture_path)
                except Exception:
                    continue
                fixture_site = str(fixture.get("site", ""))
                fixture_steps = int(fixture.get("total_steps", 0) or 0)
                site_ranks = rankings.get(fixture_site, [])
                if site_ranks:
                    best = site_ranks[0]
                    step_delta = best.step_count - fixture_steps
                    results.append({
                        "fixture": fixture_path.name,
                        "site": fixture_site,
                        "fixture_steps": fixture_steps,
                        "current_best_task": best.task_id,
                        "current_best_steps": best.step_count,
                        "current_best_efficiency": round(best.efficiency, 4),
                        "step_delta": step_delta,
                        "verdict": (
                            "IMPROVED" if step_delta < -2
                            else "REGRESSION" if step_delta > 5
                            else "STABLE"
                        ),
                    })
                else:
                    results.append({
                        "fixture": fixture_path.name,
                        "site": fixture_site,
                        "fixture_steps": fixture_steps,
                        "current_best_task": None,
                        "current_best_steps": None,
                        "current_best_efficiency": None,
                        "step_delta": None,
                        "verdict": "UNMATCHED",
                    })
            if args.json:
                _dump(results)
            else:
                lines: list[str] = ["GOLDEN COMPARE (fixtures vs current)", "=" * 50]
                for r in results:
                    if r["current_best_task"]:
                        lines.append(
                            f"  {r['fixture']:<30s} {r['verdict']:<12s} "
                            f"fixture={r['fixture_steps']}  "
                            f"current={r['current_best_steps']}  "
                            f"eff={r['current_best_efficiency']:.2f}"
                        )
                    else:
                        lines.append(
                            f"  {r['fixture']:<30s} UNMATCHED  fixture={r['fixture_steps']}"
                        )
                _emit("\n".join(lines), args, final=True)
            return 0

        return _run_command(args, _action)

    if subcmd == "profiles":
        lines: list[str] = ["OPTIMIZATION PROFILES", "=" * 50]
        for name, weights in sorted(OPTIMIZATION_PROFILES.items()):
            parts = ", ".join(f"{k}={v:.1f}" for k, v in sorted(weights.items()))
            lines.append(f"  {name:<12s}  {parts}")
        _emit("\n".join(lines), args, final=True)
        return 0

    _emit("Usage: agent-xray golden {rank,best,capture,compare,profiles}", args, final=True)
    return 1


def _add_filter_options(parser: argparse.ArgumentParser) -> None:
    """Add shared filter flags to a subparser."""
    filter_group = parser.add_argument_group("filters")
    filter_group.add_argument(
        "--grade", dest="grade_filter", default=None,
        help="Only include tasks with this grade (comma-separated: BROKEN,WEAK)",
    )
    filter_group.add_argument(
        "--site", dest="site_filter", default=None,
        help="Only include tasks whose site_name matches (substring, case-insensitive)",
    )
    filter_group.add_argument(
        "--outcome", dest="outcome_filter", default=None,
        help="Only include tasks with this outcome status (substring, case-insensitive)",
    )
    filter_group.add_argument(
        "--since", dest="since_filter", default=None,
        help="Only include tasks after this time (e.g. 2h, 30m, 1d, or ISO timestamp)",
    )


def _add_format_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format",
        choices=FORMAT_CHOICES,
        default="auto",
        help="Trace log format (default: auto-detect)",
    )


def _add_pattern_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pattern",
        help="Glob pattern to filter .jsonl files (e.g. 'agent-steps-*.jsonl'). "
        "Without this, auto-detects files containing agent traces.",
    )


def cmd_pricing(args: argparse.Namespace) -> int:
    """Manage the model pricing database."""
    from .pricing import (
        format_model_pricing,
        load_pricing,
        pricing_source,
        update_pricing_cache,
    )

    subcmd = getattr(args, "pricing_command", None)

    if subcmd == "list":
        custom = getattr(args, "pricing", None)
        pricing_data = load_pricing(custom)
        models = pricing_data.get("models", {})
        header = f"{'Model':<40} {'Input/1M':>10} {'Output/1M':>10} {'Cached/1M':>10}"
        _emit(header, args)
        _emit("-" * len(header), args)
        for model_name in sorted(models):
            entry = models[model_name]
            cached = f"${entry['cached_input']:.4f}" if "cached_input" in entry else "  -"
            _emit(
                f"{model_name:<40} ${entry.get('input', 0.0):>9.4f}"
                f" ${entry.get('output', 0.0):>9.4f} {cached:>10}",
                args,
            )
        aliases = pricing_data.get("aliases", {})
        if aliases:
            _emit("", args)
            _emit(f"Aliases ({len(aliases)}):", args)
            for alias, target in sorted(aliases.items()):
                _emit(f"  {alias} -> {target}", args)
        _emit("", args)
        _emit(f"Total: {len(models)} models, {len(aliases)} aliases", args, final=True)
        return 0

    if subcmd == "show":
        custom = getattr(args, "pricing", None)
        text = format_model_pricing(args.model_name, load_pricing(custom))
        _emit(text, args, final=True)
        return 0

    if subcmd == "update":
        ok, msg = update_pricing_cache()
        _emit(msg, args, final=True)
        return 0 if ok else 1

    if subcmd == "path":
        custom = getattr(args, "pricing", None)
        _emit(pricing_source(custom), args, final=True)
        return 0

    # No sub-subcommand: show help
    _emit("Usage: agent-xray pricing {list,show,update,path}", args, final=True)
    return 1


def cmd_baseline(args: argparse.Namespace) -> int:
    def _action() -> int:
        from .baseline import (
            build_baseline,
            format_overhead_report,
            format_prompt_impact_report,
            generate_naked_prompt,
            group_by_prompt_hash,
            load_baselines,
            measure_all_overhead,
            overhead_report_data,
            prompt_impact_data,
            save_baseline,
        )

        subcmd = getattr(args, "baseline_command", None)
        use_json = getattr(args, "json", False)

        if subcmd == "capture":
            tasks = _load(args)
            task = resolve_task(tasks, args.task_id)
            analysis = analyze_task(task)
            baseline = build_baseline(task, analysis)
            out = Path(args.out) if args.out else Path.cwd() / "baselines" / f"{analysis.site_name}.json"
            path = save_baseline(baseline, out)
            if use_json:
                _dump({"baseline": str(path), **baseline.to_dict()})
            else:
                _emit(f"Baseline saved to {path}", args, final=True)
            return 0

        if subcmd == "generate":
            tasks = _load(args)
            task = resolve_task(tasks, args.task_id)
            prompt = generate_naked_prompt(task)
            if use_json:
                _dump({"task_id": task.task_id, "naked_prompt": prompt})
            else:
                _emit(prompt, args, final=True)
            return 0

        if subcmd == "list":
            baselines_dir = Path(args.baselines_dir)
            baselines = load_baselines(baselines_dir)
            if use_json:
                _dump({name: bl.to_dict() for name, bl in baselines.items()})
            else:
                if not baselines:
                    _emit(f"No baselines found in {baselines_dir}", args, final=True)
                else:
                    lines = [f"{'Site':<20} {'Steps':>6} {'Duration':>10} {'Cost':>10}"]
                    lines.append("-" * 50)
                    for name, bl in sorted(baselines.items()):
                        lines.append(
                            f"{name:<20} {bl.step_count:>6} {bl.duration_s:>9.1f}s ${bl.cost_usd:>8.4f}"
                        )
                    _emit("\n".join(lines), args, final=True)
            return 0

        _emit("Usage: agent-xray baseline {capture,generate,list}", args, final=True)
        return 1

    return _run_command(args, _action)


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
    p_analyze.add_argument("log_dir", help="Directory or .jsonl file containing agent traces")
    p_analyze.add_argument("--days", type=int, help="Include only the N most recent days of traces")
    p_analyze.add_argument(
        "--rules",
        help="Ruleset name (default, browser_flow, coding_agent, research_agent) or path to JSON",
    )
    _add_format_option(p_analyze)
    _add_pattern_option(p_analyze)
    p_analyze.add_argument("--json", action="store_true", help="Output results as JSON")
    p_analyze.set_defaults(func=cmd_analyze)

    for name, handler, example in (
        ("surface", cmd_surface, "agent-xray surface task-123 ./traces"),
        ("reasoning", cmd_reasoning, "agent-xray reasoning task-123 ./traces --json"),
    ):
        parser_ = _add_subparser(
            sub,
            name,
            help_text=f"{name.title()} output for a task",
            example=example,
        )
        parser_.add_argument("task_id", help="Task ID or prefix to search for")
        parser_.add_argument(
            "log_dir_pos", nargs="?", default=None,
            help="Directory or .jsonl file (also accepted via --log-dir). "
            "Defaults to AGENT_XRAY_LOG_DIR env var if set.",
        )
        parser_.add_argument(
            "--log-dir", dest="log_dir_opt",
            help="Directory or .jsonl file containing agent traces",
        )
        parser_.add_argument(
            "--days", type=int, help="Include only the N most recent days of traces"
        )
        _add_format_option(parser_)
        _add_pattern_option(parser_)
        parser_.add_argument("--json", action="store_true", help="Output results as JSON")
        parser_.set_defaults(func=handler)

    p_diff = _add_subparser(
        sub,
        "diff",
        help_text="Compare two tasks",
        example="agent-xray diff task-123 task-124 ./traces",
    )
    p_diff.add_argument("task_id_1", help="First task ID to compare")
    p_diff.add_argument("task_id_2", help="Second task ID to compare")
    p_diff.add_argument(
        "log_dir_pos", nargs="?", default=None,
        help="Directory or .jsonl file (also accepted via --log-dir). "
        "Defaults to AGENT_XRAY_LOG_DIR env var if set.",
    )
    p_diff.add_argument(
        "--log-dir", dest="log_dir_opt",
        help="Directory or .jsonl file containing agent traces (alias for positional arg)",
    )
    p_diff.add_argument("--days", type=int, help="Include only the N most recent days of traces")
    _add_format_option(p_diff)
    _add_pattern_option(p_diff)
    p_diff.add_argument("--json", action="store_true", help="Output results as JSON")
    p_diff.add_argument("--summary", action="store_true", help="Show concise side-by-side comparison instead of full diff")
    p_diff.set_defaults(func=cmd_diff)

    p_grade = _add_subparser(
        sub,
        "grade",
        help_text="Grade a log directory",
        example="agent-xray grade ./traces --rules browser_flow",
    )
    p_grade.add_argument("log_dir", help="Directory or .jsonl file containing agent traces")
    p_grade.add_argument(
        "--rules",
        default=str(DEFAULT_RULES_PATH),
        help="Ruleset name (default, browser_flow, coding_agent, research_agent) or path to JSON",
    )
    p_grade.add_argument("--days", type=int, help="Include only the N most recent days of traces")
    _add_format_option(p_grade)
    _add_pattern_option(p_grade)
    _add_filter_options(p_grade)
    p_grade.add_argument("--json", action="store_true", help="Output results as JSON")
    p_grade.set_defaults(func=cmd_grade)

    p_tree = _add_subparser(
        sub,
        "tree",
        help_text="Show a day/site/task tree",
        example="agent-xray tree ./traces",
    )
    p_tree.add_argument(
        "log_dir", nargs="?", default=None,
        help="Directory or .jsonl file containing agent traces",
    )
    p_tree.add_argument(
        "--log-dir", dest="log_dir_opt", help="Directory or .jsonl file (alternative to positional)",
    )
    p_tree.add_argument("--days", type=int, help="Include only the N most recent days of traces")
    p_tree.add_argument(
        "--rules",
        help="Ruleset name or path — enriches tree with grades and scores",
    )
    _add_format_option(p_tree)
    _add_pattern_option(p_tree)
    _add_filter_options(p_tree)
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
        "--log-dir", dest="log_dir_opt", help="Directory or .jsonl file containing agent traces"
    )
    p_capture.add_argument("--days", type=int, help="Include only the N most recent days of traces")
    _add_format_option(p_capture)
    _add_pattern_option(p_capture)
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
        "--log-dir", dest="log_dir_opt", help="Directory or .jsonl file containing agent traces"
    )
    p_replay.add_argument("--days", type=int, help="Include only the N most recent days of traces")
    _add_format_option(p_replay)
    _add_pattern_option(p_replay)
    p_replay.add_argument("--json", action="store_true", help="Output results as JSON")
    p_replay.set_defaults(func=cmd_replay)

    p_flywheel = _add_subparser(
        sub,
        "flywheel",
        help_text="Run end-to-end grading, root-cause analysis, and baseline comparison",
        example="agent-xray flywheel ./traces --baseline ./baseline.json --json",
    )
    p_flywheel.add_argument("log_dir", help="Directory or .jsonl file containing agent traces")
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
    _add_pattern_option(p_flywheel)
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
        help_text="Generate a report (health, golden, broken, tools, flows, outcomes, actions, coding, research, cost, fixes, timeline, spins, compare)",
        example="agent-xray report ./traces health",
    )
    p_report.add_argument("log_dir", help="Directory or .jsonl file containing agent traces")
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
            "timeline",
            "spins",
            "compare",
            "overhead",
            "prompt-impact",
        ],
        help="Type of report to generate",
    )
    p_report.add_argument(
        "--rules",
        help="Ruleset name (default, browser_flow, coding_agent, research_agent) or path to JSON",
    )
    p_report.add_argument("--days", type=int, help="Include only the N most recent days of traces")
    p_report.add_argument(
        "--bucket",
        default="1h",
        help="Time bucket for timeline report (e.g. '15m', '1h'). Default: 1h",
    )
    _add_format_option(p_report)
    _add_pattern_option(p_report)
    _add_filter_options(p_report)
    p_report.add_argument("--day1", help="First day for compare report (YYYYMMDD)")
    p_report.add_argument("--day2", help="Second day for compare report (YYYYMMDD)")
    p_report.add_argument(
        "--baselines",
        help="Directory containing baseline JSON files (for overhead report)",
    )
    p_report.add_argument("--json", action="store_true", help="Output results as JSON")
    p_report.add_argument("--markdown", action="store_true", help="Output results as Markdown")
    p_report.set_defaults(func=cmd_report)

    p_completeness = _add_subparser(
        sub,
        "completeness",
        help_text="Check data completeness of agent traces",
        example="agent-xray completeness ./traces --json",
    )
    p_completeness.add_argument("log_dir", help="Directory or .jsonl file containing agent traces")
    p_completeness.add_argument("--days", type=int, help="Include only the N most recent days of traces")
    _add_format_option(p_completeness)
    _add_pattern_option(p_completeness)
    p_completeness.add_argument("--json", action="store_true", help="Output results as JSON")
    p_completeness.set_defaults(func=cmd_completeness)

    p_diagnose = _add_subparser(
        sub,
        "diagnose",
        help_text="Classify failures and build a prioritized fix plan",
        example="agent-xray diagnose ./traces --json",
    )
    p_diagnose.add_argument("log_dir", help="Directory or .jsonl file containing agent traces")
    p_diagnose.add_argument("--days", type=int, help="Include only the N most recent days of traces")
    p_diagnose.add_argument("--rules", default="default", help="Ruleset name for grading (default: default)")
    _add_format_option(p_diagnose)
    _add_pattern_option(p_diagnose)
    _add_filter_options(p_diagnose)
    p_diagnose.add_argument("--json", action="store_true", help="Output results as JSON")
    p_diagnose.add_argument(
        "--project-root",
        help="Project root to validate fix target paths against (env: AGENT_XRAY_PROJECT_ROOT)",
    )
    p_diagnose.set_defaults(func=cmd_diagnose)

    p_validate_targets = _add_subparser(
        sub,
        "validate-targets",
        help_text="Validate fix-plan target paths against a project directory",
        example="agent-xray validate-targets --project-root /path/to/project",
    )
    p_validate_targets.add_argument(
        "--project-root",
        help="Project root to validate fix target paths against (env: AGENT_XRAY_PROJECT_ROOT)",
    )
    p_validate_targets.add_argument(
        "--resolver",
        help="Named target resolver to use (default: active resolver)",
    )
    p_validate_targets.set_defaults(func=cmd_validate_targets)

    p_search = _add_subparser(
        sub,
        "search",
        help_text="Search tasks by user_text",
        example='agent-xray search "pizza" ./traces',
    )
    p_search.add_argument("query", help="Search string (case-insensitive substring match on user_text)")
    p_search.add_argument(
        "log_dir_pos", nargs="?", default=None,
        help="Directory or .jsonl file. Defaults to AGENT_XRAY_LOG_DIR env var if set.",
    )
    p_search.add_argument(
        "--log-dir", dest="log_dir_opt",
        help="Directory or .jsonl file containing agent traces (alias for positional arg)",
    )
    p_search.add_argument("--days", type=int, help="Include only the N most recent days of traces")
    _add_format_option(p_search)
    _add_pattern_option(p_search)
    p_search.add_argument(
        "--grade", dest="grade_filter", default=None,
        help="Only include tasks with this grade (comma-separated: BROKEN,WEAK)",
    )
    p_search.add_argument("--json", action="store_true", help="Output results as JSON")
    p_search.set_defaults(func=cmd_search)

    p_tui = _add_subparser(
        sub,
        "tui",
        help_text="Open the interactive decision-surface inspector",
        example="agent-xray tui ./traces --task-id task-123",
    )
    p_tui.add_argument("log_dir", help="Trace log directory or jsonl file to inspect")
    p_tui.add_argument("--task-id", help="Specific task id to open. Defaults to the latest task.")
    p_tui.set_defaults(func=cmd_tui)

    p_watch = _add_subparser(
        sub,
        "watch",
        help_text="Tail a JSONL log file and grade tasks as they complete in real-time",
        example="agent-xray watch path/to/agent-steps-20260328.jsonl --rules browser_flow",
    )
    p_watch.add_argument("file", help="Path to a JSONL log file to watch")
    p_watch.add_argument(
        "--rules",
        help="Ruleset name (default, browser_flow, coding_agent, research_agent) or path to JSON",
    )
    p_watch.add_argument(
        "--poll",
        type=float,
        default=2.0,
        help="Poll interval in seconds (default: 2.0)",
    )
    p_watch.add_argument("--json", action="store_true", help="Output one JSON object per completed task")
    p_watch.set_defaults(func=cmd_watch)

    # Baseline management subcommands
    p_baseline = _add_subparser(
        sub,
        "baseline",
        help_text="Capture, generate, or list baselines for overhead measurement",
        example="agent-xray baseline capture task-123 ./traces -o baselines/dominos.json",
    )
    baseline_sub = p_baseline.add_subparsers(dest="baseline_command")

    p_bl_capture = baseline_sub.add_parser("capture", help="Capture a task as a baseline")
    p_bl_capture.add_argument("task_id", help="Task id to capture")
    p_bl_capture.add_argument(
        "log_dir_pos", nargs="?", default=None,
        help="Directory or .jsonl file containing agent traces",
    )
    p_bl_capture.add_argument(
        "--log-dir", dest="log_dir_opt",
        help="Directory or .jsonl file containing agent traces (alias for positional arg)",
    )
    p_bl_capture.add_argument("-o", "--out", help="Output JSON file path")
    p_bl_capture.add_argument("--days", type=int, help="Include only the N most recent days of traces")
    _add_format_option(p_bl_capture)
    _add_pattern_option(p_bl_capture)
    p_bl_capture.add_argument("--json", action="store_true", help="Output results as JSON")
    p_bl_capture.set_defaults(func=cmd_baseline)

    p_bl_generate = baseline_sub.add_parser("generate", help="Print the naked prompt for a task")
    p_bl_generate.add_argument("task_id", help="Task id to generate prompt for")
    p_bl_generate.add_argument(
        "log_dir_pos", nargs="?", default=None,
        help="Directory or .jsonl file containing agent traces",
    )
    p_bl_generate.add_argument(
        "--log-dir", dest="log_dir_opt",
        help="Directory or .jsonl file containing agent traces (alias for positional arg)",
    )
    p_bl_generate.add_argument("--days", type=int, help="Include only the N most recent days of traces")
    _add_format_option(p_bl_generate)
    _add_pattern_option(p_bl_generate)
    p_bl_generate.add_argument("--json", action="store_true", help="Output results as JSON")
    p_bl_generate.set_defaults(func=cmd_baseline)

    p_bl_list = baseline_sub.add_parser("list", help="List all baselines in a directory")
    p_bl_list.add_argument("baselines_dir", help="Directory containing baseline JSON files")
    p_bl_list.add_argument("--json", action="store_true", help="Output results as JSON")
    p_bl_list.set_defaults(func=cmd_baseline)

    p_baseline.set_defaults(func=cmd_baseline)

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

    # Rules management subcommands
    p_rules = _add_subparser(
        sub,
        "rules",
        help_text="List, show, or scaffold rulesets",
        example="agent-xray rules list",
    )
    rules_sub = p_rules.add_subparsers(dest="rules_command")

    p_rules_list = rules_sub.add_parser("list", help="List available built-in rulesets")
    p_rules_list.set_defaults(func=cmd_rules_list)

    p_rules_show = rules_sub.add_parser("show", help="Show a ruleset's full JSON")
    p_rules_show.add_argument("name", help="Ruleset name (e.g. default, browser_flow)")
    p_rules_show.set_defaults(func=cmd_rules_show)

    p_rules_init = rules_sub.add_parser("init", help="Scaffold a custom ruleset to stdout")
    p_rules_init.add_argument("--base", default="default", help="Base ruleset to extend (default: default)")
    p_rules_init.set_defaults(func=cmd_rules_init)

    # Default: show help when no sub-subcommand given
    p_rules.set_defaults(func=lambda args: (p_rules.print_help(), 0)[1])

    # Pricing management subcommands
    p_pricing = _add_subparser(
        sub,
        "pricing",
        help_text="List, show, or update model pricing data",
        example="agent-xray pricing list",
    )
    pricing_sub = p_pricing.add_subparsers(dest="pricing_command")

    p_pricing_list = pricing_sub.add_parser("list", help="Show all known models and prices")
    p_pricing_list.add_argument(
        "--pricing", help="Path to custom pricing JSON file",
    )
    p_pricing_list.set_defaults(func=cmd_pricing)

    p_pricing_show = pricing_sub.add_parser("show", help="Show pricing for a specific model")
    p_pricing_show.add_argument("model_name", help="Model name (e.g. gpt-4.1-nano)")
    p_pricing_show.add_argument(
        "--pricing", help="Path to custom pricing JSON file",
    )
    p_pricing_show.set_defaults(func=cmd_pricing)

    p_pricing_update = pricing_sub.add_parser("update", help="Fetch latest pricing from GitHub")
    p_pricing_update.set_defaults(func=cmd_pricing)

    p_pricing_path = pricing_sub.add_parser("path", help="Show where pricing data is loaded from")
    p_pricing_path.add_argument(
        "--pricing", help="Path to custom pricing JSON file",
    )
    p_pricing_path.set_defaults(func=cmd_pricing)

    # Default: show help when no sub-subcommand given
    p_pricing.set_defaults(func=cmd_pricing)

    # Golden exemplar ranking subcommands
    p_golden = _add_subparser(
        sub,
        "golden",
        help_text="Rank, inspect, and capture golden exemplar runs",
        example="agent-xray golden rank ./traces --optimize balanced",
    )
    golden_sub = p_golden.add_subparsers(dest="golden_command")

    p_golden_rank = golden_sub.add_parser(
        "rank", help="Rank golden/good runs by efficiency within each site"
    )
    p_golden_rank.add_argument(
        "log_dir_pos", nargs="?", default=None,
        help="Directory or .jsonl file. Defaults to AGENT_XRAY_LOG_DIR env var if set.",
    )
    p_golden_rank.add_argument(
        "--log-dir", dest="log_dir_opt",
        help="Directory or .jsonl file containing agent traces",
    )
    p_golden_rank.add_argument("--rules", help="Ruleset name or path to JSON")
    p_golden_rank.add_argument(
        "--optimize", default="balanced",
        help="Optimization profile: balanced, cost, speed, steps (default: balanced)",
    )
    p_golden_rank.add_argument(
        "--site", dest="site_filter",
        help="Only show rankings for this site (substring, case-insensitive)",
    )
    p_golden_rank.add_argument("--days", type=int, help="Include only the N most recent days")
    _add_format_option(p_golden_rank)
    _add_pattern_option(p_golden_rank)
    p_golden_rank.add_argument("--json", action="store_true", help="Output as JSON")
    p_golden_rank.set_defaults(func=cmd_golden)

    p_golden_best = golden_sub.add_parser(
        "best", help="Show the top exemplar for each site"
    )
    p_golden_best.add_argument(
        "log_dir_pos", nargs="?", default=None,
        help="Directory or .jsonl file. Defaults to AGENT_XRAY_LOG_DIR env var if set.",
    )
    p_golden_best.add_argument(
        "--log-dir", dest="log_dir_opt",
        help="Directory or .jsonl file containing agent traces",
    )
    p_golden_best.add_argument("--rules", help="Ruleset name or path to JSON")
    p_golden_best.add_argument(
        "--optimize", default="balanced",
        help="Optimization profile (default: balanced)",
    )
    p_golden_best.add_argument("--days", type=int, help="Include only the N most recent days")
    _add_format_option(p_golden_best)
    _add_pattern_option(p_golden_best)
    p_golden_best.add_argument("--json", action="store_true", help="Output as JSON")
    p_golden_best.set_defaults(func=cmd_golden)

    p_golden_capture = golden_sub.add_parser(
        "capture", help="Capture the exemplar for a site as a fixture"
    )
    p_golden_capture.add_argument(
        "log_dir_pos", nargs="?", default=None,
        help="Directory or .jsonl file. Defaults to AGENT_XRAY_LOG_DIR env var if set.",
    )
    p_golden_capture.add_argument(
        "--log-dir", dest="log_dir_opt",
        help="Directory or .jsonl file containing agent traces",
    )
    p_golden_capture.add_argument("--rules", help="Ruleset name or path to JSON")
    p_golden_capture.add_argument(
        "--optimize", default="balanced",
        help="Optimization profile (default: balanced)",
    )
    p_golden_capture.add_argument(
        "--site", dest="site_name", help="Site name to capture exemplar for"
    )
    p_golden_capture.add_argument("--out", "-o", help="Output file path")
    p_golden_capture.add_argument("--days", type=int, help="Include only the N most recent days")
    _add_format_option(p_golden_capture)
    _add_pattern_option(p_golden_capture)
    p_golden_capture.add_argument("--json", action="store_true", help="Output as JSON")
    p_golden_capture.set_defaults(func=cmd_golden)

    p_golden_compare = golden_sub.add_parser(
        "compare", help="Compare current golden runs against captured fixtures"
    )
    p_golden_compare.add_argument(
        "log_dir_pos", nargs="?", default=None,
        help="Directory or .jsonl file. Defaults to AGENT_XRAY_LOG_DIR env var if set.",
    )
    p_golden_compare.add_argument(
        "--log-dir", dest="log_dir_opt",
        help="Directory or .jsonl file containing agent traces",
    )
    p_golden_compare.add_argument(
        "--fixtures", required=True,
        help="Directory containing golden fixture files",
    )
    p_golden_compare.add_argument("--rules", help="Ruleset name or path to JSON")
    p_golden_compare.add_argument(
        "--optimize", default="balanced",
        help="Optimization profile (default: balanced)",
    )
    p_golden_compare.add_argument("--days", type=int, help="Include only the N most recent days")
    _add_format_option(p_golden_compare)
    _add_pattern_option(p_golden_compare)
    p_golden_compare.add_argument("--json", action="store_true", help="Output as JSON")
    p_golden_compare.set_defaults(func=cmd_golden)

    p_golden_profiles = golden_sub.add_parser(
        "profiles", help="List available optimization profiles"
    )
    p_golden_profiles.set_defaults(func=cmd_golden)

    # Default: show help when no sub-subcommand given
    p_golden.set_defaults(func=lambda args: (p_golden.print_help(), 0)[1])

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
