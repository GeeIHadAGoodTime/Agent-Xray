from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .analyzer import analyze_task, detect_signals, resolve_task
from .schema import AgentStep, AgentTask

EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")
PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")
CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
ZIP_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")
ADDRESS_RE = re.compile(
    r"\b\d{1,5}\s+[A-Za-z0-9.\- ]+\b(?:street|st|avenue|ave|road|rd|drive|dr|lane|ln)\b", re.I
)


def _sanitize_text(value: str) -> str:
    text = EMAIL_RE.sub("*email*", value)
    text = PHONE_RE.sub("*phone*", text)
    text = CARD_RE.sub("*card_number*", text)
    text = ZIP_RE.sub("*zip*", text)
    text = ADDRESS_RE.sub("*address*", text)
    return text


def _sanitize_url(value: str) -> str:
    if not value:
        return value
    parsed = urlparse(value)
    if not parsed.scheme and not parsed.netloc:
        parsed = urlparse(f"https://{value}")
    host = "shop.example.test"
    path = parsed.path or "/"
    if parsed.fragment:
        path = f"{path}#{parsed.fragment}"
    return f"https://{host}{path}"


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, str):
        if value.startswith("http://") or value.startswith("https://"):
            return _sanitize_url(value)
        return _sanitize_text(value)
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if key.lower() in {"url", "page_url"} and isinstance(item, str):
                sanitized[key] = _sanitize_url(item)
            else:
                sanitized[key] = _sanitize_value(item)
        return sanitized
    return value


def detect_milestone(step: AgentStep) -> str | None:
    signals = detect_signals(step)
    if signals["payment_fields_strong"]:
        return "PAYMENT"
    if signals["payment_fields"]:
        return "PAYMENT"
    if signals["checkout"]:
        return "CHECKOUT"
    if signals["cart"]:
        return "CART"
    if signals["fill_real"]:
        return "FORM_FILL"
    return None


def extract_expected_content(step: AgentStep) -> list[str]:
    result = (step.tool_result or "").lower()
    expected: list[str] = []
    for keyword in (
        "card number",
        "cvv",
        "expir",
        "checkout",
        "your cart",
        "subtotal",
        "payment method",
    ):
        if keyword in result:
            expected.append(keyword)
    return expected[:6]


def build_fixture(task: AgentTask, *, sanitize: bool = True) -> dict[str, Any]:
    analysis = analyze_task(task)
    milestones_reached: list[str] = []
    seen: set[str] = set()
    steps: list[dict[str, Any]] = []
    for step in task.sorted_steps:
        milestone = detect_milestone(step)
        if milestone and milestone not in seen:
            seen.add(milestone)
            milestones_reached.append(milestone)
        step_entry = {
            "step": step.step,
            "tool_name": step.tool_name,
            "tool_input": _sanitize_value(step.tool_input) if sanitize else step.tool_input,
            "page_url": (
                _sanitize_url(step.page_url or "") if sanitize and step.page_url else step.page_url
            ),
            "expected_result_contains": (
                _sanitize_value(extract_expected_content(step))
                if sanitize
                else extract_expected_content(step)
            ),
            "milestone": milestone,
        }
        steps.append(step_entry)
    payload = {
        "task_id": task.task_id,
        "user_text": _sanitize_text(task.task_text or "") if sanitize else task.task_text,
        "site": analysis.site_name,
        "category": task.task_category,
        "grade_hint": None,
        "step_sequence": steps,
        "milestones_reached": milestones_reached,
        "total_steps": len(task.steps),
    }
    return payload


def save_fixture(task: AgentTask, output_path: str | Path, *, sanitize: bool = True) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_fixture(task, sanitize=sanitize), indent=2), encoding="utf-8")
    return path


def capture_task(
    tasks: list[AgentTask], query: str, output_path: str | Path, *, sanitize: bool = True
) -> Path:
    task = resolve_task(tasks, query)
    return save_fixture(task, output_path, sanitize=sanitize)
