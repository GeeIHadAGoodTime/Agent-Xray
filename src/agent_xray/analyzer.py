from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .schema import AgentStep, AgentTask, TaskOutcome
from .signals import SignalDetector, run_detection

ERROR_PATTERNS = [
    (r"not approved|approval denied|blocked by approval|needs the user's ok first", "approval_block"),
    (r"unknown tool", "unknown_tool"),
    (r"not available in your capability tier|permission denied|forbidden", "tier_block"),
    (r"timed out|timeout", "timeout"),
    (r"rate limit|maximum \d+ calls per|too many requests|429", "rate_limit"),
    (r"no accessibility snapshot available", "snapshot_missing"),
    (r"fill .*failed: locator", "fill_fail"),
    (r"form incomplete", "fill_incomplete"),
    (r"navigation failed|execution context was destroyed", "navigation_fail"),
    (r"failed to focus window", "window_not_found"),
    (r"validation error|field required", "validation"),
    (r"not found|404", "not_found"),
    (r"click.*failed|click.*timeout|failed to click", "click_fail"),
    (r"expecting value|invalid json|json.?decode", "parse_error"),
    (r"connection refused|connection reset|ECONNREFUSED", "connection_error"),
]
GENERIC_HOST_PREFIXES = {"www", "m", "mobile", "app"}
NO_NAVIGATION_MARKERS = ("about:blank", "new-tab-page")
DATE_RE = re.compile(r"(20\d{6})")


@dataclass(slots=True)
class TaskAnalysis:
    task: AgentTask
    unique_urls: list[str]
    unique_url_paths: list[str]
    unique_tools: list[str]
    tool_sequence: list[str]
    max_repeat_tool: str
    max_repeat_count: int
    errors: int
    error_rate: float
    total_duration_ms: int
    hallucinated_tools: int
    no_tools_steps: int
    site_name: str
    final_url: str
    timeout_like: bool
    error_kinds: dict[str, int]
    total_cost_usd: float
    avg_cost_per_step: float
    signal_metrics: dict[str, dict[str, Any]]

    @property
    def is_spin(self) -> bool:
        return self.max_repeat_count >= 3

    @property
    def step_count(self) -> int:
        return len(self.task.steps)

    @property
    def spin_is_severe(self) -> bool:
        return self.max_repeat_count >= 10

    @property
    def spin_is_moderate(self) -> bool:
        return 5 <= self.max_repeat_count < 10

    @property
    def spin_is_mild(self) -> bool:
        return 3 <= self.max_repeat_count < 5

    @property
    def error_rate_is_high(self) -> bool:
        return self.error_rate > 0.5

    @property
    def error_rate_is_medium(self) -> bool:
        return 0.2 < self.error_rate <= 0.5

    @property
    def healthy_step_range(self) -> bool:
        return 8 <= self.step_count <= 25

    @property
    def too_many_steps(self) -> bool:
        return self.step_count > 25

    @property
    def tokens_in(self) -> int:
        return sum(
            (step.model.input_tokens or 0) if step.model else 0
            for step in self.task.steps
        )

    @property
    def tokens_out(self) -> int:
        return sum(
            (step.model.output_tokens or 0) if step.model else 0
            for step in self.task.steps
        )

    def metrics(self) -> dict[str, Any]:
        metrics = {
            "step_count": self.step_count,
            "unique_urls": len(self.unique_urls),
            "unique_url_paths": len(self.unique_url_paths),
            "unique_tools": len(self.unique_tools),
            "errors": self.errors,
            "error_rate": self.error_rate,
            "hallucinated_tools": self.hallucinated_tools,
            "no_tools_steps": self.no_tools_steps,
            "max_repeat_count": self.max_repeat_count,
            "max_repeat_tool": self.max_repeat_tool,
            "is_spin": self.is_spin,
            "timeout_like": self.timeout_like,
            "total_duration_ms": self.total_duration_ms,
            "site_name": self.site_name,
            "final_url": self.final_url,
            "total_cost_usd": self.total_cost_usd,
            "avg_cost_per_step": self.avg_cost_per_step,
            # Exclusive spin tiers (only one is True)
            "spin_is_severe": self.spin_is_severe,
            "spin_is_moderate": self.spin_is_moderate,
            "spin_is_mild": self.spin_is_mild,
            # Exclusive error rate tiers
            "error_rate_is_high": self.error_rate_is_high,
            "error_rate_is_medium": self.error_rate_is_medium,
            # Step range
            "healthy_step_range": self.healthy_step_range,
            "too_many_steps": self.too_many_steps,
            # Tokens
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
        }
        for detector_name, detector_metrics in self.signal_metrics.items():
            metrics[detector_name] = detector_metrics
            metrics.update(detector_metrics)
        return metrics


def normalize_site_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")


def is_ip_address(host: str) -> bool:
    return bool(re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", host))


def site_from_host(host: str) -> str | None:
    host = host.lower().split("@")[-1].split(":")[0].strip(".")
    if not host:
        return None
    if host.startswith("www."):
        host = host[4:]
    if host == "localhost":
        return "localhost"
    if is_ip_address(host):
        return host.replace(".", "-")
    parts = [part for part in host.split(".") if part]
    while len(parts) > 1 and parts[0] in GENERIC_HOST_PREFIXES:
        parts.pop(0)
    for part in parts:
        if part not in {"com", "org", "net", "io", "gov", "edu", "co", "us"}:
            return normalize_site_label(part)
    return normalize_site_label(parts[0]) if parts else None


def extract_site_from_urlish(raw: str) -> str | None:
    text = raw.strip()
    if not text:
        return None
    lower = text.lower()
    if any(marker in lower for marker in NO_NAVIGATION_MARKERS):
        return "no-navigation"
    parsed = urlparse(text if "://" in text else f"http://{text}")
    host = parsed.netloc or parsed.path.split("/")[0]
    if not host:
        return None
    return site_from_host(host)


def extract_site_name(task: AgentTask) -> str:
    candidates: list[str] = []
    if task.task_text:
        candidates.append(task.task_text)
    for step in task.sorted_steps:
        if page_url := _page_url(step):
            candidates.append(page_url)
        for key in ("url", "service_or_url", "service_name", "title", "query"):
            value = step.tool_input.get(key)
            if value:
                candidates.append(str(value))
    for candidate in candidates:
        site = extract_site_from_urlish(candidate)
        if site and site != "no-navigation":
            return site
    if any(step.tool_name.startswith(("browser_", "desktop_")) for step in task.steps):
        return "no-navigation"
    return "unknown"


def classify_error(error: str | None) -> str:
    if not error:
        return ""
    lowered = error.lower()
    for pattern, name in ERROR_PATTERNS:
        if re.search(pattern, lowered):
            return name
    return "other"


def summarize_tool_result(step: AgentStep, limit: int = 240) -> str:
    text = (step.error or step.tool_result or "").replace("\r", " ").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _unique_urls(task: AgentTask) -> list[str]:
    seen: set[str] = set()
    urls: list[str] = []
    for step in task.sorted_steps:
        url = _page_url(step)
        if url and url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def _unique_url_paths(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for url in urls:
        parsed = urlparse(url)
        key = f"{parsed.netloc}{parsed.path}"
        if key not in seen:
            seen.add(key)
            deduped.append(url)
    return deduped


def _max_consecutive_repeat(sequence: list[str]) -> tuple[str, int]:
    if not sequence:
        return ("", 0)
    best_tool = sequence[0]
    best_count = 1
    current_count = 1
    for index in range(1, len(sequence)):
        if sequence[index] == sequence[index - 1]:
            current_count += 1
            if current_count > best_count:
                best_tool = sequence[index]
                best_count = current_count
        else:
            current_count = 1
    return (best_tool, best_count)


def _page_url(step: AgentStep) -> str | None:
    if step.browser and step.browser.page_url:
        return step.browser.page_url
    return None


def _tools_available(step: AgentStep) -> list[str] | None:
    if step.tools is None or step.tools.tools_available is None:
        return None
    return list(step.tools.tools_available)


def _step_cost(step: AgentStep) -> float:
    if step.model and step.model.cost_usd is not None:
        return float(step.model.cost_usd)
    return 0.0


def _compute_core_metrics(task: AgentTask) -> dict[str, Any]:
    urls = _unique_urls(task)
    url_paths = _unique_url_paths(urls)
    tool_sequence = [step.tool_name for step in task.sorted_steps]
    unique_tools = sorted(set(tool_sequence))
    repeat_tool, repeat_count = _max_consecutive_repeat(tool_sequence)
    errors = sum(1 for step in task.steps if step.error)
    error_kinds: Counter[str] = Counter()
    no_tools_steps = 0
    hallucinated_tools = 0
    for step in task.sorted_steps:
        if _tools_available(step) == []:
            no_tools_steps += 1
        error_kind = classify_error(step.error)
        if error_kind:
            error_kinds[error_kind] += 1
        if error_kind == "unknown_tool":
            hallucinated_tools += 1
    total_cost = sum(_step_cost(step) for step in task.steps)
    return {
        "unique_urls": urls,
        "unique_url_paths": url_paths,
        "unique_tools": unique_tools,
        "tool_sequence": tool_sequence,
        "max_repeat_tool": repeat_tool,
        "max_repeat_count": repeat_count,
        "errors": errors,
        "error_rate": errors / max(1, len(task.steps)),
        "total_duration_ms": sum(step.duration_ms or 0 for step in task.steps),
        "hallucinated_tools": hallucinated_tools,
        "no_tools_steps": no_tools_steps,
        "site_name": extract_site_name(task),
        "final_url": urls[-1] if urls else "",
        "timeout_like": (
            task.outcome is not None and task.outcome.status in {"timeout", "max_iterations"}
        )
        or len(task.steps) >= 75,
        "error_kinds": dict(error_kinds),
        "total_cost_usd": float(total_cost),
        "avg_cost_per_step": (float(total_cost) / len(task.steps)) if task.steps else 0.0,
    }


def analyze_task(
    task: AgentTask,
    detectors: list[SignalDetector] | None = None,
) -> TaskAnalysis:
    core = _compute_core_metrics(task)
    signal_metrics = run_detection(task, detectors)
    return TaskAnalysis(task=task, signal_metrics=signal_metrics, **core)


def analyze_tasks(
    tasks: list[AgentTask],
    detectors: list[SignalDetector] | None = None,
) -> dict[str, TaskAnalysis]:
    return {task.task_id: analyze_task(task, detectors=detectors) for task in tasks}


def _extract_day(path: Path, payload: dict[str, Any]) -> str | None:
    if match := DATE_RE.search(path.stem):
        return match.group(1)
    timestamp = payload.get("timestamp") or payload.get("ts")
    if isinstance(timestamp, str) and len(timestamp) >= 10:
        return timestamp[:10].replace("-", "")
    return None


def _iter_jsonl_files(log_dir: Path, days: int | None = None) -> list[Path]:
    if log_dir.is_file():
        return [log_dir]
    files = sorted(log_dir.glob("*.jsonl"))
    if days and days > 0:
        return files[-days:]
    return files


def tasks_from_steps(steps: list[AgentStep], *, source_path: Path | None = None) -> list[AgentTask]:
    tasks: dict[str, AgentTask] = {}
    for step in steps:
        task_id = step.task_id or (source_path.stem if source_path else "unknown-task")
        step.task_id = task_id
        task = tasks.setdefault(task_id, AgentTask(task_id=task_id))
        if task.day is None and source_path is not None:
            task.day = _extract_day(source_path, {"timestamp": step.timestamp})
        task.steps.append(step)
    return [tasks[key] for key in sorted(tasks)]


def load_adapted_tasks(
    log_dir: str | Path,
    *,
    format: str = "auto",
    days: int | None = None,
) -> list[AgentTask]:
    from .adapters import adapt

    root = Path(log_dir)
    if not root.exists():
        raise FileNotFoundError(f"log path does not exist: {root}")
    tasks: dict[str, AgentTask] = {}
    for path in _iter_jsonl_files(root, days=days):
        for adapted_task in tasks_from_steps(adapt(path, format=format), source_path=path):
            task = tasks.setdefault(adapted_task.task_id, AgentTask(task_id=adapted_task.task_id))
            if task.day is None:
                task.day = adapted_task.day
            task.steps.extend(adapted_task.steps)
    return [tasks[key] for key in sorted(tasks)]


def load_tasks(log_dir: str | Path, days: int | None = None) -> list[AgentTask]:
    root = Path(log_dir)
    if not root.exists():
        raise FileNotFoundError(f"log path does not exist: {root}")
    tasks: dict[str, AgentTask] = {}
    for path in _iter_jsonl_files(root, days=days):
        with path.open(encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                try:
                    raw_payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(raw_payload, dict):
                    continue
                payload = {str(key): value for key, value in raw_payload.items()}
                event = payload.get("event")
                task_id = str(payload.get("task_id") or "")
                if not task_id:
                    continue
                task = tasks.setdefault(task_id, AgentTask(task_id=task_id))
                if task.day is None:
                    task.day = _extract_day(path, payload)
                if payload.get("user_text") and not task.task_text:
                    task.task_text = str(payload["user_text"])
                if payload.get("task_category") and not task.task_category:
                    task.task_category = str(payload["task_category"])
                if event == "task_complete" or (
                    payload.get("outcome") is not None and payload.get("tool_name") in (None, "")
                ):
                    task.outcome = TaskOutcome.from_dict(payload)
                    if payload.get("user_text") and not task.task_text:
                        task.task_text = str(payload["user_text"])
                    if payload.get("task_category") and not task.task_category:
                        task.task_category = str(payload["task_category"])
                    task.metadata.update(task.outcome.metadata)
                    continue
                if payload.get("tool_name"):
                    step = AgentStep.from_dict(payload)
                    if not step.timestamp:
                        step.timestamp = None if payload.get("ts") is None else str(payload["ts"])
                    task.steps.append(step)
                    for key in (
                        "focused_set",
                        "prompt_variant",
                        "prompt_variant_full",
                        "system_prompt_text",
                        "system_context_components",
                        "prior_conversation_turns",
                        "prior_conversation_summary",
                    ):
                        value = payload.get(key)
                        if value is not None and key not in task.metadata:
                            task.metadata[key] = value
    return [tasks[key] for key in sorted(tasks)]


def resolve_task(tasks: list[AgentTask], query: str) -> AgentTask:
    by_id = {task.task_id: task for task in tasks}
    if query in by_id:
        return by_id[query]
    matches = [task for task in tasks if task.task_id.startswith(query)]
    if len(matches) == 1:
        return matches[0]
    matches = [task for task in tasks if query in task.task_id]
    if len(matches) == 1:
        return matches[0]
    available = sorted(t.task_id for t in tasks)
    hint = ", ".join(available[:10])
    if len(available) > 10:
        hint += f" ... ({len(available)} total)"
    raise KeyError(f"task id '{query}' not found. Available: {hint}")


def build_task_tree(tasks: list[AgentTask]) -> dict[str, dict[str, list[str]]]:
    tree: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for task in tasks:
        analysis = analyze_task(task)
        day = task.day or "unknown-day"
        tree[day][analysis.site_name].append(task.task_id)
    return {day: dict(sites) for day, sites in sorted(tree.items())}
