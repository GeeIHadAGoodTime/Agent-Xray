from __future__ import annotations

import json
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .schema import AgentStep, AgentTask, TaskOutcome
from .signals import SignalDetector, run_detection

ERROR_PATTERNS = [
    (
        r"not approved|approval denied|blocked by approval|needs the user's ok first",
        "approval_block",
    ),
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
    """Computed metrics and derived signals for a single task.

    Attributes:
        task: Source task used to compute the analysis.
        unique_urls: Ordered unique URLs visited by the task.
        unique_url_paths: Deduplicated URLs collapsed by host and path.
        unique_tools: Sorted unique tool names used by the task.
        tool_sequence: Ordered tool call sequence.
        max_repeat_tool: Tool with the longest consecutive repetition streak.
        max_repeat_count: Length of the longest repetition streak.
        errors: Number of steps with a recorded error.
        error_rate: Fraction of steps that recorded an error.
        total_duration_ms: Sum of step durations in milliseconds.
        hallucinated_tools: Count of unknown-tool errors.
        no_tools_steps: Count of steps that exposed zero tools.
        site_name: Best-effort site or target label inferred from the task.
        final_url: Last visited URL, if any.
        timeout_like: Whether the run ended like a timeout or max-iteration stop.
        task_completed: Whether the task outcome indicates successful completion.
        error_kinds: Error counts bucketed by classifier label.
        total_cost_usd: Sum of per-step costs in US dollars.
        avg_cost_per_step: Average step cost in US dollars.
        signal_metrics: Per-detector metrics returned by signal detectors.
    """

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
    task_completed: bool
    error_kinds: dict[str, int]
    total_cost_usd: float
    avg_cost_per_step: float
    signal_metrics: dict[str, dict[str, Any]]
    # New metrics — defaults preserve backward compat with manual construction
    rejected_tool_count: int = 0
    timed_out_flag: bool = False
    suspicious_short_flag: bool = False
    final_answer_length: int = 0
    has_final_answer: bool = False
    max_context_usage_pct: float = 0.0
    cache_read_tokens_total: int = 0
    cache_creation_tokens_total: int = 0
    # Temporal patterns
    max_step_gap_ms: int = 0
    avg_step_duration_ms: float = 0.0
    step_duration_trend: str = "stable"
    # System context analysis
    has_frustration_context: bool = False
    has_delivery_address: bool = False
    has_user_model: bool = False
    system_context_field_count: int = 0
    # Final answer quality
    final_answer_empty_but_success: bool = False
    # DOM element ref mismatch
    element_ref_mismatches: int = 0

    @property
    def task_id(self) -> str:
        return self.task.task_id

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
        return sum((step.model.input_tokens or 0) if step.model else 0 for step in self.task.steps)

    @property
    def tokens_out(self) -> int:
        return sum((step.model.output_tokens or 0) if step.model else 0 for step in self.task.steps)

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
            "task_completed": self.task_completed,
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
            # Data from production flags
            "rejected_tool_count": self.rejected_tool_count,
            "timed_out_flag": self.timed_out_flag,
            "suspicious_short_flag": self.suspicious_short_flag,
            "final_answer_length": self.final_answer_length,
            "has_final_answer": self.has_final_answer,
            "max_context_usage_pct": self.max_context_usage_pct,
            "cache_read_tokens_total": self.cache_read_tokens_total,
            "cache_creation_tokens_total": self.cache_creation_tokens_total,
            # Temporal patterns
            "max_step_gap_ms": self.max_step_gap_ms,
            "avg_step_duration_ms": self.avg_step_duration_ms,
            "step_duration_trend": self.step_duration_trend,
            # System context analysis
            "has_frustration_context": self.has_frustration_context,
            "has_delivery_address": self.has_delivery_address,
            "has_user_model": self.has_user_model,
            "system_context_field_count": self.system_context_field_count,
            # Final answer quality
            "final_answer_empty_but_success": self.final_answer_empty_but_success,
            # DOM element ref mismatch
            "element_ref_mismatches": self.element_ref_mismatches,
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
    if "://" not in text:
        looks_like_host = (
            "." in text
            or "/" in text
            or text.lower() == "localhost"
            or is_ip_address(text.split("/", 1)[0].split(":", 1)[0])
        )
        if not looks_like_host:
            return None
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
    # Consume rejected_tools across all steps
    rejected_tool_count = sum(
        len(step.tools.rejected_tools or [])
        for step in task.steps
        if step.tools
    )
    # Consume production outcome flags
    outcome_meta = task.outcome.metadata if task.outcome else {}
    timed_out_flag = bool(outcome_meta.get("timed_out", False))
    suspicious_short_flag = bool(outcome_meta.get("suspicious_short", False))
    # Final answer analysis
    final_answer = (task.outcome.final_answer or "") if task.outcome else ""
    final_answer_length = len(final_answer.strip())
    has_final_answer = final_answer_length > 0
    # Context usage peak
    context_usages = [
        step.model.context_usage_pct
        for step in task.steps
        if step.model and step.model.context_usage_pct is not None
    ]
    # Also check final_context_usage_pct from outcome
    final_ctx = outcome_meta.get("final_context_usage_pct")
    if final_ctx is not None:
        try:
            context_usages.append(float(final_ctx))
        except (TypeError, ValueError):
            pass
    max_context_usage_pct = max(context_usages) if context_usages else 0.0
    # Normalize to percentage if in 0-1 range
    if 0.0 < max_context_usage_pct <= 1.0:
        max_context_usage_pct *= 100.0
    # Cache token totals
    cache_read_tokens_total = sum(
        step.model.cache_read_tokens or 0
        for step in task.steps
        if step.model
    )
    cache_creation_tokens_total = sum(
        step.model.cache_creation_tokens or 0
        for step in task.steps
        if step.model
    )
    # --- Temporal patterns ---
    sorted_steps = task.sorted_steps
    step_durations = [s.duration_ms for s in sorted_steps if s.duration_ms is not None]
    avg_step_duration_ms = (
        (sum(step_durations) / len(step_durations)) if step_durations else 0.0
    )
    # Max gap between consecutive step timestamps
    max_step_gap_ms = 0
    timestamps_ms: list[int] = []
    for s in sorted_steps:
        if s.timestamp is not None:
            try:
                ts_float = float(s.timestamp)
                timestamps_ms.append(int(ts_float * 1000))
            except (TypeError, ValueError):
                pass
    for i in range(1, len(timestamps_ms)):
        gap = timestamps_ms[i] - timestamps_ms[i - 1]
        if gap > max_step_gap_ms:
            max_step_gap_ms = gap
    # Step duration trend
    step_duration_trend = "stable"
    if len(step_durations) >= 3:
        third = len(step_durations) // 3
        first_third_avg = sum(step_durations[:third]) / max(third, 1)
        last_third_avg = sum(step_durations[-third:]) / max(third, 1)
        if first_third_avg > 0 and last_third_avg > 0:
            if last_third_avg < first_third_avg / 2:
                step_duration_trend = "accelerating"
            elif last_third_avg > first_third_avg * 2:
                step_duration_trend = "decelerating"
    # --- System context analysis ---
    sys_ctx = task.metadata.get("system_context_components")
    if not isinstance(sys_ctx, dict):
        sys_ctx = {}
    has_frustration_context = bool(sys_ctx.get("frustration"))
    has_delivery_address = "delivery_address" in sys_ctx
    has_user_model = "user_model" in sys_ctx
    system_context_field_count = sum(1 for v in sys_ctx.values() if v)
    # --- Final answer quality ---
    task_completed_flag = (
        task.outcome is not None and task.outcome.status in {"success", "completed"}
    )
    final_answer_empty_but_success = (
        task_completed_flag and final_answer_length <= 10
    )
    # --- DOM element ref mismatch ---
    element_ref_mismatches = 0
    ref_pattern = re.compile(r"@e\d+")
    for idx in range(1, len(sorted_steps)):
        step = sorted_steps[idx]
        if not step.tool_name.startswith(("browser_click", "browser_fill")):
            continue
        ref_value = step.tool_input.get("ref", "")
        if isinstance(ref_value, str):
            ref_match = ref_pattern.search(ref_value)
        else:
            ref_match = ref_pattern.search(str(ref_value))
        if not ref_match:
            continue
        ref_str = ref_match.group(0)
        prev_step = sorted_steps[idx - 1]
        if not prev_step.tool_name.startswith("browser_snapshot"):
            continue
        prev_result = prev_step.tool_result or ""
        if ref_str not in prev_result:
            element_ref_mismatches += 1
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
        "timeout_like": timed_out_flag or (
            task.outcome is not None
            and task.outcome.status
            in {"timeout", "max_iterations", "spin_terminated", "early_abort", "failed"}
        )
        or len(task.steps) >= 75,
        "task_completed": (
            task.outcome is not None and task.outcome.status in {"success", "completed"}
        ),
        "error_kinds": dict(error_kinds),
        "total_cost_usd": float(total_cost),
        "avg_cost_per_step": (float(total_cost) / len(task.steps)) if task.steps else 0.0,
        "rejected_tool_count": rejected_tool_count,
        "timed_out_flag": timed_out_flag,
        "suspicious_short_flag": suspicious_short_flag,
        "final_answer_length": final_answer_length,
        "has_final_answer": has_final_answer,
        "max_context_usage_pct": max_context_usage_pct,
        "cache_read_tokens_total": cache_read_tokens_total,
        "cache_creation_tokens_total": cache_creation_tokens_total,
        "max_step_gap_ms": max_step_gap_ms,
        "avg_step_duration_ms": avg_step_duration_ms,
        "step_duration_trend": step_duration_trend,
        "has_frustration_context": has_frustration_context,
        "has_delivery_address": has_delivery_address,
        "has_user_model": has_user_model,
        "system_context_field_count": system_context_field_count,
        "final_answer_empty_but_success": final_answer_empty_but_success,
        "element_ref_mismatches": element_ref_mismatches,
    }


def analyze_task(
    task: AgentTask,
    detectors: list[SignalDetector] | None = None,
) -> TaskAnalysis:
    """Analyze one task and compute core plus detector-derived metrics.

    Args:
        task: Task to analyze.
        detectors: Optional detector instances. When omitted, built-in and
            installed detectors are discovered automatically.

    Returns:
        TaskAnalysis: Computed metrics and summaries for the task.

    Example:
        >>> analysis = analyze_task(task)
        >>> analysis.metrics()["step_count"]
        4
    """
    core = _compute_core_metrics(task)
    signal_metrics = run_detection(task, detectors)
    return TaskAnalysis(task=task, signal_metrics=signal_metrics, **core)


def analyze_tasks(
    tasks: list[AgentTask],
    detectors: list[SignalDetector] | None = None,
) -> dict[str, TaskAnalysis]:
    """Analyze multiple tasks and index results by task id.

    Args:
        tasks: Tasks to analyze.
        detectors: Optional shared detector instances.

    Returns:
        dict[str, TaskAnalysis]: Per-task analyses keyed by ``task_id``.
    """
    return {task.task_id: analyze_task(task, detectors=detectors) for task in tasks}


def _extract_day(path: Path, payload: dict[str, Any]) -> str | None:
    if match := DATE_RE.search(path.stem):
        return match.group(1)
    timestamp = payload.get("timestamp") or payload.get("ts")
    if isinstance(timestamp, str) and len(timestamp) >= 10:
        return timestamp[:10].replace("-", "")
    return None


def _sniff_agent_trace(path: Path) -> bool:
    """Check if a JSONL file contains agent traces by inspecting its first few lines."""
    try:
        with path.open(encoding="utf-8", errors="ignore") as f:
            for _, line in zip(range(5), f):
                try:
                    row = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if not isinstance(row, dict):
                    continue
                # Agent step files have task_id + (tool_name or event=agent_step/task_complete)
                if row.get("task_id") and (
                    row.get("tool_name")
                    or row.get("event") in {"agent_step", "task_complete"}
                ):
                    return True
    except OSError:
        pass
    return False


def _iter_jsonl_files(
    log_dir: Path, days: int | None = None, pattern: str | None = None
) -> list[Path]:
    if log_dir.is_file():
        return [log_dir]
    glob_pat = pattern or "*.jsonl"
    files = sorted(log_dir.glob(glob_pat))
    if not pattern:
        # Auto-filter: only keep files that actually contain agent traces
        files = [f for f in files if _sniff_agent_trace(f)]
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
    """Load tasks through a framework adapter.

    Args:
        log_dir: Directory or single ``.jsonl`` trace file to load.
        format: Explicit adapter format or ``"auto"`` for auto-detection.
        days: Optional number of most recent ``.jsonl`` files to include.

    Returns:
        list[AgentTask]: Normalized tasks reconstructed from adapted steps.
    """
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


def load_tasks(
    log_dir: str | Path, days: int | None = None, pattern: str | None = None
) -> list[AgentTask]:
    """Load native agent-xray JSONL traces into normalized tasks.

    Args:
        log_dir: Directory or single ``.jsonl`` trace file to load.
        days: Optional number of most recent ``.jsonl`` files to include.
        pattern: Glob pattern to filter files (e.g. ``"agent-steps-*.jsonl"``).
            When omitted, auto-detects files containing agent traces.

    Returns:
        list[AgentTask]: Parsed tasks ordered by ``task_id``.

    Example:
        >>> tasks = load_tasks("./traces")
        >>> isinstance(tasks, list)
        True
    """
    root = Path(log_dir)
    if not root.exists():
        raise FileNotFoundError(f"log path does not exist: {root}")
    tasks: dict[str, AgentTask] = {}
    skipped = 0
    total = 0
    files = _iter_jsonl_files(root, days=days, pattern=pattern)
    for path in files:
        with path.open(encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                total += 1
                try:
                    raw_payload = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                if not isinstance(raw_payload, dict):
                    skipped += 1
                    continue
                payload = {str(key): value for key, value in raw_payload.items()}
                event = payload.get("event")
                task_id = str(payload.get("task_id") or "")
                if not task_id:
                    skipped += 1
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
    if skipped > 0:
        pct = skipped / total * 100 if total else 0
        if skipped == total:
            msg = (
                f"load_tasks: all {total} lines lacked task_id — "
                f"file may not contain agent traces. "
                f"Try pointing at a specific agent step file or use pattern='agent-steps-*.jsonl'"
            )
        else:
            msg = (
                f"load_tasks: skipped {skipped}/{total} lines ({pct:.0f}%) "
                f"without task_id (not agent step records)"
            )
        warnings.warn(
            msg,
            stacklevel=2,
        )
    return [tasks[key] for key in sorted(tasks)]


def resolve_task(tasks: list[AgentTask], query: str) -> AgentTask:
    """Resolve a task by exact id, prefix, or unique substring.

    Args:
        tasks: Candidate tasks to search.
        query: Exact id, unique prefix, or unique substring match.

    Returns:
        AgentTask: The uniquely matched task.

    Raises:
        KeyError: If the query does not resolve to exactly one task.
    """
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
