from __future__ import annotations

import json
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
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
        soft_errors: Number of logical failures inferred from ``tool_result``.
        soft_error_kinds: Soft-error counts bucketed by classifier label.
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
    soft_errors: int = 0
    soft_error_kinds: dict[str, int] = field(default_factory=dict)
    # New metrics — defaults preserve backward compat with manual construction
    task_failed: bool = False
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
    final_answer_indicates_failure: bool = False
    # DOM element ref mismatch
    element_ref_mismatches: int = 0

    @property
    def task_id(self) -> str:
        return self.task.task_id

    def __post_init__(self) -> None:
        if self.soft_error_kinds is None:
            self.soft_error_kinds = {}

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
            "task_failed": self.task_failed,
            "total_duration_ms": self.total_duration_ms,
            "site_name": self.site_name,
            "final_url": self.final_url,
            "soft_errors": self.soft_errors,
            "soft_error_kinds": dict(self.soft_error_kinds),
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
            "final_answer_indicates_failure": self.final_answer_indicates_failure,
            # DOM element ref mismatch
            "element_ref_mismatches": self.element_ref_mismatches,
        }
        for detector_name, detector_metrics in self.signal_metrics.items():
            metrics[detector_name] = detector_metrics
            metrics.update(detector_metrics)
        return metrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task": self.task.to_dict(),
            "unique_urls": list(self.unique_urls),
            "unique_url_paths": list(self.unique_url_paths),
            "unique_tools": list(self.unique_tools),
            "tool_sequence": list(self.tool_sequence),
            "max_repeat_tool": self.max_repeat_tool,
            "max_repeat_count": self.max_repeat_count,
            "errors": self.errors,
            "error_rate": self.error_rate,
            "total_duration_ms": self.total_duration_ms,
            "hallucinated_tools": self.hallucinated_tools,
            "no_tools_steps": self.no_tools_steps,
            "site_name": self.site_name,
            "final_url": self.final_url,
            "timeout_like": self.timeout_like,
            "task_completed": self.task_completed,
            "error_kinds": dict(self.error_kinds),
            "soft_errors": self.soft_errors,
            "soft_error_kinds": dict(self.soft_error_kinds),
            "total_cost_usd": self.total_cost_usd,
            "avg_cost_per_step": self.avg_cost_per_step,
            "signal_metrics": {
                name: dict(metrics) for name, metrics in self.signal_metrics.items()
            },
            "task_failed": self.task_failed,
            "rejected_tool_count": self.rejected_tool_count,
            "timed_out_flag": self.timed_out_flag,
            "suspicious_short_flag": self.suspicious_short_flag,
            "final_answer_length": self.final_answer_length,
            "has_final_answer": self.has_final_answer,
            "max_context_usage_pct": self.max_context_usage_pct,
            "cache_read_tokens_total": self.cache_read_tokens_total,
            "cache_creation_tokens_total": self.cache_creation_tokens_total,
            "max_step_gap_ms": self.max_step_gap_ms,
            "avg_step_duration_ms": self.avg_step_duration_ms,
            "step_duration_trend": self.step_duration_trend,
            "has_frustration_context": self.has_frustration_context,
            "has_delivery_address": self.has_delivery_address,
            "has_user_model": self.has_user_model,
            "system_context_field_count": self.system_context_field_count,
            "final_answer_empty_but_success": self.final_answer_empty_but_success,
            "final_answer_indicates_failure": self.final_answer_indicates_failure,
            "element_ref_mismatches": self.element_ref_mismatches,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TaskAnalysis:
        def _int(value: Any, default: int = 0) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        def _float(value: Any, default: float = 0.0) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def _bool(value: Any, default: bool = False) -> bool:
            if isinstance(value, bool):
                return value
            if value is None:
                return default
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "1", "yes", "y"}:
                    return True
                if lowered in {"false", "0", "no", "n"}:
                    return False
            return bool(value)

        def _counts(value: Any) -> dict[str, int]:
            if not isinstance(value, dict):
                return {}
            return {str(key): _int(item) for key, item in value.items()}

        task_payload = payload.get("task")
        if isinstance(task_payload, AgentTask):
            task = task_payload
        elif isinstance(task_payload, dict):
            task = AgentTask.from_dict(task_payload)
        else:
            task = AgentTask(task_id=str(payload.get("task_id", "")), steps=[])

        signal_metrics_raw = payload.get("signal_metrics")
        signal_metrics: dict[str, dict[str, Any]] = {}
        if isinstance(signal_metrics_raw, dict):
            for name, metrics in signal_metrics_raw.items():
                signal_metrics[str(name)] = dict(metrics) if isinstance(metrics, dict) else {}

        return cls(
            task=task,
            unique_urls=[str(item) for item in payload.get("unique_urls", [])],
            unique_url_paths=[str(item) for item in payload.get("unique_url_paths", [])],
            unique_tools=[str(item) for item in payload.get("unique_tools", [])],
            tool_sequence=[str(item) for item in payload.get("tool_sequence", [])],
            max_repeat_tool=str(payload.get("max_repeat_tool", "")),
            max_repeat_count=_int(payload.get("max_repeat_count")),
            errors=_int(payload.get("errors")),
            error_rate=_float(payload.get("error_rate")),
            total_duration_ms=_int(payload.get("total_duration_ms")),
            hallucinated_tools=_int(payload.get("hallucinated_tools")),
            no_tools_steps=_int(payload.get("no_tools_steps")),
            site_name=str(payload.get("site_name", "")),
            final_url=str(payload.get("final_url", "")),
            timeout_like=_bool(payload.get("timeout_like")),
            task_completed=_bool(payload.get("task_completed")),
            error_kinds=_counts(payload.get("error_kinds")),
            soft_errors=_int(payload.get("soft_errors")),
            soft_error_kinds=_counts(payload.get("soft_error_kinds")),
            total_cost_usd=_float(payload.get("total_cost_usd")),
            avg_cost_per_step=_float(payload.get("avg_cost_per_step")),
            signal_metrics=signal_metrics,
            task_failed=_bool(payload.get("task_failed")),
            rejected_tool_count=_int(payload.get("rejected_tool_count")),
            timed_out_flag=_bool(payload.get("timed_out_flag")),
            suspicious_short_flag=_bool(payload.get("suspicious_short_flag")),
            final_answer_length=_int(payload.get("final_answer_length")),
            has_final_answer=_bool(payload.get("has_final_answer")),
            max_context_usage_pct=_float(payload.get("max_context_usage_pct")),
            cache_read_tokens_total=_int(payload.get("cache_read_tokens_total")),
            cache_creation_tokens_total=_int(payload.get("cache_creation_tokens_total")),
            max_step_gap_ms=_int(payload.get("max_step_gap_ms")),
            avg_step_duration_ms=_float(payload.get("avg_step_duration_ms")),
            step_duration_trend=str(payload.get("step_duration_trend", "stable")),
            has_frustration_context=_bool(payload.get("has_frustration_context")),
            has_delivery_address=_bool(payload.get("has_delivery_address")),
            has_user_model=_bool(payload.get("has_user_model")),
            system_context_field_count=_int(payload.get("system_context_field_count")),
            final_answer_empty_but_success=_bool(payload.get("final_answer_empty_but_success")),
            final_answer_indicates_failure=_bool(payload.get("final_answer_indicates_failure")),
            element_ref_mismatches=_int(payload.get("element_ref_mismatches")),
        )


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


# Patterns indicating a logical failure inside tool_result even when error is None.
# Each tuple is (compiled regex, soft-error category name).
_SOFT_ERROR_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"not on a payment page|no payment form", re.IGNORECASE), "soft_not_on_page"),
    (re.compile(r"element not found|selector not found|no (?:such )?element", re.IGNORECASE), "soft_element_missing"),
    (re.compile(r"timed?\s*out|deadline exceeded", re.IGNORECASE), "soft_timeout"),
    (re.compile(r"access denied|unauthorized|forbidden|403", re.IGNORECASE), "soft_access_denied"),
    (re.compile(r"rate limit|too many requests|429", re.IGNORECASE), "soft_rate_limit"),
    (re.compile(r"could not|unable to|failed to|cannot", re.IGNORECASE), "soft_failure"),
]

_FINAL_ANSWER_FAILURE_PATTERN = re.compile(
    r"\b(?:"
    r"error|errored|failure|failed|failing|"
    r"cannot|can't|could not|couldn't|unable to|"
    r"stuck|broke|broken|crash|crashed"
    r")\b",
    re.IGNORECASE,
)


def classify_soft_error(tool_result: str | None) -> str:
    """Classify logical failures in tool_result content (no error field set).

    Returns a soft-error category or empty string.
    """
    if not tool_result:
        return ""
    for pattern, name in _SOFT_ERROR_PATTERNS:
        if pattern.search(tool_result):
            return name
    return ""


def final_answer_indicates_failure(final_answer: str | None) -> bool:
    if not final_answer:
        return False
    normalized = " ".join(final_answer.split())
    return bool(_FINAL_ANSWER_FAILURE_PATTERN.search(normalized))


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


def _step_cost(step: AgentStep, pricing_data: dict[str, Any] | None = None) -> float:
    """Compute cost for a single step.

    Uses the trace-provided ``cost_usd`` when available.  Falls back to
    token-based estimation via the pricing database.
    """
    if step.model and step.model.cost_usd is not None:
        return float(step.model.cost_usd)
    # Estimate from token counts + pricing database
    if step.model and step.model.model_name and (
        step.model.input_tokens or step.model.output_tokens
    ):
        from .pricing import get_model_cost

        return get_model_cost(
            model_name=step.model.model_name,
            input_tokens=step.model.input_tokens or 0,
            output_tokens=step.model.output_tokens or 0,
            cached_tokens=step.model.cache_read_tokens or 0,
            pricing_data=pricing_data,
        )
    return 0.0


def _compute_core_metrics(task: AgentTask, pricing_data: dict[str, Any] | None = None) -> dict[str, Any]:
    urls = _unique_urls(task)
    url_paths = _unique_url_paths(urls)
    tool_sequence = [step.tool_name for step in task.sorted_steps]
    unique_tools = sorted(set(tool_sequence))
    repeat_tool, repeat_count = _max_consecutive_repeat(tool_sequence)
    errors = sum(1 for step in task.steps if step.error)
    error_kinds: Counter[str] = Counter()
    soft_error_kinds: Counter[str] = Counter()
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
        # Detect logical failures in tool_result content (no error field)
        if not step.error:
            soft_kind = classify_soft_error(step.tool_result)
            if soft_kind:
                soft_error_kinds[soft_kind] += 1
    total_cost = sum(_step_cost(step, pricing_data) for step in task.steps)
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
    # Production data stores context_usage_pct already in percentage units
    # (0.5 means 0.5%, not 50%). Do NOT normalize.
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
                # Try numeric epoch first, then ISO-8601
                ts_float = float(s.timestamp)
                timestamps_ms.append(int(ts_float * 1000))
            except (TypeError, ValueError):
                try:
                    from datetime import datetime, timezone
                    dt = datetime.fromisoformat(str(s.timestamp))
                    timestamps_ms.append(int(dt.timestamp() * 1000))
                except (TypeError, ValueError):
                    pass
    for i in range(1, len(timestamps_ms)):
        gap = timestamps_ms[i] - timestamps_ms[i - 1]
        if gap > max_step_gap_ms:
            max_step_gap_ms = gap
    # Step duration trend (filter out 0ms artifact steps like think/noop)
    step_duration_trend = "stable"
    real_durations = [d for d in step_durations if d > 0]
    if len(real_durations) >= 3:
        third = len(real_durations) // 3
        first_third_avg = sum(real_durations[:third]) / max(third, 1)
        last_third_avg = sum(real_durations[-third:]) / max(third, 1)
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
    final_answer_failure_flag = final_answer_indicates_failure(final_answer)
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
        # Search backward for the most recent snapshot (not just immediately prior)
        snapshot_result = None
        for back_idx in range(idx - 1, -1, -1):
            if sorted_steps[back_idx].tool_name.startswith("browser_snapshot"):
                snapshot_result = sorted_steps[back_idx].tool_result or ""
                break
        if snapshot_result is None:
            continue
        if ref_str not in snapshot_result:
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
        "total_duration_ms": (
            int(task.outcome.total_duration_s * 1000)
            if task.outcome and task.outcome.total_duration_s
            else sum(step.duration_ms or 0 for step in task.steps)
        ),
        "hallucinated_tools": hallucinated_tools,
        "no_tools_steps": no_tools_steps,
        "site_name": extract_site_name(task),
        "final_url": urls[-1] if urls else "",
        "timeout_like": timed_out_flag or (
            task.outcome is not None
            and task.outcome.status
            in {"timeout", "max_iterations", "spin_terminated"}
        )
        or len(task.steps) >= 75,
        "task_completed": (
            task.outcome is not None
            and task.outcome.status
            in {"success", "completed", "payment_gate"}
        ),
        "task_failed": (
            task.outcome is not None
            and task.outcome.status
            in {"failed", "llm_error", "early_abort"}
        ),
        "error_kinds": dict(error_kinds),
        "soft_errors": sum(soft_error_kinds.values()),
        "soft_error_kinds": dict(soft_error_kinds),
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
        "final_answer_indicates_failure": final_answer_failure_flag,
        "element_ref_mismatches": element_ref_mismatches,
    }


def analyze_task(
    task: AgentTask,
    detectors: list[SignalDetector] | None = None,
    pricing_data: dict[str, Any] | None = None,
) -> TaskAnalysis:
    """Analyze one task and compute core plus detector-derived metrics.

    Args:
        task: Task to analyze.
        detectors: Optional detector instances. When omitted, built-in and
            installed detectors are discovered automatically.
        pricing_data: Pre-loaded pricing dict for cost estimation.
            When omitted, the bundled pricing database is loaded automatically.

    Returns:
        TaskAnalysis: Computed metrics and summaries for the task.

    Example:
        >>> analysis = analyze_task(task)
        >>> analysis.metrics()["step_count"]
        4
    """
    core = _compute_core_metrics(task, pricing_data)
    signal_metrics = run_detection(task, detectors)
    return TaskAnalysis(task=task, signal_metrics=signal_metrics, **core)


def analyze_tasks(
    tasks: list[AgentTask],
    detectors: list[SignalDetector] | None = None,
    pricing_data: dict[str, Any] | None = None,
) -> dict[str, TaskAnalysis]:
    """Analyze multiple tasks and index results by task id.

    Args:
        tasks: Tasks to analyze.
        detectors: Optional shared detector instances.
        pricing_data: Pre-loaded pricing dict for cost estimation.

    Returns:
        dict[str, TaskAnalysis]: Per-task analyses keyed by ``task_id``.
    """
    return {
        task.task_id: analyze_task(task, detectors=detectors, pricing_data=pricing_data)
        for task in tasks
    }


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
    files = list(_iter_jsonl_files(root, days=days))
    for path in files:
        for adapted_task in tasks_from_steps(adapt(path, format=format), source_path=path):
            task = tasks.setdefault(adapted_task.task_id, AgentTask(task_id=adapted_task.task_id))
            if task.day is None:
                task.day = adapted_task.day
            task.steps.extend(adapted_task.steps)

    # Recover task outcomes from raw JSONL.  Adapters only emit AgentStep
    # records (lines with tool_name), so task_complete / outcome-only lines
    # are silently dropped.  This second pass restores them.
    for path in files:
        try:
            with path.open(encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(payload, dict):
                        continue
                    task_id = str(payload.get("task_id") or "")
                    if not task_id:
                        continue
                    is_outcome = (
                        payload.get("event") == "task_complete"
                        or (
                            payload.get("outcome") is not None
                            and payload.get("tool_name") in (None, "")
                        )
                    )
                    if not is_outcome:
                        continue
                    task = tasks.get(task_id)
                    if task is None:
                        # Outcome-only task (no steps were parsed by adapter)
                        task = AgentTask(task_id=task_id)
                        task.day = _extract_day(path, payload)
                        tasks[task_id] = task
                    if task.outcome is None:
                        task.outcome = TaskOutcome.from_dict(payload)
                        task.metadata.update(task.outcome.metadata)
                    if payload.get("user_text") and not task.task_text:
                        task.task_text = str(payload["user_text"])
                    if payload.get("task_category") and not task.task_category:
                        task.task_category = str(payload["task_category"])
        except OSError:
            continue

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
