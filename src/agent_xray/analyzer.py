from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .schema import AgentStep, AgentTask, TaskOutcome

TERMINAL_URL_PATTERNS = [
    r"/payment",
    r"/checkout",
    r"/billing",
    r"/order/confirm",
    r"/order/review",
    r"/place.order",
    r"/confirmation",
]
PAYMENT_FIELD_KEYWORDS = [
    "card number",
    "credit card",
    "debit card",
    "cvv",
    "cvc",
    "security code",
    "expir",
    "billing address",
    "cardholder",
    "payment method",
]
PAYMENT_FIELDS_STRONG = [
    "card number",
    "cvv",
    "cvc",
    "security code",
    "expir",
    "cardholder",
    "card details",
]
CHECKOUT_KEYWORDS = [
    "checkout",
    "place order",
    "complete order",
    "order summary",
    "proceed to payment",
    "pay now",
]
CART_KEYWORDS = [
    "your cart",
    "shopping cart",
    "cart summary",
    "subtotal",
    "view cart",
    "items in cart",
]
REAL_FILL_PATTERNS = [
    r"card",
    r"cvv",
    r"cvc",
    r"expir",
    r"security",
    r"address",
    r"zip",
    r"postal",
    r"city",
    r"state",
    r"first.?name",
    r"last.?name",
    r"email",
    r"phone",
]
ERROR_PATTERNS = [
    (r"not approved|approval denied|blocked by approval", "approval_block"),
    (r"unknown tool", "unknown_tool"),
    (r"timed out|timeout", "timeout"),
    (r"rate limit", "rate_limit"),
    (r"validation error|field required", "validation"),
    (r"not found|404", "not_found"),
    (r"click.*failed|click.*timeout|failed to click", "click_fail"),
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
    fill_count: int
    real_fill_count: int
    reached_payment: bool
    payment_fields_confirmed: bool
    reached_checkout: bool
    reached_cart: bool
    url_has_terminal: bool
    hallucinated_tools: int
    no_tools_steps: int
    site_name: str
    final_url: str
    timeout_like: bool
    error_kinds: dict[str, int]

    @property
    def is_spin(self) -> bool:
        return self.max_repeat_count >= 3

    @property
    def step_count(self) -> int:
        return len(self.task.steps)

    def metrics(self) -> dict[str, Any]:
        return {
            "step_count": self.step_count,
            "unique_urls": len(self.unique_urls),
            "unique_url_paths": len(self.unique_url_paths),
            "unique_tools": len(self.unique_tools),
            "errors": self.errors,
            "error_rate": self.error_rate,
            "fill_count": self.fill_count,
            "real_fill_count": self.real_fill_count,
            "reached_payment": self.reached_payment,
            "payment_fields_confirmed": self.payment_fields_confirmed,
            "reached_checkout": self.reached_checkout,
            "reached_cart": self.reached_cart,
            "url_has_terminal": self.url_has_terminal,
            "hallucinated_tools": self.hallucinated_tools,
            "no_tools_steps": self.no_tools_steps,
            "max_repeat_count": self.max_repeat_count,
            "is_spin": self.is_spin,
            "timeout_like": self.timeout_like,
            "total_duration_ms": self.total_duration_ms,
        }


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
        if step.page_url:
            candidates.append(step.page_url)
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


def detect_signals(step: AgentStep) -> dict[str, bool]:
    result_lower = (step.tool_result or "")[:5000].lower()
    tool_input_text = json.dumps(step.tool_input, sort_keys=True).lower()
    signals = {
        "payment_fields": any(keyword in result_lower for keyword in PAYMENT_FIELD_KEYWORDS),
        "payment_fields_strong": any(keyword in result_lower for keyword in PAYMENT_FIELDS_STRONG),
        "checkout": any(keyword in result_lower for keyword in CHECKOUT_KEYWORDS),
        "cart": any(keyword in result_lower for keyword in CART_KEYWORDS),
        "is_fill": "fill" in step.tool_name.lower() or "fields" in step.tool_input,
        "fill_real": False,
    }
    if signals["is_fill"] and any(
        re.search(pattern, tool_input_text) for pattern in REAL_FILL_PATTERNS
    ):
        signals["fill_real"] = True
    combined = f"{step.tool_input.get('text', '')} {step.tool_input.get('fields', '')}".lower()
    if re.search(r"\d{4}[\s-]?\d{4}[\s-]?\d{4}", combined):
        signals["fill_real"] = True
    if re.search(r"@[\w.-]+\.\w+", combined):
        signals["fill_real"] = True
    if re.search(r"\b\d{5}\b", combined) and "ref" in step.tool_input:
        signals["fill_real"] = True
    return signals


def summarize_tool_result(step: AgentStep, limit: int = 240) -> str:
    text = (step.error or step.tool_result or "").replace("\r", " ").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _unique_urls(task: AgentTask) -> list[str]:
    seen: set[str] = set()
    urls: list[str] = []
    for step in task.sorted_steps:
        if step.page_url and step.page_url not in seen:
            seen.add(step.page_url)
            urls.append(step.page_url)
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


def analyze_task(task: AgentTask) -> TaskAnalysis:
    urls = _unique_urls(task)
    url_paths = _unique_url_paths(urls)
    tool_sequence = [step.tool_name for step in task.sorted_steps]
    unique_tools = sorted(set(tool_sequence))
    repeat_tool, repeat_count = _max_consecutive_repeat(tool_sequence)
    errors = sum(1 for step in task.steps if step.error)
    error_kinds: Counter[str] = Counter()
    fill_count = 0
    real_fill_count = 0
    reached_payment = False
    payment_fields_confirmed = False
    reached_checkout = False
    reached_cart = False
    no_tools_steps = 0
    hallucinated_tools = 0
    for step in task.sorted_steps:
        signals = detect_signals(step)
        fill_count += int(signals["is_fill"])
        real_fill_count += int(signals["fill_real"])
        payment_fields_confirmed = payment_fields_confirmed or signals["payment_fields_strong"]
        reached_payment = reached_payment or signals["payment_fields"]
        reached_checkout = reached_checkout or signals["checkout"]
        reached_cart = reached_cart or signals["cart"]
        if step.tools_available == []:
            no_tools_steps += 1
        error_kind = classify_error(step.error)
        if error_kind:
            error_kinds[error_kind] += 1
        if error_kind == "unknown_tool":
            hallucinated_tools += 1
    url_has_terminal = any(
        any(re.search(pattern, urlparse(url).path.lower()) for pattern in TERMINAL_URL_PATTERNS)
        for url in urls
    )
    reached_payment = payment_fields_confirmed or (reached_payment and url_has_terminal)
    return TaskAnalysis(
        task=task,
        unique_urls=urls,
        unique_url_paths=url_paths,
        unique_tools=unique_tools,
        tool_sequence=tool_sequence,
        max_repeat_tool=repeat_tool,
        max_repeat_count=repeat_count,
        errors=errors,
        error_rate=errors / max(1, len(task.steps)),
        total_duration_ms=sum(step.duration_ms or 0 for step in task.steps),
        fill_count=fill_count,
        real_fill_count=real_fill_count,
        reached_payment=reached_payment,
        payment_fields_confirmed=payment_fields_confirmed,
        reached_checkout=reached_checkout,
        reached_cart=reached_cart,
        url_has_terminal=url_has_terminal,
        hallucinated_tools=hallucinated_tools,
        no_tools_steps=no_tools_steps,
        site_name=extract_site_name(task),
        final_url=urls[-1] if urls else "",
        timeout_like=(
            task.outcome is not None and task.outcome.status in {"timeout", "max_iterations"}
        )
        or len(task.steps) >= 75,
        error_kinds=dict(error_kinds),
    )


def analyze_tasks(tasks: list[AgentTask]) -> dict[str, TaskAnalysis]:
    return {task.task_id: analyze_task(task) for task in tasks}


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


def load_tasks(log_dir: str | Path, days: int | None = None) -> list[AgentTask]:
    root = Path(log_dir)
    if not root.exists():
        raise FileNotFoundError(f"log path does not exist: {root}")
    tasks: dict[str, AgentTask] = {}
    for path in _iter_jsonl_files(root, days=days):
        with path.open(encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
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
                        step.timestamp = payload.get("ts")
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
    raise KeyError(f"task id '{query}' was not uniquely found")


def build_task_tree(tasks: list[AgentTask]) -> dict[str, dict[str, list[str]]]:
    tree: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for task in tasks:
        analysis = analyze_task(task)
        day = task.day or "unknown-day"
        tree[day][analysis.site_name].append(task.task_id)
    return {day: dict(sites) for day, sites in sorted(tree.items())}
