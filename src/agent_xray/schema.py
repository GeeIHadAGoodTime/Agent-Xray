from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

GRADE_ORDER = {
    "BROKEN": 0,
    "WEAK": 1,
    "OK": 2,
    "GOOD": 3,
    "GOLDEN": 4,
}

CORE_STEP_FIELDS = {
    "task_id",
    "step",
    "tool_name",
    "tool_input",
    "tool_result",
    "error",
    "duration_ms",
    "timestamp",
    "ts",
}
MODEL_FIELD_NAMES = {
    "model_name",
    "temperature",
    "tool_choice",
    "context_window",
    "context_usage_pct",
    "compaction_count",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "compaction_method",
    "compaction_messages_before",
    "compaction_messages_after",
    "compaction_summary_preview",
    "trimmed_messages",
    "fifo_evicted_messages",
    "screenshots_evicted",
    "prompt_variant",
    "prompt_variant_full",
}
TOOLS_FIELD_NAMES = {
    "tools_available",
    "tools_available_names",
    "system_prompt_hash",
    "message_count",
    "rejected_tools",
    "focused_set",
    "tools_available_count",
    "conversation_turn_count",
}
REASONING_FIELD_NAMES = {
    "llm_reasoning",
    "correction_messages",
    "spin_intervention",
    "error_registry_context",
    "continuation_nudge",
    "force_termination",
    "hard_loop_breaker",
    "consecutive_failure_warning",
    "approval_path",
}
BROWSER_FIELD_NAMES = {
    "page_url",
    "had_screenshot",
    "had_screenshot_image",
    "snapshot_compressed",
    "snapshot_pre_compress_len",
}
RESERVED_STEP_FIELDS = (
    CORE_STEP_FIELDS
    | {"model", "tools", "reasoning", "browser", "extensions"}
    | MODEL_FIELD_NAMES
    | TOOLS_FIELD_NAMES
    | REASONING_FIELD_NAMES
    | BROWSER_FIELD_NAMES
)


def _coerce_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _coerce_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def _coerce_list_of_str(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _has_context_values(values: dict[str, Any]) -> bool:
    return any(value is not None for value in values.values())


def _merge_extensions(
    payload: dict[str, Any], explicit_extensions: dict[str, Any]
) -> dict[str, Any]:
    merged = dict(explicit_extensions)
    for key, value in payload.items():
        if key not in RESERVED_STEP_FIELDS and key not in merged:
            merged[key] = value
    return merged


@dataclass(slots=True)
class ModelContext:
    """LLM call metadata including compaction and prompt details."""

    model_name: str | None = None
    temperature: float | None = None
    tool_choice: str | None = None
    context_window: int | None = None
    context_usage_pct: float | None = None
    compaction_count: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    compaction_method: str | None = None
    compaction_messages_before: int | None = None
    compaction_messages_after: int | None = None
    compaction_summary_preview: str | None = None
    trimmed_messages: int | None = None
    fifo_evicted_messages: int | None = None
    screenshots_evicted: int | None = None
    prompt_variant: str | None = None
    prompt_variant_full: str | None = None


@dataclass(slots=True)
class ToolContext:
    """Tool availability and filtering metadata."""

    tools_available: list[str] | None = None
    system_prompt_hash: str | None = None
    message_count: int | None = None
    rejected_tools: list[str] | None = None
    focused_set: str | None = None
    tools_available_count: int | None = None
    conversation_turn_count: int | None = None


@dataclass(slots=True)
class ReasoningContext:
    """LLM reasoning, corrections, and dynamic injections."""

    llm_reasoning: str | None = None
    correction_messages: list[str] | None = None
    spin_intervention: str | None = None
    error_registry_context: str | None = None
    continuation_nudge: str | None = None
    force_termination: str | None = None
    hard_loop_breaker: str | None = None
    consecutive_failure_warning: str | None = None
    approval_path: str | None = None


@dataclass(slots=True)
class BrowserContext:
    """Browser/UI agent metadata."""

    page_url: str | None = None
    had_screenshot: bool | None = None
    snapshot_compressed: bool | None = None
    had_screenshot_image: bool | None = None
    snapshot_pre_compress_len: int | None = None


@dataclass(slots=True, init=False)
class AgentStep:
    """Universal agent step record.

    Core fields (8): task_id, step, tool_name, tool_input plus 4 optional.
    Extensions: typed contexts for model, tools, reasoning, browser.
    Catch-all: extensions dict for anything else.
    """

    task_id: str
    step: int
    tool_name: str
    tool_input: dict[str, Any]
    tool_result: str | None = None
    error: str | None = None
    duration_ms: int | None = None
    timestamp: str | None = None
    model: ModelContext | None = None
    tools: ToolContext | None = None
    reasoning: ReasoningContext | None = None
    browser: BrowserContext | None = None
    extensions: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        task_id: str,
        step: int,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_result: str | None = None,
        error: str | None = None,
        duration_ms: int | None = None,
        timestamp: str | None = None,
        model: ModelContext | None = None,
        tools: ToolContext | None = None,
        reasoning: ReasoningContext | None = None,
        browser: BrowserContext | None = None,
        extensions: dict[str, Any] | None = None,
        *,
        model_name: str | None = None,
        temperature: float | None = None,
        tool_choice: str | None = None,
        context_window: int | None = None,
        context_usage_pct: float | None = None,
        compaction_count: int | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cost_usd: float | None = None,
        tools_available: list[str] | None = None,
        system_prompt_hash: str | None = None,
        message_count: int | None = None,
        llm_reasoning: str | None = None,
        correction_messages: list[str] | None = None,
        spin_intervention: str | None = None,
        page_url: str | None = None,
        had_screenshot: bool | None = None,
        snapshot_compressed: bool | None = None,
    ) -> None:
        self.task_id = str(task_id)
        self.step = int(step)
        self.tool_name = str(tool_name)
        self.tool_input = _coerce_dict(tool_input)
        self.tool_result = _coerce_optional_str(tool_result)
        self.error = _coerce_optional_str(error)
        self.duration_ms = _coerce_optional_int(duration_ms)
        self.timestamp = _coerce_optional_str(timestamp)
        self.model = self._merge_model_context(
            model=model,
            model_name=model_name,
            temperature=temperature,
            tool_choice=tool_choice,
            context_window=context_window,
            context_usage_pct=context_usage_pct,
            compaction_count=compaction_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )
        self.tools = self._merge_tool_context(
            tools=tools,
            tools_available=tools_available,
            system_prompt_hash=system_prompt_hash,
            message_count=message_count,
        )
        self.reasoning = self._merge_reasoning_context(
            reasoning=reasoning,
            llm_reasoning=llm_reasoning,
            correction_messages=correction_messages,
            spin_intervention=spin_intervention,
        )
        self.browser = self._merge_browser_context(
            browser=browser,
            page_url=page_url,
            had_screenshot=had_screenshot,
            snapshot_compressed=snapshot_compressed,
        )
        self.extensions = dict(extensions or {})

    @staticmethod
    def _merge_model_context(
        *,
        model: ModelContext | None,
        model_name: str | None,
        temperature: float | None,
        tool_choice: str | None,
        context_window: int | None,
        context_usage_pct: float | None,
        compaction_count: int | None,
        input_tokens: int | None,
        output_tokens: int | None,
        cost_usd: float | None,
    ) -> ModelContext | None:
        values = asdict(model) if model else {}
        overrides = {
            "model_name": model_name,
            "temperature": temperature,
            "tool_choice": tool_choice,
            "context_window": context_window,
            "context_usage_pct": context_usage_pct,
            "compaction_count": compaction_count,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
        }
        for key, value in overrides.items():
            if value is not None:
                values[key] = value
        if not _has_context_values(values):
            return None
        return ModelContext(
            model_name=_coerce_optional_str(values.get("model_name")),
            temperature=_coerce_optional_float(values.get("temperature")),
            tool_choice=_coerce_optional_str(values.get("tool_choice")),
            context_window=_coerce_optional_int(values.get("context_window")),
            context_usage_pct=_coerce_optional_float(values.get("context_usage_pct")),
            compaction_count=_coerce_optional_int(values.get("compaction_count")),
            input_tokens=_coerce_optional_int(values.get("input_tokens")),
            output_tokens=_coerce_optional_int(values.get("output_tokens")),
            cost_usd=_coerce_optional_float(values.get("cost_usd")),
            compaction_method=_coerce_optional_str(values.get("compaction_method")),
            compaction_messages_before=_coerce_optional_int(values.get("compaction_messages_before")),
            compaction_messages_after=_coerce_optional_int(values.get("compaction_messages_after")),
            compaction_summary_preview=_coerce_optional_str(values.get("compaction_summary_preview")),
            trimmed_messages=_coerce_optional_int(values.get("trimmed_messages")),
            fifo_evicted_messages=_coerce_optional_int(values.get("fifo_evicted_messages")),
            screenshots_evicted=_coerce_optional_int(values.get("screenshots_evicted")),
            prompt_variant=_coerce_optional_str(values.get("prompt_variant")),
            prompt_variant_full=_coerce_optional_str(values.get("prompt_variant_full")),
        )

    @staticmethod
    def _merge_tool_context(
        *,
        tools: ToolContext | None,
        tools_available: list[str] | None,
        system_prompt_hash: str | None,
        message_count: int | None,
    ) -> ToolContext | None:
        values = asdict(tools) if tools else {}
        overrides = {
            "tools_available": tools_available,
            "system_prompt_hash": system_prompt_hash,
            "message_count": message_count,
        }
        for key, value in overrides.items():
            if value is not None:
                values[key] = value
        if not _has_context_values(values):
            return None
        return ToolContext(
            tools_available=_coerce_list_of_str(values.get("tools_available")),
            system_prompt_hash=_coerce_optional_str(values.get("system_prompt_hash")),
            message_count=_coerce_optional_int(values.get("message_count")),
            rejected_tools=_coerce_list_of_str(values.get("rejected_tools")),
            focused_set=_coerce_optional_str(values.get("focused_set")),
            tools_available_count=_coerce_optional_int(values.get("tools_available_count")),
            conversation_turn_count=_coerce_optional_int(values.get("conversation_turn_count")),
        )

    @staticmethod
    def _merge_reasoning_context(
        *,
        reasoning: ReasoningContext | None,
        llm_reasoning: str | None,
        correction_messages: list[str] | None,
        spin_intervention: str | None,
    ) -> ReasoningContext | None:
        values = asdict(reasoning) if reasoning else {}
        overrides = {
            "llm_reasoning": llm_reasoning,
            "correction_messages": correction_messages,
            "spin_intervention": spin_intervention,
        }
        for key, value in overrides.items():
            if value is not None:
                values[key] = value
        if not _has_context_values(values):
            return None
        return ReasoningContext(
            llm_reasoning=_coerce_optional_str(values.get("llm_reasoning")),
            correction_messages=_coerce_list_of_str(values.get("correction_messages")),
            spin_intervention=_coerce_optional_str(values.get("spin_intervention")),
            error_registry_context=_coerce_optional_str(values.get("error_registry_context")),
            continuation_nudge=_coerce_optional_str(values.get("continuation_nudge")),
            force_termination=_coerce_optional_str(values.get("force_termination")),
            hard_loop_breaker=_coerce_optional_str(values.get("hard_loop_breaker")),
            consecutive_failure_warning=_coerce_optional_str(values.get("consecutive_failure_warning")),
            approval_path=_coerce_optional_str(values.get("approval_path")),
        )

    @staticmethod
    def _merge_browser_context(
        *,
        browser: BrowserContext | None,
        page_url: str | None,
        had_screenshot: bool | None,
        snapshot_compressed: bool | None,
    ) -> BrowserContext | None:
        values = asdict(browser) if browser else {}
        overrides = {
            "page_url": page_url,
            "had_screenshot": had_screenshot,
            "snapshot_compressed": snapshot_compressed,
        }
        for key, value in overrides.items():
            if value is not None:
                values[key] = value
        if not _has_context_values(values):
            return None
        return BrowserContext(
            page_url=_coerce_optional_str(values.get("page_url")),
            had_screenshot=_coerce_optional_bool(values.get("had_screenshot")),
            snapshot_compressed=_coerce_optional_bool(values.get("snapshot_compressed")),
            had_screenshot_image=_coerce_optional_bool(values.get("had_screenshot_image")),
            snapshot_pre_compress_len=_coerce_optional_int(values.get("snapshot_pre_compress_len")),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AgentStep:
        model_payload = _coerce_dict(payload.get("model"))
        tools_payload = _coerce_dict(payload.get("tools"))
        reasoning_payload = _coerce_dict(payload.get("reasoning"))
        browser_payload = _coerce_dict(payload.get("browser"))

        if "model_name" in payload and "model_name" not in model_payload:
            model_payload["model_name"] = payload.get("model_name")
        if "temperature" in payload and "temperature" not in model_payload:
            model_payload["temperature"] = payload.get("temperature")
        if "tool_choice" in payload and "tool_choice" not in model_payload:
            model_payload["tool_choice"] = payload.get("tool_choice")
        if "context_window" in payload and "context_window" not in model_payload:
            model_payload["context_window"] = payload.get("context_window")
        if "context_usage_pct" in payload and "context_usage_pct" not in model_payload:
            model_payload["context_usage_pct"] = payload.get("context_usage_pct")
        if "compaction_count" in payload and "compaction_count" not in model_payload:
            model_payload["compaction_count"] = payload.get("compaction_count")
        if "input_tokens" in payload and "input_tokens" not in model_payload:
            model_payload["input_tokens"] = payload.get("input_tokens")
        if "output_tokens" in payload and "output_tokens" not in model_payload:
            model_payload["output_tokens"] = payload.get("output_tokens")
        if "cost_usd" in payload and "cost_usd" not in model_payload:
            model_payload["cost_usd"] = payload.get("cost_usd")
        for _mf in (
            "compaction_method", "compaction_messages_before", "compaction_messages_after",
            "compaction_summary_preview", "trimmed_messages", "fifo_evicted_messages",
            "screenshots_evicted", "prompt_variant", "prompt_variant_full",
        ):
            if _mf in payload and _mf not in model_payload:
                model_payload[_mf] = payload.get(_mf)

        tools_available = payload.get("tools_available")
        if tools_available is None:
            tools_available = payload.get("tools_available_names")
        if tools_available is not None and "tools_available" not in tools_payload:
            tools_payload["tools_available"] = tools_available
        if "system_prompt_hash" in payload and "system_prompt_hash" not in tools_payload:
            tools_payload["system_prompt_hash"] = payload.get("system_prompt_hash")
        if "message_count" in payload and "message_count" not in tools_payload:
            tools_payload["message_count"] = payload.get("message_count")
        for _tf in ("rejected_tools", "focused_set", "tools_available_count", "conversation_turn_count"):
            if _tf in payload and _tf not in tools_payload:
                tools_payload[_tf] = payload.get(_tf)

        if "llm_reasoning" in payload and "llm_reasoning" not in reasoning_payload:
            reasoning_payload["llm_reasoning"] = payload.get("llm_reasoning")
        if "correction_messages" in payload and "correction_messages" not in reasoning_payload:
            reasoning_payload["correction_messages"] = payload.get("correction_messages")
        if "spin_intervention" in payload and "spin_intervention" not in reasoning_payload:
            reasoning_payload["spin_intervention"] = payload.get("spin_intervention")
        for _rf in (
            "error_registry_context", "continuation_nudge", "force_termination",
            "hard_loop_breaker", "consecutive_failure_warning", "approval_path",
        ):
            if _rf in payload and _rf not in reasoning_payload:
                reasoning_payload[_rf] = payload.get(_rf)

        if "page_url" in payload and "page_url" not in browser_payload:
            browser_payload["page_url"] = payload.get("page_url")
        if "had_screenshot" in payload and "had_screenshot" not in browser_payload:
            browser_payload["had_screenshot"] = payload.get("had_screenshot")
        if (
            "had_screenshot_image" in payload
            and "had_screenshot" not in browser_payload
            and "had_screenshot" not in payload
        ):
            browser_payload["had_screenshot"] = payload.get("had_screenshot_image")
        if "had_screenshot_image" in payload and "had_screenshot_image" not in browser_payload:
            browser_payload["had_screenshot_image"] = payload.get("had_screenshot_image")
        if "snapshot_compressed" in payload and "snapshot_compressed" not in browser_payload:
            browser_payload["snapshot_compressed"] = payload.get("snapshot_compressed")
        if "snapshot_pre_compress_len" in payload and "snapshot_pre_compress_len" not in browser_payload:
            browser_payload["snapshot_pre_compress_len"] = payload.get("snapshot_pre_compress_len")

        step_value = _coerce_optional_int(payload.get("step", 0))
        timestamp_value = payload.get("timestamp") or payload.get("ts")
        return cls(
            task_id=str(payload.get("task_id", "")),
            step=step_value if step_value is not None else 0,
            tool_name=str(payload.get("tool_name", "")),
            tool_input=_coerce_dict(payload.get("tool_input")),
            tool_result=_coerce_optional_str(payload.get("tool_result")),
            error=_coerce_optional_str(payload.get("error")),
            duration_ms=_coerce_optional_int(payload.get("duration_ms")),
            timestamp=_coerce_optional_str(timestamp_value),
            model=(
                None
                if not _has_context_values(model_payload)
                else ModelContext(
                    model_name=_coerce_optional_str(model_payload.get("model_name")),
                    temperature=_coerce_optional_float(model_payload.get("temperature")),
                    tool_choice=_coerce_optional_str(model_payload.get("tool_choice")),
                    context_window=_coerce_optional_int(model_payload.get("context_window")),
                    context_usage_pct=_coerce_optional_float(
                        model_payload.get("context_usage_pct")
                    ),
                    compaction_count=_coerce_optional_int(model_payload.get("compaction_count")),
                    input_tokens=_coerce_optional_int(model_payload.get("input_tokens")),
                    output_tokens=_coerce_optional_int(model_payload.get("output_tokens")),
                    cost_usd=_coerce_optional_float(model_payload.get("cost_usd")),
                    compaction_method=_coerce_optional_str(model_payload.get("compaction_method")),
                    compaction_messages_before=_coerce_optional_int(model_payload.get("compaction_messages_before")),
                    compaction_messages_after=_coerce_optional_int(model_payload.get("compaction_messages_after")),
                    compaction_summary_preview=_coerce_optional_str(model_payload.get("compaction_summary_preview")),
                    trimmed_messages=_coerce_optional_int(model_payload.get("trimmed_messages")),
                    fifo_evicted_messages=_coerce_optional_int(model_payload.get("fifo_evicted_messages")),
                    screenshots_evicted=_coerce_optional_int(model_payload.get("screenshots_evicted")),
                    prompt_variant=_coerce_optional_str(model_payload.get("prompt_variant")),
                    prompt_variant_full=_coerce_optional_str(model_payload.get("prompt_variant_full")),
                )
            ),
            tools=(
                None
                if not _has_context_values(tools_payload)
                else ToolContext(
                    tools_available=_coerce_list_of_str(tools_payload.get("tools_available")),
                    system_prompt_hash=_coerce_optional_str(
                        tools_payload.get("system_prompt_hash")
                    ),
                    message_count=_coerce_optional_int(tools_payload.get("message_count")),
                    rejected_tools=_coerce_list_of_str(tools_payload.get("rejected_tools")),
                    focused_set=_coerce_optional_str(tools_payload.get("focused_set")),
                    tools_available_count=_coerce_optional_int(tools_payload.get("tools_available_count")),
                    conversation_turn_count=_coerce_optional_int(tools_payload.get("conversation_turn_count")),
                )
            ),
            reasoning=(
                None
                if not _has_context_values(reasoning_payload)
                else ReasoningContext(
                    llm_reasoning=_coerce_optional_str(reasoning_payload.get("llm_reasoning")),
                    correction_messages=_coerce_list_of_str(
                        reasoning_payload.get("correction_messages")
                    ),
                    spin_intervention=_coerce_optional_str(
                        reasoning_payload.get("spin_intervention")
                    ),
                    error_registry_context=_coerce_optional_str(reasoning_payload.get("error_registry_context")),
                    continuation_nudge=_coerce_optional_str(reasoning_payload.get("continuation_nudge")),
                    force_termination=_coerce_optional_str(reasoning_payload.get("force_termination")),
                    hard_loop_breaker=_coerce_optional_str(reasoning_payload.get("hard_loop_breaker")),
                    consecutive_failure_warning=_coerce_optional_str(reasoning_payload.get("consecutive_failure_warning")),
                    approval_path=_coerce_optional_str(reasoning_payload.get("approval_path")),
                )
            ),
            browser=(
                None
                if not _has_context_values(browser_payload)
                else BrowserContext(
                    page_url=_coerce_optional_str(browser_payload.get("page_url")),
                    had_screenshot=_coerce_optional_bool(browser_payload.get("had_screenshot")),
                    snapshot_compressed=_coerce_optional_bool(
                        browser_payload.get("snapshot_compressed")
                    ),
                    had_screenshot_image=_coerce_optional_bool(browser_payload.get("had_screenshot_image")),
                    snapshot_pre_compress_len=_coerce_optional_int(browser_payload.get("snapshot_pre_compress_len")),
                )
            ),
            extensions=_merge_extensions(payload, _coerce_dict(payload.get("extensions"))),
        )

    @property
    def model_name(self) -> str | None:
        return self.model.model_name if self.model else None

    @property
    def page_url(self) -> str | None:
        return self.browser.page_url if self.browser else None

    @property
    def llm_reasoning(self) -> str | None:
        return self.reasoning.llm_reasoning if self.reasoning else None

    @property
    def tools_available(self) -> list[str] | None:
        return self.tools.tools_available if self.tools else None

    @property
    def tools_available_names(self) -> list[str] | None:
        return self.tools.tools_available if self.tools else None

    @property
    def temperature(self) -> float | None:
        return self.model.temperature if self.model else None

    @property
    def tool_choice(self) -> str | None:
        return self.model.tool_choice if self.model else None

    @property
    def message_count(self) -> int | None:
        return self.tools.message_count if self.tools else None

    @property
    def system_prompt_hash(self) -> str | None:
        return self.tools.system_prompt_hash if self.tools else None

    @property
    def context_usage_pct(self) -> float | None:
        return self.model.context_usage_pct if self.model else None

    @property
    def context_window(self) -> int | None:
        return self.model.context_window if self.model else None

    @property
    def compaction_count(self) -> int | None:
        return self.model.compaction_count if self.model else None

    @property
    def snapshot_compressed(self) -> bool | None:
        return self.browser.snapshot_compressed if self.browser else None

    @property
    def had_screenshot(self) -> bool | None:
        return self.browser.had_screenshot if self.browser else None

    @property
    def correction_messages(self) -> list[str] | None:
        return self.reasoning.correction_messages if self.reasoning else None

    @property
    def spin_intervention(self) -> str | None:
        return self.reasoning.spin_intervention if self.reasoning else None

    @property
    def input_tokens(self) -> int | None:
        return self.model.input_tokens if self.model else None

    @property
    def output_tokens(self) -> int | None:
        return self.model.output_tokens if self.model else None

    @property
    def cost_usd(self) -> float | None:
        return self.model.cost_usd if self.model else None

    @property
    def rejected_tools(self) -> list[str] | None:
        return self.tools.rejected_tools if self.tools else None

    @property
    def focused_set(self) -> str | None:
        return self.tools.focused_set if self.tools else None

    @property
    def approval_path(self) -> str | None:
        return self.reasoning.approval_path if self.reasoning else None

    @property
    def compaction_method(self) -> str | None:
        return self.model.compaction_method if self.model else None

    @property
    def prompt_variant(self) -> str | None:
        return self.model.prompt_variant if self.model else None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TaskOutcome:
    task_id: str
    status: str
    final_answer: str | None = None
    total_steps: int | None = None
    total_duration_s: float | None = None
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TaskOutcome:
        metadata = {
            key: value
            for key, value in payload.items()
            if key
            not in {
                "event",
                "task_id",
                "outcome",
                "status",
                "final_answer",
                "total_steps",
                "total_duration_s",
                "timestamp",
                "ts",
            }
        }
        return cls(
            task_id=str(payload.get("task_id", "")),
            status=str(payload.get("outcome") or payload.get("status") or ""),
            final_answer=_coerce_optional_str(payload.get("final_answer")),
            total_steps=_coerce_optional_int(payload.get("total_steps")),
            total_duration_s=_coerce_optional_float(payload.get("total_duration_s")),
            timestamp=_coerce_optional_str(payload.get("timestamp") or payload.get("ts")),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentTask:
    task_id: str
    steps: list[AgentStep] = field(default_factory=list)
    task_text: str | None = None
    task_category: str | None = None
    day: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    outcome: TaskOutcome | None = None

    @property
    def sorted_steps(self) -> list[AgentStep]:
        return sorted(self.steps, key=lambda step: step.step)

    @classmethod
    def from_steps(
        cls,
        steps: list[AgentStep],
        *,
        task_id: str | None = None,
        task_text: str | None = None,
        task_category: str | None = None,
        metadata: dict[str, Any] | None = None,
        outcome: TaskOutcome | None = None,
    ) -> AgentTask:
        resolved_task_id = task_id or (steps[0].task_id if steps else "task")
        return cls(
            task_id=resolved_task_id,
            steps=list(steps),
            task_text=task_text,
            task_category=task_category,
            metadata=dict(metadata or {}),
            outcome=outcome,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_text": self.task_text,
            "task_category": self.task_category,
            "day": self.day,
            "metadata": self.metadata,
            "outcome": self.outcome.to_dict() if self.outcome else None,
            "steps": [step.to_dict() for step in self.sorted_steps],
        }


MODEL_CONTEXT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "model_name": {"type": ["string", "null"]},
        "temperature": {"type": ["number", "null"]},
        "tool_choice": {"type": ["string", "null"]},
        "context_window": {"type": ["integer", "null"], "minimum": 0},
        "context_usage_pct": {"type": ["number", "null"]},
        "compaction_count": {"type": ["integer", "null"], "minimum": 0},
        "input_tokens": {"type": ["integer", "null"], "minimum": 0},
        "output_tokens": {"type": ["integer", "null"], "minimum": 0},
        "cost_usd": {"type": ["number", "null"]},
        "compaction_method": {"type": ["string", "null"]},
        "compaction_messages_before": {"type": ["integer", "null"], "minimum": 0},
        "compaction_messages_after": {"type": ["integer", "null"], "minimum": 0},
        "compaction_summary_preview": {"type": ["string", "null"]},
        "trimmed_messages": {"type": ["integer", "null"], "minimum": 0},
        "fifo_evicted_messages": {"type": ["integer", "null"], "minimum": 0},
        "screenshots_evicted": {"type": ["integer", "null"], "minimum": 0},
        "prompt_variant": {"type": ["string", "null"]},
        "prompt_variant_full": {"type": ["string", "null"]},
    },
    "additionalProperties": False,
}

TOOL_CONTEXT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tools_available": {"type": ["array", "null"], "items": {"type": "string"}},
        "system_prompt_hash": {"type": ["string", "null"]},
        "message_count": {"type": ["integer", "null"], "minimum": 0},
        "rejected_tools": {"type": ["array", "null"], "items": {"type": "string"}},
        "focused_set": {"type": ["string", "null"]},
        "tools_available_count": {"type": ["integer", "null"], "minimum": 0},
        "conversation_turn_count": {"type": ["integer", "null"], "minimum": 0},
    },
    "additionalProperties": False,
}

REASONING_CONTEXT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "llm_reasoning": {"type": ["string", "null"]},
        "correction_messages": {"type": ["array", "null"], "items": {"type": "string"}},
        "spin_intervention": {"type": ["string", "null"]},
        "error_registry_context": {"type": ["string", "null"]},
        "continuation_nudge": {"type": ["string", "null"]},
        "force_termination": {"type": ["string", "null"]},
        "hard_loop_breaker": {"type": ["string", "null"]},
        "consecutive_failure_warning": {"type": ["string", "null"]},
        "approval_path": {"type": ["string", "null"]},
    },
    "additionalProperties": False,
}

BROWSER_CONTEXT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "page_url": {"type": ["string", "null"]},
        "had_screenshot": {"type": ["boolean", "null"]},
        "snapshot_compressed": {"type": ["boolean", "null"]},
        "had_screenshot_image": {"type": ["boolean", "null"]},
        "snapshot_pre_compress_len": {"type": ["integer", "null"], "minimum": 0},
    },
    "additionalProperties": False,
}

AGENT_STEP_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "AgentStep",
    "type": "object",
    "required": ["task_id", "step", "tool_name", "tool_input"],
    "properties": {
        "task_id": {"type": "string"},
        "step": {"type": "integer", "minimum": 0},
        "tool_name": {"type": "string"},
        "tool_input": {"type": "object"},
        "tool_result": {"type": ["string", "null"]},
        "error": {"type": ["string", "null"]},
        "duration_ms": {"type": ["integer", "null"], "minimum": 0},
        "timestamp": {"type": ["string", "null"]},
        "model": {"anyOf": [{"type": "null"}, MODEL_CONTEXT_JSON_SCHEMA]},
        "tools": {"anyOf": [{"type": "null"}, TOOL_CONTEXT_JSON_SCHEMA]},
        "reasoning": {"anyOf": [{"type": "null"}, REASONING_CONTEXT_JSON_SCHEMA]},
        "browser": {"anyOf": [{"type": "null"}, BROWSER_CONTEXT_JSON_SCHEMA]},
        "extensions": {"type": "object"},
        "model_name": {"type": ["string", "null"], "deprecated": True},
        "temperature": {"type": ["number", "null"], "deprecated": True},
        "tool_choice": {"type": ["string", "null"], "deprecated": True},
        "message_count": {"type": ["integer", "null"], "minimum": 0, "deprecated": True},
        "tools_available": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "deprecated": True,
        },
        "tools_available_names": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "deprecated": True,
        },
        "llm_reasoning": {"type": ["string", "null"], "deprecated": True},
        "page_url": {"type": ["string", "null"], "deprecated": True},
        "system_prompt_hash": {"type": ["string", "null"], "deprecated": True},
        "context_usage_pct": {"type": ["number", "null"], "deprecated": True},
        "context_window": {"type": ["integer", "null"], "minimum": 0, "deprecated": True},
        "compaction_count": {"type": ["integer", "null"], "minimum": 0, "deprecated": True},
        "snapshot_compressed": {"type": ["boolean", "null"], "deprecated": True},
        "had_screenshot": {"type": ["boolean", "null"], "deprecated": True},
        "had_screenshot_image": {"type": ["boolean", "null"], "deprecated": True},
        "correction_messages": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "deprecated": True,
        },
        "spin_intervention": {"type": ["string", "null"], "deprecated": True},
        "input_tokens": {"type": ["integer", "null"], "minimum": 0, "deprecated": True},
        "output_tokens": {"type": ["integer", "null"], "minimum": 0, "deprecated": True},
        "cost_usd": {"type": ["number", "null"], "deprecated": True},
    },
    "additionalProperties": True,
}

TASK_OUTCOME_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "TaskOutcome",
    "type": "object",
    "required": ["task_id", "status"],
    "properties": {
        "task_id": {"type": "string"},
        "status": {"type": "string"},
        "final_answer": {"type": ["string", "null"]},
        "total_steps": {"type": ["integer", "null"], "minimum": 0},
        "total_duration_s": {"type": ["number", "null"], "minimum": 0},
        "timestamp": {"type": ["string", "null"]},
        "metadata": {"type": "object"},
    },
    "additionalProperties": True,
}

AGENT_TASK_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "AgentTask",
    "type": "object",
    "required": ["task_id", "steps"],
    "properties": {
        "task_id": {"type": "string"},
        "task_text": {"type": ["string", "null"]},
        "task_category": {"type": ["string", "null"]},
        "day": {"type": ["string", "null"]},
        "metadata": {"type": "object"},
        "outcome": {"anyOf": [{"type": "null"}, TASK_OUTCOME_JSON_SCHEMA]},
        "steps": {"type": "array", "items": AGENT_STEP_JSON_SCHEMA},
    },
    "additionalProperties": True,
}
