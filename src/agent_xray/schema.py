from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, cast, overload

SCHEMA_VERSION = "1.0"

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
    "schema_version",
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
    "llm_decision",
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
    "browser_tiers_used",
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
    """Structured metadata captured for one model call.

    Attributes:
        model_name: Model identifier recorded for the step.
        temperature: Sampling temperature used by the model call.
        tool_choice: Tool-choice mode or explicit tool override.
        context_window: Maximum context window exposed to the model.
        context_usage_pct: Fraction of the context window consumed.
        compaction_count: Number of prompt compactions applied.
        input_tokens: Input token count attributed to the step.
        output_tokens: Output token count attributed to the step.
        cost_usd: Estimated per-step cost in US dollars.
        compaction_method: Name of the compaction strategy, if any.
        compaction_messages_before: Message count before compaction.
        compaction_messages_after: Message count after compaction.
        compaction_summary_preview: Short preview of the compaction summary.
        trimmed_messages: Number of messages trimmed from the prompt.
        fifo_evicted_messages: Number of FIFO-evicted messages.
        screenshots_evicted: Number of screenshots evicted from context.
        prompt_variant: Short label for the active prompt variant.
        prompt_variant_full: Fully qualified prompt variant identifier.
    """

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
    """Metadata describing the tools visible to the agent at a step.

    Attributes:
        tools_available: Tool names exposed to the model.
        system_prompt_hash: Stable hash of the effective system prompt.
        message_count: Number of messages currently in the prompt history.
        rejected_tools: Tools removed by routing or policy logic.
        focused_set: Name of the focused tool subset, if any.
        tools_available_count: Explicit tool count from the source trace.
        conversation_turn_count: Number of conversational turns represented.
    """

    tools_available: list[str] | None = None
    system_prompt_hash: str | None = None
    message_count: int | None = None
    rejected_tools: list[str] | None = None
    focused_set: str | None = None
    tools_available_count: int | None = None
    conversation_turn_count: int | None = None


@dataclass(slots=True)
class ReasoningContext:
    """Reasoning metadata and runtime interventions for a step.

    Attributes:
        llm_reasoning: Captured reasoning or scratchpad text.
        correction_messages: Corrective messages injected before the decision.
        spin_intervention: Loop-breaking intervention text, if any.
        error_registry_context: Recent error context surfaced to the model.
        continuation_nudge: Prompt fragment nudging the model to continue.
        force_termination: Forced-stop instruction injected into context.
        hard_loop_breaker: Strong loop-breaker message from the runner.
        consecutive_failure_warning: Warning about repeated recent failures.
        approval_path: Approval-policy path relevant to the step.
    """

    llm_reasoning: str | None = None
    llm_decision: str | None = None
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
    """Browser-specific state captured alongside a step.

    Attributes:
        page_url: Current page URL seen by the browser agent.
        had_screenshot: Whether a screenshot was available.
        snapshot_compressed: Whether the snapshot was compressed.
        had_screenshot_image: Whether an image screenshot was attached.
        snapshot_pre_compress_len: Snapshot size before compression.
    """

    page_url: str | None = None
    had_screenshot: bool | None = None
    snapshot_compressed: bool | None = None
    had_screenshot_image: bool | None = None
    snapshot_pre_compress_len: int | None = None
    browser_tiers_used: list[str] | None = None


MODEL_CONTEXT_COERCIONS: dict[str, Callable[[Any], Any]] = {
    "model_name": _coerce_optional_str,
    "temperature": _coerce_optional_float,
    "tool_choice": _coerce_optional_str,
    "context_window": _coerce_optional_int,
    "context_usage_pct": _coerce_optional_float,
    "compaction_count": _coerce_optional_int,
    "input_tokens": _coerce_optional_int,
    "output_tokens": _coerce_optional_int,
    "cost_usd": _coerce_optional_float,
    "compaction_method": _coerce_optional_str,
    "compaction_messages_before": _coerce_optional_int,
    "compaction_messages_after": _coerce_optional_int,
    "compaction_summary_preview": _coerce_optional_str,
    "trimmed_messages": _coerce_optional_int,
    "fifo_evicted_messages": _coerce_optional_int,
    "screenshots_evicted": _coerce_optional_int,
    "prompt_variant": _coerce_optional_str,
    "prompt_variant_full": _coerce_optional_str,
}
TOOL_CONTEXT_COERCIONS: dict[str, Callable[[Any], Any]] = {
    "tools_available": _coerce_list_of_str,
    "system_prompt_hash": _coerce_optional_str,
    "message_count": _coerce_optional_int,
    "rejected_tools": _coerce_list_of_str,
    "focused_set": _coerce_optional_str,
    "tools_available_count": _coerce_optional_int,
    "conversation_turn_count": _coerce_optional_int,
}
REASONING_CONTEXT_COERCIONS: dict[str, Callable[[Any], Any]] = {
    "llm_reasoning": _coerce_optional_str,
    "llm_decision": _coerce_optional_str,
    "correction_messages": _coerce_list_of_str,
    "spin_intervention": _coerce_optional_str,
    "error_registry_context": _coerce_optional_str,
    "continuation_nudge": _coerce_optional_str,
    "force_termination": _coerce_optional_str,
    "hard_loop_breaker": _coerce_optional_str,
    "consecutive_failure_warning": _coerce_optional_str,
    "approval_path": _coerce_optional_str,
}
BROWSER_CONTEXT_COERCIONS: dict[str, Callable[[Any], Any]] = {
    "page_url": _coerce_optional_str,
    "had_screenshot": _coerce_optional_bool,
    "snapshot_compressed": _coerce_optional_bool,
    "had_screenshot_image": _coerce_optional_bool,
    "snapshot_pre_compress_len": _coerce_optional_int,
    "browser_tiers_used": _coerce_list_of_str,
}
CONTEXT_FACTORIES = {
    "model": ModelContext,
    "tools": ToolContext,
    "reasoning": ReasoningContext,
    "browser": BrowserContext,
}
CONTEXT_FIELD_COERCIONS = {
    "model": MODEL_CONTEXT_COERCIONS,
    "tools": TOOL_CONTEXT_COERCIONS,
    "reasoning": REASONING_CONTEXT_COERCIONS,
    "browser": BROWSER_CONTEXT_COERCIONS,
}
FIELD_MAP: dict[str, tuple[str, Callable[[Any], Any]]] = {
    **{
        field_name: ("model", coercion_fn)
        for field_name, coercion_fn in MODEL_CONTEXT_COERCIONS.items()
    },
    **{
        field_name: ("tools", coercion_fn)
        for field_name, coercion_fn in TOOL_CONTEXT_COERCIONS.items()
    },
    "tools_available_names": ("tools", _coerce_list_of_str),
    **{
        field_name: ("reasoning", coercion_fn)
        for field_name, coercion_fn in REASONING_CONTEXT_COERCIONS.items()
    },
    **{
        field_name: ("browser", coercion_fn)
        for field_name, coercion_fn in BROWSER_CONTEXT_COERCIONS.items()
    },
}
FIELD_ALIASES = {
    "tools_available_names": "tools_available",
}


@overload
def _build_context(
    context_name: Literal["model"], payload: dict[str, Any]
) -> ModelContext | None: ...


@overload
def _build_context(
    context_name: Literal["tools"], payload: dict[str, Any]
) -> ToolContext | None: ...


@overload
def _build_context(
    context_name: Literal["reasoning"], payload: dict[str, Any]
) -> ReasoningContext | None: ...


@overload
def _build_context(
    context_name: Literal["browser"], payload: dict[str, Any]
) -> BrowserContext | None: ...


def _build_context(
    context_name: str, payload: dict[str, Any]
) -> ModelContext | ToolContext | ReasoningContext | BrowserContext | None:
    if not _has_context_values(payload):
        return None
    coerced_payload = {
        field_name: coercion_fn(payload.get(field_name))
        for field_name, coercion_fn in CONTEXT_FIELD_COERCIONS[context_name].items()
    }
    return cast(  # type narrowing handled by @overload
        ModelContext | ToolContext | ReasoningContext | BrowserContext,
        CONTEXT_FACTORIES[context_name](**coerced_payload),
    )


def _resolve_schema_version(payload: dict[str, Any], *, scope: str) -> str:
    schema_version = _coerce_optional_str(payload.get("schema_version")) or SCHEMA_VERSION
    if schema_version != SCHEMA_VERSION:
        warnings.warn(
            (
                f"{scope} schema_version '{schema_version}' does not match "
                f"current schema version '{SCHEMA_VERSION}'."
            ),
            stacklevel=2,
        )
    return schema_version


def _validate_task_id(payload: dict[str, Any]) -> str:
    if "task_id" not in payload:
        return str(payload.get("task_id", ""))
    task_id = _coerce_optional_str(payload.get("task_id"))
    if task_id is None or not task_id.strip():
        raise ValueError("task_id must be a non-empty string when provided")
    return task_id


def _validate_step(payload: dict[str, Any]) -> int:
    if "step" not in payload:
        return 0
    step = _coerce_optional_int(payload.get("step"))
    if step is None or step < 0:
        raise ValueError("step must be a non-negative integer when provided")
    return step


@dataclass(slots=True, init=False)
class AgentStep:
    """Normalized record for a single agent step and its typed sub-contexts.

    Args:
        task_id: Stable task identifier used to group related steps.
        step: Step index within the task.
        tool_name: Tool invoked by the agent.
        tool_input: Structured tool input payload.
        tool_result: Optional textual tool output.
        error: Optional error message for the step.
        duration_ms: Step duration in milliseconds.
        timestamp: Step timestamp as a string.
        schema_version: Schema version for serialized compatibility.
        model: Pre-built model metadata for the step.
        tools: Pre-built tool metadata for the step.
        reasoning: Pre-built reasoning metadata for the step.
        browser: Pre-built browser metadata for the step.
        extensions: Additional non-schema fields preserved from the trace.

    Example:
        >>> step = AgentStep(
        ...     task_id="checkout-1",
        ...     step=1,
        ...     tool_name="browser_navigate",
        ...     tool_input={"url": "https://shop.example.test"},
        ...     page_url="https://shop.example.test",
        ... )
        >>> step.page_url
        'https://shop.example.test'
    """

    task_id: str
    step: int
    tool_name: str
    tool_input: dict[str, Any]
    tool_result: str | None = None
    error: str | None = None
    duration_ms: int | None = None
    timestamp: str | None = None
    schema_version: str = SCHEMA_VERSION
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
        schema_version: str = SCHEMA_VERSION,
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
        self.schema_version = _coerce_optional_str(schema_version) or SCHEMA_VERSION
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
        return _build_context("model", values)

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
        return _build_context("tools", values)

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
        return _build_context("reasoning", values)

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
        return _build_context("browser", values)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AgentStep:
        """Build an :class:`AgentStep` from flat or nested serialized data.

        Args:
            payload: Raw step payload. Nested context objects are accepted under
                ``model``, ``tools``, ``reasoning``, and ``browser``. Legacy flat
                fields are hoisted into those contexts automatically.

        Returns:
            A normalized ``AgentStep`` instance.

        Example:
            >>> step = AgentStep.from_dict(
            ...     {"task_id": "task-1", "step": "2", "tool_name": "respond", "tool_input": {}}
            ... )
            >>> step.step
            2
        """
        context_payloads = {
            context_name: _coerce_dict(payload.get(context_name))
            for context_name in CONTEXT_FACTORIES
        }
        for field_name, (target_context, coercion_fn) in FIELD_MAP.items():
            if field_name not in payload:
                continue
            target_payload = context_payloads[target_context]
            target_field = FIELD_ALIASES.get(field_name, field_name)
            if target_field not in target_payload:
                target_payload[target_field] = coercion_fn(payload.get(field_name))

        # Extract token counts from nested llm_usage if present
        if "llm_usage" in payload and isinstance(payload["llm_usage"], dict):
            usage = payload["llm_usage"]
            model_data = context_payloads.get("model", {})
            if "input_tokens" not in model_data and "input_tokens" in usage:
                model_data["input_tokens"] = usage["input_tokens"]
            if "output_tokens" not in model_data and "output_tokens" in usage:
                model_data["output_tokens"] = usage["output_tokens"]
            if "total_tokens" not in model_data and "total_tokens" in usage:
                model_data["total_tokens"] = usage["total_tokens"]
            context_payloads["model"] = model_data

        if (
            "had_screenshot_image" in payload
            and "had_screenshot" not in context_payloads["browser"]
            and "had_screenshot" not in payload
        ):
            context_payloads["browser"]["had_screenshot"] = _coerce_optional_bool(
                payload.get("had_screenshot_image")
            )

        timestamp_value = payload.get("timestamp") or payload.get("ts")
        return cls(
            task_id=_validate_task_id(payload),
            step=_validate_step(payload),
            tool_name=str(payload.get("tool_name", "")),
            tool_input=_coerce_dict(payload.get("tool_input")),
            tool_result=_coerce_optional_str(payload.get("tool_result")),
            error=_coerce_optional_str(payload.get("error")),
            duration_ms=_coerce_optional_int(payload.get("duration_ms")),
            timestamp=_coerce_optional_str(timestamp_value),
            schema_version=_resolve_schema_version(payload, scope="AgentStep"),
            model=_build_context("model", context_payloads["model"]),
            tools=_build_context("tools", context_payloads["tools"]),
            reasoning=_build_context("reasoning", context_payloads["reasoning"]),
            browser=_build_context("browser", context_payloads["browser"]),
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
        """Serialize the step to a plain dictionary.

        Returns:
            dict[str, Any]: JSON-friendly data suitable for ``json.dumps()``
            or round-tripping through :meth:`from_dict`.
        """
        return asdict(self)


@dataclass(slots=True)
class TaskOutcome:
    """Outcome metadata emitted once a task finishes.

    Attributes:
        task_id: Identifier of the completed task.
        status: Outcome label such as ``success`` or ``failed``.
        final_answer: Final answer text, if captured.
        total_steps: Total step count for the task.
        total_duration_s: Total task runtime in seconds.
        timestamp: Completion timestamp.
        metadata: Additional completion fields preserved from the trace.
    """

    task_id: str
    status: str
    final_answer: str | None = None
    total_steps: int | None = None
    total_duration_s: float | None = None
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TaskOutcome:
        """Build a ``TaskOutcome`` from a serialized payload.

        Args:
            payload: Raw outcome payload loaded from JSON.

        Returns:
            TaskOutcome: Parsed outcome data with unknown keys stored in
            ``metadata``.
        """
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
        """Serialize the outcome to a plain dictionary.

        Returns:
            dict[str, Any]: JSON-friendly outcome data.
        """
        return asdict(self)


@dataclass(slots=True)
class AgentTask:
    """Task-level container for ordered steps, metadata, and final outcome.

    Attributes:
        task_id: Stable task identifier.
        schema_version: Schema version for the serialized task payload.
        steps: Raw step records associated with the task.
        task_text: Original user instruction, when available.
        task_category: Optional task category label.
        day: Day bucket derived from the source log path or timestamps.
        metadata: Additional task-level metadata recovered while loading.
        outcome: Terminal outcome metadata, if present.

    Example:
        >>> task = AgentTask.from_dict({"task_id": "task-1", "steps": []})
        >>> task.task_id
        'task-1'
    """

    task_id: str
    schema_version: str = SCHEMA_VERSION
    steps: list[AgentStep] = field(default_factory=list)
    task_text: str | None = None
    task_category: str | None = None
    day: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    outcome: TaskOutcome | None = None

    @property
    def sorted_steps(self) -> list[AgentStep]:
        """Return steps ordered by their ``step`` index."""
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
        schema_version: str = SCHEMA_VERSION,
    ) -> AgentTask:
        """Build an ``AgentTask`` from an existing list of ``AgentStep`` objects.

        Args:
            steps: Step objects to associate with the task.
            task_id: Optional explicit task identifier.
            task_text: Optional user-facing task description.
            task_category: Optional task category label.
            metadata: Optional task-level metadata.
            outcome: Optional terminal outcome for the task.
            schema_version: Schema version for the resulting task.

        Returns:
            AgentTask: A normalized task containing the provided steps.
        """
        resolved_task_id = task_id or (steps[0].task_id if steps else "task")
        return cls(
            task_id=resolved_task_id,
            schema_version=_coerce_optional_str(schema_version) or SCHEMA_VERSION,
            steps=list(steps),
            task_text=task_text,
            task_category=task_category,
            metadata=dict(metadata or {}),
            outcome=outcome,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AgentTask:
        """Build an :class:`AgentTask` from serialized task data.

        Args:
            payload: Raw task payload including optional ``steps`` and ``outcome`` data.

        Returns:
            A normalized ``AgentTask`` instance.

        Example:
            >>> task = AgentTask.from_dict({"task_id": "task-1", "steps": []})
            >>> task.schema_version
            '1.0'
        """
        steps_payload = payload.get("steps")
        steps = (
            [
                step if isinstance(step, AgentStep) else AgentStep.from_dict(_coerce_dict(step))
                for step in steps_payload
            ]
            if isinstance(steps_payload, list)
            else []
        )

        outcome_payload = payload.get("outcome")
        if isinstance(outcome_payload, TaskOutcome):
            outcome = outcome_payload
        elif isinstance(outcome_payload, dict):
            outcome = TaskOutcome.from_dict(outcome_payload)
        else:
            outcome = None

        return cls(
            task_id=_validate_task_id(payload),
            schema_version=_resolve_schema_version(payload, scope="AgentTask"),
            steps=steps,
            task_text=_coerce_optional_str(payload.get("task_text")),
            task_category=_coerce_optional_str(payload.get("task_category")),
            day=_coerce_optional_str(payload.get("day")),
            metadata=_coerce_dict(payload.get("metadata")),
            outcome=outcome,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the task with steps sorted by ``step``.

        Returns:
            dict[str, Any]: JSON-friendly task data including nested outcome
            and step payloads.
        """
        return {
            "task_id": self.task_id,
            "schema_version": self.schema_version,
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
        "schema_version": {"type": "string"},
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
        "schema_version": {"type": "string"},
        "task_text": {"type": ["string", "null"]},
        "task_category": {"type": ["string", "null"]},
        "day": {"type": ["string", "null"]},
        "metadata": {"type": "object"},
        "outcome": {"anyOf": [{"type": "null"}, TASK_OUTCOME_JSON_SCHEMA]},
        "steps": {"type": "array", "items": AGENT_STEP_JSON_SCHEMA},
    },
    "additionalProperties": True,
}
