from __future__ import annotations

import math
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from agent_xray.schema import (
    AgentStep,
    _coerce_list_of_str,
    _coerce_optional_bool,
    _coerce_optional_float,
    _coerce_optional_int,
)

json_scalar = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(max_size=32),
)
json_value = st.recursive(
    json_scalar,
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(st.text(max_size=16), children, max_size=4),
    ),
    max_leaves=20,
)


def _jsonish_dicts() -> st.SearchStrategy[dict[str, Any]]:
    return st.dictionaries(st.text(max_size=16), json_value, max_size=20)


def _optional_strs() -> st.SearchStrategy[str | None]:
    return st.one_of(st.none(), st.text(max_size=24))


def _optional_ints() -> st.SearchStrategy[int | None]:
    return st.one_of(st.none(), st.integers(min_value=0, max_value=500_000))


def _optional_floats() -> st.SearchStrategy[float | None]:
    return st.one_of(
        st.none(),
        st.floats(min_value=0, max_value=100_000, allow_nan=False, allow_infinity=False),
    )


def _optional_bool_lists() -> st.SearchStrategy[list[str] | None]:
    return st.one_of(st.none(), st.lists(st.text(max_size=16), max_size=5))


def _step_payloads() -> st.SearchStrategy[dict[str, Any]]:
    model_strategy = st.fixed_dictionaries(
        {
            "model_name": _optional_strs(),
            "temperature": _optional_floats(),
            "tool_choice": _optional_strs(),
            "context_window": _optional_ints(),
            "context_usage_pct": _optional_floats(),
            "compaction_count": _optional_ints(),
            "input_tokens": _optional_ints(),
            "output_tokens": _optional_ints(),
            "cost_usd": _optional_floats(),
            "compaction_method": _optional_strs(),
            "compaction_messages_before": _optional_ints(),
            "compaction_messages_after": _optional_ints(),
            "compaction_summary_preview": _optional_strs(),
            "trimmed_messages": _optional_ints(),
            "fifo_evicted_messages": _optional_ints(),
            "screenshots_evicted": _optional_ints(),
            "prompt_variant": _optional_strs(),
            "prompt_variant_full": _optional_strs(),
        }
    )
    tools_strategy = st.fixed_dictionaries(
        {
            "tools_available": _optional_bool_lists(),
            "system_prompt_hash": _optional_strs(),
            "message_count": _optional_ints(),
            "rejected_tools": _optional_bool_lists(),
            "focused_set": _optional_strs(),
            "tools_available_count": _optional_ints(),
            "conversation_turn_count": _optional_ints(),
        }
    )
    reasoning_strategy = st.fixed_dictionaries(
        {
            "llm_reasoning": _optional_strs(),
            "correction_messages": _optional_bool_lists(),
            "spin_intervention": _optional_strs(),
            "error_registry_context": _optional_strs(),
            "continuation_nudge": _optional_strs(),
            "force_termination": _optional_strs(),
            "hard_loop_breaker": _optional_strs(),
            "consecutive_failure_warning": _optional_strs(),
            "approval_path": _optional_strs(),
        }
    )
    browser_strategy = st.fixed_dictionaries(
        {
            "page_url": _optional_strs(),
            "had_screenshot": st.one_of(st.none(), st.booleans()),
            "snapshot_compressed": st.one_of(st.none(), st.booleans()),
            "had_screenshot_image": st.one_of(st.none(), st.booleans()),
            "snapshot_pre_compress_len": _optional_ints(),
        }
    )
    return st.fixed_dictionaries(
        {
            "task_id": st.text(min_size=1, max_size=24).filter(lambda value: bool(value.strip())),
            "step": st.integers(min_value=0, max_value=10_000),
            "tool_name": st.text(max_size=24),
            "tool_input": st.dictionaries(st.text(max_size=16), json_value, max_size=8),
            "tool_result": _optional_strs(),
            "error": _optional_strs(),
            "duration_ms": _optional_ints(),
            "timestamp": _optional_strs(),
            "model": st.one_of(st.none(), model_strategy),
            "tools": st.one_of(st.none(), tools_strategy),
            "reasoning": st.one_of(st.none(), reasoning_strategy),
            "browser": st.one_of(st.none(), browser_strategy),
            "extensions": st.dictionaries(st.text(max_size=16), json_value, max_size=8),
        }
    )


@settings(deadline=None, max_examples=150)
@given(_jsonish_dicts())
def test_agent_step_from_dict_accepts_arbitrary_dict_inputs(payload: dict[str, Any]) -> None:
    step = AgentStep.from_dict(payload)
    assert isinstance(step.task_id, str)
    assert isinstance(step.step, int)
    assert isinstance(step.tool_name, str)
    assert isinstance(step.tool_input, dict)
    assert isinstance(step.extensions, dict)


@settings(deadline=None, max_examples=150)
@given(json_value)
def test_coerce_optional_int_matches_contract(value: Any) -> None:
    coerced = _coerce_optional_int(value)
    if value is None or value == "":
        assert coerced is None
    else:
        try:
            assert coerced == int(value)
        except (TypeError, ValueError):
            assert coerced is None


@settings(deadline=None, max_examples=150)
@given(json_value)
def test_coerce_optional_float_matches_contract(value: Any) -> None:
    coerced = _coerce_optional_float(value)
    if value is None or value == "":
        assert coerced is None
    else:
        try:
            expected = float(value)
            if math.isnan(expected):
                assert coerced is not None and math.isnan(coerced)
            else:
                assert coerced == expected
        except (TypeError, ValueError):
            assert coerced is None


@settings(deadline=None, max_examples=150)
@given(
    st.one_of(
        st.none(),
        st.booleans(),
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.text(max_size=16),
        st.lists(st.integers(), max_size=4),
    )
)
def test_coerce_optional_bool_matches_contract(value: Any) -> None:
    coerced = _coerce_optional_bool(value)
    if value is None or value == "":
        assert coerced is None
        return
    if isinstance(value, bool):
        assert coerced is value
        return
    if isinstance(value, int):
        if value in {0, 1}:
            assert coerced is bool(value)
        else:
            assert coerced is None
        return
    if isinstance(value, float):
        if value in {0.0, 1.0}:
            assert coerced is bool(int(value))
        else:
            assert coerced is None
        return
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1"}:
            assert coerced is True
            return
        if lowered in {"false", "0"}:
            assert coerced is False
            return
        assert coerced is None
        return
    assert coerced is None


@settings(deadline=None, max_examples=150)
@given(
    st.one_of(
        st.none(),
        st.booleans(),
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.text(max_size=16),
        st.lists(json_scalar, max_size=4),
    )
)
def test_coerce_list_of_str_matches_contract(value: Any) -> None:
    coerced = _coerce_list_of_str(value)
    if value is None:
        assert coerced is None
    elif isinstance(value, list):
        assert coerced == [str(item) for item in value]
    else:
        assert coerced == [str(value)]


@settings(deadline=None, max_examples=100)
@given(_step_payloads())
def test_agent_step_roundtrip_preserves_data(payload: dict[str, Any]) -> None:
    original = AgentStep.from_dict(payload)
    roundtrip = AgentStep.from_dict(original.to_dict())
    assert roundtrip.to_dict() == original.to_dict()


@settings(deadline=None, max_examples=100)
@given(_step_payloads())
def test_agent_step_roundtrip_preserves_extensions(payload: dict[str, Any]) -> None:
    step = AgentStep.from_dict(payload)
    assert AgentStep.from_dict(step.to_dict()).extensions == step.extensions
