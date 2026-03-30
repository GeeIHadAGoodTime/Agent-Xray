from __future__ import annotations

import pytest

from agent_xray.schema import AgentStep, TaskOutcome, _coerce_optional_bool


def test_agent_step_from_dict_with_minimal_data() -> None:
    step = AgentStep.from_dict(
        {
            "task_id": "task-1",
            "step": 1,
            "tool_name": "respond",
            "tool_input": {},
        }
    )

    assert step.task_id == "task-1"
    assert step.step == 1
    assert step.tool_name == "respond"
    assert step.tool_input == {}
    assert step.model is None
    assert step.tools is None
    assert step.reasoning is None
    assert step.browser is None


def test_agent_step_from_dict_with_full_contexts() -> None:
    step = AgentStep.from_dict(
        {
            "task_id": "task-1",
            "step": "2",
            "tool_name": "browser_click",
            "tool_input": {"ref": "checkout"},
            "tool_result": "opened checkout",
            "error": None,
            "duration_ms": "125",
            "timestamp": "2026-03-30T15:00:00Z",
            "model": {
                "model_name": "gpt-5.4",
                "temperature": "0.2",
                "tool_choice": "auto",
                "context_window": "128000",
                "context_usage_pct": "67.5",
                "compaction_count": "2",
                "input_tokens": "321",
                "output_tokens": "123",
                "total_tokens": "444",
                "cache_read_tokens": "10",
                "cache_creation_tokens": "5",
                "cost_usd": "0.14",
                "prompt_variant": "commerce-v2",
            },
            "tools": {
                "tools_available": ["browser_click", "respond"],
                "system_prompt_hash": "abc123",
                "message_count": "6",
                "rejected_tools": ["web_search"],
                "focused_set": "checkout",
                "tools_available_count": "2",
                "conversation_turn_count": "3",
            },
            "reasoning": {
                "llm_reasoning": "Checkout button is visible.",
                "llm_decision": "click it",
                "correction_messages": ["avoid duplicate clicks"],
                "spin_intervention": "redirect",
                "approval_path": "auto",
            },
            "browser": {
                "page_url": "https://shop.example.test/checkout",
                "had_screenshot": "true",
                "snapshot_compressed": "false",
                "had_screenshot_image": 1,
                "snapshot_pre_compress_len": "2048",
                "browser_tiers_used": ["dom", "vision"],
            },
            "llm_usage": {
                "input_tokens": 500,
                "output_tokens": 200,
                "total_tokens": 700,
                "cache_read_tokens": 50,
                "cache_creation_tokens": 25,
            },
        }
    )

    assert step.step == 2
    assert step.duration_ms == 125
    assert step.model is not None and step.model.model_name == "gpt-5.4"
    assert step.input_tokens == 321
    assert step.output_tokens == 123
    assert step.tools is not None and step.tools.rejected_tools == ["web_search"]
    assert step.reasoning is not None and step.reasoning.llm_decision == "click it"
    assert step.browser is not None and step.browser.browser_tiers_used == ["dom", "vision"]
    assert step.browser.had_screenshot is True
    assert step.browser.snapshot_compressed is False


def test_task_outcome_from_dict_round_trip_preserves_metadata() -> None:
    original = TaskOutcome.from_dict(
        {
            "task_id": "task-1",
            "outcome": "success",
            "final_answer": "done",
            "total_steps": "4",
            "total_duration_s": "2.5",
            "ts": "2026-03-30T15:10:00Z",
            "custom_note": "kept",
        }
    )

    round_tripped = TaskOutcome.from_dict(original.to_dict())

    assert round_tripped.task_id == "task-1"
    assert round_tripped.status == "success"
    assert round_tripped.total_steps == 4
    assert round_tripped.total_duration_s == pytest.approx(2.5)
    assert round_tripped.metadata == {"custom_note": "kept"}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        (None, None),
        ("true", True),
        ("false", False),
        ("maybe", None),
        (1, True),
        (0, False),
        ("", None),
    ],
)
def test_coerce_optional_bool_edge_cases(value: object, expected: bool | None) -> None:
    assert _coerce_optional_bool(value) is expected


def test_from_dict_empty_payload_returns_valid_minimal_step() -> None:
    step = AgentStep.from_dict({})

    assert step.task_id == ""
    assert step.step == 0
    assert step.tool_name == ""
    assert step.tool_input == {}
    assert step.extensions == {}


def test_unknown_fields_are_preserved_in_extensions() -> None:
    step = AgentStep.from_dict(
        {
            "task_id": "task-1",
            "step": 1,
            "tool_name": "respond",
            "tool_input": {},
            "custom_flag": True,
            "model": {"model_name": "gpt-5", "quantization": "int8"},
            "tools": {"tools_available": ["respond"], "router_hint": "focused"},
            "reasoning": {"llm_reasoning": "done", "debug_tag": "trace-1"},
            "browser": {"page_url": "https://example.test", "viewport": "mobile"},
        }
    )

    assert step.extensions["custom_flag"] is True
    assert step.extensions["model"] == {"quantization": "int8"}
    assert step.extensions["tools"] == {"router_hint": "focused"}
    assert step.extensions["reasoning"] == {"debug_tag": "trace-1"}
    assert step.extensions["browser"] == {"viewport": "mobile"}
