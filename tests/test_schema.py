from __future__ import annotations

from agent_xray.schema import AGENT_STEP_JSON_SCHEMA, AgentStep, AgentTask, TaskOutcome


def test_agent_step_round_trip() -> None:
    step = AgentStep.from_dict(
        {
            "task_id": "task-1",
            "step": 1,
            "tool_name": "browser_navigate",
            "tool_input": {"url": "https://example.test"},
            "tool_result": "ok",
            "tools_available": ["browser_navigate"],
        }
    )
    data = step.to_dict()
    assert data["task_id"] == "task-1"
    assert data["tool_input"]["url"] == "https://example.test"
    assert step.tools_available == ["browser_navigate"]


def test_agent_step_schema_required_fields() -> None:
    assert set(AGENT_STEP_JSON_SCHEMA["required"]) == {"task_id", "step", "tool_name", "tool_input"}


def test_agent_step_model_cost_round_trip() -> None:
    step = AgentStep.from_dict(
        {
            "task_id": "task-1",
            "step": 2,
            "tool_name": "respond",
            "tool_input": {},
            "model": {
                "model_name": "gpt-4o",
                "input_tokens": 100,
                "output_tokens": 50,
                "cost_usd": 0.0125,
            },
        }
    )
    assert step.model is not None
    assert step.model.model_name == "gpt-4o"
    assert step.cost_usd == 0.0125


def test_agent_task_from_steps_uses_step_task_id() -> None:
    task = AgentTask.from_steps([AgentStep("task-42", 1, "search", {"q": "hello"})])
    assert task.task_id == "task-42"
    assert len(task.steps) == 1


def test_from_dict_empty_dict_returns_defaults() -> None:
    step = AgentStep.from_dict({})
    assert step.task_id == ""
    assert step.step == 0
    assert step.tool_name == ""
    assert step.tool_input == {}


def test_from_dict_string_step_coercion() -> None:
    step = AgentStep.from_dict(
        {"task_id": "task-1", "step": "3", "tool_name": "respond", "tool_input": {}}
    )
    assert step.step == 3


def test_from_dict_extra_fields_go_to_extensions() -> None:
    step = AgentStep.from_dict(
        {
            "task_id": "task-1",
            "step": 1,
            "tool_name": "respond",
            "tool_input": {},
            "custom_flag": True,
        }
    )
    assert step.extensions["custom_flag"] is True


def test_from_dict_nested_model_context() -> None:
    step = AgentStep.from_dict(
        {
            "task_id": "task-1",
            "step": 1,
            "tool_name": "respond",
            "tool_input": {},
            "model": {"model_name": "gpt-5", "cost_usd": 0.25},
            "tools": {"tools_available": ["respond"]},
            "reasoning": {"llm_reasoning": "Use the response tool."},
            "browser": {"page_url": "https://example.test"},
        }
    )
    assert step.model_name == "gpt-5"
    assert step.cost_usd == 0.25
    assert step.tools_available_names == ["respond"]
    assert step.llm_reasoning == "Use the response tool."
    assert step.page_url == "https://example.test"


def test_from_dict_flat_and_nested_equivalent() -> None:
    flat = AgentStep.from_dict(
        {
            "task_id": "task-1",
            "step": 1,
            "tool_name": "respond",
            "tool_input": {},
            "model_name": "gpt-5",
            "cost_usd": 0.5,
            "tools_available_names": ["respond"],
            "llm_reasoning": "Think.",
            "page_url": "https://example.test",
        }
    )
    nested = AgentStep.from_dict(
        {
            "task_id": "task-1",
            "step": 1,
            "tool_name": "respond",
            "tool_input": {},
            "model": {"model_name": "gpt-5", "cost_usd": 0.5},
            "tools": {"tools_available": ["respond"]},
            "reasoning": {"llm_reasoning": "Think."},
            "browser": {"page_url": "https://example.test"},
        }
    )
    assert flat.model_name == nested.model_name
    assert flat.cost_usd == nested.cost_usd
    assert flat.tools_available_names == nested.tools_available_names
    assert flat.llm_reasoning == nested.llm_reasoning
    assert flat.page_url == nested.page_url


def test_to_dict_round_trip_with_extensions(sample_step: AgentStep) -> None:
    cloned = AgentStep.from_dict(sample_step.to_dict())
    assert cloned.extensions == sample_step.extensions
    assert cloned.tools_available_names == sample_step.tools_available_names
    assert cloned.cost_usd == sample_step.cost_usd


def test_tools_available_names_property(sample_step: AgentStep) -> None:
    assert sample_step.tools_available_names == sample_step.tools_available


def test_cost_usd_property(sample_step: AgentStep) -> None:
    assert sample_step.cost_usd == 0.014


def test_task_outcome_from_dict_merges_nested_metadata_without_double_nesting() -> None:
    outcome = TaskOutcome.from_dict(
        {
            "task_id": "task-1",
            "status": "success",
            "metadata": {"timed_out": True, "source": "nested"},
            "final_context_usage_pct": 92.0,
        }
    )
    assert outcome.metadata == {
        "timed_out": True,
        "source": "nested",
        "final_context_usage_pct": 92.0,
    }
    assert outcome.to_dict()["metadata"]["source"] == "nested"


def test_task_outcome_round_trip_preserves_metadata_structure() -> None:
    original = TaskOutcome(
        task_id="task-1",
        status="success",
        metadata={"source": "nested", "details": {"attempt": 2}},
    )

    cloned = TaskOutcome.from_dict(original.to_dict())

    assert cloned.metadata == original.metadata


def test_task_outcome_from_dict_flattens_double_nested_metadata() -> None:
    outcome = TaskOutcome.from_dict(
        {
            "task_id": "task-1",
            "status": "success",
            "metadata": {
                "metadata": {"source": "nested"},
                "details": {"attempt": 2},
            },
        }
    )

    assert outcome.metadata == {
        "source": "nested",
        "details": {"attempt": 2},
    }


# ── New context fields (decision surface expansion) ──────────────────


def test_from_dict_compaction_fields_flat() -> None:
    step = AgentStep.from_dict(
        {
            "task_id": "t",
            "step": 1,
            "tool_name": "x",
            "tool_input": {},
            "compaction_method": "summarize",
            "compaction_messages_before": 20,
            "compaction_messages_after": 10,
            "compaction_summary_preview": "Earlier context...",
            "trimmed_messages": 5,
            "fifo_evicted_messages": 2,
            "screenshots_evicted": 1,
            "prompt_variant": "commerce_v3",
        }
    )
    assert step.model is not None
    assert step.model.compaction_method == "summarize"
    assert step.model.compaction_messages_before == 20
    assert step.model.compaction_messages_after == 10
    assert step.model.compaction_summary_preview == "Earlier context..."
    assert step.model.trimmed_messages == 5
    assert step.model.fifo_evicted_messages == 2
    assert step.model.screenshots_evicted == 1
    assert step.model.prompt_variant == "commerce_v3"


def test_from_dict_tool_context_new_fields() -> None:
    step = AgentStep.from_dict(
        {
            "task_id": "t",
            "step": 1,
            "tool_name": "x",
            "tool_input": {},
            "rejected_tools": ["web_search", "respond"],
            "focused_set": "browser_payment",
            "tools_available_count": 5,
            "conversation_turn_count": 3,
        }
    )
    assert step.tools is not None
    assert step.tools.rejected_tools == ["web_search", "respond"]
    assert step.tools.focused_set == "browser_payment"
    assert step.tools.tools_available_count == 5
    assert step.tools.conversation_turn_count == 3
    assert step.rejected_tools == ["web_search", "respond"]
    assert step.focused_set == "browser_payment"


def test_from_dict_reasoning_context_new_fields() -> None:
    step = AgentStep.from_dict(
        {
            "task_id": "t",
            "step": 1,
            "tool_name": "x",
            "tool_input": {},
            "error_registry_context": "timeout on /cart",
            "continuation_nudge": "proceed_to_checkout",
            "force_termination": "max_iter",
            "hard_loop_breaker": "tool_repeat_3x",
            "consecutive_failure_warning": "3 failures",
            "approval_path": "auto_approved",
        }
    )
    assert step.reasoning is not None
    assert step.reasoning.error_registry_context == "timeout on /cart"
    assert step.reasoning.continuation_nudge == "proceed_to_checkout"
    assert step.reasoning.force_termination == "max_iter"
    assert step.reasoning.hard_loop_breaker == "tool_repeat_3x"
    assert step.reasoning.consecutive_failure_warning == "3 failures"
    assert step.reasoning.approval_path == "auto_approved"
    assert step.approval_path == "auto_approved"


def test_from_dict_browser_context_new_fields() -> None:
    step = AgentStep.from_dict(
        {
            "task_id": "t",
            "step": 1,
            "tool_name": "x",
            "tool_input": {},
            "page_url": "https://example.test",
            "had_screenshot_image": True,
            "snapshot_pre_compress_len": 9500,
        }
    )
    assert step.browser is not None
    assert step.browser.had_screenshot_image is True
    assert step.browser.snapshot_pre_compress_len == 9500


def test_from_dict_all_new_fields_nested() -> None:
    step = AgentStep.from_dict(
        {
            "task_id": "t",
            "step": 1,
            "tool_name": "x",
            "tool_input": {},
            "model": {
                "compaction_method": "truncate",
                "trimmed_messages": 3,
                "prompt_variant": "v2",
                "prompt_variant_full": "research_deep_v2",
            },
            "tools": {
                "rejected_tools": ["a"],
                "focused_set": "research",
                "conversation_turn_count": 7,
            },
            "reasoning": {
                "error_registry_context": "404 errors",
                "hard_loop_breaker": "url_stuck",
            },
            "browser": {
                "snapshot_pre_compress_len": 3000,
                "had_screenshot_image": False,
            },
        }
    )
    assert step.model.compaction_method == "truncate"
    assert step.model.prompt_variant_full == "research_deep_v2"
    assert step.tools.rejected_tools == ["a"]
    assert step.reasoning.hard_loop_breaker == "url_stuck"
    assert step.browser.snapshot_pre_compress_len == 3000


def test_json_schemas_include_new_fields() -> None:
    from agent_xray.schema import (
        BROWSER_CONTEXT_JSON_SCHEMA,
        MODEL_CONTEXT_JSON_SCHEMA,
        REASONING_CONTEXT_JSON_SCHEMA,
        TOOL_CONTEXT_JSON_SCHEMA,
    )

    model_props = MODEL_CONTEXT_JSON_SCHEMA["properties"]
    assert "compaction_method" in model_props
    assert "trimmed_messages" in model_props
    assert "prompt_variant" in model_props
    assert "screenshots_evicted" in model_props

    tool_props = TOOL_CONTEXT_JSON_SCHEMA["properties"]
    assert "rejected_tools" in tool_props
    assert "focused_set" in tool_props
    assert "conversation_turn_count" in tool_props

    reasoning_props = REASONING_CONTEXT_JSON_SCHEMA["properties"]
    assert "error_registry_context" in reasoning_props
    assert "approval_path" in reasoning_props
    assert "hard_loop_breaker" in reasoning_props

    browser_props = BROWSER_CONTEXT_JSON_SCHEMA["properties"]
    assert "snapshot_pre_compress_len" in browser_props
    assert "had_screenshot_image" in browser_props
