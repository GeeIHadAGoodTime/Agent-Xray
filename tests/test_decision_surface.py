"""Tests for the full decision surface — verifying all 23 information surfaces
are correctly propagated from schema through to surface output."""

from __future__ import annotations

from agent_xray.schema import (
    AgentStep,
    AgentTask,
    BrowserContext,
    ModelContext,
    ReasoningContext,
    TaskOutcome,
    ToolContext,
)
from agent_xray.surface import (
    format_surface_text,
    reasoning_for_task,
    surface_for_task,
)


def _full_surface_task() -> AgentTask:
    """Build a task that exercises every decision surface field."""
    return AgentTask(
        task_id="surface-test",
        task_text="Buy headphones on shop.example.test",
        task_category="commerce",
        metadata={
            "system_prompt_text": "You are a shopping assistant.",
            "system_context_components": {
                "browser": "enabled",
                "payment": "stripe_checkout",
                "persona": "helpful_shopper",
            },
            "prior_conversation_summary": "User previously browsed electronics.",
        },
        steps=[
            AgentStep(
                task_id="surface-test",
                step=1,
                tool_name="browser_navigate",
                tool_input={"url": "https://shop.example.test"},
                tool_result="Homepage loaded with product listings.",
                duration_ms=850,
                timestamp="2026-03-26T12:00:00Z",
                model=ModelContext(
                    model_name="gpt-4.1-nano",
                    temperature=0.1,
                    tool_choice="auto",
                    context_window=128000,
                    context_usage_pct=15.0,
                    compaction_count=0,
                    input_tokens=200,
                    output_tokens=50,
                    cost_usd=0.002,
                    compaction_method=None,
                    compaction_messages_before=None,
                    compaction_messages_after=None,
                    compaction_summary_preview=None,
                    trimmed_messages=None,
                    fifo_evicted_messages=None,
                    screenshots_evicted=None,
                    prompt_variant="commerce_v3",
                    prompt_variant_full=None,
                ),
                tools=ToolContext(
                    tools_available=["browser_navigate", "browser_click", "browser_fill_ref", "browser_snapshot"],
                    system_prompt_hash="abc123",
                    message_count=2,
                    rejected_tools=["web_search", "respond"],
                    focused_set="browser_commerce",
                    tools_available_count=4,
                    conversation_turn_count=1,
                ),
                reasoning=ReasoningContext(
                    llm_reasoning="I need to navigate to the store first.",
                    correction_messages=None,
                    spin_intervention=None,
                    error_registry_context=None,
                    continuation_nudge=None,
                    force_termination=None,
                    hard_loop_breaker=None,
                    consecutive_failure_warning=None,
                    approval_path=None,
                ),
                browser=BrowserContext(
                    page_url="https://shop.example.test/",
                    had_screenshot=True,
                    snapshot_compressed=False,
                    had_screenshot_image=True,
                    snapshot_pre_compress_len=4500,
                ),
            ),
            AgentStep(
                task_id="surface-test",
                step=2,
                tool_name="browser_click",
                tool_input={"ref": "add-to-cart"},
                tool_result="Added to cart. Subtotal: $99.",
                duration_ms=300,
                timestamp="2026-03-26T12:01:00Z",
                model=ModelContext(
                    model_name="gpt-4.1-nano",
                    temperature=0.1,
                    tool_choice="auto",
                    context_window=128000,
                    context_usage_pct=45.0,
                    compaction_count=1,
                    input_tokens=800,
                    output_tokens=60,
                    cost_usd=0.008,
                    compaction_method="summarize",
                    compaction_messages_before=12,
                    compaction_messages_after=6,
                    compaction_summary_preview="User navigated to store...",
                    trimmed_messages=2,
                    fifo_evicted_messages=1,
                    screenshots_evicted=3,
                    prompt_variant="commerce_v3",
                    prompt_variant_full=None,
                ),
                tools=ToolContext(
                    tools_available=["browser_click", "browser_fill_ref", "browser_snapshot"],
                    system_prompt_hash="abc123",
                    message_count=8,
                    rejected_tools=["web_search"],
                    focused_set="browser_commerce",
                    tools_available_count=3,
                    conversation_turn_count=2,
                ),
                reasoning=ReasoningContext(
                    llm_reasoning="Product found, adding to cart.",
                    correction_messages=["Don't click buy before checking price."],
                    spin_intervention="reassess_after_retry",
                    error_registry_context="Prior error: timeout on /products page",
                    continuation_nudge="continue_checkout_flow",
                    force_termination=None,
                    hard_loop_breaker=None,
                    consecutive_failure_warning="2 consecutive failures on click",
                    approval_path="auto_approved:browser_click",
                ),
                browser=BrowserContext(
                    page_url="https://shop.example.test/cart",
                    had_screenshot=True,
                    snapshot_compressed=True,
                    had_screenshot_image=False,
                    snapshot_pre_compress_len=8200,
                ),
            ),
        ],
        outcome=TaskOutcome(
            task_id="surface-test",
            status="success",
            total_steps=2,
            total_duration_s=1.15,
            final_answer="Added to cart.",
        ),
    )


# ── Core surface structure ──────────────────────────────────────────


def test_surface_task_level_fields():
    task = _full_surface_task()
    surface = surface_for_task(task)
    assert surface["task_id"] == "surface-test"
    assert surface["task_text"] == "Buy headphones on shop.example.test"
    assert surface["task_category"] == "commerce"
    assert surface["prompt_text"] == "You are a shopping assistant."
    assert surface["system_context_components"]["browser"] == "enabled"
    assert surface["system_context_components"]["payment"] == "stripe_checkout"
    assert surface["prior_conversation_summary"] == "User previously browsed electronics."
    assert surface["outcome"]["status"] == "success"


def test_surface_step_count():
    task = _full_surface_task()
    surface = surface_for_task(task)
    assert len(surface["steps"]) == 2


# ── LLM decision context ────────────────────────────────────────────


def test_surface_model_context():
    task = _full_surface_task()
    surface = surface_for_task(task)
    step1 = surface["steps"][0]
    assert step1["model_name"] == "gpt-4.1-nano"
    assert step1["temperature"] == 0.1
    assert step1["tool_choice"] == "auto"
    assert step1["prompt_variant"] == "commerce_v3"
    assert step1["model"]["input_tokens"] == 200
    assert step1["model"]["output_tokens"] == 50
    assert step1["model"]["cost_usd"] == 0.002


# ── Tool availability (THE decision surface) ────────────────────────


def test_surface_tools_available():
    task = _full_surface_task()
    surface = surface_for_task(task)
    step1 = surface["steps"][0]
    assert step1["tools_available_names"] == [
        "browser_navigate", "browser_click", "browser_fill_ref", "browser_snapshot"
    ]
    assert step1["tools_available_count"] == 4


def test_surface_rejected_tools():
    task = _full_surface_task()
    surface = surface_for_task(task)
    step1 = surface["steps"][0]
    assert step1["rejected_tools"] == ["web_search", "respond"]


def test_surface_focused_set():
    task = _full_surface_task()
    surface = surface_for_task(task)
    step1 = surface["steps"][0]
    assert step1["focused_set"] == "browser_commerce"


def test_surface_tools_available_count_from_metadata():
    """When tools_available_count is set in metadata, it should be used directly."""
    task = _full_surface_task()
    surface = surface_for_task(task)
    step2 = surface["steps"][1]
    assert step2["tools_available_count"] == 3


def test_surface_tools_available_count_fallback():
    """When tools_available_count is None, fall back to len(tool_names)."""
    task = AgentTask(
        task_id="t",
        steps=[
            AgentStep(
                "t", 1, "click", {},
                tools=ToolContext(
                    tools_available=["a", "b"],
                    tools_available_count=None,
                ),
            ),
        ],
    )
    surface = surface_for_task(task)
    assert surface["steps"][0]["tools_available_count"] == 2


# ── Context pressure ────────────────────────────────────────────────


def test_surface_context_pressure():
    task = _full_surface_task()
    surface = surface_for_task(task)
    step1 = surface["steps"][0]
    assert step1["context_usage_pct"] == 15.0
    assert step1["context_window"] == 128000
    assert step1["message_count"] == 2
    assert step1["conversation_turn_count"] == 1
    assert step1["system_prompt_hash"] == "abc123"


# ── Compaction & trimming ────────────────────────────────────────────


def test_surface_compaction_fields():
    task = _full_surface_task()
    surface = surface_for_task(task)
    step2 = surface["steps"][1]
    assert step2["compaction_count"] == 1
    assert step2["compaction_method"] == "summarize"
    assert step2["compaction_messages_before"] == 12
    assert step2["compaction_messages_after"] == 6
    assert step2["compaction_summary_preview"] == "User navigated to store..."
    assert step2["trimmed_messages"] == 2
    assert step2["fifo_evicted_messages"] == 1
    assert step2["screenshots_evicted"] == 3


def test_surface_no_compaction_when_none():
    task = _full_surface_task()
    surface = surface_for_task(task)
    step1 = surface["steps"][0]
    assert step1["compaction_method"] is None
    assert step1["trimmed_messages"] is None


# ── Dynamic injections ───────────────────────────────────────────────


def test_surface_dynamic_injections_step2():
    task = _full_surface_task()
    surface = surface_for_task(task)
    step2 = surface["steps"][1]
    assert step2["correction_messages"] == ["Don't click buy before checking price."]
    assert step2["spin_intervention"] == "reassess_after_retry"
    assert step2["error_registry_context"] == "Prior error: timeout on /products page"
    assert step2["continuation_nudge"] == "continue_checkout_flow"
    assert step2["consecutive_failure_warning"] == "2 consecutive failures on click"
    assert step2["approval_path"] == "auto_approved:browser_click"


def test_surface_no_injections_step1():
    task = _full_surface_task()
    surface = surface_for_task(task)
    step1 = surface["steps"][0]
    assert step1["correction_messages"] == []
    assert step1["spin_intervention"] is None
    assert step1["error_registry_context"] is None
    assert step1["force_termination"] is None
    assert step1["hard_loop_breaker"] is None


def test_surface_force_termination():
    task = AgentTask(
        task_id="t",
        steps=[
            AgentStep(
                "t", 1, "respond", {},
                reasoning=ReasoningContext(
                    force_termination="max_iterations_exceeded",
                    hard_loop_breaker="tool_repeat_3x",
                ),
            ),
        ],
    )
    surface = surface_for_task(task)
    step = surface["steps"][0]
    assert step["force_termination"] == "max_iterations_exceeded"
    assert step["hard_loop_breaker"] == "tool_repeat_3x"


# ── Browser state ────────────────────────────────────────────────────


def test_surface_browser_state():
    task = _full_surface_task()
    surface = surface_for_task(task)
    step1 = surface["steps"][0]
    assert step1["page_url"] == "https://shop.example.test/"
    assert step1["had_screenshot"] is True
    assert step1["had_screenshot_image"] is True
    assert step1["snapshot_compressed"] is False
    assert step1["snapshot_pre_compress_len"] == 4500


def test_surface_browser_compressed():
    task = _full_surface_task()
    surface = surface_for_task(task)
    step2 = surface["steps"][1]
    assert step2["snapshot_compressed"] is True
    assert step2["snapshot_pre_compress_len"] == 8200


# ── Conversation history ─────────────────────────────────────────────


def test_surface_conversation_history_accumulates():
    task = _full_surface_task()
    surface = surface_for_task(task)
    hist1 = surface["steps"][0]["conversation_history"]
    assert hist1[0] == {"role": "user", "content": "Buy headphones on shop.example.test"}
    assert len(hist1) == 1

    hist2 = surface["steps"][1]["conversation_history"]
    assert len(hist2) >= 3
    assert hist2[0]["role"] == "user"
    assert hist2[1]["role"] == "assistant_reasoning"
    assert hist2[2]["role"] == "tool_call"


# ── Prompt extraction paths ─────────────────────────────────────────


def test_prompt_from_task_metadata():
    task = _full_surface_task()
    surface = surface_for_task(task)
    assert surface["prompt_text"] == "You are a shopping assistant."


def test_prompt_from_step_extensions():
    task = AgentTask(
        task_id="t",
        steps=[
            AgentStep(
                "t", 1, "click", {},
                extensions={"system_prompt_text": "You are a browser agent."},
            ),
        ],
    )
    surface = surface_for_task(task)
    assert surface["prompt_text"] == "You are a browser agent."


def test_prompt_metadata_takes_priority_over_extensions():
    task = AgentTask(
        task_id="t",
        metadata={"system_prompt_text": "From metadata."},
        steps=[
            AgentStep(
                "t", 1, "click", {},
                extensions={"system_prompt_text": "From extensions."},
            ),
        ],
    )
    surface = surface_for_task(task)
    assert surface["prompt_text"] == "From metadata."


def test_system_components_from_metadata():
    task = _full_surface_task()
    surface = surface_for_task(task)
    assert surface["system_context_components"]["persona"] == "helpful_shopper"


def test_system_components_from_step_extensions():
    task = AgentTask(
        task_id="t",
        steps=[
            AgentStep(
                "t", 1, "click", {},
                extensions={"system_context_components": {"mode": "research"}},
            ),
        ],
    )
    surface = surface_for_task(task)
    assert surface["system_context_components"]["mode"] == "research"


# ── from_dict round-trip ─────────────────────────────────────────────


def test_from_dict_preserves_all_new_fields():
    """Flat payload with all 23 surfaces should survive from_dict -> surface."""
    payload = {
        "task_id": "rt",
        "step": 1,
        "tool_name": "browser_click",
        "tool_input": {"ref": "btn"},
        "tool_result": "clicked",
        "model_name": "gpt-4.1-nano",
        "temperature": 0.2,
        "tool_choice": "required",
        "context_window": 64000,
        "context_usage_pct": 72.5,
        "compaction_count": 2,
        "compaction_method": "sliding_window",
        "compaction_messages_before": 20,
        "compaction_messages_after": 10,
        "compaction_summary_preview": "Previous steps...",
        "trimmed_messages": 5,
        "fifo_evicted_messages": 3,
        "screenshots_evicted": 1,
        "prompt_variant": "payment_v2",
        "tools_available": ["browser_click", "browser_fill_ref"],
        "rejected_tools": ["respond"],
        "focused_set": "payment",
        "tools_available_count": 2,
        "conversation_turn_count": 5,
        "message_count": 14,
        "system_prompt_hash": "hash456",
        "llm_reasoning": "Clicking checkout button.",
        "correction_messages": ["Use fill_ref not click for forms."],
        "spin_intervention": "redirect_to_alternative",
        "error_registry_context": "timeout errors on /cart",
        "continuation_nudge": "proceed_to_payment",
        "force_termination": None,
        "hard_loop_breaker": None,
        "consecutive_failure_warning": "3 failures on fill_ref",
        "approval_path": "user_confirmed",
        "page_url": "https://shop.example.test/checkout",
        "had_screenshot": True,
        "had_screenshot_image": False,
        "snapshot_compressed": True,
        "snapshot_pre_compress_len": 12000,
    }
    step = AgentStep.from_dict(payload)
    task = AgentTask(task_id="rt", steps=[step])
    surface = surface_for_task(task)
    s = surface["steps"][0]

    # Model context
    assert s["model_name"] == "gpt-4.1-nano"
    assert s["temperature"] == 0.2
    assert s["tool_choice"] == "required"
    assert s["prompt_variant"] == "payment_v2"
    assert s["context_usage_pct"] == 72.5
    assert s["context_window"] == 64000

    # Compaction
    assert s["compaction_count"] == 2
    assert s["compaction_method"] == "sliding_window"
    assert s["compaction_messages_before"] == 20
    assert s["compaction_messages_after"] == 10
    assert s["compaction_summary_preview"] == "Previous steps..."
    assert s["trimmed_messages"] == 5
    assert s["fifo_evicted_messages"] == 3
    assert s["screenshots_evicted"] == 1

    # Tools
    assert s["tools_available_names"] == ["browser_click", "browser_fill_ref"]
    assert s["tools_available_count"] == 2
    assert s["rejected_tools"] == ["respond"]
    assert s["focused_set"] == "payment"
    assert s["conversation_turn_count"] == 5
    assert s["message_count"] == 14
    assert s["system_prompt_hash"] == "hash456"

    # Reasoning & injections
    assert s["llm_reasoning"] == "Clicking checkout button."
    assert s["correction_messages"] == ["Use fill_ref not click for forms."]
    assert s["spin_intervention"] == "redirect_to_alternative"
    assert s["error_registry_context"] == "timeout errors on /cart"
    assert s["continuation_nudge"] == "proceed_to_payment"
    assert s["consecutive_failure_warning"] == "3 failures on fill_ref"
    assert s["approval_path"] == "user_confirmed"

    # Browser
    assert s["page_url"] == "https://shop.example.test/checkout"
    assert s["had_screenshot"] is True
    assert s["had_screenshot_image"] is False
    assert s["snapshot_compressed"] is True
    assert s["snapshot_pre_compress_len"] == 12000


def test_from_dict_nested_model_preserves_new_fields():
    """Nested model dict with new fields should propagate correctly."""
    payload = {
        "task_id": "n",
        "step": 1,
        "tool_name": "click",
        "tool_input": {},
        "model": {
            "model_name": "opus",
            "compaction_method": "truncate",
            "trimmed_messages": 8,
            "prompt_variant": "research_v1",
        },
        "tools": {
            "tools_available": ["search"],
            "rejected_tools": ["click"],
            "focused_set": "research",
            "conversation_turn_count": 3,
        },
        "reasoning": {
            "llm_reasoning": "Searching for info",
            "error_registry_context": "prior 404",
            "approval_path": "auto",
        },
        "browser": {
            "page_url": "https://example.test",
            "snapshot_pre_compress_len": 500,
        },
    }
    step = AgentStep.from_dict(payload)
    assert step.model.compaction_method == "truncate"
    assert step.model.trimmed_messages == 8
    assert step.model.prompt_variant == "research_v1"
    assert step.tools.rejected_tools == ["click"]
    assert step.tools.focused_set == "research"
    assert step.tools.conversation_turn_count == 3
    assert step.reasoning.error_registry_context == "prior 404"
    assert step.reasoning.approval_path == "auto"
    assert step.browser.snapshot_pre_compress_len == 500


# ── format_surface_text ──────────────────────────────────────────────


def test_format_surface_text_includes_prompt():
    task = _full_surface_task()
    surface = surface_for_task(task)
    text = format_surface_text(surface)
    assert "SYSTEM PROMPT:" in text
    assert "You are a shopping assistant." in text


def test_format_surface_text_includes_components():
    task = _full_surface_task()
    surface = surface_for_task(task)
    text = format_surface_text(surface)
    assert "PROMPT COMPONENTS:" in text
    assert "browser: enabled" in text


def test_format_surface_text_includes_prior_context():
    task = _full_surface_task()
    surface = surface_for_task(task)
    text = format_surface_text(surface)
    assert "PRIOR CONTEXT:" in text
    assert "previously browsed electronics" in text


def test_format_surface_text_includes_rejected_tools():
    task = _full_surface_task()
    surface = surface_for_task(task)
    text = format_surface_text(surface)
    assert "rejected:" in text
    assert "web_search" in text


def test_format_surface_text_includes_focused_set():
    task = _full_surface_task()
    surface = surface_for_task(task)
    text = format_surface_text(surface)
    assert "focused_set: browser_commerce" in text


def test_format_surface_text_includes_compaction():
    task = _full_surface_task()
    surface = surface_for_task(task)
    text = format_surface_text(surface)
    assert "compacted: summarize" in text
    assert "12 -> 6 messages" in text


def test_format_surface_text_includes_injections():
    task = _full_surface_task()
    surface = surface_for_task(task)
    text = format_surface_text(surface)
    assert "SPIN: reassess_after_retry" in text
    assert "CORRECTION: Don't click buy" in text
    assert "ERROR_CONTEXT:" in text
    assert "NUDGE: continue_checkout_flow" in text
    assert "APPROVAL: auto_approved:browser_click" in text


def test_format_surface_text_includes_prompt_variant():
    task = _full_surface_task()
    surface = surface_for_task(task)
    text = format_surface_text(surface)
    assert "prompt=commerce_v3" in text


# ── reasoning_for_task ───────────────────────────────────────────────


def test_reasoning_includes_injections():
    task = _full_surface_task()
    reasoning = reasoning_for_task(task)
    chain = reasoning["reasoning_chain"]
    assert chain[1]["spin_intervention"] == "reassess_after_retry"
    assert chain[1]["correction_messages"] == ["Don't click buy before checking price."]


# ── Edge cases ───────────────────────────────────────────────────────


def test_surface_no_model_no_tools_no_reasoning_no_browser():
    """Minimal step with no context objects should not crash."""
    task = AgentTask(
        task_id="bare",
        steps=[AgentStep("bare", 1, "noop", {})],
    )
    surface = surface_for_task(task)
    step = surface["steps"][0]
    assert step["model_name"] is None
    assert step["tools_available_names"] == []
    assert step["tools_available_count"] == 0
    assert step["rejected_tools"] is None
    assert step["llm_reasoning"] == ""
    assert step["page_url"] is None
    assert step["compaction_method"] is None
    assert step["correction_messages"] == []


def test_surface_empty_task():
    task = AgentTask(task_id="empty")
    surface = surface_for_task(task)
    assert surface["steps"] == []
    assert surface["prompt_text"] is None
    assert surface["system_context_components"] is None
