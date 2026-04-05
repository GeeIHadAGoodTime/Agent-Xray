from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_xray.analyzer import analyze_task
from agent_xray.contrib import novviola as novviola_mod
from agent_xray.contrib import task_bank as task_bank_mod
import agent_xray.diagnose as diagnose_mod
from agent_xray.diagnose import get_target_resolver
from agent_xray.grader import GradeResult, SignalResult
from agent_xray.schema import AgentStep, AgentTask, TaskOutcome


def _step(
    step: int,
    tool_name: str,
    tool_input: dict[str, object] | None = None,
    *,
    tool_result: str | None = None,
    error: str | None = None,
    page_url: str | None = None,
    tools_available: list[str] | None = None,
    llm_reasoning: str | None = None,
    cost_usd: float | None = None,
) -> AgentStep:
    return AgentStep(
        task_id="task-1",
        step=step,
        tool_name=tool_name,
        tool_input=tool_input or {},
        tool_result=tool_result,
        error=error,
        page_url=page_url,
        tools_available=tools_available,
        llm_reasoning=llm_reasoning,
        cost_usd=cost_usd,
    )


def _task(
    *steps: AgentStep,
    task_text: str = "Add the blue mug to cart and stop at checkout on shop.example.test.",
    task_category: str = "commerce",
    status: str = "success",
    final_answer: str | None = "Stopped at checkout and did not enter payment details.",
) -> AgentTask:
    return AgentTask(
        task_id="task-1",
        task_text=task_text,
        task_category=task_category,
        steps=list(steps),
        outcome=TaskOutcome(
            task_id="task-1",
            status=status,
            total_steps=len(steps),
            total_duration_s=float(len(steps)),
            final_answer=final_answer,
            timestamp="2026-04-05T12:00:00Z",
        ),
    )


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_load_task_bank_accepts_json_array(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "task_bank.json",
        [
            {
                "id": "checkout-gate",
                "user_text": "Add the blue mug to cart and stop at checkout.",
                "success_criteria": {"must_reach_checkout": True},
            }
        ],
    )

    bank = task_bank_mod.load_task_bank(path)

    assert bank[0]["id"] == "checkout-gate"
    assert bank[0]["success_criteria"]["must_reach_checkout"] is True


def test_load_task_bank_accepts_wrapped_tasks_payload(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "wrapped.json",
        {
            "tasks": [
                {
                    "id": "research-1",
                    "user_text": "Summarize three sources.",
                    "success_criteria": {"min_urls": 3},
                }
            ]
        },
    )

    bank = task_bank_mod.load_task_bank(path)

    assert [entry["id"] for entry in bank] == ["research-1"]


def test_load_task_bank_rejects_unsupported_shape(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "bad.json", {"entries": []})

    with pytest.raises(ValueError, match="Unsupported task bank format"):
        task_bank_mod.load_task_bank(path)


def test_validate_task_bank_entries_accepts_empty_bank() -> None:
    result = task_bank_mod.validate_task_bank_entries([])

    assert result.valid is True
    assert result.errors == []
    assert result.warnings == []


def test_validate_task_bank_entries_reports_missing_fields_and_duplicate_ids() -> None:
    result = task_bank_mod.validate_task_bank_entries(
        [
            {"id": "dup", "user_text": "Missing criteria"},
            {"id": "dup", "user_text": "", "success_criteria": {}},
        ]
    )

    assert result.valid is False
    assert any("missing required field(s): success_criteria" in error for error in result.errors)
    assert any("duplicate task id 'dup'" in error for error in result.errors)
    assert any("field 'user_text' must be a non-empty string" in error for error in result.errors)


def test_validate_task_bank_entries_reports_invalid_criterion_types_and_values() -> None:
    result = task_bank_mod.validate_task_bank_entries(
        [
            {
                "id": "broken",
                "user_text": "Fill checkout fields.",
                "success_criteria": {
                    "must_fill_fields": ["email", ""],
                    "max_steps": -1,
                    "min_tool_count": "two",
                },
            }
        ]
    )

    assert result.valid is False
    assert any("criterion 'must_fill_fields' must contain non-empty strings" in error for error in result.errors)
    assert any("criterion 'min_tool_count' must be int, got str" in error for error in result.errors)
    assert any("criterion 'max_steps' must be >= 0" in warning for warning in result.warnings)


def test_validate_task_bank_entries_warns_for_unknown_criteria_and_answer_type() -> None:
    result = task_bank_mod.validate_task_bank_entries(
        [
            {
                "id": "warn-only",
                "user_text": "Do the thing.",
                "success_criteria": {
                    "unknown_flag": True,
                    "answer_type": "narrative",
                },
            }
        ]
    )

    assert result.valid is True
    assert result.errors == []
    assert any("unknown criterion 'unknown_flag'" in warning for warning in result.warnings)
    assert any("criterion 'answer_type' must be one of" in warning for warning in result.warnings)


def test_validate_task_bank_from_path_loads_and_validates(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "bank.json",
        [
            {
                "id": "empty-list-warning",
                "user_text": "Use web search.",
                "success_criteria": {"must_use_tools": []},
            }
        ],
    )

    result = task_bank_mod.validate_task_bank(path)

    assert result.valid is True
    assert result.errors == []
    assert any("criterion 'must_use_tools' must not be empty" in warning for warning in result.warnings)


def test_match_task_to_bank_prefers_site_and_category_aligned_entry() -> None:
    task = _task(
        _step(1, "browser_navigate", {"url": "https://shop.example.test/cart"}, page_url="https://shop.example.test/cart"),
        task_text="Add the blue mug to cart and stop at checkout on shop.example.test.",
    )
    analysis = analyze_task(task)
    bank = [
        {
            "id": "wrong-site",
            "user_text": "Add the blue mug to cart and stop at checkout on shop.example.test.",
            "category": "commerce",
            "site": "other.example.test",
            "success_criteria": {},
        },
        {
            "id": "right-site",
            "user_text": "Add the blue mug to cart and stop at checkout on shop.example.test.",
            "category": "commerce",
            "site": "shop.example.test",
            "success_criteria": {},
        },
    ]

    match = task_bank_mod.match_task_to_bank(task, bank, analysis=analysis)

    assert match is not None
    assert match["id"] == "right-site"


def test_match_task_to_bank_returns_none_for_empty_text_or_low_similarity() -> None:
    blank_task = _task(_step(1, "web_search"), task_text="")
    unrelated_task = _task(_step(1, "web_search"), task_text="Research EU battery regulations.")
    bank = [
        {
            "id": "commerce-only",
            "user_text": "Buy socks and proceed to checkout.",
            "success_criteria": {},
        }
    ]

    assert task_bank_mod.match_task_to_bank(blank_task, bank) is None
    assert task_bank_mod.match_task_to_bank(unrelated_task, bank, threshold=0.9) is None


def test_evaluate_task_criteria_reports_pass_and_fail_lines() -> None:
    task = _task(
        _step(
            1,
            "browser_fill_ref",
            {"fields": ["email", "address"], "text": "alice@example.com 123 Main St"},
            tool_result="Shipping form accepted.",
            page_url="https://shop.example.test/checkout",
        ),
        _step(
            2,
            "respond",
            {},
            tool_result="Stopped at checkout.",
            tools_available=["respond"],
        ),
        final_answer="Stopped at checkout and did not enter payment details.",
    )
    analysis = analyze_task(task)

    results = task_bank_mod.evaluate_task_criteria(
        task,
        analysis,
        {
            "must_answer_contains": ["checkout"],
            "must_fill_fields": ["email", "address"],
            "must_use_tools": ["browser_fill_ref"],
            "max_steps": 1,
        },
    )

    assert any(line.startswith("[PASS] must_answer_contains") for line in results)
    assert any(line.startswith("[PASS] must_fill_fields") for line in results)
    assert any(line.startswith("[PASS] must_use_tools") for line in results)
    assert any(line.startswith("[FAIL] max_steps") for line in results)


def test_evaluate_task_criteria_handles_invalid_and_unknown_criteria() -> None:
    task = _task(_step(1, "respond"), final_answer=None, status="failed")
    analysis = analyze_task(task)

    results = task_bank_mod.evaluate_task_criteria(
        task,
        analysis,
        {
            "must_answer_contains": [],
            "unknown_flag": True,
            "must_have_answer": True,
        },
    )

    assert "[FAIL] must_answer_contains: must_answer_contains requires a non-empty list" in results
    assert "[FAIL] unknown_flag: unknown criterion 'unknown_flag' (not implemented)" in results
    assert any(line.startswith("[FAIL] must_have_answer") for line in results)


def test_evaluate_task_criteria_detects_payment_fill_and_checkout_success() -> None:
    task = _task(
        _step(
            1,
            "browser_click",
            {"ref": "add-to-cart"},
            tool_result="Added to cart. Your cart subtotal is $15.",
            page_url="https://shop.example.test/cart",
        ),
        _step(
            2,
            "browser_click",
            {"ref": "checkout"},
            tool_result="Checkout page loaded.",
            page_url="https://shop.example.test/checkout",
        ),
        _step(
            3,
            "browser_fill_ref",
            {
                "ref": "payment-form",
                "fields": ["card number", "cvv", "expiration"],
                "text": "4111 1111 1111 1111 123 12/29",
            },
            tool_result="card number and cvv accepted",
            page_url="https://shop.example.test/payment",
        ),
    )
    analysis = analyze_task(task)

    results = task_bank_mod.evaluate_task_criteria(
        task,
        analysis,
        {
            "must_reach_cart": True,
            "must_reach_checkout": True,
            "payment_fields_visible": True,
            "must_not_fill_payment": True,
        },
    )

    assert any(line.startswith("[PASS] must_reach_cart") for line in results)
    assert any(line.startswith("[PASS] must_reach_checkout") for line in results)
    assert any(line.startswith("[PASS] payment_fields_visible") for line in results)
    assert any(line.startswith("[FAIL] must_not_fill_payment") for line in results)


def test_grade_with_task_bank_adds_signals_and_downgrades_on_critical_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bank_path = _write_json(
        tmp_path / "task_bank.json",
        [
            {
                "id": "checkout-gate",
                "user_text": "Add the blue mug to cart and stop at checkout on shop.example.test.",
                "category": "commerce",
                "site": "shop.example.test",
                "success_criteria": {"must_reach_url": r"/payment"},
            }
        ],
    )
    task = _task(
        _step(1, "browser_click", {"ref": "add-to-cart"}, tool_result="Added to cart.", page_url="https://shop.example.test/cart"),
        _step(2, "browser_click", {"ref": "checkout"}, tool_result="Checkout loaded.", page_url="https://shop.example.test/checkout"),
    )

    monkeypatch.setattr(task_bank_mod, "load_rules", lambda rules=None: SimpleNamespace(name="fake"))
    monkeypatch.setattr(
        task_bank_mod,
        "grade_task",
        lambda task, ruleset, analysis=None: GradeResult(
            task_id=task.task_id,
            grade="GOLDEN",
            score=12,
            reasons=["base"],
            metrics={},
            signals=[
                SignalResult(
                    name="baseline",
                    passed=True,
                    points=12,
                    actual=True,
                    reason="baseline grade",
                )
            ],
        ),
    )

    [result] = task_bank_mod.grade_with_task_bank([task], bank_path)

    assert result.grade == "GOOD"
    assert any(signal.name == "task_bank_match" and signal.actual == "checkout-gate" for signal in result.signals)
    criteria_signal = next(signal for signal in result.signals if signal.name == "task_bank_criteria")
    assert criteria_signal.passed is False
    assert criteria_signal.actual == {"passed": 0, "failed": 1, "total": 1}
    assert any(reason.startswith("[FAIL] must_reach_url") for reason in result.reasons)
    assert any(reason.startswith("[DOWNGRADE]") for reason in result.reasons)


def test_grade_with_task_bank_leaves_grade_unchanged_when_bank_is_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bank_path = _write_json(tmp_path / "empty.json", [])
    task = _task(_step(1, "respond"))
    baseline = GradeResult(
        task_id=task.task_id,
        grade="OK",
        score=3,
        reasons=["base"],
        metrics={},
        signals=[],
    )

    monkeypatch.setattr(task_bank_mod, "load_rules", lambda rules=None: SimpleNamespace(name="fake"))
    monkeypatch.setattr(task_bank_mod, "grade_task", lambda task, ruleset, analysis=None: baseline)

    [result] = task_bank_mod.grade_with_task_bank([task], bank_path)

    assert result.grade == "OK"
    assert result.reasons == ["base"]
    assert result.signals == []


def test_novviola_resolver_refines_prompt_bug_targets_from_section_and_pattern() -> None:
    resolver = novviola_mod.NovviolaTargetResolver()

    targets = resolver.resolve(
        "prompt_bug",
        [
            "prompt_section=response_format: malformed final answer",
            "The agent hallucinated an unknown.tool during recovery.",
        ],
    )

    assert targets[0] == "services/llm/prompts/sections/response_format.py"
    assert "mcp_hub/client_hub.py" in targets
    assert "services/llm/prompts/builder.py" in targets


def test_novviola_verify_commands_return_task_specific_and_fallback_commands() -> None:
    commands = novviola_mod.NovviolaVerifyCommands()

    task_specific = commands.get("timeout", task_id="task-123")
    fallback = commands.get("totally_unknown")

    assert 'grep "task-123"' in task_specific
    assert "agent-steps-*.jsonl" in task_specific
    assert fallback.startswith("tail -20 logs/structured/agent-steps-")


def test_novviola_register_makes_resolver_available() -> None:
    importlib.reload(novviola_mod)

    previous = diagnose_mod._ACTIVE_TARGET_RESOLVER
    try:
        novviola_mod.register()

        resolver = get_target_resolver("novviola")
        assert isinstance(resolver, novviola_mod.NovviolaTargetResolver)
    finally:
        diagnose_mod._ACTIVE_TARGET_RESOLVER = previous
