from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_xray.contrib import novviola as novviola_mod
from agent_xray.contrib import task_bank as task_bank_mod
from agent_xray.schema import AgentStep, AgentTask, TaskOutcome


class StubAnalysis:
    def __init__(
        self,
        *,
        unique_urls: list[str] | None = None,
        unique_url_paths: list[str] | None = None,
        unique_tools: list[str] | None = None,
        site_name: str = "",
        metrics_data: dict[str, object] | None = None,
        is_spin: bool = False,
        total_cost_usd: float = 0.0,
        errors: int = 0,
        error_rate: float = 0.0,
    ) -> None:
        self.unique_urls = list(unique_urls or [])
        self.unique_url_paths = list(unique_url_paths or [])
        self.unique_tools = list(unique_tools or [])
        self.site_name = site_name
        self._metrics_data = dict(metrics_data or {})
        self.is_spin = is_spin
        self.total_cost_usd = total_cost_usd
        self.errors = errors
        self.error_rate = error_rate

    def metrics(self) -> dict[str, object]:
        return dict(self._metrics_data)


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _step(
    step: int,
    tool_name: str,
    tool_input: dict[str, object] | None = None,
    *,
    tool_result: str | None = None,
    error: str | None = None,
    page_url: str | None = None,
) -> AgentStep:
    return AgentStep(
        task_id="task-1",
        step=step,
        tool_name=tool_name,
        tool_input=tool_input or {},
        tool_result=tool_result,
        error=error,
        page_url=page_url,
    )


def _task(
    *steps: AgentStep,
    task_text: str = "Add the blue mug to cart.",
    task_category: str = "commerce",
    status: str = "success",
    final_answer: str | None = "Done.",
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


def test_validate_task_bank_accepts_valid_schema(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "valid_bank.json",
        {
            "tasks": [
                {
                    "id": "valid-task",
                    "user_text": "Summarize the document and cite the source.",
                    "success_criteria": {
                        "must_answer_contains": ["source"],
                        "max_steps": 4,
                        "min_tool_count": 1,
                    },
                }
            ]
        },
    )

    result = task_bank_mod.validate_task_bank(path)

    assert result.valid is True
    assert result.errors == []
    assert result.warnings == []


def test_validate_task_bank_accepts_empty_bank(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "empty_bank.json", [])

    result = task_bank_mod.validate_task_bank(path)

    assert result.valid is True
    assert result.errors == []
    assert result.warnings == []


def test_validate_task_bank_reports_non_object_and_missing_required_fields(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "broken_bank.json",
        [
            "not-an-object",
            {"id": "only-id"},
        ],
    )

    result = task_bank_mod.validate_task_bank(path)

    assert result.valid is False
    assert any("expected an object, got str" in error for error in result.errors)
    assert any("missing required field(s): success_criteria, user_text" in error for error in result.errors)


def test_validate_task_bank_reports_invalid_schema_details(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "invalid_schema.json",
        [
            {
                "id": "",
                "user_text": "Checkout.",
                "success_criteria": [],
            },
            {
                "id": "dup",
                "user_text": "First.",
                "success_criteria": {},
            },
            {
                "id": "dup",
                "user_text": "Second.",
                "success_criteria": {
                    "must_use_tools": [],
                    "min_urls": -1,
                },
            },
        ],
    )

    result = task_bank_mod.validate_task_bank(path)

    assert result.valid is False
    assert any("field 'id' must be a non-empty string" in error for error in result.errors)
    assert any("field 'success_criteria' must be an object" in error for error in result.errors)
    assert any("duplicate task id 'dup'" in error for error in result.errors)
    assert any("criterion 'must_use_tools' must not be empty" in warning for warning in result.warnings)
    assert any("criterion 'min_urls' must be >= 0" in warning for warning in result.warnings)


def test_match_task_to_bank_allows_strong_text_similarity_without_shared_tokens() -> None:
    task = _task(task_text="aaaaaaaaab", task_category="research")
    bank = [
        {
            "id": "near-match",
            "user_text": "aaaaaaaaac",
            "success_criteria": {},
        }
    ]
    analysis = StubAnalysis(site_name="docs.example.test")

    match = task_bank_mod.match_task_to_bank(task, bank, analysis=analysis, threshold=0.2)

    assert match is not None
    assert match["id"] == "near-match"


def test_match_task_to_bank_keeps_first_entry_when_scores_tie() -> None:
    task = _task(task_text="Buy blue mug for pickup", task_category="commerce")
    bank = [
        {
            "id": "first",
            "user_text": "Buy blue mug for pickup",
            "success_criteria": {},
        },
        {
            "id": "second",
            "user_text": "Buy blue mug for pickup",
            "success_criteria": {},
        },
    ]
    analysis = StubAnalysis(site_name="shop.example.test")

    match = task_bank_mod.match_task_to_bank(task, bank, analysis=analysis)

    assert match is not None
    assert match["id"] == "first"


def test_match_task_to_bank_ignores_stopword_only_overlap() -> None:
    task = _task(task_text="the and or", task_category="research")
    bank = [
        {
            "id": "stopwords-only",
            "user_text": "and or the",
            "success_criteria": {},
        }
    ]
    analysis = StubAnalysis(site_name="docs.example.test")

    match = task_bank_mod.match_task_to_bank(task, bank, analysis=analysis)

    assert match is None


def test_evaluate_task_criteria_must_reach_url_matches_query_and_case_insensitive() -> None:
    task = _task(task_text="Open checkout.")
    analysis = StubAnalysis(
        unique_urls=["https://shop.example.test/CHECKOUT?step=Review"],
        unique_url_paths=["shop.example.test/checkout"],
    )

    results = task_bank_mod.evaluate_task_criteria(
        task,
        analysis,
        {"must_reach_url": r"checkout\?step=review"},
    )

    assert results == [
        "[PASS] must_reach_url: URL matched: https://shop.example.test/CHECKOUT?step=Review"
    ]


def test_evaluate_task_criteria_must_answer_contains_checks_status_when_answer_missing() -> None:
    task = _task(final_answer=None, status="Checkout ready for review")
    analysis = StubAnalysis()

    results = task_bank_mod.evaluate_task_criteria(
        task,
        analysis,
        {"must_answer_contains": ["checkout", "review"]},
    )

    assert any(line.startswith("[PASS] must_answer_contains") for line in results)


def test_evaluate_task_criteria_answer_type_action_uses_final_answer_markers() -> None:
    task = _task(
        final_answer="Draft reply created and queued for send.",
        status="failed",
    )
    analysis = StubAnalysis()

    results = task_bank_mod.evaluate_task_criteria(
        task,
        analysis,
        {"answer_type": "action"},
    )

    assert results == ["[PASS] answer_type: action confirmation present"]


def test_evaluate_task_criteria_answer_type_consultative_rejects_short_answer() -> None:
    task = _task(final_answer="This answer is too short.")
    analysis = StubAnalysis()

    results = task_bank_mod.evaluate_task_criteria(
        task,
        analysis,
        {"answer_type": "consultative"},
    )

    assert results == ["[FAIL] answer_type: answer too short for consultative (25 chars)"]


def test_evaluate_task_criteria_min_urls_counts_unique_paths_not_query_variants() -> None:
    task = _task(task_text="Browse multiple results.")
    analysis = StubAnalysis(
        unique_urls=[
            "https://shop.example.test/search?q=mug",
            "https://shop.example.test/search?q=cup",
        ],
        unique_url_paths=["shop.example.test/search"],
    )

    results = task_bank_mod.evaluate_task_criteria(
        task,
        analysis,
        {"min_urls": 2},
    )

    assert results == ["[FAIL] min_urls: only 1 unique URLs (need 2)"]


def test_evaluate_task_criteria_payment_gate_passes_without_payment_fill() -> None:
    task = _task(
        _step(1, "browser_click", {"ref": "checkout"}, page_url="https://shop.example.test/checkout"),
        status="payment_gate",
        final_answer="Stopped at PAYMENT_GATE without entering payment info.",
    )
    analysis = StubAnalysis()

    results = task_bank_mod.evaluate_task_criteria(
        task,
        analysis,
        {"must_not_fill_payment": True},
    )

    assert results == [
        "[PASS] must_not_fill_payment: stopped at payment gate without filling payment"
    ]


def test_evaluate_task_criteria_payment_fields_visible_reports_partial_payment_progress() -> None:
    task = _task(task_text="Reach payment page.")
    analysis = StubAnalysis(
        metrics_data={
            "commerce": {
                "reached_payment": True,
                "payment_fields_confirmed": False,
            }
        }
    )

    results = task_bank_mod.evaluate_task_criteria(
        task,
        analysis,
        {"payment_fields_visible": True},
    )

    assert results == [
        "[FAIL] payment_fields_visible: payment page reached but card fields not confirmed strong"
    ]


def test_evaluate_task_criteria_must_have_answer_accepts_completed_status_without_text() -> None:
    task = _task(final_answer=None, status="completed")
    analysis = StubAnalysis()

    results = task_bank_mod.evaluate_task_criteria(
        task,
        analysis,
        {"must_have_answer": True},
    )

    assert results == ["[PASS] must_have_answer: outcome=completed (answer may be in TTS)"]


def test_novviola_prompt_bug_deduplicates_section_targets_and_keeps_builder_fallback() -> None:
    resolver = novviola_mod.NovviolaTargetResolver()

    targets = resolver.resolve(
        "prompt_bug",
        [
            "prompt_section=browser: mismatch",
            "prompt_section=browser: duplicate section mention",
        ],
    )

    assert targets[0] == "mcp_servers/browser/server.py"
    assert targets.count("mcp_servers/browser/server.py") == 1
    assert targets[-1] == "services/llm/prompts/builder.py"


def test_novviola_resolver_returns_no_direct_code_fix_for_empty_target_mapping() -> None:
    resolver = novviola_mod.NovviolaTargetResolver()

    targets = resolver.resolve("model_limit", ["task too large for the current model"])

    assert targets == ["(no direct code fix)"]
