from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_xray import TaskBank, TaskBankEntry, load_task_bank
from agent_xray.contrib.task_bank import ALLOWED_ANSWER_TYPES, validate_task_bank_entries


def _write_task_bank(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_load_task_bank_from_json_array(tmp_path: Path) -> None:
    path = _write_task_bank(
        tmp_path / "task_bank.json",
        [
            {
                "id": "checkout-gate",
                "category": "commerce",
                "user_text": "Add the blue mug to cart and stop at the payment gate.",
                "success_criteria": {"must_reach_checkout": True},
                "difficulty": "medium",
                "optimal_chain": ["plan", "browse", "verify"],
                "test_command": "python -m pytest tests/test_checkout.py -q",
            }
        ],
    )

    bank = load_task_bank(path)

    assert isinstance(bank, TaskBank)
    assert len(bank) == 1
    entry = next(iter(bank))
    assert isinstance(entry, TaskBankEntry)
    assert entry.id == "checkout-gate"
    assert entry.category == "commerce"
    assert entry.success_criteria == {"must_reach_checkout": True}
    assert entry.optimal_chain == ["plan", "browse", "verify"]
    assert entry.test_command == "python -m pytest tests/test_checkout.py -q"


def test_task_bank_accepts_wrapped_tasks_payload(tmp_path: Path) -> None:
    path = _write_task_bank(
        tmp_path / "wrapped_task_bank.json",
        {
            "tasks": [
                {
                    "id": "research-citations",
                    "category": "research",
                    "user_text": "Summarize the latest launch with citations.",
                    "success_criteria": {"min_sources": 3},
                    "difficulty": "hard",
                    "optimal_chain": ["search", "read", "synthesize"],
                }
            ]
        },
    )

    bank = TaskBank.load_json(path)

    assert len(bank) == 1
    assert bank.entries[0].test_command is None


def test_task_bank_filtering_is_case_insensitive(tmp_path: Path) -> None:
    path = _write_task_bank(
        tmp_path / "task_bank.json",
        [
            {
                "id": "commerce-easy",
                "category": "Commerce",
                "user_text": "Buy one item.",
                "success_criteria": {"must_reach_cart": True},
                "difficulty": "Easy",
                "optimal_chain": ["browse"],
            },
            {
                "id": "commerce-hard",
                "category": "commerce",
                "user_text": "Complete checkout.",
                "success_criteria": {"must_reach_checkout": True},
                "difficulty": "Hard",
                "optimal_chain": ["browse", "fill", "verify"],
            },
            {
                "id": "research-hard",
                "category": "research",
                "user_text": "Compare three providers.",
                "success_criteria": {"min_sources": 3},
                "difficulty": "hard",
                "optimal_chain": ["search", "read", "synthesize"],
            },
        ],
    )

    bank = load_task_bank(path)

    commerce_bank = bank.filter_by_category("commerce")
    hard_bank = bank.filter_by_difficulty("HARD")
    commerce_hard_bank = bank.filter(category="commerce", difficulty="hard")

    assert [entry.id for entry in commerce_bank] == ["commerce-easy", "commerce-hard"]
    assert [entry.id for entry in hard_bank] == ["commerce-hard", "research-hard"]
    assert [entry.id for entry in commerce_hard_bank] == ["commerce-hard"]


def test_load_task_bank_rejects_invalid_entry_shape(tmp_path: Path) -> None:
    path = _write_task_bank(
        tmp_path / "invalid_task_bank.json",
        [
            {
                "id": "broken",
                "category": "commerce",
                "user_text": "Do the thing.",
                "success_criteria": [],
                "difficulty": "medium",
                "optimal_chain": ["browse"],
            }
        ],
    )

    with pytest.raises(ValueError, match="success_criteria"):
        load_task_bank(path)


@pytest.mark.parametrize("answer_type", sorted(ALLOWED_ANSWER_TYPES))
def test_validate_task_bank_entries_accepts_extended_answer_types(answer_type: str) -> None:
    result = validate_task_bank_entries([
        {
            "id": f"task-{answer_type}",
            "user_text": "Handle the task.",
            "success_criteria": {"answer_type": answer_type},
        }
    ])

    assert result.valid is True
    assert result.errors == []
    assert result.warnings == []


def test_validate_task_bank_entries_reports_unknown_criteria_as_warning() -> None:
    result = validate_task_bank_entries([
        {
            "id": "novviola-florist",
            "user_text": "Find a local florist and place the order.",
            "success_criteria": {"local_florist_allowed": True},
        }
    ])

    assert result.valid is True
    assert result.errors == []
    assert result.warnings
    assert "unknown criterion 'local_florist_allowed'" in result.warnings[0]
