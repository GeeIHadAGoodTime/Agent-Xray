from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _require_string(payload: Mapping[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"task bank entry field '{field_name}' must be a non-empty string")
    return value


def _require_dict(payload: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    value = payload.get(field_name)
    if not isinstance(value, dict):
        raise ValueError(f"task bank entry field '{field_name}' must be an object")
    return dict(value)


def _require_string_list(payload: Mapping[str, Any], field_name: str) -> list[str]:
    value = payload.get(field_name)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"task bank entry field '{field_name}' must be a list of strings")
    return list(value)


@dataclass(slots=True, frozen=True)
class TaskBankEntry:
    id: str
    category: str
    user_text: str
    success_criteria: dict[str, Any]
    difficulty: str
    optimal_chain: list[str]
    test_command: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TaskBankEntry:
        test_command = payload.get("test_command")
        if test_command is not None and not isinstance(test_command, str):
            raise ValueError("task bank entry field 'test_command' must be a string when present")
        return cls(
            id=_require_string(payload, "id"),
            category=_require_string(payload, "category"),
            user_text=_require_string(payload, "user_text"),
            success_criteria=_require_dict(payload, "success_criteria"),
            difficulty=_require_string(payload, "difficulty"),
            optimal_chain=_require_string_list(payload, "optimal_chain"),
            test_command=test_command,
        )


@dataclass(slots=True)
class TaskBank:
    entries: list[TaskBankEntry] = field(default_factory=list)

    @classmethod
    def load_json(cls, path: str | Path) -> TaskBank:
        resolved = Path(path)
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("tasks")
        if not isinstance(payload, list):
            raise ValueError("task bank JSON must be an array or an object with a 'tasks' array")
        entries: list[TaskBankEntry] = []
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError("each task bank entry must be an object")
            entries.append(TaskBankEntry.from_dict(item))
        return cls(entries=entries)

    def filter(
        self,
        *,
        category: str | None = None,
        difficulty: str | None = None,
    ) -> TaskBank:
        category_key = category.casefold() if category is not None else None
        difficulty_key = difficulty.casefold() if difficulty is not None else None
        return TaskBank(
            entries=[
                entry
                for entry in self.entries
                if (category_key is None or entry.category.casefold() == category_key)
                and (difficulty_key is None or entry.difficulty.casefold() == difficulty_key)
            ]
        )

    def filter_by_category(self, category: str) -> TaskBank:
        return self.filter(category=category)

    def filter_by_difficulty(self, difficulty: str) -> TaskBank:
        return self.filter(difficulty=difficulty)

    def __iter__(self) -> Iterator[TaskBankEntry]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)


def load_task_bank(path: str | Path) -> TaskBank:
    return TaskBank.load_json(path)
