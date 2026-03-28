from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from agent_xray.cli import build_parser, cmd_analyze, cmd_grade, cmd_root_cause, cmd_task_bank


def _write_task_bank(path: Path) -> Path:
    payload = {
        "tasks": [
            {
                "id": "checkout-wireless-headset",
                "user_text": "Buy the wireless headset and complete checkout on shop.example.test.",
                "site": "shop.example.test",
                "category": "commerce",
                "success_criteria": {
                    "must_reach_checkout": True,
                    "must_reach_url": "receipt/ready",
                },
            }
        ]
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_invalid_task_bank(path: Path) -> Path:
    payload = {
        "tasks": [
            {
                "id": "",
                "user_text": "broken bank entry",
                "success_criteria": {"made_up_rule": True},
            }
        ]
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_root_cause_parser_exists() -> None:
    parser = build_parser()
    args = parser.parse_args(["root-cause", "./traces", "--task-bank", "task_bank.json"])
    assert args.command == "root-cause"
    assert args.task_bank == "task_bank.json"


def test_task_bank_parser_exists() -> None:
    parser = build_parser()
    args = parser.parse_args(["task-bank", "validate", "task_bank.json"])
    assert args.command == "task-bank"
    assert args.task_bank_command == "validate"
    assert args.path == "task_bank.json"


def test_cmd_grade_task_bank_caps_golden_to_good(
    tmp_trace_dir,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    task_bank = _write_task_bank(tmp_path / "task_bank.json")
    result = cmd_grade(
        Namespace(
            log_dir=tmp_trace_dir,
            days=None,
            rules=None,
            format="auto",
            pattern=None,
            json=True,
            task_bank=task_bank,
            grade_filter=None,
            site_filter=None,
            outcome_filter=None,
            since_filter=None,
            expected_rejections=[],
            verbose=False,
            quiet=False,
            no_color=True,
        )
    )
    payload = json.loads(capsys.readouterr().out)
    grades = {item["task_id"]: item for item in payload["tasks"]}
    assert result == 0
    assert grades["golden-task"]["grade"] == "GOOD"
    assert any("must_reach_url" in reason for reason in grades["golden-task"]["reasons"])


def test_cmd_analyze_task_bank_uses_criterion_aware_grades(
    tmp_trace_dir,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    task_bank = _write_task_bank(tmp_path / "task_bank.json")
    result = cmd_analyze(
        Namespace(
            log_dir=tmp_trace_dir,
            days=None,
            rules=None,
            format="auto",
            pattern=None,
            json=True,
            task_bank=task_bank,
            verbose=False,
            quiet=False,
            no_color=True,
        )
    )
    payload = json.loads(capsys.readouterr().out)
    grades = {item["task_id"]: item["grade"] for item in payload["tasks"]}
    assert result == 0
    assert grades["golden-task"] == "GOOD"


def test_cmd_root_cause_json(
    tmp_trace_dir,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    task_bank = _write_task_bank(tmp_path / "task_bank.json")
    result = cmd_root_cause(
        Namespace(
            log_dir=tmp_trace_dir,
            days=None,
            rules="default",
            format="auto",
            pattern=None,
            json=True,
            task_bank=task_bank,
            grade_filter=None,
            site_filter=None,
            outcome_filter=None,
            since_filter=None,
            expected_rejections=[],
            verbose=False,
            quiet=False,
            no_color=True,
        )
    )
    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["summary"]["classified_failures"] >= 1
    assert any(item["task_id"] == "broken-task" for item in payload["tasks"])


def test_cmd_task_bank_list_show_validate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    task_bank = _write_task_bank(tmp_path / "task_bank.json")

    result = cmd_task_bank(
        Namespace(
            task_bank_command="list",
            path=task_bank,
            json=False,
            verbose=False,
            quiet=False,
            no_color=True,
        )
    )
    out = capsys.readouterr().out
    assert result == 0
    assert "checkout-wireless-headset" in out

    result = cmd_task_bank(
        Namespace(
            task_bank_command="show",
            path=task_bank,
            task_id="checkout-wireless-headset",
            json=False,
            verbose=False,
            quiet=False,
            no_color=True,
        )
    )
    out = capsys.readouterr().out
    assert result == 0
    assert "success_criteria" in out
    assert "must_reach_checkout" in out

    result = cmd_task_bank(
        Namespace(
            task_bank_command="validate",
            path=task_bank,
            json=False,
            verbose=False,
            quiet=False,
            no_color=True,
        )
    )
    out = capsys.readouterr().out
    assert result == 0
    assert "TASK BANK VALID" in out


def test_cmd_task_bank_validate_invalid_bank(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    task_bank = _write_invalid_task_bank(tmp_path / "invalid_task_bank.json")
    result = cmd_task_bank(
        Namespace(
            task_bank_command="validate",
            path=task_bank,
            json=True,
            verbose=False,
            quiet=False,
            no_color=True,
        )
    )
    payload = json.loads(capsys.readouterr().out)
    assert result == 1
    assert payload["valid"] is False
    assert payload["errors"]
