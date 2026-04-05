"""Comprehensive tests for the enforce engine."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_xray.enforce import (
    ChangeRecord,
    ChallengeResult,
    DiffHunk,
    EnforceConfig,
    _load_challenges,
    _load_iterations,
    _load_session,
    _save_challenge,
    _save_iteration,
    _save_session,
    _session_dir,
    build_enforce_report,
    enforce_challenge,
    enforce_check,
    enforce_init,
    enforce_plan,
    enforce_reset,
    enforce_status,
)
from agent_xray.enforce_audit import (
    audit_change,
    challenge_iterations,
    detect_assertion_weakening,
    detect_early_return,
    detect_exception_swallowing,
    detect_hardcoded_values,
    detect_import_removal,
    detect_mock_insertion,
    detect_special_case_branching,
    detect_test_file_modification,
)
from agent_xray.enforce_report import (
    format_enforce_json,
    format_enforce_markdown,
    format_enforce_text,
    generate_report,
)


def _result(
    passed: int,
    failed: int,
    *,
    errors: int = 0,
    skipped: int = 0,
    failed_names: list[str] | None = None,
    passed_names: list[str] | None = None,
    output: str | None = None,
) -> object:
    from agent_xray.enforce import TestResult

    failed_names = failed_names or []
    passed_names = passed_names or []
    total = passed + failed + errors + skipped
    return TestResult(
        exit_code=0 if failed == 0 and errors == 0 else 1,
        passed=passed,
        failed=failed,
        errors=errors,
        skipped=skipped,
        total=total,
        duration_seconds=1.25,
        output=output or f"{passed} passed, {failed} failed",
        test_names_passed=passed_names,
        test_names_failed=failed_names,
    )


def _baseline():
    return _result(
        3,
        2,
        failed_names=[
            "tests/test_app.py::test_alpha",
            "tests/test_app.py::test_beta",
        ],
        passed_names=[
            "tests/test_app.py::test_gamma",
            "tests/test_app.py::test_delta",
            "tests/test_app.py::test_epsilon",
        ],
    )


def _improved():
    return _result(
        4,
        1,
        failed_names=["tests/test_app.py::test_beta"],
        passed_names=[
            "tests/test_app.py::test_alpha",
            "tests/test_app.py::test_gamma",
            "tests/test_app.py::test_delta",
            "tests/test_app.py::test_epsilon",
        ],
    )


def _mixed():
    return _result(
        3,
        2,
        failed_names=[
            "tests/test_app.py::test_beta",
            "tests/test_app.py::test_zeta",
        ],
        passed_names=[
            "tests/test_app.py::test_alpha",
            "tests/test_app.py::test_gamma",
            "tests/test_app.py::test_delta",
            "tests/test_app.py::test_epsilon",
        ],
    )


def _regressed():
    return _result(
        2,
        3,
        failed_names=[
            "tests/test_app.py::test_alpha",
            "tests/test_app.py::test_beta",
            "tests/test_app.py::test_zeta",
        ],
        passed_names=[
            "tests/test_app.py::test_gamma",
            "tests/test_app.py::test_delta",
        ],
    )


def _all_green():
    return _result(
        5,
        0,
        passed_names=[
            "tests/test_app.py::test_alpha",
            "tests/test_app.py::test_beta",
            "tests/test_app.py::test_gamma",
            "tests/test_app.py::test_delta",
            "tests/test_app.py::test_epsilon",
        ],
    )


def _config(tmp_path: Path, **overrides: object) -> EnforceConfig:
    config = EnforceConfig(
        test_command="python -m pytest -q",
        project_root=str(tmp_path),
        max_iterations=10,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _diff(
    *,
    file: str = "src/app.py",
    before_line: str = "    return old_value",
    after_lines: list[str] | None = None,
) -> str:
    after_lines = after_lines or ["    return new_value"]
    header = [
        f"diff --git a/{file} b/{file}",
        f"--- a/{file}",
        f"+++ b/{file}",
        "@@ -1,1 +1,2 @@",
        f"-{before_line}",
    ]
    return "\n".join([*header, *[f"+{line}" for line in after_lines]]) + "\n"


def _default_check_kwargs(after_result=None, **overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "_run_tests_fn": lambda cmd, cwd: after_result or _improved(),
        "_audit_fn": lambda diff, files_modified, allow_test_modification: ("VALID", [], []),
        "_git_diff_fn": lambda cwd: " src/app.py | 2 +-\n 1 file changed, 1 insertion(+), 1 deletion(-)",
        "_git_names_fn": lambda cwd: ["src/app.py"],
        "_git_commit_fn": lambda msg, cwd: "deadbeef",
        "_git_revert_fn": lambda head, cwd: True,
        "_git_head_fn": lambda cwd: "head123",
        "_git_diff_content_fn": lambda cwd: _diff(
            after_lines=[
                "    if value is None:",
                "        return 0",
            ],
        ),
    }
    kwargs.update(overrides)
    return kwargs


def _record(iteration: int, **overrides: object) -> ChangeRecord:
    record = ChangeRecord(
        iteration=iteration,
        files_modified=["src/app.py"],
        hypothesis=f"iteration {iteration}",
        before=_baseline(),
        after=_improved(),
        tests_improved=["tests/test_app.py::test_alpha"],
        tests_regressed=[],
        net_improvement=1,
        audit_verdict="VALID",
        decision="RECOMMEND_COMMIT",
        diff_stat=" src/app.py | 2 +-",
        prediction_accuracy="accurate" if iteration == 1 else "no_prediction",
        predicted_tests=["test_alpha"] if iteration == 1 else [],
        actual_improved=["tests/test_app.py::test_alpha"] if iteration == 1 else [],
        prediction_match_pct=100.0 if iteration == 1 else 0.0,
        diff_hunks=[
            DiffHunk(
                file="src/app.py",
                line_number=1,
                removed_lines=["L1: -    return old_value"],
                added_lines=["L1: +    return new_value"],
            ).to_dict()
        ],
        review_summary="1 improved, 0 regressed (net +1). Audit: VALID. 1 file(s), 2 diff line(s). Recommend: commit.",
    )
    for key, value in overrides.items():
        setattr(record, key, value)
    return record


def test_session_persistence_round_trip(tmp_path: Path) -> None:
    config = _config(tmp_path)
    baseline = _baseline()

    session_dir = _save_session(config, baseline, str(tmp_path))
    loaded_config, loaded_baseline, session_data = _load_session(str(tmp_path))

    assert session_dir == _session_dir(str(tmp_path))
    assert loaded_config.test_command == config.test_command
    assert loaded_baseline.failed == baseline.failed
    assert session_data["iteration_count"] == 0
    assert "started_at" in session_data


def test_session_state_tracks_iterations_and_challenges(tmp_path: Path) -> None:
    _save_session(_config(tmp_path), _baseline(), str(tmp_path))
    _save_iteration(_record(1), str(tmp_path))
    _save_iteration(_record(2, decision="RECOMMEND_REVERT", audit_verdict="GAMING"), str(tmp_path))
    _save_challenge(
        ChallengeResult(iteration_range=(1, 2), changes_reviewed=2, findings=["reviewed"]),
        1,
        str(tmp_path),
    )

    iterations = _load_iterations(str(tmp_path))
    challenges = _load_challenges(str(tmp_path))

    assert [item.iteration for item in iterations] == [1, 2]
    assert challenges[0].iteration_range == (1, 2)
    assert challenges[0].changes_reviewed == 2


def test_enforce_status_counts_commit_revert_and_gaming(tmp_path: Path) -> None:
    _save_session(_config(tmp_path), _baseline(), str(tmp_path))
    _save_iteration(_record(1), str(tmp_path))
    _save_iteration(
        _record(
            2,
            decision="RECOMMEND_REVERT",
            audit_verdict="GAMING",
            after=_regressed(),
        ),
        str(tmp_path),
    )

    status = enforce_status(str(tmp_path))

    assert status["iterations"] == 2
    assert status["committed"] == 1
    assert status["reverted"] == 1
    assert status["gaming_detected"] == 1


def test_enforce_reset_removes_session_directory(tmp_path: Path) -> None:
    _save_session(_config(tmp_path), _baseline(), str(tmp_path))

    assert enforce_reset(str(tmp_path)) is True
    assert not _session_dir(str(tmp_path)).exists()
    assert enforce_reset(str(tmp_path)) is False


def test_enforce_init_creates_session_and_uses_baseline_command(tmp_path: Path) -> None:
    calls: list[tuple[str, str | None]] = []
    config = _config(
        tmp_path,
        test_command="python -m pytest tests -q",
        baseline_command="python -m pytest tests/smoke -q",
    )

    def fake_runner(cmd: str, cwd: str | None):
        calls.append((cmd, cwd))
        return _baseline()

    baseline, session_dir = enforce_init(config, _run_tests_fn=fake_runner)

    assert baseline.failed == 2
    assert session_dir.exists()
    assert (session_dir / "session.json").exists()
    assert (session_dir / "baseline.json").exists()
    assert calls == [("python -m pytest tests/smoke -q", str(tmp_path))]


def test_enforce_init_stashes_when_requested(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path, stash_first=True)
    calls = {"stash": 0}

    monkeypatch.setattr("agent_xray.enforce._git_has_uncommitted_changes", lambda cwd=None: True)
    monkeypatch.setattr("agent_xray.enforce._git_stash", lambda cwd=None: calls.__setitem__("stash", calls["stash"] + 1) or True)
    monkeypatch.setattr("agent_xray.enforce._git_head_hash", lambda cwd=None: "head123")

    _, session_dir = enforce_init(config, _run_tests_fn=lambda cmd, cwd: _baseline())
    session_data = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))

    assert calls["stash"] == 1
    assert session_data["stash_saved"] is True


def test_enforce_check_returns_no_changes_without_running_tests(tmp_path: Path) -> None:
    _save_session(_config(tmp_path), _baseline(), str(tmp_path))

    def fail_runner(cmd: str, cwd: str | None):
        raise AssertionError("tests should not run when there is no diff")

    record = enforce_check(
        project_root=str(tmp_path),
        _run_tests_fn=fail_runner,
        _git_diff_fn=lambda cwd: "",
        _git_names_fn=lambda cwd: [],
        _git_diff_content_fn=lambda cwd: "",
    )

    assert record.decision == "NO_CHANGES"
    assert record.files_modified == []
    assert record.after is None


def test_enforce_check_recommends_commit_and_parses_diff_hunks(tmp_path: Path) -> None:
    _save_session(_config(tmp_path), _baseline(), str(tmp_path))
    enforce_plan("fix alpha", ["test_alpha"], project_root=str(tmp_path))

    record = enforce_check("fix alpha", project_root=str(tmp_path), **_default_check_kwargs())

    assert record.decision == "RECOMMEND_COMMIT"
    assert record.tests_improved == ["tests/test_app.py::test_alpha"]
    assert record.net_improvement == 1
    assert record.recommended_action == "commit"
    assert record.prediction_accuracy == "accurate"
    assert record.prediction_match_pct == 100.0
    assert record.diff_hunks[0]["file"] == "src/app.py"
    assert record.meta_analysis["classification"] in {"root_cause_fix", "symptom_patch"}
    assert "Recommend: commit." in record.review_summary


def test_enforce_check_uses_previous_iteration_after_as_before(tmp_path: Path) -> None:
    _save_session(_config(tmp_path), _baseline(), str(tmp_path))
    _save_iteration(_record(1, after=_improved()), str(tmp_path))

    record = enforce_check(
        "finish fix",
        project_root=str(tmp_path),
        **_default_check_kwargs(after_result=_all_green()),
    )

    assert record.iteration == 2
    assert record.before is not None
    assert record.before.failed == 1
    assert record.after is not None
    assert record.after.failed == 0


def test_enforce_check_recommends_investigate_for_suspicious_but_improving_change(tmp_path: Path) -> None:
    _save_session(_config(tmp_path), _baseline(), str(tmp_path))

    record = enforce_check(
        "suspicious fix",
        project_root=str(tmp_path),
        **_default_check_kwargs(
            _audit_fn=lambda diff, files_modified, allow_test_modification: (
                "SUSPICIOUS",
                ["mock_stub_insertion: Mock/stub added"],
                ["mock_stub_insertion"],
            ),
        ),
    )

    assert record.decision == "RECOMMEND_COMMIT"
    assert record.audit_verdict == "SUSPICIOUS"
    assert record.recommended_action == "investigate"


def test_enforce_check_rejects_change_that_exceeds_size_limits(tmp_path: Path) -> None:
    config = _config(tmp_path, max_files_per_change=1, max_diff_lines=1)
    _save_session(config, _baseline(), str(tmp_path))

    record = enforce_check(
        "too large",
        project_root=str(tmp_path),
        **_default_check_kwargs(
            _git_names_fn=lambda cwd: ["src/app.py", "src/other.py"],
            _git_diff_content_fn=lambda cwd: _diff(after_lines=["    x = 1", "    y = 2"]),
        ),
    )

    assert record.decision == "REJECTED"
    assert record.recommended_action == "split"
    assert any("Change too large" in reason for reason in record.audit_reasons)


def test_enforce_check_reverts_on_regression_and_sets_root_cause(tmp_path: Path) -> None:
    _save_session(_config(tmp_path), _baseline(), str(tmp_path))

    record = enforce_check(
        "bad fix",
        project_root=str(tmp_path),
        **_default_check_kwargs(
            after_result=_regressed(),
            _git_diff_content_fn=lambda cwd: _diff(
                after_lines=["import sys"],
                before_line="import os",
            ),
        ),
    )

    assert record.decision == "RECOMMEND_REVERT"
    assert record.recommended_action == "revert"
    assert record.regression_root_cause != ""
    assert "import" in record.regression_root_cause.lower()


def test_enforce_check_reverts_mixed_signal_change_with_investigate_recommendation(tmp_path: Path) -> None:
    _save_session(_config(tmp_path), _baseline(), str(tmp_path))

    record = enforce_check(
        "mixed result",
        project_root=str(tmp_path),
        **_default_check_kwargs(after_result=_mixed()),
    )

    assert record.tests_improved == ["tests/test_app.py::test_alpha"]
    assert record.tests_regressed == ["tests/test_app.py::test_zeta"]
    assert record.decision == "RECOMMEND_REVERT"
    assert record.recommended_action == "investigate"


def test_enforce_check_marks_rule_violations_as_gaming(tmp_path: Path) -> None:
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps({"forbidden_patterns": [r"print\("]}), encoding="utf-8")
    _save_session(_config(tmp_path, rules_file=str(rules_path)), _baseline(), str(tmp_path))

    record = enforce_check(
        "debug print",
        project_root=str(tmp_path),
        **_default_check_kwargs(
            _git_diff_content_fn=lambda cwd: _diff(after_lines=['    print("debug")']),
        ),
    )

    assert record.audit_verdict == "GAMING"
    assert record.decision == "RECOMMEND_REVERT"
    assert record.rule_violations
    assert "print" in record.rule_violations[0]


@pytest.mark.parametrize(
    ("name", "detector", "payload"),
    [
        (
            "test_modification",
            lambda data: detect_test_file_modification(data),
            ["tests/test_app.py"],
        ),
        (
            "hardcoded_values",
            detect_hardcoded_values,
            "+    return 42\n",
        ),
        (
            "special_case",
            detect_special_case_branching,
            "+    if test_mode:\n+        return cached\n",
        ),
        (
            "mock_insertion",
            detect_mock_insertion,
            "+    mock = Mock()\n",
        ),
        (
            "assertion_weakening",
            detect_assertion_weakening,
            "-    assert result == expected\n+    pass\n",
        ),
        (
            "exception_swallowing",
            detect_exception_swallowing,
            "+    except Exception:\n+        pass\n",
        ),
        (
            "early_return",
            detect_early_return,
            "+    return None\n",
        ),
        (
            "import_removal",
            detect_import_removal,
            "-import os\n-from pathlib import Path\n",
        ),
    ],
)
def test_gaming_detectors_all_fire(
    name: str,
    detector,
    payload: str | list[str],
) -> None:
    signal = detector(payload)

    assert signal is not None, name
    assert signal.description
    assert signal.confidence > 0.0


def test_audit_change_combines_multiple_gaming_signals() -> None:
    verdict, reasons, signals = audit_change(
        diff="+    except Exception: pass\n+    return 42\n",
        files_modified=["tests/test_app.py"],
    )

    assert verdict == "GAMING"
    assert "test_file_modification" in signals
    assert "hardcoded_value_insertion" in signals
    assert "exception_swallowing" in signals
    assert reasons


def test_enforce_challenge_reviews_only_new_iterations(tmp_path: Path) -> None:
    _save_session(_config(tmp_path), _baseline(), str(tmp_path))
    _save_iteration(_record(1), str(tmp_path))
    _save_iteration(_record(2), str(tmp_path))

    first = enforce_challenge(str(tmp_path))

    _save_iteration(_record(3, files_modified=["src/shared.py"]), str(tmp_path))
    second = enforce_challenge(str(tmp_path))

    assert first.iteration_range == (1, 2)
    assert second.iteration_range == (3, 3)
    assert len(_load_challenges(str(tmp_path))) == 2


def test_challenge_iterations_flags_hot_files_and_vetoes_gaming() -> None:
    result = challenge_iterations(
        [
            _record(1, files_modified=["src/shared.py"], audit_verdict="SUSPICIOUS"),
            _record(2, files_modified=["src/shared.py"], audit_verdict="GAMING", decision="RECOMMEND_COMMIT"),
            _record(3, files_modified=["src/shared.py"], audit_verdict="VALID"),
        ]
    )

    assert 2 in result.vetoed
    assert any("consecutive suspicious" in finding.lower() for finding in result.findings)
    assert any("hot files" in finding.lower() for finding in result.findings)


def test_build_enforce_report_aggregates_counts_and_maps(tmp_path: Path) -> None:
    _save_session(_config(tmp_path), _baseline(), str(tmp_path))
    _save_iteration(_record(1, files_modified=["src/app.py", "src/shared.py"]), str(tmp_path))
    _save_iteration(
        _record(
            2,
            files_modified=["src/shared.py"],
            decision="RECOMMEND_REVERT",
            audit_verdict="GAMING",
            after=_regressed(),
            prediction_accuracy="wrong",
            predicted_tests=["test_beta"],
            actual_improved=[],
            prediction_match_pct=0.0,
        ),
        str(tmp_path),
    )
    _save_challenge(
        ChallengeResult(iteration_range=(1, 2), changes_reviewed=2, vetoed=[2], findings=["review"]),
        1,
        str(tmp_path),
    )

    report = build_enforce_report(str(tmp_path))

    assert report.total_iterations == 2
    assert report.committed_count == 1
    assert report.reverted_count == 1
    assert report.gaming_detected_count == 1
    assert report.change_map["src/shared.py"] == 2
    assert report.prediction_accuracy_summary["accurate"] == 1
    assert report.prediction_accuracy_summary["wrong"] == 1


def test_format_enforce_text_renders_summary_iterations_and_challenges(tmp_path: Path) -> None:
    _save_session(_config(tmp_path), _baseline(), str(tmp_path))
    _save_iteration(_record(1), str(tmp_path))
    _save_challenge(
        ChallengeResult(iteration_range=(1, 1), changes_reviewed=1, findings=["No issues"]),
        1,
        str(tmp_path),
    )

    text = format_enforce_text(build_enforce_report(str(tmp_path)), color=False)

    assert "ENFORCEMENT REPORT" in text
    assert "Iteration 1" in text
    assert "Adversarial Challenges" in text
    assert "Change Map" in text


def test_format_enforce_json_renders_grade_timeline_and_change_map(tmp_path: Path) -> None:
    _save_session(_config(tmp_path), _baseline(), str(tmp_path))
    _save_iteration(_record(1, commit_hash="abcdef1"), str(tmp_path))

    payload = json.loads(format_enforce_json(build_enforce_report(str(tmp_path))))

    assert "grade" in payload
    assert payload["timeline"][0]["number"] == 1
    assert payload["timeline"][0]["commit_hash"] == "abcdef1"
    assert payload["detailed_change_map"][0]["file"] == "src/app.py"


def test_format_enforce_markdown_renders_sections_and_diff_detail(tmp_path: Path) -> None:
    _save_session(_config(tmp_path), _baseline(), str(tmp_path))
    _save_iteration(_record(1), str(tmp_path))
    _save_challenge(
        ChallengeResult(iteration_range=(1, 1), changes_reviewed=1, findings=["Clean"]),
        1,
        str(tmp_path),
    )

    markdown = format_enforce_markdown(build_enforce_report(str(tmp_path)))

    assert "# Enforcement Report" in markdown
    assert "## Summary" in markdown
    assert "### Iteration 1 - RECOMMEND_COMMIT" in markdown
    assert "```diff" in markdown
    assert "## Adversarial Challenges" in markdown


@pytest.mark.parametrize(
    ("fmt", "needle"),
    [
        ("text", "ENFORCEMENT REPORT"),
        ("json", '"config"'),
        ("markdown", "# Enforcement Report"),
    ],
)
def test_generate_report_supports_text_json_and_markdown(
    tmp_path: Path,
    fmt: str,
    needle: str,
) -> None:
    _save_session(_config(tmp_path), _baseline(), str(tmp_path))
    _save_iteration(_record(1), str(tmp_path))

    rendered = generate_report(str(tmp_path), format=fmt, color=False)

    assert needle in rendered
