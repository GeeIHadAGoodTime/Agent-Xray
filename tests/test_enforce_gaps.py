"""Tests for all 13 enforce mode gaps.

GAP 1:  Autonomous loop (enforce_auto)
GAP 2:  Meta-analysis of successful changes
GAP 3:  Project-rule audit awareness
GAP 4:  Line-item detail (diff hunks)
GAP 5:  Good move assessment (classify_diff_quality)
GAP 6:  Real adversarial challenge (cross-iteration analysis)
GAP 7:  Change-size enforcement
GAP 8:  Pre-change hypothesis requirement (enforce_plan)
GAP 9:  Predicted-vs-actual comparison
GAP 10: Advisory not enforced (enforce_guard)
GAP 11: Integration with grading (optional, tested via report)
GAP 12: Regression root cause
GAP 13: Cumulative report improvements
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_xray.enforce import (
    ChangeRecord,
    ChallengeResult,
    DiffHunk,
    EnforceConfig,
    EnforceReport,
    TestResult,
    _evaluate_prediction,
    _format_agent_command,
    _heuristic_regression_cause,
    _load_plan,
    _load_project_rules,
    _meta_analyze,
    _parse_diff_hunks,
    _save_session,
    _save_iteration,
    _session_dir,
    build_enforce_report,
    enforce_auto,
    enforce_check,
    enforce_guard,
    enforce_plan,
    enforce_status,
)
from agent_xray.enforce_audit import (
    GamingSignal,
    challenge_iterations,
    classify_diff_quality,
    detect_rule_violations,
    _group_diff_by_file,
)
from agent_xray.enforce_report import (
    format_enforce_json,
    format_enforce_markdown,
    format_enforce_text,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _baseline() -> TestResult:
    return TestResult(
        exit_code=1, passed=8, failed=2, errors=0, skipped=0,
        total=10, duration_seconds=3.5, output="8 passed, 2 failed in 3.50s",
        test_names_passed=["test_a", "test_b", "test_c", "test_d",
                           "test_e", "test_f", "test_g", "test_h"],
        test_names_failed=["test_x", "test_y"],
    )


def _improved() -> TestResult:
    return TestResult(
        exit_code=1, passed=9, failed=1, errors=0, skipped=0,
        total=10, duration_seconds=3.2, output="9 passed, 1 failed in 3.20s",
        test_names_passed=["test_a", "test_b", "test_c", "test_d",
                           "test_e", "test_f", "test_g", "test_h", "test_x"],
        test_names_failed=["test_y"],
    )


def _all_pass() -> TestResult:
    return TestResult(
        exit_code=0, passed=10, failed=0, errors=0, skipped=0,
        total=10, duration_seconds=2.0, output="10 passed in 2.00s",
        test_names_passed=["test_a", "test_b", "test_c", "test_d",
                           "test_e", "test_f", "test_g", "test_h",
                           "test_x", "test_y"],
        test_names_failed=[],
    )


def _regressed() -> TestResult:
    return TestResult(
        exit_code=1, passed=7, failed=3, errors=0, skipped=0,
        total=10, duration_seconds=3.8, output="7 passed, 3 failed in 3.80s",
        test_names_passed=["test_a", "test_b", "test_c", "test_d",
                           "test_e", "test_f", "test_g"],
        test_names_failed=["test_h", "test_x", "test_y"],
    )


def _config(tmp_path: Path) -> EnforceConfig:
    return EnforceConfig(
        test_command="echo '8 passed, 2 failed in 3.50s'",
        project_root=str(tmp_path),
        max_iterations=10,
    )


def _make_check_stubs(**overrides):
    """Create a dict of mock functions for enforce_check."""
    defaults = dict(
        _run_tests_fn=lambda cmd, cwd: _improved(),
        _audit_fn=lambda diff, files_modified, allow_test_modification: ("VALID", [], []),
        _git_diff_fn=lambda cwd: "1 file changed",
        _git_names_fn=lambda cwd: ["src/main.py"],
        _git_commit_fn=lambda msg, cwd: "abc123",
        _git_revert_fn=lambda h, cwd: True,
        _git_head_fn=lambda cwd: "def456",
        _git_diff_content_fn=lambda cwd: "+    result = compute()\n",
    )
    defaults.update(overrides)
    return defaults


SAMPLE_DIFF = """\
diff --git a/src/main.py b/src/main.py
--- a/src/main.py
+++ b/src/main.py
@@ -45,3 +45,3 @@ def process():
-    threshold = 5
+    threshold = 3
     return compute(threshold)
@@ -100,2 +100,4 @@ def validate():
+    if value is None:
+        return False
     return True
"""


# ===========================================================================
# GAP 1: Autonomous loop (enforce_auto)
# ===========================================================================

class TestEnforceAuto:
    def test_auto_basic_loop(self, tmp_path: Path):
        """Auto loop should init, run iterations, and produce a report."""
        call_count = {"agent": 0, "tests": 0}

        def mock_tests(cmd, cwd):
            call_count["tests"] += 1
            if call_count["tests"] >= 3:
                return _all_pass()
            return _baseline()

        def mock_shell(cmd, cwd):
            call_count["agent"] += 1
            return 0, "Agent made changes"

        config = EnforceConfig(
            test_command="test",
            project_root=str(tmp_path),
            max_iterations=10,
            challenge_every=0,
        )
        report = enforce_auto(
            config, "agent-cmd",
            _run_tests_fn=mock_tests,
            _run_shell_fn=mock_shell,
        )
        assert isinstance(report, EnforceReport)
        assert report.total_iterations >= 1
        assert call_count["agent"] >= 1

    def test_auto_stops_when_all_pass(self, tmp_path: Path):
        """Auto loop should stop early when all tests pass."""
        def mock_tests(cmd, cwd):
            return _all_pass()

        config = EnforceConfig(
            test_command="test",
            project_root=str(tmp_path),
            max_iterations=50,
        )
        report = enforce_auto(
            config, "agent-cmd",
            _run_tests_fn=mock_tests,
            _run_shell_fn=lambda cmd, cwd: (0, "ok"),
        )
        # Should stop immediately since baseline already passes
        assert report.total_iterations == 0

    def test_auto_respects_max_iterations(self, tmp_path: Path):
        """Auto loop should not exceed max_iterations."""
        config = EnforceConfig(
            test_command="test",
            project_root=str(tmp_path),
            max_iterations=3,
            challenge_every=0,
        )
        report = enforce_auto(
            config, "agent-cmd",
            _run_tests_fn=lambda cmd, cwd: _baseline(),
            _run_shell_fn=lambda cmd, cwd: (0, "ok"),
        )
        assert report.total_iterations <= 3

    def test_auto_uses_configured_test_timeout(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        timeouts: list[int] = []

        def mock_run_tests(cmd, cwd, *, timeout=120):
            timeouts.append(timeout)
            return _all_pass()

        monkeypatch.setattr("agent_xray.enforce.run_tests", mock_run_tests)

        config = EnforceConfig(
            test_command="test",
            project_root=str(tmp_path),
            max_iterations=1,
            challenge_every=0,
            test_timeout=17,
        )
        report = enforce_auto(
            config,
            "agent-cmd",
            _run_shell_fn=lambda cmd, cwd: (0, "ok"),
        )
        assert report.total_iterations == 0
        assert timeouts == [17]

    def test_agent_command_templates_use_baseline_context_first(self, tmp_path: Path):
        commands: list[str] = []
        test_calls = {"count": 0}

        def mock_tests(cmd, cwd):
            test_calls["count"] += 1
            if test_calls["count"] == 1:
                return TestResult(
                    exit_code=1,
                    passed=3,
                    failed=2,
                    errors=0,
                    skipped=0,
                    total=5,
                    duration_seconds=1.0,
                    output="Assertion 'boom'\nline two",
                    test_names_failed=[
                        "tests/test_alpha.py::test_one",
                        "tests/test_beta.py::test_two",
                    ],
                )
            return _all_pass()

        def mock_shell(cmd, cwd):
            commands.append(cmd)
            return 0, "ok"

        config = EnforceConfig(
            test_command="test",
            project_root=str(tmp_path),
            max_iterations=1,
            challenge_every=0,
        )
        report = enforce_auto(
            config,
            "agent --msg '{iteration}|{fail_count}|{pass_count}|{total_count}|{failing_tests}|{last_error}|{hypothesis}'",
            _run_tests_fn=mock_tests,
            _run_shell_fn=mock_shell,
        )
        assert report.total_iterations == 1
        assert len(commands) == 1
        assert "1|2|3|5|" in commands[0]
        assert "tests/test_alpha.py::test_one, tests/test_beta.py::test_two" in commands[0]
        assert "Assertion \\'boom\\'\\nline two" in commands[0]
        assert commands[0].endswith("|'")

    def test_agent_command_templates_use_latest_iteration_context(self, tmp_path: Path):
        commands: list[str] = []
        test_calls = {"count": 0}

        def mock_tests(cmd, cwd):
            test_calls["count"] += 1
            if test_calls["count"] == 1:
                return TestResult(
                    exit_code=1,
                    passed=4,
                    failed=2,
                    errors=0,
                    skipped=0,
                    total=6,
                    duration_seconds=1.0,
                    output="first failure",
                    test_names_failed=[
                        "tests/test_alpha.py::test_one",
                        "tests/test_beta.py::test_two",
                    ],
                )
            if test_calls["count"] == 2:
                return TestResult(
                    exit_code=1,
                    passed=5,
                    failed=1,
                    errors=0,
                    skipped=0,
                    total=6,
                    duration_seconds=1.0,
                    output="second failure",
                    test_names_failed=["tests/test_beta.py::test_two"],
                )
            return _all_pass()

        def mock_shell(cmd, cwd):
            commands.append(cmd)
            return 0, "ok"

        config = EnforceConfig(
            test_command="test",
            project_root=str(tmp_path),
            max_iterations=2,
            challenge_every=0,
        )
        enforce_auto(
            config,
            "agent '{iteration}|{failing_tests}|{fail_count}|{hypothesis}|{last_error}'",
            _run_tests_fn=mock_tests,
            _run_shell_fn=mock_shell,
        )
        assert len(commands) == 2
        assert "1|tests/test_alpha.py::test_one, tests/test_beta.py::test_two|2||first failure" in commands[0]
        assert "2|tests/test_beta.py::test_two|1|auto iteration 1|second failure" in commands[1]


class TestAutoTemplateFormatting:
    def test_unknown_placeholders_pass_through(self):
        result = _format_agent_command(
            "agent '{unknown_key}' '{iteration}'",
            _baseline(),
            3,
            "hypothesis",
        )
        assert result == "agent '{unknown_key}' '3'"


# ===========================================================================
# GAP 2: Meta-analysis of successful changes
# ===========================================================================

class TestMetaAnalysis:
    def test_basic_meta_analysis(self):
        before = _baseline()
        after = _improved()
        diff = "+    result = compute()\n-    old = bad()\n"
        files = ["src/main.py"]
        meta = _meta_analyze(before, after, diff, files)
        assert "classification" in meta
        assert "localized" in meta
        assert "additive" in meta
        assert "tests_fixed" in meta
        assert meta["localized"] is True

    def test_meta_analysis_root_cause_fix(self):
        before = _baseline()
        after = _improved()
        diff = "-    return bad\n+    return compute(data)\n"
        files = ["src/main.py"]
        meta = _meta_analyze(before, after, diff, files)
        assert meta["classification"] == "root_cause_fix"

    def test_meta_analysis_test_update(self):
        before = _baseline()
        after = _improved()
        diff = "+    assert result == expected\n"
        files = ["tests/test_main.py"]
        meta = _meta_analyze(before, after, diff, files)
        assert meta["classification"] == "test_update"

    def test_meta_analysis_config_change(self):
        before = _baseline()
        after = _improved()
        diff = "+    timeout = 30\n"
        files = ["config/settings.json"]
        meta = _meta_analyze(before, after, diff, files)
        assert meta["classification"] == "configuration_change"

    def test_meta_analysis_scattered(self):
        before = _baseline()
        after = _improved()
        diff = "+    fix\n"
        files = ["src/a.py", "src/b.py", "src/c.py"]
        meta = _meta_analyze(before, after, diff, files)
        assert meta["localized"] is False

    def test_meta_analysis_additive(self):
        diff = "+line1\n+line2\n+line3\n-old\n"
        meta = _meta_analyze(_baseline(), _improved(), diff, ["src/a.py"])
        assert meta["additive"] is True
        assert meta["lines_added"] == 3
        assert meta["lines_removed"] == 1

    def test_meta_in_change_record(self, tmp_path: Path):
        """Committed changes should have meta_analysis populated."""
        cfg = _config(tmp_path)
        _save_session(cfg, _baseline(), str(tmp_path))
        stubs = _make_check_stubs()
        record = enforce_check("fix test_x", project_root=str(tmp_path), **stubs)
        assert record.decision == "RECOMMEND_COMMIT"
        assert record.meta_analysis != {}
        assert "classification" in record.meta_analysis


# ===========================================================================
# GAP 3: Project-rule audit awareness
# ===========================================================================

class TestProjectRules:
    def test_detect_forbidden_pattern(self):
        rules = {"forbidden_patterns": [r"os\.getenv"]}
        diff = '+    value = os.getenv("KEY")\n'
        sig = detect_rule_violations(diff, ["src/main.py"], rules)
        assert sig is not None
        assert sig.name == "rule_violation"
        assert "os.getenv" in sig.description.lower() or "forbidden" in sig.description.lower()

    def test_detect_forbidden_file(self):
        rules = {"forbidden_file_patterns": [r"\.env"]}
        sig = detect_rule_violations("", [".env", "src/main.py"], rules)
        assert sig is not None
        assert "forbidden file" in sig.description.lower()

    def test_detect_max_files_exceeded(self):
        rules = {"max_files_per_change": 2}
        sig = detect_rule_violations("", ["a.py", "b.py", "c.py"], rules)
        assert sig is not None
        assert "too many" in sig.description.lower()

    def test_no_violations(self):
        rules = {"forbidden_patterns": [r"os\.getenv"], "max_files_per_change": 10}
        diff = "+    result = compute()\n"
        sig = detect_rule_violations(diff, ["src/main.py"], rules)
        assert sig is None

    def test_empty_rules(self):
        sig = detect_rule_violations("+code\n", ["src/main.py"], {})
        assert sig is None

    def test_load_project_rules_file(self, tmp_path: Path):
        rules = {"forbidden_patterns": ["print\\("]}
        rules_path = tmp_path / "rules.json"
        rules_path.write_text(json.dumps(rules))
        loaded = _load_project_rules(str(rules_path))
        assert loaded is not None
        assert "forbidden_patterns" in loaded

    def test_load_project_rules_missing(self):
        loaded = _load_project_rules("/nonexistent/rules.json")
        assert loaded is None

    def test_load_project_rules_none(self):
        loaded = _load_project_rules(None)
        assert loaded is None

    def test_rules_in_enforce_check(self, tmp_path: Path):
        """Rules file violations should show in audit results."""
        rules = {"forbidden_patterns": [r"print\("]}
        rules_path = tmp_path / "rules.json"
        rules_path.write_text(json.dumps(rules))

        cfg = EnforceConfig(
            test_command="test",
            project_root=str(tmp_path),
            rules_file=str(rules_path),
        )
        _save_session(cfg, _baseline(), str(tmp_path))

        stubs = _make_check_stubs(
            _git_diff_content_fn=lambda cwd: '+    print("debug")\n',
        )
        record = enforce_check("add debug", project_root=str(tmp_path), **stubs)
        assert len(record.rule_violations) > 0


# ===========================================================================
# GAP 4: Line-item diff detail (_parse_diff_hunks)
# ===========================================================================

class TestParseDiffHunks:
    def test_basic_hunk(self):
        hunks = _parse_diff_hunks(SAMPLE_DIFF)
        assert len(hunks) >= 1
        assert hunks[0].file == "src/main.py"
        assert hunks[0].line_number > 0

    def test_hunk_has_added_and_removed(self):
        hunks = _parse_diff_hunks(SAMPLE_DIFF)
        first = hunks[0]
        assert len(first.removed_lines) > 0
        assert len(first.added_lines) > 0

    def test_multiple_hunks(self):
        hunks = _parse_diff_hunks(SAMPLE_DIFF)
        assert len(hunks) == 2  # Two @@ sections

    def test_empty_diff(self):
        hunks = _parse_diff_hunks("")
        assert hunks == []

    def test_no_changes_diff(self):
        hunks = _parse_diff_hunks("diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n")
        assert hunks == []

    def test_diff_hunk_round_trip(self):
        hunk = DiffHunk(file="a.py", line_number=10, removed_lines=["L10: -old"], added_lines=["L10: +new"])
        d = hunk.to_dict()
        h2 = DiffHunk.from_dict(d)
        assert h2.file == "a.py"
        assert h2.line_number == 10
        assert h2.removed_lines == ["L10: -old"]

    def test_hunks_in_change_record(self, tmp_path: Path):
        """enforce_check should populate diff_hunks."""
        cfg = _config(tmp_path)
        _save_session(cfg, _baseline(), str(tmp_path))
        stubs = _make_check_stubs(
            _git_diff_content_fn=lambda cwd: SAMPLE_DIFF,
        )
        record = enforce_check("fix", project_root=str(tmp_path), **stubs)
        assert len(record.diff_hunks) >= 1
        assert record.diff_hunks[0]["file"] == "src/main.py"

    def test_multi_file_diff(self):
        diff = """\
diff --git a/src/a.py b/src/a.py
--- a/src/a.py
+++ b/src/a.py
@@ -1,3 +1,3 @@
-old_a
+new_a
diff --git a/src/b.py b/src/b.py
--- a/src/b.py
+++ b/src/b.py
@@ -5,2 +5,2 @@
-old_b
+new_b
"""
        hunks = _parse_diff_hunks(diff)
        assert len(hunks) == 2
        files = {h.file for h in hunks}
        assert "src/a.py" in files
        assert "src/b.py" in files


# ===========================================================================
# GAP 5: Good move assessment (classify_diff_quality)
# ===========================================================================

class TestClassifyChangeQuality:
    def test_behavioral_improvement(self):
        diff = "+    threshold = 3\n-    threshold = 5\n"
        result = classify_diff_quality(diff, ["src/config.py"], 0)
        assert result == "behavioral_improvement"

    def test_bug_fix(self):
        diff = "+    if value is None:\n+        return False\n"
        result = classify_diff_quality(diff, ["src/main.py"], 1)
        assert result == "bug_fix"

    def test_test_improvement(self):
        diff = "+    assert result == expected\n+    assert other == value\n"
        result = classify_diff_quality(diff, ["tests/test_main.py"], 1)
        assert result == "test_improvement"

    def test_neutral(self):
        result = classify_diff_quality("", [], 0)
        assert result == "neutral"

    def test_refactor(self):
        diff = "\n".join(
            [f"+    new_line_{i}" for i in range(8)]
            + [f"-    old_line_{i}" for i in range(8)]
        )
        result = classify_diff_quality(diff, ["src/main.py"], 0)
        assert result == "refactor"

    def test_quality_in_record(self, tmp_path: Path):
        cfg = _config(tmp_path)
        _save_session(cfg, _baseline(), str(tmp_path))
        stubs = _make_check_stubs()
        record = enforce_check("fix", project_root=str(tmp_path), **stubs)
        assert record.change_quality != ""


# ===========================================================================
# GAP 6: Real adversarial challenge (cross-iteration analysis)
# ===========================================================================

class TestAdversarialChallengeEnhanced:
    def _make_iter(self, num, *, verdict="VALID", decision="COMMITTED",
                   files=None, improved=None, regressed=None,
                   before_passed=None, after_passed=None,
                   before_failed=None, after_failed=None):
        before = TestResult(
            exit_code=1, passed=before_passed or 5, failed=before_failed or 3,
            errors=0, skipped=0, total=8, duration_seconds=1.0, output="",
            test_names_passed=[], test_names_failed=[],
        )
        after = TestResult(
            exit_code=1, passed=after_passed or 6, failed=after_failed or 2,
            errors=0, skipped=0, total=8, duration_seconds=1.0, output="",
            test_names_passed=[], test_names_failed=[],
        )
        return ChangeRecord(
            iteration=num,
            files_modified=files or ["src/main.py"],
            audit_verdict=verdict,
            decision=decision,
            before=before, after=after,
            tests_improved=improved or [],
            tests_regressed=regressed or [],
            diff_stat="1 file changed" if decision == "COMMITTED" else "",
        )

    def test_detects_test_flip_flop(self):
        iters = [
            self._make_iter(1, improved=["test_a"]),
            self._make_iter(2),
            self._make_iter(3, regressed=["test_a"]),
        ]
        result = challenge_iterations(iters)
        assert any("flip-flop" in f.lower() for f in result.findings)

    def test_detects_dependency_risk(self):
        iters = [
            self._make_iter(1, files=["src/core.py"]),
            self._make_iter(2, files=["src/other.py"]),
            self._make_iter(3, files=["src/core.py"]),
        ]
        result = challenge_iterations(iters)
        assert any("dependency" in f.lower() or "risk" in f.lower() for f in result.findings)

    def test_detects_coverage_gap(self):
        after = TestResult(
            exit_code=1, passed=5, failed=3, errors=0, skipped=0,
            total=8, duration_seconds=1.0, output="",
            test_names_passed=[], test_names_failed=["tests/other/test_x.py::test_fail"],
        )
        iters = [ChangeRecord(
            iteration=1, files_modified=["src/core/main.py"],
            audit_verdict="VALID", decision="COMMITTED",
            before=_baseline(), after=after,
            tests_improved=[], tests_regressed=[],
        )]
        result = challenge_iterations(iters)
        assert any("coverage" in f.lower() or "gap" in f.lower() for f in result.findings)


# ===========================================================================
# GAP 7: Change-size enforcement
# ===========================================================================

class TestChangeSizeEnforcement:
    def test_reject_too_many_files(self, tmp_path: Path):
        cfg = EnforceConfig(
            test_command="test",
            project_root=str(tmp_path),
            max_files_per_change=2,
        )
        _save_session(cfg, _baseline(), str(tmp_path))

        stubs = _make_check_stubs(
            _git_names_fn=lambda cwd: ["a.py", "b.py", "c.py"],
        )
        record = enforce_check("big change", project_root=str(tmp_path), **stubs)
        assert record.decision == "REJECTED"
        assert any("too large" in r.lower() for r in record.audit_reasons)

    def test_reject_too_many_diff_lines(self, tmp_path: Path):
        cfg = EnforceConfig(
            test_command="test",
            project_root=str(tmp_path),
            max_diff_lines=5,
        )
        _save_session(cfg, _baseline(), str(tmp_path))

        big_diff = "\n".join([f"+line{i}" for i in range(20)])
        stubs = _make_check_stubs(
            _git_diff_content_fn=lambda cwd: big_diff,
        )
        record = enforce_check("big diff", project_root=str(tmp_path), **stubs)
        assert record.decision == "REJECTED"

    def test_rejected_changes_include_actionable_guidance(self, tmp_path: Path):
        cfg = EnforceConfig(
            test_command="test",
            project_root=str(tmp_path),
            max_files_per_change=1,
        )
        _save_session(cfg, _baseline(), str(tmp_path))

        stubs = _make_check_stubs(
            _git_names_fn=lambda cwd: ["a.py", "b.py"],
        )
        record = enforce_check("split me", project_root=str(tmp_path), **stubs)
        assert record.decision == "REJECTED"
        assert any("split this change" in reason.lower() for reason in record.audit_reasons)

    def test_accepts_within_limits(self, tmp_path: Path):
        cfg = EnforceConfig(
            test_command="test",
            project_root=str(tmp_path),
            max_files_per_change=10,
            max_diff_lines=500,
        )
        _save_session(cfg, _baseline(), str(tmp_path))

        stubs = _make_check_stubs()
        record = enforce_check("small change", project_root=str(tmp_path), **stubs)
        assert record.decision != "REJECTED"

    def test_config_defaults(self):
        cfg = EnforceConfig(test_command="test")
        assert cfg.max_files_per_change == 5
        assert cfg.max_diff_lines == 200

    def test_config_round_trip(self):
        cfg = EnforceConfig(test_command="test", max_files_per_change=3, max_diff_lines=100)
        d = cfg.to_dict()
        cfg2 = EnforceConfig.from_dict(d)
        assert cfg2.max_files_per_change == 3
        assert cfg2.max_diff_lines == 100


# ===========================================================================
# GAP 8: Pre-change hypothesis requirement (enforce_plan)
# ===========================================================================

class TestEnforcePlan:
    def test_plan_and_check(self, tmp_path: Path):
        cfg = _config(tmp_path)
        _save_session(cfg, _baseline(), str(tmp_path))

        result = enforce_plan(
            "Fix test_x by lowering threshold",
            ["test_x"],
            project_root=str(tmp_path),
        )
        assert result["status"] == "plan_registered"

        # Verify plan was saved
        plan = _load_plan(str(tmp_path))
        assert plan is not None
        assert plan["hypothesis"] == "Fix test_x by lowering threshold"
        assert plan["expected_tests"] == ["test_x"]

    def test_plan_requires_session(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            enforce_plan("fix", project_root=str(tmp_path))

    def test_plan_consumed_after_check(self, tmp_path: Path):
        cfg = _config(tmp_path)
        _save_session(cfg, _baseline(), str(tmp_path))

        enforce_plan("fix test_x", ["test_x"], project_root=str(tmp_path))
        stubs = _make_check_stubs()
        enforce_check("fix test_x", project_root=str(tmp_path), **stubs)

        plan = _load_plan(str(tmp_path))
        assert plan is None  # Plan should be consumed

    def test_plan_empty_expected(self, tmp_path: Path):
        cfg = _config(tmp_path)
        _save_session(cfg, _baseline(), str(tmp_path))
        result = enforce_plan("just refactoring", project_root=str(tmp_path))
        assert result["expected_tests"] == []


# ===========================================================================
# GAP 9: Predicted-vs-actual comparison
# ===========================================================================

class TestPredictionAccuracy:
    def test_accurate_prediction(self):
        plan = {"expected_tests": ["test_x"]}
        acc, predicted, actual, pct = _evaluate_prediction(plan, ["test_x"], [])
        assert acc == "accurate"
        assert pct == 100.0

    def test_partial_prediction(self):
        plan = {"expected_tests": ["test_x", "test_y"]}
        acc, predicted, actual, pct = _evaluate_prediction(plan, ["test_x"], [])
        assert acc == "partial"
        assert pct == 50.0

    def test_wrong_prediction(self):
        plan = {"expected_tests": ["test_x"]}
        acc, predicted, actual, pct = _evaluate_prediction(plan, ["test_z"], [])
        assert acc == "wrong"
        assert pct == 0.0

    def test_no_prediction(self):
        acc, predicted, actual, pct = _evaluate_prediction(None, ["test_x"], [])
        assert acc == "no_prediction"

    def test_empty_expected(self):
        plan = {"expected_tests": []}
        acc, _, _, _ = _evaluate_prediction(plan, ["test_x"], [])
        assert acc == "no_prediction"

    def test_prediction_in_record(self, tmp_path: Path):
        cfg = _config(tmp_path)
        _save_session(cfg, _baseline(), str(tmp_path))
        enforce_plan("fix test_x", ["test_x"], project_root=str(tmp_path))

        stubs = _make_check_stubs()
        record = enforce_check("fix test_x", project_root=str(tmp_path), **stubs)
        assert record.prediction_accuracy in ("accurate", "partial", "wrong", "no_prediction")
        assert isinstance(record.prediction_match_pct, float)
        assert isinstance(record.predicted_tests, list)

    def test_no_plan_warning(self, tmp_path: Path):
        """Check without plan should include a warning."""
        cfg = _config(tmp_path)
        _save_session(cfg, _baseline(), str(tmp_path))
        stubs = _make_check_stubs()
        record = enforce_check(project_root=str(tmp_path), **stubs)
        assert record.prediction_accuracy == "no_prediction"


# ===========================================================================
# GAP 10: enforce_guard
# ===========================================================================

class TestEnforceGuard:
    def test_guard_clean(self, tmp_path: Path):
        cfg = _config(tmp_path)
        _save_session(cfg, _baseline(), str(tmp_path))
        result = enforce_guard(
            project_root=str(tmp_path),
            _git_head_fn=lambda cwd: "abc123",
            _git_names_fn=lambda cwd: [],
        )
        assert result["status"] == "clean"
        assert len(result["warnings"]) == 0

    def test_guard_uncommitted(self, tmp_path: Path):
        cfg = _config(tmp_path)
        _save_session(cfg, _baseline(), str(tmp_path))
        result = enforce_guard(
            project_root=str(tmp_path),
            _git_head_fn=lambda cwd: "abc123",
            _git_names_fn=lambda cwd: ["untracked.py", "dirty.py"],
        )
        assert result["status"] == "warning"
        assert len(result["warnings"]) > 0
        assert "untracked.py" in result["uncommitted_files"]

    def test_guard_no_session(self, tmp_path: Path):
        result = enforce_guard(
            project_root=str(tmp_path),
            _git_head_fn=lambda cwd: "abc",
            _git_names_fn=lambda cwd: [],
        )
        assert result["status"] == "no_session"


# ===========================================================================
# GAP 12: Regression root cause
# ===========================================================================

class TestRegressionRootCause:
    def test_heuristic_import_removal(self):
        diff = "-import os\n-from pathlib import Path\n"
        cause = _heuristic_regression_cause(diff, ["test_x"], ["src/main.py"])
        assert "import" in cause.lower()

    def test_heuristic_function_sig_change(self):
        diff = "-def process(a, b):\n+def process(a, b, c=None):\n"
        cause = _heuristic_regression_cause(diff, ["test_x"], ["src/main.py"])
        assert "function" in cause.lower() or "signature" in cause.lower()

    def test_heuristic_config_change(self):
        diff = "+timeout = 5\n"
        cause = _heuristic_regression_cause(diff, ["test_x"], ["config/settings.py"])
        assert "config" in cause.lower() or "settings" in cause.lower()

    def test_heuristic_generic(self):
        diff = "+something = 1\n"
        cause = _heuristic_regression_cause(diff, ["test_x", "test_y"], ["src/main.py"])
        assert "2 test" in cause

    def test_no_regressions(self):
        cause = _heuristic_regression_cause("+code\n", [], ["src/main.py"])
        assert cause == ""

    def test_regression_cause_in_record(self, tmp_path: Path):
        """Reverted changes with regressions should have regression_root_cause."""
        cfg = EnforceConfig(
            test_command="test",
            project_root=str(tmp_path),
            require_improvement=True,
        )
        _save_session(cfg, _baseline(), str(tmp_path))

        stubs = _make_check_stubs(
            _run_tests_fn=lambda cmd, cwd: _regressed(),
            _git_diff_content_fn=lambda cwd: "-import os\n+import sys\n",
        )
        record = enforce_check("bad change", project_root=str(tmp_path), **stubs)
        assert record.decision == "RECOMMEND_REVERT"
        assert record.regression_root_cause != ""

    def test_require_improvement_reverts_neutral_results(self, tmp_path: Path):
        cfg = EnforceConfig(
            test_command="test",
            project_root=str(tmp_path),
            require_improvement=True,
        )
        _save_session(cfg, _baseline(), str(tmp_path))

        stubs = _make_check_stubs(
            _run_tests_fn=lambda cmd, cwd: _baseline(),
        )
        record = enforce_check("no-op change", project_root=str(tmp_path), **stubs)
        assert record.decision == "RECOMMEND_REVERT"
        assert record.recommended_action == "revert"


# ===========================================================================
# GAP 13: Cumulative report improvements
# ===========================================================================

class TestCumulativeReport:
    def _setup_session(self, tmp_path: Path, num_iters: int = 3):
        cfg = _config(tmp_path)
        _save_session(cfg, _baseline(), str(tmp_path))
        for i in range(1, num_iters + 1):
            rec = ChangeRecord(
                iteration=i,
                files_modified=[f"src/file{i}.py", "src/shared.py"],
                decision="COMMITTED" if i % 2 == 1 else "REVERTED",
                audit_verdict="VALID" if i % 2 == 1 else "GAMING",
                before=_baseline(),
                after=_improved() if i % 2 == 1 else _regressed(),
                diff_stat=f"{i} file changed" if i % 2 == 1 else "",
                prediction_accuracy="accurate" if i == 1 else ("wrong" if i == 2 else "partial"),
            )
            _save_iteration(rec, str(tmp_path))
        return tmp_path

    def test_prediction_accuracy_summary(self, tmp_path: Path):
        self._setup_session(tmp_path)
        report = build_enforce_report(str(tmp_path))
        pa = report.prediction_accuracy_summary
        assert "accurate" in pa
        assert "wrong" in pa
        assert "partial" in pa
        assert pa["total_with_prediction"] > 0

    def test_efficiency_ratio(self, tmp_path: Path):
        self._setup_session(tmp_path)
        report = build_enforce_report(str(tmp_path))
        assert 0.0 <= report.efficiency_ratio <= 1.0
        # 2 committed out of 3 = 0.667
        assert abs(report.efficiency_ratio - 0.667) < 0.01

    def test_change_map(self, tmp_path: Path):
        self._setup_session(tmp_path)
        report = build_enforce_report(str(tmp_path))
        assert "src/shared.py" in report.change_map
        assert report.change_map["src/shared.py"] == 3  # Modified in all 3 iterations

    def test_cumulative_diff(self, tmp_path: Path):
        self._setup_session(tmp_path)
        report = build_enforce_report(str(tmp_path))
        assert report.cumulative_diff != ""  # Should have committed diffs

    def test_rejected_count(self, tmp_path: Path):
        cfg = _config(tmp_path)
        _save_session(cfg, _baseline(), str(tmp_path))
        rec = ChangeRecord(
            iteration=1, decision="REJECTED",
            audit_verdict="VALID", before=_baseline(), after=_baseline(),
        )
        _save_iteration(rec, str(tmp_path))
        report = build_enforce_report(str(tmp_path))
        assert report.rejected_count == 1

    def test_text_report_efficiency(self, tmp_path: Path):
        self._setup_session(tmp_path)
        report = build_enforce_report(str(tmp_path))
        text = format_enforce_text(report, color=False)
        assert "Efficiency" in text

    def test_text_report_prediction(self, tmp_path: Path):
        self._setup_session(tmp_path)
        report = build_enforce_report(str(tmp_path))
        text = format_enforce_text(report, color=False)
        assert "Prediction" in text

    def test_text_report_change_map(self, tmp_path: Path):
        self._setup_session(tmp_path)
        report = build_enforce_report(str(tmp_path))
        text = format_enforce_text(report, color=False)
        assert "Change Map" in text

    def test_markdown_report_prediction(self, tmp_path: Path):
        self._setup_session(tmp_path)
        report = build_enforce_report(str(tmp_path))
        md = format_enforce_markdown(report)
        assert "Prediction Accuracy" in md

    def test_markdown_report_efficiency(self, tmp_path: Path):
        self._setup_session(tmp_path)
        report = build_enforce_report(str(tmp_path))
        md = format_enforce_markdown(report)
        assert "Efficiency" in md

    def test_markdown_report_change_map(self, tmp_path: Path):
        self._setup_session(tmp_path)
        report = build_enforce_report(str(tmp_path))
        md = format_enforce_markdown(report)
        assert "## Change Map" in md

    def test_json_report_new_fields(self, tmp_path: Path):
        self._setup_session(tmp_path)
        report = build_enforce_report(str(tmp_path))
        json_str = format_enforce_json(report)
        data = json.loads(json_str)
        assert "prediction_accuracy_summary" in data
        assert "efficiency_ratio" in data
        assert "change_map" in data
        assert "rejected_count" in data


# ===========================================================================
# GAP 4+5+9+12 in reports (rendering the new fields)
# ===========================================================================

class TestReportNewFields:
    def _make_report_with_all_fields(self):
        change = ChangeRecord(
            iteration=1,
            files_modified=["src/main.py"],
            hypothesis="Fix threshold",
            before=_baseline(),
            after=_improved(),
            tests_improved=["test_x"],
            net_improvement=1,
            audit_verdict="VALID",
            decision="COMMITTED",
            commit_hash="abc001",
            change_quality="bug_fix",
            prediction_accuracy="accurate",
            predicted_tests=["test_x"],
            actual_improved=["test_x"],
            prediction_match_pct=100.0,
            regression_root_cause="",
            meta_analysis={"classification": "root_cause_fix", "localized": True},
            diff_hunks=[{
                "file": "src/main.py",
                "line_number": 47,
                "removed_lines": ["L47: -    threshold = 5"],
                "added_lines": ["L47: +    threshold = 3"],
            }],
            rule_violations=[],
        )
        regressed_change = ChangeRecord(
            iteration=2,
            files_modified=["src/other.py"],
            hypothesis="Refactor imports",
            before=_improved(),
            after=_regressed(),
            tests_regressed=["test_h"],
            net_improvement=-1,
            audit_verdict="VALID",
            decision="REVERTED",
            regression_root_cause="removed import needed by test_h",
            rule_violations=["Forbidden pattern 'print(' found"],
        )
        return EnforceReport(
            config=EnforceConfig(test_command="pytest"),
            changes=[change, regressed_change],
            total_iterations=2,
            committed_count=1,
            reverted_count=1,
            baseline_result=_baseline(),
            final_result=_improved(),
            net_improvement=1,
            prediction_accuracy_summary={
                "accurate": 1, "partial": 0, "wrong": 0,
                "no_prediction": 1, "total_with_prediction": 1, "accuracy_pct": 100.0,
            },
            efficiency_ratio=0.5,
            change_map={"src/main.py": 1, "src/other.py": 1},
        )

    def test_text_report_diff_hunks(self):
        report = self._make_report_with_all_fields()
        text = format_enforce_text(report, color=False)
        assert "src/main.py" in text
        assert "threshold" in text

    def test_text_report_change_quality(self):
        report = self._make_report_with_all_fields()
        text = format_enforce_text(report, color=False)
        assert "bug_fix" in text

    def test_text_report_prediction_accuracy(self):
        report = self._make_report_with_all_fields()
        text = format_enforce_text(report, color=False)
        assert "accurate" in text

    def test_text_report_regression_cause(self):
        report = self._make_report_with_all_fields()
        text = format_enforce_text(report, color=False)
        assert "import" in text.lower()

    def test_text_report_meta_classification(self):
        report = self._make_report_with_all_fields()
        text = format_enforce_text(report, color=False)
        assert "root_cause_fix" in text

    def test_text_report_rule_violations(self):
        report = self._make_report_with_all_fields()
        text = format_enforce_text(report, color=False)
        assert "print(" in text

    def test_markdown_report_diff_hunks(self):
        report = self._make_report_with_all_fields()
        md = format_enforce_markdown(report)
        assert "```diff" in md
        assert "threshold" in md

    def test_markdown_report_quality(self):
        report = self._make_report_with_all_fields()
        md = format_enforce_markdown(report)
        assert "**Quality:**" in md

    def test_markdown_report_prediction(self):
        report = self._make_report_with_all_fields()
        md = format_enforce_markdown(report)
        assert "**Prediction:**" in md

    def test_markdown_report_regression_cause(self):
        report = self._make_report_with_all_fields()
        md = format_enforce_markdown(report)
        assert "**Regression cause:**" in md

    def test_markdown_report_classification(self):
        report = self._make_report_with_all_fields()
        md = format_enforce_markdown(report)
        assert "**Classification:**" in md

    def test_markdown_report_rule_violations(self):
        report = self._make_report_with_all_fields()
        md = format_enforce_markdown(report)
        assert "Rule violations:" in md

    def test_json_report_all_new_fields(self):
        report = self._make_report_with_all_fields()
        json_str = format_enforce_json(report)
        data = json.loads(json_str)
        change = data["changes"][0]
        assert "diff_hunks" in change
        assert "change_quality" in change
        assert "prediction_accuracy" in change
        assert "meta_analysis" in change
        assert "rule_violations" in change
        assert change["change_quality"] == "bug_fix"
        assert change["prediction_accuracy"] == "accurate"
        assert change["meta_analysis"]["classification"] == "root_cause_fix"

    def test_rejected_in_text(self):
        change = ChangeRecord(
            iteration=1, decision="REJECTED", audit_verdict="VALID",
            before=_baseline(), after=_baseline(),
        )
        report = EnforceReport(
            config=EnforceConfig(test_command="test"),
            changes=[change],
            total_iterations=1,
            rejected_count=1,
            baseline_result=_baseline(),
        )
        text = format_enforce_text(report, color=False)
        assert "Rejected:" in text

    def test_rejected_in_markdown(self):
        change = ChangeRecord(
            iteration=1, decision="REJECTED", audit_verdict="VALID",
            before=_baseline(), after=_baseline(),
        )
        report = EnforceReport(
            config=EnforceConfig(test_command="test"),
            changes=[change],
            total_iterations=1,
            rejected_count=1,
            baseline_result=_baseline(),
        )
        md = format_enforce_markdown(report)
        assert "REJECTED" in md


# ===========================================================================
# Backward compatibility
# ===========================================================================

class TestBackwardCompatibility:
    def test_change_record_from_old_dict(self):
        """Old session files without new fields should still load."""
        old_dict = {
            "iteration": 1,
            "files_modified": ["a.py"],
            "before": {
                "exit_code": 1, "passed": 5, "failed": 3, "errors": 0,
                "skipped": 0, "total": 8, "duration_seconds": 1.0,
                "output": "", "timestamp": "2026-01-01T00:00:00",
            },
            "after": {
                "exit_code": 0, "passed": 8, "failed": 0, "errors": 0,
                "skipped": 0, "total": 8, "duration_seconds": 1.0,
                "output": "", "timestamp": "2026-01-01T00:01:00",
            },
            "decision": "COMMITTED",
            "audit_verdict": "VALID",
        }
        rec = ChangeRecord.from_dict(old_dict)
        assert rec.iteration == 1
        assert rec.meta_analysis == {}
        assert rec.diff_hunks == []
        assert rec.change_quality == ""
        assert rec.prediction_accuracy == ""
        assert rec.predicted_tests == []
        assert rec.regression_root_cause == ""
        assert rec.rule_violations == []

    def test_enforce_config_from_old_dict(self):
        """Old config without new fields should load with defaults."""
        old_dict = {
            "test_command": "pytest",
            "max_iterations": 50,
        }
        cfg = EnforceConfig.from_dict(old_dict)
        assert cfg.max_files_per_change == 5
        assert cfg.max_diff_lines == 200
        assert cfg.rules_file is None

    def test_enforce_report_from_old_session(self, tmp_path: Path):
        """Report from old session (without new fields) should still build."""
        cfg = EnforceConfig(test_command="test", project_root=str(tmp_path))
        _save_session(cfg, _baseline(), str(tmp_path))
        _save_iteration(ChangeRecord(
            iteration=1, decision="COMMITTED", audit_verdict="VALID",
            before=_baseline(), after=_improved(),
        ), str(tmp_path))
        report = build_enforce_report(str(tmp_path))
        assert report.total_iterations == 1
        assert report.efficiency_ratio > 0
        assert isinstance(report.prediction_accuracy_summary, dict)
        assert isinstance(report.change_map, dict)


# ===========================================================================
# _group_diff_by_file helper
# ===========================================================================

class TestGroupDiffByFile:
    def test_basic(self):
        diff = """\
diff --git a/a.py b/a.py
+added_a
-removed_a
diff --git a/b.py b/b.py
+added_b
"""
        result = _group_diff_by_file(diff)
        assert result["a.py"] == (1, 1)
        assert result["b.py"] == (1, 0)

    def test_empty(self):
        assert _group_diff_by_file("") == {}
