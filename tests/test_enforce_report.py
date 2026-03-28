"""Tests for enforcement report generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_xray.enforce import (
    ChangeRecord,
    ChallengeResult,
    EnforceConfig,
    EnforceReport,
    TestResult,
    _save_iteration,
    _save_session,
)
from agent_xray.enforce_report import (
    _build_detailed_change_map,
    _build_timeline,
    _improvements_per_iteration,
    check_against_rules,
    format_enforce_json,
    format_enforce_markdown,
    format_enforce_text,
    format_rules_violations,
    generate_report,
    grade_enforce_session,
    load_project_rules,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _baseline() -> TestResult:
    return TestResult(
        exit_code=1, passed=8, failed=2, errors=0, skipped=0,
        total=10, duration_seconds=3.5, output="8 passed, 2 failed",
    )


def _improved() -> TestResult:
    return TestResult(
        exit_code=0, passed=10, failed=0, errors=0, skipped=0,
        total=10, duration_seconds=2.8, output="10 passed",
    )


def _config() -> EnforceConfig:
    return EnforceConfig(test_command="pytest tests/ -x", max_iterations=50)


def _change(iteration: int, decision: str = "COMMITTED", verdict: str = "VALID") -> ChangeRecord:
    return ChangeRecord(
        iteration=iteration,
        files_modified=["src/main.py"],
        hypothesis=f"Fix iteration {iteration}",
        before=_baseline(),
        after=_improved(),
        tests_improved=["test_x"],
        tests_regressed=[],
        net_improvement=1,
        audit_verdict=verdict,
        decision=decision,
        commit_hash=f"abc{iteration:03d}",
        gaming_signals=["test_file_modification"] if verdict == "GAMING" else [],
        audit_reasons=[f"Reason for {verdict}"],
    )


def _challenge() -> ChallengeResult:
    return ChallengeResult(
        iteration_range=(1, 3),
        changes_reviewed=3,
        vetoed=[2],
        findings=["Consecutive suspicious changes", "Hot file detected"],
    )


def _report(
    *,
    num_changes: int = 3,
    include_challenge: bool = True,
) -> EnforceReport:
    changes = [_change(i + 1) for i in range(num_changes)]
    if num_changes > 1:
        changes[1] = _change(2, decision="REVERTED", verdict="GAMING")
    challenges = [_challenge()] if include_challenge else []
    return EnforceReport(
        config=_config(),
        changes=changes,
        challenges=challenges,
        total_iterations=num_changes,
        committed_count=num_changes - 1 if num_changes > 1 else num_changes,
        reverted_count=1 if num_changes > 1 else 0,
        vetoed_count=0,
        gaming_detected_count=1 if num_changes > 1 else 0,
        baseline_result=_baseline(),
        final_result=_improved(),
        net_improvement=2,
        duration_seconds=120.0,
    )


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

class TestFormatEnforceText:
    def test_contains_header(self):
        text = format_enforce_text(_report(), color=False)
        assert "ENFORCEMENT REPORT" in text

    def test_contains_summary(self):
        text = format_enforce_text(_report(), color=False)
        assert "Iterations: 3" in text
        assert "Committed:" in text
        assert "Reverted:" in text
        assert "Gaming:" in text

    def test_contains_baseline(self):
        text = format_enforce_text(_report(), color=False)
        assert "8 passed" in text
        assert "2 failed" in text

    def test_contains_iterations(self):
        text = format_enforce_text(_report(), color=False)
        assert "Iteration 1" in text
        assert "Iteration 2" in text
        assert "Iteration 3" in text

    def test_contains_challenge(self):
        text = format_enforce_text(_report(), color=True)
        assert "Adversarial" in text
        assert "Consecutive" in text

    def test_no_color(self):
        text = format_enforce_text(_report(), color=False)
        assert "\033[" not in text

    def test_with_color(self):
        text = format_enforce_text(_report(), color=True)
        assert "\033[" in text

    def test_empty_report(self):
        report = EnforceReport(
            config=_config(),
            baseline_result=_baseline(),
            final_result=_baseline(),
        )
        text = format_enforce_text(report, color=False)
        assert "ENFORCEMENT REPORT" in text
        assert "Iterations: 0" in text

    def test_gaming_signals_shown(self):
        report = _report(num_changes=2)
        text = format_enforce_text(report, color=False)
        assert "test_file_modification" in text

    def test_net_improvement_shown(self):
        text = format_enforce_text(_report(), color=False)
        assert "+2" in text


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------

class TestFormatEnforceJson:
    def test_valid_json(self):
        result = format_enforce_json(_report())
        data = json.loads(result)
        assert isinstance(data, dict)

    def test_contains_all_fields(self):
        result = format_enforce_json(_report())
        data = json.loads(result)
        assert "config" in data
        assert "changes" in data
        assert "challenges" in data
        assert "total_iterations" in data
        assert "committed_count" in data
        assert "reverted_count" in data
        assert "gaming_detected_count" in data
        assert "baseline_result" in data
        assert "final_result" in data
        assert "net_improvement" in data

    def test_changes_structure(self):
        result = format_enforce_json(_report())
        data = json.loads(result)
        assert len(data["changes"]) == 3
        change = data["changes"][0]
        assert "iteration" in change
        assert "files_modified" in change
        assert "audit_verdict" in change
        assert "decision" in change

    def test_empty_report_json(self):
        report = EnforceReport(
            config=_config(),
            baseline_result=_baseline(),
        )
        result = format_enforce_json(report)
        data = json.loads(result)
        assert data["total_iterations"] == 0
        assert len(data["changes"]) == 0


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

class TestFormatEnforceMarkdown:
    def test_header(self):
        md = format_enforce_markdown(_report())
        assert "# Enforcement Report" in md

    def test_summary_table(self):
        md = format_enforce_markdown(_report())
        assert "| Metric | Value |" in md
        assert "Iterations" in md
        assert "Committed" in md

    def test_baseline_table(self):
        md = format_enforce_markdown(_report())
        assert "Baseline vs Final" in md
        assert "| Baseline |" in md
        assert "| Final |" in md

    def test_iteration_sections(self):
        md = format_enforce_markdown(_report())
        assert "### Iteration 1" in md
        assert "### Iteration 2" in md

    def test_challenge_section(self):
        md = format_enforce_markdown(_report())
        assert "## Adversarial Challenges" in md
        assert "Challenge 1" in md

    def test_hypothesis_shown(self):
        md = format_enforce_markdown(_report())
        assert "**Hypothesis:**" in md

    def test_files_shown(self):
        md = format_enforce_markdown(_report())
        assert "`src/main.py`" in md

    def test_commit_hash_shown(self):
        md = format_enforce_markdown(_report())
        assert "`abc001`" in md

    def test_empty_report_markdown(self):
        report = EnforceReport(
            config=_config(),
            baseline_result=_baseline(),
        )
        md = format_enforce_markdown(report)
        assert "# Enforcement Report" in md
        assert "| Iterations | 0 |" in md


# ---------------------------------------------------------------------------
# generate_report (convenience)
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_text_format(self, tmp_path: Path):
        cfg = EnforceConfig(test_command="pytest", project_root=str(tmp_path))
        _save_session(cfg, _baseline(), str(tmp_path))
        result = generate_report(str(tmp_path), format="text", color=False)
        assert "ENFORCEMENT REPORT" in result

    def test_json_format(self, tmp_path: Path):
        cfg = EnforceConfig(test_command="pytest", project_root=str(tmp_path))
        _save_session(cfg, _baseline(), str(tmp_path))
        result = generate_report(str(tmp_path), format="json")
        data = json.loads(result)
        assert "config" in data

    def test_markdown_format(self, tmp_path: Path):
        cfg = EnforceConfig(test_command="pytest", project_root=str(tmp_path))
        _save_session(cfg, _baseline(), str(tmp_path))
        result = generate_report(str(tmp_path), format="markdown")
        assert "# Enforcement Report" in result

    def test_with_iterations(self, tmp_path: Path):
        cfg = EnforceConfig(test_command="pytest", project_root=str(tmp_path))
        _save_session(cfg, _baseline(), str(tmp_path))
        _save_iteration(ChangeRecord(
            iteration=1, decision="COMMITTED", audit_verdict="VALID",
            before=_baseline(), after=_improved(),
        ), str(tmp_path))
        result = generate_report(str(tmp_path), format="text", color=False)
        assert "Iteration 1" in result
        assert "Iterations: 1" in result


# ---------------------------------------------------------------------------
# Additional helpers for grading / cumulative tests
# ---------------------------------------------------------------------------

def _tr(passed: int, failed: int) -> TestResult:
    """Quick TestResult builder."""
    return TestResult(
        exit_code=0 if failed == 0 else 1,
        passed=passed,
        failed=failed,
        errors=0,
        skipped=0,
        total=passed + failed,
        duration_seconds=1.0,
        output=f"{passed} passed, {failed} failed",
    )


def _change_with(
    iteration: int,
    *,
    decision: str = "COMMITTED",
    verdict: str = "VALID",
    before_failed: int = 5,
    after_failed: int = 3,
    files: list[str] | None = None,
    gaming_signals: list[str] | None = None,
    net_improvement: int | None = None,
    started_at: str = "",
    commit_hash: str = "",
) -> ChangeRecord:
    """Flexible ChangeRecord builder for tests."""
    before = _tr(10 - before_failed, before_failed)
    after = _tr(10 - after_failed, after_failed)
    if net_improvement is None:
        net_improvement = before_failed - after_failed
    return ChangeRecord(
        iteration=iteration,
        files_modified=files or ["src/main.py"],
        hypothesis=f"Fix iteration {iteration}",
        before=before,
        after=after,
        net_improvement=net_improvement,
        audit_verdict=verdict,
        decision=decision,
        commit_hash=commit_hash,
        gaming_signals=gaming_signals or (["test_deletion"] if verdict == "GAMING" else []),
        started_at=started_at,
    )


def _grading_report(
    changes: list[ChangeRecord],
    *,
    baseline_failed: int = 10,
    final_failed: int = 0,
) -> EnforceReport:
    """Build a report for grading tests."""
    committed = sum(1 for c in changes if c.decision == "COMMITTED")
    reverted = sum(1 for c in changes if c.decision == "REVERTED")
    gaming = sum(1 for c in changes if c.gaming_signals)
    return EnforceReport(
        config=_config(),
        changes=changes,
        total_iterations=len(changes),
        committed_count=committed,
        reverted_count=reverted,
        gaming_detected_count=gaming,
        baseline_result=_tr(10 - baseline_failed, baseline_failed),
        final_result=_tr(10 - final_failed, final_failed),
        net_improvement=baseline_failed - final_failed,
        duration_seconds=60.0,
    )


# ---------------------------------------------------------------------------
# GAP 3: Rules File Awareness
# ---------------------------------------------------------------------------

class TestLoadProjectRules:
    def test_load_project_rules_basic(self, tmp_path: Path):
        rules = {
            "forbidden_patterns": ["except: pass", "noqa"],
            "required_patterns": ["logger."],
            "max_complexity": 10,
            "banned_imports": ["os.system"],
            "custom_rules": [
                {
                    "name": "no-print",
                    "pattern": "^\\+.*\\bprint\\(",
                    "description": "No print statements",
                    "confidence": 0.5,
                }
            ],
        }
        rules_file = tmp_path / "rules.json"
        rules_file.write_text(json.dumps(rules), encoding="utf-8")

        loaded = load_project_rules(str(rules_file))
        assert loaded["forbidden_patterns"] == ["except: pass", "noqa"]
        assert loaded["required_patterns"] == ["logger."]
        assert loaded["max_complexity"] == 10
        assert loaded["banned_imports"] == ["os.system"]
        assert len(loaded["custom_rules"]) == 1
        assert loaded["custom_rules"][0]["name"] == "no-print"

    def test_load_project_rules_missing_file(self):
        result = load_project_rules("/nonexistent/path/rules.json")
        assert result == {}

    def test_load_project_rules_invalid_json(self, tmp_path: Path):
        bad = tmp_path / "bad.json"
        bad.write_text("not json at all", encoding="utf-8")
        assert load_project_rules(str(bad)) == {}

    def test_load_project_rules_non_dict(self, tmp_path: Path):
        arr = tmp_path / "arr.json"
        arr.write_text("[1, 2, 3]", encoding="utf-8")
        assert load_project_rules(str(arr)) == {}


class TestCheckAgainstRules:
    def test_check_against_rules_forbidden(self):
        diff = (
            "+++ b/src/foo.py\n"
            "+    except: pass\n"
            "+    x = 1  # noqa\n"
            " normal line\n"
        )
        rules = {"forbidden_patterns": ["except: pass", "noqa"]}
        violations = check_against_rules(diff, rules)
        assert len(violations) == 2
        assert any("except: pass" in v for v in violations)
        assert any("noqa" in v for v in violations)

    def test_check_against_rules_banned_imports(self):
        diff = (
            "+import os\n"
            "+from os import system\n"
            "+import os.system\n"
        )
        rules = {"banned_imports": ["os.system"]}
        violations = check_against_rules(diff, rules)
        assert len(violations) >= 1
        assert any("os.system" in v for v in violations)

    def test_check_against_rules_custom(self):
        diff = "+    print('hello')\n+    logger.info('ok')\n"
        rules = {
            "custom_rules": [
                {
                    "name": "no-print",
                    "pattern": "^\\+.*\\bprint\\(",
                    "description": "No print statements",
                    "confidence": 0.5,
                }
            ],
        }
        violations = check_against_rules(diff, rules)
        assert len(violations) == 1
        assert "no-print" in violations[0]
        assert "No print statements" in violations[0]

    def test_required_under_threshold(self):
        diff = "+line1\n+line2\n"
        rules = {"required_patterns": ["logger."]}
        assert check_against_rules(diff, rules) == []

    def test_required_over_threshold_missing(self):
        diff = "\n".join(f"+line{i}" for i in range(12)) + "\n"
        rules = {"required_patterns": ["logger."]}
        violations = check_against_rules(diff, rules)
        assert len(violations) == 1
        assert "logger." in violations[0]

    def test_required_over_threshold_present(self):
        lines = [f"+line{i}" for i in range(11)]
        lines.append("+    logger.info('hello')")
        diff = "\n".join(lines) + "\n"
        rules = {"required_patterns": ["logger."]}
        assert check_against_rules(diff, rules) == []

    def test_empty_diff(self):
        assert check_against_rules("", {"forbidden_patterns": ["x"]}) == []

    def test_empty_rules(self):
        assert check_against_rules("+something\n", {}) == []

    def test_bad_regex_in_custom_rule(self):
        diff = "+something\n"
        rules = {"custom_rules": [{"name": "bad", "pattern": "[invalid(", "description": "bad"}]}
        assert check_against_rules(diff, rules) == []


class TestFormatRulesViolations:
    def test_no_violations(self):
        result = format_rules_violations([], color=False)
        assert "No rule violations" in result

    def test_with_violations(self):
        viols = ["Found bad pattern", "Missing import"]
        result = format_rules_violations(viols, color=False)
        assert "2" in result
        assert "Found bad pattern" in result
        assert "Missing import" in result


# ---------------------------------------------------------------------------
# GAP 11: Grading Integration
# ---------------------------------------------------------------------------

class TestGradeEnforceSession:
    def test_grade_session_all_committed(self):
        """All committed with improvements should get A."""
        changes = [
            _change_with(1, before_failed=10, after_failed=7, commit_hash="aaa"),
            _change_with(2, before_failed=7, after_failed=4, commit_hash="bbb"),
            _change_with(3, before_failed=4, after_failed=0, commit_hash="ccc"),
        ]
        report = _grading_report(changes, baseline_failed=10, final_failed=0)
        result = grade_enforce_session(report)
        assert result["grade"] == "A"
        assert result["score"] >= 90

    def test_grade_session_with_gaming(self):
        """Gaming detections should lower grade significantly."""
        changes = [
            _change_with(1, before_failed=10, after_failed=8, commit_hash="a"),
            _change_with(2, decision="REVERTED", verdict="GAMING",
                         before_failed=8, after_failed=5),
            _change_with(3, decision="REVERTED", verdict="GAMING",
                         before_failed=8, after_failed=6),
            _change_with(4, before_failed=8, after_failed=5, commit_hash="d"),
        ]
        report = _grading_report(changes, baseline_failed=10, final_failed=5)
        result = grade_enforce_session(report)
        # 2 gaming (-30), 2 reverted (-10): should be well below A
        assert result["grade"] in ("C", "D", "F")
        assert result["score"] < 75

    def test_grade_session_net_regression(self):
        """Final worse than baseline should have regression penalty.

        With many reverted + gaming + waste + regression, score drops heavily.
        """
        changes = [
            _change_with(i, decision="REVERTED", verdict="GAMING",
                         before_failed=5, after_failed=5, net_improvement=0)
            for i in range(1, 7)
        ]
        report = _grading_report(changes, baseline_failed=5, final_failed=8)
        result = grade_enforce_session(report)
        # 100 - 6*5 (reverted) - 6*15 (gaming) - 6*3 (waste) - 10 (regression)
        # = 100 - 30 - 90 - 18 - 10 = -48 -> clamped to 0
        assert result["grade"] == "F"
        assert result["score"] == 0

    def test_grade_perfect_session(self):
        """Perfect session: all committed, all improving, final=0."""
        changes = [
            _change_with(1, before_failed=5, after_failed=2, commit_hash="a"),
            _change_with(2, before_failed=2, after_failed=0, commit_hash="b"),
        ]
        report = _grading_report(changes, baseline_failed=5, final_failed=0)
        result = grade_enforce_session(report)
        # 100 + 2*2 (good commits) + 10 (all pass) = 114 -> clamped 100
        assert result["grade"] == "A"
        assert result["score"] == 100

    def test_grade_all_waste(self):
        """Every iteration wastes (net_improvement=0) => low grade."""
        changes = [
            _change_with(i, before_failed=5, after_failed=5, net_improvement=0,
                         commit_hash=f"c{i}")
            for i in range(1, 11)
        ]
        report = _grading_report(changes, baseline_failed=5, final_failed=5)
        result = grade_enforce_session(report)
        # 100 - 10*3 (waste) = 70
        assert result["score"] == 70
        assert result["grade"] == "C"

    def test_grade_factors_present(self):
        changes = [
            _change_with(1, decision="REVERTED", verdict="GAMING",
                         before_failed=5, after_failed=3),
            _change_with(2, before_failed=5, after_failed=3, commit_hash="abc"),
        ]
        report = _grading_report(changes, baseline_failed=5, final_failed=3)
        result = grade_enforce_session(report)
        assert "factors" in result
        assert isinstance(result["factors"], dict)


# ---------------------------------------------------------------------------
# GAP 13: Cumulative Report Improvements
# ---------------------------------------------------------------------------

class TestEfficiencyMetricInText:
    def test_efficiency_metric_in_text(self):
        report = _grading_report(
            [_change_with(1, before_failed=10, after_failed=5, commit_hash="a")],
            baseline_failed=10,
            final_failed=5,
        )
        text = format_enforce_text(report, color=False)
        assert "Improvements/iteration" in text

    def test_improvements_per_iteration_value(self):
        report = _grading_report(
            [
                _change_with(1, before_failed=10, after_failed=5, commit_hash="a"),
                _change_with(2, before_failed=5, after_failed=2, commit_hash="b"),
            ],
            baseline_failed=10,
            final_failed=2,
        )
        ipi = _improvements_per_iteration(report)
        # net_improvement = 10 - 2 = 8, iterations = 2 => 4.0
        assert ipi == 4.0


class TestChangeMapInMarkdown:
    def test_change_map_in_markdown(self):
        changes = [
            _change_with(1, files=["src/foo.py", "src/bar.py"], commit_hash="a"),
            _change_with(2, files=["src/foo.py"], commit_hash="b"),
            _change_with(3, decision="REVERTED", files=["src/foo.py", "src/baz.py"]),
        ]
        report = _grading_report(changes, baseline_failed=5, final_failed=2)
        md = format_enforce_markdown(report)
        assert "## Change Map" in md
        assert "src/foo.py" in md
        assert "src/bar.py" in md

    def test_detailed_change_map_breakdown(self):
        changes = [
            _change_with(1, files=["src/foo.py"], commit_hash="a"),
            _change_with(2, files=["src/foo.py"], commit_hash="b"),
            _change_with(3, decision="REVERTED", files=["src/foo.py"]),
        ]
        report = _grading_report(changes, baseline_failed=5, final_failed=2)
        dmap = _build_detailed_change_map(report)
        assert len(dmap) == 1
        assert dmap[0]["file"] == "src/foo.py"
        assert dmap[0]["total"] == 3
        assert dmap[0]["committed"] == 2
        assert dmap[0]["reverted"] == 1


class TestTimelineInText:
    def test_timeline_in_text(self):
        changes = [
            _change_with(1, before_failed=5, after_failed=3,
                         commit_hash="abc1234", started_at="2026-03-28T10:30:15Z"),
        ]
        report = _grading_report(changes, baseline_failed=5, final_failed=3)
        text = format_enforce_text(report, color=False)
        assert "Timeline" in text
        assert "10:30:15" in text
        assert "abc1234" in text

    def test_timeline_gaming_flag(self):
        changes = [
            _change_with(1, decision="REVERTED", verdict="GAMING",
                         before_failed=5, after_failed=3, started_at="2026-03-28T10:31:00Z"),
        ]
        report = _grading_report(changes, baseline_failed=5, final_failed=5)
        timeline = _build_timeline(report)
        assert timeline[0]["gaming"] is True


class TestGradeInJson:
    def test_grade_in_json(self):
        changes = [
            _change_with(1, before_failed=5, after_failed=3, commit_hash="abc"),
        ]
        report = _grading_report(changes, baseline_failed=5, final_failed=3)
        raw = format_enforce_json(report)
        data = json.loads(raw)
        assert "grade" in data
        assert "grade" in data["grade"]
        assert "score" in data["grade"]
        assert "factors" in data["grade"]

    def test_improvements_per_iteration_in_json(self):
        report = _grading_report(
            [_change_with(1, before_failed=10, after_failed=5, commit_hash="a")],
            baseline_failed=10,
            final_failed=5,
        )
        raw = format_enforce_json(report)
        data = json.loads(raw)
        assert "improvements_per_iteration" in data
        assert data["improvements_per_iteration"] == 5.0

    def test_timeline_in_json(self):
        changes = [
            _change_with(1, commit_hash="aaa", started_at="2026-03-28T10:00:00Z"),
        ]
        report = _grading_report(changes)
        raw = format_enforce_json(report)
        data = json.loads(raw)
        assert "timeline" in data
        assert data["timeline"][0]["timestamp"] == "10:00:00"

    def test_detailed_change_map_in_json(self):
        changes = [
            _change_with(1, files=["src/a.py"], commit_hash="a"),
            _change_with(2, files=["src/a.py", "src/b.py"], commit_hash="b"),
        ]
        report = _grading_report(changes)
        raw = format_enforce_json(report)
        data = json.loads(raw)
        assert "detailed_change_map" in data
        assert len(data["detailed_change_map"]) == 2


class TestGradeInMarkdown:
    def test_grade_in_markdown(self):
        changes = [
            _change_with(1, before_failed=5, after_failed=3, commit_hash="abc"),
        ]
        report = _grading_report(changes, baseline_failed=5, final_failed=3)
        md = format_enforce_markdown(report)
        assert "## Grade" in md
        assert "/100" in md

    def test_timeline_in_markdown(self):
        changes = [
            _change_with(1, commit_hash="abc", started_at="2026-03-28T10:00:00Z"),
        ]
        report = _grading_report(changes)
        md = format_enforce_markdown(report)
        assert "## Timeline" in md
        assert "10:00:00" in md
