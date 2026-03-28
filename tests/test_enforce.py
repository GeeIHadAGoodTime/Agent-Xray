"""Tests for the enforcement engine core."""

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
    _git_diff_names,
    _git_diff_stat,
    _iteration_count,
    _load_iterations,
    _load_session,
    _save_iteration,
    _save_session,
    _session_dir,
    build_enforce_report,
    compare_test_results,
    enforce_check,
    enforce_diff,
    enforce_init,
    enforce_reset,
    enforce_status,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _baseline() -> TestResult:
    return TestResult(
        exit_code=1,
        passed=8,
        failed=2,
        errors=0,
        skipped=0,
        total=10,
        duration_seconds=3.5,
        output="8 passed, 2 failed in 3.50s",
        test_names_passed=["test_a", "test_b", "test_c", "test_d",
                           "test_e", "test_f", "test_g", "test_h"],
        test_names_failed=["test_x", "test_y"],
        timestamp="2026-03-28T12:00:00+00:00",
    )


def _improved() -> TestResult:
    return TestResult(
        exit_code=1,
        passed=9,
        failed=1,
        errors=0,
        skipped=0,
        total=10,
        duration_seconds=3.2,
        output="9 passed, 1 failed in 3.20s",
        test_names_passed=["test_a", "test_b", "test_c", "test_d",
                           "test_e", "test_f", "test_g", "test_h", "test_x"],
        test_names_failed=["test_y"],
        timestamp="2026-03-28T12:01:00+00:00",
    )


def _regressed() -> TestResult:
    return TestResult(
        exit_code=1,
        passed=7,
        failed=3,
        errors=0,
        skipped=0,
        total=10,
        duration_seconds=3.8,
        output="7 passed, 3 failed in 3.80s",
        test_names_passed=["test_a", "test_b", "test_c", "test_d",
                           "test_e", "test_f", "test_g"],
        test_names_failed=["test_h", "test_x", "test_y"],
        timestamp="2026-03-28T12:02:00+00:00",
    )


def _config(tmp_path: Path) -> EnforceConfig:
    return EnforceConfig(
        test_command="echo '8 passed, 2 failed in 3.50s'",
        project_root=str(tmp_path),
        max_iterations=10,
    )


# ---------------------------------------------------------------------------
# TestResult
# ---------------------------------------------------------------------------

class TestTestResult:
    def test_round_trip(self):
        tr = _baseline()
        d = tr.to_dict()
        tr2 = TestResult.from_dict(d)
        assert tr2.passed == 8
        assert tr2.failed == 2
        assert tr2.total == 10
        assert tr2.test_names_failed == ["test_x", "test_y"]

    def test_auto_timestamp(self):
        tr = TestResult(exit_code=0, passed=1, failed=0, errors=0,
                        skipped=0, total=1, duration_seconds=0.1, output="ok")
        assert tr.timestamp != ""


# ---------------------------------------------------------------------------
# EnforceConfig
# ---------------------------------------------------------------------------

class TestEnforceConfig:
    def test_round_trip(self):
        cfg = EnforceConfig(test_command="pytest", max_iterations=20)
        d = cfg.to_dict()
        cfg2 = EnforceConfig.from_dict(d)
        assert cfg2.test_command == "pytest"
        assert cfg2.max_iterations == 20

    def test_defaults(self):
        cfg = EnforceConfig(test_command="pytest")
        assert cfg.max_iterations == 50
        assert cfg.challenge_every == 5
        assert cfg.require_improvement is True
        assert cfg.allow_test_modification is False
        assert cfg.git_auto_commit is True
        assert cfg.git_auto_revert is True


# ---------------------------------------------------------------------------
# ChangeRecord
# ---------------------------------------------------------------------------

class TestChangeRecord:
    def test_round_trip(self):
        rec = ChangeRecord(
            iteration=1,
            files_modified=["foo.py"],
            before=_baseline(),
            after=_improved(),
            tests_improved=["test_x"],
            net_improvement=1,
            audit_verdict="VALID",
            decision="COMMITTED",
            commit_hash="abc123",
        )
        d = rec.to_dict()
        rec2 = ChangeRecord.from_dict(d)
        assert rec2.iteration == 1
        assert rec2.before is not None
        assert rec2.before.passed == 8
        assert rec2.after is not None
        assert rec2.after.passed == 9
        assert rec2.tests_improved == ["test_x"]

    def test_minimal(self):
        rec = ChangeRecord(iteration=1)
        assert rec.files_modified == []
        assert rec.before is None
        assert rec.decision == ""


# ---------------------------------------------------------------------------
# ChallengeResult
# ---------------------------------------------------------------------------

class TestChallengeResult:
    def test_round_trip(self):
        cr = ChallengeResult(
            iteration_range=(1, 5),
            changes_reviewed=5,
            vetoed=[3],
            findings=["Found issue"],
        )
        d = cr.to_dict()
        cr2 = ChallengeResult.from_dict(d)
        assert cr2.iteration_range == (1, 5)
        assert cr2.vetoed == [3]


# ---------------------------------------------------------------------------
# compare_test_results
# ---------------------------------------------------------------------------

class TestCompareTestResults:
    def test_improvement(self):
        improved, regressed, unchanged = compare_test_results(_baseline(), _improved())
        assert "test_x" in improved
        assert len(regressed) == 0

    def test_regression(self):
        improved, regressed, unchanged = compare_test_results(_baseline(), _regressed())
        assert "test_h" in regressed

    def test_no_change(self):
        improved, regressed, unchanged = compare_test_results(_baseline(), _baseline())
        assert improved == []
        assert regressed == []

    def test_count_fallback(self):
        """When no test names, compare counts."""
        before = TestResult(
            exit_code=1, passed=5, failed=3, errors=0, skipped=0,
            total=8, duration_seconds=1.0, output="",
        )
        after = TestResult(
            exit_code=1, passed=7, failed=1, errors=0, skipped=0,
            total=8, duration_seconds=1.0, output="",
        )
        improved, regressed, unchanged = compare_test_results(before, after)
        assert len(improved) > 0  # Should show count-based improvement

    def test_failed_only_names_still_show_improvement(self):
        before = TestResult(
            exit_code=1, passed=5, failed=2, errors=0, skipped=0,
            total=7, duration_seconds=1.0, output="",
            test_names_failed=["tests/test_api.py::test_alpha", "tests/test_api.py::test_beta"],
        )
        after = TestResult(
            exit_code=1, passed=6, failed=1, errors=0, skipped=0,
            total=7, duration_seconds=1.0, output="",
            test_names_failed=["tests/test_api.py::test_beta"],
        )
        improved, regressed, unchanged = compare_test_results(before, after)
        assert improved == ["tests/test_api.py::test_alpha"]
        assert regressed == []


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------

class TestSessionPersistence:
    def test_save_and_load_session(self, tmp_path: Path):
        cfg = _config(tmp_path)
        baseline = _baseline()
        _save_session(cfg, baseline, str(tmp_path))
        cfg2, bl2, data = _load_session(str(tmp_path))
        assert cfg2.test_command == cfg.test_command
        assert bl2.passed == 8

    def test_load_missing_session(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="No enforcement session"):
            _load_session(str(tmp_path))

    def test_save_and_load_iterations(self, tmp_path: Path):
        cfg = _config(tmp_path)
        _save_session(cfg, _baseline(), str(tmp_path))
        rec = ChangeRecord(
            iteration=1, files_modified=["a.py"],
            before=_baseline(), after=_improved(),
            decision="COMMITTED",
        )
        _save_iteration(rec, str(tmp_path))
        loaded = _load_iterations(str(tmp_path))
        assert len(loaded) == 1
        assert loaded[0].iteration == 1

    def test_iteration_count(self, tmp_path: Path):
        cfg = _config(tmp_path)
        _save_session(cfg, _baseline(), str(tmp_path))
        assert _iteration_count(str(tmp_path)) == 0
        _save_iteration(ChangeRecord(iteration=1), str(tmp_path))
        assert _iteration_count(str(tmp_path)) == 1
        _save_iteration(ChangeRecord(iteration=2), str(tmp_path))
        assert _iteration_count(str(tmp_path)) == 2


class TestGitDiffFiltering:
    def test_git_diff_names_ignores_crlf_warnings(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            "agent_xray.enforce._run_shell",
            lambda command, cwd=None: (
                0,
                "warning: in the working copy of foo.py, LF will be replaced by CRLF\n"
                "src/foo.py\n"
                "tests/test_foo.py\n",
            ),
        )
        assert _git_diff_names() == ["src/foo.py", "tests/test_foo.py"]

    def test_git_diff_stat_ignores_crlf_warnings(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            "agent_xray.enforce._run_shell",
            lambda command, cwd=None: (
                0,
                "warning: in the working copy of foo.py, LF will be replaced by CRLF\n"
                " src/foo.py | 2 +-\n"
                " 1 file changed, 1 insertion(+), 1 deletion(-)\n",
            ),
        )
        assert _git_diff_stat() == (
            "src/foo.py | 2 +-\n1 file changed, 1 insertion(+), 1 deletion(-)"
        )


# ---------------------------------------------------------------------------
# enforce_init
# ---------------------------------------------------------------------------

class TestEnforceInit:
    def test_basic_init(self, tmp_path: Path):
        cfg = _config(tmp_path)

        def mock_run_tests(cmd, cwd):
            return _baseline()

        baseline, sd = enforce_init(cfg, _run_tests_fn=mock_run_tests)
        assert baseline.passed == 8
        assert sd.exists()
        assert (sd / "session.json").exists()
        assert (sd / "baseline.json").exists()

    def test_session_dir_created(self, tmp_path: Path):
        cfg = _config(tmp_path)

        def mock_run(cmd, cwd):
            return _baseline()

        _, sd = enforce_init(cfg, _run_tests_fn=mock_run)
        assert (sd / "iterations").is_dir()
        assert (sd / "challenges").is_dir()

    def test_stash_first_stashes_and_reset_pops(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        cfg = EnforceConfig(
            test_command="pytest",
            project_root=str(tmp_path),
            stash_first=True,
        )
        calls = {"stash": 0, "pop": 0}

        monkeypatch.setattr("agent_xray.enforce._git_has_uncommitted_changes", lambda cwd=None: True)
        monkeypatch.setattr("agent_xray.enforce._git_head_hash", lambda cwd=None: "abc123")

        def mock_stash(cwd=None):
            calls["stash"] += 1
            return True

        def mock_pop(cwd=None):
            calls["pop"] += 1
            return True

        monkeypatch.setattr("agent_xray.enforce._git_stash", mock_stash)
        monkeypatch.setattr("agent_xray.enforce._git_stash_pop", mock_pop)

        _, sd = enforce_init(cfg, _run_tests_fn=lambda cmd, cwd: _baseline())
        session_data = json.loads((sd / "session.json").read_text(encoding="utf-8"))

        assert calls["stash"] == 1
        assert session_data["stash_saved"] is True
        assert enforce_reset(str(tmp_path)) is True
        assert calls["pop"] == 1

    def test_init_adds_session_dir_to_existing_gitignore(self, tmp_path: Path):
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("node_modules/\n", encoding="utf-8")

        cfg = _config(tmp_path)
        enforce_init(cfg, _run_tests_fn=lambda cmd, cwd: _baseline())

        lines = gitignore.read_text(encoding="utf-8").splitlines()
        assert ".agent-xray-enforce/" in lines

    def test_init_does_not_create_gitignore(self, tmp_path: Path):
        cfg = _config(tmp_path)
        enforce_init(cfg, _run_tests_fn=lambda cmd, cwd: _baseline())
        assert not (tmp_path / ".gitignore").exists()


# ---------------------------------------------------------------------------
# enforce_check
# ---------------------------------------------------------------------------

class TestEnforceCheck:
    def test_basic_check_improvement(self, tmp_path: Path):
        cfg = _config(tmp_path)
        _save_session(cfg, _baseline(), str(tmp_path))

        call_count = {"n": 0}

        def mock_run(cmd, cwd):
            return _improved()

        def mock_audit(diff, files_modified, allow_test_modification):
            return "VALID", ["No gaming detected"], []

        record = enforce_check(
            "fix test_x",
            project_root=str(tmp_path),
            _run_tests_fn=mock_run,
            _audit_fn=mock_audit,
            _git_diff_fn=lambda cwd: "1 file changed",
            _git_names_fn=lambda cwd: ["foo.py"],
            _git_commit_fn=lambda msg, cwd: "abc123",
            _git_revert_fn=lambda h, cwd: True,
            _git_head_fn=lambda cwd: "def456",
            _git_diff_content_fn=lambda cwd: "+fix",
        )
        assert record.iteration == 1
        assert record.decision == "COMMITTED"
        assert record.audit_verdict == "VALID"
        assert record.hypothesis == "fix test_x"

    def test_check_revert_on_regression(self, tmp_path: Path):
        cfg = EnforceConfig(
            test_command="test",
            project_root=str(tmp_path),
            require_improvement=True,
        )
        _save_session(cfg, _baseline(), str(tmp_path))

        reverted = {"called": False}

        def mock_revert(h, cwd):
            reverted["called"] = True
            return True

        record = enforce_check(
            "bad change",
            project_root=str(tmp_path),
            _run_tests_fn=lambda cmd, cwd: _regressed(),
            _audit_fn=lambda diff, files_modified, allow_test_modification: (
                "VALID", [], []
            ),
            _git_diff_fn=lambda cwd: "",
            _git_names_fn=lambda cwd: ["bar.py"],
            _git_commit_fn=lambda msg, cwd: None,
            _git_revert_fn=mock_revert,
            _git_head_fn=lambda cwd: "aaa",
            _git_diff_content_fn=lambda cwd: "",
        )
        assert record.decision == "REVERTED"
        assert reverted["called"]

    def test_check_revert_on_gaming(self, tmp_path: Path):
        cfg = _config(tmp_path)
        _save_session(cfg, _baseline(), str(tmp_path))

        record = enforce_check(
            "gaming change",
            project_root=str(tmp_path),
            _run_tests_fn=lambda cmd, cwd: _improved(),
            _audit_fn=lambda diff, files_modified, allow_test_modification: (
                "GAMING", ["test_file_modification"], ["test_file_modification"]
            ),
            _git_diff_fn=lambda cwd: "",
            _git_names_fn=lambda cwd: ["tests/test_foo.py"],
            _git_commit_fn=lambda msg, cwd: None,
            _git_revert_fn=lambda h, cwd: True,
            _git_head_fn=lambda cwd: "bbb",
            _git_diff_content_fn=lambda cwd: "",
        )
        assert record.decision == "REVERTED"
        assert record.audit_verdict == "GAMING"

    def test_sequential_iterations(self, tmp_path: Path):
        cfg = _config(tmp_path)
        _save_session(cfg, _baseline(), str(tmp_path))

        stubs = dict(
            _run_tests_fn=lambda cmd, cwd: _improved(),
            _audit_fn=lambda diff, files_modified, allow_test_modification: (
                "VALID", [], []
            ),
            _git_diff_fn=lambda cwd: "",
            _git_names_fn=lambda cwd: ["a.py"],
            _git_commit_fn=lambda msg, cwd: "c1",
            _git_revert_fn=lambda h, cwd: True,
            _git_head_fn=lambda cwd: "h1",
            _git_diff_content_fn=lambda cwd: "",
        )
        r1 = enforce_check("first", project_root=str(tmp_path), **stubs)
        r2 = enforce_check("second", project_root=str(tmp_path), **stubs)
        assert r1.iteration == 1
        assert r2.iteration == 2

    def test_check_no_session(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            enforce_check(project_root=str(tmp_path))


class TestEnforceDiff:
    def test_diff_preview_shows_rejection_for_large_change(self, tmp_path: Path):
        cfg = EnforceConfig(
            test_command="pytest",
            project_root=str(tmp_path),
            max_diff_lines=1,
        )
        _save_session(cfg, _baseline(), str(tmp_path))

        result = enforce_diff(
            project_root=str(tmp_path),
            _git_names_fn=lambda cwd: ["src/foo.py"],
            _git_diff_content_fn=lambda cwd: (
                "diff --git a/src/foo.py b/src/foo.py\n"
                "--- a/src/foo.py\n"
                "+++ b/src/foo.py\n"
                "@@ -1 +1 @@\n"
                "-old line\n"
                "+new line\n"
            ),
        )

        assert result["files"] == ["src/foo.py"]
        assert result["file_count"] == 1
        assert result["diff_line_count"] == 2
        assert result["would_reject"] is True
        assert "diff lines exceeds limit of 1" in result["reject_reason"]
        assert "+new line" in result["diff_lines"]


# ---------------------------------------------------------------------------
# enforce_status
# ---------------------------------------------------------------------------

class TestEnforceStatus:
    def test_status_after_init(self, tmp_path: Path):
        cfg = _config(tmp_path)
        _save_session(cfg, _baseline(), str(tmp_path))
        status = enforce_status(str(tmp_path))
        assert status["session_active"] is True
        assert status["iterations"] == 0
        assert status["baseline"]["passed"] == 8

    def test_status_after_iterations(self, tmp_path: Path):
        cfg = _config(tmp_path)
        _save_session(cfg, _baseline(), str(tmp_path))
        _save_iteration(ChangeRecord(
            iteration=1, decision="COMMITTED", audit_verdict="VALID",
            after=_improved(),
        ), str(tmp_path))
        _save_iteration(ChangeRecord(
            iteration=2, decision="REVERTED", audit_verdict="GAMING",
            after=_regressed(),
        ), str(tmp_path))
        status = enforce_status(str(tmp_path))
        assert status["iterations"] == 2
        assert status["committed"] == 1
        assert status["reverted"] == 1
        assert status["gaming_detected"] == 1


# ---------------------------------------------------------------------------
# enforce_reset
# ---------------------------------------------------------------------------

class TestEnforceReset:
    def test_reset_existing(self, tmp_path: Path):
        cfg = _config(tmp_path)
        _save_session(cfg, _baseline(), str(tmp_path))
        assert enforce_reset(str(tmp_path)) is True
        assert not _session_dir(str(tmp_path)).exists()

    def test_reset_nonexistent(self, tmp_path: Path):
        assert enforce_reset(str(tmp_path)) is False


# ---------------------------------------------------------------------------
# build_enforce_report
# ---------------------------------------------------------------------------

class TestBuildEnforceReport:
    def test_basic_report(self, tmp_path: Path):
        cfg = _config(tmp_path)
        _save_session(cfg, _baseline(), str(tmp_path))
        _save_iteration(ChangeRecord(
            iteration=1, decision="COMMITTED", audit_verdict="VALID",
            before=_baseline(), after=_improved(),
        ), str(tmp_path))
        report = build_enforce_report(str(tmp_path))
        assert report.total_iterations == 1
        assert report.committed_count == 1
        assert report.baseline_result is not None
        assert report.baseline_result.passed == 8

    def test_empty_report(self, tmp_path: Path):
        cfg = _config(tmp_path)
        _save_session(cfg, _baseline(), str(tmp_path))
        report = build_enforce_report(str(tmp_path))
        assert report.total_iterations == 0
        assert report.committed_count == 0

    def test_report_to_dict(self, tmp_path: Path):
        cfg = _config(tmp_path)
        _save_session(cfg, _baseline(), str(tmp_path))
        report = build_enforce_report(str(tmp_path))
        d = report.to_dict()
        assert "config" in d
        assert "baseline_result" in d
        assert d["total_iterations"] == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_all_tests_pass(self, tmp_path: Path):
        """Baseline with zero failures."""
        all_pass = TestResult(
            exit_code=0, passed=10, failed=0, errors=0, skipped=0,
            total=10, duration_seconds=1.0, output="10 passed",
        )
        cfg = _config(tmp_path)
        _save_session(cfg, all_pass, str(tmp_path))
        status = enforce_status(str(tmp_path))
        assert status["baseline"]["failed"] == 0

    def test_all_tests_fail(self, tmp_path: Path):
        """Baseline with all failures."""
        all_fail = TestResult(
            exit_code=1, passed=0, failed=10, errors=0, skipped=0,
            total=10, duration_seconds=1.0, output="10 failed",
        )
        cfg = _config(tmp_path)
        _save_session(cfg, all_fail, str(tmp_path))
        status = enforce_status(str(tmp_path))
        assert status["baseline"]["passed"] == 0

    def test_zero_tests(self, tmp_path: Path):
        """No tests at all."""
        empty = TestResult(
            exit_code=0, passed=0, failed=0, errors=0, skipped=0,
            total=0, duration_seconds=0.0, output="no tests ran",
        )
        cfg = _config(tmp_path)
        _save_session(cfg, empty, str(tmp_path))
        status = enforce_status(str(tmp_path))
        assert status["baseline"]["total"] == 0

    def test_empty_diff_check(self, tmp_path: Path):
        """Check with no actual changes."""
        cfg = _config(tmp_path)
        _save_session(cfg, _baseline(), str(tmp_path))
        record = enforce_check(
            project_root=str(tmp_path),
            _run_tests_fn=lambda cmd, cwd: _baseline(),
            _audit_fn=lambda diff, files_modified, allow_test_modification: (
                "VALID", ["No gaming"], []
            ),
            _git_diff_fn=lambda cwd: "",
            _git_names_fn=lambda cwd: [],
            _git_commit_fn=lambda msg, cwd: "c1",
            _git_revert_fn=lambda h, cwd: True,
            _git_head_fn=lambda cwd: "aaa",
            _git_diff_content_fn=lambda cwd: "",
        )
        assert record.files_modified == []
        assert record.net_improvement == 0
