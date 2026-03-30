from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from agent_xray.diagnose import (
    FixPlanEntry,
    build_fix_plan,
    format_fix_plan_text,
    get_target_resolver,
    list_all_targets,
    register_target_resolver,
    validate_fix_targets,
)
from agent_xray.root_cause import ROOT_CAUSES, RootCauseResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result(task_id: str, root_cause: str, score: int, **kwargs) -> RootCauseResult:
    return RootCauseResult(
        task_id=task_id,
        root_cause=root_cause,
        grade="BROKEN",
        score=score,
        confidence="high",
        evidence=kwargs.get("evidence", [f"test evidence for {root_cause}"]),
        prompt_section=kwargs.get("prompt_section"),
        prompt_fix_hint=kwargs.get("prompt_fix_hint"),
    )


def _entry(targets: list[str], evidence: list[str] | None = None) -> FixPlanEntry:
    return FixPlanEntry(
        priority=1,
        root_cause="test",
        count=1,
        impact=5,
        severity=3,
        investigate_task="t1",
        targets=targets,
        fix_hint="fix it",
        verify_command="agent-xray surface t1",
        evidence=list(evidence or []),
    )


# ---------------------------------------------------------------------------
# validate_fix_targets
# ---------------------------------------------------------------------------


class TestValidateFixTargets:
    def test_valid_paths(self, tmp_path: Path) -> None:
        """Targets that exist on disk should not generate stale warnings."""
        (tmp_path / "intent").mkdir()
        (tmp_path / "intent" / "pipeline.py").write_text("# code")
        (tmp_path / "config.json").write_text("{}")

        entry = _entry(["intent/pipeline.py", "config.json"])
        result = validate_fix_targets([entry], project_root=tmp_path)
        assert result is not None
        assert len(result) == 1
        assert not any("STALE_TARGET" in e for e in result[0].evidence)

    def test_stale_paths(self, tmp_path: Path) -> None:
        """Missing files should get STALE_TARGET warnings in evidence."""
        entry = _entry(["services/llm/prompts/sections/research.py"])
        result = validate_fix_targets([entry], project_root=tmp_path)
        assert len(result) == 1
        stale = [e for e in result[0].evidence if "STALE_TARGET" in e]
        assert len(stale) == 1
        assert "research.py" in stale[0]

    def test_no_project_root_passthrough(self) -> None:
        """When project_root is None, entries pass through unchanged."""
        entry = _entry(["nonexistent/file.py"])
        original_evidence = list(entry.evidence)
        result = validate_fix_targets([entry], project_root=None)
        assert result[0].evidence == original_evidence

    def test_nonexistent_project_root(self, tmp_path: Path) -> None:
        """A non-directory project_root should be a no-op."""
        bogus = tmp_path / "does_not_exist"
        entry = _entry(["some/file.py"])
        original_evidence = list(entry.evidence)
        result = validate_fix_targets([entry], project_root=bogus)
        assert result[0].evidence == original_evidence

    def test_description_targets_skipped(self, tmp_path: Path) -> None:
        """Targets that look like descriptions (no slash, no extension) are skipped."""
        entry = _entry([
            "tool registry / tool-routing rules",
            "approval or permission policy",
            "some description without path",
        ])
        original_evidence = list(entry.evidence)
        result = validate_fix_targets([entry], project_root=tmp_path)
        assert result[0].evidence == original_evidence

    def test_mixed_path_and_description(self, tmp_path: Path) -> None:
        """Only path-like targets get validated; descriptions pass through."""
        (tmp_path / "real").mkdir()
        (tmp_path / "real" / "file.py").write_text("# exists")

        entry = _entry([
            "real/file.py",
            "nonexistent/module.py",
            "a plain description",
        ])
        result = validate_fix_targets([entry], project_root=tmp_path)
        stale = [e for e in result[0].evidence if "STALE_TARGET" in e]
        assert len(stale) == 1
        assert "nonexistent/module.py" in stale[0]

    def test_multiple_stale_targets(self, tmp_path: Path) -> None:
        """Multiple stale paths each produce their own warning."""
        entry = _entry(["a/b.py", "c/d.js", "e/f.ts"])
        result = validate_fix_targets([entry], project_root=tmp_path)
        stale = [e for e in result[0].evidence if "STALE_TARGET" in e]
        assert len(stale) == 3

    def test_non_code_extension_skipped(self, tmp_path: Path) -> None:
        """Paths with non-code extensions should not be validated."""
        entry = _entry(["images/logo.png", "data/output.csv"])
        original_evidence = list(entry.evidence)
        result = validate_fix_targets([entry], project_root=tmp_path)
        assert result[0].evidence == original_evidence

    def test_all_code_extensions(self, tmp_path: Path) -> None:
        """All supported code extensions should trigger validation."""
        extensions = [".py", ".js", ".ts", ".json", ".yaml", ".yml", ".toml", ".cfg", ".md"]
        targets = [f"dir/file{ext}" for ext in extensions]
        entry = _entry(targets)
        result = validate_fix_targets([entry], project_root=tmp_path)
        stale = [e for e in result[0].evidence if "STALE_TARGET" in e]
        assert len(stale) == len(extensions)

    def test_string_project_root(self, tmp_path: Path) -> None:
        """project_root as a string should work the same as Path."""
        entry = _entry(["missing/file.py"])
        result = validate_fix_targets([entry], project_root=str(tmp_path))
        stale = [e for e in result[0].evidence if "STALE_TARGET" in e]
        assert len(stale) == 1


# ---------------------------------------------------------------------------
# list_all_targets
# ---------------------------------------------------------------------------


class TestListAllTargets:
    def test_returns_all_root_causes(self) -> None:
        """list_all_targets should return entries for all known root causes."""
        result = list_all_targets()
        # Every root cause in ROOT_CAUSES that has FIX_TARGETS should appear
        for cause in ROOT_CAUSES:
            # The default resolver returns targets for all defined root causes
            assert cause in result, f"missing root cause: {cause}"

    def test_custom_resolver(self) -> None:
        """list_all_targets should work with a provided resolver."""

        class TestResolver:
            def resolve(self, root_cause: str, evidence: list[str]) -> list[str]:
                return [f"target_for_{root_cause}"]

        result = list_all_targets(TestResolver())
        for cause in ROOT_CAUSES:
            assert result[cause] == [f"target_for_{cause}"]

    def test_empty_resolver(self) -> None:
        """Root causes returning empty targets should be excluded."""

        class EmptyResolver:
            def resolve(self, root_cause: str, evidence: list[str]) -> list[str]:
                return []

        result = list_all_targets(EmptyResolver())
        assert result == {}


# ---------------------------------------------------------------------------
# format_fix_plan_text stale warnings
# ---------------------------------------------------------------------------


class TestFormatStaleWarnings:
    def test_stale_target_shown_prominently(self, tmp_path: Path) -> None:
        """STALE_TARGET evidence should render with the warning marker."""
        entry = _entry(["missing/module.py"])
        validate_fix_targets([entry], project_root=tmp_path)
        text = format_fix_plan_text([entry])
        assert "STALE TARGET" in text
        assert "missing/module.py" in text

    def test_no_stale_no_warning(self, tmp_path: Path) -> None:
        """When no targets are stale, the warning marker should not appear."""
        (tmp_path / "real").mkdir()
        (tmp_path / "real" / "file.py").write_text("# ok")
        entry = _entry(["real/file.py"])
        validate_fix_targets([entry], project_root=tmp_path)
        text = format_fix_plan_text([entry])
        assert "STALE TARGET" not in text

    def test_stale_evidence_not_duplicated(self, tmp_path: Path) -> None:
        """STALE_TARGET evidence should appear in warnings, not in Evidence line."""
        entry = _entry(["missing/module.py"], evidence=["some real evidence"])
        validate_fix_targets([entry], project_root=tmp_path)
        text = format_fix_plan_text([entry])
        lines = text.split("\n")
        evidence_lines = [l for l in lines if l.strip().startswith("Evidence:")]
        # The Evidence line should show original evidence but not STALE_TARGET
        for line in evidence_lines:
            assert "STALE_TARGET" not in line
        # But the stale warning line should still be there
        assert any("STALE TARGET" in l for l in lines)


# ---------------------------------------------------------------------------
# CLI --project-root flag parsing
# ---------------------------------------------------------------------------


class TestCliProjectRoot:
    def test_diagnose_parser_has_project_root(self) -> None:
        """The diagnose subcommand should accept --project-root."""
        from agent_xray.cli import build_parser

        parser = build_parser()
        # Parse with a dummy log_dir (won't actually run)
        args = parser.parse_args(["diagnose", ".", "--project-root", "/some/path"])
        assert args.project_root == "/some/path"

    def test_diagnose_parser_project_root_optional(self) -> None:
        """--project-root should be optional."""
        from agent_xray.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["diagnose", "."])
        assert getattr(args, "project_root", None) is None

    def test_validate_targets_parser_exists(self) -> None:
        """The validate-targets subcommand should exist."""
        from agent_xray.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(
            ["validate-targets", "--project-root", "/some/path"]
        )
        assert args.project_root == "/some/path"

    def test_validate_targets_resolver_flag(self) -> None:
        """validate-targets should accept --resolver."""
        from agent_xray.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(
            ["validate-targets", "--project-root", "/p", "--resolver", "custom"]
        )
        assert args.resolver == "custom"


# ---------------------------------------------------------------------------
# validate-targets command output and exit code
# ---------------------------------------------------------------------------


class TestValidateTargetsCommand:
    def test_exit_code_zero_when_all_valid(self, tmp_path: Path) -> None:
        """Exit code 0 when no path-like targets are stale."""
        from agent_xray.cli import cmd_validate_targets

        args = type("Args", (), {
            "project_root": str(tmp_path),
            "resolver": None,
        })()
        # Default targets are descriptions (no slashes with code extensions),
        # so all should pass
        code = cmd_validate_targets(args)
        assert code == 0

    def test_exit_code_one_when_stale(self, tmp_path: Path) -> None:
        """Exit code 1 when a path-like resolver returns stale paths."""

        class StaleResolver:
            def resolve(self, root_cause: str, evidence: list[str]) -> list[str]:
                if root_cause == "routing_bug":
                    return ["nonexistent/file.py"]
                return ["tool registry"]

        register_target_resolver("_test_stale", StaleResolver())

        args = type("Args", (), {
            "project_root": str(tmp_path),
            "resolver": "_test_stale",
        })()

        from agent_xray.cli import cmd_validate_targets

        code = cmd_validate_targets(args)
        assert code == 1

    def test_output_format(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Output should include TARGET VALIDATION header and Summary line."""

        class MixedResolver:
            def resolve(self, root_cause: str, evidence: list[str]) -> list[str]:
                if root_cause == "routing_bug":
                    return ["nonexistent/file.py", "a description"]
                return []

        register_target_resolver("_test_mixed", MixedResolver())

        args = type("Args", (), {
            "project_root": str(tmp_path),
            "resolver": "_test_mixed",
        })()

        from agent_xray.cli import cmd_validate_targets

        cmd_validate_targets(args)
        captured = capsys.readouterr()
        assert "TARGET VALIDATION" in captured.out
        assert "Summary:" in captured.out
        assert "[STALE]" in captured.out
        assert "[OK]" in captured.out

    def test_missing_project_root_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Should print error and return 1 when no project root given."""
        # Ensure env var is also unset
        env_backup = os.environ.pop("AGENT_XRAY_PROJECT_ROOT", None)
        try:
            args = type("Args", (), {
                "project_root": None,
                "resolver": None,
            })()

            from agent_xray.cli import cmd_validate_targets

            code = cmd_validate_targets(args)
            assert code == 1
            captured = capsys.readouterr()
            output = (captured.out + captured.err).lower()
            assert "project-root" in output or "project_root" in output
        finally:
            if env_backup is not None:
                os.environ["AGENT_XRAY_PROJECT_ROOT"] = env_backup


# ---------------------------------------------------------------------------
# Integration: build_fix_plan + validate_fix_targets
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_build_then_validate(self, tmp_path: Path) -> None:
        """Full pipeline: build a fix plan, then validate targets."""
        results = [_result("t1", "spin", -5)]
        plan = build_fix_plan(results)
        # Default targets for spin are descriptions, not paths
        validate_fix_targets(plan, project_root=tmp_path)
        # Should have no stale warnings since targets are descriptions
        assert not any(
            "STALE_TARGET" in e
            for entry in plan
            for e in entry.evidence
        )

    def test_build_validate_with_path_resolver(self, tmp_path: Path) -> None:
        """Custom resolver returning file paths should get validated."""

        class PathResolver:
            def resolve(self, root_cause: str, evidence: list[str]) -> list[str]:
                return ["intent/pipeline.py", "missing/module.py"]

        results = [_result("t1", "spin", -5)]
        plan = build_fix_plan(results, target_resolver=PathResolver())

        # Create one of the two target files
        (tmp_path / "intent").mkdir()
        (tmp_path / "intent" / "pipeline.py").write_text("# ok")

        validate_fix_targets(plan, project_root=tmp_path)
        stale = [
            e for entry in plan for e in entry.evidence if "STALE_TARGET" in e
        ]
        assert len(stale) == 1
        assert "missing/module.py" in stale[0]
