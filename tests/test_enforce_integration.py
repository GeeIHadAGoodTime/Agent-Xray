from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from agent_xray.enforce import (
    EnforceConfig,
    build_enforce_report,
    enforce_challenge,
    enforce_check,
    enforce_init,
    enforce_reset,
    enforce_status,
)


def _run_git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def _pytest_command() -> str:
    return subprocess.list2cmdline([sys.executable, "-m", "pytest", "tests", "-q"])


@pytest.mark.integration
def test_enforce_end_to_end_real_git_repo() -> None:
    repo_dir = Path(tempfile.mkdtemp(prefix="agent-xray-enforce-"))
    try:
        _run_git(repo_dir, "init")
        _run_git(repo_dir, "config", "user.name", "Agent Xray Tests")
        _run_git(repo_dir, "config", "user.email", "agent-xray-tests@example.com")

        (repo_dir / "calc.py").write_text(
            "def add(a: int, b: int) -> int:\n"
            "    return a - b\n",
            encoding="utf-8",
        )
        tests_dir = repo_dir / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_calc.py").write_text(
            "from calc import add\n\n\n"
            "def test_add() -> None:\n"
            "    assert add(2, 3) == 5\n",
            encoding="utf-8",
        )

        _run_git(repo_dir, "add", ".")
        _run_git(repo_dir, "commit", "-m", "Initial broken fixture")

        config = EnforceConfig(
            test_command=_pytest_command(),
            project_root=str(repo_dir),
            max_iterations=5,
        )

        baseline, session_dir = enforce_init(config)
        assert session_dir.exists()
        assert baseline.failed == 1

        (repo_dir / "calc.py").write_text(
            "def add(a: int, b: int) -> int:\n"
            "    return a + b\n",
            encoding="utf-8",
        )

        record = enforce_check(
            hypothesis="Fix add arithmetic bug",
            project_root=str(repo_dir),
        )
        assert record.decision == "COMMITTED"
        assert record.after is not None
        assert record.after.failed == 0
        assert record.commit_hash

        status = enforce_status(str(repo_dir))
        assert status["iterations"] == 1
        assert status["committed"] == 1

        challenge = enforce_challenge(str(repo_dir))
        assert challenge.changes_reviewed == 1
        assert challenge.iteration_range == (1, 1)
        assert challenge.findings

        report = build_enforce_report(str(repo_dir))
        assert report.total_iterations == 1
        assert report.changes
        assert report.final_result is not None
        assert report.final_result.failed == 0

        assert enforce_reset(str(repo_dir)) is True
        assert not session_dir.exists()
    finally:
        shutil.rmtree(repo_dir, ignore_errors=True)
