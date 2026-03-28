"""Enforcement engine -- controlled experiment loop for disciplined agent changes.

Provides session management, test execution, git integration, and iteration
tracking for agents making incremental, empirically-verified code changes.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DiffHunk:
    """A parsed hunk from a unified diff showing what changed at line level."""

    file: str
    line_number: int
    removed_lines: list[str] = field(default_factory=list)
    added_lines: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiffHunk:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class EnforceConfig:
    """Configuration for an enforcement session."""

    test_command: str
    max_iterations: int = 50
    challenge_every: int = 5
    require_improvement: bool = True
    allow_test_modification: bool = False
    git_auto_commit: bool = True
    git_auto_revert: bool = True
    baseline_command: str | None = None
    project_root: str | None = None
    report_path: str | None = None
    stash_first: bool = False
    # GAP 7: Change-size enforcement
    max_files_per_change: int = 5
    max_diff_lines: int = 200
    # GAP 3: Project-rule audit awareness
    rules_file: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnforceConfig:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class TestResult:
    """Captured test suite results."""

    exit_code: int
    passed: int
    failed: int
    errors: int
    skipped: int
    total: int
    duration_seconds: float
    output: str
    test_names_passed: list[str] = field(default_factory=list)
    test_names_failed: list[str] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TestResult:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class ChangeRecord:
    """One iteration of the enforcement loop."""

    iteration: int
    # What changed
    files_modified: list[str] = field(default_factory=list)
    diff_summary: str = ""
    diff_stat: str = ""
    hypothesis: str = ""
    # Measurement
    before: TestResult | None = None
    after: TestResult | None = None
    tests_improved: list[str] = field(default_factory=list)
    tests_regressed: list[str] = field(default_factory=list)
    tests_unchanged: int = 0
    net_improvement: int = 0
    # Audit
    audit_verdict: str = ""
    audit_reasons: list[str] = field(default_factory=list)
    gaming_signals: list[str] = field(default_factory=list)
    # Decision
    decision: str = ""
    commit_hash: str | None = None
    # Timing
    started_at: str = ""
    completed_at: str = ""
    # GAP 2: Meta-analysis of successful changes
    meta_analysis: dict[str, Any] = field(default_factory=dict)
    # GAP 4: Line-item diff detail
    diff_hunks: list[dict[str, Any]] = field(default_factory=list)
    # GAP 5: Change quality classification
    change_quality: str = ""
    # GAP 8/9: Prediction tracking
    prediction_accuracy: str = ""  # "accurate", "partial", "wrong", "no_prediction"
    predicted_tests: list[str] = field(default_factory=list)
    actual_improved: list[str] = field(default_factory=list)
    prediction_match_pct: float = 0.0
    # GAP 12: Regression root cause
    regression_root_cause: str = ""
    # GAP 3: Rule violations
    rule_violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChangeRecord:
        d = dict(data)
        if "before" in d and isinstance(d["before"], dict):
            d["before"] = TestResult.from_dict(d["before"])
        if "after" in d and isinstance(d["after"], dict):
            d["after"] = TestResult.from_dict(d["after"])
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ChallengeResult:
    """Result of an adversarial review."""

    iteration_range: tuple[int, int] = (0, 0)
    changes_reviewed: int = 0
    vetoed: list[int] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration_range": list(self.iteration_range),
            "changes_reviewed": self.changes_reviewed,
            "vetoed": self.vetoed,
            "findings": self.findings,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChallengeResult:
        d = dict(data)
        if "iteration_range" in d:
            ir = d["iteration_range"]
            d["iteration_range"] = tuple(ir) if isinstance(ir, (list, tuple)) else (0, 0)
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class EnforceReport:
    """Complete enforcement session report."""

    config: EnforceConfig
    changes: list[ChangeRecord] = field(default_factory=list)
    challenges: list[ChallengeResult] = field(default_factory=list)
    total_iterations: int = 0
    committed_count: int = 0
    reverted_count: int = 0
    vetoed_count: int = 0
    rejected_count: int = 0
    gaming_detected_count: int = 0
    baseline_result: TestResult | None = None
    final_result: TestResult | None = None
    net_improvement: int = 0
    duration_seconds: float = 0.0
    # GAP 13: Cumulative report improvements
    prediction_accuracy_summary: dict[str, Any] = field(default_factory=dict)
    efficiency_ratio: float = 0.0
    change_map: dict[str, int] = field(default_factory=dict)
    cumulative_diff: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "changes": [c.to_dict() for c in self.changes],
            "challenges": [c.to_dict() for c in self.challenges],
            "total_iterations": self.total_iterations,
            "committed_count": self.committed_count,
            "reverted_count": self.reverted_count,
            "vetoed_count": self.vetoed_count,
            "rejected_count": self.rejected_count,
            "gaming_detected_count": self.gaming_detected_count,
            "baseline_result": self.baseline_result.to_dict() if self.baseline_result else None,
            "final_result": self.final_result.to_dict() if self.final_result else None,
            "net_improvement": self.net_improvement,
            "duration_seconds": self.duration_seconds,
            "prediction_accuracy_summary": self.prediction_accuracy_summary,
            "efficiency_ratio": self.efficiency_ratio,
            "change_map": self.change_map,
            "cumulative_diff": self.cumulative_diff,
        }


# ---------------------------------------------------------------------------
# Test output parsing
# ---------------------------------------------------------------------------

# Individual field patterns for pytest summary (order-independent)
_PYTEST_PASSED_FIELD = re.compile(r"(\d+)\s+passed")
_PYTEST_FAILED_FIELD = re.compile(r"(\d+)\s+failed")
_PYTEST_ERRORS_FIELD = re.compile(r"(\d+)\s+errors?")
_PYTEST_SKIPPED_FIELD = re.compile(r"(\d+)\s+skipped")
_PYTEST_DESELECTED_FIELD = re.compile(r"(\d+)\s+deselected")
_PYTEST_WARNINGS_FIELD = re.compile(r"(\d+)\s+warnings?")
_PYTEST_DURATION_FIELD = re.compile(r"in\s+([\d.]+)s")
# A line is a pytest summary if it has "passed", "failed", or "error" with counts
_PYTEST_SUMMARY_LINE = re.compile(r"\d+\s+(?:passed|failed|errors?)")

_PYTEST_PASSED_RE = re.compile(
    r"^\s*(?:PASSED|\.)\s+(\S+)",
    re.MULTILINE,
)

_PYTEST_FAILED_RE = re.compile(
    r"^\s*(?:FAILED|F)\s+(\S+)",
    re.MULTILINE,
)

# More reliable: look for lines like "tests/test_foo.py::test_bar PASSED"
_PYTEST_VERBOSE_RESULT_RE = re.compile(
    r"^(\S+::\S+)\s+(PASSED|FAILED|ERROR|SKIPPED)",
    re.MULTILINE,
)
_PYTEST_SHORT_SUMMARY_RESULT_RE = re.compile(
    r"^(FAILED|ERROR)\s+(\S+)",
    re.MULTILINE,
)

# Generic number extraction pattern
_GENERIC_NUMBERS_RE = re.compile(
    r"(\d+)\s+(?:passed|pass|ok|success)",
    re.IGNORECASE,
)
_GENERIC_FAIL_RE = re.compile(
    r"(\d+)\s+(?:failed|fail|error|failure)",
    re.IGNORECASE,
)


def _append_unique_name(names: list[str], name: str) -> None:
    """Append a test name once while preserving order."""
    if name and name not in names:
        names.append(name)


def parse_test_output(output: str, exit_code: int) -> TestResult:
    """Parse test runner output into structured results.

    Tries pytest format first, then generic patterns, then exit-code-only.
    """
    ts = datetime.now(timezone.utc).isoformat()

    # Try pytest verbose output first for individual test names
    passed_names: list[str] = []
    failed_names: list[str] = []
    for m in _PYTEST_VERBOSE_RESULT_RE.finditer(output):
        name, status = m.group(1), m.group(2)
        if status == "PASSED":
            _append_unique_name(passed_names, name)
        elif status in ("FAILED", "ERROR"):
            _append_unique_name(failed_names, name)

    # pytest -q still exposes failing node IDs in "short test summary info".
    for m in _PYTEST_SHORT_SUMMARY_RESULT_RE.finditer(output):
        status, name = m.group(1), m.group(2)
        if status in ("FAILED", "ERROR"):
            _append_unique_name(failed_names, name)

    # Try pytest summary line (search from bottom up)
    passed = 0
    failed = 0
    errors = 0
    skipped = 0
    duration = 0.0
    found_pytest = False

    for line in reversed(output.splitlines()):
        if _PYTEST_SUMMARY_LINE.search(line):
            m_p = _PYTEST_PASSED_FIELD.search(line)
            m_f = _PYTEST_FAILED_FIELD.search(line)
            m_e = _PYTEST_ERRORS_FIELD.search(line)
            m_s = _PYTEST_SKIPPED_FIELD.search(line)
            m_d = _PYTEST_DURATION_FIELD.search(line)
            if m_p or m_f or m_e:
                passed = int(m_p.group(1)) if m_p else 0
                failed = int(m_f.group(1)) if m_f else 0
                errors = int(m_e.group(1)) if m_e else 0
                skipped = int(m_s.group(1)) if m_s else 0
                duration = float(m_d.group(1)) if m_d else 0.0
                found_pytest = True
                break

    if found_pytest:
        total = passed + failed + errors + skipped
        return TestResult(
            exit_code=exit_code,
            passed=passed,
            failed=failed,
            errors=errors,
            skipped=skipped,
            total=total,
            duration_seconds=duration,
            output=output,
            test_names_passed=passed_names,
            test_names_failed=failed_names,
            timestamp=ts,
        )

    # Try generic patterns
    generic_passed_m = _GENERIC_NUMBERS_RE.search(output)
    generic_failed_m = _GENERIC_FAIL_RE.search(output)
    if generic_passed_m or generic_failed_m:
        gp = int(generic_passed_m.group(1)) if generic_passed_m else 0
        gf = int(generic_failed_m.group(1)) if generic_failed_m else 0
        return TestResult(
            exit_code=exit_code,
            passed=gp,
            failed=gf,
            errors=0,
            skipped=0,
            total=gp + gf,
            duration_seconds=0.0,
            output=output,
            test_names_passed=passed_names,
            test_names_failed=failed_names,
            timestamp=ts,
        )

    # Fallback: exit code only
    is_pass = exit_code == 0
    return TestResult(
        exit_code=exit_code,
        passed=1 if is_pass else 0,
        failed=0 if is_pass else 1,
        errors=0,
        skipped=0,
        total=1,
        duration_seconds=0.0,
        output=output,
        test_names_passed=passed_names,
        test_names_failed=failed_names,
        timestamp=ts,
    )


# ---------------------------------------------------------------------------
# Shell / git helpers
# ---------------------------------------------------------------------------

def _run_shell(command: str, cwd: str | None = None) -> tuple[int, str]:
    """Run a shell command and return (exit_code, combined_output)."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=False,
            cwd=cwd,
            timeout=600,
        )
        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")
        return result.returncode, stdout + stderr
    except subprocess.TimeoutExpired:
        return 1, "Command timed out after 600 seconds"
    except Exception as e:
        return 1, str(e)


def _filter_git_output_lines(output: str) -> list[str]:
    """Drop git warning noise that should not be treated as diff content."""
    filtered: list[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("warning:"):
            continue
        filtered.append(stripped)
    return filtered


def _git_diff_stat(cwd: str | None = None) -> str:
    """Get git diff --stat for staged and unstaged changes."""
    _, out = _run_shell(
        f"git diff --stat HEAD -- . ':!{SESSION_DIR_NAME}'", cwd,
    )
    return "\n".join(_filter_git_output_lines(out))


def _git_diff_names(cwd: str | None = None) -> list[str]:
    """Get list of modified file names."""
    _, out = _run_shell(
        f"git diff --name-only HEAD -- . ':!{SESSION_DIR_NAME}'", cwd,
    )
    return _filter_git_output_lines(out)


def _git_stage_all(cwd: str | None = None) -> bool:
    code, _ = _run_shell("git add -A", cwd)
    return code == 0


def _git_commit(message: str, cwd: str | None = None) -> str | None:
    """Commit staged changes. Returns commit hash or None."""
    _git_stage_all(cwd)
    code, out = _run_shell(f'git commit -m "{message}"', cwd)
    if code != 0:
        return None
    # Extract commit hash
    _, hash_out = _run_shell("git rev-parse --short HEAD", cwd)
    return hash_out.strip() or None


def _git_revert_to(commit_hash: str, cwd: str | None = None) -> bool:
    """Hard reset to a given commit."""
    code, _ = _run_shell(f"git reset --hard {commit_hash}", cwd)
    return code == 0


def _git_head_hash(cwd: str | None = None) -> str:
    _, out = _run_shell("git rev-parse --short HEAD", cwd)
    return out.strip()


def _git_stash(cwd: str | None = None) -> bool:
    code, _ = _run_shell("git stash", cwd)
    return code == 0


def _git_stash_pop(cwd: str | None = None) -> bool:
    code, _ = _run_shell("git stash pop", cwd)
    return code == 0


def _git_has_uncommitted_changes(cwd: str | None = None) -> bool:
    """Return True when the repo has tracked or untracked changes."""
    _, out = _run_shell(
        f"git status --porcelain --untracked-files=all -- . ':!{SESSION_DIR_NAME}'",
        cwd,
    )
    return bool(_filter_git_output_lines(out))


def _git_diff_content(cwd: str | None = None) -> str:
    """Get full diff content."""
    _, out = _run_shell(
        f"git diff HEAD -- . ':!{SESSION_DIR_NAME}'", cwd,
    )
    return out


# ---------------------------------------------------------------------------
# Session directory management
# ---------------------------------------------------------------------------

SESSION_DIR_NAME = ".agent-xray-enforce"


def _session_dir(project_root: str | None = None) -> Path:
    root = Path(project_root) if project_root else Path.cwd()
    return root / SESSION_DIR_NAME


def _ensure_session_dir(project_root: str | None = None) -> Path:
    d = _session_dir(project_root)
    d.mkdir(parents=True, exist_ok=True)
    (d / "iterations").mkdir(exist_ok=True)
    (d / "challenges").mkdir(exist_ok=True)
    return d


def _maybe_add_session_dir_to_gitignore(project_root: str | None = None) -> None:
    """Append the session directory to .gitignore when the file already exists."""
    root = Path(project_root) if project_root else Path.cwd()
    gitignore_path = root / ".gitignore"
    if not gitignore_path.exists():
        return

    content = gitignore_path.read_text(encoding="utf-8")
    if SESSION_DIR_NAME in content:
        return

    prefix = ""
    if content and not content.endswith(("\n", "\r")):
        prefix = "\n"
    gitignore_path.write_text(
        f"{content}{prefix}{SESSION_DIR_NAME}/\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------

def _save_session(
    config: EnforceConfig,
    baseline: TestResult,
    project_root: str | None,
    *,
    stash_saved: bool = False,
) -> Path:
    """Save session configuration and baseline."""
    sd = _ensure_session_dir(project_root)
    session_data = {
        "config": config.to_dict(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "iteration_count": 0,
        "head_hash_at_init": _git_head_hash(project_root),
        "stash_saved": stash_saved,
    }
    (sd / "session.json").write_text(json.dumps(session_data, indent=2), encoding="utf-8")
    (sd / "baseline.json").write_text(json.dumps(baseline.to_dict(), indent=2), encoding="utf-8")
    return sd


def _load_session(project_root: str | None = None) -> tuple[EnforceConfig, TestResult, dict[str, Any]]:
    """Load session config and baseline. Raises FileNotFoundError if not initialized."""
    sd = _session_dir(project_root)
    session_path = sd / "session.json"
    baseline_path = sd / "baseline.json"
    if not session_path.exists():
        raise FileNotFoundError(
            f"No enforcement session found in {sd}. Run 'agent-xray enforce init' first."
        )
    session_data = json.loads(session_path.read_text(encoding="utf-8"))
    config = EnforceConfig.from_dict(session_data["config"])
    baseline = TestResult.from_dict(json.loads(baseline_path.read_text(encoding="utf-8")))
    return config, baseline, session_data


def _update_session_iteration_count(project_root: str | None, count: int) -> None:
    sd = _session_dir(project_root)
    session_path = sd / "session.json"
    data = json.loads(session_path.read_text(encoding="utf-8"))
    data["iteration_count"] = count
    session_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _save_iteration(record: ChangeRecord, project_root: str | None = None) -> Path:
    sd = _session_dir(project_root)
    path = sd / "iterations" / f"{record.iteration:03d}.json"
    path.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
    _update_session_iteration_count(project_root, record.iteration)
    return path


def _load_iterations(project_root: str | None = None) -> list[ChangeRecord]:
    sd = _session_dir(project_root)
    idir = sd / "iterations"
    if not idir.exists():
        return []
    records: list[ChangeRecord] = []
    for path in sorted(idir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        records.append(ChangeRecord.from_dict(data))
    return records


def _save_challenge(result: ChallengeResult, index: int, project_root: str | None = None) -> Path:
    sd = _session_dir(project_root)
    path = sd / "challenges" / f"challenge_{index:03d}.json"
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return path


def _load_challenges(project_root: str | None = None) -> list[ChallengeResult]:
    sd = _session_dir(project_root)
    cdir = sd / "challenges"
    if not cdir.exists():
        return []
    results: list[ChallengeResult] = []
    for path in sorted(cdir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        results.append(ChallengeResult.from_dict(data))
    return results


def _iteration_count(project_root: str | None = None) -> int:
    """Count existing iterations."""
    sd = _session_dir(project_root)
    idir = sd / "iterations"
    if not idir.exists():
        return 0
    return len(list(idir.glob("*.json")))


# ---------------------------------------------------------------------------
# Core enforcement operations
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Diff parsing (GAP 4)
# ---------------------------------------------------------------------------

_DIFF_FILE_RE = re.compile(r"^diff --git a/(.+?) b/(.+)$", re.MULTILINE)
_DIFF_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", re.MULTILINE)


def _parse_diff_hunks(diff: str) -> list[DiffHunk]:
    """Parse a unified diff into structured DiffHunk objects with line-level detail."""
    hunks: list[DiffHunk] = []
    if not diff or not diff.strip():
        return hunks

    current_file = ""
    lines = diff.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        # Detect file header
        file_match = _DIFF_FILE_RE.match(line)
        if file_match:
            current_file = file_match.group(2)
            i += 1
            continue

        # Detect hunk header
        hunk_match = _DIFF_HUNK_RE.match(line)
        if hunk_match:
            start_line = int(hunk_match.group(1))
            i += 1
            removed: list[str] = []
            added: list[str] = []
            line_num = start_line
            while i < len(lines):
                hline = lines[i]
                if hline.startswith("diff --git") or hline.startswith("@@"):
                    break
                if hline.startswith("-") and not hline.startswith("---"):
                    removed.append(f"L{line_num}: {hline}")
                elif hline.startswith("+") and not hline.startswith("+++"):
                    added.append(f"L{line_num}: {hline}")
                    line_num += 1
                else:
                    line_num += 1
                i += 1

            if removed or added:
                hunks.append(DiffHunk(
                    file=current_file,
                    line_number=start_line,
                    removed_lines=removed,
                    added_lines=added,
                ))
            continue

        i += 1

    return hunks


# ---------------------------------------------------------------------------
# Meta-analysis (GAP 2)
# ---------------------------------------------------------------------------

def _meta_analyze(
    before: TestResult,
    after: TestResult,
    diff: str,
    files: list[str],
) -> dict[str, Any]:
    """Analyze a successful change to classify its nature.

    Returns a dict with keys:
    - tests_fixed: list of tests that went fail->pass
    - tests_broken: list of tests that went pass->fail
    - localized: bool (one file vs many)
    - additive: bool (more added than removed)
    - classification: str
    """
    tests_fixed = sorted(set(after.test_names_passed) - set(before.test_names_passed))
    tests_broken = sorted(set(after.test_names_failed) - set(before.test_names_failed))

    localized = len(files) <= 1
    # Count added vs removed lines
    added = sum(1 for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff.splitlines() if line.startswith("-") and not line.startswith("---"))
    additive = added > removed

    # Classify
    test_files = [f for f in files if any(p in f.lower() for p in ("test_", "_test.", "tests/", "spec/"))]
    source_files = [f for f in files if f not in test_files]

    if test_files and not source_files:
        classification = "test_update"
    elif any(p in f.lower() for f in files for p in ("config", "settings", ".env", ".json", ".yaml", ".toml")):
        classification = "configuration_change"
    elif removed > added and tests_fixed:
        classification = "root_cause_fix"
    elif tests_fixed:
        classification = "root_cause_fix"
    elif not tests_fixed and not tests_broken and added > removed:
        classification = "symptom_patch"
    else:
        classification = "symptom_patch"

    return {
        "tests_fixed": tests_fixed,
        "tests_broken": tests_broken,
        "localized": localized,
        "additive": additive,
        "lines_added": added,
        "lines_removed": removed,
        "classification": classification,
        "source_files": source_files,
        "test_files": test_files,
    }


# ---------------------------------------------------------------------------
# Prediction plan (GAP 8)
# ---------------------------------------------------------------------------

def _save_plan(hypothesis: str, expected_tests: list[str], project_root: str | None) -> None:
    """Save the current iteration plan (hypothesis + expected tests)."""
    sd = _ensure_session_dir(project_root)
    plan_data = {
        "hypothesis": hypothesis,
        "expected_tests": expected_tests,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (sd / "current_plan.json").write_text(json.dumps(plan_data, indent=2), encoding="utf-8")


def _load_plan(project_root: str | None) -> dict[str, Any] | None:
    """Load the current plan if one exists."""
    sd = _session_dir(project_root)
    plan_path = sd / "current_plan.json"
    if not plan_path.exists():
        return None
    data = json.loads(plan_path.read_text(encoding="utf-8"))
    return data


def _clear_plan(project_root: str | None) -> None:
    """Clear the current plan after it's been consumed."""
    sd = _session_dir(project_root)
    plan_path = sd / "current_plan.json"
    if plan_path.exists():
        plan_path.unlink()


def _evaluate_prediction(
    plan: dict[str, Any] | None,
    improved: list[str],
    regressed: list[str],
) -> tuple[str, list[str], list[str], float]:
    """Compare actual results to the plan's predictions.

    Returns (accuracy, predicted_tests, actual_improved, match_pct).
    """
    if plan is None:
        return "no_prediction", [], list(improved), 0.0

    expected = set(plan.get("expected_tests", []))
    actual_improved_set = set(improved)

    if not expected:
        return "no_prediction", [], list(improved), 0.0

    correct = expected & actual_improved_set
    match_pct = len(correct) / len(expected) * 100 if expected else 0.0

    if expected == actual_improved_set:
        accuracy = "accurate"
    elif correct:
        accuracy = "partial"
    else:
        accuracy = "wrong"

    return accuracy, list(expected), list(improved), match_pct


# ---------------------------------------------------------------------------
# Regression root cause heuristic (GAP 12)
# ---------------------------------------------------------------------------

def _heuristic_regression_cause(diff: str, regressed: list[str], files: list[str]) -> str:
    """Provide a heuristic explanation for why a change caused regressions."""
    if not regressed:
        return ""
    causes: list[str] = []
    # Check if the diff modifies error handling
    if "except" in diff and ("pass" in diff or "return" in diff):
        causes.append("error handling modification may have masked real failures")
    # Check if imports were removed
    removed_imports = [l for l in diff.splitlines() if l.startswith("-") and ("import " in l or "from " in l)]
    if removed_imports:
        causes.append(f"removed {len(removed_imports)} import(s) which may be needed by regressed tests")
    # Check if function signatures changed
    if re.search(r"^[-+]\s*def\s+\w+\(", diff, re.MULTILINE):
        causes.append("function signature change may have broken callers")
    # Check if config/constants changed
    if any("config" in f.lower() or "constant" in f.lower() or "settings" in f.lower() for f in files):
        causes.append("configuration/constants change may have cascading effects")
    if not causes:
        causes.append(f"change to {', '.join(files[:3])} caused {len(regressed)} test(s) to regress")
    return "; ".join(causes)


# ---------------------------------------------------------------------------
# Project rules loading (GAP 3)
# ---------------------------------------------------------------------------

def _load_project_rules(rules_file: str | None) -> dict[str, Any] | None:
    """Load project rules from a JSON file. Returns None if not configured."""
    if not rules_file:
        return None
    path = Path(rules_file)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def run_tests(test_command: str, cwd: str | None = None) -> TestResult:
    """Run the test command and parse results."""
    exit_code, output = _run_shell(test_command, cwd)
    return parse_test_output(output, exit_code)


def compare_test_results(before: TestResult, after: TestResult) -> tuple[list[str], list[str], int]:
    """Compare two test results. Returns (improved, regressed, unchanged count).

    Improved = tests that went from failed to passed.
    Regressed = tests that went from passed to failed.
    """
    before_passed = set(before.test_names_passed)
    before_failed = set(before.test_names_failed)
    after_passed = set(after.test_names_passed)
    after_failed = set(after.test_names_failed)

    # Use either explicit pass transitions or disappearance/appearance in the
    # failed set. This keeps pytest -q output useful even without passed names.
    improved = sorted((after_passed - before_passed) | (before_failed - after_failed))
    regressed = sorted((after_failed - before_failed) | (before_passed - after_passed))

    # If we don't have individual test names, compare counts
    if not before_passed and not before_failed and not after_passed and not after_failed:
        pass_delta = after.passed - before.passed
        fail_delta = after.failed - before.failed
        if pass_delta > 0 and fail_delta <= 0:
            improved = [f"(+{pass_delta} passed)"]
        if fail_delta > 0:
            regressed = [f"(+{fail_delta} failed)"]

    all_tests = before_passed | before_failed | after_passed | after_failed
    unchanged = len(all_tests) - len(improved) - len(regressed)
    if unchanged < 0:
        unchanged = max(0, before.total - len(improved) - len(regressed))

    return improved, regressed, unchanged


def _diff_lines(diff_content: str) -> list[str]:
    """Return the raw unified diff as individual lines."""
    return diff_content.splitlines()


def _diff_line_count(diff_content: str) -> int:
    """Count only added/removed diff lines, excluding file headers."""
    return len(
        [
            line
            for line in diff_content.splitlines()
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
        ]
    )


def _change_reject_reason(
    config: EnforceConfig,
    files_modified: list[str],
    diff_line_count: int,
) -> str:
    """Return the size-based rejection reason, if any."""
    if len(files_modified) > config.max_files_per_change:
        return (
            f"Change too large: {len(files_modified)} files exceeds limit of "
            f"{config.max_files_per_change} -- break into smaller iterations"
        )
    if diff_line_count > config.max_diff_lines:
        return (
            f"Change too large: {diff_line_count} diff lines exceeds limit of "
            f"{config.max_diff_lines} -- break into smaller iterations"
        )
    return ""


def enforce_init(
    config: EnforceConfig,
    *,
    _run_tests_fn: Any = None,
) -> tuple[TestResult, Path]:
    """Initialize an enforcement session.

    Runs the test suite to capture a baseline and creates the session directory.
    Returns the baseline result and session directory path.
    """
    project_root = config.project_root
    runner = _run_tests_fn or run_tests
    sd = _ensure_session_dir(project_root)
    _maybe_add_session_dir_to_gitignore(project_root)

    stash_saved = False
    if config.stash_first and _git_has_uncommitted_changes(project_root):
        if not _git_stash(project_root):
            raise RuntimeError("Failed to stash uncommitted changes before enforce init.")
        stash_saved = True

    baseline = runner(config.baseline_command or config.test_command, project_root)
    _save_session(config, baseline, project_root, stash_saved=stash_saved)
    return baseline, sd


def enforce_check(
    hypothesis: str = "",
    *,
    project_root: str | None = None,
    _run_tests_fn: Any = None,
    _audit_fn: Any = None,
    _git_diff_fn: Any = None,
    _git_names_fn: Any = None,
    _git_commit_fn: Any = None,
    _git_revert_fn: Any = None,
    _git_head_fn: Any = None,
    _git_diff_content_fn: Any = None,
) -> ChangeRecord:
    """Check the current state after an agent made a change.

    This is the main iteration step:
    1. Capture what changed (diff)
    2. Run tests (after)
    3. Compare with baseline or previous iteration
    4. Audit for gaming
    5. Check change-size limits
    6. Evaluate prediction accuracy
    7. Commit or revert
    8. Run meta-analysis on committed changes
    """
    from .enforce_audit import audit_change, classify_diff_quality, detect_rule_violations

    config, baseline, session_data = _load_session(project_root)
    iterations = _load_iterations(project_root)
    iteration_num = len(iterations) + 1

    runner = _run_tests_fn or run_tests
    audit_fn = _audit_fn or audit_change
    diff_stat_fn = _git_diff_fn or _git_diff_stat
    diff_names_fn = _git_names_fn or _git_diff_names
    commit_fn = _git_commit_fn or _git_commit
    revert_fn = _git_revert_fn or _git_revert_to
    head_fn = _git_head_fn or _git_head_hash
    diff_content_fn = _git_diff_content_fn or _git_diff_content

    started_at = datetime.now(timezone.utc).isoformat()

    # Capture what changed
    files_modified = diff_names_fn(project_root)
    diff_stat = diff_stat_fn(project_root)
    diff_content = diff_content_fn(project_root)

    # GAP 4: Parse diff hunks
    diff_hunks = _parse_diff_hunks(diff_content)

    # Determine "before" state
    if iterations:
        before = iterations[-1].after or baseline
    else:
        before = baseline

    # Snapshot current HEAD before potential commit
    head_before = head_fn(project_root)

    # Run tests
    after = runner(config.test_command, project_root)

    # Compare
    improved, regressed, unchanged = compare_test_results(before, after)
    net = len(improved) - len(regressed)

    # Also compare with overall pass/fail counts
    count_improvement = (after.passed - before.passed) - (after.failed - before.failed)
    if not improved and not regressed and count_improvement != 0:
        net = count_improvement

    # Audit
    verdict, reasons, gaming_signals = audit_fn(
        diff=diff_content,
        files_modified=files_modified,
        allow_test_modification=config.allow_test_modification,
    )

    # GAP 3: Check project rules
    rule_violations: list[str] = []
    rules = _load_project_rules(config.rules_file)
    if rules:
        rule_signal = detect_rule_violations(diff_content, files_modified, rules)
        if rule_signal:
            rule_violations = [rule_signal.description]
            gaming_signals.append(rule_signal.name)
            reasons.append(f"rule_violation: {rule_signal.description}")
            if rule_signal.confidence >= 0.6:
                verdict = "GAMING"

    # GAP 5: Classify change quality
    test_delta = after.passed - before.passed
    change_quality = classify_diff_quality(diff_content, files_modified, test_delta)

    # GAP 8: Evaluate prediction
    plan = _load_plan(project_root)
    pred_accuracy, predicted_tests, actual_improved, pred_match_pct = _evaluate_prediction(
        plan, improved, regressed,
    )
    _clear_plan(project_root)

    if plan is None and not hypothesis:
        reasons.append("No enforce_plan was registered before this check (prediction tracking disabled)")

    # Decision logic
    decision = "COMMITTED"
    commit_hash = None

    # GAP 7: Change-size enforcement
    diff_line_count = _diff_line_count(diff_content)
    reject_reason = _change_reject_reason(config, files_modified, diff_line_count)
    if reject_reason:
        decision = "REJECTED"
        reasons.append(reject_reason)
        reasons.append("Guidance: Split this change into smaller iterations touching fewer files.")
    elif verdict == "GAMING":
        decision = "REVERTED"
        reasons.append("Gaming detected -- auto-reverting")
    elif regressed and config.require_improvement:
        decision = "REVERTED"
        reasons.append(f"Regressions detected: {', '.join(regressed[:5])}")
    elif net < 0 and config.require_improvement:
        decision = "REVERTED"
        reasons.append("Net negative improvement")
    elif net == 0 and config.require_improvement and not improved:
        decision = "COMMITTED"  # Allow neutral changes unless they regress

    # GAP 12: Regression root cause
    regression_root_cause = ""
    if regressed:
        regression_root_cause = _heuristic_regression_cause(diff_content, regressed, files_modified)

    # Execute decision
    if decision == "COMMITTED" and config.git_auto_commit:
        msg = f"enforce: iteration {iteration_num}"
        if hypothesis:
            msg += f" - {hypothesis[:72]}"
        commit_hash = commit_fn(msg, project_root)
    elif decision == "REVERTED" and config.git_auto_revert:
        revert_fn(head_before, project_root)
    # REJECTED: don't commit or revert -- leave changes for the agent to split

    # GAP 2: Meta-analysis for committed changes
    meta = {}
    if decision == "COMMITTED":
        meta = _meta_analyze(before, after, diff_content, files_modified)

    record = ChangeRecord(
        iteration=iteration_num,
        files_modified=files_modified,
        diff_summary=hypothesis or f"Iteration {iteration_num}",
        diff_stat=diff_stat,
        hypothesis=hypothesis,
        before=before,
        after=after,
        tests_improved=improved,
        tests_regressed=regressed,
        tests_unchanged=unchanged,
        net_improvement=net,
        audit_verdict=verdict,
        audit_reasons=reasons,
        gaming_signals=gaming_signals,
        decision=decision,
        commit_hash=commit_hash,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc).isoformat(),
        meta_analysis=meta,
        diff_hunks=[h.to_dict() for h in diff_hunks],
        change_quality=change_quality,
        prediction_accuracy=pred_accuracy,
        predicted_tests=predicted_tests,
        actual_improved=actual_improved,
        prediction_match_pct=pred_match_pct,
        regression_root_cause=regression_root_cause,
        rule_violations=rule_violations,
    )

    _save_iteration(record, project_root)
    return record


def enforce_plan(
    hypothesis: str,
    expected_tests: list[str] | None = None,
    *,
    project_root: str | None = None,
) -> dict[str, Any]:
    """Register what the agent intends to do before doing it (GAP 8).

    Call this BEFORE making changes. Then call enforce_check after making changes.
    The check will compare actual results to this plan.
    """
    _load_session(project_root)  # Ensure session exists
    expected = expected_tests or []
    _save_plan(hypothesis, expected, project_root)
    return {
        "hypothesis": hypothesis,
        "expected_tests": expected,
        "status": "plan_registered",
    }


def enforce_diff(
    *,
    project_root: str | None = None,
    _git_names_fn: Any = None,
    _git_diff_content_fn: Any = None,
) -> dict[str, Any]:
    """Preview the current diff against enforce change-size rules."""
    config, _, _ = _load_session(project_root)
    names_fn = _git_names_fn or _git_diff_names
    diff_content_fn = _git_diff_content_fn or _git_diff_content

    files = names_fn(project_root)
    diff_content = diff_content_fn(project_root)
    diff_lines = _diff_lines(diff_content)
    diff_line_count = _diff_line_count(diff_content)
    reject_reason = _change_reject_reason(config, files, diff_line_count)

    return {
        "files": files,
        "file_count": len(files),
        "diff_lines": diff_lines,
        "diff_line_count": diff_line_count,
        "would_reject": bool(reject_reason),
        "reject_reason": reject_reason,
    }


def enforce_guard(
    *,
    project_root: str | None = None,
    _git_head_fn: Any = None,
    _git_names_fn: Any = None,
) -> dict[str, Any]:
    """Check if uncommitted changes exist that weren't processed through enforce (GAP 10).

    Returns a dict with status and any warnings.
    """
    head_fn = _git_head_fn or _git_head_hash
    names_fn = _git_names_fn or _git_diff_names

    try:
        config, baseline, session_data = _load_session(project_root)
    except FileNotFoundError:
        return {"status": "no_session", "warnings": ["No enforcement session active."]}

    iterations = _load_iterations(project_root)
    current_head = head_fn(project_root)
    uncommitted = names_fn(project_root)

    warnings_list: list[str] = []
    if uncommitted:
        warnings_list.append(
            f"Found {len(uncommitted)} uncommitted file(s) not processed through enforce_check: "
            f"{', '.join(uncommitted[:5])}"
        )

    last_known_hash = session_data.get("head_hash_at_init", "")
    if iterations:
        for it in reversed(iterations):
            if it.commit_hash:
                last_known_hash = it.commit_hash
                break

    return {
        "status": "warning" if warnings_list else "clean",
        "warnings": warnings_list,
        "current_head": current_head,
        "last_known_hash": last_known_hash,
        "uncommitted_files": uncommitted,
    }


def enforce_status(project_root: str | None = None) -> dict[str, Any]:
    """Return current session status."""
    config, baseline, session_data = _load_session(project_root)
    iterations = _load_iterations(project_root)

    committed = sum(1 for i in iterations if i.decision == "COMMITTED")
    reverted = sum(1 for i in iterations if i.decision == "REVERTED")
    vetoed = sum(1 for i in iterations if i.decision == "VETOED")
    gaming = sum(1 for i in iterations if i.audit_verdict == "GAMING")

    return {
        "session_active": True,
        "config": config.to_dict(),
        "started_at": session_data.get("started_at", ""),
        "baseline": {
            "passed": baseline.passed,
            "failed": baseline.failed,
            "total": baseline.total,
        },
        "iterations": len(iterations),
        "committed": committed,
        "reverted": reverted,
        "vetoed": vetoed,
        "gaming_detected": gaming,
        "max_iterations": config.max_iterations,
        "last_result": iterations[-1].after.to_dict() if iterations and iterations[-1].after else None,
    }


def enforce_challenge(
    project_root: str | None = None,
    *,
    _load_iterations_fn: Any = None,
) -> ChallengeResult:
    """Run adversarial review on recent changes."""
    from .enforce_audit import challenge_iterations

    config, baseline, _ = _load_session(project_root)
    loader = _load_iterations_fn or _load_iterations
    iterations = loader(project_root)
    challenges = _load_challenges(project_root)

    # Determine range to review
    last_reviewed = 0
    if challenges:
        last_reviewed = max(c.iteration_range[1] for c in challenges)

    start = last_reviewed + 1
    end = len(iterations)
    if start > end:
        return ChallengeResult(
            iteration_range=(start, end),
            changes_reviewed=0,
            findings=["No new iterations to review."],
        )

    to_review = [i for i in iterations if start <= i.iteration <= end]
    result = challenge_iterations(to_review)
    result.iteration_range = (start, end)

    _save_challenge(result, len(challenges) + 1, project_root)
    return result


def enforce_reset(project_root: str | None = None) -> bool:
    """Reset/abandon the enforcement session."""
    sd = _session_dir(project_root)
    if sd.exists():
        session_path = sd / "session.json"
        if session_path.exists():
            session_data = json.loads(session_path.read_text(encoding="utf-8"))
            if session_data.get("stash_saved") and not _git_stash_pop(project_root):
                raise RuntimeError("Failed to restore stashed changes during enforce reset.")
        shutil.rmtree(sd)
        return True
    return False


def build_enforce_report(project_root: str | None = None) -> EnforceReport:
    """Build a complete enforcement report from session data."""
    config, baseline, session_data = _load_session(project_root)
    iterations = _load_iterations(project_root)
    challenges = _load_challenges(project_root)

    committed = sum(1 for i in iterations if i.decision == "COMMITTED")
    reverted = sum(1 for i in iterations if i.decision == "REVERTED")
    vetoed = sum(1 for i in iterations if i.decision == "VETOED")
    rejected = sum(1 for i in iterations if i.decision == "REJECTED")
    gaming = sum(1 for i in iterations if i.audit_verdict == "GAMING")

    final = iterations[-1].after if iterations and iterations[-1].after else baseline
    net = final.passed - baseline.passed

    started = session_data.get("started_at", "")
    ended = datetime.now(timezone.utc).isoformat()
    dur = 0.0
    if started:
        try:
            s = datetime.fromisoformat(started)
            e = datetime.fromisoformat(ended)
            dur = (e - s).total_seconds()
        except (ValueError, TypeError):
            pass

    # GAP 13: Cumulative report improvements

    # Prediction accuracy summary
    pred_counts: dict[str, int] = {"accurate": 0, "partial": 0, "wrong": 0, "no_prediction": 0}
    for it in iterations:
        acc = it.prediction_accuracy or "no_prediction"
        if acc in pred_counts:
            pred_counts[acc] += 1
        else:
            pred_counts[acc] = 1
    total_with_prediction = pred_counts["accurate"] + pred_counts["partial"] + pred_counts["wrong"]
    pred_accuracy_pct = (
        (pred_counts["accurate"] / total_with_prediction * 100)
        if total_with_prediction > 0
        else 0.0
    )
    prediction_accuracy_summary = {
        **pred_counts,
        "total_with_prediction": total_with_prediction,
        "accuracy_pct": round(pred_accuracy_pct, 1),
    }

    # Efficiency ratio
    efficiency_ratio = committed / len(iterations) if iterations else 0.0

    # Change map: which files modified most often
    change_map: dict[str, int] = {}
    for it in iterations:
        for f in it.files_modified:
            change_map[f] = change_map.get(f, 0) + 1
    # Sort by frequency
    change_map = dict(sorted(change_map.items(), key=lambda x: -x[1]))

    # Cumulative diff of all committed changes (combine diff_stats)
    committed_diffs: list[str] = []
    for it in iterations:
        if it.decision == "COMMITTED" and it.diff_stat:
            committed_diffs.append(it.diff_stat)
    cumulative_diff = "\n".join(committed_diffs) if committed_diffs else ""

    return EnforceReport(
        config=config,
        changes=iterations,
        challenges=challenges,
        total_iterations=len(iterations),
        committed_count=committed,
        reverted_count=reverted,
        vetoed_count=vetoed,
        rejected_count=rejected,
        gaming_detected_count=gaming,
        baseline_result=baseline,
        final_result=final,
        net_improvement=net,
        duration_seconds=dur,
        prediction_accuracy_summary=prediction_accuracy_summary,
        efficiency_ratio=round(efficiency_ratio, 3),
        change_map=change_map,
        cumulative_diff=cumulative_diff,
    )


# ---------------------------------------------------------------------------
# Autonomous loop (GAP 1)
# ---------------------------------------------------------------------------

def enforce_auto(
    config: EnforceConfig,
    agent_cmd: str,
    *,
    _run_tests_fn: Any = None,
    _run_shell_fn: Any = None,
) -> EnforceReport:
    """Run the full autonomous enforcement loop (GAP 1).

    1. Initialize session (capture baseline)
    2. Loop: run agent_cmd -> enforce_check -> commit/revert -> repeat
    3. Stop when max_iterations reached or all tests pass
    4. Generate and return report
    """
    runner = _run_tests_fn or run_tests
    shell_fn = _run_shell_fn or _run_shell
    project_root = config.project_root

    # Step 1: Initialize
    baseline, sd = enforce_init(config, _run_tests_fn=runner)

    # If all tests pass already, return immediately
    if baseline.failed == 0 and baseline.errors == 0:
        return build_enforce_report(project_root)

    latest_result = baseline
    latest_hypothesis = ""

    for iteration in range(1, config.max_iterations + 1):
        rendered_agent_cmd = _format_agent_command(
            agent_cmd,
            latest_result,
            iteration,
            latest_hypothesis,
        )

        # Step 2: Run agent command
        shell_fn(rendered_agent_cmd, project_root)

        # Step 3: enforce_check
        record = enforce_check(
            hypothesis=f"auto iteration {iteration}",
            project_root=project_root,
            _run_tests_fn=runner,
        )

        if record.after is not None:
            latest_result = record.after
        latest_hypothesis = record.hypothesis

        # Step 4: Run challenge at configured intervals
        iter_count = _iteration_count(project_root)
        if config.challenge_every > 0 and iter_count % config.challenge_every == 0 and iter_count > 0:
            enforce_challenge(project_root)

        # Step 5: Check stopping condition -- all tests pass
        if record.after and record.after.failed == 0 and record.after.errors == 0:
            break

    # Step 6: Generate report
    return build_enforce_report(project_root)


class _SafeTemplateDict(defaultdict[str]):
    """format_map mapping that preserves unknown placeholders."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _escape_shell_value(value: str) -> str:
    """Escape user-controlled text before interpolating into a shell command."""
    return (
        value
        .replace("\\", "\\\\")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
        .replace("`", "\\`")
        .replace('"', '\\"')
        .replace("'", "\\'")
    )


def _format_agent_command(
    agent_cmd: str,
    result: TestResult | None,
    iteration: int,
    hypothesis: str,
) -> str:
    """Substitute auto-mode template variables into the agent command."""
    failed_names = result.test_names_failed if result else []
    failing_tests = ", ".join(failed_names) if failed_names else "unknown"
    fail_count = (result.failed + result.errors) if result else 0
    pass_count = result.passed if result else 0
    total_count = result.total if result else 0
    last_error = result.output[-500:] if result and result.output else ""

    context = _SafeTemplateDict(str)
    context.update(
        {
            "failing_tests": _escape_shell_value(failing_tests),
            "fail_count": str(fail_count),
            "pass_count": str(pass_count),
            "total_count": str(total_count),
            "iteration": str(iteration),
            "last_error": _escape_shell_value(last_error),
            "hypothesis": _escape_shell_value(hypothesis),
        }
    )
    return agent_cmd.format_map(context)
