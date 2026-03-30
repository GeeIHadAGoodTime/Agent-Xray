"""Gaming detection and self-audit heuristics for enforcement mode.

Analyzes diffs to detect test gaming, assertion weakening, exception
swallowing, and other anti-patterns that agents use to fake passing tests.
Also provides positive change classification and project-rule enforcement.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .enforce import ChangeRecord


# ---------------------------------------------------------------------------
# Gaming signal dataclass
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class GamingSignal:
    """A single gaming detection signal."""

    name: str
    confidence: float  # 0.0 - 1.0
    description: str
    line_evidence: str = ""  # The suspicious line(s) from the diff


# ---------------------------------------------------------------------------
# Heuristic detectors
# ---------------------------------------------------------------------------

# Patterns for detecting test directory files
_TEST_PATH_PATTERNS = (
    re.compile(r"(?:^|\/)tests?\/", re.IGNORECASE),
    re.compile(r"(?:^|\/)test_\w+\.py", re.IGNORECASE),
    re.compile(r"(?:^|\/)\w+_test\.py", re.IGNORECASE),
    re.compile(r"(?:^|\/)conftest\.py", re.IGNORECASE),
    re.compile(r"(?:^|\/)spec\/", re.IGNORECASE),
)


def _is_test_file(path: str) -> bool:
    """Check if a file path is a test file."""
    return any(p.search(path) for p in _TEST_PATH_PATTERNS)


def _matches_allowlist(text: str, patterns: list[str] | tuple[str, ...] | None) -> bool:
    """Return True when any case-insensitive allowlist pattern appears in text."""
    if not text or not patterns:
        return False
    lowered = text.casefold()
    return any(pattern.casefold() in lowered for pattern in patterns if pattern)


def detect_test_file_modification(
    files_modified: list[str],
    allow_test_modification: bool = False,
    project_allowlist: list[str] | None = None,
) -> GamingSignal | None:
    """Detect modifications to test files."""
    if allow_test_modification:
        return None
    allowlist = tuple(item.lower() for item in (project_allowlist or []))
    test_files = [
        f for f in files_modified
        if _is_test_file(f)
        and not any(entry in f.lower() for entry in allowlist)
    ]
    if not test_files:
        return None
    return GamingSignal(
        name="test_file_modification",
        confidence=0.3,
        description=f"Test files modified: {', '.join(test_files[:3])}",
        line_evidence=", ".join(test_files),
    )


# Patterns for hardcoded value insertion
_HARDCODED_RETURN_RE = re.compile(
    r"^\+\s*return\s+(?:\d+|['\"].*?['\"]|True|False|None|\[.*?\]|\{.*?\})\s*$",
    re.MULTILINE,
)
_HARDCODED_RETURN_ALLOWLIST = (
    "no visible effect",
    "not found",
    "error",
    "success",
)


def detect_hardcoded_values(
    diff: str,
    allowlist: list[str] | None = None,
) -> GamingSignal | None:
    """Detect insertion of hardcoded return values."""
    matches = _HARDCODED_RETURN_RE.findall(diff)
    if not matches:
        return None
    combined_allowlist = [*_HARDCODED_RETURN_ALLOWLIST, *(allowlist or [])]
    # Only flag if the return is suspiciously simple
    suspicious = [
        m
        for m in matches
        if len(m.strip()) < 40 and not _matches_allowlist(m, combined_allowlist)
    ]
    if not suspicious:
        return None
    return GamingSignal(
        name="hardcoded_value_insertion",
        confidence=0.5,
        description=f"Hardcoded return value(s) inserted ({len(suspicious)} occurrence(s))",
        line_evidence=suspicious[0].strip() if suspicious else "",
    )


# Patterns for special-case test branching
_TEST_MODE_BRANCH_RE = re.compile(
    r"^\+\s*if\s+.*(?:test_mode|TEST_MODE|testing|is_test|mock|MOCK|fixture|FIXTURE)",
    re.MULTILINE | re.IGNORECASE,
)


def detect_special_case_branching(diff: str) -> GamingSignal | None:
    """Detect if-conditions that branch on test-specific flags."""
    matches = _TEST_MODE_BRANCH_RE.findall(diff)
    if not matches:
        return None
    return GamingSignal(
        name="special_case_branching",
        confidence=0.6,
        description=f"Test-specific branch condition(s) added ({len(matches)})",
        line_evidence=matches[0].strip() if matches else "",
    )


# Patterns for mock/stub insertion
_MOCK_INSERTION_RE = re.compile(
    r"^\+\s*(?:mock|Mock|MagicMock|patch|@patch|monkeypatch|stub|Stub)",
    re.MULTILINE,
)


def detect_mock_insertion(diff: str) -> GamingSignal | None:
    """Detect insertion of mocks or stubs that bypass real logic."""
    matches = _MOCK_INSERTION_RE.findall(diff)
    if not matches:
        return None
    return GamingSignal(
        name="mock_stub_insertion",
        confidence=0.2,
        description=f"Mock/stub added ({len(matches)} occurrence(s))",
        line_evidence=matches[0].strip() if matches else "",
    )


# Patterns for assertion weakening
_ASSERTION_WEAKEN_PATTERNS = [
    # assertEqual -> assertIn
    (re.compile(r"^\-\s*.*assertEqual.*$", re.MULTILINE),
     re.compile(r"^\+\s*.*assertIn.*$", re.MULTILINE)),
    # == -> in
    (re.compile(r"^\-\s*.*assert\s+.*==.*$", re.MULTILINE),
     re.compile(r"^\+\s*.*assert\s+.*\bin\b.*$", re.MULTILINE)),
    # Removing assert entirely
    (re.compile(r"^\-\s*assert\s+", re.MULTILINE),
     re.compile(r"^\+\s*(?:pass|#).*$", re.MULTILINE)),
]


def detect_assertion_weakening(diff: str) -> GamingSignal | None:
    """Detect weakening of test assertions."""
    lines = diff.splitlines()
    removals = [
        line for line in lines if line.startswith("-") and "assert" in line.lower()
    ]

    if not removals:
        return None

    # Check for specific weakening patterns
    for rm_pat, add_pat in _ASSERTION_WEAKEN_PATTERNS:
        rm_matches = rm_pat.findall(diff)
        add_matches = add_pat.findall(diff)
        if rm_matches and add_matches:
            return GamingSignal(
                name="assertion_weakening",
                confidence=0.7,
                description="Test assertion weakened (strict -> lenient)",
                line_evidence=f"Removed: {rm_matches[0].strip()[:60]}",
            )

    # Check for assert removal without replacement
    assert_removed = [
        line for line in lines
        if line.startswith("-") and re.search(r"\bassert\b", line)
    ]
    assert_added = [
        line for line in lines
        if line.startswith("+") and re.search(r"\bassert\b", line)
    ]
    if len(assert_removed) > len(assert_added):
        return GamingSignal(
            name="assertion_weakening",
            confidence=0.6,
            description=f"Net {len(assert_removed) - len(assert_added)} assertion(s) removed",
            line_evidence=assert_removed[0].strip()[:60] if assert_removed else "",
        )

    return None


# Pattern for exception swallowing
_EXCEPTION_SWALLOW_RE = re.compile(
    r"^\+\s*except(?:\s+\w+)?(?:\s+as\s+\w+)?:\s*(?:\n\+\s*pass|\s+pass)",
    re.MULTILINE,
)
_BARE_EXCEPT_PASS_RE = re.compile(
    r"^\+\s*except.*:.*\n\+\s*pass",
    re.MULTILINE,
)


def detect_exception_swallowing(diff: str) -> GamingSignal | None:
    """Detect bare except: pass patterns."""
    matches = list(_BARE_EXCEPT_PASS_RE.finditer(diff))
    if not matches:
        # Also check single-line pattern
        single = re.findall(r"^\+\s*except.*:\s*pass", diff, re.MULTILINE)
        if not single:
            return None
        return GamingSignal(
            name="exception_swallowing",
            confidence=0.7,
            description=f"Exception swallowing pattern(s) added ({len(single)})",
            line_evidence=single[0].strip() if single else "",
        )
    return GamingSignal(
        name="exception_swallowing",
        confidence=0.7,
        description=f"Exception swallowing (except: pass) added ({len(matches)})",
        line_evidence=matches[0].group().strip()[:60] if matches else "",
    )


# Pattern for early return
_EARLY_RETURN_RE = re.compile(
    r"^\+\s*return\s*(?:None|True|False|0|\[\]|\{\}|\"\")?(?:\s*#.*)?$",
    re.MULTILINE,
)


def detect_early_return(diff: str) -> GamingSignal | None:
    """Detect suspicious early returns that bypass logic."""
    matches = _EARLY_RETURN_RE.findall(diff)
    if not matches:
        return None
    # Only flag if the return is at a suspicious location (added, not replacing)
    added_returns = [m for m in matches if m.strip()]
    if not added_returns:
        return None
    return GamingSignal(
        name="early_return",
        confidence=0.4,
        description=f"Early return statement(s) added ({len(added_returns)})",
        line_evidence=added_returns[0].strip() if added_returns else "",
    )


# Pattern for import removal
_IMPORT_REMOVAL_RE = re.compile(
    r"^\-\s*(?:import|from)\s+",
    re.MULTILINE,
)


def detect_import_removal(diff: str) -> GamingSignal | None:
    """Detect removal of imports."""
    removed = _IMPORT_REMOVAL_RE.findall(diff)
    added_imports = re.findall(r"^\+\s*(?:import|from)\s+", diff, re.MULTILINE)
    net_removed = len(removed) - len(added_imports)
    if net_removed <= 0:
        return None
    return GamingSignal(
        name="import_removal",
        confidence=0.3,
        description=f"Net {net_removed} import(s) removed",
        line_evidence=removed[0].strip() if removed else "",
    )


# ---------------------------------------------------------------------------
# Combined audit function
# ---------------------------------------------------------------------------

ALL_DETECTORS = [
    detect_hardcoded_values,
    detect_special_case_branching,
    detect_mock_insertion,
    detect_assertion_weakening,
    detect_exception_swallowing,
    detect_early_return,
    detect_import_removal,
]


def audit_change(
    diff: str,
    files_modified: list[str] | None = None,
    allow_test_modification: bool = False,
    project_allowlist: list[str] | None = None,
) -> tuple[str, list[str], list[str]]:
    """Audit a change for gaming signals.

    Returns:
        (verdict, reasons, gaming_signal_names)
        verdict is one of: "VALID", "SUSPICIOUS", "GAMING"
    """
    signals: list[GamingSignal] = []
    allowlist = project_allowlist or []

    # Test file modification check
    if files_modified:
        sig = detect_test_file_modification(
            files_modified,
            allow_test_modification,
            project_allowlist=project_allowlist,
        )
        if sig:
            signals.append(sig)

    # Run all diff-based detectors
    for detector in ALL_DETECTORS:
        if detector is detect_hardcoded_values:
            sig = detector(diff, allowlist)
        else:
            sig = detector(diff)
        if sig and not _matches_allowlist(
            f"{sig.line_evidence}\n{sig.description}",
            allowlist,
        ):
            signals.append(sig)

    if not signals:
        return "VALID", ["No gaming signals detected"], []

    # Compute combined score
    # Use max confidence, not sum (avoid double-counting)
    max_confidence = max(s.confidence for s in signals)
    avg_confidence = sum(s.confidence for s in signals) / len(signals)
    # Weight toward max but consider breadth
    combined = max_confidence * 0.7 + avg_confidence * 0.3

    reasons = [f"{s.name}: {s.description} (confidence: {s.confidence:.1f})" for s in signals]
    signal_names = [s.name for s in signals]

    if combined > 0.6:
        return "GAMING", reasons, signal_names
    if combined > 0.3:
        return "SUSPICIOUS", reasons, signal_names
    return "VALID", reasons, signal_names


# ---------------------------------------------------------------------------
# Adversarial challenge
# ---------------------------------------------------------------------------

def challenge_iterations(
    iterations: list[ChangeRecord],
) -> Any:
    """Review a set of iterations for cumulative gaming patterns.

    Looks for:
    - Multiple suspicious iterations in sequence
    - Gradual assertion weakening across iterations
    - Accumulating exception swallowing
    """
    from .enforce import ChallengeResult

    if not iterations:
        return ChallengeResult(
            changes_reviewed=0,
            findings=["No iterations to review."],
        )

    findings: list[str] = []
    vetoed: list[int] = []

    # Check for consecutive suspicious/gaming verdicts
    gaming_streak = 0
    for it in iterations:
        if it.audit_verdict in ("GAMING", "SUSPICIOUS"):
            gaming_streak += 1
        else:
            gaming_streak = 0
        if gaming_streak >= 2:
            findings.append(
                f"Consecutive suspicious changes at iterations "
                f"{it.iteration - 1}-{it.iteration}"
            )

    # Check for net regression across the reviewed range
    if iterations:
        first = iterations[0]
        last = iterations[-1]
        if first.before and last.after:
            if last.after.failed > first.before.failed:
                findings.append(
                    f"Net regression: failures increased from "
                    f"{first.before.failed} to {last.after.failed}"
                )

    # Check for repeated modifications to the same files
    file_counts: dict[str, int] = {}
    for it in iterations:
        for f in it.files_modified:
            file_counts[f] = file_counts.get(f, 0) + 1
    hot_files = {f: c for f, c in file_counts.items() if c >= 3}
    if hot_files:
        files_str = ", ".join(f"{f} ({c}x)" for f, c in sorted(hot_files.items(), key=lambda x: -x[1])[:3])
        findings.append(f"Hot files modified repeatedly: {files_str}")

    # Veto iterations with GAMING verdict
    for it in iterations:
        if it.audit_verdict == "GAMING" and it.decision not in ("REVERTED", "RECOMMEND_REVERT"):
            vetoed.append(it.iteration)
            findings.append(f"Iteration {it.iteration} vetoed: gaming detected but not reverted")

    # GAP 6: Cross-iteration analysis
    # Check for test flip-flop: test_A fixed in iteration N, broken again in iteration M
    test_history: dict[str, list[tuple[int, str]]] = {}  # test_name -> [(iter, "improved"/"regressed")]
    for it in iterations:
        for t in it.tests_improved:
            test_history.setdefault(t, []).append((it.iteration, "improved"))
        for t in it.tests_regressed:
            test_history.setdefault(t, []).append((it.iteration, "regressed"))
    for test_name, history in test_history.items():
        states = [s for _, s in history]
        if "improved" in states and "regressed" in states:
            iters = [str(i) for i, _ in history]
            findings.append(
                f"Test flip-flop: {test_name} changed state across iterations {', '.join(iters)}"
            )

    # Dependency detection: if two committed changes modify the same function
    committed_files: dict[str, list[int]] = {}
    for it in iterations:
        if it.decision in ("COMMITTED", "RECOMMEND_COMMIT"):
            for f in it.files_modified:
                committed_files.setdefault(f, []).append(it.iteration)
    for f, iters_list in committed_files.items():
        if len(iters_list) >= 2:
            findings.append(
                f"Dependency risk: {f} modified in committed iterations "
                f"{', '.join(str(i) for i in iters_list)} -- check for interaction bugs"
            )

    # Coverage gap: if all changes target one subsystem but failures in another
    all_modified_dirs: set[str] = set()
    all_failed_dirs: set[str] = set()
    for it in iterations:
        for f in it.files_modified:
            parts = f.split("/")
            if len(parts) > 1:
                all_modified_dirs.add(parts[0])
        for t in it.tests_regressed:
            parts = t.split("/")
            if len(parts) > 1:
                all_failed_dirs.add(parts[0])
        if it.after:
            for t in it.after.test_names_failed:
                parts = t.split("/")
                if len(parts) > 1:
                    all_failed_dirs.add(parts[0])
    untouched_failure_dirs = all_failed_dirs - all_modified_dirs
    if untouched_failure_dirs and all_modified_dirs:
        findings.append(
            f"Coverage gap: failing tests in {', '.join(sorted(untouched_failure_dirs))} "
            f"but changes only in {', '.join(sorted(all_modified_dirs))}"
        )

    # Cumulative gaming check: run gaming detection on aggregated diff
    combined_diff_parts: list[str] = []
    combined_files: set[str] = set()
    for it in iterations:
        if it.decision in ("COMMITTED", "RECOMMEND_COMMIT") and it.diff_stat:
            combined_diff_parts.append(it.diff_stat)
            combined_files.update(it.files_modified)
    if combined_diff_parts:
        combined_diff_text = "\n".join(combined_diff_parts)
        cumulative_signals: list[GamingSignal] = []
        for detector in ALL_DETECTORS:
            sig = detector(combined_diff_text)
            if sig:
                cumulative_signals.append(sig)
        if cumulative_signals:
            sig_names = [s.name for s in cumulative_signals]
            findings.append(
                f"Cumulative gaming signals in aggregate diff: {', '.join(sig_names)}"
            )

    # ------------------------------------------------------------------
    # Check 5 (NEW): Cumulative gaming signal count
    # ------------------------------------------------------------------
    total_gaming_signals = sum(len(it.gaming_signals) for it in iterations)
    if total_gaming_signals > 5:
        findings.append(
            f"High cumulative gaming signal density: "
            f"{total_gaming_signals} signals across "
            f"{len(iterations)} iterations"
        )

    # ------------------------------------------------------------------
    # Check 6 (NEW): Assertion erosion
    # ------------------------------------------------------------------
    assertion_removals = 0
    for it in iterations:
        for reason in it.audit_reasons:
            lower = reason.lower()
            if "assertion" in lower and (
                "removed" in lower or "weakened" in lower or "deleted" in lower
            ):
                assertion_removals += 1
    if assertion_removals > 2:
        findings.append(
            f"Cumulative assertion erosion: {assertion_removals} "
            f"assertions weakened/removed across iterations"
        )

    # ------------------------------------------------------------------
    # Check 7 (NEW): Diminishing returns
    # ------------------------------------------------------------------
    trailing_zeros = 0
    for it in reversed(iterations):
        if it.net_improvement == 0:
            trailing_zeros += 1
        else:
            break
    if trailing_zeros >= 3:
        findings.append(
            f"Diminishing returns: last {trailing_zeros} iterations "
            f"produced no improvement"
        )

    # ------------------------------------------------------------------
    # Check 8 (NEW): Coverage gap -- persistent failures never addressed
    # ------------------------------------------------------------------
    regressed_counter: Counter[str] = Counter()
    improved_set: set[str] = set()
    for it in iterations:
        for t in it.tests_regressed:
            regressed_counter[t] += 1
        for t in it.tests_improved:
            improved_set.add(t)
    persistent = {
        t for t, n in regressed_counter.items()
        if n >= 2 and t not in improved_set
    }
    if persistent:
        names_str = ", ".join(sorted(persistent))
        findings.append(
            f"Persistent failures never addressed: {names_str}"
        )

    # ------------------------------------------------------------------
    # Check 9 (NEW): Scope creep
    # ------------------------------------------------------------------
    if len(iterations) >= 4:
        mid = len(iterations) // 2
        first_half = iterations[:mid]
        second_half = iterations[mid:]

        avg_first = (
            sum(len(it.files_modified) for it in first_half) / len(first_half)
            if first_half
            else 0.0
        )
        avg_second = (
            sum(len(it.files_modified) for it in second_half) / len(second_half)
            if second_half
            else 0.0
        )

        if avg_second > avg_first and avg_first > 0:
            findings.append(
                f"Scope creep detected: avg {avg_first:.1f} "
                f"files/change -> {avg_second:.1f} files/change"
            )

    if not findings:
        findings.append("No adversarial findings -- changes look clean.")

    return ChallengeResult(
        iteration_range=(
            iterations[0].iteration if iterations else 0,
            iterations[-1].iteration if iterations else 0,
        ),
        changes_reviewed=len(iterations),
        vetoed=vetoed,
        findings=findings,
    )


# ---------------------------------------------------------------------------
# Change quality classification (GAP 5)
# ---------------------------------------------------------------------------

def classify_diff_quality(diff: str, files: list[str], test_delta: int) -> str:
    """Classify a change positively based on its diff nature.

    Returns one of:
    - "behavioral_improvement": threshold/config/prompt changes
    - "bug_fix": error path, null check, off-by-one fixes
    - "refactor": restructure without behavior change
    - "test_improvement": test coverage improvement
    - "neutral": no clear classification
    """
    if not diff and not files:
        return "neutral"

    lines = diff.splitlines()
    added = [
        line for line in lines if line.startswith("+") and not line.startswith("+++")
    ]
    removed = [
        line for line in lines if line.startswith("-") and not line.startswith("---")
    ]

    # Check for test files
    test_files = [f for f in files if any(p in f.lower() for p in ("test_", "_test.", "tests/", "spec/", "conftest"))]
    source_files = [f for f in files if f not in test_files]

    if test_files and not source_files and test_delta > 0:
        return "test_improvement"

    # Check for config/threshold changes
    config_patterns = re.compile(
        r"(?:threshold|timeout|limit|max_|min_|rate|delay|interval|config|setting)",
        re.IGNORECASE,
    )
    config_changes = sum(
        1 for line in added + removed if config_patterns.search(line)
    )
    if config_changes > 0 and len(added) + len(removed) <= config_changes * 3:
        return "behavioral_improvement"

    # Check for bug fix patterns
    bugfix_patterns = [
        re.compile(r"(?:if\s+\w+\s+is\s+None|is\s+not\s+None)", re.IGNORECASE),
        re.compile(r"(?:try:|except\s+\w+)", re.IGNORECASE),
        re.compile(r"(?:>=?\s*0|<=?\s*len|boundary|bounds|off.by.one)", re.IGNORECASE),
        re.compile(r"(?:\.get\(|getattr\(|hasattr\()", re.IGNORECASE),
    ]
    bugfix_evidence = sum(1 for line in added for pat in bugfix_patterns if pat.search(line))
    if bugfix_evidence >= 1 and test_delta >= 0:
        return "bug_fix"

    # Check for refactoring (similar line count, different structure)
    if abs(len(added) - len(removed)) <= 3 and len(added) > 5 and test_delta == 0:
        return "refactor"

    # Behavioral improvement if prompt/text changes
    prompt_patterns = re.compile(r"(?:prompt|instruction|message|template|text)", re.IGNORECASE)
    if any(prompt_patterns.search(line) for line in added):
        return "behavioral_improvement"

    return "neutral"


# ---------------------------------------------------------------------------
# Project rule violations (GAP 3)
# ---------------------------------------------------------------------------

def detect_rule_violations(
    diff: str,
    files: list[str],
    rules: dict[str, Any],
) -> GamingSignal | None:
    """Check a diff against project-defined rules.

    Rules format:
    {
        "forbidden_patterns": ["os\\.getenv", "print\\("],
        "required_in_new_files": ["from __future__ import annotations"],
        "max_files_per_change": 3,
        "forbidden_file_patterns": ["\\.env", "credentials"]
    }
    """
    violations: list[str] = []

    # Check forbidden patterns in added lines
    forbidden = rules.get("forbidden_patterns", [])
    for pattern_str in forbidden:
        try:
            pat = re.compile(pattern_str)
        except re.error:
            continue
        for line in diff.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                if pat.search(line):
                    violations.append(f"Forbidden pattern '{pattern_str}' found: {line.strip()[:60]}")
                    break

    # Check forbidden file patterns
    forbidden_files = rules.get("forbidden_file_patterns", [])
    for pattern_str in forbidden_files:
        try:
            pat = re.compile(pattern_str)
        except re.error:
            continue
        for f in files:
            if pat.search(f):
                violations.append(f"Forbidden file pattern '{pattern_str}' matched: {f}")
                break

    # Check max files per change
    max_files = rules.get("max_files_per_change")
    if max_files is not None and len(files) > int(max_files):
        violations.append(
            f"Too many files modified: {len(files)} > {max_files}"
        )

    # Check required in new files (heuristic: files that only have additions)
    required = rules.get("required_in_new_files", [])
    if required:
        # Find files that appear to be newly added (only + lines, no - lines)
        added_lines_by_file = _group_diff_by_file(diff)
        for fname, (adds, removes) in added_lines_by_file.items():
            if adds > 0 and removes == 0:
                for req_pattern in required:
                    if req_pattern not in diff:
                        violations.append(
                            f"New file {fname} missing required pattern: {req_pattern}"
                        )
                        break

    if not violations:
        return None

    return GamingSignal(
        name="rule_violation",
        confidence=0.6,
        description="; ".join(violations[:3]),
        line_evidence=violations[0] if violations else "",
    )


# ---------------------------------------------------------------------------
# Gap 2: Meta-analysis of successful changes
# ---------------------------------------------------------------------------

def _file_extension(path: str) -> str:
    """Extract file extension from a path."""
    _, ext = os.path.splitext(path)
    return ext if ext else "(no ext)"


def _file_directory(path: str) -> str:
    """Extract the first directory component from a path."""
    parts = path.replace("\\", "/").split("/")
    return parts[0] if len(parts) > 1 else "(root)"


def analyze_successful_changes(
    iterations: list[ChangeRecord],
) -> dict[str, Any]:
    """Analyse patterns across COMMITTED iterations.

    Returns a dictionary with:
    - avg_files_per_change: float
    - avg_net_improvement: float
    - common_file_patterns: dict mapping extension/directory to count
    - single_file_changes: int
    - multi_file_changes: int
    - success_rate: float  (committed / total)
    - patterns: list[str]  human-readable observations
    """
    committed = [r for r in iterations if r.decision in ("COMMITTED", "RECOMMEND_COMMIT")]
    total = len(iterations)

    if not committed:
        return {
            "avg_files_per_change": 0.0,
            "avg_net_improvement": 0.0,
            "common_file_patterns": {},
            "single_file_changes": 0,
            "multi_file_changes": 0,
            "success_rate": 0.0,
            "patterns": [],
        }

    # File counts
    file_counts = [len(r.files_modified) for r in committed]
    avg_files = sum(file_counts) / len(committed)
    avg_improvement = sum(r.net_improvement for r in committed) / len(committed)

    single = sum(1 for c in file_counts if c == 1)
    multi = sum(1 for c in file_counts if c >= 2)

    # Extension & directory frequency
    ext_counter: Counter[str] = Counter()
    dir_counter: Counter[str] = Counter()
    for r in committed:
        for f in r.files_modified:
            ext_counter[_file_extension(f)] += 1
            dir_counter[_file_directory(f)] += 1

    # Merge into one "common_file_patterns" dict
    common: dict[str, int] = {}
    for ext, cnt in ext_counter.most_common(5):
        common[ext] = cnt
    for d, cnt in dir_counter.most_common(3):
        common[d + "/"] = cnt

    success_rate = len(committed) / total if total else 0.0

    # Build human-readable patterns
    patterns: list[str] = []

    if committed:
        single_pct = round(single / len(committed) * 100)
        if single_pct > 0:
            patterns.append(
                f"{single_pct}% of successful changes touched a single file"
            )

    if ext_counter:
        top_ext, _top_cnt = ext_counter.most_common(1)[0]
        if dir_counter:
            top_dir, _ = dir_counter.most_common(1)[0]
            patterns.append(
                f"Most improvements came from {top_ext} files in {top_dir}/"
            )

    return {
        "avg_files_per_change": round(avg_files, 2),
        "avg_net_improvement": round(avg_improvement, 2),
        "common_file_patterns": common,
        "single_file_changes": single,
        "multi_file_changes": multi,
        "success_rate": round(success_rate, 4),
        "patterns": patterns,
    }


# ---------------------------------------------------------------------------
# Gap 5: "Good Move" quality assessment (ChangeRecord-level)
# ---------------------------------------------------------------------------

def classify_change_quality(record: ChangeRecord) -> str:
    """Classify a single ChangeRecord into a quality bucket.

    Returns one of:
        EXCELLENT, GOOD, NEUTRAL, POOR, HARMFUL
    """
    has_gaming = bool(record.gaming_signals)
    has_reg = bool(record.tests_regressed)

    _commit_decisions = ("COMMITTED", "RECOMMEND_COMMIT")
    _revert_decisions = ("REVERTED", "RECOMMEND_REVERT")

    # HARMFUL: reverted AND (gaming OR large net loss)
    if record.decision in _revert_decisions and (has_gaming or record.net_improvement < -2):
        return "HARMFUL"

    # POOR: reverted without gaming, OR zero improvement with regressions
    if record.decision in _revert_decisions and not has_gaming:
        return "POOR"
    if (
        record.net_improvement == 0
        and has_reg
        and record.decision in _commit_decisions
    ):
        return "POOR"

    # EXCELLENT: big improvement, clean, committed
    if (
        record.net_improvement >= 3
        and not has_gaming
        and not has_reg
        and record.decision in _commit_decisions
    ):
        return "EXCELLENT"

    # GOOD: moderate improvement, clean, committed
    if (
        record.net_improvement >= 1
        and not has_gaming
        and record.decision in _commit_decisions
    ):
        return "GOOD"

    # NEUTRAL: zero improvement, no regressions, committed
    if (
        record.net_improvement == 0
        and not has_reg
        and record.decision in _commit_decisions
    ):
        return "NEUTRAL"

    # Fall-through: anything else is NEUTRAL (e.g. PENDING)
    return "NEUTRAL"


def quality_distribution(
    iterations: list[ChangeRecord],
) -> dict[str, int]:
    """Count each quality level across all iterations.

    Returns a dict like ``{"EXCELLENT": 2, "GOOD": 3, ...}``.
    """
    dist: dict[str, int] = {
        "EXCELLENT": 0,
        "GOOD": 0,
        "NEUTRAL": 0,
        "POOR": 0,
        "HARMFUL": 0,
    }
    for r in iterations:
        quality = classify_change_quality(r)
        dist[quality] += 1
    return dist


# ---------------------------------------------------------------------------
# Helpers for _group_diff_by_file
# ---------------------------------------------------------------------------

def _group_diff_by_file(diff: str) -> dict[str, tuple[int, int]]:
    """Group diff lines by file, counting adds and removes per file."""
    result: dict[str, tuple[int, int]] = {}
    current_file = ""
    for line in diff.splitlines():
        m = re.match(r"^diff --git a/(.+?) b/(.+)$", line)
        if m:
            current_file = m.group(2)
            result[current_file] = (0, 0)
            continue
        if not current_file:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            adds, removes = result.get(current_file, (0, 0))
            result[current_file] = (adds + 1, removes)
        elif line.startswith("-") and not line.startswith("---"):
            adds, removes = result.get(current_file, (0, 0))
            result[current_file] = (adds, removes + 1)
    return result
