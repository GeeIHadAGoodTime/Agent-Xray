"""Report generation for enforcement sessions.

Produces text, JSON, and Markdown reports from enforcement session data.
Includes grading, project-rule checking, and cumulative metrics.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .enforce import (
    ChangeRecord,
    ChallengeResult,
    DiffHunk,
    EnforceReport,
    TestResult,
    build_enforce_report,
)


# ---------------------------------------------------------------------------
# ANSI color helpers (terminal only)
# ---------------------------------------------------------------------------

ANSI_RESET = "\033[0m"
ANSI_COLORS = {
    "header": "\033[1;36m",
    "good": "\033[32m",
    "bad": "\033[31m",
    "warn": "\033[33m",
    "dim": "\033[90m",
    "bold": "\033[1m",
}


def _c(text: str, key: str, *, color: bool = True) -> str:
    if not color:
        return text
    code = ANSI_COLORS.get(key, "")
    if not code:
        return text
    return f"{code}{text}{ANSI_RESET}"


# ---------------------------------------------------------------------------
# GAP 3: Rules File Awareness
# ---------------------------------------------------------------------------

def load_project_rules(rules_path: str) -> dict[str, Any]:
    """Load project-specific rules from a JSON file.

    Expected format::

        {
          "forbidden_patterns": ["except: pass", "noqa", "type: ignore"],
          "required_patterns": ["logger."],
          "max_complexity": 10,
          "banned_imports": ["os.system"],
          "custom_rules": [
            {"name": "no-print", "pattern": "^\\\\+.*\\\\bprint\\\\(",
             "description": "No print statements", "confidence": 0.5}
          ]
        }

    Returns the parsed dict, or an empty dict if the file is missing or
    malformed.  Never raises.
    """
    try:
        with open(rules_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return {}
        return data
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def check_against_rules(diff: str, rules: dict[str, Any]) -> list[str]:
    """Check a diff string against project rules.

    Inspects only *added* lines (starting with ``+`` but not ``+++``).

    Returns a list of human-readable violation strings.
    """
    violations: list[str] = []

    # Extract added lines (skip diff header lines like +++ b/file)
    added_lines: list[str] = []
    for raw_line in diff.splitlines():
        if raw_line.startswith("+") and not raw_line.startswith("+++"):
            added_lines.append(raw_line)

    # --- forbidden_patterns ---
    for pattern in rules.get("forbidden_patterns", []):
        for line in added_lines:
            content = line[1:]  # strip leading '+'
            if pattern in content:
                violations.append(
                    f"Forbidden pattern '{pattern}' found: {line.strip()}"
                )

    # --- banned_imports ---
    for banned in rules.get("banned_imports", []):
        for line in added_lines:
            content = line[1:]
            stripped = content.strip()
            if stripped.startswith(("import ", "from ")):
                if banned in stripped:
                    violations.append(
                        f"Banned import '{banned}' found: {stripped}"
                    )

    # --- custom_rules (regex-based) ---
    for custom in rules.get("custom_rules", []):
        name = custom.get("name", "unnamed")
        pat = custom.get("pattern", "")
        desc = custom.get("description", "")
        if not pat:
            continue
        try:
            regex = re.compile(pat)
        except re.error:
            continue
        for line in added_lines:
            if regex.search(line):
                violations.append(f"[{name}] {desc}: {line.strip()}")

    # --- required_patterns (only when change is >10 added lines) ---
    if len(added_lines) > 10:
        for required in rules.get("required_patterns", []):
            found = any(required in line for line in added_lines)
            if not found:
                violations.append(
                    f"Required pattern '{required}' not found in "
                    f"{len(added_lines)} added lines"
                )

    return violations


def format_rules_violations(violations: list[str], *, color: bool = True) -> str:
    """Format a list of rule violations into a readable string."""
    if not violations:
        return _c("No rule violations found.", "good", color=color)

    header = _c(f"Rule Violations ({len(violations)}):", "bad", color=color)
    lines = [header]
    for v in violations:
        bullet = _c("  - ", "warn", color=color)
        lines.append(f"{bullet}{v}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# GAP 11: Grading Integration
# ---------------------------------------------------------------------------

def grade_enforce_session(report: EnforceReport) -> dict[str, Any]:
    """Grade the overall enforcement session.

    Returns a dict with ``grade`` (A-F), ``score`` (0-100), and ``factors``.

    Grading formula:
    - Start at 100
    - Subtract 5 per REVERTED iteration
    - Subtract 15 per GAMING detection
    - Subtract 3 per iteration with net_improvement == 0 (waste)
    - Add 2 per COMMITTED iteration with net_improvement > 0
    - Subtract 10 if final has more failures than baseline (net regression)
    - Add 10 if final has 0 failures (all tests pass)
    - Clamp to 0-100
    """
    score = 100
    factors: dict[str, int] = {}

    # -5 per REVERTED iteration
    reverted = report.reverted_count
    if reverted:
        penalty = reverted * 5
        factors["reverted_penalty"] = -penalty
        score -= penalty

    # -15 per GAMING detection
    gaming = report.gaming_detected_count
    if gaming:
        penalty = gaming * 15
        factors["gaming_penalty"] = -penalty
        score -= penalty

    # -3 per waste iteration (net_improvement == 0)
    waste = 0
    for change in report.changes:
        if change.net_improvement == 0:
            waste += 1
    if waste:
        penalty = waste * 3
        factors["waste_penalty"] = -penalty
        score -= penalty

    # +2 per COMMITTED iteration with net_improvement > 0
    good_commits = 0
    for change in report.changes:
        if change.decision == "COMMITTED" and change.net_improvement > 0:
            good_commits += 1
    if good_commits:
        bonus = good_commits * 2
        factors["good_commit_bonus"] = bonus
        score += bonus

    # -10 if final has more failures than baseline (net regression)
    baseline_failed = report.baseline_result.failed if report.baseline_result else 0
    final_failed = report.final_result.failed if report.final_result else 0
    if final_failed > baseline_failed:
        factors["net_regression_penalty"] = -10
        score -= 10

    # +10 if all tests pass at end
    if report.final_result and report.final_result.failed == 0:
        factors["all_pass_bonus"] = 10
        score += 10

    # Clamp 0-100
    score = max(0, min(100, score))

    # Map to letter grade
    if score >= 90:
        grade = "A"
    elif score >= 75:
        grade = "B"
    elif score >= 60:
        grade = "C"
    elif score >= 45:
        grade = "D"
    else:
        grade = "F"

    return {"grade": grade, "score": score, "factors": factors}


# ---------------------------------------------------------------------------
# GAP 13: Cumulative metrics helpers
# ---------------------------------------------------------------------------

def _improvements_per_iteration(report: EnforceReport) -> float:
    """net_improvement / max(total_iterations, 1)."""
    return report.net_improvement / max(report.total_iterations, 1)


def _build_detailed_change_map(
    report: EnforceReport,
) -> list[dict[str, Any]]:
    """Per-file change summary with committed/reverted breakdown.

    Returns list sorted by total modifications descending::

        [{"file": "src/foo.py", "total": 5, "committed": 3, "reverted": 2}, ...]
    """
    from collections import Counter

    totals: Counter[str] = Counter()
    committed: Counter[str] = Counter()
    reverted: Counter[str] = Counter()

    for change in report.changes:
        for f in change.files_modified:
            totals[f] += 1
            if change.decision == "COMMITTED":
                committed[f] += 1
            elif change.decision == "REVERTED":
                reverted[f] += 1

    result = []
    for f in sorted(totals, key=lambda x: -totals[x]):
        result.append({
            "file": f,
            "total": totals[f],
            "committed": committed.get(f, 0),
            "reverted": reverted.get(f, 0),
        })
    return result


def _build_timeline(report: EnforceReport) -> list[dict[str, Any]]:
    """Build a timeline of iterations.

    Returns::

        [{"number": 1, "timestamp": "10:30:15", "status": "COMMITTED",
          "delta": "+2 tests", "commit_hash": "abc1234", "gaming": False}, ...]
    """
    timeline = []
    for change in report.changes:
        delta = change.net_improvement
        if delta > 0:
            delta_str = f"+{delta} tests"
        elif delta < 0:
            delta_str = f"{delta} tests"
        else:
            delta_str = "0 tests"

        # Use started_at for timestamp, extract time portion if ISO format
        ts = change.started_at or ""
        if "T" in ts:
            ts = ts.split("T", 1)[1][:8]  # HH:MM:SS

        timeline.append({
            "number": change.iteration,
            "timestamp": ts,
            "status": change.decision,
            "delta": delta_str,
            "commit_hash": change.commit_hash or "",
            "gaming": bool(change.gaming_signals),
        })
    return timeline


# ---------------------------------------------------------------------------
# Diff hunk formatting helpers (GAP 4)
# ---------------------------------------------------------------------------

def _format_diff_hunks_text(hunks: list[dict[str, Any]], *, color: bool = True) -> list[str]:
    """Format diff hunks for text output."""
    lines: list[str] = []
    for hunk in hunks[:10]:  # Limit to 10 hunks per iteration
        file_name = hunk.get("file", "")
        lines.append(f"    File: {file_name}")
        for rl in hunk.get("removed_lines", [])[:3]:
            lines.append(f"      {_c(rl, 'bad', color=color)}")
        for al in hunk.get("added_lines", [])[:3]:
            lines.append(f"      {_c(al, 'good', color=color)}")
    return lines


def _format_diff_hunks_markdown(hunks: list[dict[str, Any]]) -> list[str]:
    """Format diff hunks for markdown output."""
    lines: list[str] = []
    if not hunks:
        return lines
    lines.append("")
    lines.append("**Diff detail:**")
    lines.append("```diff")
    for hunk in hunks[:10]:
        file_name = hunk.get("file", "")
        lines.append(f"# {file_name}")
        for rl in hunk.get("removed_lines", [])[:3]:
            lines.append(rl)
        for al in hunk.get("added_lines", [])[:3]:
            lines.append(al)
    lines.append("```")
    return lines


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def _format_test_result_line(label: str, tr: TestResult, *, color: bool = True) -> str:
    status = f"{tr.passed} passed, {tr.failed} failed"
    if tr.errors:
        status += f", {tr.errors} errors"
    if tr.skipped:
        status += f", {tr.skipped} skipped"
    return f"  {label}: {status} (exit {tr.exit_code})"


def _format_change_text(change: ChangeRecord, *, color: bool = True) -> str:
    lines: list[str] = []
    verdict_color = "good" if change.audit_verdict == "VALID" else (
        "bad" if change.audit_verdict == "GAMING" else "warn"
    )
    decision_color = "good" if change.decision == "COMMITTED" else (
        "warn" if change.decision == "REJECTED" else "bad"
    )

    header = f"Iteration {change.iteration}"
    if change.commit_hash:
        header += f" [{change.commit_hash}]"
    lines.append(_c(header, "bold", color=color))

    if change.hypothesis:
        lines.append(f"  Hypothesis: {change.hypothesis}")
    if change.files_modified:
        files_str = ", ".join(change.files_modified[:5])
        if len(change.files_modified) > 5:
            files_str += f" (+{len(change.files_modified) - 5} more)"
        lines.append(f"  Files: {files_str}")

    if change.before:
        lines.append(_format_test_result_line("Before", change.before, color=color))
    if change.after:
        lines.append(_format_test_result_line("After ", change.after, color=color))

    if change.tests_improved:
        improved_str = ", ".join(change.tests_improved[:3])
        if len(change.tests_improved) > 3:
            improved_str += f" (+{len(change.tests_improved) - 3})"
        lines.append(f"  Improved: {_c(improved_str, 'good', color=color)}")
    if change.tests_regressed:
        regressed_str = ", ".join(change.tests_regressed[:3])
        if len(change.tests_regressed) > 3:
            regressed_str += f" (+{len(change.tests_regressed) - 3})"
        lines.append(f"  Regressed: {_c(regressed_str, 'bad', color=color)}")

    lines.append(f"  Net: {change.net_improvement:+d}")
    lines.append(
        f"  Verdict: {_c(change.audit_verdict, verdict_color, color=color)}"
        f"  Decision: {_c(change.decision, decision_color, color=color)}"
    )

    if change.gaming_signals:
        lines.append(f"  Gaming signals: {', '.join(change.gaming_signals)}")

    # GAP 5: Change quality
    if change.change_quality and change.change_quality != "neutral":
        lines.append(f"  Quality: {change.change_quality}")

    # GAP 9: Prediction accuracy
    if change.prediction_accuracy and change.prediction_accuracy != "no_prediction":
        lines.append(
            f"  Prediction: {change.prediction_accuracy} ({change.prediction_match_pct:.0f}%)"
        )

    # GAP 12: Regression root cause
    if change.regression_root_cause:
        lines.append(f"  Regression cause: {change.regression_root_cause}")

    # GAP 2: Meta-analysis
    if change.meta_analysis:
        classification = change.meta_analysis.get("classification", "")
        if classification:
            lines.append(f"  Classification: {classification}")

    # GAP 4: Diff hunks
    if change.diff_hunks:
        lines.extend(_format_diff_hunks_text(change.diff_hunks, color=color))

    # GAP 3: Rule violations
    if change.rule_violations:
        for rv in change.rule_violations[:3]:
            lines.append(f"  Rule violation: {rv}")

    return "\n".join(lines)


def format_enforce_text(report: EnforceReport, *, color: bool = True) -> str:
    """Format enforcement report as terminal text."""
    lines: list[str] = []

    # Header
    lines.append(_c("ENFORCEMENT REPORT", "header", color=color))
    lines.append("=" * 60)
    lines.append("")

    # Summary
    lines.append(_c("Summary", "bold", color=color))
    lines.append(f"  Iterations: {report.total_iterations}")
    lines.append(f"  Committed: {_c(str(report.committed_count), 'good', color=color)}")
    lines.append(f"  Reverted:  {_c(str(report.reverted_count), 'bad', color=color)}")
    if report.rejected_count:
        lines.append(f"  Rejected:  {_c(str(report.rejected_count), 'warn', color=color)}")
    lines.append(f"  Vetoed:    {_c(str(report.vetoed_count), 'warn', color=color)}")
    lines.append(f"  Gaming:    {_c(str(report.gaming_detected_count), 'bad', color=color)}")
    lines.append(f"  Net improvement: {report.net_improvement:+d} tests")
    if report.duration_seconds > 0:
        lines.append(f"  Duration: {report.duration_seconds:.0f}s")
    # GAP 13: Efficiency
    if report.total_iterations > 0:
        lines.append(f"  Efficiency: {report.efficiency_ratio:.1%} (committed/total)")
    # GAP 13: Improvements per iteration
    ipi = _improvements_per_iteration(report)
    lines.append(f"  Improvements/iteration: {ipi:+.2f}")
    lines.append("")

    # GAP 11: Grade
    grade_info = grade_enforce_session(report)
    grade_letter = grade_info["grade"]
    grade_score = grade_info["score"]
    grade_color = "good" if grade_letter in ("A", "B") else (
        "warn" if grade_letter == "C" else "bad"
    )
    lines.append(_c("Grade", "bold", color=color))
    lines.append(
        f"  {_c(grade_letter, grade_color, color=color)}  ({grade_score}/100)"
    )
    if grade_info["factors"]:
        for factor, value in grade_info["factors"].items():
            sign = "+" if value > 0 else ""
            lines.append(f"    {factor}: {sign}{value}")
    lines.append("")

    # GAP 13: Prediction accuracy summary
    if report.prediction_accuracy_summary.get("total_with_prediction", 0) > 0:
        pa = report.prediction_accuracy_summary
        lines.append(_c("Prediction Accuracy", "bold", color=color))
        lines.append(f"  Accurate: {pa.get('accurate', 0)}")
        lines.append(f"  Partial:  {pa.get('partial', 0)}")
        lines.append(f"  Wrong:    {pa.get('wrong', 0)}")
        lines.append(f"  Overall:  {pa.get('accuracy_pct', 0):.1f}%")
        lines.append("")

    # Baseline vs final
    if report.baseline_result:
        lines.append(_c("Baseline", "bold", color=color))
        lines.append(_format_test_result_line("Start", report.baseline_result, color=color))
    if report.final_result:
        lines.append(_format_test_result_line("Final", report.final_result, color=color))
    lines.append("")

    # Iterations
    if report.changes:
        lines.append(_c("Iterations", "header", color=color))
        lines.append("-" * 40)
        for change in report.changes:
            lines.append(_format_change_text(change, color=color))
            lines.append("")

    # Challenges
    if report.challenges:
        lines.append(_c("Adversarial Challenges", "header", color=color))
        lines.append("-" * 40)
        for ch in report.challenges:
            lines.append(f"  Review range: {ch.iteration_range[0]}-{ch.iteration_range[1]}")
            lines.append(f"  Changes reviewed: {ch.changes_reviewed}")
            if ch.vetoed:
                lines.append(f"  Vetoed: {ch.vetoed}")
            for finding in ch.findings:
                lines.append(f"    - {finding}")
            lines.append("")

    # GAP 13: Change map (detailed with committed/reverted breakdown)
    detailed_map = _build_detailed_change_map(report)
    if detailed_map:
        lines.append(_c("Change Map (most modified files)", "bold", color=color))
        for entry in detailed_map[:10]:
            f = entry["file"]
            t = entry["total"]
            c = entry["committed"]
            r = entry["reverted"]
            lines.append(
                f"  {f:<40} (modified {t}x, {c} committed, {r} reverted)"
            )
        lines.append("")

    # GAP 13: Timeline
    timeline = _build_timeline(report)
    if timeline:
        lines.append(_c("Timeline", "bold", color=color))
        for entry in timeline:
            ts = entry["timestamp"] or "        "
            num = f"#{entry['number']}"
            status = entry["status"]
            delta = entry["delta"]
            commit_tag = f"  [{entry['commit_hash'][:7]}]" if entry["commit_hash"] else ""
            gaming_tag = "  (gaming)" if entry["gaming"] else ""
            lines.append(
                f"  {num:<4} {ts:<10} {status:<10} {delta:<12}{commit_tag}{gaming_tag}"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------

def format_enforce_json(report: EnforceReport) -> str:
    """Format enforcement report as JSON."""
    data = report.to_dict()

    # GAP 11: Grade
    data["grade"] = grade_enforce_session(report)

    # GAP 13: Improvements per iteration
    data["improvements_per_iteration"] = _improvements_per_iteration(report)

    # GAP 13: Detailed change map
    data["detailed_change_map"] = _build_detailed_change_map(report)

    # GAP 13: Timeline
    data["timeline"] = _build_timeline(report)

    return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def format_enforce_markdown(report: EnforceReport) -> str:
    """Format enforcement report as Markdown."""
    lines: list[str] = []

    lines.append("# Enforcement Report")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Iterations | {report.total_iterations} |")
    lines.append(f"| Committed | {report.committed_count} |")
    lines.append(f"| Reverted | {report.reverted_count} |")
    if report.rejected_count:
        lines.append(f"| Rejected | {report.rejected_count} |")
    lines.append(f"| Vetoed | {report.vetoed_count} |")
    lines.append(f"| Gaming detected | {report.gaming_detected_count} |")
    lines.append(f"| Net improvement | {report.net_improvement:+d} tests |")
    if report.duration_seconds > 0:
        lines.append(f"| Duration | {report.duration_seconds:.0f}s |")
    if report.total_iterations > 0:
        lines.append(f"| Efficiency | {report.efficiency_ratio:.1%} |")
    ipi = _improvements_per_iteration(report)
    lines.append(f"| Improvements/iteration | {ipi:+.2f} |")
    lines.append("")

    # GAP 11: Grade
    grade_info = grade_enforce_session(report)
    lines.append("## Grade")
    lines.append("")
    lines.append(f"**{grade_info['grade']}** ({grade_info['score']}/100)")
    lines.append("")
    if grade_info["factors"]:
        lines.append("| Factor | Points |")
        lines.append("|--------|--------|")
        for factor, value in grade_info["factors"].items():
            sign = "+" if value > 0 else ""
            lines.append(f"| {factor} | {sign}{value} |")
        lines.append("")

    # GAP 13: Prediction accuracy summary
    if report.prediction_accuracy_summary.get("total_with_prediction", 0) > 0:
        pa = report.prediction_accuracy_summary
        lines.append("## Prediction Accuracy")
        lines.append("")
        lines.append("| Outcome | Count |")
        lines.append("|---------|-------|")
        lines.append(f"| Accurate | {pa.get('accurate', 0)} |")
        lines.append(f"| Partial | {pa.get('partial', 0)} |")
        lines.append(f"| Wrong | {pa.get('wrong', 0)} |")
        lines.append(f"| Overall accuracy | {pa.get('accuracy_pct', 0):.1f}% |")
        lines.append("")

    # Baseline vs final
    if report.baseline_result:
        lines.append("## Baseline vs Final")
        lines.append("")
        lines.append("| | Passed | Failed | Total |")
        lines.append("|-|--------|--------|-------|")
        lines.append(
            f"| Baseline | {report.baseline_result.passed}"
            f" | {report.baseline_result.failed}"
            f" | {report.baseline_result.total} |"
        )
        if report.final_result:
            lines.append(
                f"| Final | {report.final_result.passed}"
                f" | {report.final_result.failed}"
                f" | {report.final_result.total} |"
            )
        lines.append("")

    # Iteration details
    if report.changes:
        lines.append("## Iterations")
        lines.append("")
        for change in report.changes:
            status_label = {
                "COMMITTED": "COMMITTED",
                "REVERTED": "REVERTED",
                "VETOED": "VETOED",
                "REJECTED": "REJECTED",
            }.get(change.decision, change.decision)
            lines.append(f"### Iteration {change.iteration} - {status_label}")
            lines.append("")
            if change.hypothesis:
                lines.append(f"**Hypothesis:** {change.hypothesis}")
                lines.append("")
            if change.files_modified:
                lines.append("**Files modified:**")
                for f in change.files_modified[:10]:
                    lines.append(f"- `{f}`")
                lines.append("")
            # GAP 4: Diff hunks in markdown
            if change.diff_hunks:
                lines.extend(_format_diff_hunks_markdown(change.diff_hunks))
                lines.append("")
            if change.before and change.after:
                lines.append(
                    f"**Tests:** {change.before.passed}/{change.before.total}"
                    f" -> {change.after.passed}/{change.after.total}"
                    f" (net {change.net_improvement:+d})"
                )
                lines.append("")
            lines.append(f"**Verdict:** {change.audit_verdict} | **Decision:** {change.decision}")
            if change.commit_hash:
                lines.append(f"**Commit:** `{change.commit_hash}`")
            if change.gaming_signals:
                lines.append(f"**Gaming signals:** {', '.join(change.gaming_signals)}")
            # GAP 5: Change quality
            if change.change_quality and change.change_quality != "neutral":
                lines.append(f"**Quality:** {change.change_quality}")
            # GAP 9: Prediction
            if change.prediction_accuracy and change.prediction_accuracy != "no_prediction":
                lines.append(
                    f"**Prediction:** {change.prediction_accuracy} "
                    f"({change.prediction_match_pct:.0f}%)"
                )
            # GAP 12: Regression root cause
            if change.regression_root_cause:
                lines.append(f"**Regression cause:** {change.regression_root_cause}")
            # GAP 2: Meta-analysis classification
            if change.meta_analysis and change.meta_analysis.get("classification"):
                lines.append(f"**Classification:** {change.meta_analysis['classification']}")
            if change.audit_reasons:
                lines.append("")
                lines.append("Audit notes:")
                for reason in change.audit_reasons:
                    lines.append(f"- {reason}")
            # GAP 3: Rule violations
            if change.rule_violations:
                lines.append("")
                lines.append("Rule violations:")
                for rv in change.rule_violations:
                    lines.append(f"- {rv}")
            lines.append("")

    # Challenges
    if report.challenges:
        lines.append("## Adversarial Challenges")
        lines.append("")
        for i, ch in enumerate(report.challenges, 1):
            lines.append(f"### Challenge {i}")
            lines.append(f"- Range: iterations {ch.iteration_range[0]}-{ch.iteration_range[1]}")
            lines.append(f"- Changes reviewed: {ch.changes_reviewed}")
            if ch.vetoed:
                lines.append(f"- Vetoed iterations: {ch.vetoed}")
            lines.append("")
            for finding in ch.findings:
                lines.append(f"- {finding}")
            lines.append("")

    # GAP 13: Change map (detailed with committed/reverted breakdown)
    detailed_map = _build_detailed_change_map(report)
    if detailed_map:
        lines.append("## Change Map")
        lines.append("")
        lines.append("| File | Total | Committed | Reverted |")
        lines.append("|------|-------|-----------|----------|")
        for entry in detailed_map[:10]:
            lines.append(
                f"| `{entry['file']}` | {entry['total']} | "
                f"{entry['committed']} | {entry['reverted']} |"
            )
        lines.append("")

    # GAP 13: Timeline
    timeline = _build_timeline(report)
    if timeline:
        lines.append("## Timeline")
        lines.append("")
        lines.append("| # | Time | Status | Delta | Commit | Gaming |")
        lines.append("|---|------|--------|-------|--------|--------|")
        for entry in timeline:
            ts = entry["timestamp"] or "-"
            commit = entry["commit_hash"][:7] if entry["commit_hash"] else "-"
            gaming = "Yes" if entry["gaming"] else ""
            lines.append(
                f"| {entry['number']} | {ts} | {entry['status']} | "
                f"{entry['delta']} | {commit} | {gaming} |"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def generate_report(
    project_root: str | None = None,
    *,
    format: str = "text",
    color: bool = True,
) -> str:
    """Generate an enforcement report in the specified format."""
    report = build_enforce_report(project_root)
    if format == "json":
        return format_enforce_json(report)
    if format == "markdown":
        return format_enforce_markdown(report)
    return format_enforce_text(report, color=color)
