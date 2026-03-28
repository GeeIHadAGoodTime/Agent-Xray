"""Task bank adapter for NOVVIOLA-style task_bank.json files.

Converts curated task banks with per-task success criteria into
agent-xray compatible evaluation. Preserves the exact fuzzy matching
algorithm from step-log-analysis (SequenceMatcher ratio + token overlap)
so that bank-to-task pairing is consistent across both systems.

Usage:

    from agent_xray.contrib.task_bank import grade_with_task_bank

    results = grade_with_task_bank(tasks, "path/to/task_bank.json")
    for r in results:
        print(f"{r.task_id}: {r.grade} ({r.score})")

The adapter works in three phases:

1. **Load** the task bank JSON (list of dicts or ``{"tasks": [...]}``).
2. **Match** each bank entry to an ``AgentTask`` using fuzzy text similarity
   with token overlap, stopword filtering, site/category hints, and quality
   tiebreaking.
3. **Evaluate** per-task criteria (``must_reach_url``, ``must_fill_fields``,
   ``payment_fields_visible``, etc.) and fold the results into the standard
   ``GradeResult`` produced by agent-xray's ruleset grader.
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..analyzer import TaskAnalysis, analyze_task
from ..grader import GradeResult, RuleSet, SignalResult, grade_task, load_rules
from ..schema import AgentTask

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FUZZY_THRESHOLD = 0.45

_MATCH_STOPWORDS = frozenset({
    "a", "an", "and", "at", "for", "from", "how", "in", "is", "it",
    "my", "of", "on", "or", "the", "to", "what", "which", "with", "your",
})

# All known criteria names that evaluate_task_criteria handles.
KNOWN_CRITERIA = frozenset({
    "must_answer_contains",
    "answer_type",
    "must_reach_url",
    "must_fill_fields",
    "min_urls",
    "max_steps",
    "payment_fields_visible",
    "must_not_fill_payment",
    "must_reach_cart",
    "must_reach_checkout",
    "must_use_tools",
    "no_browser_needed",
    "must_have_answer",
    "min_tool_count",
})


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_task_bank(path: str | Path) -> list[dict[str, Any]]:
    """Load a task_bank.json file and return the tasks list.

    Accepts both a bare JSON array and an object with a ``"tasks"`` key.

    Args:
        path: Filesystem path to the task bank JSON file.

    Returns:
        list[dict[str, Any]]: Parsed bank entries, each containing at
        minimum ``id``, ``user_text``, and ``success_criteria``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file contains an unsupported format.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Task bank not found at {resolved}")
    with open(resolved, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        tasks = data.get("tasks")
        if isinstance(tasks, list):
            return tasks
    raise ValueError(f"Unsupported task bank format at {resolved}")


# ---------------------------------------------------------------------------
# Fuzzy matching  (ported verbatim from step-log-analysis grader.py)
# ---------------------------------------------------------------------------


def _fuzzy_score(a: str, b: str) -> float:
    """SequenceMatcher ratio between two strings (case-insensitive)."""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _token_set(text: str) -> set[str]:
    """Extract lowercase alphanumeric tokens, dropping stopwords."""
    return {
        tok
        for tok in re.findall(r"[a-z0-9]+", text.lower())
        if tok and tok not in _MATCH_STOPWORDS
    }


def _token_coverage(a: str, b: str) -> float:
    """Overlap coverage of the shorter token set."""
    a_tokens = _token_set(a)
    b_tokens = _token_set(b)
    if not a_tokens or not b_tokens:
        return 0.0
    shared = len(a_tokens & b_tokens)
    return shared / max(1, min(len(a_tokens), len(b_tokens)))


def _site_from_urls(task: AgentTask, analysis: TaskAnalysis) -> str:
    """Extract a site hint from the task's URLs."""
    for url in analysis.unique_urls:
        host = urlparse(url).netloc.lower().replace("www.", "")
        if host:
            return host
    return ""


def match_task_to_bank(
    task: AgentTask,
    bank: list[dict[str, Any]],
    *,
    analysis: TaskAnalysis | None = None,
    threshold: float = FUZZY_THRESHOLD,
) -> dict[str, Any] | None:
    """Fuzzy-match an AgentTask to the best bank entry.

    Uses the same SequenceMatcher + token overlap algorithm from
    step-log-analysis so that matching behaviour is identical.

    Args:
        task: The agent-xray task to match.
        bank: Loaded task bank entries.
        analysis: Optional precomputed analysis (avoids recomputation).
        threshold: Minimum combined score to accept a match.

    Returns:
        The best-matching bank entry dict, or ``None`` if no entry
        exceeds the threshold.
    """
    if analysis is None:
        analysis = analyze_task(task)
    log_text = task.task_text or ""
    if not log_text:
        return None

    log_site = _site_from_urls(task, analysis) or analysis.site_name
    log_category = task.task_category or ""

    best_entry: dict[str, Any] | None = None
    best_score = 0.0

    for bt in bank:
        bank_text = bt.get("user_text", "")
        bank_site = bt.get("site", "")
        bank_cat = bt.get("category", "")

        if not bank_text:
            continue

        text_score = _fuzzy_score(bank_text, log_text)
        overlap_score = _token_coverage(bank_text, log_text)
        shared_tokens = _token_set(bank_text) & _token_set(log_text)

        # Require at least one shared non-stopword token unless text
        # similarity is very strong.
        if not shared_tokens and text_score < 0.65:
            continue

        cat_match = bool(
            log_category and bank_cat and log_category.lower() == bank_cat.lower()
        )
        site_match = False
        if cat_match and bank_site and bank_site != "none":
            site_match = bank_site.lower() in log_site.lower()

        signal_score = (0.7 * overlap_score) + (0.3 * text_score)
        if cat_match and site_match:
            if signal_score < 0.30:
                continue
            signal_score += 0.05

        if signal_score > best_score:
            best_score = signal_score
            best_entry = bt

    if best_entry is not None and best_score >= threshold:
        return best_entry
    return None


# ---------------------------------------------------------------------------
# Criterion evaluation
# ---------------------------------------------------------------------------


def _evaluate_must_answer_contains(
    value: list[str],
    task: AgentTask,
    analysis: TaskAnalysis,
) -> tuple[bool, str]:
    if not isinstance(value, list) or not value:
        return False, "must_answer_contains requires a non-empty list"
    haystacks: list[str] = []
    if task.outcome and task.outcome.final_answer:
        haystacks.append(task.outcome.final_answer.lower())
    if task.outcome and task.outcome.status:
        haystacks.append(task.outcome.status.lower())
    combined = "\n".join(haystacks)
    if not combined:
        return False, "no final answer or outcome recorded"
    missing = [n for n in value if str(n).lower() not in combined]
    if not missing:
        return True, "answer contained: %s" % ", ".join(str(x) for x in value)
    return False, "missing answer keywords: %s" % ", ".join(str(x) for x in missing)


def _evaluate_answer_type(
    value: str,
    task: AgentTask,
    analysis: TaskAnalysis,
) -> tuple[bool, str]:
    expected = str(value).strip().lower()
    final_answer = ""
    outcome_status = ""
    if task.outcome:
        final_answer = (task.outcome.final_answer or "").strip()
        outcome_status = (task.outcome.status or "").strip().lower()

    if expected == "factual":
        if final_answer:
            return True, "factual answer present (%d chars)" % len(final_answer)
        return False, "no factual answer recorded"

    if expected == "action":
        if outcome_status in {"success", "payment_gate", "cancelled", "completed"}:
            return True, "action outcome=%s" % outcome_status
        if final_answer:
            markers = (
                "sent", "scheduled", "added", "created", "playing",
                "turned", "set ", "activated", "draft", "queued", "payment_gate",
            )
            if any(m in final_answer.lower() for m in markers):
                return True, "action confirmation present"
        return False, "no action confirmation found"

    if expected == "consultative":
        if len(final_answer) >= 50:
            return True, "consultative answer present (%d chars)" % len(final_answer)
        if final_answer:
            return False, "answer too short for consultative (%d chars)" % len(final_answer)
        return False, "no consultative answer recorded"

    return True, "unknown answer_type '%s' (skipped)" % value


def _evaluate_must_reach_url(
    value: str,
    task: AgentTask,
    analysis: TaskAnalysis,
) -> tuple[bool, str]:
    pattern = value
    for url in analysis.unique_urls:
        parsed = urlparse(url)
        full = parsed.netloc + parsed.path
        if parsed.query:
            full += "?" + parsed.query
        if re.search(pattern, full, re.IGNORECASE):
            return True, "URL matched: %s" % url
    return False, "no URL matched pattern /%s/" % pattern


def _evaluate_must_fill_fields(
    value: list[str],
    task: AgentTask,
    analysis: TaskAnalysis,
) -> tuple[bool, str]:
    filled: set[str] = set()
    for step in task.sorted_steps:
        is_fill = "fill" in step.tool_name.lower() or "fields" in step.tool_input
        if not is_fill:
            continue
        inp = json.dumps(step.tool_input).lower()
        for field_pat in value:
            if re.search(field_pat, inp, re.IGNORECASE):
                filled.add(field_pat)
    missing = set(value) - filled
    if not missing:
        return True, "all fields filled: %s" % ", ".join(value)
    return False, "missing fills: %s" % ", ".join(missing)


def _evaluate_min_urls(
    value: int,
    task: AgentTask,
    analysis: TaskAnalysis,
) -> tuple[bool, str]:
    actual = len(analysis.unique_url_paths)
    if actual >= value:
        return True, "%d unique URLs (need %d)" % (actual, value)
    return False, "only %d unique URLs (need %d)" % (actual, value)


def _evaluate_max_steps(
    value: int,
    task: AgentTask,
    analysis: TaskAnalysis,
) -> tuple[bool, str]:
    actual = len(task.steps)
    if actual <= value:
        return True, "%d steps (limit %d)" % (actual, value)
    return False, "%d steps exceeded limit of %d" % (actual, value)


def _evaluate_payment_fields_visible(
    value: bool,
    task: AgentTask,
    analysis: TaskAnalysis,
) -> tuple[bool, str]:
    metrics = analysis.metrics()
    confirmed = metrics.get("commerce", {}).get("payment_fields_confirmed", False)
    reached = metrics.get("commerce", {}).get("reached_payment", False)
    if value and confirmed:
        return True, "payment card fields confirmed visible"
    if value and reached:
        return False, "payment page reached but card fields not confirmed strong"
    if value:
        return False, "payment fields never detected"
    return True, "not required"


def _evaluate_must_not_fill_payment(
    value: bool,
    task: AgentTask,
    analysis: TaskAnalysis,
) -> tuple[bool, str]:
    if not value:
        return True, "not required"
    evidence: list[str] = []
    for idx, step in enumerate(task.sorted_steps):
        payload = json.dumps(step.tool_input).lower()
        if not payload:
            continue
        has_card_number = bool(re.search(r"\b(?:\d[ -]?){13,19}\b", payload))
        has_expiry = bool(re.search(r"\b(?:0[1-9]|1[0-2])/\d{2,4}\b", payload))
        has_cvv = bool(re.search(r'"(?:cvv|cvc|security[_ ]?code)"\s*:\s*"\d{3,4}"', payload))
        has_payment_key = any(
            tok in payload
            for tok in (
                '"card_number"', '"cc-number"', '"exp_month"',
                '"exp_year"', '"holder_name"', '"billing_zip"',
            )
        )
        if has_card_number or has_expiry or has_cvv or has_payment_key:
            evidence.append("step %d %s" % (idx, step.tool_name))
    if evidence:
        return False, "payment details filled: %s" % ", ".join(evidence[:5])
    outcome_status = (task.outcome.status or "") if task.outcome else ""
    final_answer = (task.outcome.final_answer or "") if task.outcome else ""
    if outcome_status == "payment_gate" or "payment_gate" in final_answer.lower():
        return True, "stopped at payment gate without filling payment"
    return True, "no payment-fill evidence detected"


def _evaluate_must_reach_cart(
    value: bool,
    task: AgentTask,
    analysis: TaskAnalysis,
) -> tuple[bool, str]:
    metrics = analysis.metrics()
    reached = metrics.get("commerce", {}).get("reached_cart", False)
    if value and reached:
        return True, "cart reached"
    if value:
        return False, "cart not reached"
    return True, "not required"


def _evaluate_must_reach_checkout(
    value: bool,
    task: AgentTask,
    analysis: TaskAnalysis,
) -> tuple[bool, str]:
    metrics = analysis.metrics()
    reached = metrics.get("commerce", {}).get("reached_checkout", False)
    if value and reached:
        return True, "checkout reached"
    if value:
        return False, "checkout not reached"
    return True, "not required"


def _evaluate_must_use_tools(
    value: list[str],
    task: AgentTask,
    analysis: TaskAnalysis,
) -> tuple[bool, str]:
    used = analysis.unique_tools
    found: list[str] = []
    missing: list[str] = []
    for tool_pat in value:
        if any(tool_pat.lower() in t.lower() for t in used):
            found.append(tool_pat)
        else:
            missing.append(tool_pat)
    if not missing:
        return True, "all required tools used: %s" % ", ".join(value)
    return False, "missing tools: %s (used: %s)" % (
        ", ".join(missing),
        ", ".join(sorted(used)),
    )


def _evaluate_no_browser_needed(
    value: bool,
    task: AgentTask,
    analysis: TaskAnalysis,
) -> tuple[bool, str]:
    browser_tools = {
        t for t in analysis.unique_tools
        if "browser" in t.lower() or "navigate" in t.lower()
    }
    if value and not browser_tools:
        return True, "no browser tools used"
    if value and browser_tools:
        return True, "browser used (%s) but not a hard failure" % ", ".join(browser_tools)
    return True, "not checked"


def _evaluate_must_have_answer(
    value: bool,
    task: AgentTask,
    analysis: TaskAnalysis,
) -> tuple[bool, str]:
    final_answer = ""
    outcome_status = ""
    if task.outcome:
        final_answer = task.outcome.final_answer or ""
        outcome_status = task.outcome.status or ""
    if value and final_answer:
        return True, "answer present (%d chars)" % len(final_answer)
    if value and outcome_status in {"success", "completed"}:
        return True, "outcome=%s (answer may be in TTS)" % outcome_status
    if value:
        return False, "no final answer recorded"
    return True, "not required"


def _evaluate_min_tool_count(
    value: int,
    task: AgentTask,
    analysis: TaskAnalysis,
) -> tuple[bool, str]:
    actual = len(task.steps)
    if actual >= value:
        return True, "%d tool calls (need %d)" % (actual, value)
    return False, "only %d tool calls (need %d)" % (actual, value)


_CRITERION_DISPATCH: dict[str, Any] = {
    "must_answer_contains": _evaluate_must_answer_contains,
    "answer_type": _evaluate_answer_type,
    "must_reach_url": _evaluate_must_reach_url,
    "must_fill_fields": _evaluate_must_fill_fields,
    "min_urls": _evaluate_min_urls,
    "max_steps": _evaluate_max_steps,
    "payment_fields_visible": _evaluate_payment_fields_visible,
    "must_not_fill_payment": _evaluate_must_not_fill_payment,
    "must_reach_cart": _evaluate_must_reach_cart,
    "must_reach_checkout": _evaluate_must_reach_checkout,
    "must_use_tools": _evaluate_must_use_tools,
    "no_browser_needed": _evaluate_no_browser_needed,
    "must_have_answer": _evaluate_must_have_answer,
    "min_tool_count": _evaluate_min_tool_count,
}


def evaluate_task_criteria(
    task: AgentTask,
    analysis: TaskAnalysis,
    criteria: dict[str, Any],
) -> list[str]:
    """Evaluate per-task success criteria from a bank entry.

    Args:
        task: The agent-xray task to evaluate.
        analysis: Precomputed task analysis.
        criteria: ``success_criteria`` dict from a bank entry.

    Returns:
        list[str]: Human-readable pass/fail strings for each criterion,
        prefixed with ``[PASS]`` or ``[FAIL]``.
    """
    results: list[str] = []
    for cname, cvalue in criteria.items():
        evaluator = _CRITERION_DISPATCH.get(cname)
        if evaluator is not None:
            passed, explanation = evaluator(cvalue, task, analysis)
        else:
            passed = True
            explanation = "unknown criterion '%s' (skipped)" % cname
        tag = "[PASS]" if passed else "[FAIL]"
        results.append("%s %s: %s" % (tag, cname, explanation))
    return results


# ---------------------------------------------------------------------------
# Convenience grading function
# ---------------------------------------------------------------------------


def grade_with_task_bank(
    tasks: list[AgentTask],
    bank_path: str | Path,
    rules: str | Path | RuleSet | None = None,
    *,
    threshold: float = FUZZY_THRESHOLD,
) -> list[GradeResult]:
    """Grade tasks using both a ruleset and task-bank criteria.

    This is the main entry point. For each task it:

    1. Fuzzy-matches the task to a bank entry (if any).
    2. Grades the task against the given ruleset (default rules if omitted).
    3. Evaluates per-task bank criteria and folds pass/fail results into
       the ``GradeResult.reasons`` list.
    4. Optionally downgrades ``GOLDEN`` to ``GOOD`` if critical bank criteria
       fail (``must_reach_cart``, ``must_reach_url``, ``payment_fields_visible``).

    Args:
        tasks: Agent-xray tasks to grade.
        bank_path: Path to the task_bank.json file.
        rules: Ruleset name, path, or loaded ``RuleSet``. Defaults to the
            bundled ``"default"`` ruleset.
        threshold: Fuzzy match threshold for bank pairing.

    Returns:
        list[GradeResult]: One result per task (same order as input).
    """
    bank = load_task_bank(bank_path)
    if isinstance(rules, RuleSet):
        ruleset = rules
    else:
        ruleset = load_rules(rules)

    results: list[GradeResult] = []
    for task in tasks:
        analysis = analyze_task(task)
        result = grade_task(task, ruleset, analysis=analysis)
        bank_entry = match_task_to_bank(
            task, bank, analysis=analysis, threshold=threshold,
        )
        if bank_entry is not None:
            criteria = bank_entry.get("success_criteria", {})
            criterion_lines = evaluate_task_criteria(task, analysis, criteria)
            # Fold criterion lines into the grade result reasons
            result.reasons.extend(criterion_lines)
            # Inject bank metadata as extra signals
            bank_id = bank_entry.get("id", "unknown")
            result.signals.append(
                SignalResult(
                    name="task_bank_match",
                    passed=True,
                    points=0,
                    actual=bank_id,
                    reason="matched bank entry %s" % bank_id,
                )
            )
            # Count bank-criteria pass/fail
            n_pass = sum(1 for l in criterion_lines if l.startswith("[PASS]"))
            n_fail = sum(1 for l in criterion_lines if l.startswith("[FAIL]"))
            n_total = n_pass + n_fail
            result.signals.append(
                SignalResult(
                    name="task_bank_criteria",
                    passed=n_fail == 0,
                    points=0,
                    actual={"passed": n_pass, "failed": n_fail, "total": n_total},
                    reason="%d/%d bank criteria passed" % (n_pass, n_total),
                )
            )
            # Downgrade GOLDEN -> GOOD when critical bank criteria fail
            critical_criteria = {
                "must_reach_cart", "must_reach_url",
                "payment_fields_visible", "must_reach_checkout",
            }
            if result.grade == "GOLDEN" and n_fail > 0:
                failed_names = set()
                for line in criterion_lines:
                    if line.startswith("[FAIL]"):
                        crit_name = line.split("]", 1)[1].strip().split(":")[0].strip()
                        failed_names.add(crit_name)
                if failed_names & critical_criteria:
                    result.grade = "GOOD"
                    result.reasons.append(
                        "capped at GOOD: critical bank criteria failed (%s)"
                        % ", ".join(failed_names & critical_criteria)
                    )
        results.append(result)
    return results


__all__ = [
    "FUZZY_THRESHOLD",
    "KNOWN_CRITERIA",
    "evaluate_task_criteria",
    "grade_with_task_bank",
    "load_task_bank",
    "match_task_to_bank",
]
