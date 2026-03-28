"""Golden exemplar ranking system -- rank golden/good runs by efficiency."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .analyzer import TaskAnalysis, analyze_task
from .capture import build_fixture, detect_milestone
from .grader import GradeResult, RuleSet, grade_task, grade_tasks, load_rules
from .schema import AgentTask

# Built-in optimization profiles: weights for (steps, duration, cost, errors)
OPTIMIZATION_PROFILES: dict[str, dict[str, float]] = {
    "balanced": {"steps": 0.3, "duration": 0.3, "cost": 0.2, "errors": 0.2},
    "cost": {"steps": 0.1, "duration": 0.1, "cost": 0.7, "errors": 0.1},
    "speed": {"steps": 0.1, "duration": 0.7, "cost": 0.1, "errors": 0.1},
    "steps": {"steps": 0.7, "duration": 0.1, "cost": 0.1, "errors": 0.1},
}


@dataclass(slots=True)
class GoldenRank:
    """A ranked golden/good run with efficiency metrics."""

    task_id: str
    grade: str
    score: int
    efficiency: float  # 0.0 - 1.0, higher is better
    tier: str  # "EXEMPLAR", "EFFICIENT", "BASELINE"
    site_name: str
    step_count: int
    duration_s: float
    cost_usd: float
    error_count: int
    unique_tools: int
    milestones: list[str]
    flow_summary: str  # e.g. "cart->checkout->payment"
    optimization_notes: list[str]  # why this run is/isn't efficient

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "grade": self.grade,
            "score": self.score,
            "efficiency": round(self.efficiency, 4),
            "tier": self.tier,
            "site_name": self.site_name,
            "step_count": self.step_count,
            "duration_s": self.duration_s,
            "cost_usd": self.cost_usd,
            "error_count": self.error_count,
            "unique_tools": self.unique_tools,
            "milestones": self.milestones,
            "flow_summary": self.flow_summary,
            "optimization_notes": self.optimization_notes,
        }


def _resolve_profile(optimize: str | dict[str, float]) -> dict[str, float]:
    """Resolve an optimization profile from name or custom dict."""
    if isinstance(optimize, dict):
        return optimize
    if optimize not in OPTIMIZATION_PROFILES:
        raise ValueError(
            f"Unknown optimization profile: {optimize!r}. "
            f"Available: {', '.join(sorted(OPTIMIZATION_PROFILES))}"
        )
    return OPTIMIZATION_PROFILES[optimize]


@dataclass(slots=True)
class _SiteStats:
    """Min/max stats for a site used in normalization."""

    min_steps: int
    max_steps: int
    min_duration: float
    max_duration: float
    min_cost: float
    max_cost: float
    min_errors: int
    max_errors: int


def _compute_site_stats(
    analyses: list[TaskAnalysis],
) -> _SiteStats:
    """Compute min/max stats from a group of analyses for normalization."""
    steps = [a.step_count for a in analyses]
    durations = [a.total_duration_ms / 1000.0 for a in analyses]
    costs = [a.total_cost_usd for a in analyses]
    errors = [a.errors for a in analyses]
    return _SiteStats(
        min_steps=min(steps),
        max_steps=max(steps),
        min_duration=min(durations),
        max_duration=max(durations),
        min_cost=min(costs),
        max_cost=max(costs),
        min_errors=min(errors),
        max_errors=max(errors),
    )


def _normalize(value: float, low: float, high: float) -> float:
    """Min-max normalize a value to [0, 1]. Lower raw value = higher score."""
    if high == low:
        return 1.0
    # Invert: lower is better, so (high - value) / (high - low)
    return (high - value) / (high - low)


def compute_efficiency(
    analysis: TaskAnalysis,
    profile: dict[str, float],
    site_stats: _SiteStats,
) -> float:
    """Compute efficiency score for one task given site-level normalization stats.

    Uses min-max normalization within the site.  Each dimension is normalized
    independently (lower is better) and then combined via the profile weights.

    Args:
        analysis: Precomputed analysis for the task.
        profile: Weight dict with keys ``steps``, ``duration``, ``cost``, ``errors``.
        site_stats: Min/max statistics for the site group.

    Returns:
        Efficiency score in ``[0.0, 1.0]``.  Higher is better.
    """
    step_score = _normalize(analysis.step_count, site_stats.min_steps, site_stats.max_steps)
    duration_s = analysis.total_duration_ms / 1000.0
    duration_score = _normalize(duration_s, site_stats.min_duration, site_stats.max_duration)
    cost_score = _normalize(analysis.total_cost_usd, site_stats.min_cost, site_stats.max_cost)
    error_score = _normalize(analysis.errors, site_stats.min_errors, site_stats.max_errors)

    weighted = (
        profile.get("steps", 0.25) * step_score
        + profile.get("duration", 0.25) * duration_score
        + profile.get("cost", 0.25) * cost_score
        + profile.get("errors", 0.25) * error_score
    )
    return max(0.0, min(1.0, weighted))


def _milestones_for_task(task: AgentTask) -> list[str]:
    """Extract ordered unique milestones from a task."""
    milestones: list[str] = []
    seen: set[str] = set()
    for step in task.sorted_steps:
        milestone = detect_milestone(step)
        if milestone and milestone not in seen:
            seen.add(milestone)
            milestones.append(milestone)
    return milestones


def _flow_summary(milestones: list[str]) -> str:
    """Produce a short flow summary from milestones."""
    if not milestones:
        return "(no milestones)"
    return "->".join(m.lower() for m in milestones)


def _assign_tiers(ranks: list[GoldenRank]) -> None:
    """Assign EXEMPLAR/EFFICIENT/BASELINE tiers in-place.

    Top 10% (at least 1) = EXEMPLAR, top 33% = EFFICIENT, rest = BASELINE.
    """
    if not ranks:
        return
    n = len(ranks)
    exemplar_cutoff = max(1, math.ceil(n * 0.10))
    efficient_cutoff = max(exemplar_cutoff, math.ceil(n * 0.33))
    for i, rank in enumerate(ranks):
        if i < exemplar_cutoff:
            rank.tier = "EXEMPLAR"
        elif i < efficient_cutoff:
            rank.tier = "EFFICIENT"
        else:
            rank.tier = "BASELINE"


def _efficiency_notes(analysis: TaskAnalysis, site_stats: _SiteStats) -> list[str]:
    """Generate optimization notes for a single analysis."""
    notes: list[str] = []
    if analysis.step_count == site_stats.min_steps and site_stats.min_steps < site_stats.max_steps:
        notes.append("fewest steps in site group")
    if analysis.step_count == site_stats.max_steps and site_stats.min_steps < site_stats.max_steps:
        notes.append("most steps in site group")
    duration_s = analysis.total_duration_ms / 1000.0
    if duration_s == site_stats.min_duration and site_stats.min_duration < site_stats.max_duration:
        notes.append("fastest run in site group")
    if duration_s == site_stats.max_duration and site_stats.min_duration < site_stats.max_duration:
        notes.append("slowest run in site group")
    if analysis.errors == 0:
        notes.append("zero errors")
    elif analysis.errors == site_stats.max_errors and site_stats.max_errors > 0:
        notes.append(f"most errors ({analysis.errors}) in site group")
    if analysis.total_cost_usd == site_stats.min_cost and site_stats.min_cost < site_stats.max_cost:
        notes.append("lowest cost in site group")
    if analysis.is_spin:
        notes.append(f"tool spin detected ({analysis.max_repeat_tool} x{analysis.max_repeat_count})")
    return notes


def rank_golden_runs(
    tasks: list[AgentTask],
    rules: RuleSet | None = None,
    optimize: str | dict[str, float] = "balanced",
) -> dict[str, list[GoldenRank]]:
    """Grade all tasks, filter to GOLDEN/GOOD, group by site, rank by efficiency.

    Args:
        tasks: All tasks to evaluate.
        rules: Ruleset for grading.  Defaults to the built-in default.
        optimize: Profile name or custom weight dict.

    Returns:
        Dict keyed by ``site_name``, each value a list of :class:`GoldenRank`
        sorted by efficiency descending.
    """
    rules = rules or load_rules()
    profile = _resolve_profile(optimize)

    grades = grade_tasks(tasks, rules)
    grade_by_id = {g.task_id: g for g in grades}
    analyses = {t.task_id: analyze_task(t) for t in tasks}

    # Filter to GOLDEN and GOOD only
    golden_ids = {g.task_id for g in grades if g.grade in ("GOLDEN", "GOOD")}
    if not golden_ids:
        return {}

    # Group by site
    site_groups: dict[str, list[str]] = {}
    for task_id in golden_ids:
        site = analyses[task_id].site_name
        site_groups.setdefault(site, []).append(task_id)

    result: dict[str, list[GoldenRank]] = {}
    for site, task_ids in sorted(site_groups.items()):
        site_analyses = [analyses[tid] for tid in task_ids]
        stats = _compute_site_stats(site_analyses)

        ranks: list[GoldenRank] = []
        for tid in task_ids:
            a = analyses[tid]
            g = grade_by_id[tid]
            task = next(t for t in tasks if t.task_id == tid)
            milestones = _milestones_for_task(task)
            eff = compute_efficiency(a, profile, stats)
            notes = _efficiency_notes(a, stats)
            ranks.append(
                GoldenRank(
                    task_id=tid,
                    grade=g.grade,
                    score=g.score,
                    efficiency=eff,
                    tier="BASELINE",  # assigned below
                    site_name=site,
                    step_count=a.step_count,
                    duration_s=round(a.total_duration_ms / 1000.0, 2),
                    cost_usd=round(a.total_cost_usd, 4),
                    error_count=a.errors,
                    unique_tools=len(a.unique_tools),
                    milestones=milestones,
                    flow_summary=_flow_summary(milestones),
                    optimization_notes=notes,
                )
            )
        # Sort by efficiency descending (stable: ties keep original order)
        ranks.sort(key=lambda r: -r.efficiency)
        _assign_tiers(ranks)
        result[site] = ranks

    return result


def find_exemplars(
    tasks: list[AgentTask],
    rules: RuleSet | None = None,
    optimize: str | dict[str, float] = "balanced",
) -> list[GoldenRank]:
    """Return just the top-ranked (EXEMPLAR) run per site.

    Args:
        tasks: All tasks to evaluate.
        rules: Ruleset for grading.
        optimize: Profile name or custom weight dict.

    Returns:
        One :class:`GoldenRank` per site that has at least one golden/good run.
    """
    rankings = rank_golden_runs(tasks, rules=rules, optimize=optimize)
    exemplars: list[GoldenRank] = []
    for site_ranks in rankings.values():
        for rank in site_ranks:
            if rank.tier == "EXEMPLAR":
                exemplars.append(rank)
                break
    return exemplars


def explain_efficiency_gap(
    exemplar_analysis: TaskAnalysis,
    other_analysis: TaskAnalysis,
) -> list[str]:
    """Compare two analyses and explain WHY the exemplar is more efficient.

    Checks: fewer steps, fewer errors, less context used, better tool selection,
    fewer retries, reached deeper milestones.

    Args:
        exemplar_analysis: The higher-ranked analysis.
        other_analysis: The lower-ranked analysis.

    Returns:
        Human-readable explanations of the efficiency gap.
    """
    explanations: list[str] = []

    # Step count
    delta_steps = other_analysis.step_count - exemplar_analysis.step_count
    if delta_steps > 0:
        explanations.append(
            f"exemplar used {delta_steps} fewer steps "
            f"({exemplar_analysis.step_count} vs {other_analysis.step_count})"
        )
    elif delta_steps < 0:
        explanations.append(
            f"exemplar used {abs(delta_steps)} more steps but was more efficient overall"
        )

    # Errors
    delta_errors = other_analysis.errors - exemplar_analysis.errors
    if delta_errors > 0:
        explanations.append(
            f"exemplar had {delta_errors} fewer errors "
            f"({exemplar_analysis.errors} vs {other_analysis.errors})"
        )

    # Duration
    ex_dur = exemplar_analysis.total_duration_ms / 1000.0
    ot_dur = other_analysis.total_duration_ms / 1000.0
    if ot_dur > 0 and ex_dur < ot_dur * 0.8:
        explanations.append(
            f"exemplar was {ot_dur - ex_dur:.1f}s faster "
            f"({ex_dur:.1f}s vs {ot_dur:.1f}s)"
        )

    # Cost
    if other_analysis.total_cost_usd > 0 and (
        exemplar_analysis.total_cost_usd < other_analysis.total_cost_usd * 0.8
    ):
        explanations.append(
            f"exemplar cost ${exemplar_analysis.total_cost_usd:.4f} vs "
            f"${other_analysis.total_cost_usd:.4f}"
        )

    # Tool diversity
    ex_tools = set(exemplar_analysis.unique_tools)
    ot_tools = set(other_analysis.unique_tools)
    only_other = ot_tools - ex_tools
    if only_other:
        explanations.append(
            f"other run used extra tools not in exemplar: {', '.join(sorted(only_other))}"
        )

    # Spin detection
    if other_analysis.is_spin and not exemplar_analysis.is_spin:
        explanations.append(
            f"other run had tool spin ({other_analysis.max_repeat_tool} "
            f"x{other_analysis.max_repeat_count}), exemplar did not"
        )

    # Context usage
    if (
        other_analysis.max_context_usage_pct > 0
        and exemplar_analysis.max_context_usage_pct < other_analysis.max_context_usage_pct * 0.7
    ):
        explanations.append(
            f"exemplar used less context "
            f"({exemplar_analysis.max_context_usage_pct:.1f}% vs "
            f"{other_analysis.max_context_usage_pct:.1f}%)"
        )

    # Milestones
    ex_task = exemplar_analysis.task
    ot_task = other_analysis.task
    ex_milestones = _milestones_for_task(ex_task)
    ot_milestones = _milestones_for_task(ot_task)
    if len(ex_milestones) > len(ot_milestones):
        explanations.append(
            f"exemplar reached more milestones "
            f"({', '.join(ex_milestones)} vs {', '.join(ot_milestones) or 'none'})"
        )
    elif len(ot_milestones) > len(ex_milestones):
        explanations.append(
            f"other run reached more milestones but was less efficient overall"
        )

    if not explanations:
        explanations.append("runs are very similar in efficiency metrics")

    return explanations


def format_golden_ranking(
    rankings: dict[str, list[GoldenRank]],
    optimize: str | dict[str, float] = "balanced",
) -> str:
    """Format the ranking as terminal text.

    Args:
        rankings: Output from :func:`rank_golden_runs`.
        optimize: Profile name (used in header).

    Returns:
        Human-readable ranking text.
    """
    profile_name = optimize if isinstance(optimize, str) else "custom"
    lines: list[str] = [
        f"GOLDEN RANKING ({profile_name})",
        "=" * 60,
    ]

    if not rankings:
        lines.append("(no golden/good runs found)")
        return "\n".join(lines)

    for site, ranks in sorted(rankings.items()):
        golden_count = len(ranks)
        lines.append(f"\n{site} ({golden_count} golden/good runs)")
        for rank in ranks:
            tier_stars = {"EXEMPLAR": "***", "EFFICIENT": "**", "BASELINE": "*"}.get(
                rank.tier, "*"
            )
            task_short = rank.task_id[:12]
            lines.append(
                f"  {tier_stars:<4s} {task_short:<14s} eff={rank.efficiency:.2f}  "
                f"{rank.step_count} steps  {rank.duration_s:5.0f}s  "
                f"${rank.cost_usd:.2f}  {rank.flow_summary}"
            )

    # Exemplar insights
    exemplar_notes: list[str] = []
    for site, ranks in sorted(rankings.items()):
        for rank in ranks:
            if rank.tier == "EXEMPLAR" and rank.optimization_notes:
                exemplar_notes.append(
                    f"  {site}: {rank.task_id[:12]} -- {'; '.join(rank.optimization_notes)}"
                )
                break

    if exemplar_notes:
        lines.append("")
        lines.append("EXEMPLAR INSIGHTS:")
        lines.extend(exemplar_notes)

    return "\n".join(lines)


def capture_exemplar(
    tasks: list[AgentTask],
    rules: RuleSet | None = None,
    site: str | None = None,
    optimize: str | dict[str, float] = "balanced",
    output_path: str | Path | None = None,
) -> Path:
    """Find the exemplar for a site, build a fixture, add efficiency metadata, save.

    Args:
        tasks: All tasks.
        rules: Ruleset for grading.
        site: Target site name.  When omitted, uses the first exemplar found.
        optimize: Profile name or custom weight dict.
        output_path: Where to save the fixture.  Defaults to ``./exemplars/<site>.json``.

    Returns:
        Path to the saved fixture file.

    Raises:
        KeyError: If no exemplar is found for the requested site.
    """
    rankings = rank_golden_runs(tasks, rules=rules, optimize=optimize)

    target_rank: GoldenRank | None = None
    if site:
        site_ranks = rankings.get(site)
        if not site_ranks:
            available = sorted(rankings.keys())
            raise KeyError(
                f"No golden/good runs for site {site!r}. "
                f"Available sites: {', '.join(available) if available else '(none)'}"
            )
        target_rank = site_ranks[0]
    else:
        for site_ranks in rankings.values():
            for rank in site_ranks:
                if rank.tier == "EXEMPLAR":
                    target_rank = rank
                    break
            if target_rank is not None:
                break

    if target_rank is None:
        raise KeyError("No exemplar runs found in any site")

    task = next(t for t in tasks if t.task_id == target_rank.task_id)
    fixture = build_fixture(task)
    fixture["efficiency_metadata"] = target_rank.to_dict()

    if output_path is None:
        dest = Path("exemplars") / f"{target_rank.site_name}.json"
    else:
        dest = Path(output_path)

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    return dest
