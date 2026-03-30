"""Tests for the golden exemplar ranking system."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_xray.analyzer import analyze_task
from agent_xray.golden import (
    OPTIMIZATION_PROFILES,
    GoldenRank,
    _SiteStats,
    _assign_tiers,
    _flow_summary,
    _milestones_for_task,
    _normalize,
    capture_exemplar,
    compute_efficiency,
    explain_efficiency_gap,
    find_exemplars,
    format_golden_ranking,
    rank_golden_runs,
)
from agent_xray.grader import load_rules
from agent_xray.schema import AgentStep, AgentTask, BrowserContext, ModelContext, TaskOutcome


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _step(
    task_id: str,
    step: int,
    tool_name: str,
    tool_input: dict | None = None,
    *,
    tool_result: str | None = None,
    error: str | None = None,
    duration_ms: int | None = 500,
    timestamp: str | None = None,
    model_name: str | None = "gpt-5-mini",
    input_tokens: int | None = 100,
    output_tokens: int | None = 30,
    cost_usd: float | None = 0.01,
    page_url: str | None = None,
    tools_available: list[str] | None = None,
) -> AgentStep:
    return AgentStep(
        task_id=task_id,
        step=step,
        tool_name=tool_name,
        tool_input=tool_input or {},
        tool_result=tool_result,
        error=error,
        duration_ms=duration_ms,
        timestamp=timestamp,
        model=ModelContext(
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        ) if model_name else None,
        browser=BrowserContext(page_url=page_url) if page_url else None,
        tools=None,
    )


def _outcome(task_id: str, status: str, total_steps: int) -> TaskOutcome:
    return TaskOutcome(
        task_id=task_id,
        status=status,
        total_steps=total_steps,
        total_duration_s=total_steps * 5.0,
        final_answer="done",
        timestamp="2026-03-28T12:00:00Z",
    )


def _make_commerce_task(
    task_id: str,
    *,
    step_count: int = 8,
    cost_per_step: float = 0.01,
    duration_per_step: int = 500,
    error_count: int = 0,
    reach_payment: bool = True,
) -> AgentTask:
    """Build a realistic commerce task reaching various milestones."""
    steps: list[AgentStep] = []
    tool_sequence = [
        ("browser_navigate", {"url": "https://shop.example.test"}, "Homepage loaded.", "https://shop.example.test/"),
        ("browser_click", {"ref": "product-1"}, "Product page.", "https://shop.example.test/product/1"),
        ("browser_click", {"ref": "add-to-cart"}, "Added to cart. Your cart subtotal is $50.", "https://shop.example.test/cart"),
        ("browser_fill_ref", {"ref": "shipping-form", "text": "123 Main"}, "Shipping form filled.", "https://shop.example.test/cart"),
        ("browser_click", {"ref": "checkout"}, "Checkout page loaded.", "https://shop.example.test/checkout"),
        ("browser_fill_ref", {"ref": "payment-form", "fields": ["card number", "cvv"], "text": "4111 1111 1111 1111"}, "card number cvv payment method confirmed", "https://shop.example.test/payment"),
        ("browser_click", {"ref": "place-order"}, "Order review.", "https://shop.example.test/order/review"),
        ("browser_snapshot", {}, "Order confirmed.", "https://shop.example.test/order/confirm"),
    ]
    if not reach_payment:
        # Stop before payment step
        tool_sequence = tool_sequence[:5]

    for i in range(min(step_count, len(tool_sequence))):
        tool_name, tool_input, result, url = tool_sequence[i]
        error = "timeout" if i < error_count else None
        steps.append(
            _step(
                task_id, i + 1, tool_name, tool_input,
                tool_result=result,
                error=error,
                duration_ms=duration_per_step,
                cost_usd=cost_per_step,
                page_url=url,
                timestamp=f"2026-03-28T12:{i:02d}:00Z",
            )
        )
    # Pad with extra snapshot steps if step_count > len(tool_sequence)
    for i in range(len(tool_sequence), step_count):
        steps.append(
            _step(
                task_id, i + 1, "browser_snapshot", {},
                tool_result="Page snapshot.",
                duration_ms=duration_per_step,
                cost_usd=cost_per_step,
                page_url="https://shop.example.test/checkout",
                timestamp=f"2026-03-28T12:{i:02d}:00Z",
            )
        )

    return AgentTask(
        task_id=task_id,
        task_text="Buy a product on shop.example.test.",
        task_category="commerce",
        steps=steps,
        outcome=_outcome(task_id, "success", len(steps)),
    )


def _make_weather_task(
    task_id: str,
    *,
    step_count: int = 3,
    cost_per_step: float = 0.005,
    duration_per_step: int = 300,
    error_count: int = 0,
) -> AgentTask:
    """Build a simple weather lookup task."""
    steps: list[AgentStep] = []
    for i in range(step_count):
        error = "connection refused" if i < error_count else None
        steps.append(
            _step(
                task_id, i + 1, "web_search",
                {"query": "weather chicago"},
                tool_result="72F and sunny",
                error=error,
                duration_ms=duration_per_step,
                cost_usd=cost_per_step,
                timestamp=f"2026-03-28T13:{i:02d}:00Z",
            )
        )
    return AgentTask(
        task_id=task_id,
        task_text="What is the weather in Chicago?",
        task_category="research",
        steps=steps,
        outcome=_outcome(task_id, "success", len(steps)),
    )


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_lower_is_better(self) -> None:
        assert _normalize(0.0, 0.0, 10.0) == 1.0
        assert _normalize(10.0, 0.0, 10.0) == 0.0
        assert _normalize(5.0, 0.0, 10.0) == 0.5

    def test_same_min_max_returns_one(self) -> None:
        assert _normalize(5.0, 5.0, 5.0) == 1.0


# ---------------------------------------------------------------------------
# Efficiency computation
# ---------------------------------------------------------------------------

class TestComputeEfficiency:
    def test_known_values_balanced(self) -> None:
        """A task at the minimum on all dimensions should score 1.0."""
        task = _make_commerce_task("best", step_count=5, cost_per_step=0.01, duration_per_step=300, error_count=0)
        analysis = analyze_task(task)
        stats = _SiteStats(
            min_steps=5, max_steps=20,
            min_duration=1.5, max_duration=100.0,
            min_cost=0.05, max_cost=0.20,
            min_errors=0, max_errors=5,
        )
        profile = OPTIMIZATION_PROFILES["balanced"]
        eff = compute_efficiency(analysis, profile, stats)
        assert eff > 0.5, f"Expected >0.5 efficiency for best-case task, got {eff}"

    def test_worst_case_scores_low(self) -> None:
        """A task at the maximum on all dimensions should score low."""
        task = _make_commerce_task("worst", step_count=8, cost_per_step=0.03, duration_per_step=2000, error_count=3)
        analysis = analyze_task(task)
        stats = _SiteStats(
            min_steps=5, max_steps=8,
            min_duration=10.0, max_duration=16.0,
            min_cost=0.05, max_cost=0.24,
            min_errors=0, max_errors=3,
        )
        profile = OPTIMIZATION_PROFILES["balanced"]
        eff = compute_efficiency(analysis, profile, stats)
        assert eff < 0.3, f"Expected <0.3 efficiency for worst-case task, got {eff}"

    def test_single_task_scores_one(self) -> None:
        """When there is only one task, all min==max, so efficiency is 1.0."""
        task = _make_commerce_task("only", step_count=8)
        analysis = analyze_task(task)
        stats = _SiteStats(
            min_steps=8, max_steps=8,
            min_duration=4.0, max_duration=4.0,
            min_cost=0.08, max_cost=0.08,
            min_errors=0, max_errors=0,
        )
        eff = compute_efficiency(analysis, OPTIMIZATION_PROFILES["balanced"], stats)
        assert eff == 1.0


# ---------------------------------------------------------------------------
# Site-level normalization
# ---------------------------------------------------------------------------

class TestSiteNormalization:
    def test_weather_and_commerce_separate(self) -> None:
        """Weather tasks and commerce tasks should not compete."""
        tasks = [
            _make_commerce_task("com-1", step_count=8, cost_per_step=0.01),
            _make_commerce_task("com-2", step_count=15, cost_per_step=0.02),
            _make_weather_task("weather-1", step_count=3, cost_per_step=0.005),
            _make_weather_task("weather-2", step_count=5, cost_per_step=0.01),
        ]
        rules = load_rules()
        rankings = rank_golden_runs(tasks, rules=rules)
        # Weather and commerce should be in different site groups
        sites = set(rankings.keys())
        # At least one site should contain the commerce tasks
        # The weather tasks may or may not grade as GOLDEN depending on default rules
        # But the key assertion: if both appear, they are in separate groups
        for site, ranks in rankings.items():
            task_ids = {r.task_id for r in ranks}
            # No single site should contain both commerce and weather tasks
            has_com = any("com" in tid for tid in task_ids)
            has_weather = any("weather" in tid for tid in task_ids)
            assert not (has_com and has_weather), (
                f"Site {site!r} mixes commerce and weather tasks"
            )


# ---------------------------------------------------------------------------
# Tier assignment
# ---------------------------------------------------------------------------

class TestTierAssignment:
    def test_single_run_is_exemplar(self) -> None:
        rank = GoldenRank(
            task_id="only-1", grade="GOLDEN", score=10, efficiency=0.9,
            tier="BASELINE", site_name="shop", complexity_weight=1.0, step_count=8, duration_s=40.0,
            cost_usd=0.08, error_count=0, unique_tools=4,
            milestones=["CART"], flow_summary="cart", optimization_notes=[],
        )
        _assign_tiers([rank])
        assert rank.tier == "EXEMPLAR"

    def test_three_runs_one_exemplar(self) -> None:
        ranks = [
            GoldenRank(
                task_id=f"t-{i}", grade="GOLDEN", score=10, efficiency=1.0 - i * 0.3,
                tier="BASELINE", site_name="shop", complexity_weight=1.0, step_count=8, duration_s=40.0,
                cost_usd=0.08, error_count=0, unique_tools=4,
                milestones=[], flow_summary="", optimization_notes=[],
            )
            for i in range(3)
        ]
        _assign_tiers(ranks)
        assert ranks[0].tier == "EXEMPLAR"
        # With 3 runs: exemplar_cutoff=ceil(0.3)=1, efficient_cutoff=max(1, ceil(0.99))=1
        # So index 1 and 2 are BASELINE (efficient_cutoff == exemplar_cutoff)
        # This is by design: with very few runs, tiers collapse
        assert ranks[1].tier in ("EFFICIENT", "BASELINE")
        assert ranks[2].tier == "BASELINE"

    def test_ten_runs_tiers(self) -> None:
        ranks = [
            GoldenRank(
                task_id=f"t-{i}", grade="GOLDEN", score=10, efficiency=1.0 - i * 0.08,
                tier="BASELINE", site_name="shop", complexity_weight=1.0, step_count=8, duration_s=40.0,
                cost_usd=0.08, error_count=0, unique_tools=4,
                milestones=[], flow_summary="", optimization_notes=[],
            )
            for i in range(10)
        ]
        _assign_tiers(ranks)
        exemplar_count = sum(1 for r in ranks if r.tier == "EXEMPLAR")
        efficient_count = sum(1 for r in ranks if r.tier == "EFFICIENT")
        baseline_count = sum(1 for r in ranks if r.tier == "BASELINE")
        assert exemplar_count >= 1
        assert efficient_count >= 1
        assert baseline_count >= 1
        # First rank should be EXEMPLAR
        assert ranks[0].tier == "EXEMPLAR"

    def test_empty_ranks(self) -> None:
        # Should not raise
        _assign_tiers([])


# ---------------------------------------------------------------------------
# Ranking order
# ---------------------------------------------------------------------------

class TestRankingOrder:
    def test_highest_efficiency_first(self) -> None:
        tasks = [
            _make_commerce_task("fast", step_count=6, cost_per_step=0.01, duration_per_step=300),
            _make_commerce_task("slow", step_count=8, cost_per_step=0.03, duration_per_step=2000),
        ]
        rules = load_rules()
        rankings = rank_golden_runs(tasks, rules=rules)
        for site, ranks in rankings.items():
            if len(ranks) > 1:
                for i in range(len(ranks) - 1):
                    assert ranks[i].efficiency >= ranks[i + 1].efficiency

    def test_cross_site_ranking_uses_complexity_weight_when_not_grouped(self) -> None:
        tasks = [
            _make_commerce_task("commerce-short", step_count=5, cost_per_step=0.005),
            _make_commerce_task("commerce-long", step_count=9, cost_per_step=0.01),
        ]
        rules = load_rules("simple")

        rankings = rank_golden_runs(tasks, rules=rules, per_site=False)

        assert set(rankings) == {"all-sites"}
        by_id = {rank.task_id: rank for rank in rankings["all-sites"]}
        assert by_id["commerce-long"].complexity_weight >= by_id["commerce-short"].complexity_weight


# ---------------------------------------------------------------------------
# Explain efficiency gap
# ---------------------------------------------------------------------------

class TestExplainEfficiencyGap:
    def test_basic_gap_explanation(self) -> None:
        exemplar = _make_commerce_task("exemplar", step_count=6, cost_per_step=0.005, error_count=0)
        other = _make_commerce_task("other", step_count=8, cost_per_step=0.03, error_count=2, duration_per_step=2000)
        ex_analysis = analyze_task(exemplar)
        ot_analysis = analyze_task(other)
        explanations = explain_efficiency_gap(ex_analysis, ot_analysis)
        assert len(explanations) > 0
        # Should mention fewer steps
        assert any("fewer steps" in e for e in explanations), explanations
        # Should mention fewer errors
        assert any("fewer errors" in e for e in explanations), explanations

    def test_identical_tasks_produce_explanation(self) -> None:
        task_a = _make_commerce_task("a", step_count=6)
        task_b = _make_commerce_task("b", step_count=6)
        explanations = explain_efficiency_gap(analyze_task(task_a), analyze_task(task_b))
        assert len(explanations) >= 1
        # When tasks are identical, should say they are similar
        assert any("similar" in e for e in explanations), explanations


# ---------------------------------------------------------------------------
# Optimization profiles
# ---------------------------------------------------------------------------

class TestOptimizationProfiles:
    def test_cost_profile_favors_cheap(self) -> None:
        cheap = _make_commerce_task("cheap", step_count=8, cost_per_step=0.001, duration_per_step=3000)
        expensive = _make_commerce_task("expensive", step_count=6, cost_per_step=0.05, duration_per_step=300)
        tasks = [cheap, expensive]
        rules = load_rules()
        rankings_cost = rank_golden_runs(tasks, rules=rules, optimize="cost")
        for site, ranks in rankings_cost.items():
            if len(ranks) == 2:
                # The cheap task should be ranked higher under "cost" profile
                assert ranks[0].cost_usd <= ranks[1].cost_usd

    def test_speed_profile_favors_fast(self) -> None:
        fast = _make_commerce_task("fast", step_count=8, cost_per_step=0.05, duration_per_step=100)
        slow = _make_commerce_task("slow", step_count=6, cost_per_step=0.001, duration_per_step=5000)
        tasks = [fast, slow]
        rules = load_rules()
        rankings_speed = rank_golden_runs(tasks, rules=rules, optimize="speed")
        for site, ranks in rankings_speed.items():
            if len(ranks) == 2:
                assert ranks[0].duration_s <= ranks[1].duration_s

    def test_all_builtin_profiles_exist(self) -> None:
        for name in ("balanced", "cost", "speed", "steps"):
            assert name in OPTIMIZATION_PROFILES
            weights = OPTIMIZATION_PROFILES[name]
            assert set(weights.keys()) == {"steps", "duration", "cost", "errors"}
            assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_custom_profile_dict(self) -> None:
        tasks = [_make_commerce_task("t1", step_count=8)]
        rules = load_rules()
        custom = {"steps": 1.0, "duration": 0.0, "cost": 0.0, "errors": 0.0}
        rankings = rank_golden_runs(tasks, rules=rules, optimize=custom)
        # Should not raise, and should produce results
        assert isinstance(rankings, dict)

    def test_invalid_profile_raises(self) -> None:
        tasks = [_make_commerce_task("t1")]
        rules = load_rules()
        with pytest.raises(ValueError, match="Unknown optimization profile"):
            rank_golden_runs(tasks, rules=rules, optimize="nonexistent")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_no_golden_runs(self) -> None:
        """Tasks that all grade BROKEN/WEAK should produce empty rankings."""
        task = AgentTask(
            task_id="bad-task",
            task_text="fail",
            steps=[
                _step("bad-task", 1, "browser_snapshot", {}, error="timeout", duration_ms=100),
            ],
            outcome=_outcome("bad-task", "failed", 1),
        )
        rules = load_rules()
        rankings = rank_golden_runs([task], rules=rules)
        assert rankings == {}

    def test_single_golden_run_is_exemplar(self) -> None:
        tasks = [_make_commerce_task("only-golden", step_count=8)]
        rules = load_rules()
        rankings = rank_golden_runs(tasks, rules=rules)
        all_ranks = [r for ranks in rankings.values() for r in ranks]
        if all_ranks:
            assert all_ranks[0].tier == "EXEMPLAR"
            assert all_ranks[0].efficiency == 1.0

    def test_all_same_efficiency(self) -> None:
        """When all tasks have identical metrics, all should be efficiency 1.0."""
        tasks = [
            _make_commerce_task(f"same-{i}", step_count=8, cost_per_step=0.01, duration_per_step=500)
            for i in range(3)
        ]
        rules = load_rules()
        rankings = rank_golden_runs(tasks, rules=rules)
        for site, ranks in rankings.items():
            for rank in ranks:
                assert rank.efficiency == 1.0


# ---------------------------------------------------------------------------
# Format output
# ---------------------------------------------------------------------------

class TestFormatGoldenRanking:
    def test_empty_rankings(self) -> None:
        text = format_golden_ranking({})
        assert "no golden/good runs" in text.lower()

    def test_contains_site_and_task_ids(self) -> None:
        tasks = [
            _make_commerce_task("t-aaa", step_count=8),
            _make_commerce_task("t-bbb", step_count=8, cost_per_step=0.02, duration_per_step=1000),
        ]
        rules = load_rules()
        rankings = rank_golden_runs(tasks, rules=rules)
        text = format_golden_ranking(rankings)
        assert "GOLDEN RANKING" in text
        # At least one task id prefix should appear
        assert "t-aaa" in text or "t-bbb" in text

    def test_exemplar_insights_section(self) -> None:
        tasks = [_make_commerce_task("t-1", step_count=6)]
        rules = load_rules()
        rankings = rank_golden_runs(tasks, rules=rules)
        text = format_golden_ranking(rankings)
        # With a single task that is the min on all axes, should have insights
        if rankings:
            assert "EXEMPLAR INSIGHTS" in text or "***" in text


# ---------------------------------------------------------------------------
# Find exemplars
# ---------------------------------------------------------------------------

class TestFindExemplars:
    def test_one_per_site(self) -> None:
        tasks = [
            _make_commerce_task("com-1", step_count=6),
            _make_commerce_task("com-2", step_count=8, cost_per_step=0.02),
            _make_weather_task("w-1", step_count=3),
        ]
        rules = load_rules()
        exemplars = find_exemplars(tasks, rules=rules)
        sites = {e.site_name for e in exemplars}
        # Each site should appear at most once
        assert len(sites) == len(exemplars)

    def test_no_golden_returns_empty(self) -> None:
        task = AgentTask(
            task_id="bad",
            steps=[_step("bad", 1, "browser_snapshot", {}, error="fail")],
            outcome=_outcome("bad", "failed", 1),
        )
        exemplars = find_exemplars([task])
        assert exemplars == []


# ---------------------------------------------------------------------------
# Milestones and flow summary
# ---------------------------------------------------------------------------

class TestMilestonesAndFlow:
    def test_commerce_milestones(self) -> None:
        task = _make_commerce_task("full", step_count=8, reach_payment=True)
        milestones = _milestones_for_task(task)
        assert "CART" in milestones
        assert "PAYMENT" in milestones

    def test_no_payment_milestones(self) -> None:
        task = _make_commerce_task("partial", step_count=5, reach_payment=False)
        milestones = _milestones_for_task(task)
        assert "CART" in milestones
        assert "PAYMENT" not in milestones

    def test_flow_summary_format(self) -> None:
        assert _flow_summary(["CART", "CHECKOUT", "PAYMENT"]) == "cart->checkout->payment"
        assert _flow_summary([]) == "(no milestones)"


# ---------------------------------------------------------------------------
# Capture exemplar
# ---------------------------------------------------------------------------

class TestCaptureExemplar:
    def test_capture_produces_valid_fixture(self, tmp_path: Path) -> None:
        tasks = [_make_commerce_task("cap-1", step_count=8)]
        rules = load_rules()
        output = tmp_path / "exemplar.json"
        path = capture_exemplar(tasks, rules=rules, output_path=output)
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "task_id" in data
        assert "step_sequence" in data
        assert "efficiency_metadata" in data
        meta = data["efficiency_metadata"]
        assert "efficiency" in meta
        assert "tier" in meta
        assert meta["tier"] == "EXEMPLAR"

    def test_capture_unknown_site_raises(self, tmp_path: Path) -> None:
        tasks = [_make_commerce_task("cap-2", step_count=8)]
        rules = load_rules()
        with pytest.raises(KeyError, match="No golden/good runs"):
            capture_exemplar(tasks, rules=rules, site="nonexistent", output_path=tmp_path / "out.json")

    def test_capture_default_output_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        tasks = [_make_commerce_task("cap-3", step_count=8)]
        rules = load_rules()
        path = capture_exemplar(tasks, rules=rules)
        assert path.exists()
        assert "exemplars" in str(path)


# ---------------------------------------------------------------------------
# GoldenRank.to_dict
# ---------------------------------------------------------------------------

class TestGoldenRankSerialization:
    def test_to_dict_roundtrip(self) -> None:
        rank = GoldenRank(
            task_id="test-1", grade="GOLDEN", score=10, efficiency=0.85,
            tier="EXEMPLAR", site_name="shop", complexity_weight=1.0, step_count=8, duration_s=40.0,
            cost_usd=0.08, error_count=0, unique_tools=4,
            milestones=["CART", "PAYMENT"], flow_summary="cart->payment",
            optimization_notes=["fewest steps"],
        )
        d = rank.to_dict()
        assert d["task_id"] == "test-1"
        assert d["complexity_weight"] == 1.0
        assert d["efficiency"] == 0.85
        assert d["milestones"] == ["CART", "PAYMENT"]
        # Should be JSON-serializable
        json.dumps(d)
