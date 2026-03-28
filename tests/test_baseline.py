"""Tests for the baseline / overhead measurement system."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_xray.analyzer import analyze_task, analyze_tasks
from agent_xray.baseline import (
    Baseline,
    OverheadResult,
    PromptHashGroup,
    build_baseline,
    format_overhead_report,
    format_prompt_impact_report,
    generate_naked_prompt,
    group_by_prompt_hash,
    load_baseline,
    load_baselines,
    measure_all_overhead,
    measure_overhead,
    overhead_report_data,
    prompt_impact_data,
    save_baseline,
)
from agent_xray.grader import grade_tasks, load_rules
from agent_xray.schema import AgentStep, AgentTask, TaskOutcome, ToolContext


def _step(
    task_id: str,
    step: int,
    tool_name: str,
    tool_input: dict | None = None,
    *,
    tool_result: str | None = None,
    error: str | None = None,
    duration_ms: int | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cost_usd: float | None = None,
    page_url: str | None = None,
    system_prompt_hash: str | None = None,
) -> AgentStep:
    return AgentStep(
        task_id=task_id,
        step=step,
        tool_name=tool_name,
        tool_input=tool_input or {},
        tool_result=tool_result,
        error=error,
        duration_ms=duration_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        page_url=page_url,
        system_prompt_hash=system_prompt_hash,
    )


def _outcome(task_id: str, status: str, total_steps: int) -> TaskOutcome:
    return TaskOutcome(
        task_id=task_id,
        status=status,
        total_steps=total_steps,
        total_duration_s=total_steps * 0.5,
    )


def _browser_task(task_id: str = "browser-task", *, prompt_hash: str | None = None) -> AgentTask:
    """A small browser task with known structure."""
    steps = [
        _step(
            task_id, 1, "browser_navigate",
            {"url": "https://shop.example.test"},
            tool_result="Homepage loaded.",
            duration_ms=800,
            input_tokens=120, output_tokens=40, cost_usd=0.01,
            page_url="https://shop.example.test/",
            system_prompt_hash=prompt_hash,
        ),
        _step(
            task_id, 2, "browser_click",
            {"ref": "product-link", "label": "Wireless Headset"},
            tool_result="Product page loaded.",
            duration_ms=400,
            input_tokens=100, output_tokens=30, cost_usd=0.008,
            page_url="https://shop.example.test/products/headset",
            system_prompt_hash=prompt_hash,
        ),
        _step(
            task_id, 3, "browser_click",
            {"ref": "add-to-cart"},
            tool_result="Added to cart.",
            duration_ms=300,
            input_tokens=90, output_tokens=25, cost_usd=0.006,
            page_url="https://shop.example.test/cart",
            system_prompt_hash=prompt_hash,
        ),
        _step(
            task_id, 4, "browser_fill_ref",
            {"ref": "address-form", "fields": ["address", "zip"], "text": "123 Main St 60601"},
            tool_result="Form submitted.",
            duration_ms=600,
            input_tokens=110, output_tokens=35, cost_usd=0.009,
            page_url="https://shop.example.test/checkout",
            system_prompt_hash=prompt_hash,
        ),
    ]
    return AgentTask(
        task_id=task_id,
        task_text="Buy a wireless headset from shop.example.test.",
        task_category="commerce",
        steps=steps,
        outcome=_outcome(task_id, "success", len(steps)),
    )


def _bloated_task(task_id: str = "bloated-task", *, prompt_hash: str | None = None) -> AgentTask:
    """A task that does the same thing as _browser_task but with extra overhead."""
    base = _browser_task(task_id, prompt_hash=prompt_hash)
    extra_steps = [
        _step(
            task_id, 5, "web_search",
            {"query": "shop.example.test headset"},
            tool_result="Found shop.example.test",
            duration_ms=500, input_tokens=100, output_tokens=30, cost_usd=0.007,
            system_prompt_hash=prompt_hash,
        ),
        _step(
            task_id, 6, "browser_navigate",
            {"url": "https://shop.example.test/search"},
            tool_result="Search page.",
            duration_ms=400, input_tokens=90, output_tokens=25, cost_usd=0.005,
            page_url="https://shop.example.test/search",
            system_prompt_hash=prompt_hash,
        ),
        _step(
            task_id, 7, "browser_snapshot",
            {},
            tool_result="Snapshot captured.",
            duration_ms=200, input_tokens=80, output_tokens=20, cost_usd=0.004,
            page_url="https://shop.example.test/search",
            system_prompt_hash=prompt_hash,
        ),
        _step(
            task_id, 8, "browser_snapshot",
            {},
            tool_result="Snapshot captured.",
            duration_ms=200, input_tokens=80, output_tokens=20, cost_usd=0.004,
            page_url="https://shop.example.test/search",
            system_prompt_hash=prompt_hash,
        ),
        _step(
            task_id, 9, "browser_snapshot",
            {},
            tool_result="Snapshot captured.",
            duration_ms=200, input_tokens=80, output_tokens=20, cost_usd=0.004,
            page_url="https://shop.example.test/search",
            system_prompt_hash=prompt_hash,
        ),
        _step(
            task_id, 10, "browser_click",
            {"ref": "confirm"},
            tool_result="Confirmed.",
            duration_ms=300, input_tokens=90, output_tokens=25, cost_usd=0.005,
            page_url="https://shop.example.test/confirm",
            system_prompt_hash=prompt_hash,
        ),
    ]
    base.steps.extend(extra_steps)
    base.outcome = _outcome(task_id, "success", len(base.steps))
    return base


# ---------------------------------------------------------------------------
# generate_naked_prompt
# ---------------------------------------------------------------------------


class TestGenerateNakedPrompt:
    def test_browser_task_produces_sentences(self):
        task = _browser_task()
        prompt = generate_naked_prompt(task)
        assert "Go to https://shop.example.test" in prompt
        assert "Click" in prompt
        assert "Fill" in prompt

    def test_empty_task(self):
        task = AgentTask(task_id="empty", steps=[])
        prompt = generate_naked_prompt(task)
        assert prompt == ""

    def test_web_search_step(self):
        task = AgentTask(
            task_id="search-task",
            steps=[
                _step("search-task", 1, "web_search", {"query": "best pizza near me"}),
            ],
        )
        prompt = generate_naked_prompt(task)
        assert "Search the web for" in prompt
        assert "best pizza near me" in prompt

    def test_coding_steps(self):
        task = AgentTask(
            task_id="code-task",
            steps=[
                _step("code-task", 1, "read_file", {"path": "src/main.py"}),
                _step("code-task", 2, "edit_file", {"path": "src/main.py"}),
                _step("code-task", 3, "run_tests", {"command": "pytest tests/"}),
                _step("code-task", 4, "git_commit", {"message": "Fix bug"}),
            ],
        )
        prompt = generate_naked_prompt(task)
        assert "Read src/main.py" in prompt
        assert "Edit src/main.py" in prompt
        assert "Run pytest tests/" in prompt
        assert "Commit with message 'Fix bug'" in prompt

    def test_respond_step(self):
        task = AgentTask(
            task_id="respond-task",
            steps=[_step("respond-task", 1, "respond", {})],
        )
        prompt = generate_naked_prompt(task)
        assert "Respond with the result" in prompt


# ---------------------------------------------------------------------------
# build_baseline
# ---------------------------------------------------------------------------


class TestBuildBaseline:
    def test_captures_correct_metrics(self):
        task = _browser_task()
        analysis = analyze_task(task)
        baseline = build_baseline(task, analysis)

        assert baseline.task_id == "browser-task"
        assert baseline.site_name == analysis.site_name
        assert baseline.step_count == 4
        assert baseline.duration_s == analysis.total_duration_ms / 1000.0
        assert baseline.total_tokens_in == analysis.tokens_in
        assert baseline.total_tokens_out == analysis.tokens_out
        assert baseline.cost_usd == analysis.total_cost_usd
        assert baseline.error_count == 0
        assert isinstance(baseline.tool_sequence, list)
        assert len(baseline.tool_sequence) == 4
        assert baseline.naked_prompt != ""

    def test_user_text_captured(self):
        task = _browser_task()
        analysis = analyze_task(task)
        baseline = build_baseline(task, analysis)
        assert baseline.user_text == task.task_text


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------


class TestSaveLoadRoundTrip:
    def test_round_trip(self, tmp_path: Path):
        task = _browser_task()
        analysis = analyze_task(task)
        baseline = build_baseline(task, analysis)

        path = save_baseline(baseline, tmp_path / "test.json")
        loaded = load_baseline(path)

        assert loaded.task_id == baseline.task_id
        assert loaded.site_name == baseline.site_name
        assert loaded.step_count == baseline.step_count
        assert abs(loaded.duration_s - baseline.duration_s) < 0.01
        assert loaded.total_tokens_in == baseline.total_tokens_in
        assert loaded.total_tokens_out == baseline.total_tokens_out
        assert abs(loaded.cost_usd - baseline.cost_usd) < 0.0001
        assert loaded.error_count == baseline.error_count
        assert loaded.milestones == baseline.milestones
        assert loaded.tool_sequence == baseline.tool_sequence
        assert loaded.naked_prompt == baseline.naked_prompt

    def test_creates_parent_directories(self, tmp_path: Path):
        task = _browser_task()
        analysis = analyze_task(task)
        baseline = build_baseline(task, analysis)

        path = save_baseline(baseline, tmp_path / "deep" / "nested" / "bl.json")
        assert path.exists()
        loaded = load_baseline(path)
        assert loaded.task_id == baseline.task_id


class TestLoadBaselines:
    def test_loads_multiple(self, tmp_path: Path):
        for i, name in enumerate(["alpha", "beta"]):
            bl = Baseline(
                task_id=f"task-{i}",
                site_name=name,
                user_text=f"Task {name}",
                step_count=i + 3,
                duration_s=float(i + 1),
                total_tokens_in=100 * (i + 1),
                total_tokens_out=50 * (i + 1),
                cost_usd=0.01 * (i + 1),
                error_count=0,
                milestones=[],
                tool_sequence=["browser_navigate"],
                naked_prompt=f"Do {name}",
            )
            save_baseline(bl, tmp_path / f"{name}.json")

        baselines = load_baselines(tmp_path)
        assert len(baselines) == 2
        assert "alpha" in baselines
        assert "beta" in baselines

    def test_empty_dir(self, tmp_path: Path):
        baselines = load_baselines(tmp_path)
        assert baselines == {}

    def test_nonexistent_dir(self, tmp_path: Path):
        baselines = load_baselines(tmp_path / "nope")
        assert baselines == {}


# ---------------------------------------------------------------------------
# measure_overhead
# ---------------------------------------------------------------------------


class TestMeasureOverhead:
    def _make_baseline(self) -> Baseline:
        task = _browser_task()
        analysis = analyze_task(task)
        return build_baseline(task, analysis)

    def test_correct_percentages(self):
        baseline = self._make_baseline()
        bloated = _bloated_task()
        analysis = analyze_task(bloated)
        result = measure_overhead(bloated, analysis, "GOOD", baseline)

        # 10 steps vs 4 baseline = 150% overhead
        assert result.step_overhead_pct == pytest.approx(150.0)
        assert result.baseline_steps == 4
        assert result.actual_steps == 10
        assert result.overhead_category == "bloated"

    def test_efficient_overhead(self):
        baseline = self._make_baseline()
        # Same task = 0% overhead
        task = _browser_task("same-task")
        analysis = analyze_task(task)
        result = measure_overhead(task, analysis, "GOLDEN", baseline)

        assert result.step_overhead_pct == pytest.approx(0.0)
        assert result.overhead_category == "efficient"

    def test_success_delta_same(self):
        baseline = self._make_baseline()
        task = _browser_task("same-delta")
        analysis = analyze_task(task)
        result = measure_overhead(task, analysis, "GOOD", baseline)
        assert result.success_delta == "same"


class TestOverheadCategories:
    def _baseline(self, steps: int = 4) -> Baseline:
        return Baseline(
            task_id="bl",
            site_name="shop",
            user_text="test",
            step_count=steps,
            duration_s=2.0,
            total_tokens_in=400,
            total_tokens_out=100,
            cost_usd=0.03,
            error_count=0,
            milestones=[],
            tool_sequence=["browser_navigate"] * steps,
            naked_prompt="test",
        )

    def _task_with_n_steps(self, n: int) -> tuple[AgentTask, str]:
        task = AgentTask(
            task_id=f"task-{n}",
            steps=[
                _step(
                    f"task-{n}", i, "browser_navigate",
                    {"url": "https://shop.example.test"},
                    duration_ms=500,
                    page_url="https://shop.example.test/",
                )
                for i in range(1, n + 1)
            ],
            task_text="test",
            outcome=_outcome(f"task-{n}", "success", n),
        )
        return task, "OK"

    def test_efficient(self):
        baseline = self._baseline(10)
        task, grade = self._task_with_n_steps(14)  # 40% overhead
        analysis = analyze_task(task)
        result = measure_overhead(task, analysis, grade, baseline)
        assert result.overhead_category == "efficient"

    def test_acceptable(self):
        baseline = self._baseline(10)
        task, grade = self._task_with_n_steps(20)  # 100% overhead
        analysis = analyze_task(task)
        result = measure_overhead(task, analysis, grade, baseline)
        assert result.overhead_category == "acceptable"

    def test_bloated(self):
        baseline = self._baseline(10)
        task, grade = self._task_with_n_steps(30)  # 200% overhead
        analysis = analyze_task(task)
        result = measure_overhead(task, analysis, grade, baseline)
        assert result.overhead_category == "bloated"

    def test_pathological(self):
        baseline = self._baseline(10)
        task, grade = self._task_with_n_steps(45)  # 350% overhead
        analysis = analyze_task(task)
        result = measure_overhead(task, analysis, grade, baseline)
        assert result.overhead_category == "pathological"


class TestContributingFactors:
    def test_extra_tools_detected(self):
        baseline_task = _browser_task()
        baseline_analysis = analyze_task(baseline_task)
        baseline = build_baseline(baseline_task, baseline_analysis)

        bloated = _bloated_task()
        analysis = analyze_task(bloated)
        result = measure_overhead(bloated, analysis, "GOOD", baseline)

        factor_text = " ".join(result.contributing_factors)
        assert "Extra tools" in factor_text or "extra step" in factor_text

    def test_repeated_tools_detected(self):
        baseline_task = _browser_task()
        baseline_analysis = analyze_task(baseline_task)
        baseline = build_baseline(baseline_task, baseline_analysis)

        bloated = _bloated_task()
        analysis = analyze_task(bloated)
        result = measure_overhead(bloated, analysis, "WEAK", baseline)

        factor_text = " ".join(result.contributing_factors)
        # browser_snapshot repeated 3x
        assert "repeated" in factor_text or "Redundant" in factor_text or "extra step" in factor_text


# ---------------------------------------------------------------------------
# measure_all_overhead
# ---------------------------------------------------------------------------


class TestMeasureAllOverhead:
    def test_matches_only_site_baselines(self, tmp_path: Path):
        task = _browser_task()
        analysis = analyze_task(task)
        baseline = build_baseline(task, analysis)

        # Baseline keyed by site_name
        baselines = {baseline.site_name: baseline}
        grades = {"browser-task": "GOOD"}

        results = measure_all_overhead([task], grades, baselines)
        assert len(results) == 1
        assert results[0].task_id == "browser-task"

    def test_no_matching_baseline(self):
        task = _browser_task()
        results = measure_all_overhead([task], {"browser-task": "GOOD"}, {})
        assert len(results) == 0


# ---------------------------------------------------------------------------
# group_by_prompt_hash
# ---------------------------------------------------------------------------


class TestGroupByPromptHash:
    def test_groups_tasks_by_hash(self):
        tasks = [
            _browser_task("t1", prompt_hash="hash-a"),
            _browser_task("t2", prompt_hash="hash-a"),
            _bloated_task("t3", prompt_hash="hash-b"),
        ]
        analyses = {t.task_id: analyze_task(t) for t in tasks}
        grades = {"t1": "GOLDEN", "t2": "GOOD", "t3": "BROKEN"}

        groups = group_by_prompt_hash(tasks, analyses, grades)
        assert len(groups) == 2

        # Sorted by task count desc
        assert groups[0].task_count == 2
        assert groups[0].prompt_hash == "hash-a"
        assert groups[1].task_count == 1
        assert groups[1].prompt_hash == "hash-b"

    def test_golden_and_broken_rates(self):
        tasks = [
            _browser_task("t1", prompt_hash="hash-x"),
            _browser_task("t2", prompt_hash="hash-x"),
        ]
        analyses = {t.task_id: analyze_task(t) for t in tasks}
        grades = {"t1": "GOLDEN", "t2": "BROKEN"}

        groups = group_by_prompt_hash(tasks, analyses, grades)
        assert len(groups) == 1
        g = groups[0]
        assert g.golden_rate == pytest.approx(0.5)
        assert g.broken_rate == pytest.approx(0.5)

    def test_unknown_hash_when_none(self):
        task = _browser_task("no-hash")
        analyses = {task.task_id: analyze_task(task)}
        grades = {"no-hash": "OK"}

        groups = group_by_prompt_hash([task], analyses, grades)
        assert len(groups) == 1
        assert groups[0].prompt_hash == "unknown"

    def test_sample_task_ids_limited_to_3(self):
        tasks = [_browser_task(f"t{i}", prompt_hash="same") for i in range(10)]
        analyses = {t.task_id: analyze_task(t) for t in tasks}
        grades = {t.task_id: "OK" for t in tasks}

        groups = group_by_prompt_hash(tasks, analyses, grades)
        assert len(groups[0].sample_task_ids) == 3

    def test_with_baselines_computes_overhead(self):
        baseline_task = _browser_task("bl")
        baseline_analysis = analyze_task(baseline_task)
        baseline = build_baseline(baseline_task, baseline_analysis)
        baselines = {baseline.site_name: baseline}

        tasks = [
            _browser_task("t1", prompt_hash="hash-a"),
            _bloated_task("t2", prompt_hash="hash-a"),
        ]
        analyses = {t.task_id: analyze_task(t) for t in tasks}
        grades = {"t1": "GOLDEN", "t2": "WEAK"}

        groups = group_by_prompt_hash(tasks, analyses, grades, baselines)
        assert len(groups) == 1
        # Should have computed overhead
        assert groups[0].avg_overhead_pct != 0.0

    def test_single_task_per_hash(self):
        tasks = [
            _browser_task("lone", prompt_hash="single"),
        ]
        analyses = {tasks[0].task_id: analyze_task(tasks[0])}
        grades = {"lone": "GOOD"}

        groups = group_by_prompt_hash(tasks, analyses, grades)
        assert len(groups) == 1
        assert groups[0].task_count == 1
        assert groups[0].avg_steps == 4.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_baseline_with_zero_steps(self):
        baseline = Baseline(
            task_id="zero",
            site_name="shop",
            user_text="test",
            step_count=0,
            duration_s=0.0,
            total_tokens_in=0,
            total_tokens_out=0,
            cost_usd=0.0,
            error_count=0,
            milestones=[],
            tool_sequence=[],
            naked_prompt="",
        )
        task = _browser_task()
        analysis = analyze_task(task)
        result = measure_overhead(task, analysis, "OK", baseline)

        # With 0 baseline steps, overhead pct should be 0.0 (safe division)
        assert result.step_overhead_pct == 0.0
        assert result.overhead_category == "efficient"

    def test_no_matching_baseline_returns_empty(self):
        results = measure_all_overhead([], {}, {})
        assert results == []

    def test_to_dict_round_trips(self):
        task = _browser_task()
        analysis = analyze_task(task)
        baseline = build_baseline(task, analysis)
        d = baseline.to_dict()
        assert isinstance(d, dict)
        assert d["task_id"] == "browser-task"
        assert d["step_count"] == 4

    def test_overhead_result_to_dict(self):
        baseline = Baseline(
            task_id="bl",
            site_name="shop",
            user_text="test",
            step_count=4,
            duration_s=2.0,
            total_tokens_in=400,
            total_tokens_out=100,
            cost_usd=0.03,
            error_count=0,
            milestones=[],
            tool_sequence=["browser_navigate"],
            naked_prompt="test",
        )
        task = _browser_task()
        analysis = analyze_task(task)
        result = measure_overhead(task, analysis, "GOOD", baseline)
        d = result.to_dict()
        assert "step_overhead_pct" in d
        assert "overhead_category" in d
        assert isinstance(d["contributing_factors"], list)

    def test_prompt_hash_group_to_dict(self):
        g = PromptHashGroup(
            prompt_hash="abc123",
            task_count=5,
            avg_steps=10.5,
            avg_duration_s=3.2,
            avg_cost=0.045,
            avg_errors=1.2,
            golden_rate=0.6,
            broken_rate=0.1,
            avg_overhead_pct=75.3,
            sample_task_ids=["t1", "t2"],
        )
        d = g.to_dict()
        assert d["prompt_hash"] == "abc123"
        assert d["task_count"] == 5
        assert d["avg_steps"] == 10.5


# ---------------------------------------------------------------------------
# Format functions
# ---------------------------------------------------------------------------


class TestFormatOverheadReport:
    def test_header_present(self):
        text = format_overhead_report([])
        assert "PROMPT OVERHEAD ANALYSIS" in text
        assert "No tasks matched" in text

    def test_with_results(self):
        baseline_task = _browser_task()
        baseline_analysis = analyze_task(baseline_task)
        baseline = build_baseline(baseline_task, baseline_analysis)

        bloated = _bloated_task()
        analysis = analyze_task(bloated)
        result = measure_overhead(bloated, analysis, "GOOD", baseline)

        text = format_overhead_report([result])
        assert "PROMPT OVERHEAD ANALYSIS" in text
        assert "OVERHEAD DISTRIBUTION" in text
        assert "Bloated" in text or "bloated" in text

    def test_with_hash_groups(self):
        tasks = [
            _browser_task("t1", prompt_hash="hash-a"),
            _bloated_task("t2", prompt_hash="hash-b"),
        ]
        analyses = {t.task_id: analyze_task(t) for t in tasks}
        grades = {"t1": "GOLDEN", "t2": "WEAK"}

        baseline_task = _browser_task("bl")
        baseline = build_baseline(baseline_task, analyze_task(baseline_task))
        baselines = {baseline.site_name: baseline}

        grade_map = {t.task_id: grades[t.task_id] for t in tasks}
        results = measure_all_overhead(tasks, grade_map, baselines)
        hash_groups = group_by_prompt_hash(tasks, analyses, grades, baselines)

        text = format_overhead_report(results, hash_groups)
        assert "PROMPT HASH CORRELATION" in text


class TestFormatPromptImpactReport:
    def test_header_present(self):
        text = format_prompt_impact_report([])
        assert "PROMPT IMPACT ANALYSIS" in text
        assert "No prompt hash data" in text

    def test_with_groups(self):
        tasks = [
            _browser_task("t1", prompt_hash="hash-a"),
            _bloated_task("t2", prompt_hash="hash-b"),
        ]
        analyses = {t.task_id: analyze_task(t) for t in tasks}
        grades = {"t1": "GOLDEN", "t2": "BROKEN"}
        groups = group_by_prompt_hash(tasks, analyses, grades)

        text = format_prompt_impact_report(groups)
        assert "PROMPT IMPACT ANALYSIS" in text
        assert "hash-a" in text or "hash-b" in text


class TestOverheadReportData:
    def test_returns_structured_data(self):
        data = overhead_report_data([], [])
        assert data["total_measured"] == 0
        assert "distribution" in data
        assert "tasks" in data
        assert "hash_groups" in data


class TestPromptImpactData:
    def test_returns_structured_data(self):
        data = prompt_impact_data([])
        assert data["total_groups"] == 0
        assert "groups" in data
