"""Tests for the model pricing system."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent_xray.pricing import (
    _BUNDLED_PATH,
    _reset_cache,
    format_model_pricing,
    get_model_cost,
    list_models,
    load_pricing,
    pricing_source,
)


@pytest.fixture(autouse=True)
def _clear_pricing_cache():
    """Reset the in-memory pricing cache between tests."""
    _reset_cache()
    yield
    _reset_cache()


# -- Bundled data integrity --------------------------------------------------


def test_bundled_pricing_exists():
    assert _BUNDLED_PATH.exists(), f"Bundled pricing not found at {_BUNDLED_PATH}"


def test_bundled_pricing_valid_json():
    data = json.loads(_BUNDLED_PATH.read_text(encoding="utf-8"))
    assert "models" in data
    assert "_meta" in data
    assert isinstance(data["models"], dict)
    assert len(data["models"]) > 10, "Expected at least 10 models in bundled pricing"


def test_bundled_pricing_has_key_models():
    data = load_pricing()
    models = data["models"]
    for model in ("gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "claude-sonnet-4-20250514"):
        assert model in models, f"Expected {model} in bundled pricing"


def test_bundled_pricing_entry_has_required_fields():
    data = load_pricing()
    for name, entry in data["models"].items():
        assert "input" in entry, f"Model {name} missing 'input' price"
        assert "output" in entry, f"Model {name} missing 'output' price"
        assert isinstance(entry["input"], (int, float)), f"Model {name} input not numeric"
        assert isinstance(entry["output"], (int, float)), f"Model {name} output not numeric"


def test_bundled_pricing_has_aliases():
    data = load_pricing()
    assert "aliases" in data
    aliases = data["aliases"]
    assert len(aliases) > 0
    # Each alias target must exist in models
    models = data["models"]
    for alias, target in aliases.items():
        assert target in models, f"Alias {alias} -> {target}, but {target} not in models"


# -- load_pricing priority ---------------------------------------------------


def test_load_pricing_returns_bundled_by_default():
    data = load_pricing()
    assert "models" in data
    assert "gpt-4.1-nano" in data["models"]


def test_load_pricing_custom_path(tmp_path):
    custom = tmp_path / "custom.json"
    custom.write_text(
        json.dumps({
            "models": {"my-model": {"input": 1.0, "output": 2.0}},
            "aliases": {},
        }),
        encoding="utf-8",
    )
    data = load_pricing(str(custom))
    assert "my-model" in data["models"]
    assert "gpt-4.1-nano" not in data["models"]


def test_load_pricing_env_var(tmp_path, monkeypatch):
    custom = tmp_path / "env_pricing.json"
    custom.write_text(
        json.dumps({
            "models": {"env-model": {"input": 0.5, "output": 1.0}},
            "aliases": {},
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT_XRAY_PRICING", str(custom))
    data = load_pricing()
    assert "env-model" in data["models"]


# -- Model resolution --------------------------------------------------------


def test_exact_match():
    cost = get_model_cost("gpt-4.1-nano", input_tokens=1_000_000, output_tokens=0)
    assert cost == pytest.approx(0.10, abs=0.001)


def test_alias_resolution():
    cost = get_model_cost("claude-sonnet-4", input_tokens=1_000_000, output_tokens=0)
    assert cost == pytest.approx(3.0, abs=0.01)


def test_prefix_match_dated_model():
    # A model like "gpt-4.1-nano-2025-04-14" should match "gpt-4.1-nano"
    cost = get_model_cost("gpt-4.1-nano-2025-04-14", input_tokens=1_000_000, output_tokens=0)
    assert cost == pytest.approx(0.10, abs=0.001)


def test_unknown_model_returns_zero():
    cost = get_model_cost("totally-unknown-model-xyz", input_tokens=1000, output_tokens=500)
    assert cost == 0.0


# -- Cost calculation --------------------------------------------------------


def test_cost_calculation_input_only():
    # gpt-4.1-nano: input=$0.10/1M
    cost = get_model_cost("gpt-4.1-nano", input_tokens=500_000, output_tokens=0)
    assert cost == pytest.approx(0.05, abs=0.001)


def test_cost_calculation_output_only():
    # gpt-4.1-nano: output=$0.40/1M
    cost = get_model_cost("gpt-4.1-nano", input_tokens=0, output_tokens=1_000_000)
    assert cost == pytest.approx(0.40, abs=0.001)


def test_cost_calculation_mixed():
    # gpt-4.1: input=$2.00/1M, output=$8.00/1M
    cost = get_model_cost("gpt-4.1", input_tokens=100_000, output_tokens=50_000)
    expected = (100_000 * 2.0 + 50_000 * 8.0) / 1_000_000
    assert cost == pytest.approx(expected, abs=0.0001)


def test_cost_calculation_with_cache():
    # gpt-4.1: input=$2.00/1M, cached=$0.50/1M, output=$8.00/1M
    cost = get_model_cost(
        "gpt-4.1",
        input_tokens=100_000,
        output_tokens=50_000,
        cached_tokens=60_000,
    )
    regular = 100_000 - 60_000  # 40_000
    expected = (regular * 2.0 + 60_000 * 0.5 + 50_000 * 8.0) / 1_000_000
    assert cost == pytest.approx(expected, abs=0.0001)


def test_cost_calculation_cached_falls_back_to_input_price():
    # gpt-4-turbo has no cached_input, should use input price for cached tokens
    cost = get_model_cost(
        "gpt-4-turbo",
        input_tokens=100_000,
        output_tokens=0,
        cached_tokens=50_000,
    )
    # Without cached_input, all input is priced at input rate
    expected = (50_000 * 10.0 + 50_000 * 10.0) / 1_000_000
    assert cost == pytest.approx(expected, abs=0.0001)


def test_zero_tokens_returns_zero():
    cost = get_model_cost("gpt-4.1", input_tokens=0, output_tokens=0)
    assert cost == 0.0


# -- Round 12 regression: Anthropic cached_input rates -----------------------


@pytest.mark.parametrize("model,expected_cached", [
    ("claude-3.5-sonnet-20241022", 0.30),
    ("claude-sonnet-4-20250514", 0.30),
    ("claude-opus-4-20250514", 1.50),
    ("claude-haiku-4-20250414", 0.08),
    ("claude-sonnet-4-6-20260320", 0.30),
    ("claude-opus-4-6-20260320", 1.50),
    ("claude-haiku-4-5-20251001", 0.08),
])
def test_anthropic_cached_input_is_10pct_of_input(model, expected_cached):
    """Anthropic cached_input should be ~10% of input, not ~50%."""
    data = load_pricing()
    info = data.get("models", {}).get(model)
    assert info is not None, f"Model {model} not found in pricing"
    assert "cached_input" in info, f"Model {model} missing cached_input"
    assert info["cached_input"] == pytest.approx(expected_cached, abs=0.01), (
        f"Model {model}: cached_input={info['cached_input']}, expected={expected_cached} "
        f"(~10% of input={info['input']})"
    )


# -- list_models --------------------------------------------------------------


def test_list_models_returns_sorted():
    models = list_models()
    assert len(models) > 10
    assert models == sorted(models)


def test_list_models_with_custom_data():
    custom_data = {
        "models": {"zebra": {"input": 1.0, "output": 2.0}, "alpha": {"input": 0.5, "output": 1.0}},
        "aliases": {},
    }
    models = list_models(custom_data)
    assert models == ["alpha", "zebra"]


# -- pricing_source -----------------------------------------------------------


def test_pricing_source_bundled():
    source = pricing_source()
    assert "bundled" in source


def test_pricing_source_custom():
    source = pricing_source("/tmp/custom.json")
    assert "custom" in source
    assert "/tmp/custom.json" in source


# -- format_model_pricing -----------------------------------------------------


def test_format_model_pricing_known():
    text = format_model_pricing("gpt-4.1-nano")
    assert "gpt-4.1-nano" in text
    assert "Input" in text
    assert "Output" in text


def test_format_model_pricing_unknown():
    text = format_model_pricing("nonexistent-model-123")
    assert "not found" in text


# -- Integration with analyzer ------------------------------------------------


def test_analyzer_uses_pricing_for_missing_cost():
    """When cost_usd is None but tokens and model are present, pricing should kick in."""
    from agent_xray.analyzer import analyze_task
    from agent_xray.schema import AgentStep, AgentTask, ModelContext, TaskOutcome

    step = AgentStep(
        task_id="pricing-test",
        step=1,
        tool_name="browser_click",
        tool_input={},
        model=ModelContext(
            model_name="gpt-4.1-nano",
            input_tokens=10_000,
            output_tokens=5_000,
            cost_usd=None,  # No trace cost
        ),
    )
    task = AgentTask(
        task_id="pricing-test",
        steps=[step],
        outcome=TaskOutcome(
            task_id="pricing-test",
            status="success",
            total_steps=1,
        ),
    )
    analysis = analyze_task(task)
    # gpt-4.1-nano: 10K in * $0.10/1M + 5K out * $0.40/1M
    expected = (10_000 * 0.10 + 5_000 * 0.40) / 1_000_000
    assert analysis.total_cost_usd == pytest.approx(expected, abs=0.0001)
    assert analysis.total_cost_usd > 0


def test_analyzer_prefers_trace_cost_over_pricing():
    """When cost_usd is provided in the trace, pricing should not override it."""
    from agent_xray.analyzer import analyze_task
    from agent_xray.schema import AgentStep, AgentTask, ModelContext, TaskOutcome

    step = AgentStep(
        task_id="trace-cost-test",
        step=1,
        tool_name="browser_click",
        tool_input={},
        model=ModelContext(
            model_name="gpt-4.1-nano",
            input_tokens=10_000,
            output_tokens=5_000,
            cost_usd=0.999,  # Explicit trace cost
        ),
    )
    task = AgentTask(
        task_id="trace-cost-test",
        steps=[step],
        outcome=TaskOutcome(
            task_id="trace-cost-test",
            status="success",
            total_steps=1,
        ),
    )
    analysis = analyze_task(task)
    assert analysis.total_cost_usd == pytest.approx(0.999, abs=0.0001)


# -- Cost report integration --------------------------------------------------


def test_cost_report_shows_pricing_coverage():
    """Cost report data includes pricing_coverage section."""
    from agent_xray.analyzer import analyze_tasks
    from agent_xray.reports import report_cost, report_cost_data
    from agent_xray.schema import AgentStep, AgentTask, ModelContext, TaskOutcome

    step = AgentStep(
        task_id="coverage-test",
        step=1,
        tool_name="browser_click",
        tool_input={},
        model=ModelContext(
            model_name="gpt-4.1-nano",
            input_tokens=10_000,
            output_tokens=5_000,
        ),
    )
    task = AgentTask(
        task_id="coverage-test",
        steps=[step],
        outcome=TaskOutcome(
            task_id="coverage-test",
            status="success",
            total_steps=1,
        ),
    )
    analyses = analyze_tasks([task])
    data = report_cost_data([task], analyses)
    assert "pricing_coverage" in data
    assert data["pricing_coverage"]["priced_tasks"] == 1
    assert data["pricing_coverage"]["total_tasks"] == 1

    text = report_cost([task], analyses)
    assert "Pricing coverage:" in text


def test_cost_report_warns_about_unpriced_models():
    """Cost report warns when models have no pricing data."""
    from agent_xray.analyzer import analyze_tasks
    from agent_xray.reports import report_cost, report_cost_data
    from agent_xray.schema import AgentStep, AgentTask, ModelContext, TaskOutcome

    step = AgentStep(
        task_id="unpriced-test",
        step=1,
        tool_name="some_tool",
        tool_input={},
        model=ModelContext(
            model_name="totally-unknown-model",
            input_tokens=10_000,
            output_tokens=5_000,
        ),
    )
    task = AgentTask(
        task_id="unpriced-test",
        steps=[step],
        outcome=TaskOutcome(
            task_id="unpriced-test",
            status="success",
            total_steps=1,
        ),
    )
    analyses = analyze_tasks([task])
    data = report_cost_data([task], analyses)
    assert "totally-unknown-model" in data["pricing_coverage"]["unpriced_models"]

    text = report_cost([task], analyses)
    assert "Pricing unavailable" in text
    assert "totally-unknown-model" in text
