from __future__ import annotations

from pathlib import Path

from agent_xray.adapters import adapt, autodetect

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_autodetect_openai() -> None:
    assert autodetect(FIXTURES / "openai_trace.jsonl") == "openai"


def test_autodetect_langchain() -> None:
    assert autodetect(FIXTURES / "langchain_trace.jsonl") == "langchain"


def test_autodetect_anthropic() -> None:
    assert autodetect(FIXTURES / "anthropic_trace.jsonl") == "anthropic"


def test_autodetect_crewai() -> None:
    assert autodetect(FIXTURES / "crewai_trace.jsonl") == "crewai"


def test_autodetect_generic() -> None:
    assert autodetect(FIXTURES / "generic_trace.jsonl") == "generic"


def test_adapt_openai_produces_valid_steps() -> None:
    steps = adapt(FIXTURES / "openai_trace.jsonl", format="openai")
    assert len(steps) == 4
    assert steps[0].task_id == "run_market_001"
    assert steps[0].tool_name == "web_search"
    assert steps[0].input_tokens == 120
    assert steps[0].output_tokens == 60
    assert steps[2].tool_result == "NVDA closed at 913.52 USD."
    assert steps[3].tool_input["subject"] == "Market update"


def test_adapt_langchain_pairs_start_end() -> None:
    steps = adapt(FIXTURES / "langchain_trace.jsonl", format="langchain")
    assert len(steps) == 4
    assert steps[0].task_id == "task-lang-1"
    assert steps[0].tool_result == "Found 4 industry reports."
    assert steps[0].duration_ms == 2000
    assert steps[-1].task_id == "task-lang-2"


def test_adapt_anthropic_matches_tool_use_ids() -> None:
    steps = adapt(FIXTURES / "anthropic_trace.jsonl", format="anthropic")
    assert len(steps) == 4
    assert steps[1].tool_name == "browser_open"
    assert steps[1].tool_result == "Page loaded successfully."
    assert steps[3].output_tokens == 27


def test_adapt_crewai_extracts_agent_role() -> None:
    steps = adapt(FIXTURES / "crewai_trace.jsonl", format="crewai")
    assert len(steps) == 8
    assert steps[0].task_id == "Find market data"
    assert steps[0].tool_input["agent_role"] == "Researcher"
    assert steps[2].tool_input["value"] == "table.market"


def test_adapt_generic_round_trips() -> None:
    steps = adapt(FIXTURES / "generic_trace.jsonl", format="generic")
    assert len(steps) == 8
    assert steps[0].to_dict()["tool_input"]["query"] == "AI chip market 2026"
    assert steps[4].tool_input["value"] == "competitor revenue 2025"
    assert steps[-1].tool_result == "Revenue grew 12%."
