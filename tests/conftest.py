from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_xray.schema import (
    AgentStep,
    AgentTask,
    BrowserContext,
    ModelContext,
    ReasoningContext,
    TaskOutcome,
    ToolContext,
)


def _step(
    task_id: str,
    step: int,
    tool_name: str,
    tool_input: dict[str, object] | None = None,
    *,
    tool_result: str | None = None,
    error: str | None = None,
    duration_ms: int | None = None,
    timestamp: str | None = None,
    model_name: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cost_usd: float | None = None,
    tools_available: list[str] | None = None,
    llm_reasoning: str | None = None,
    page_url: str | None = None,
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
        model_name=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        tools_available=tools_available,
        llm_reasoning=llm_reasoning,
        page_url=page_url,
    )


def _outcome(task_id: str, status: str, total_steps: int, final_answer: str | None) -> TaskOutcome:
    return TaskOutcome(
        task_id=task_id,
        status=status,
        total_steps=total_steps,
        total_duration_s=total_steps * 0.5,
        final_answer=final_answer,
        timestamp="2026-03-26T12:30:00Z",
    )


def _write_tasks(path: Path, tasks: list[AgentTask]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    trace_path = path / "trace_20260326.jsonl"
    lines: list[str] = []
    for task in tasks:
        for index, step in enumerate(task.sorted_steps):
            payload = step.to_dict()
            if index == 0:
                payload["user_text"] = task.task_text
                payload["task_category"] = task.task_category
            lines.append(json.dumps(payload, sort_keys=True))
        if task.outcome is not None:
            lines.append(
                json.dumps(
                    {
                        "event": "task_complete",
                        "task_id": task.task_id,
                        "status": task.outcome.status,
                        "final_answer": task.outcome.final_answer,
                        "total_steps": task.outcome.total_steps,
                        "total_duration_s": task.outcome.total_duration_s,
                        "timestamp": task.outcome.timestamp,
                    },
                    sort_keys=True,
                )
            )
    trace_path.write_text("\n".join(lines), encoding="utf-8")
    return path


@pytest.fixture
def sample_step() -> AgentStep:
    return AgentStep(
        task_id="sample-task",
        step=1,
        tool_name="browser_click",
        tool_input={"ref": "checkout-button", "note": "email alice@example.com"},
        tool_result="Opened checkout for alice@example.com",
        duration_ms=325,
        timestamp="2026-03-26T12:00:00Z",
        model=ModelContext(
            model_name="gpt-5-mini",
            temperature=0.1,
            tool_choice="auto",
            context_window=128000,
            context_usage_pct=0.42,
            compaction_count=1,
            input_tokens=180,
            output_tokens=72,
            cost_usd=0.014,
        ),
        tools=ToolContext(
            tools_available=["browser_click", "browser_snapshot", "browser_fill_ref"],
            system_prompt_hash="prompt-v1",
            message_count=6,
        ),
        reasoning=ReasoningContext(
            llm_reasoning="The checkout button is visible, so clicking should advance the flow.",
            correction_messages=["Avoid duplicate clicks."],
            spin_intervention="reassess_after_two_retries",
        ),
        browser=BrowserContext(
            page_url="https://shop.example.test/checkout",
            had_screenshot=True,
            snapshot_compressed=False,
        ),
        extensions={"session_id": "sess-123"},
    )


@pytest.fixture
def golden_task() -> AgentTask:
    task_id = "golden-task"
    steps = [
        _step(
            task_id,
            1,
            "browser_navigate",
            {"url": "https://shop.example.test"},
            tool_result="Homepage with wireless headset listing.",
            duration_ms=900,
            timestamp="2026-03-26T12:00:00Z",
            model_name="gpt-5-mini",
            input_tokens=120,
            output_tokens=40,
            cost_usd=0.01,
            tools_available=["browser_navigate", "browser_click", "browser_fill_ref"],
            llm_reasoning="Open the storefront first.",
            page_url="https://shop.example.test/",
        ),
        _step(
            task_id,
            2,
            "browser_click",
            {"ref": "product-wireless-headset"},
            tool_result="Product detail page loaded.",
            duration_ms=450,
            timestamp="2026-03-26T12:01:00Z",
            model_name="gpt-5-mini",
            input_tokens=100,
            output_tokens=28,
            cost_usd=0.008,
            tools_available=["browser_click", "browser_fill_ref"],
            page_url="https://shop.example.test/products/wireless-headset",
        ),
        _step(
            task_id,
            3,
            "browser_click",
            {"ref": "add-to-cart"},
            tool_result="Added to cart. Your cart subtotal is $129.",
            duration_ms=300,
            timestamp="2026-03-26T12:02:00Z",
            page_url="https://shop.example.test/cart",
        ),
        _step(
            task_id,
            4,
            "browser_fill_ref",
            {"ref": "shipping-form", "fields": ["address", "zip"], "text": "123 Main St 60601"},
            tool_result="Shipping form accepted.",
            duration_ms=600,
            timestamp="2026-03-26T12:03:00Z",
            page_url="https://shop.example.test/cart",
        ),
        _step(
            task_id,
            5,
            "browser_click",
            {"ref": "proceed-to-checkout"},
            tool_result="Checkout page with address review.",
            duration_ms=280,
            timestamp="2026-03-26T12:04:00Z",
            page_url="https://shop.example.test/checkout",
        ),
        _step(
            task_id,
            6,
            "browser_fill_ref",
            {
                "ref": "payment-form",
                "fields": ["card number", "cvv", "expiration"],
                "text": "4111 1111 1111 1111 123 12/29",
            },
            tool_result="card number cvv expir payment method confirmed",
            duration_ms=720,
            timestamp="2026-03-26T12:05:00Z",
            page_url="https://shop.example.test/payment",
        ),
        _step(
            task_id,
            7,
            "browser_click",
            {"ref": "place-order"},
            tool_result="Order review submitted.",
            duration_ms=320,
            timestamp="2026-03-26T12:06:00Z",
            page_url="https://shop.example.test/order/review",
        ),
        _step(
            task_id,
            8,
            "browser_snapshot",
            {},
            tool_result="Order confirmation page loaded.",
            duration_ms=250,
            timestamp="2026-03-26T12:07:00Z",
            page_url="https://shop.example.test/order/confirmation",
        ),
    ]
    return AgentTask(
        task_id=task_id,
        task_text="Buy the wireless headset and complete checkout on shop.example.test.",
        task_category="commerce",
        steps=steps,
        outcome=_outcome(task_id, "success", len(steps), "Order placed."),
    )


@pytest.fixture
def multi_step_commerce_task(golden_task: AgentTask) -> AgentTask:
    return golden_task


@pytest.fixture
def broken_task() -> AgentTask:
    task_id = "broken-task"
    steps = [
        _step(
            task_id,
            1,
            "browser_snapshot",
            {},
            error="Timed out waiting for checkout.",
            timestamp="2026-03-26T13:00:00Z",
            page_url="https://shop.example.test/checkout",
        ),
        _step(
            task_id,
            2,
            "browser_snapshot",
            {},
            error="Timed out waiting for checkout.",
            timestamp="2026-03-26T13:01:00Z",
            page_url="https://shop.example.test/checkout",
        ),
        _step(
            task_id,
            3,
            "browser_snapshot",
            {},
            error="Timed out waiting for checkout.",
            timestamp="2026-03-26T13:02:00Z",
            page_url="https://shop.example.test/checkout",
        ),
    ]
    return AgentTask(
        task_id=task_id,
        task_text="Try to recover the stuck checkout flow on shop.example.test.",
        task_category="commerce",
        steps=steps,
        outcome=_outcome(task_id, "failed", len(steps), None),
    )


@pytest.fixture
def coding_task() -> AgentTask:
    task_id = "coding-task"
    steps = [
        _step(task_id, 1, "read_file", {"path": "src/parser.py"}, tool_result="parser source"),
        _step(task_id, 2, "edit_file", {"path": "src/parser.py"}, tool_result="updated parser"),
        _step(
            task_id,
            3,
            "run_tests",
            {"command": "python -m pytest tests/test_parser.py"},
            tool_result="1 failed, 6 passed",
        ),
        _step(task_id, 4, "edit_file", {"path": "tests/test_parser.py"}, tool_result="updated test"),
        _step(
            task_id,
            5,
            "run_tests",
            {"command": "python -m pytest tests/test_parser.py"},
            tool_result="7 passed",
        ),
        _step(
            task_id,
            6,
            "git_commit",
            {"message": "Fix parser whitespace handling"},
            tool_result="Committed changes.",
        ),
    ]
    return AgentTask(
        task_id=task_id,
        task_text="Fix the parser regression in src/parser.py and verify with tests.",
        task_category="coding",
        steps=steps,
        outcome=_outcome(task_id, "success", len(steps), "Parser fixed."),
    )


@pytest.fixture
def research_task() -> AgentTask:
    task_id = "research-task"
    steps = [
        _step(
            task_id,
            1,
            "web_search",
            {"query": "agent observability best practices"},
            tool_result="Top results: https://a.example.test/report and https://c.example.test/guide",
        ),
        _step(
            task_id,
            2,
            "read_url",
            {"url": "https://a.example.test/report"},
            tool_result="According to https://a.example.test/report, traces need step metadata.",
            page_url="https://a.example.test/report",
        ),
        _step(
            task_id,
            3,
            "web_search",
            {"query": "LLM tracing semantic conventions"},
            tool_result="Source: https://b.example.test/spec",
        ),
        _step(
            task_id,
            4,
            "read_url",
            {"url": "https://b.example.test/spec"},
            tool_result="Reference doc at https://b.example.test/spec",
            page_url="https://b.example.test/spec",
        ),
        _step(
            task_id,
            5,
            "respond",
            {},
            tool_result=(
                "According to https://a.example.test/report, https://b.example.test/spec, "
                "and https://c.example.test/guide, durable traces need citations and tool context."
            ),
        ),
    ]
    return AgentTask(
        task_id=task_id,
        task_text="Research three sources about agent observability and summarize them.",
        task_category="research",
        steps=steps,
        outcome=_outcome(task_id, "success", len(steps), "Research summary delivered."),
    )


@pytest.fixture
def adapter_formats() -> tuple[str, ...]:
    return ("generic", "openai", "langchain", "anthropic", "crewai", "otel")


@pytest.fixture(params=("generic", "openai", "langchain", "anthropic", "crewai", "otel"))
def adapter_format(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture
def adapter_trace_paths() -> dict[str, Path]:
    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    return {
        "generic": fixtures_dir / "generic_trace.jsonl",
        "openai": fixtures_dir / "openai_trace.jsonl",
        "langchain": fixtures_dir / "langchain_trace.jsonl",
        "anthropic": fixtures_dir / "anthropic_trace.jsonl",
        "crewai": fixtures_dir / "crewai_trace.jsonl",
        "otel": fixtures_dir / "otel_trace.json",
    }


@pytest.fixture
def large_task() -> AgentTask:
    task_id = "large-task"
    urls = [
        "https://shop.example.test/",
        "https://shop.example.test/search?q=headset",
        "https://shop.example.test/products/wireless-headset",
        "https://shop.example.test/cart",
        "https://shop.example.test/checkout",
        "https://shop.example.test/payment",
    ]
    tools = [
        "browser_navigate",
        "browser_snapshot",
        "browser_click",
        "browser_fill_ref",
        "browser_wait",
    ]
    steps: list[AgentStep] = []
    for index in range(1, 61):
        tool_name = tools[(index - 1) % len(tools)]
        page_url = urls[(index - 1) % len(urls)]
        tool_input: dict[str, object] = {"ref": f"element-{index}"}
        if tool_name == "browser_navigate":
            tool_input = {"url": page_url}
        elif tool_name == "browser_fill_ref":
            tool_input = {"ref": f"field-{index}", "text": f"value-{index}"}
        steps.append(
            _step(
                task_id,
                index,
                tool_name,
                tool_input,
                tool_result=f"step-{index} completed on {page_url}",
                duration_ms=100 + index,
                timestamp=f"2026-03-26T12:{index % 60:02d}:00Z",
                model_name="gpt-5-mini",
                input_tokens=120 + index,
                output_tokens=40 + (index % 7),
                cost_usd=0.001,
                tools_available=[
                    "browser_navigate",
                    "browser_snapshot",
                    "browser_click",
                    "browser_fill_ref",
                    "browser_wait",
                ],
                llm_reasoning=f"Advance the checkout flow at step {index}.",
                page_url=page_url,
            )
        )
    return AgentTask(
        task_id=task_id,
        task_text="Perform a long commerce flow with repeated navigation and checkout updates.",
        task_category="commerce",
        steps=steps,
        outcome=_outcome(task_id, "success", len(steps), "Large trace completed."),
    )


@pytest.fixture
def write_trace_dir(tmp_path):
    def _write(name: str, tasks: list[AgentTask]) -> Path:
        return _write_tasks(tmp_path / name, tasks)

    return _write


@pytest.fixture
def clone_task():
    def _clone(
        task: AgentTask,
        task_id: str,
        *,
        model_name: str | None = None,
        cost_usd: float | None = None,
    ) -> AgentTask:
        steps: list[AgentStep] = []
        for step in task.sorted_steps:
            payload = step.to_dict()
            payload["task_id"] = task_id
            model_payload = dict(payload.get("model") or {})
            if model_name is not None:
                model_payload["model_name"] = model_name
            if cost_usd is not None:
                model_payload["cost_usd"] = cost_usd
            if model_payload:
                payload["model"] = model_payload
            steps.append(AgentStep.from_dict(payload))
        outcome = TaskOutcome.from_dict(task.outcome.to_dict()) if task.outcome else None
        if outcome is not None:
            outcome.task_id = task_id
        return AgentTask(
            task_id=task_id,
            steps=steps,
            task_text=task.task_text,
            task_category=task.task_category,
            day=task.day,
            metadata=dict(task.metadata),
            outcome=outcome,
        )

    return _clone


@pytest.fixture
def tmp_trace_dir(
    write_trace_dir,
    golden_task: AgentTask,
    broken_task: AgentTask,
    coding_task: AgentTask,
    research_task: AgentTask,
) -> Path:
    return write_trace_dir(
        "sample-traces",
        [golden_task, broken_task, coding_task, research_task],
    )
