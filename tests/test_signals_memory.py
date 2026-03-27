from __future__ import annotations

from agent_xray.schema import AgentStep, AgentTask
from agent_xray.signals.memory import MemoryDetector


def _step(
    step: int,
    tool_name: str,
    tool_input: dict[str, object] | None = None,
    *,
    tool_result: str | None = None,
    error: str | None = None,
    llm_reasoning: str | None = None,
) -> AgentStep:
    return AgentStep(
        task_id="task-memory",
        step=step,
        tool_name=tool_name,
        tool_input=tool_input or {},
        tool_result=tool_result,
        error=error,
        llm_reasoning=llm_reasoning,
    )


def test_memory_detector_detects_store_operation() -> None:
    detector = MemoryDetector()
    signals = detector.detect_step(
        _step(1, "memory_store", {"memory_key": "customer.profile", "value": "vip"})
    )
    assert signals["is_memory_store"] is True
    assert signals["memory_key"] == "customer.profile"


def test_memory_detector_detects_recall_operation() -> None:
    detector = MemoryDetector()
    signals = detector.detect_step(
        _step(1, "memory_recall", {"key": "customer.profile"}, tool_result="vip customer")
    )
    assert signals["is_memory_recall"] is True
    assert signals["is_rag_query"] is False


def test_memory_detector_detects_rag_query() -> None:
    detector = MemoryDetector()
    signals = detector.detect_step(
        _step(1, "semantic_search", {"query": "refund policy", "top_k": 3}, tool_result="3 chunks")
    )
    assert signals["is_rag_query"] is True
    assert signals["is_memory_recall"] is True


def test_memory_detector_summarizes_recall_hit_rate() -> None:
    detector = MemoryDetector()
    steps = [
        _step(1, "memory_recall", {"key": "profile"}, tool_result="customer is vip"),
        _step(2, "memory_recall", {"key": "missing"}, tool_result="memory miss"),
    ]
    summary = detector.summarize(
        AgentTask(task_id="task-memory", steps=steps),
        [detector.detect_step(step) for step in steps],
    )
    assert summary["recall_hit_rate"] == 0.5


def test_memory_detector_counts_context_injections() -> None:
    detector = MemoryDetector()
    steps = [
        _step(1, "inject_context", {"memory_key": "profile"}, tool_result="Injected context window."),
        _step(2, "respond", {}, llm_reasoning="Injected context from memory before answering."),
    ]
    summary = detector.summarize(
        AgentTask(task_id="task-memory", steps=steps),
        [detector.detect_step(step) for step in steps],
    )
    assert summary["context_injections"] == 2


def test_memory_detector_counts_operations_and_unique_keys() -> None:
    detector = MemoryDetector()
    steps = [
        _step(1, "memory_store", {"memory_key": "profile"}),
        _step(2, "memory_recall", {"key": "profile"}, tool_result="hit"),
        _step(3, "forget_memory", {"memory_key": "session.cache"}),
    ]
    summary = detector.summarize(
        AgentTask(task_id="task-memory", steps=steps),
        [detector.detect_step(step) for step in steps],
    )
    assert summary["memory_operations"] == 3
    assert summary["unique_keys"] == 2
