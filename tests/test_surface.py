from __future__ import annotations

from agent_xray.schema import AgentStep, AgentTask
from agent_xray.surface import diff_tasks, reasoning_for_task, surface_for_task


def test_surface_contains_history() -> None:
    task = AgentTask(
        task_id="task-1",
        task_text="inspect page",
        steps=[
            AgentStep(
                "task-1",
                1,
                "browser_navigate",
                {"url": "https://example.test"},
                tool_result="loaded",
            ),
            AgentStep(
                "task-1",
                2,
                "browser_snapshot",
                {},
                tool_result="checkout page",
                llm_reasoning="I should inspect the page",
            ),
        ],
    )
    surface = surface_for_task(task)
    assert surface["steps"][0]["conversation_history"][0]["content"] == "inspect page"
    reasoning = reasoning_for_task(task)
    assert reasoning["reasoning_chain"][1]["reasoning"] == "I should inspect the page"


def test_diff_detects_divergence() -> None:
    left = AgentTask(task_id="left", steps=[AgentStep("left", 1, "a", {}, tool_result="ok")])
    right = AgentTask(task_id="right", steps=[AgentStep("right", 1, "b", {}, tool_result="ok")])
    diff = diff_tasks(left, right)
    assert diff["diverged_at_step"] == 1


def test_surface_history_windowing() -> None:
    task = AgentTask(
        task_id="windowed",
        task_text="inspect long run",
        steps=[
            AgentStep("windowed", 1, "tool_1", {}, tool_result="ok"),
            AgentStep("windowed", 2, "tool_2", {}, tool_result="ok"),
            AgentStep("windowed", 3, "tool_3", {}, tool_result="ok"),
            AgentStep("windowed", 4, "tool_4", {}, tool_result="ok"),
            AgentStep("windowed", 5, "tool_5", {}, tool_result="ok"),
            AgentStep("windowed", 6, "tool_6", {}, tool_result="ok"),
            AgentStep("windowed", 7, "tool_7", {}, tool_result="ok"),
        ],
    )

    surface = surface_for_task(task, max_history_steps=5)
    history = surface["steps"][-1]["conversation_history"]

    assert history[0]["content"] == "inspect long run"
    assert history[1]["content"] == "tool_1 {}"
    assert history[2]["content"] == "ok"
    assert history[3]["content"] == "[...8 steps omitted...]"
    assert history[-2]["content"] == "tool_6 {}"
    assert history[-1]["content"] == "ok"


def test_surface_completeness_for_sparse_trace() -> None:
    task = AgentTask(task_id="bare", steps=[AgentStep("bare", 1, "noop", {})])

    surface = surface_for_task(task)
    step = surface["steps"][0]

    assert step["completeness"] == 0.5
    assert len(step["missing_surfaces"]) == 23
    assert "memory" in step["missing_surfaces"]
    assert "rag" in step["missing_surfaces"]


def test_surface_includes_memory_and_rag_fields() -> None:
    task = AgentTask(
        task_id="memory-rag",
        steps=[
            AgentStep(
                "memory-rag",
                1,
                "search",
                {"query": "policies"},
                extensions={
                    "memory_query": "customer preferences",
                    "memory_results": ["prefers email", "enterprise tier"],
                    "memory_store_key": "customer:42",
                    "rag_query": "refund policy",
                    "rag_documents_count": 2,
                    "rag_relevance_scores": [0.97, 0.84],
                },
            ),
        ],
    )

    surface = surface_for_task(task)
    step = surface["steps"][0]

    assert step["memory_query"] == "customer preferences"
    assert step["memory_results"] == ["prefers email", "enterprise tier"]
    assert step["memory_store_key"] == "customer:42"
    assert step["rag_query"] == "refund policy"
    assert step["rag_documents_count"] == 2
    assert step["rag_relevance_scores"] == [0.97, 0.84]
    assert "memory" not in step["missing_surfaces"]
    assert "rag" not in step["missing_surfaces"]


def test_diff_aligns_steps_by_tool_name_sequence() -> None:
    left = AgentTask(
        task_id="left",
        steps=[
            AgentStep("left", 1, "search", {"q": "a"}),
            AgentStep("left", 2, "click", {"ref": "cta"}),
            AgentStep("left", 3, "respond", {"text": "done"}),
        ],
    )
    right = AgentTask(
        task_id="right",
        steps=[
            AgentStep("right", 1, "search", {"q": "a"}),
            AgentStep("right", 2, "snapshot", {}),
            AgentStep("right", 3, "click", {"ref": "cta"}),
            AgentStep("right", 4, "respond", {"text": "done"}),
        ],
    )

    diff = diff_tasks(left, right)

    assert diff["diverged_at_step"] == 2
    assert diff["divergence_point"]["status"] == "insert"
    assert diff["divergence_point"]["right_step"] == 2
    assert [entry["status"] for entry in diff["step_alignment"]] == [
        "match",
        "insert",
        "match",
        "match",
    ]
    assert 0.0 < diff["similarity_score"] < 1.0
