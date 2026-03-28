from __future__ import annotations

from agent_xray.schema import AgentStep, AgentTask
from agent_xray.surface import (
    SimilarityBreakdown,
    diff_tasks,
    format_diff_summary,
    format_surface_text,
    reasoning_for_task,
    surface_for_task,
)


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


def test_missing_surfaces_summarized_once_not_per_step() -> None:
    """BUG #8: missing_surfaces should summarize once at task level, not per-step."""
    task = AgentTask(
        task_id="noisy",
        steps=[
            AgentStep("noisy", i, "noop", {})
            for i in range(1, 31)  # 30 sparse steps
        ],
    )
    surface = surface_for_task(task)

    # Task-level summary should exist
    summary = surface["missing_surfaces_summary"]
    assert isinstance(summary, dict)
    assert len(summary) > 0

    # Each field should report "missing in 30/30 steps"
    for field, description in summary.items():
        assert "missing in 30/30 steps" in description

    # Per-step missing_surfaces still exist for backward compat
    for step in surface["steps"]:
        assert isinstance(step["missing_surfaces"], list)


def test_missing_surfaces_summary_partial() -> None:
    """BUG #8: Summary should count correctly when a surface is present in some steps."""
    from agent_xray.schema import ToolContext

    task = AgentTask(
        task_id="partial",
        steps=[
            # Step 1 has tool context, step 2 does not
            AgentStep(
                "partial", 1, "search", {},
                tools=ToolContext(
                    tools_available=["search", "click"],
                    message_count=5,
                ),
            ),
            AgentStep("partial", 2, "click", {}),
            AgentStep("partial", 3, "respond", {}),
        ],
    )
    surface = surface_for_task(task)
    summary = surface["missing_surfaces_summary"]

    # tools_available_names is present in step 1 but missing in steps 2 and 3
    assert "tools_available_names" not in summary or "2/3" in summary.get("tools_available_names", "")


def test_format_surface_text_shows_task_level_missing() -> None:
    """BUG #8: Text output should show task-level missing summary, not per-step noise."""
    task = AgentTask(
        task_id="summary-test",
        steps=[
            AgentStep("summary-test", i, "noop", {})
            for i in range(1, 6)
        ],
    )
    surface = surface_for_task(task)
    text = format_surface_text(surface)

    # Should contain the task-level summary header
    assert "MISSING SURFACES (task-level summary)" in text
    # Should contain "missing in 5/5 steps" for fields missing everywhere
    assert "missing in 5/5 steps" in text


# --- BUG #12: Similarity metric tests ---


def test_diff_similarity_returns_breakdown() -> None:
    """diff_tasks should return a structured similarity breakdown, not just a float."""
    left = AgentTask(
        task_id="left",
        steps=[
            AgentStep("left", 1, "search", {"q": "a"}, tool_result="ok"),
            AgentStep("left", 2, "click", {"ref": "cta"}, tool_result="ok"),
        ],
    )
    right = AgentTask(
        task_id="right",
        steps=[
            AgentStep("right", 1, "search", {"q": "a"}, tool_result="ok"),
            AgentStep("right", 2, "click", {"ref": "cta"}, tool_result="ok"),
        ],
    )

    diff = diff_tasks(left, right)

    # Backward-compatible float key still exists
    assert isinstance(diff["similarity_score"], float)

    # New structured breakdown exists
    sim = diff["similarity"]
    assert isinstance(sim, dict)
    assert "tool_sequence_ratio" in sim
    assert "exact_signature_matches" in sim
    assert "total_steps" in sim
    assert "tool_name_matches" in sim
    assert "tool_name_and_input_matches" in sim
    assert "score" in sim
    assert "description" in sim

    # For identical tasks, all metrics should be at max
    assert sim["exact_signature_matches"] == 2
    assert sim["total_steps"] == 2
    assert sim["tool_name_matches"] == 2
    assert sim["tool_name_and_input_matches"] == 2
    assert sim["score"] == 1.0


def test_diff_similarity_distinguishes_tool_name_vs_input_matches() -> None:
    """Similarity breakdown should separately count tool name matches vs full matches."""
    left = AgentTask(
        task_id="left",
        steps=[
            AgentStep("left", 1, "search", {"q": "a"}, tool_result="ok"),
            AgentStep("left", 2, "click", {"ref": "button-1"}, tool_result="ok"),
        ],
    )
    right = AgentTask(
        task_id="right",
        steps=[
            AgentStep("right", 1, "search", {"q": "a"}, tool_result="ok"),
            AgentStep("right", 2, "click", {"ref": "button-2"}, tool_result="ok"),
        ],
    )

    diff = diff_tasks(left, right)
    sim = diff["similarity"]

    # Both steps have matching tool names
    assert sim["tool_name_matches"] == 2
    # Only step 1 has matching tool input
    assert sim["tool_name_and_input_matches"] == 1
    # Exact signature matches depends on full signature (tool_name + input + url + error)
    # Step 2 has different input so only step 1 is an exact match
    assert sim["exact_signature_matches"] == 1
    # Score should be between 0 and 1
    assert 0.0 < sim["score"] < 1.0


def test_diff_similarity_description_is_human_readable() -> None:
    """The description field should contain 'X of Y' phrasing."""
    left = AgentTask(
        task_id="left",
        steps=[
            AgentStep("left", 1, "a", {}, tool_result="ok"),
            AgentStep("left", 2, "b", {}, tool_result="ok"),
            AgentStep("left", 3, "c", {}, tool_result="ok"),
        ],
    )
    right = AgentTask(
        task_id="right",
        steps=[
            AgentStep("right", 1, "a", {}, tool_result="ok"),
            AgentStep("right", 2, "x", {}, tool_result="ok"),
            AgentStep("right", 3, "c", {}, tool_result="ok"),
        ],
    )

    diff = diff_tasks(left, right)
    desc = diff["similarity"]["description"]

    assert "of 3" in desc
    assert "exact matches" in desc
    assert "same tool name" in desc


def test_diff_similarity_zero_steps() -> None:
    """Empty tasks should produce a score of 1.0 and appropriate description."""
    left = AgentTask(task_id="left", steps=[])
    right = AgentTask(task_id="right", steps=[])

    diff = diff_tasks(left, right)
    sim = diff["similarity"]

    assert sim["score"] == 1.0
    assert sim["total_steps"] == 0
    assert "zero steps" in sim["description"].lower()


def test_diff_similarity_completely_different() -> None:
    """Completely different tasks should have low similarity."""
    left = AgentTask(
        task_id="left",
        steps=[
            AgentStep("left", 1, "a", {"x": 1}, tool_result="ok"),
            AgentStep("left", 2, "b", {"y": 2}, tool_result="ok"),
        ],
    )
    right = AgentTask(
        task_id="right",
        steps=[
            AgentStep("right", 1, "c", {"z": 3}, tool_result="ok"),
            AgentStep("right", 2, "d", {"w": 4}, tool_result="ok"),
        ],
    )

    diff = diff_tasks(left, right)
    sim = diff["similarity"]

    assert sim["exact_signature_matches"] == 0
    assert sim["tool_name_matches"] == 0
    assert sim["score"] == 0.0


def test_format_diff_summary_shows_description() -> None:
    """format_diff_summary should render the human-readable similarity description."""
    left = AgentTask(
        task_id="left-task",
        steps=[
            AgentStep("left-task", 1, "search", {"q": "a"}, tool_result="ok"),
            AgentStep("left-task", 2, "click", {"ref": "cta"}, tool_result="ok"),
        ],
    )
    right = AgentTask(
        task_id="right-task",
        steps=[
            AgentStep("right-task", 1, "search", {"q": "b"}, tool_result="ok"),
            AgentStep("right-task", 2, "click", {"ref": "cta"}, tool_result="ok"),
        ],
    )

    diff = diff_tasks(left, right)
    text = format_diff_summary(diff)

    assert "similarity:" in text
    assert "of 2" in text
    assert "exact matches" in text


def test_similarity_breakdown_to_dict() -> None:
    """SimilarityBreakdown.to_dict() should return a JSON-serializable dict."""
    import json

    breakdown = SimilarityBreakdown(
        tool_sequence_ratio=0.8,
        exact_signature_matches=3,
        total_steps=5,
        tool_name_matches=4,
        tool_name_and_input_matches=3,
        score=0.76,
        description="3 of 5 steps are exact matches; 4 of 5 share the same tool name",
    )

    d = breakdown.to_dict()
    assert d["score"] == 0.76
    assert d["total_steps"] == 5
    # Should be JSON-serializable
    assert json.loads(json.dumps(d))["exact_signature_matches"] == 3
