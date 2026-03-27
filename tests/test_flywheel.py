from __future__ import annotations

import json

from agent_xray.flywheel import run_flywheel


def test_flywheel_baseline_comparison(tmp_path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    trace = log_dir / "trace.jsonl"
    trace.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "task_id": "task-1",
                        "step": 1,
                        "tool_name": "web_search",
                        "tool_input": {"q": "laptop"},
                        "tool_result": "results",
                    }
                ),
                json.dumps(
                    {
                        "task_id": "task-1",
                        "step": 2,
                        "tool_name": "browser_navigate",
                        "tool_input": {"url": "https://shop.example.test"},
                        "tool_result": "checkout",
                        "page_url": "https://shop.example.test/checkout",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "grade_distribution": {"GOOD": 1},
                "task_grades": {"task-1": "GOOD"},
            }
        ),
        encoding="utf-8",
    )

    result = run_flywheel(log_dir, baseline_path=baseline_path)

    assert result.total_tasks == 1
    assert result.baseline_grade_distribution == {"GOOD": 1}
    assert result.grade_deltas is not None
    assert result.trend in {"stable", "improving", "degrading"}
