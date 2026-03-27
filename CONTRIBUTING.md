# Contributing

## Getting Started

```bash
git clone https://github.com/agent-xray/agent-xray.git
cd agent-xray
python -m pip install -e ".[all]"
python -m pytest
```

Useful local checks:

```bash
python -m ruff check src tests
python -m ruff format --check src tests
python -m mypy src/agent_xray --strict
python -m build --no-isolation
```

## Write Your First Signal Detector

Signal detectors live under [`src/agent_xray/signals`](src/agent_xray/signals/__init__.py). Each detector:

- inspects one step at a time with `detect_step()`
- summarizes those booleans into task-level metrics with `summarize()`
- returns flat metrics that a JSON ruleset can score

Minimal detector:

```python
from __future__ import annotations

from typing import Any

from agent_xray.schema import AgentStep, AgentTask


class ScreenshotRecoveryDetector:
    name = "screenshot_recovery"

    def detect_step(self, step: AgentStep) -> dict[str, bool]:
        browser = step.browser
        return {
            "used_screenshot": bool(browser and browser.had_screenshot),
            "succeeded": step.error is None and bool(step.tool_result),
        }

    def summarize(self, task: AgentTask, step_signals: list[dict[str, bool]]) -> dict[str, Any]:
        screenshot_steps = sum(1 for item in step_signals if item["used_screenshot"])
        return {
            "screenshot_steps": screenshot_steps,
            "recovered_after_visual_check": any(
                step_signals[index]["used_screenshot"] and step_signals[index + 1]["succeeded"]
                for index in range(len(step_signals) - 1)
            ),
        }
```

Try it locally:

```python
from agent_xray.analyzer import analyze_task

analysis = analyze_task(task, detectors=[ScreenshotRecoveryDetector()])
print(analysis.metrics()["recovered_after_visual_check"])
```

To ship an in-repo detector:

1. Add a new module under `src/agent_xray/signals/`.
2. Instantiate it in `discover_detectors()` inside [`src/agent_xray/signals/__init__.py`](src/agent_xray/signals/__init__.py).
3. Add or update tests that exercise both `detect_step()` and `summarize()`.
4. Add a ruleset entry if the new metric should affect grading.

External plugins can register detectors through the `agent_xray.signals` entry-point group.

## Write An Adapter

Adapters convert trace records from one framework into `AgentStep` objects. Put new adapters under `src/agent_xray/adapters/`.

Minimal adapter example:

```python
from __future__ import annotations

from pathlib import Path

from agent_xray.adapters import _iter_json_objects, _normalize_tool_input
from agent_xray.schema import AgentStep


def load(path: Path) -> list[AgentStep]:
    steps: list[AgentStep] = []
    for _, payload in _iter_json_objects(path):
        tool_name = payload.get("tool_name")
        if tool_name is None:
            continue
        steps.append(
            AgentStep.from_dict(
                {
                    "task_id": str(payload.get("task_id") or path.stem),
                    "step": len(steps) + 1,
                    "tool_name": str(tool_name),
                    "tool_input": _normalize_tool_input(payload.get("tool_input")),
                    "tool_result": payload.get("tool_result"),
                    "error": payload.get("error"),
                    "timestamp": payload.get("timestamp") or payload.get("ts"),
                }
            )
        )
    return steps
```

Then register the new format key in [`src/agent_xray/adapters/__init__.py`](src/agent_xray/adapters/__init__.py) so `--format your_format` works.

Adapter rules:

- Normalize into `AgentStep`; do not add framework-specific branches throughout the analyzer.
- Preserve timestamps, tool inputs, model metadata, and browser context when available.
- Drop malformed records quietly unless a warning materially helps debugging.
- Add focused tests with tiny fixture files rather than giant real-world traces.

## Add A Rule Set

Rulesets live in `src/agent_xray/rules/*.json`.

Start from `default.json` or `browser_flow.json`, then:

1. Pick the metrics you want to score.
2. Define signal entries using `field`, `op`, `value`, `points`, and `reason`.
3. Set `thresholds` or `grade_thresholds`.
4. Add `golden_requirements` if a run must hit a terminal condition before it can earn `GOLDEN`.

Example:

```json
{
  "name": "my_rules",
  "signals": [
    {
      "label": "uses_multiple_tools",
      "field": "unique_tools",
      "op": "gte",
      "value": 2,
      "points": 2,
      "reason": "used more than one tool"
    }
  ],
  "thresholds": {
    "GOLDEN": 6,
    "GOOD": 4,
    "OK": 2,
    "WEAK": 0
  }
}
```

Run it with:

```bash
agent-xray grade ./traces --rules ./src/agent_xray/rules/my_rules.json
```

## Code Style

- Formatting and linting: `ruff` is the source of truth.
- Typing: `mypy --strict` must stay green for the core package.
- Tests: every behavior change needs tests, especially adapters, detector metrics, replay logic, and rule evaluation.
- Docs: keep README, changelog, and CLI help aligned with the real surface area.
- Compatibility: avoid Windows-only paths or shell examples in user-facing docs.

## PR Process

1. Open a focused PR. One behavior change beats five loosely related edits.
2. Include tests and any rules or fixtures needed to explain the change.
3. Update `README.md`, `CHANGELOG.md`, or `docs/` when the public surface changes.
4. Run `pytest`, `ruff`, `mypy`, and `python -m build --no-isolation` before asking for review.
5. Call out tradeoffs, backwards-compatibility risks, and any follow-up work in the PR description.
