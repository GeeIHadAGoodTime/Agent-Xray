# Contributing

## Development Setup

```bash
git clone https://github.com/agent-xray/agent-xray.git
cd agent-xray
python -m pip install -e ".[all]"
python -m pytest tests -q
```

Common local checks:

```bash
python -m ruff check src tests
python -m ruff format --check src tests
python -m mypy src/agent_xray --strict
python -m build --no-isolation
```

Useful habits while developing:

- keep fixtures tiny and purpose-built
- prefer `agent-xray quickstart` or small JSONL files over large production traces while iterating
- update docs when the public CLI, schema, or extension surface changes

## Project Shape

Core directories:

- `src/agent_xray/schema.py`: canonical types used everywhere else
- `src/agent_xray/adapters/`: framework-specific trace loaders
- `src/agent_xray/signals/`: detector packs that emit domain metrics
- `src/agent_xray/grader.py`: JSON rules evaluation
- `src/agent_xray/root_cause.py`: failure classification
- `src/agent_xray/reports.py`: text and JSON report builders
- `src/agent_xray/cli.py`: CLI wiring
- `tests/`: focused unit and integration coverage

Design rule: new framework- or domain-specific behavior should stay at the edges. Normalize first, then reuse the shared schema, analyzer, grader, and reporting pipeline.

## How To Add A New Adapter

Adapters convert a foreign trace format into `AgentStep` objects.

1. Create a new module under `src/agent_xray/adapters/`.

   Typical shape:

   ```python
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

2. Register the format key in `src/agent_xray/adapters/__init__.py`.

   Add it to `FORMATS` so `agent-xray --format <name>` can resolve the adapter.

3. Preserve as much structured metadata as possible.

   Important fields:

   - timestamps and durations
   - tool inputs and tool outputs
   - model metadata
   - browser context
   - reasoning or intervention metadata

4. Add tests.

   Minimum coverage:

   - adapter fixture file under `tests/fixtures/`
   - positive parse case in `tests/test_adapters.py`
   - edge-case behavior for malformed or partial records

Adapter rules:

- normalize into `AgentStep` instead of branching inside the analyzer
- skip malformed records quietly unless a warning materially helps the user
- keep fixtures small enough that failures are obvious from the diff

## How To Add A New Signal Detector

Signal detectors add task metrics without changing analyzer logic.

1. Create a new module under `src/agent_xray/signals/`.
2. Implement the `SignalDetector` protocol:

   ```python
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

       def summarize(
           self,
           task: AgentTask,
           step_signals: list[dict[str, bool]],
       ) -> dict[str, Any]:
           screenshot_steps = sum(1 for item in step_signals if item["used_screenshot"])
           return {
               "screenshot_steps": screenshot_steps,
               "recovered_after_visual_check": any(
                   step_signals[index]["used_screenshot"]
                   and step_signals[index + 1]["succeeded"]
                   for index in range(len(step_signals) - 1)
               ),
           }
   ```

3. Wire it into `discover_detectors()` in `src/agent_xray/signals/__init__.py`, or expose it through the `agent_xray.signals` entry-point group if it lives outside the repo.
4. Add tests for both `detect_step()` and `summarize()`.
5. Add or update a ruleset if the new metric should affect grades.

Detector guidance:

- keep step signals simple and boolean where possible
- emit flat, stable metric names from `summarize()`
- prefer domain metrics over prompt-specific heuristics when you can measure both

## How To Add A New Report Type

Reports are a CLI surface, so treat them as public API.

1. Add `report_<name>()` and `report_<name>_data()` to `src/agent_xray/reports.py`.

   Convention:

   - text function returns a terminal-friendly string
   - data function returns JSON-serializable data

2. Wire both into `cmd_report()` in `src/agent_xray/cli.py`.
3. Add the new report name to the CLI parser choices.
4. Add tests:

   - text and JSON coverage in `tests/test_reports.py`
   - CLI coverage in `tests/test_reports_cli.py`

5. Update `README.md` or `docs/` if the new report is user-facing.

A report is a good fit when:

- it summarizes an existing set of metrics
- it helps a human make a decision faster
- it does not need new normalization logic

If you need new raw metrics first, add those in a detector or the analyzer before adding the report.

## Rules And Root-Cause Extensions

You do not need to change Python code to add a ruleset.

- put a JSON file in `src/agent_xray/rules/` or anywhere on disk
- run it with `agent-xray grade ./traces --rules ./path/to/rules.json`
- document it in [`docs/custom-rules.md`](docs/custom-rules.md) if it becomes part of the public surface

If you need custom fix-plan targets, use the target-resolver extension point in `src/agent_xray/diagnose.py`:

- implement the `TargetResolver` protocol
- register it with `register_target_resolver()`
- pass it to `build_fix_plan(..., target_resolver=...)`

## Testing Expectations

Before opening a PR, run the checks relevant to your change:

```bash
python -m pytest tests -q
python -m ruff check src tests
python -m ruff format --check src tests
python -m mypy src/agent_xray --strict
```

If you changed packaging or install behavior, also run:

```bash
python -m build --no-isolation
```

## PR Review Checklist

Use this checklist before asking for review:

- the change is focused and has a clear reason to exist
- adapters still normalize into `AgentStep` instead of bypassing the schema
- new signals or reports have tests
- CLI changes update help text and docs
- public behavior changes update `README.md`, `docs/`, or both
- new rulesets include realistic thresholds and, when needed, `golden_requirements`
- no unrelated files were reformatted or rewritten
- `pytest`, `ruff`, and `mypy` pass locally for the touched area

## Style Notes

- Prefer ASCII in docs, fixtures, and source unless the file already uses Unicode intentionally.
- Keep terminal examples shell-agnostic when possible; avoid Windows-only commands in user-facing docs.
- Favor small explicit helpers over clever normalization logic.
- If a trace field is optional, preserve it when available rather than inventing fallback values.
