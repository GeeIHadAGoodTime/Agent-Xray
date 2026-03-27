# agent-xray

`agent-xray` turns raw agent step logs into something you can debug: grades, replay surfaces, root-cause buckets, fixture capture, and regression checks.

It was extracted from an internal step-log-analysis workflow and rebuilt as a standalone Python package with no hard dependency on any single agent stack.

## Quick Start

```powershell
cd J:\PROJECTS\agent-xray
python -m pip install -e .

agent-xray analyze .\logs --json
agent-xray grade .\logs --rules .\src\agent_xray\rules\browser_flow.json
agent-xray surface 51f3fd8d --log-dir .\logs
agent-xray reasoning 51f3fd8d --log-dir .\logs
agent-xray diff task-a task-b --log-dir .\logs --json
agent-xray tree --log-dir .\logs
agent-xray capture 51f3fd8d --log-dir .\logs --out .\captured\golden.json
agent-xray replay .\captured\golden.json --log-dir .\logs
```

Set `AGENT_XRAY_LOG_DIR` if you want `surface`, `reasoning`, `diff`, `tree`, `capture`, and `replay` to default to a shared log directory.

## Pitch

Most agent traces are only useful in the moment they were recorded. `agent-xray` makes them reusable:

- It defines a formal JSONL schema for portable step logs.
- It grades runs with configurable JSON rules instead of hardcoded repo logic.
- It reconstructs the information surface the model saw at each step.
- It classifies likely failure modes so fix work can be prioritized.
- It captures sanitized golden fixtures and checks new runs against them.

## Architecture

```text
JSONL step logs
  -> adapter.py        normalize external trace formats into AgentStep
  -> schema.py         AgentStep / AgentTask / TaskOutcome
  -> analyzer.py       URL, fill, error, spin, and site-level signals
  -> grader.py         configurable scoring via JSON rules
  -> surface.py        prompt/tool/context replay and run diffs
  -> root_cause.py     failure bucket classification
  -> capture.py        sanitized golden fixture generation
  -> replay.py         fixture-vs-run comparison
  -> diagnose.py       prioritized fix plan
  -> flywheel.py       end-to-end pass across grading + diagnosis + replay
```

## JSONL Schema

Core step shape:

```python
from dataclasses import dataclass

@dataclass
class AgentStep:
    task_id: str
    step: int
    tool_name: str
    tool_input: dict
    tool_result: str | None = None
    llm_reasoning: str | None = None
    error: str | None = None
    duration_ms: int | None = None
    timestamp: str | None = None
    model_name: str | None = None
    temperature: float | None = None
    tool_choice: str | None = None
    message_count: int | None = None
    tools_available: list[str] | None = None
    page_url: str | None = None
    system_prompt_hash: str | None = None
    context_usage_pct: float | None = None
    context_window: int | None = None
    compaction_count: int | None = None
    snapshot_compressed: bool | None = None
    had_screenshot: bool | None = None
    correction_messages: list[str] | None = None
    spin_intervention: str | None = None
```

`schema.py` also defines:

- `AgentTask`: a collection of steps plus metadata and optional task outcome
- `TaskOutcome`: final status, answer, duration, and metadata
- JSON Schema dictionaries for the three objects

## Grading

Rules live in JSON files. Two bundled rule sets ship with the package:

- `src/agent_xray/rules/default.json`: generic reliability and spin signals
- `src/agent_xray/rules/browser_flow.json`: browser and commerce progression signals

Rule files define:

- positive and negative signals
- metric comparisons like `gte`, `lte`, and `equals`
- point values
- grade thresholds
- optional `golden_requirements` to cap `GOLDEN` unless a terminal condition was visibly reached

Example custom rule:

```json
{
  "name": "custom",
  "signals": [
    {
      "name": "must_use_multiple_tools",
      "metric": "unique_tools",
      "gte": 2,
      "points": 2,
      "reason": "+2 used more than one tool"
    }
  ],
  "grade_thresholds": {
    "GOLDEN": 2,
    "GOOD": 1,
    "OK": 0,
    "WEAK": 0
  }
}
```

## Information Surfaces

The surface replay is built around the idea that a bad tool call is rarely just a bad tool call. The model acted under a specific surface:

- prompt text or prompt hash
- tools available at the step
- reasoning text before the action
- conversation history carried into the step
- context pressure signals like message count and compactions
- page URL and browser state
- injected corrections or spin interventions

`agent-xray surface <task_id>` prints those layers in order.

## Root Causes

The classifier is heuristic on purpose. It tries to separate operational failures from prompt failures:

- `routing_bug`
- `approval_block`
- `spin`
- `environment_drift`
- `tool_bug`
- `tool_selection_bug`
- `early_abort`
- `stuck_loop`
- `reasoning_bug`
- `prompt_bug`
- `model_limit`

Use it for triage, not for absolutes.

## CLI Reference

```text
agent-xray analyze <log_dir> [--days N] [--json]
agent-xray surface <task_id> [--log-dir DIR] [--json]
agent-xray reasoning <task_id> [--log-dir DIR] [--json]
agent-xray diff <task_id_1> <task_id_2> [--log-dir DIR] [--json]
agent-xray grade <log_dir> [--rules rules.json] [--json]
agent-xray tree [--log-dir DIR] [--json]
agent-xray capture <task_id> [--log-dir DIR] [--out fixture.json] [--json]
agent-xray replay <golden.json> [--log-dir DIR] [--json]
```

## Adapters

See `examples/adapters/` for converters from:

- OpenAI Agents SDK traces
- LangChain action traces
- Claude Code style step logs
- generic JSONL records

The package itself does not force one trace format. You adapt into `AgentStep` and everything else works on top.

## Pluggable Interfaces

`adapter.py` exposes:

- `ToolRegistry`: optional source of tool names and descriptions
- `PromptBuilder`: optional prompt reconstruction hook
- `StaticToolRegistry` and `StaticPromptBuilder`: minimal built-ins

`runner.py` exposes:

```python
class TaskRunner(Protocol):
    async def send(self, task_text: str) -> str: ...
    async def get_status(self, task_id: str) -> str: ...
```

and includes `GenericHTTPRunner`.

## Fixtures

Golden fixtures are JSON snapshots of successful runs with sensitive content sanitized:

- emails become `*email*`
- phone numbers become `*phone*`
- card numbers become `*card_number*`
- ZIPs become `*zip*`
- URLs become `https://shop.example.test/...`

See `examples/golden_run.json`.

## Contributing

1. Keep the log schema stable.
2. Prefer new rule files over hardcoded grading branches.
3. Add tests for any new metric, rule operator, or replay behavior.
4. Keep adapters as examples unless the source format is stable enough for core support.

## Development

```powershell
python -m compileall .\src\agent_xray
python -m pytest
```
