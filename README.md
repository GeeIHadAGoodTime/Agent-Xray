# agent-xray

**See what your agent saw.**

[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-181717?logo=github)](https://github.com/GeeIHadAGoodTime/Agent-Xray/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/agent-xray)](https://pypi.org/project/agent-xray/)
[![Python](https://img.shields.io/pypi/pyversions/agent-xray)](https://pypi.org/project/agent-xray/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

`agent-xray` is a local-first debugger for AI agent decisions. It reconstructs the exact decision surface the model saw at each step: prompt context, available tools, conversation history, page state, context pressure, and corrective signals.

When a run goes sideways, most tracing tools tell you what happened. `agent-xray` tells you what the model had in front of it when it made the choice.

## Why agent-xray?

- Local-first: inspect traces on your machine, in CI, or offline.
- Framework-agnostic: normalize JSONL, OpenAI, LangChain, Anthropic, CrewAI, and OTel traces into one schema.
- Decision-surface replay: rebuild prompt, tool, reasoning, and browser context step by step.
- Practical triage: grade runs, classify likely root causes, capture golden fixtures, and compare regressions over time.
- Typed and scriptable: use it as a CLI, import it as a library, or wire it into pytest.

| Feature | agent-xray |
| --- | --- |
| Runs locally | Yes |
| Works offline | Yes |
| Account required | No |
| Framework-agnostic | Yes |
| Rule-based grading | Yes |
| Root-cause heuristics | Yes |
| Enforce mode | Yes |
| Golden replay | Yes |
| Optional TUI | Yes |

> Use your tracing stack to collect runs. Use `agent-xray` to understand why a specific decision failed.

## How agent-xray Compares

| Feature | agent-xray | LangSmith | Langfuse | Arize Phoenix | Braintrust | AgentOps |
|---------|-----------|-----------|----------|---------------|------------|----------|
| Local-first | Yes | No | Self-host option | Self-host option | No | No |
| Fully offline | Yes | No | No | Partial | No | No |
| Open source | MIT | Proprietary | MIT (server) | Apache 2.0 | Proprietary | Proprietary |
| No account needed | Yes | No | No | No | No | No |
| Zero dependencies | Yes | Many | Many | Many | Many | Many |
| Framework agnostic | Yes | LangChain-first | LangChain-first | Yes | Yes | Yes |
| Decision surface replay | Yes | No | No | No | No | No |
| Pluggable signal detection | Yes | No | No | No | Custom evals | No |
| Root-cause classification | Yes | No | No | No | No | No |
| Enforce mode (agent referee) | Yes | No | No | No | No | No |
| Golden fixture regression | Yes | Dataset comparison | No | No | Dataset comparison | No |
| Interactive TUI | Yes | Web UI | Web UI | Web UI | Web UI | Web UI |
| Model A/B comparison | Yes | Experiments | No | No | Experiments | No |
| pytest plugin | Yes | No | No | No | No | No |
| Cost tracking | Per-decision | Per-trace | Per-trace | Per-trace | Per-trace | Per-session |
| Pricing | Free forever | Free tier + paid | Free tier + paid | Free tier + paid | Free tier + paid | Free tier + paid |

agent-xray is complementary, not competing. Use LangSmith or Langfuse to collect production traces at scale. Use agent-xray to deeply debug why a specific agent decision went wrong -- locally, offline, with no account required. Think of it as the pytest to their Sauce Labs.

## Install

```bash
pip install agent-xray
```

Optional extras:

```bash
pip install "agent-xray[all]"
```

`[all]` pulls in the optional runner, TUI, OTel adapter, lint, typecheck, and test dependencies.

## Documentation

- [Integration guide](docs/integration.md) -- connect your agent in 5 minutes
- [Tutorial](docs/tutorial.md) -- instrument a simple agent and analyze the output
- [Architecture overview](docs/architecture.md)
- [Custom rules guide](docs/custom-rules.md)
- [Contribution guide](CONTRIBUTING.md)

## Quick Start

```bash
# Create a demo trace directory with bundled sample data
agent-xray quickstart

# Analyze a directory of traces
agent-xray analyze ./traces

# Reconstruct the full decision surface for one task
agent-xray surface task-001 --log-dir ./traces

# Grade tasks with a bundled ruleset
agent-xray grade ./traces --rules browser_flow

# Compare two task runs step-by-step
agent-xray diff task-a task-b --log-dir ./traces

# Run the full flywheel: grading, root causes, fixture replay, baseline deltas
agent-xray flywheel ./traces --fixture-dir ./captured --baseline ./baseline.json

# Compare two model runs across matched tasks
agent-xray compare ./runs-gpt4 ./runs-gpt5

# Open the interactive inspector
agent-xray tui ./traces
```

Set `AGENT_XRAY_LOG_DIR` if you want task-centric commands like `surface`, `reasoning`, `tree`, `capture`, and `replay` to default to one shared trace directory.

## Bundled Example

If you want a zero-setup demo, run:

```bash
agent-xray quickstart
```

If you want to see the native trace format directly, paste this inline example into `./traces/demo.jsonl`:

```bash
mkdir -p traces
cat > traces/demo.jsonl <<'JSONL'
{"task_id":"demo-checkout","step":1,"tool_name":"browser_navigate","tool_input":{"url":"https://demo-shop.example.test"},"tool_result":"Homepage loaded.","timestamp":"2026-03-27T12:00:00Z","duration_ms":900,"user_text":"Buy the blue mug on demo-shop.example.test and stop once checkout is visible.","task_category":"commerce","model_name":"gpt-5-mini","temperature":0.0,"tool_choice":"auto","tools_available":["browser_navigate","browser_click","browser_fill_ref","browser_snapshot"],"message_count":1,"llm_reasoning":"Open the storefront first.","page_url":"https://demo-shop.example.test/"}
{"task_id":"demo-checkout","step":2,"tool_name":"browser_click","tool_input":{"ref":"product-blue-mug"},"tool_result":"Product page opened.","timestamp":"2026-03-27T12:00:04Z","duration_ms":420,"model_name":"gpt-5-mini","temperature":0.0,"tool_choice":"auto","tools_available":["browser_click","browser_fill_ref","browser_snapshot"],"message_count":2,"llm_reasoning":"Open the mug detail page.","page_url":"https://demo-shop.example.test/products/blue-mug"}
{"task_id":"demo-checkout","step":3,"tool_name":"browser_fill_ref","tool_input":{"ref":"shipping-form","fields":["email","zip"],"text":"alex@example.test 60601"},"tool_result":"Shipping details accepted.","timestamp":"2026-03-27T12:00:09Z","duration_ms":610,"model_name":"gpt-5-mini","temperature":0.0,"tool_choice":"auto","tools_available":["browser_click","browser_fill_ref","browser_snapshot"],"message_count":3,"llm_reasoning":"Provide the minimum details required to continue.","page_url":"https://demo-shop.example.test/checkout"}
{"task_id":"demo-checkout","step":4,"tool_name":"browser_snapshot","tool_input":{},"tool_result":"Checkout page is visible with the order summary.","timestamp":"2026-03-27T12:00:12Z","duration_ms":250,"model_name":"gpt-5-mini","temperature":0.0,"tool_choice":"auto","tools_available":["browser_click","browser_snapshot"],"message_count":4,"llm_reasoning":"Verify that checkout is now visible.","page_url":"https://demo-shop.example.test/checkout"}
{"event":"task_complete","task_id":"demo-checkout","status":"success","final_answer":"Checkout page is open for the blue mug.","total_steps":4,"total_duration_s":2.18,"timestamp":"2026-03-27T12:00:12Z"}
JSONL

agent-xray analyze ./traces
agent-xray surface demo-checkout --log-dir ./traces
agent-xray grade ./traces
```

Expected summaries:

```text
Analyzed 1 task(s) with rules=default
{'GOLDEN': 0, 'GOOD': 1, 'OK': 0, 'WEAK': 0, 'BROKEN': 0}
```

```text
GRADE SUMMARY
Tasks: 1
Rules: default

  GOLDEN: 0
  GOOD: 1
  OK: 0
  WEAK: 0
  BROKEN: 0
```

The [tutorial](docs/tutorial.md) includes the full expected `surface` output and explains how to read it.

## What Is A Decision Surface?

At each step, an agent acts with a specific view of the world:

- A system prompt or prompt hash
- A conversation history, including compression side effects
- A tool set that may change step to step
- Model metadata like token usage, tool choice, and context pressure
- Browser cues like URL, screenshots, and page transitions
- Corrections, interventions, or retries from previous failures

`agent-xray` reconstructs that surface so you can inspect the exact conditions behind a tool choice instead of guessing from the final trace alone.

## Supported Frameworks

| Framework | Format Flag | Status |
| --- | --- | --- |
| Generic JSONL | `--format generic` | Stable |
| OpenAI Agents / Responses-style traces | `--format openai` | Stable |
| Anthropic Messages traces | `--format anthropic` | Stable |
| LangChain / LangGraph traces | `--format langchain` | Stable |
| CrewAI traces | `--format crewai` | Stable |
| OpenTelemetry GenAI spans | `--format otel` | Experimental |
| Auto-detect | `--format auto` | Stable |

## Integration

The fastest way to start tracing your agent is to write one JSON line per tool call into a `traces/` directory:

```python
import json, time
from pathlib import Path

trace = Path("traces/run.jsonl")
trace.parent.mkdir(exist_ok=True)

# In your agent loop, after each tool call:
with trace.open("a") as f:
    f.write(json.dumps({
        "task_id": "my-task",
        "step": step,
        "tool_name": tool_name,
        "tool_input": tool_args,
        "tool_result": result_text,
        "model_name": "gpt-4.1-mini",
    }) + "\n")
```

Then analyze:

```bash
agent-xray analyze ./traces
agent-xray surface my-task --log-dir ./traces
```

Working examples for each framework are in [`examples/`](examples/):

| Framework | Example |
| --- | --- |
| Anthropic SDK | [`instrument_anthropic.py`](examples/instrument_anthropic.py) |
| OpenAI SDK | [`instrument_openai.py`](examples/instrument_openai.py) |
| MCP (Playwright, etc.) | [`instrument_mcp.py`](examples/instrument_mcp.py) |
| LangChain | [`instrument_langchain.py`](examples/instrument_langchain.py) |

See the full [integration guide](docs/integration.md) for the JSONL format reference, field descriptions, and common pitfalls.

## Grading System

Every task is analyzed once, then scored by a JSON ruleset. Bundled rules include:

- `default`: general reliability and loop-detection signals
- `browser_flow`: browser and commerce progression signals
- `coding_agent`: file-edit, test, lint, and shell behavior
- `research_agent`: browsing and evidence-oriented signals

Rule files support both legacy metric syntax and the newer field/operator form:

```json
{
  "name": "browser_flow",
  "signals": [
    {
      "label": "high_error_rate",
      "field": "error_rate",
      "op": "gte",
      "value": 0.5,
      "points": -3,
      "reason": "error rate is too high"
    },
    {
      "label": "good_tool_diversity",
      "field": "unique_tools",
      "op": "gte",
      "value": 3,
      "points": 2,
      "reason": "used multiple tools"
    }
  ],
  "grade_thresholds": {
    "GOLDEN": 8,
    "GOOD": 5,
    "OK": 2,
    "WEAK": 0
  }
}
```

## Enforce Mode

Enforce mode turns `agent-xray` into an **advisor** for AI agents making code changes. It enforces one-change-at-a-time discipline, runs A/B testing against baseline, detects gaming patterns, and recommends commit-or-revert on every iteration — with evidence.

### Why enforce mode?

AI agents left to fix failing tests will often game them: weaken assertions, insert hardcoded values, swallow exceptions, or add special-case branches. Enforce mode catches this automatically and reverts bad changes before they accumulate.

### Quick start

```bash
# Initialize a session (captures baseline test results)
agent-xray enforce init --test-cmd "pytest tests/ -q"

# Agent makes a change, then checks in
agent-xray enforce check

# Adversarial audit of all iterations so far
agent-xray enforce challenge

# View current session status
agent-xray enforce status

# Generate a full report (text, json, or markdown)
agent-xray enforce report --format markdown

# Reset and start over
agent-xray enforce reset
```

### The enforce loop

Each `enforce check` iteration:

1. Captures the git diff and parses it into `DiffHunk`s
2. Runs the test suite and compares results against baseline
3. Detects gaming signals (8 heuristic detectors)
4. Classifies change quality (productive, refactor, test-only, mixed, suspicious)
5. Flags changes that exceed size limits (max files and diff lines per change)
6. Checks project-specific rules if a rules file is configured
7. Surfaces likely root causes for any regressions with evidence
8. Reports a decision recommendation: **COMMIT** (clear improvement), **REVERT** (regression or gaming detected), or **REJECTED** (size/rule violation)

The decision is a recommendation with evidence. The calling agent or user decides whether to act on it.

### Gaming detection

Eight heuristic detectors catch common gaming patterns:

| Detector | What it catches |
| --- | --- |
| `detect_test_file_modification` | Agent edits test files to make them pass |
| `detect_hardcoded_values` | Hardcoded return values that bypass real logic |
| `detect_special_case_branching` | `if test_mode` or `if x == expected_value` branches |
| `detect_mock_insertion` | Mock objects added to bypass real dependencies |
| `detect_assertion_weakening` | Replacing strict assertions with permissive ones |
| `detect_exception_swallowing` | Broad `except: pass` blocks hiding failures |
| `detect_early_return` | Short-circuit returns before real logic executes |
| `detect_import_removal` | Removing imports to silence import errors |

Combined confidence: **VALID** (<0.3), **SUSPICIOUS** (0.3-0.6), **GAMING** (>0.6). A GAMING verdict recommends revert with specific evidence of which detectors fired.

### Adversarial challenges

`enforce challenge` runs 9 cross-iteration checks:

- **Flip-flop detection**: same file toggling between states
- **Dependency risk**: changes concentrated in high-coupling files
- **Coverage gap**: improvements that don't touch test files
- **Cumulative gaming**: gaming signals aggregated across the full diff
- **Assertion erosion**: weakening assertions across iterations
- **Diminishing returns**: progress plateaus (3+ trailing zero-improvement iterations)
- **Persistent failures**: same tests keep failing across iterations
- **Scope creep**: later iterations more wasteful than earlier ones

### Hypothesis tracking

```bash
# Plan a hypothesis before making changes
agent-xray enforce plan --hypothesis "timeout is caused by missing await"

# Guard: verify the change matches the hypothesis
agent-xray enforce guard
```

With `--require-hypothesis`, every `enforce check` requires a registered hypothesis. After the change, agent-xray compares predicted test improvements against actual results and scores prediction accuracy.

### Auto mode

```bash
# Run the full loop with an AI agent command
agent-xray enforce auto --agent-cmd "codex exec '{hypothesis}'" --max-iter 20
```

Template variables available in `--agent-cmd`: `{failing_tests}`, `{fail_count}`, `{pass_count}`, `{total_count}`, `{iteration}`, `{last_error}`, `{hypothesis}`.

### Session grading

Every session gets an A-F grade based on:
- Reverted iterations (-5 each)
- Gaming detections (-15 each)
- Wasted iterations with no improvement (-3 each)
- Good commits (+2 each)
- Net regression from baseline (-10)
- All tests passing at end (+10)

### Library usage

```python
from agent_xray import (
    EnforceConfig, enforce_init, enforce_check, enforce_challenge,
    build_enforce_report, format_enforce_text
)

config = EnforceConfig(
    test_command="pytest tests/ -q",
    max_files_per_change=5,
    max_diff_lines=200,
)

session = enforce_init(config)
# ... agent makes changes ...
result = enforce_check(session)
print(result.decision)  # COMMIT, REVERT, or REJECTED (recommendation with evidence)

report = build_enforce_report(session)
print(format_enforce_text(report))
```

## Root Cause Classification

When a task grades poorly, `agent-xray` classifies the most likely failure mode using a 19-category cascade classifier:

| Root Cause | What it means |
| --- | --- |
| `routing_bug` | The task never got the right tool exposure |
| `approval_block` | Policy or permission gates blocked progress |
| `delegation_failure` | Multi-agent workflow failed at delegation boundary |
| `test_failure_loop` | Agent kept rerunning failing tests without changing approach |
| `spin` | The same action repeated without forward movement |
| `environment_drift` | The target environment changed underneath the agent |
| `tool_bug` | A tool call failed or returned unusable output |
| `insufficient_sources` | Research task answered before gathering enough evidence |
| `valid_alternative_path` | Goal achieved through an unexpected but valid path |
| `consultative_success` | Well-reasoned consultative answer without browser interaction |
| `tool_selection_bug` | The right tool existed, but the model chose poorly |
| `early_abort` | The run ended before enough work was attempted |
| `stuck_loop` | The run kept acting without meaningful state progress |
| `memory_overload` | Context pressure degraded the agent's later reasoning |
| `reasoning_bug` | Strategy was wrong even with adequate tools |
| `tool_rejection_mismatch` | A needed tool was actively rejected by policy |
| `prompt_bug` | Instructions likely nudged the model toward failure |
| `model_limit` | The task appears to exceed model capability |
| `unclassified` | No specific root cause identified from available evidence |

The classifier also detects **soft errors** — logical failures in tool results (e.g., "NOT ON A PAYMENT PAGE") even when the tool reported success.

## Library Usage

```python
from agent_xray import AgentStep, AgentTask, surface_for_task, classify_task, grade_task, load_rules

records = [
    {"task_id": "task-1", "step": 1, "tool_name": "browser_navigate", "tool_input": {"url": "https://example.test"}},
    {"task_id": "task-1", "step": 2, "tool_name": "browser_snapshot", "tool_input": {}, "tool_result": "checkout"},
]

steps = [AgentStep.from_dict(record) for record in records]
task = AgentTask.from_steps(steps, task_text="Inspect the checkout flow")

grade = grade_task(task, load_rules("browser_flow"))
surface = surface_for_task(task)

print(grade.grade, grade.score)
print(surface["steps"][0]["tools_available_names"])

if grade.grade in {"WEAK", "BROKEN"}:
    cause = classify_task(task, grade)
    print(cause.root_cause, cause.evidence)
```

## Architecture

```text
JSONL / framework trace files
  -> adapters/            normalize source formats into AgentStep
  -> schema.py            AgentStep / AgentTask / typed contexts
  -> signals/             detector packs for commerce, coding, and research
  -> analyzer.py          task metrics, cost tracking, soft-error detection
  -> grader.py            configurable JSON rules
  -> surface.py           decision-surface replay, reasoning chain, tree, diff
  -> root_cause.py        19-category failure classifier with cascade ordering
  -> capture.py           sanitized golden fixtures
  -> replay.py            fixture-vs-run comparison
  -> flywheel.py          end-to-end quality loop and baseline comparison
  -> comparison.py        model-vs-model divergence analysis
  -> enforce.py           controlled experiment loop (session, check, auto)
  -> enforce_audit.py     gaming detection and adversarial challenges
  -> enforce_report.py    enforce session reports (text, JSON, markdown)
```

## Badge Placeholders

If you are forking `agent-xray` or copying the README structure into another project, replace `<owner>` and `<repo>` in the following badge templates:

```md
[![PyPI](https://img.shields.io/pypi/v/agent-xray)](https://pypi.org/project/agent-xray/)
[![CI](https://img.shields.io/github/actions/workflow/status/<owner>/<repo>/ci.yml?branch=main)](https://github.com/<owner>/<repo>/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/codecov/c/github/<owner>/<repo>)](https://codecov.io/gh/<owner>/<repo>)
```

## Contributing

`agent-xray` is designed to be extended. Add a detector, adapter, rule set, or CLI improvement and keep the behavior observable.

See [CONTRIBUTING.md](CONTRIBUTING.md) for:

- local setup
- your first signal detector
- writing a new adapter
- adding a rule set
- style, test, and PR expectations

## License

MIT
