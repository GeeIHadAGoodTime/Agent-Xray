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

Task-bank-aware grading is also supported when you have curated task expectations:

```bash
agent-xray grade ./traces --task-bank ./task_bank.json
agent-xray analyze ./traces --task-bank ./task_bank.json
agent-xray root-cause ./traces --task-bank ./task_bank.json
```

`contrib/task_bank.py` adds fuzzy task matching plus per-task success criteria such as `must_reach_url`, `must_fill_fields`, `must_use_tools`, `must_reach_checkout`, `must_not_fill_payment`, and `must_have_answer`.

## Full CLI Reference

### Filtering and Common Flags

Most commands accept cross-cutting flags for filtering, output format, and environment configuration. These apply broadly and are listed here once rather than repeated per command.

#### Task filters

| Flag | Applies to | Description |
|------|-----------|-------------|
| `--grade BROKEN,WEAK` | grade, root-cause, diagnose, report, search, tree | Filter tasks by grade (comma-separated) |
| `--site dominos.com` | grade, root-cause, diagnose, report, tree | Filter tasks by site name |
| `--outcome success` | grade, root-cause, diagnose, report, tree | Filter tasks by outcome status |
| `--since 2h` / `--since 1d` | grade, root-cause, diagnose, report, tree | Time-based filter with relative durations |
| `--pattern "agent-steps-*.jsonl"` | analyze, grade, surface, root-cause, diagnose, report, completeness, compare | Glob pattern to filter JSONL files |
| `--days N` | analyze, grade, surface, root-cause, diagnose, report, completeness | Include only N most recent days |

#### Grading and rules

| Flag | Applies to | Description |
|------|-----------|-------------|
| `--rules browser_flow` | grade, root-cause, diagnose, report, compare, flywheel, golden, watch, tree | Select domain-specific ruleset |
| `--task-bank task_bank.json` | analyze, grade, root-cause, diagnose, report, surface | Enable expectation-aware grading with success criteria |
| `--expected-rejection ask_user` | grade, root-cause, diagnose (repeatable) | Exclude intentional tool rejections from mismatch counts |

#### Output format

| Flag | Applies to | Description |
|------|-----------|-------------|
| `--json` | Most commands | Machine-readable JSON output |
| `--markdown` | report | Markdown-formatted report output |

#### Environment variables

| Variable | Applies to | Description |
|----------|-----------|-------------|
| `AGENT_XRAY_LOG_DIR` | All commands taking `log_dir` | Default log directory |
| `AGENT_XRAY_PROJECT_ROOT` | diagnose, validate-targets | Default project root |
| `AGENT_XRAY_PRICING` | pricing, report cost | Custom pricing JSON path |
| `NO_COLOR` | All commands | Disable ANSI color output |

### Core inspection commands

| Command | What it does | Example |
| --- | --- | --- |
| `agent-xray analyze <log-dir>` | Grades a trace set and returns a high-level distribution summary | `agent-xray analyze ./traces --rules browser_flow --task-bank ./task_bank.json` |
| `agent-xray surface <task-id> [log-dir]` | Reconstructs the full decision surface for one task | `agent-xray surface golden-task ./traces --json` |
| `agent-xray reasoning <task-id> [log-dir]` | Extracts the reasoning chain for a task | `agent-xray reasoning golden-task ./traces` |
| `agent-xray diff <task-a> <task-b> [log-dir]` | Compares two tasks step by step | `agent-xray diff task-a task-b ./traces --summary` |
| `agent-xray grade <log-dir>` | Produces per-task grades, reasons, and root-cause output | `agent-xray grade ./traces --rules coding_agent --json` |
| `agent-xray root-cause <log-dir>` | Classifies likely failure modes for weak or broken tasks | `agent-xray root-cause ./traces --rules browser_flow --json` |
| `agent-xray tree [log-dir]` | Groups tasks into a day/site/task tree and can enrich it with grades | `agent-xray tree ./traces --rules default` |
| `agent-xray search <text> [log-dir]` | Searches tasks by `user_text`, optionally filtering by grade | `agent-xray search "checkout" ./traces --grade BROKEN,WEAK` |
| `agent-xray compare <left-dir> <right-dir>` | Compares matched tasks across two run directories | `agent-xray compare ./runs/gpt-4.1 ./runs/gpt-5 --json` |
| `agent-xray tui <log-dir>` | Opens the optional interactive inspector | `agent-xray tui ./traces --task-id broken-task` |
| `agent-xray quickstart` | Creates a sample trace set and walkthrough | `agent-xray quickstart` |

### Fixture, replay, and flywheel commands

| Command | What it does | Example |
| --- | --- | --- |
| `agent-xray capture <task-id>` | Captures a sanitized golden fixture from a task | `agent-xray capture golden-task --log-dir ./traces --out ./fixtures/golden-task.json` |
| `agent-xray replay <fixture>` | Replays a captured fixture against current traces | `agent-xray replay ./fixtures/golden-task.json --log-dir ./traces` |
| `agent-xray flywheel <log-dir>` | Runs grading, root-cause analysis, fix-plan generation, fixture replay, and baseline comparison | `agent-xray flywheel ./traces --fixture-dir ./fixtures --baseline ./last-flywheel.json --json` |
| `agent-xray record -- <command...>` | Records JSON-emitting agent output from a subprocess into native trace files | `agent-xray record --output-dir ./traces -- python my_agent.py` |

### Reporting, monitoring, and diagnosis commands

| Command | What it does | Example |
| --- | --- | --- |
| `agent-xray report <log-dir> <type>` | Generates text, JSON, or Markdown reports | `agent-xray report ./traces actions --markdown` |
| `agent-xray watch <file>` | Live-tails a JSONL file and grades tasks as they complete | `agent-xray watch ./traces/agent-steps-20260328.jsonl --rules browser_flow` |
| `agent-xray completeness <log-dir>` | Checks trace completeness across observability dimensions | `agent-xray completeness ./traces --json` |
| `agent-xray diagnose <log-dir>` | Builds a prioritized fix plan from root causes and validates targets when `--project-root` is set | `agent-xray diagnose ./traces --project-root . --json` |
| `agent-xray validate-targets --project-root <dir>` | Validates fix-plan file-path targets on disk | `agent-xray validate-targets --project-root . --resolver novviola` |

### Rules, pricing, golden, baseline, and task-bank subcommands

| Command | What it does | Example |
| --- | --- | --- |
| `agent-xray rules list` | Lists built-in rulesets | `agent-xray rules list` |
| `agent-xray rules show <name>` | Prints a ruleset JSON | `agent-xray rules show browser_flow` |
| `agent-xray rules init` | Scaffolds a custom ruleset | `agent-xray rules init --base default > my_rules.json` |
| `agent-xray pricing list` | Lists all models in the pricing database | `agent-xray pricing list` |
| `agent-xray pricing show <model>` | Shows pricing for one model | `agent-xray pricing show gpt-4.1-mini` |
| `agent-xray pricing update` | Refreshes the local pricing cache from GitHub | `agent-xray pricing update` |
| `agent-xray pricing path` | Shows the active pricing source path | `agent-xray pricing path` |
| `agent-xray golden rank [log-dir]` | Ranks `GOLDEN`/`GOOD` runs by efficiency within each site | `agent-xray golden rank ./traces --optimize balanced` |
| `agent-xray golden best [log-dir]` | Shows the exemplar run for each site | `agent-xray golden best ./traces --optimize speed` |
| `agent-xray golden capture [log-dir]` | Captures the current exemplar fixture for a site | `agent-xray golden capture ./traces --site shop --out ./exemplars/shop.json` |
| `agent-xray golden compare [log-dir]` | Compares current exemplars to captured fixtures | `agent-xray golden compare ./traces --fixtures ./exemplars` |
| `agent-xray golden profiles` | Lists built-in optimization profiles | `agent-xray golden profiles` |
| `agent-xray baseline capture <task-id> [log-dir]` | Saves a baseline JSON for a task/site | `agent-xray baseline capture golden-task ./traces -o ./baselines/shop.json` |
| `agent-xray baseline generate <task-id> [log-dir]` | Prints the naked prompt for a task | `agent-xray baseline generate golden-task ./traces` |
| `agent-xray baseline list <dir>` | Lists saved baselines | `agent-xray baseline list ./baselines` |
| `agent-xray task-bank list <path>` | Lists task-bank entries | `agent-xray task-bank list ./task_bank.json` |
| `agent-xray task-bank show <path> <task-id>` | Shows one task-bank entry | `agent-xray task-bank show ./task_bank.json checkout-wireless-headset` |
| `agent-xray task-bank validate <path>` | Validates task-bank schema and criterion names | `agent-xray task-bank validate ./task_bank.json` |

### Enforce subcommands

| Command | What it does | Example |
| --- | --- | --- |
| `agent-xray enforce init` | Starts an enforcement session and captures a baseline | `agent-xray enforce init --test "pytest tests/ -q"` |
| `agent-xray enforce check` | Grades the current iteration and recommends commit, revert, or reject | `agent-xray enforce check --hypothesis "missing await"` |
| `agent-xray enforce diff` | Shows current diff size and rejection status | `agent-xray enforce diff` |
| `agent-xray enforce status` | Shows the active session state | `agent-xray enforce status` |
| `agent-xray enforce challenge` | Runs adversarial checks across iterations | `agent-xray enforce challenge` |
| `agent-xray enforce report` | Generates a text, JSON, or Markdown session report | `agent-xray enforce report --format markdown` |
| `agent-xray enforce reset` | Abandons the active session | `agent-xray enforce reset` |
| `agent-xray enforce auto` | Runs the full autonomous enforce loop | `agent-xray enforce auto --test "pytest tests/ -q" --agent-cmd "codex exec '{hypothesis}'"` |
| `agent-xray enforce plan` | Registers a hypothesis and expected tests before a change | `agent-xray enforce plan --hypothesis "timeout is caused by missing await"` |
| `agent-xray enforce guard` | Checks for changes that bypassed the enforce pipeline | `agent-xray enforce guard` |

## Signal Detector Packs

`agent-xray` ships six built-in detector packs. Their metrics are merged into `TaskAnalysis.metrics()` and can be used directly in rules:

| Detector | Class | What it measures |
| --- | --- | --- |
| Commerce | `CommerceDetector` | `reached_payment`, `reached_checkout`, `reached_cart`, fill counts, payment-field visibility, and suspiciously-short terminal runs |
| Coding | `CodingDetector` | file edits, test/build/lint/git/shell activity, error counts, `test_to_edit_ratio`, and unique files touched |
| Research | `ResearchDetector` | searches, reads, source diversity, citations, synthesis steps, and URL references |
| Planning | `PlanningDetector` | plans created, plan steps executed, plan revisions, and plan completion rate |
| Memory | `MemoryDetector` | memory store/recall activity, recall hit rate, RAG queries, unique keys, and context injections |
| Multi-agent | `MultiAgentDetector` | delegation count, unique agents, delegation success rate, and delegation depth |

### Custom signal plugins

Built-ins are loaded first, then `discover_detectors()` loads third-party detectors from the `agent_xray.signals` entry-point group.

```toml
[project.entry-points."agent_xray.signals"]
my_detector = "my_package.detectors:MyDetector"
```

```python
from agent_xray.schema import AgentStep, AgentTask


class MyDetector:
    name = "my_detector"

    def detect_step(self, step: AgentStep) -> dict[str, bool]:
        return {"is_special": "special" in step.tool_name.lower()}

    def summarize(self, task: AgentTask, step_signals: list[dict[str, bool]]) -> dict[str, int]:
        return {"special_steps": sum(1 for item in step_signals if item["is_special"])}
```

If your detector instance satisfies the `SignalDetector` protocol, `run_detection()` will include its summary under `analysis.signal_metrics["my_detector"]`.

## Task Bank System

`src/agent_xray/contrib/task_bank.py` adds a criterion-aware grading layer for curated task banks.

### What the module exposes

- `load_task_bank(path)`: loads a bare JSON array or `{"tasks": [...]}` payload
- `match_task_to_bank(task, bank, analysis=None, threshold=0.45)`: fuzzy-matches an `AgentTask` to the best bank entry using `SequenceMatcher` plus token overlap
- `evaluate_task_criteria(task, analysis, criteria)`: evaluates one bank entry's `success_criteria`
- `grade_with_task_bank(tasks, bank_path, rules=None)`: runs normal grading, appends task-bank reasons/signals, and caps `GOLDEN` to `GOOD` when critical criteria fail
- `validate_task_bank(path)` / `validate_task_bank_entries(entries)`: validates schema, duplicate IDs, criterion names, and criterion value types

### Supported criteria

`task_bank.py` currently evaluates 14 criterion names:

- `must_answer_contains`
- `answer_type`
- `must_reach_url`
- `must_fill_fields`
- `min_urls`
- `max_steps`
- `payment_fields_visible`
- `must_not_fill_payment`
- `must_reach_cart`
- `must_reach_checkout`
- `must_use_tools`
- `no_browser_needed`
- `must_have_answer`
- `min_tool_count`

### Task-bank schema

```json
{
  "tasks": [
    {
      "id": "checkout-wireless-headset",
      "user_text": "Buy the wireless headset and complete checkout on shop.example.test.",
      "site": "shop.example.test",
      "category": "commerce",
      "success_criteria": {
        "must_reach_checkout": true,
        "must_reach_url": "order/confirmation",
        "must_use_tools": ["browser_click", "browser_fill_ref"]
      }
    }
  ]
}
```

### CLI usage

```bash
agent-xray task-bank validate ./task_bank.json
agent-xray task-bank list ./task_bank.json
agent-xray task-bank show ./task_bank.json checkout-wireless-headset

agent-xray grade ./traces --task-bank ./task_bank.json
agent-xray analyze ./traces --task-bank ./task_bank.json
agent-xray root-cause ./traces --task-bank ./task_bank.json
```

The task bank is intentionally opt-in. It does not replace rules; it augments grading with task-specific success criteria.

## Reports, Watch Mode, and Completeness

### Core report types (14)

The `report` command exposes fourteen core trace reports:

| Report type | What it focuses on | Example |
| --- | --- | --- |
| `health` | overall grade distribution and task health | `agent-xray report ./traces health` |
| `golden` | high-performing `GOLDEN`/`GOOD` runs | `agent-xray report ./traces golden` |
| `broken` | weak and broken tasks plus likely why | `agent-xray report ./traces broken` |
| `tools` | tool effectiveness and error rates | `agent-xray report ./traces tools` |
| `flows` | task-flow detection across commerce, coding, research, and browser patterns | `agent-xray report ./traces flows` |
| `outcomes` | outcome/status distribution | `agent-xray report ./traces outcomes` |
| `actions` | prioritized action items | `agent-xray report ./traces actions` |
| `cost` | cost summaries by model and task | `agent-xray report ./traces cost --markdown` |
| `fixes` | grouped fix-plan output built from root causes | `agent-xray report ./traces fixes` |
| `compare` | day-over-day comparison (the CLI name is `compare`) | `agent-xray report ./traces compare --day1 20260326 --day2 20260327` |
| `coding` | coding-task activity and verification patterns | `agent-xray report ./traces coding` |
| `research` | research-task sourcing and synthesis patterns | `agent-xray report ./traces research` |
| `timeline` | time-bucketed task throughput and error patterns | `agent-xray report ./traces timeline --bucket 15m` |
| `spins` | spin sequences grouped by tool, site, and pattern | `agent-xray report ./traces spins` |

Every report supports text output by default, plus `--json` and `--markdown`.

### Baseline-derived report types

Two more report modes are built on top of `baseline.py`:

- `report overhead`: compares current runs to captured baselines with `measure_all_overhead()`
- `report prompt-impact`: groups runs by prompt hash with `group_by_prompt_hash()`

### Watch / live-tail mode

`watch.py` implements `watch_file()`, which incrementally tails a JSONL file, rebuilds completed tasks from accumulated steps, grades each task as soon as its `task_complete` event lands, and prints a running tally.

```bash
agent-xray watch ./traces/agent-steps-20260328.jsonl --rules browser_flow
agent-xray watch ./traces/agent-steps-20260328.jsonl --json
```

### Completeness checker

`check_completeness()` audits trace coverage across 14 dimensions:

- `outcome_records`
- `tool_schemas`
- `model_name`
- `cache_tokens`
- `final_answer`
- `system_prompt`
- `rejected_tools`
- `approval_path`
- `conversation_history`
- `step_durations`
- `system_context`
- `llm_reasoning`
- `step_data_loss`
- `target_validity`

The CLI command `agent-xray completeness` checks the trace-driven dimensions directly. The optional `target_validity` dimension is evaluated when `check_completeness(project_root=...)` is used programmatically.

## Diagnosis and Fix Plans

`diagnose.py` turns root-cause output into a prioritized fix queue.

- `build_fix_plan(results)`: groups failures by root cause and ranks them by impact and severity
- `format_fix_plan_text(plan)`: renders a CLI-friendly fix plan
- `validate_fix_targets(plan, project_root)`: flags stale path targets
- `list_all_targets(resolver)`: lists every known target returned by a resolver
- `register_target_resolver(name, resolver, make_default=False)`: registers a custom resolver
- `DefaultTargetResolver`: the built-in resolver that returns conceptual search targets rather than file paths

`contrib/novviola.py` is the project-specific example of a resolver plugin. It exposes:

- `NovviolaTargetResolver`: maps root causes to NOVVIOLA file paths
- `NovviolaVerifyCommands`: returns NOVVIOLA-specific verification commands
- `register()`: installs the NOVVIOLA resolver as the default active resolver

## Golden Ranking and Optimization Profiles

`golden.py` ranks high-quality runs by efficiency within each site.

- `rank_golden_runs(tasks, rules=None, optimize="balanced")`: returns per-site rankings
- `find_exemplars(tasks, rules=None, optimize="balanced")`: returns the best run per site
- `capture_exemplar(...)`: writes the current exemplar as a fixture with efficiency metadata
- `explain_efficiency_gap(exemplar_analysis, other_analysis)`: explains why one run is more efficient than another

Built-in optimization profiles:

| Profile | Weights |
| --- | --- |
| `balanced` | `steps=0.3`, `duration=0.3`, `cost=0.2`, `errors=0.2` |
| `cost` | `steps=0.1`, `duration=0.1`, `cost=0.7`, `errors=0.1` |
| `speed` | `steps=0.1`, `duration=0.7`, `cost=0.1`, `errors=0.1` |
| `steps` | `steps=0.7`, `duration=0.1`, `cost=0.1`, `errors=0.1` |

CLI examples:

```bash
agent-xray golden rank ./traces --optimize balanced
agent-xray golden best ./traces --optimize speed
agent-xray golden capture ./traces --site shop --out ./exemplars/shop.json
agent-xray golden compare ./traces --fixtures ./exemplars
agent-xray golden profiles
```

## Baseline and Overhead Measurement

`baseline.py` exposes the baseline workflow:

- `generate_naked_prompt(task)`: reconstructs a stripped-down imperative prompt from a task
- `build_baseline(task, analysis)`: captures steps, duration, tokens, cost, errors, milestones, tool sequence, and naked prompt
- `save_baseline()` / `load_baseline()` / `load_baselines()`: persist baselines as JSON
- `measure_overhead()` / `measure_all_overhead()`: compare current tasks to captured baselines
- `group_by_prompt_hash()`: group tasks by `system_prompt_hash`
- `format_overhead_report()` / `format_prompt_impact_report()`: render baseline-derived reports

CLI examples:

```bash
agent-xray baseline capture golden-task ./traces -o ./baselines/shop.json
agent-xray baseline generate golden-task ./traces
agent-xray baseline list ./baselines

agent-xray report ./traces overhead --baselines ./baselines
agent-xray report ./traces prompt-impact --baselines ./baselines
```

## Pricing Database

`pricing.py` provides bundled pricing, a local cache, alias resolution, and live refreshes.

- Source precedence: explicit `--pricing` path, `AGENT_XRAY_PRICING`, local cache, bundled `src/agent_xray/pricing.json`
- `load_pricing()`: loads the active pricing database
- `get_model_cost()`: computes USD cost from input, output, and cached tokens
- `update_pricing_cache()`: fetches the latest pricing JSON from GitHub
- `list_models()`: returns all known models
- `pricing_source()`: reports where pricing is being loaded from
- `format_model_pricing()`: pretty-prints one model entry

The CLI subcommand for `pricing_source()` is `agent-xray pricing path`.

## Auto-Instrumentation and Recording

`instrument/` contains four integration surfaces:

| SDK / integration | Entry point | What it does |
| --- | --- | --- |
| OpenAI Python SDK | `OpenAIInstrumentor` | monkey-patches chat completion calls and records tool calls |
| Anthropic Python SDK | `AnthropicInstrumentor` | monkey-patches `messages.create` and records tool-use blocks |
| LangChain / LangGraph | `XRayCallbackHandler` | callback handler that records tool start/end events |
| MCP clients | `XRayMCPProxy` | wraps `call_tool` / `list_tools` and logs every tool request |

Supporting pieces:

- `auto_instrument(output_dir, task_id=None)`: auto-detects and patches installed Anthropic/OpenAI SDKs
- `xray_trace(...)`: decorator-based Anthropic tracing helper
- `StepRecorder`: thread-safe native JSONL writer used by all instrumentors and by `agent-xray record`

## MCP Server Tools

`src/agent_xray/mcp_server.py` exposes 48 tools:

| MCP tool | What it returns |
| --- | --- |
| `enforce_init` | initializes an enforcement session and baseline |
| `enforce_check` | evaluates the current working tree against the active session |
| `enforce_diff` | previews the current diff and size rejection status |
| `enforce_plan` | registers a hypothesis and expected tests |
| `enforce_guard` | detects out-of-band changes |
| `enforce_status` | returns the current enforcement session state |
| `enforce_challenge` | runs adversarial review on unreviewed iterations |
| `enforce_reset` | abandons the active enforcement session |
| `enforce_report` | emits the full enforcement report in JSON, text, or Markdown |
| `analyze` | loads traces, analyzes tasks, and returns per-task analysis plus grade summary |
| `grade` | grades traces and returns distribution plus per-task grade details |
| `root_cause` | classifies failure causes and returns grouped distribution plus per-task results |
| `completeness` | reports trace completeness warnings and coverage scores |
| `surface_task` | extracts full detail for one task: tool chain, LLM I/O, state, outcome |
| `search_tasks` | finds tasks by user_text substring with optional grade filter |
| `diagnose` | produces structured diagnostics with tool gaps, approval blocks, fix plan |
| `compare_runs` | side-by-side diff of two trace directories |
| `report` | generates any of the 16 report types (health, broken, tools, spins, etc.) |
| `diff_tasks` | compares two tasks side by side: tool calls, timing, outcomes |
| `reasoning` | extracts the model's reasoning chain for a specific task |
| `tree` | shows day/site/task hierarchy with optional grade enrichment |
| `golden_rank` | ranks golden/good runs by efficiency within each site |
| `golden_compare` | compares current runs against golden fixtures for regressions |
| `task_bank_validate` | validates task bank schema and criterion names |
| `task_bank_list` | lists all task entries in a bank |
| `flywheel` | runs grading + root-cause + baseline comparison in one shot |
| `capture_task` | captures a task as a sanitized fixture for replay and regression |
| `pricing_show` | shows per-token pricing for a specific model |
| `replay` | compares a saved golden fixture against current traces (IMPROVED/REGRESSION/STABLE) |
| `validate_targets` | checks that fix-plan target file paths actually exist on disk |
| `rules_list` | lists all available built-in rulesets with names and descriptions |
| `rules_show` | shows a ruleset's full configuration (signals, thresholds, requirements) |
| `rules_init` | generates a scaffold for a custom ruleset based on an existing one |
| `baseline_capture` | captures a golden task's metrics as a baseline for overhead measurement |
| `baseline_list` | lists all saved baselines in a directory with their metrics |
| `golden_best` | finds the single best exemplar per site for baseline/fixture capture |
| `golden_profiles` | shows available optimization profiles with weight distributions |
| `pricing_list` | lists all known models with input/output/cached token pricing |
| `baseline_generate` | generates a naked prompt (system message only) for baseline comparison |
| `task_bank_show` | shows a single task bank entry by ID or prefix match |
| `format_detect` | auto-detects the trace format of a log file with confidence score |
| `triage` | **START HERE** — one-call investigation: grades, surfaces worst failure, returns fix plan |
| `gaming_audit` | run 9 gaming detectors on a diff (test-gaming, hardcoded values, mock injection) |
| `pricing_update` | fetch latest model pricing from GitHub and update local cache |
| `inspect_task` | comprehensive single-task report: grade + root cause + surface + reasoning in one call |
| `signal_detect` | run signal detectors on a single task, optionally filtering to one detector by name |
| `match_task` | fuzzy-match a task to the best task bank entry by user text, site, and category |
| `golden_capture` | capture a golden exemplar task for future comparison and efficiency benchmarking |

Run the MCP server over stdio with:

```bash
python -m agent_xray.mcp_server
```

## Programmatic Runners, Protocols, and Reports

Additional modules that are part of the public surface:

- `runner.py`: `TaskRunner` protocol plus `StaticRunner` and `GenericHTTPRunner` for programmatic task execution
- `protocols.py`: `ToolRegistry`, `PromptBuilder`, `StepAdapter`, `StaticToolRegistry`, `StaticPromptBuilder`, `coerce_step()`, and `coerce_steps()`
- `reports.py`: text, JSON, and Markdown renderers for every report type
- `watch.py`: live-tail grading helpers and formatted tally output
- `pricing.py`: pricing cache, alias resolution, and cost formatting

## Pytest Plugin

agent-xray ships a `pytest` plugin (registered via the `pytest11` entry point) that provides an `xray` fixture for grading agent traces inside test suites:

```python
def test_checkout_agent(xray):
    steps = [{"step": 1, "tool_name": "navigate", "tool_result": "ok"}, ...]
    report = xray.analyze(steps)
    assert report.grade in ("GOLDEN", "GOOD")
    assert report.error_rate < 0.1
```

The `XrayReport` returned by `xray.analyze()` includes `grade`, `score`, `reasons`, `root_cause`, `unique_tools`, and `error_rate`.

Pass `--xray-rules <path>` to use custom grading rules:

```bash
pytest --xray-rules my_rules.json tests/
```

## Evaluation Drift Detection

`flywheel.py` does more than aggregate grades. It also detects evaluator drift.

- `IntegrityLock`: records the expected and actual SHA-256 of a tracked file or module
- `_build_integrity_locks(...)`: captures hashes for the active rules file, any task-bank files, `agent_xray.grader`, and `agent_xray.replay`
- `check_integrity(locks)`: re-hashes those inputs before the classification/fix-plan phase
- `run_flywheel(..., task_bank_paths=[...])`: raises `EVALUATION_DRIFT` if evaluator inputs changed during the run

This prevents a long flywheel run from mixing grades produced under different rules or replay logic.

## Enforce Mode

Enforce mode turns `agent-xray` into an **advisor** for AI agents making code changes. It enforces one-change-at-a-time discipline, runs A/B testing against baseline, detects gaming patterns, and recommends commit-or-revert on every iteration — with evidence.

### Why enforce mode?

AI agents left to fix failing tests will often game them: weaken assertions, insert hardcoded values, swallow exceptions, or add special-case branches. Enforce mode catches this automatically and surfaces evidence so you can revert bad changes before they accumulate.

### Quick start

```bash
# Initialize a session (captures baseline test results)
agent-xray enforce init --test "pytest tests/ -q"

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
8. Reports a decision: **RECOMMEND_COMMIT** (clear improvement), **RECOMMEND_REVERT** (regression or gaming detected), or **REJECTED** (size/rule violation)

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
print(result.decision)  # RECOMMEND_COMMIT, RECOMMEND_REVERT, or REJECTED
print(result.review_summary)  # Socratic summary with evidence
print(result.recommended_action)  # "commit", "revert", "split", or "investigate"

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
  -> signals/             detector packs for commerce, coding, research, planning, memory, and multi-agent traces
  -> analyzer.py          task metrics, cost tracking, soft-error detection
  -> contrib/task_bank.py criterion-aware grading with fuzzy task matching
  -> grader.py            configurable JSON rules
  -> surface.py           decision-surface replay, reasoning chain, tree, diff
  -> root_cause.py        19-category failure classifier with cascade ordering
  -> capture.py           sanitized golden fixtures
  -> replay.py            fixture-vs-run comparison
  -> flywheel.py          end-to-end quality loop and baseline comparison
  -> comparison.py        model-vs-model divergence analysis
  -> reports.py           health, tools, flows, cost, fixes, timeline, and spin reports
  -> watch.py             live-tail grading as JSONL files grow
  -> pricing.py           pricing database, aliases, cache, and live refresh
  -> baseline.py          naked prompt generation and overhead measurement
  -> instrument/          OpenAI, Anthropic, LangChain, and MCP auto-instrumentation
  -> runner.py            TaskRunner protocol plus HTTP runner
  -> protocols.py         ToolRegistry / PromptBuilder / StepAdapter protocols
  -> mcp_server.py        28 MCP tools for enforce, analysis, and investigation
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
