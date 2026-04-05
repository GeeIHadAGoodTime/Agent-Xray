# agent-xray

**Reconstruct what your AI agent saw at each decision point — so you can debug behavioral failures that produce no error logs.**

[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-181717?logo=github)](https://github.com/GeeIHadAGoodTime/Agent-Xray/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/agent-xray)](https://pypi.org/project/agent-xray/)
[![Python](https://img.shields.io/pypi/pyversions/agent-xray)](https://pypi.org/project/agent-xray/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Why agent-xray?

LLM agents fail silently. There are no stack traces. No error logs. The agent just does the wrong thing, and you are left reading a raw trace trying to figure out why.

Existing observability tools show you **what happened** — traces, spans, latencies. agent-xray shows you **why** — by reconstructing the *decision surface* at each step: what was in the prompt, which tools were available, what the model was reasoning about, what the page looked like, and how context pressure was shaping choices.

agent-xray surfaces evidence and keeps agents on track. It does not make determinations for you.

- **Local-first**: inspect traces on your machine, in CI, or fully offline. No account, no telemetry, no network calls.
- **Framework-agnostic**: normalize traces from OpenAI, Anthropic, LangChain, CrewAI, OpenTelemetry, or plain JSONL into one schema.
- **Zero dependencies**: the core library has no required dependencies. Optional extras add framework-specific instrumentation.

> Use your tracing stack to collect runs. Use `agent-xray` to understand why a specific decision went wrong.

## The 5-Tool Workflow

Most investigations follow the same five steps. This is what 90% of users need:

```
triage  →  surface_task  →  grade  →  root_cause  →  inspect_task
```

| Step | Tool | What it does |
|------|------|-------------|
| 1 | **`triage`** | One call to see the full picture: structural grade distribution, the worst failure surfaced step-by-step, and a prioritized fix plan. Start here. |
| 2 | **`surface_task`** | Reconstructs the complete decision surface for a single task — prompt context, tool set, conversation history, browser state, model reasoning, corrections. This is the evidence you need to understand *why* the agent chose what it chose. |
| 3 | **`grade`** | Assigns structural execution grades to all tasks in a trace set. Useful as a triage signal to separate clean executions from problematic ones. |
| 4 | **`root_cause`** | Classifies likely failure modes (spin, tool bug, early abort, routing bug, etc.) for tasks that graded poorly. Points you toward the right subsystem to fix. |
| 5 | **`inspect_task`** | All-in-one deep dive on a single task: structural grade + root cause + full surface + reasoning chain in one call. Use this when you know which task to investigate. |

```bash
# The typical investigation
agent-xray triage ./traces
agent-xray surface task-123 --log-dir ./traces
agent-xray inspect task-123 ./traces
```

## What Grades Mean

> **Grades measure execution structure, not output correctness.** Default rules are generic starting points — create [custom rulesets](docs/custom-rules.md) that encode your product's definition of quality before running optimization campaigns. See [Adapt the Rules Before You Optimize](ONBOARDING.md#adapt-the-rules-before-you-optimize).

This distinction matters. Structural grades are triage signals, not verdicts:

| Grade | Meaning |
|-------|---------|
| **GOLDEN** | Structurally clean execution: good tool diversity, no loops, low error rate, task completed |
| **GOOD** | Minor structural issues but generally sound execution |
| **OK** | Some structural concerns worth investigating |
| **WEAK** | Significant structural problems detected |
| **BROKEN** | Structural failures: spinning, high error rate, tool bugs, early aborts |

Grades evaluate **how** the agent executed — tool diversity, loop resistance, error rate, completion signals. They do **not** evaluate whether the agent achieved the right outcome or produced correct output.

- A `GOLDEN` task could still have the wrong answer.
- A `BROKEN` task might have eventually succeeded through a messy path.
- Use `surface_task` to verify what actually happened at each decision point.
- Use `task_bank_validate` with curated success criteria to check correctness.

**Surface evidence, let the agent decide.**

## The Flywheel

The flywheel is the disciplined improvement loop that turns individual investigations into systematic gains:

```
triage
  → identify worst failure
    → surface the decision point
      → fix root cause
        → enforce_check (controlled experiment)
          → compare_runs (quantify impact across ALL tasks)
            → repeat
```

Each step produces evidence for the next. The enforce workflow provides controlled experiments with gaming detection — so you know whether the fix actually improved things or just papered over the failure. `compare_runs` quantifies before/after impact across your entire task set, not just the one you fixed.

```bash
# Run the full flywheel in one shot
agent-xray flywheel ./traces --fixture-dir ./fixtures --baseline ./last-flywheel.json

# Or step by step:
agent-xray triage ./traces                                    # Find the worst problem
agent-xray surface broken-task --log-dir ./traces              # See what the agent saw
# ... fix the root cause ...
agent-xray enforce init --test "pytest tests/ -q"              # Baseline before change
agent-xray enforce check                                       # Evaluate the fix
agent-xray compare ./traces-before ./traces-after              # Quantify across all tasks
```

The flywheel also detects **evaluator drift** — if your rules or replay logic change mid-run, it raises `EVALUATION_DRIFT` rather than mixing grades from different rule versions.

## Quick Start

```bash
pip install agent-xray
```

Generate a demo trace set and walkthrough:

```bash
agent-xray quickstart
```

Or paste this inline example into `./traces/demo.jsonl`:

```bash
mkdir -p traces
cat > traces/demo.jsonl <<'JSONL'
{"task_id":"demo-checkout","step":1,"tool_name":"browser_navigate","tool_input":{"url":"https://demo-shop.example.test"},"tool_result":"Homepage loaded.","timestamp":"2026-03-27T12:00:00Z","duration_ms":900,"user_text":"Buy the blue mug on demo-shop.example.test and stop once checkout is visible.","task_category":"commerce","model_name":"gpt-5-mini","temperature":0.0,"tool_choice":"auto","tools_available":["browser_navigate","browser_click","browser_fill_ref","browser_snapshot"],"message_count":1,"llm_reasoning":"Open the storefront first.","page_url":"https://demo-shop.example.test/"}
{"task_id":"demo-checkout","step":2,"tool_name":"browser_click","tool_input":{"ref":"product-blue-mug"},"tool_result":"Product page opened.","timestamp":"2026-03-27T12:00:04Z","duration_ms":420,"model_name":"gpt-5-mini","temperature":0.0,"tool_choice":"auto","tools_available":["browser_click","browser_fill_ref","browser_snapshot"],"message_count":2,"llm_reasoning":"Open the mug detail page.","page_url":"https://demo-shop.example.test/products/blue-mug"}
{"task_id":"demo-checkout","step":3,"tool_name":"browser_fill_ref","tool_input":{"ref":"shipping-form","fields":["email","zip"],"text":"alex@example.test 60601"},"tool_result":"Shipping details accepted.","timestamp":"2026-03-27T12:00:09Z","duration_ms":610,"model_name":"gpt-5-mini","temperature":0.0,"tool_choice":"auto","tools_available":["browser_click","browser_fill_ref","browser_snapshot"],"message_count":3,"llm_reasoning":"Provide the minimum details required to continue.","page_url":"https://demo-shop.example.test/checkout"}
{"task_id":"demo-checkout","step":4,"tool_name":"browser_snapshot","tool_input":{},"tool_result":"Checkout page is visible with the order summary.","timestamp":"2026-03-27T12:00:12Z","duration_ms":250,"model_name":"gpt-5-mini","temperature":0.0,"tool_choice":"auto","tools_available":["browser_click","browser_snapshot"],"message_count":4,"llm_reasoning":"Verify that checkout is now visible.","page_url":"https://demo-shop.example.test/checkout"}
{"event":"task_complete","task_id":"demo-checkout","status":"success","final_answer":"Checkout page is open for the blue mug.","total_steps":4,"total_duration_s":2.18,"timestamp":"2026-03-27T12:00:12Z"}
JSONL

# Start here — one call to see the full picture
agent-xray triage ./traces

# Reconstruct the decision surface for the task
agent-xray surface demo-checkout --log-dir ./traces

# Structural grade
agent-xray grade ./traces
```

The [tutorial](docs/tutorial.md) walks through the full expected output and explains how to read each section.

## Quality Gates in CI

Add structural quality assertions to your existing test suite — no dashboard, no account, no workflow change:

```python
def test_checkout_agent(xray):
    steps = run_my_agent("Buy the blue mug on demo-shop.example.test")
    report = xray.analyze(steps)
    assert report.grade in ("GOLDEN", "GOOD"), f"Agent graded {report.grade}: {report.reasons}"
    assert report.error_rate < 0.1
```

```bash
pytest --xray-rules my_rules.json tests/
```

Or add a quality gate to GitHub Actions:

```yaml
# .github/workflows/agent-quality.yml
name: Agent Quality
on: [push]
jobs:
  grade:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install agent-xray
      - run: python run_agent.py  # your agent writes traces to ./traces/
      - run: agent-xray triage ./traces
      - run: agent-xray grade ./traces --rules my_rules.json
```

The pytest plugin is registered automatically via the `pytest11` entry point — `pip install agent-xray` is all you need.

## Supported Frameworks

| Framework | Format Flag | Support Level |
| --- | --- | --- |
| Generic JSONL | `--format generic` | Full — native format, all features |
| OpenAI Agents / Responses-style traces | `--format openai` | Full |
| Anthropic Messages traces | `--format anthropic` | Full |
| LangChain / LangGraph traces | `--format langchain` | Good — callback-based, covers most workflows |
| OpenTelemetry GenAI spans | `--format otel` | Good — experimental, GenAI semantic conventions |
| CrewAI traces | `--format crewai` | Basic — fixture/replay support, limited live tracing |
| Auto-detect | `--format auto` | Stable |

**Planned**: deeper LangGraph state-graph support, AutoGPT adapter.

Use `agent-xray format-detect <file>` to check what format your traces are in and get a confidence score.

## Install

```bash
pip install agent-xray
```

Optional extras for specific integrations:

```bash
pip install "agent-xray[openai]"       # OpenAI SDK auto-instrumentation
pip install "agent-xray[anthropic]"    # Anthropic SDK auto-instrumentation
pip install "agent-xray[langchain]"    # LangChain callback handler
pip install "agent-xray[otel]"         # OpenTelemetry adapter
pip install "agent-xray[tui]"          # Interactive terminal inspector
pip install "agent-xray[mcp]"          # MCP server and proxy
pip install "agent-xray[all]"          # Everything
```

Zero required dependencies for the core library. Python 3.10+.

## Full Tool Reference

### CLI Quick Reference

```bash
agent-xray analyze ./traces
agent-xray capture task-123 ./traces
agent-xray compare ./traces-before ./traces-after
agent-xray completeness ./traces
agent-xray diagnose ./traces
agent-xray diff task-a task-b ./traces
agent-xray flywheel ./traces
agent-xray grade ./traces
agent-xray inspect task-123 ./traces
agent-xray quickstart
agent-xray reasoning task-123 ./traces
agent-xray record -- python your_agent.py
agent-xray replay ./fixture.json ./traces
agent-xray report ./traces
agent-xray root-cause ./traces
agent-xray search "checkout" ./traces
agent-xray signal-detect task-123 ./traces
agent-xray surface task-123 --log-dir ./traces
agent-xray tree ./traces
agent-xray triage ./traces
agent-xray tui ./traces
agent-xray validate-targets ./fix-plan.json
agent-xray watch ./traces/demo.jsonl
agent-xray enforce check
agent-xray rules list
agent-xray pricing list
agent-xray golden rank ./traces
agent-xray baseline capture ./traces
agent-xray task-bank validate ./task-bank.json
```

### Core 5 — The Investigation Workflow

| Tool | What it does |
|------|-------------|
| `triage` | One-call investigation: structural grade distribution, worst failure surfaced step-by-step, prioritized fix plan |
| `surface_task` | Reconstructs the full decision surface for one task: prompt, tools, reasoning, browser state, corrections |
| `grade` | Structural execution grades for all tasks with configurable JSON rulesets |
| `root_cause` | Classifies failure modes using a 22-category cascade classifier with evidence |
| `inspect_task` | All-in-one single-task deep dive: structural grade + root cause + surface + reasoning |

### Investigation (10 tools)

| Tool | What it does |
|------|-------------|
| `analyze` | High-level trace analysis with per-task metrics and grade summary |
| `reasoning` | Extracts the model's reasoning chain for a specific task |
| `search_tasks` | Finds tasks by user_text substring with optional grade filter |
| `diff_tasks` | Compares two tasks side by side: tool calls, timing, outcomes, evidence |
| `compare_runs` | Side-by-side comparison of two trace directories across matched tasks |
| `tree` | Day/site/task hierarchy with optional grade enrichment |
| `diagnose` | Structured diagnostics with tool gaps, approval blocks, and fix plan |
| `completeness` | Audits trace coverage across 14 observability dimensions |
| `signal_detect` | Runs domain signal detectors (commerce, coding, research, etc.) on a single task |
| `format_detect` | Auto-detects trace format of a log file with confidence score |

### Fixtures and Replay (4 tools)

| Tool | What it does |
|------|-------------|
| `capture_task` | Captures a sanitized golden fixture for replay and regression testing |
| `replay` | Compares saved golden fixture against current traces (IMPROVED/REGRESSION/STABLE) |
| `match_task` | Fuzzy-matches a task to the best task bank entry |
| `flywheel` | End-to-end quality loop: grading + root cause + baseline comparison + drift detection |

### Golden and Baseline (7 tools)

| Tool | What it does |
|------|-------------|
| `golden_rank` | Ranks golden/good runs by efficiency within each site |
| `golden_best` | Best exemplar per site for baseline/fixture capture |
| `golden_compare` | Compares current runs against golden fixtures for regressions |
| `golden_capture` | Captures a golden exemplar for future comparison |
| `golden_profiles` | Shows optimization profiles with weight distributions |
| `baseline_capture` | Captures task metrics as a baseline for overhead measurement |
| `baseline_list` | Lists saved baselines with their metrics |
| `baseline_generate` | Generates a naked prompt for baseline comparison |

### Enforce — Controlled Experiments (10 tools)

| Tool | What it does |
|------|-------------|
| `enforce_init` | Starts an enforcement session and captures baseline test results |
| `enforce_check` | Evaluates current changes: recommends commit, revert, or reject with evidence |
| `enforce_diff` | Previews current diff size and rejection status |
| `enforce_plan` | Registers a hypothesis and expected test outcomes before a change |
| `enforce_guard` | Detects out-of-band changes that bypassed the enforce pipeline |
| `enforce_status` | Returns the current session state |
| `enforce_quick` | Runs a single enforce init+check cycle in one call and returns the combined review payload |
| `enforce_challenge` | Adversarial review across iterations: flip-flops, gaming, scope creep |
| `enforce_reset` | Abandons the active session |
| `enforce_report` | Full session report in JSON, text, or Markdown |
| `preflight_diff` | Checks git diff against project guardrails before an enforce iteration |

### Rules, Pricing, Task Bank, and Reports (13 tools)

| Tool | What it does |
|------|-------------|
| `rules_list` | Lists built-in rulesets |
| `rules_show` | Shows a ruleset's signals, thresholds, and requirements |
| `rules_init` | Scaffolds a custom ruleset from an existing one |
| `pricing_list` | Lists all known models with token pricing |
| `pricing_show` | Per-token pricing for a specific model |
| `pricing_update` | Refreshes local pricing cache from GitHub |
| `task_bank_validate` | Validates task bank schema and criterion names |
| `task_bank_list` | Lists all task bank entries |
| `task_bank_show` | Shows a single task bank entry by ID or prefix |
| `report` | 16 report types: health, golden, broken, tools, flows, cost, spins, timeline, and more |
| `validate_targets` | Checks that fix-plan file-path targets exist on disk |
| `gaming_audit` | Runs 9 gaming detectors on a diff (test-gaming, hardcoded values, mock insertion) |

**Total: 37 MCP tools** available via `python -m agent_xray.mcp_server` or the `agent-xray-mcp` entrypoint.

## Structural Grading System

Structural grades are produced by configurable JSON rulesets. Bundled rulesets:

| Ruleset | Domain |
|---------|--------|
| `default` | General reliability and loop-detection signals |
| `simple` | Short 1-2 step tasks (timers, memory writes, single-query lookups) |
| `browser_flow` | Browser and commerce progression signals |
| `coding_agent` | File-edit, test, lint, and shell behavior |
| `research_agent` | Browsing and evidence-gathering signals |

Rules use a field/operator format:

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

Create custom rulesets:

```bash
agent-xray rules init --base default > my_rules.json
agent-xray grade ./traces --rules my_rules.json
```

### Root Cause Classification

When a task grades poorly (structural signal), `root_cause` classifies the likely failure mode using a 22-category cascade classifier. Each classification includes evidence from the decision surface:

| Root Cause | Evidence pattern |
| --- | --- |
| `spin` | Same action repeated without forward movement |
| `tool_bug` | Tool call failed or returned unusable output |
| `early_abort` | Run ended before enough work was attempted |
| `routing_bug` | Task never got the right tool exposure |
| `approval_block` | Policy or permission gates blocked progress |
| `stuck_loop` | Kept acting without meaningful state progress |
| `tool_selection_bug` | Right tool existed but model chose poorly |
| `reasoning_bug` | Strategy was wrong even with adequate tools |
| `test_failure_loop` | Kept rerunning failing tests without changing approach |
| `environment_drift` | Target environment changed underneath the agent |
| `memory_overload` | Context pressure degraded later reasoning |
| `delegation_failure` | Multi-agent workflow failed at delegation boundary |
| `insufficient_sources` | Research task answered before gathering enough evidence |
| `tool_rejection_mismatch` | Needed tool actively rejected by policy |
| `prompt_bug` | Instructions likely nudged the model toward failure |
| `model_limit` | Task appears to exceed model capability |
| `valid_alternative_path` | Goal achieved through an unexpected but valid path |
| `consultative_success` | Well-reasoned answer without browser interaction |
| `context_overflow` | Context-window pressure degraded later reasoning quality |
| `rate_limit_cascade` | Repeated rate-limit errors cascaded into failure |
| `timeout` | Task hit time or step limit before completing |
| `unclassified` | No specific root cause identified from available evidence |

The classifier also detects **soft errors** — logical failures in tool results (e.g., "NOT ON A PAYMENT PAGE") even when the tool reported technical success.

### Task Bank — Correctness Validation

Structural grades assess execution quality. For correctness, use the task bank system with curated success criteria:

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

14 supported criteria: `must_answer_contains`, `answer_type`, `must_reach_url`, `must_fill_fields`, `min_urls`, `max_steps`, `payment_fields_visible`, `must_not_fill_payment`, `must_reach_cart`, `must_reach_checkout`, `must_use_tools`, `no_browser_needed`, `must_have_answer`, `min_tool_count`.

```bash
agent-xray task-bank validate ./task_bank.json
agent-xray grade ./traces --task-bank ./task_bank.json
```

## Integration

### Minimal: write JSONL directly

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

### Auto-instrumentation

```python
from agent_xray.instrument import auto_instrument

# Patches installed OpenAI/Anthropic SDKs automatically
auto_instrument(output_dir="./traces", task_id="my-task")
```

| SDK / Integration | Instrumentor | How it works |
| --- | --- | --- |
| OpenAI Python SDK | `OpenAIInstrumentor` | Monkey-patches chat completion calls and records tool calls |
| Anthropic Python SDK | `AnthropicInstrumentor` | Monkey-patches `messages.create` and records tool-use blocks |
| LangChain / LangGraph | `XRayCallbackHandler` | Callback handler that records tool start/end events |
| MCP clients | `XRayMCPProxy` | Wraps `call_tool` / `list_tools` and logs every tool request |

Working examples for each framework are in [`examples/`](examples/):

| Framework | Example |
| --- | --- |
| Anthropic SDK | [`instrument_anthropic.py`](examples/instrument_anthropic.py) |
| OpenAI SDK | [`instrument_openai.py`](examples/instrument_openai.py) |
| MCP (Playwright, etc.) | [`instrument_mcp.py`](examples/instrument_mcp.py) |
| LangChain | [`instrument_langchain.py`](examples/instrument_langchain.py) |

See the full [integration guide](docs/integration.md) for the JSONL format reference, field descriptions, and common pitfalls.

### Record mode

Capture output from any JSON-emitting subprocess:

```bash
agent-xray record --output-dir ./traces -- python my_agent.py
```

### Pytest plugin

agent-xray ships a pytest plugin (registered via the `pytest11` entry point) that provides an `xray` fixture for asserting on structural execution quality inside test suites:

```python
def test_checkout_agent(xray):
    steps = [{"step": 1, "tool_name": "navigate", "tool_result": "ok"}, ...]
    report = xray.analyze(steps)
    assert report.grade in ("GOLDEN", "GOOD")
    assert report.error_rate < 0.1
```

```bash
pytest --xray-rules my_rules.json tests/
```

## Enforce Mode — Controlled Experiments

Enforce mode turns agent-xray into an **advisor** for AI agents making code changes. It enforces one-change-at-a-time discipline, runs A/B testing against baseline, detects gaming patterns, and surfaces evidence for commit-or-revert decisions.

### Why enforce mode?

AI agents left to fix failing tests will often game them: weaken assertions, insert hardcoded values, swallow exceptions, or add special-case branches. Enforce mode catches this automatically and surfaces the evidence so you can revert bad changes before they accumulate.

### The enforce loop

Each `enforce check` iteration:

1. Captures the git diff and parses it into hunks
2. Runs the test suite and compares results against baseline
3. Detects gaming signals (8 heuristic detectors)
4. Classifies change quality (productive, refactor, test-only, mixed, suspicious)
5. Flags changes that exceed size limits
6. Checks project-specific rules if configured
7. Surfaces likely root causes for any regressions with evidence
8. Reports a recommendation: **RECOMMEND_COMMIT**, **RECOMMEND_REVERT**, or **REJECTED** — always with evidence

The recommendation is not a verdict. The calling agent or user decides whether to act on it.

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

Combined confidence: **VALID** (<0.3), **SUSPICIOUS** (0.3-0.6), **GAMING** (>0.6).

### Adversarial challenges

`enforce challenge` runs 9 cross-iteration checks: flip-flop detection, dependency risk, coverage gaps, cumulative gaming, assertion erosion, diminishing returns, persistent failures, and scope creep.

```bash
agent-xray enforce init --test "pytest tests/ -q"
agent-xray enforce check --hypothesis "missing await"
agent-xray enforce challenge
agent-xray enforce report --format markdown
```

## Signal Detector Packs

Six built-in detector packs. Their metrics merge into `TaskAnalysis.metrics()` and can be referenced directly in rules:

| Detector | What it measures |
| --- | --- |
| **Commerce** | `reached_payment`, `reached_checkout`, `reached_cart`, fill counts, payment-field visibility |
| **Coding** | File edits, test/build/lint/git/shell activity, error counts, `test_to_edit_ratio` |
| **Research** | Searches, reads, source diversity, citations, synthesis steps, URL references |
| **Planning** | Plans created, steps executed, revisions, completion rate |
| **Memory** | Memory store/recall activity, recall hit rate, RAG queries, unique keys |
| **Multi-agent** | Delegation count, unique agents, delegation success rate, depth |

Concrete detector classes: `CommerceDetector`, `CodingDetector`, `ResearchDetector`, `PlanningDetector`, `MemoryDetector`, `MultiAgentDetector`.

### Custom signal plugins

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

## Library Usage

```python
from agent_xray import AgentStep, AgentTask, surface_for_task, classify_task, grade_task, load_rules

records = [
    {"task_id": "task-1", "step": 1, "tool_name": "browser_navigate", "tool_input": {"url": "https://example.test"}},
    {"task_id": "task-1", "step": 2, "tool_name": "browser_snapshot", "tool_input": {}, "tool_result": "checkout"},
]

steps = [AgentStep.from_dict(record) for record in records]
task = AgentTask.from_steps(steps, task_text="Inspect the checkout flow")

# Structural grade — a triage signal, not a verdict
grade = grade_task(task, load_rules("browser_flow"))

# Decision surface — the evidence
surface = surface_for_task(task)

print(grade.grade, grade.score)
print(surface["steps"][0]["tools_available_names"])

# Root cause — classified from evidence in the decision surface
if grade.grade in {"WEAK", "BROKEN"}:
    cause = classify_task(task, grade)
    print(cause.root_cause, cause.evidence)
```

## Architecture

```text
JSONL / framework trace files
  -> adapters/            normalize source formats into AgentStep
  -> schema.py            AgentStep / AgentTask / typed contexts
  -> signals/             detector packs (commerce, coding, research, planning, memory, multi-agent)
  -> analyzer.py          task metrics, cost tracking, soft-error detection
  -> surface.py           decision-surface reconstruction, reasoning chains, tree, diff
  -> grader.py            configurable JSON rulesets (structural execution grades)
  -> root_cause.py        22-category failure classifier with cascade ordering
  -> contrib/task_bank.py correctness validation with fuzzy task matching
  -> capture.py           sanitized golden fixtures
  -> replay.py            fixture-vs-run comparison
  -> flywheel.py          end-to-end quality loop, baseline comparison, drift detection
  -> enforce.py           controlled experiment loop (session, check, auto)
  -> enforce_audit.py     gaming detection and adversarial challenges
  -> enforce_report.py    enforce session reports (text, JSON, markdown)
  -> comparison.py        model-vs-model divergence analysis
  -> reports.py           16 report types (health, tools, flows, cost, spins, timeline, etc.)
  -> watch.py             live-tail grading as JSONL files grow
  -> pricing.py           pricing database, aliases, cache, and live refresh
  -> baseline.py          naked prompt generation and overhead measurement
  -> instrument/          OpenAI, Anthropic, LangChain, and MCP auto-instrumentation
  -> runner.py            TaskRunner protocol plus HTTP runner
  -> protocols.py         ToolRegistry / PromptBuilder / StepAdapter protocols
  -> mcp_server.py        37 MCP tools for investigation, enforce, and analysis
```

## How agent-xray Compares

| Feature | agent-xray | LangSmith | Langfuse | Arize Phoenix | Braintrust | AgentOps |
|---------|-----------|-----------|----------|---------------|------------|----------|
| Local-first | Yes | No | Self-host option | Self-host option | No | No |
| Fully offline | Yes | No | No | Partial | No | No |
| Open source | MIT | Proprietary | MIT (server) | Apache 2.0 | Proprietary | Proprietary |
| No account needed | Yes | No | No | No | No | No |
| Zero dependencies | Yes | Many | Many | Many | Many | Many |
| Framework agnostic | Yes | LangChain-first | LangChain-first | Yes | Yes | Yes |
| Decision surface reconstruction | Yes | No | No | No | No | No |
| Root-cause classification | Yes | No | No | No | No | No |
| Enforce mode (controlled experiments) | Yes | No | No | No | No | No |
| Golden fixture regression | Yes | Dataset comparison | No | No | Dataset comparison | No |
| Gaming detection | Yes | No | No | No | No | No |
| pytest plugin | Yes | No | No | No | No | No |
| Pricing | Free forever | Free tier + paid | Free tier + paid | Free tier + paid | Free tier + paid | Free tier + paid |

agent-xray is complementary, not competing. Use LangSmith or Langfuse to collect production traces at scale. Use agent-xray to deeply debug why a specific agent decision went wrong — locally, offline, with no account required. Think of it as the pytest to their Sauce Labs.

## Documentation

- [Integration guide](docs/integration.md) — connect your agent in 5 minutes
- [Tutorial](docs/tutorial.md) — instrument a simple agent and analyze the output
- [Architecture overview](docs/architecture.md)
- [Custom rules guide](docs/custom-rules.md)
- [Contribution guide](CONTRIBUTING.md)

## Contributing

`agent-xray` is designed to be extended. Add a detector, adapter, ruleset, or CLI improvement and keep the behavior observable.

See [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Local setup
- Your first signal detector
- Writing a new adapter
- Adding a ruleset
- Style, test, and PR expectations

## Public API Index

Public exports referenced by tests and available for library users include:

`StaticToolRegistry`, `StaticPromptBuilder`, `check_completeness`, `DefaultTargetResolver`, `build_fix_plan`, `format_fix_plan_text`, `list_all_targets`, `register_target_resolver`, `validate_fix_targets`, `load_task_bank`, `rank_golden_runs`, `find_exemplars`, `explain_efficiency_gap`, `capture_exemplar`, `generate_naked_prompt`, `build_baseline`, `measure_overhead`, `group_by_prompt_hash`, `suggest_baseline_capture`, `SignalDetector`, `discover_detectors`, `EnforceConfig`, `build_enforce_report`, `format_enforce_text`.

## License

MIT
