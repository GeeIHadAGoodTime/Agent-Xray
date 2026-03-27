# agent-xray Architecture

## Pipeline Overview

```text
traces
  -> adapters
  -> schema
  -> analyzer
  -> grader
  -> root_cause
  -> reports
```

Expanded view:

```text
JSONL trace files / framework exports
    |
    v
adapters/ (generic, OpenAI, Anthropic, LangChain, CrewAI, OTel)
    |
    v
schema.py (AgentStep, AgentTask, TaskOutcome, typed contexts)
    |
    +--> surface.py (decision-surface replay, reasoning, diff, tree)
    |
    v
analyzer.py (TaskAnalysis, derived metrics, detector execution)
    |
    +--> signals/ (commerce, coding, research, plugins)
    |
    v
grader.py (RuleSet, GradeResult, JSON rule evaluation)
    |
    v
root_cause.py (RootCauseResult, failure classification)
    |
    +--> diagnose.py (fix plans, target resolvers)
    |
    v
reports.py / cli.py (text and JSON summaries)
```

## How The Subsystems Connect

### 1. Traces and adapters

`agent-xray` starts with newline-delimited JSON files. If the file is already in the native `AgentStep` shape, `load_tasks()` can parse it directly. If it comes from another framework, `load_adapted_tasks()` calls `adapters.adapt()` and converts each source record into a normalized `AgentStep`.

The important design constraint is that every downstream subsystem only speaks in terms of `AgentStep`, `AgentTask`, and `TaskOutcome`. Framework-specific logic stays in `src/agent_xray/adapters/`.

### 2. Schema

`schema.py` defines the canonical in-memory model:

- `AgentStep`: one tool-use step
- `AgentTask`: one task and its ordered steps
- `TaskOutcome`: terminal task metadata
- `ModelContext`, `ToolContext`, `ReasoningContext`, `BrowserContext`: typed sub-contexts for step metadata

Everything else in the package assumes those types, which is why the adapters normalize aggressively and preserve unknown fields in `extensions`.

### 3. Analyzer

`analyzer.py` converts raw tasks into metrics:

- step count
- unique tools
- unique URLs
- error rate
- loop/spin signals
- site inference
- token and cost totals

The analyzer also runs detector packs from `signals/` and merges those detector metrics into the final `TaskAnalysis.metrics()` map. That flattened metrics map is the contract used by the grader.

### 4. Grader

`grader.py` evaluates a `TaskAnalysis` against a JSON `RuleSet`.

Each rule:

- resolves a metric path such as `error_rate` or `coding.test_runs`
- applies an operator such as `gte` or `equals`
- adds `points` on pass or `else_points` on failure

The final raw score is mapped onto `GOLDEN`, `GOOD`, `OK`, `WEAK`, or `BROKEN` using the ruleset thresholds. Optional `golden_requirements` can downgrade an otherwise `GOLDEN` run to `GOOD`.

### 5. Root cause classification

`root_cause.py` only classifies low-grade runs. It combines:

- grade result
- task analysis
- task structure and available tools

into a `RootCauseResult` such as `routing_bug`, `spin`, `tool_bug`, `prompt_bug`, or `model_limit`.

This stage is intentionally heuristic. It does not try to prove causality; it tries to narrow the first place a human should look.

### 6. Reports and diagnosis

`reports.py` turns tasks, grades, and analyses into terminal-friendly summaries such as:

- health
- golden
- broken
- tools
- flows
- outcomes
- actions
- coding
- research

`diagnose.py` sits one level lower than reports. It groups `RootCauseResult` values into ranked fix-plan items and can resolve those items into investigation targets through a pluggable target-resolver interface.

## Extension Points

### Signals

Signal detectors add domain-specific metrics without changing the analyzer core.

- Interface: `signals.SignalDetector`
- Discovery: built-ins from `signals/__init__.py` plus `agent_xray.signals` entry points
- Use when: you need metrics such as commerce milestones, coding behavior, or research evidence

Detector output becomes available to rules as nested fields like `commerce.reached_checkout` and `coding.test_runs`.

### Adapters

Adapters let `agent-xray` ingest foreign trace formats.

- Entry point: `adapters.adapt(path, format=...)`
- Registration: `FORMATS` in `src/agent_xray/adapters/__init__.py`
- Use when: your framework emits JSONL that is not already in native `AgentStep` form

The adapter boundary is the package's main isolation layer for framework-specific logic.

### Rules

Rules are pure JSON configuration.

- Location: `src/agent_xray/rules/*.json` or any external path
- Loader: `load_rules()`
- Runtime consumer: `grade_task()` and `grade_tasks()`

Because rules operate on flattened metric paths, you can add a new detector and then score it without changing the grader.

### Target resolvers

Target resolvers map a root cause plus evidence into concrete investigation targets.

- Protocol: `diagnose.TargetResolver`
- Registration: `register_target_resolver()`
- Lookup: `get_target_resolver()`
- Runtime consumer: `build_fix_plan(..., target_resolver=...)`

The built-in resolver uses `FIX_TARGETS` and `PROMPT_SECTION_TARGETS`, but teams can replace it with a resolver that points at their real prompt files, router configs, or service owners.

## Practical Mental Model

If you need one sentence for the whole package, use this:

`agent-xray` normalizes traces into one schema, turns that schema into metrics, scores those metrics with rules, classifies the likely failure mode, and then renders the result in surfaces, reports, and fix plans.
