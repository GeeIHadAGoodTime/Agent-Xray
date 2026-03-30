# agent-xray: Comprehensive Capabilities Audit

Version: 1.16.0 | Audit date: 2026-03-29

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Complete Tool/Command Registry](#complete-toolcommand-registry)
3. [MCP vs CLI vs Internal-Only Matrix](#mcp-vs-cli-vs-internal-only-matrix)
4. [Deep Dive: ENFORCE Workflow](#deep-dive-enforce-workflow)
5. [Deep Dive: GRADING System](#deep-dive-grading-system)
6. [Signal Detectors (Plugin System)](#signal-detectors-plugin-system)
7. [Root Cause Classification](#root-cause-classification)
8. [Adapters (Format Support)](#adapters-format-support)
9. [Auto-Instrumentation](#auto-instrumentation)
10. [Report Types](#report-types)
11. [Baseline & Overhead Measurement](#baseline--overhead-measurement)
12. [Golden Ranking & Efficiency](#golden-ranking--efficiency)
13. [Task Bank](#task-bank)
14. [Flywheel (End-to-End Quality Loop)](#flywheel-end-to-end-quality-loop)
15. [Fixture Capture & Replay](#fixture-capture--replay)
16. [Pricing System](#pricing-system)
17. [Pytest Plugin](#pytest-plugin)
18. [Extension Points & Plugin Hooks](#extension-points--plugin-hooks)
19. [Dark Abilities (CLI-only, Not in MCP)](#dark-abilities-cli-only-not-in-mcp)
20. [Hidden/Underutilized Features](#hiddenunderutilized-features)
21. [Recommended Workflows](#recommended-workflows)

---

## Architecture Overview

agent-xray is a trace analyzer for AI agent systems. It reads JSONL step logs, parses them into a normalized schema (`AgentStep` -> `AgentTask`), and provides grading, root cause classification, comparison, reporting, and an A/B-test-disciplined enforcement workflow.

**Key data flow:**
```
JSONL logs -> Adapters -> AgentStep/AgentTask -> Analyzer -> Grader -> Root Cause -> Reports
                                                    |            |
                                                    v            v
                                              Signal Detectors  Fix Plan / Diagnose
```

**Source files (35 Python modules):**

| File | Lines | Purpose |
|------|-------|---------|
| `schema.py` | ~1100 | Core data model: `AgentStep`, `AgentTask`, `TaskOutcome`, context objects (Model, Tool, Reasoning, Browser) |
| `analyzer.py` | ~1300 | `TaskAnalysis` with 50+ metrics, error classification, task loading |
| `grader.py` | ~570 | JSON ruleset evaluation, grade assignment (GOLDEN/GOOD/OK/WEAK/BROKEN) |
| `root_cause.py` | ~1100 | 17 root cause classifiers with configurable thresholds |
| `surface.py` | ~1100 | Decision surface reconstruction, task tree, diff, reasoning extraction |
| `diagnose.py` | ~500 | Fix plan builder, target resolution, investigation hints |
| `completeness.py` | ~450 | 14-dimension data completeness checker |
| `comparison.py` | ~460 | Model/day/run comparison with divergence detection |
| `reports.py` | ~2500 | 12 report types, each in text/data/markdown variants |
| `golden.py` | ~530 | Efficiency ranking with 4 optimization profiles |
| `baseline.py` | ~840 | Naked prompt generation, overhead measurement, prompt-hash grouping |
| `flywheel.py` | ~360 | End-to-end quality loop with integrity checking |
| `capture.py` | ~160 | Task fixture capture with PII sanitization |
| `replay.py` | ~155 | Fixture replay with milestone comparison |
| `enforce.py` | ~1800 | A/B testing enforcement workflow with session persistence |
| `enforce_audit.py` | ~905 | 8 gaming detectors, challenge system, rule violations |
| `enforce_report.py` | ~100+ | Enforce session reporting |
| `watch.py` | ~237 | Live JSONL tail with real-time grading |
| `pricing.py` | ~231 | Layered pricing database (bundled/cache/remote) |
| `task_bank.py` | ~108 | TaskBank/TaskBankEntry data model |
| `contrib/task_bank.py` | ~752 | Fuzzy matching, 14 criteria evaluators |
| `contrib/novviola.py` | ~406 | NOVVIOLA-specific target resolver |
| `protocols.py` | ~97 | ToolRegistry, PromptBuilder, StepAdapter protocols |
| `runner.py` | ~84 | TaskRunner protocol, GenericHTTPRunner |
| `mcp_server.py` | ~968 | FastMCP server with 22+ tools |
| `cli.py` | ~3500 | 30+ CLI subcommands |
| `tui/app.py` | ~80+ | Textual-based interactive inspector |
| `pytest_plugin.py` | ~73 | pytest fixture for in-test grading |
| `signals/__init__.py` | ~60 | Signal detector registry with entry_point plugin discovery |
| `signals/commerce.py` | ~292 | Commerce/checkout flow detection |
| `signals/coding.py` | ~137 | Code editing/verification behavior |
| `signals/research.py` | ~97 | Search/citation/synthesis detection |
| `signals/multi_agent.py` | ~225 | Delegation/handoff orchestration |
| `signals/memory.py` | ~171 | Memory/RAG/context injection |
| `signals/planning.py` | ~145 | Plan creation/revision/execution |
| `adapters/__init__.py` | ~338 | Format auto-detection, 7 adapter registry |
| `adapters/*.py` (6 files) | - | Generic, OpenAI, OpenAI Chat, Anthropic, LangChain, CrewAI, OTel |
| `instrument/__init__.py` | ~30 | Auto-instrumentation hooks |
| `instrument/base.py` | ~100+ | StepRecorder with 50MB file rotation |

---

## Complete Tool/Command Registry

### MCP Tools (22 exposed via `mcp_server.py`)

| MCP Tool | Maps To | Description |
|----------|---------|-------------|
| `analyze` | `analyzer.analyze_task` | Analyze a single task, return 50+ metrics |
| `grade` | `grader.grade_task` | Grade tasks with rulesets |
| `root_cause` | `root_cause.classify_task` | Classify root cause of WEAK/BROKEN tasks |
| `completeness` | `completeness.check_completeness` | 14-dimension data completeness check |
| `surface_task` | `surface.surface_for_task` | Reconstruct full decision surface for a task |
| `search_tasks` | `analyzer.search_tasks` | Search tasks by text/site/grade (limit 25) |
| `diagnose` | `diagnose.build_fix_plan` | Build prioritized fix plan from root causes |
| `compare_runs` | `comparison.compare_model_runs` | Compare two log directories (model/day/run) |
| `report` | `reports.*` | Generate health/golden/broken/tools/flows/outcomes/actions/cost/fixes/coding/research/timeline/spin reports |
| `diff_tasks` | `surface.diff_tasks` | Diff two tasks step-by-step |
| `reasoning` | `surface.extract_reasoning_chain` | Extract full reasoning chain for a task |
| `tree` | `surface.build_task_tree` / `analyzer.build_task_tree` | Build hierarchical task tree |
| `golden_rank` | `golden.rank_golden_runs` | Rank golden/good runs by efficiency |
| `golden_compare` | `golden.explain_efficiency_gap` | Explain efficiency gap between two tasks |
| `task_bank_validate` | `contrib.task_bank.grade_with_task_bank` | Grade tasks against curated expectations |
| `task_bank_list` | `task_bank.TaskBank.load_json` | List task bank entries with filtering |
| `flywheel` | `flywheel.run_flywheel` | Full quality loop (grade + root cause + baseline) |
| `capture_task` | `capture.capture_task` | Capture task as fixture with PII sanitization |
| `pricing_show` | `pricing.load_pricing` | Show pricing data for a model |
| `enforce_init` | `enforce.EnforceSession.init` | Initialize enforce session |
| `enforce_check` | `enforce.EnforceSession.check` | Check test results after a change |
| `enforce_diff` | `enforce.EnforceSession.diff` | Show current git diff in enforce scope |
| `enforce_plan` | `enforce.EnforceSession.plan` | Record planned hypothesis |
| `enforce_guard` | `enforce.EnforceSession.guard` | Guard against gaming (run audit) |
| `enforce_status` | `enforce.EnforceSession.status` | Show enforce session status |
| `enforce_challenge` | `enforce.EnforceSession.challenge` | Cross-iteration challenge analysis |
| `enforce_reset` | `enforce.EnforceSession.reset` | Reset enforce session |
| `enforce_report` | `enforce_report.generate_report` | Generate enforce session report |

### CLI Subcommands (30+)

| CLI Command | MCP Exposed? | Description |
|------------|-------------|-------------|
| `agent-xray grade <dir>` | Yes | Grade all tasks in a directory |
| `agent-xray surface <task_id> <dir>` | Yes | Surface replay for a task |
| `agent-xray diagnose <dir>` | Yes | Build fix plan |
| `agent-xray completeness <dir>` | Yes | Data completeness check |
| `agent-xray compare <left_dir> <right_dir>` | Yes | Compare two runs |
| `agent-xray report <type> <dir>` | Yes | Generate reports (12 types) |
| `agent-xray golden rank <dir>` | Yes | Golden efficiency ranking |
| `agent-xray golden compare <task1> <task2> <dir>` | Yes | Explain efficiency gap |
| `agent-xray golden best <dir>` | **NO** | Find best exemplar per site |
| `agent-xray golden capture <task_id> <dir> <output>` | Yes (via capture_task) | Capture exemplar fixture |
| `agent-xray golden profiles` | **NO** | Show available optimization profiles |
| `agent-xray task-bank validate <dir>` | Yes | Validate against task bank |
| `agent-xray task-bank list` | Yes | List task bank entries |
| `agent-xray flywheel <dir>` | Yes | Full quality loop |
| `agent-xray enforce init` | Yes | Initialize enforce session |
| `agent-xray enforce check` | Yes | Check after change |
| `agent-xray enforce diff` | Yes | Show enforce diff |
| `agent-xray enforce plan` | Yes | Record hypothesis |
| `agent-xray enforce guard` | Yes | Audit for gaming |
| `agent-xray enforce status` | Yes | Session status |
| `agent-xray enforce challenge` | Yes | Cross-iteration challenge |
| `agent-xray enforce reset` | Yes | Reset session |
| `agent-xray enforce report` | Yes | Generate session report |
| `agent-xray enforce auto` | **NO** | **Autonomous enforce loop** (--agent-cmd) |
| `agent-xray tui <dir>` | **NO** | Interactive TUI inspector |
| `agent-xray watch <file>` | **NO** | Live tail JSONL with real-time grading |
| `agent-xray quickstart` | **NO** | Demo walkthrough |
| `agent-xray record <cmd>` | **NO** | Capture tool calls from subprocess stdout |
| `agent-xray replay <fixture> <dir>` | **NO** | Compare fixture to current logs |
| `agent-xray validate-targets <dir>` | **NO** | Check fix-plan target paths exist |
| `agent-xray baseline capture <task_id> <dir> <output>` | **NO** | Capture baseline from task |
| `agent-xray baseline generate <dir> <output_dir>` | **NO** | Generate baselines for all golden tasks |
| `agent-xray baseline list <dir>` | **NO** | List saved baselines |
| `agent-xray rules list` | **NO** | List available rulesets |
| `agent-xray rules show <name>` | **NO** | Show ruleset details |
| `agent-xray rules init <name> <output>` | **NO** | Scaffold a new ruleset |
| `agent-xray pricing list` | **NO** | List model pricing |
| `agent-xray pricing update` | **NO** | Fetch latest pricing from GitHub |
| `agent-xray pricing path` | **NO** | Show pricing file path |
| `agent-xray report overhead <dir>` | **NO** | Prompt overhead report |
| `agent-xray report prompt-impact <dir>` | **NO** | Prompt-hash impact report |
| `agent-xray report compare <dir> <day1> <day2>` | **NO** | Day-over-day comparison report |

### CLI-Only Filtering Options (Not Available in MCP)

| Flag | Description |
|------|-------------|
| `--days N` | Filter to last N days of logs |
| `--pattern GLOB` | Filter log files by glob pattern |
| `--site SITE` | Filter to specific site name |
| `--grade GRADE` | Filter to specific grade |
| `--outcome STATUS` | Filter by outcome status |
| `--since DATE` | Filter to logs since date |
| `--markdown` | Output in GitHub-flavored Markdown |
| `--json` | Output in JSON |
| `--rules PATH` | Custom rules file |
| `--pricing PATH` | Custom pricing file |
| `--project-root PATH` | Project root for target validation |
| `--profile PROFILE` | Optimization profile (balanced/cost/speed/steps) |
| `--task-bank PATH` | Task bank JSON file |
| `--baseline-dir PATH` | Baseline directory for comparison |

---

## MCP vs CLI vs Internal-Only Matrix

### DARK ABILITIES (CLI-only, powerful, NOT in MCP)

| Capability | CLI Command | Why It Matters | Risk of Missing It |
|-----------|------------|---------------|-------------------|
| **Autonomous enforce loop** | `enforce auto --agent-cmd "..."` | Runs the full enforce cycle autonomously with gaming detection | Agent must manually orchestrate enforce init/check/guard loop |
| **Interactive TUI** | `tui <dir>` | Rich interactive step inspector with keybindings (d=Diff, r=Root Cause, g=Grade, s=Surface) | No equivalent; agent must call surface/grade/root_cause individually |
| **Live watch mode** | `watch <file>` | Real-time JSONL tail with instant grading on task_complete | Agent must poll or re-run grade periodically |
| **Record subprocess** | `record <cmd>` | Captures tool calls from any subprocess stdout as JSONL | Must manually instrument the subprocess |
| **Fixture replay** | `replay <fixture> <dir>` | Compare saved golden fixture to current run (IMPROVED/REGRESSION/STABLE) | Must manually call capture + compare |
| **Validate targets** | `validate-targets --project-root <path>` | Check that fix-plan file paths actually exist on disk | Stale targets in fix plans go unnoticed |
| **Baseline management** | `baseline capture/generate/list` | Capture minimal-prompt baselines, generate for all golden tasks, list saved | Cannot measure prompt overhead without baselines |
| **Rules management** | `rules list/show/init` | Discover bundled rulesets, inspect their contents, scaffold custom ones | Agent doesn't know what rulesets exist or how to create new ones |
| **Pricing management** | `pricing list/update/path` | View pricing DB, fetch updates from GitHub, find cache path | Agent can only use bundled pricing, cannot update |
| **Golden best** | `golden best <dir>` | Find single best exemplar per site | Must call golden rank and manually filter |
| **Golden profiles** | `golden profiles` | Show the 4 optimization profiles with weight details | Agent doesn't know what profiles exist |
| **Overhead report** | `report overhead <dir>` | Prompt overhead vs baseline with contributing factor analysis | Overhead measurement invisible |
| **Prompt-impact report** | `report prompt-impact <dir>` | Per-prompt-hash performance comparison | Cannot A/B test prompt variants |
| **Day comparison report** | `report compare <dir> <day1> <day2>` | Side-by-side day-over-day comparison | Must manually compute from separate grade runs |
| **Quickstart** | `quickstart` | Demo walkthrough for new users | N/A for agents, but useful for onboarding |
| **Markdown output** | `--markdown` flag on most commands | GFM output for docs/PRs/issues | MCP returns JSON or truncated text |

### Internal-Only Functions (Neither CLI nor MCP)

| Function | Location | Purpose |
|----------|----------|---------|
| `generate_naked_prompt(task)` | `baseline.py` | Convert golden task to minimal instruction prompt |
| `build_baseline(task, analysis)` | `baseline.py` | Create baseline snapshot from task metrics |
| `measure_overhead(task, analysis, grade, baseline)` | `baseline.py` | Measure overhead of one task vs baseline |
| `group_by_prompt_hash(tasks, analyses, grades)` | `baseline.py` | A/B grouping by system_prompt_hash |
| `format_overhead_report(results, hash_groups)` | `baseline.py` | Format overhead report text |
| `format_prompt_impact_report(hash_groups)` | `baseline.py` | Format prompt impact text |
| `audit_change(hunks, config)` | `enforce_audit.py` | Gaming detection on diff hunks |
| `challenge_iterations(records, config)` | `enforce_audit.py` | Cross-iteration challenge analysis (9 checks) |
| `classify_diff_quality(hunks)` | `enforce_audit.py` | Classify change as behavioral_improvement/bug_fix/refactor/etc |
| `classify_change_quality(record, config)` | `enforce_audit.py` | Rate change as EXCELLENT/GOOD/NEUTRAL/POOR/HARMFUL |
| `detect_rule_violations(hunks, rules)` | `enforce_audit.py` | Project-specific rule checking |
| `build_fixture(task, sanitize)` | `capture.py` | Build fixture dict with PII scrubbing |
| `compare_fixture_to_task(fixture, task)` | `replay.py` | Milestone-based regression detection |
| `coerce_step(record)` / `coerce_steps(records)` | `protocols.py` | Normalize raw dicts to AgentStep |
| `format_info(path)` | `adapters/__init__.py` | Detect trace format with confidence score |
| `autodetect(path)` | `adapters/__init__.py` | Best-guess adapter selection |
| `adapt(path, format)` | `adapters/__init__.py` | Load any trace format into AgentSteps |
| `check_integrity(locks)` | `flywheel.py` | Verify evaluation drift during flywheel |
| `validate_fix_targets(plan, project_root)` | `diagnose.py` | Check target paths exist on disk |
| `list_all_targets(resolver)` | `diagnose.py` | Get all targets for all root causes |
| `register_target_resolver(name, resolver)` | `diagnose.py` | Register custom target resolver |

---

## Deep Dive: ENFORCE Workflow

The enforce workflow is a disciplined A/B testing loop for code changes. It prevents test gaming, tracks predictions, and maintains session state across iterations.

### Configuration (`EnforceConfig`)

| Field | Default | Description |
|-------|---------|-------------|
| `test_command` | (required) | Shell command to run tests |
| `max_iterations` | 50 | Maximum enforce iterations before forced stop |
| `challenge_every` | 5 | Run cross-iteration challenge every N iterations |
| `require_improvement` | `True` | Each iteration must improve or maintain test count |
| `allow_test_modification` | `True` | Whether test files can be modified |
| `stash_first` | `True` | Git stash before starting |
| `max_files_per_change` | 5 | Maximum files changed per iteration |
| `max_diff_lines` | 200 | Maximum diff lines per iteration |
| `rules_file` | `None` | Path to project rules (forbidden patterns, etc.) |
| `scope` | `None` | Git pathspec to limit scope (multi-agent repos) |
| `test_timeout` | 300 | Seconds before test command times out |

### Session Lifecycle

1. **`enforce init`** -- Creates `.agent-xray-enforce/` directory, runs baseline tests, records initial pass/fail counts, stashes uncommitted work
2. **`enforce plan`** -- Agent records hypothesis: what it expects to change, which tests it expects to fix, predicted outcome
3. *Agent makes code changes*
4. **`enforce check`** -- Runs tests, compares to baseline, records new pass/fail, calculates delta
5. **`enforce guard`** -- Runs 8 gaming detectors on the diff, returns VALID/SUSPICIOUS/GAMING verdict
6. **`enforce challenge`** (every N iterations) -- Runs 9 cross-iteration checks for systemic gaming
7. **`enforce diff`** -- Shows current diff in scope
8. **`enforce status`** -- Shows iteration count, pass/fail trend, current state
9. **`enforce report`** -- Generates full session report (text/JSON/Markdown)
10. **`enforce reset`** -- Wipes session state

### Gaming Detection (8 Heuristic Detectors)

| Detector | Confidence | What It Catches |
|----------|------------|----------------|
| `detect_test_file_modification` | 0.7 | Agent edited test files to make them pass |
| `detect_hardcoded_values` | 0.5 | Agent hardcoded expected values |
| `detect_special_case_branching` | 0.6 | Added `if test_mode` or `if env == "test"` branches |
| `detect_mock_insertion` | 0.4 | Inserted mocks/patches to bypass real behavior |
| `detect_assertion_weakening` | 0.7 | Changed assertEqual to assertIn, made assertions less strict |
| `detect_exception_swallowing` | 0.7 | Added bare `except: pass` or catch-all error handling |
| `detect_early_return` | 0.4 | Added early return before the code-under-test executes |
| `detect_import_removal` | 0.3 | Removed imports to disable functionality |

**Verdict calculation:** `max_confidence * 0.7 + avg_confidence * 0.3` -- GAMING (>0.6), SUSPICIOUS (>0.3), VALID (<0.3)

### Cross-Iteration Challenge (9 Checks)

1. **Consecutive suspicious** -- Multiple SUSPICIOUS verdicts in a row
2. **Net regression** -- Total test count decreased vs baseline
3. **Hot files** -- Same file modified in 3+ iterations (thrashing)
4. **Test flip-flop** -- Tests that pass then fail then pass again
5. **Dependency risk** -- Changes to dependency files (requirements, setup, config)
6. **Coverage gap** -- New code without corresponding test changes
7. **Cumulative gaming signals** -- Gaming detector hits accumulating over time
8. **Assertion erosion** -- Assertions getting progressively weaker
9. **Diminishing returns / Persistent failures / Scope creep**

### Prediction Tracking (GAP 8)

When the agent calls `enforce plan`, it records:
- What hypothesis it's testing
- Which specific tests it expects to fix
- What it expects the outcome to be

After `enforce check`, the system evaluates prediction accuracy. This is a meta-analysis signal -- agents that predict well are making principled changes, not shotgunning.

### Meta-Analysis (GAP 2)

Each change is classified as:
- `root_cause_fix` -- Addresses the actual cause of failure
- `test_update` -- Modifies test expectations
- `configuration_change` -- Changes settings/config
- `symptom_patch` -- Masks the symptom without fixing root cause

### Autonomous Mode (`enforce auto`)

CLI-only. Runs the full enforce loop in a subprocess:
```bash
agent-xray enforce auto --test-cmd "pytest tests/" --agent-cmd "claude-code --task 'fix the tests'"
```
This is the most powerful enforce feature and is NOT exposed via MCP. The agent must use `--agent-cmd` to specify what to run between iterations.

---

## Deep Dive: GRADING System

### Ruleset Structure

```json
{
  "name": "browser_flow",
  "description": "...",
  "extends": "default",
  "signals": [...],
  "grade_thresholds": {"GOLDEN": 8, "GOOD": 5, "OK": 2, "WEAK": 0},
  "golden_requirements": [...]
}
```

### Signal Evaluation

Each signal maps a metric to points:

```json
{
  "name": "payment_reached",
  "metric": "commerce.reached_payment_confidence",
  "in": ["url_match", "action_sequence"],
  "points": 4,
  "reason": "+4 payment reached"
}
```

**Supported operators:** `gte`, `gt`, `lte`, `lt`, `equals`, `in`, `contains_any`, `ne`, `not_in`

**Metric resolution:** The grader resolves metrics by looking up the analysis object. Dotted paths like `commerce.reached_payment_confidence` are resolved through `signal_metrics["commerce"]["reached_payment_confidence"]`.

**`else_points`/`else_reason`:** Some signals award negative points when the condition is NOT met.

### Golden Requirements

Golden requirements are hard gates. Even if the score exceeds the GOLDEN threshold, the task is downgraded to GOOD if any golden requirement fails.

### Grade Thresholds

| Grade | Default | Browser Flow | Coding Agent | Research Agent |
|-------|---------|-------------|-------------|---------------|
| GOLDEN | 4 | 8 | 7 | 7 |
| GOOD | 3 | 5 | 4 | 4 |
| OK | 1 | 2 | 1 | 1 |
| WEAK | 0 | 0 | -2 | -2 |

Everything below WEAK is BROKEN.

### Bundled Rulesets

| Name | File | Focus |
|------|------|-------|
| `default` | `rules/default.json` | Generic execution quality (tool diversity, loop resistance, reliability) |
| `browser_flow` | `rules/browser_flow.json` | Commerce/navigation (URL progression, milestones, fills, checkout/payment) |
| `coding_agent` | `rules/coding_agent.json` | Development (test-to-edit ratio, verify cycles, lint runs) |
| `research_agent` | `rules/research_agent.json` | Research (source diversity, synthesis, citations) |

### Ruleset Inheritance

Rulesets can `"extends": "default"` to inherit signals and thresholds, then override or add new ones.

### Ruleset Validation

`validate_rules()` checks for: unknown fields, missing thresholds, overlapping rules with contradictory scoring, invalid operator names.

---

## Signal Detectors (Plugin System)

### Built-in Detectors

| Detector | Key Metrics | Flow Detection |
|----------|------------|----------------|
| **CommerceDetector** | `reached_cart`, `reached_checkout`, `reached_payment`, `fill_count`, `real_fill_count`, `payment_fill_count`, `payment_fields_confirmed`, `milestone_confidence` | Cart->Checkout->Payment flow with confidence levels (none/keyword_match/url_match/action_sequence) |
| **CodingDetector** | `file_operations`, `test_runs`, `build_runs`, `lint_runs`, `git_operations`, `test_to_edit_ratio`, `has_test_verify_cycle`, `unique_files_touched` | Edit->Test->Fix->Verify cycle |
| **ResearchDetector** | `search_count`, `read_count`, `source_diversity`, `citation_count`, `has_synthesis_step`, `search_to_read_ratio` | Search->Read->Synthesize flow |
| **MultiAgentDetector** | `delegation_count`, `unique_agents`, `delegation_success_rate`, `max_delegation_depth` | Delegation/handoff patterns with target extraction and success tracking |
| **MemoryDetector** | `memory_operations`, `unique_keys`, `recall_hit_rate`, `rag_queries`, `context_injections` | Store/recall/forget/RAG patterns with miss detection |
| **PlanningDetector** | `plans_created`, `plan_steps_executed`, `plan_revisions`, `plan_completion_rate` | Plan creation/execution/revision tracking |

### Plugin Extension

Custom detectors can be registered via `entry_points`:

```toml
[project.entry-points."agent_xray.signals"]
my_detector = "my_package.my_module:MyDetector"
```

The detector must implement the `SignalDetector` protocol:
```python
class SignalDetector(Protocol):
    name: str
    def detect_step(self, step: AgentStep) -> dict[str, Any]: ...
    def summarize(self, task: AgentTask, step_signals: list[dict]) -> dict[str, Any]: ...
```

---

## Root Cause Classification

### 17 Root Cause Categories

| Root Cause | Severity | Description |
|-----------|----------|-------------|
| `approval_block` | 5 | Tool blocked by approval/permission gate |
| `tool_rejection_mismatch` | 5 | Tools rejected that should have been available |
| `routing_bug` | 4 | Steps with zero tools available |
| `environment_drift` | 4 | Timeout/click-fail/not-found errors dominate |
| `tool_bug` | 4 | Tool implementation errors dominate |
| `memory_overload` | 4 | Context usage >85%, quality degrades |
| `spin` | 3 | Same tool repeated 5+ times consecutively |
| `stuck_loop` | 3 | Low URL diversity + many steps |
| `tool_selection_bug` | 3 | Wrong tool chosen despite right one available |
| `delegation_failure` | 3 | Sub-agent delegation returned errors |
| `test_failure_loop` | 3 | Same failing test signature repeated |
| `early_abort` | 2 | Task completed in <3 steps |
| `model_limit` | 2 | Exceeded 50 steps (context/capability limit) |
| `prompt_bug` | 2 | Reasoning shows prompt confusion/uncertainty |
| `insufficient_sources` | 2 | Research task with too few sources |
| `reasoning_bug` | 1 | Model made poor decisions despite good tools |
| `valid_alternative_path` | 0 | Non-browser path that may be correct |
| `consultative_success` | 0 | Correct consultative response |
| `unclassified` | 1 | No pattern matched |

### Classification Config

`ClassificationConfig` has 12 tunable thresholds:

| Parameter | Default | What It Controls |
|-----------|---------|-----------------|
| `spin_threshold` | 5 | Min consecutive repeats for spin classification |
| `high_error_rate` | 0.5 | Error rate threshold for error-dominant classification |
| `model_limit_steps` | 50 | Step count threshold for model_limit |
| `stuck_loop_min_steps` | 5 | Min steps for stuck_loop classification |
| `early_abort_max_steps` | 3 | Max steps for early_abort classification |
| `test_failure_loop_min_runs` | 2 | Min repeated test failures for loop classification |
| `insufficient_sources_min_searches` | 2 | Min searches to avoid insufficient_sources |
| `low_source_diversity_threshold` | 2 | Min unique domains for research tasks |
| `memory_overload_usage_pct` | 85.0 | Context usage % triggering memory_overload |
| `memory_overload_short_output_chars` | 80 | Short output threshold under memory pressure |
| `expected_rejections` | `frozenset()` | Tools whose rejection is intentional policy |

### Prompt Bug Patterns

12 regex patterns that detect prompt-level confusion in LLM reasoning:
- "not sure if.*should", "confused.*instructions", "seems.*contradictory"
- "prompt.*unclear", "instructions.*ambiguous"
- And 10 more patterns mapping to specific prompt sections (research, tools, browser, payment, planning)

---

## Adapters (Format Support)

7 trace format adapters with auto-detection:

| Format | Module | Detection Signals |
|--------|--------|-------------------|
| `generic` | `adapters/generic.py` | Nested `tool_name` + `task_id` fields |
| `openai` | `adapters/openai_sdk.py` | `object: "run"` or `object: "run_step"` |
| `openai_chat` | `adapters/openai_chat.py` | `role: "assistant"` + `tool_calls` array |
| `anthropic` | `adapters/anthropic.py` | Content blocks with `type: "tool_use"` or `"tool_result"` |
| `langchain` | `adapters/langchain.py` | Events: `agent_action`, `on_tool_start`, `on_tool_end` |
| `crewai` | `adapters/crewai.py` | `agent_role` field or `crew_*` prefixed keys |
| `otel` | `adapters/otel.py` | `resourceSpans` or `scopeSpans` in first 8KB |

Auto-detection uses heuristic scoring (pattern matching) plus load-count verification (actually try parsing with each adapter). Confidence score returned as 0.0-1.0.

---

## Auto-Instrumentation

`instrument/` provides SDK-specific monkey-patching:

| SDK | Instrumentor | What It Patches |
|-----|-------------|----------------|
| Anthropic | `AnthropicInstrumentor` | `messages.create` / `messages.stream` |
| OpenAI | `OpenAIInstrumentor` | `chat.completions.create` |
| LangChain | `XRayCallbackHandler` | LangChain callback interface |
| MCP | `XRayMCPProxy` | MCP tool call interception |

`auto_instrument()` detects which SDKs are installed and patches them automatically. `xray_trace` is a context manager for manual instrumentation.

`StepRecorder` in `instrument/base.py` provides thread-safe JSONL writing with 50MB file rotation and buffered writes.

---

## Report Types

12 report types, each available in text/data/markdown variants:

| Report | CLI Name | What It Shows |
|--------|----------|---------------|
| **Health Dashboard** | `health` | Grade distribution, error/spin/timeout/hallucination/approval counts, day trends |
| **Golden Report** | `golden` | GOLDEN/GOOD tasks ranked by score with milestones, fills, sites |
| **Broken Report** | `broken` | BROKEN tasks with WHY breakdown (spin/timeout/errors/hallucinated/stuck/routing) |
| **Tool Effectiveness** | `tools` | Per-tool call counts, error rates, avg latency, BROKEN/HIGH ERR flags |
| **Flow Analysis** | `flows` | Domain-specific flow funnels (commerce: Cart->Checkout->Payment; coding: Edit->Test->Fix->Verify; research: Search->Read->Synthesize; browser: Browse->Form->Complete) |
| **Outcome Distribution** | `outcomes` | Outcome status counts with grade cross-tab |
| **Action Items** | `actions` | Prioritized operational fixes (broken tools, spin epidemics, routing gaps, hallucinated tools, approval blocks) |
| **Cost Analysis** | `cost` | Token usage, cost by model/category/day, most expensive tasks, pricing coverage |
| **Fix Plan** | `fixes` | Root-cause-grouped fix plan with targets, evidence, severity, impact |
| **Coding Report** | `coding` | File operations, test runs, lint/git ops, edit-test cycle rate |
| **Research Report** | `research` | Searches, reads, citations, unique domains, synthesis rate |
| **Timeline Report** | `timeline` | Hourly bucketed grade distribution with avg duration and error rate |
| **Spin Analysis** | `spin` | (In reports.py lines 2483+) Consecutive-repeat sequence detection |
| **Day Comparison** | `compare` (CLI only) | Side-by-side day-over-day metrics |
| **Overhead** | `overhead` (CLI only) | Per-site step/duration/cost overhead vs baseline |
| **Prompt Impact** | `prompt-impact` (CLI only) | Per-prompt-hash performance A/B comparison |

---

## Baseline & Overhead Measurement

### Naked Prompt Generation

`generate_naked_prompt(task)` converts a golden task's tool sequence into minimal imperative instructions. Uses `_TOOL_VERB_MAP` to translate tool names to verbs (e.g., `browser_navigate` -> "Go to", `browser_click` -> "Click").

This answers: "What is the simplest possible instruction that would reproduce this task's behavior?"

### Baseline Capture

`build_baseline(task, analysis)` snapshots: step count, duration, tokens, cost, errors, milestones, tool sequence, and the naked prompt.

### Overhead Measurement

`measure_overhead()` compares one task against its site baseline:
- Step overhead %
- Duration overhead %
- Cost overhead %
- Success delta (milestone comparison: better/same/worse)
- Overhead category: efficient (<50%), acceptable (50-150%), bloated (150-300%), pathological (>300%)
- Contributing factors: extra tools not in baseline, repeated tools, more errors, extra steps, redundant web_search before navigate

### Prompt-Hash Grouping

`group_by_prompt_hash()` aggregates tasks by `system_prompt_hash` to enable A/B comparison of different prompt variants. Shows avg steps, duration, cost, golden rate, broken rate, and overhead per prompt variant.

---

## Golden Ranking & Efficiency

### Optimization Profiles

| Profile | Step Weight | Duration Weight | Cost Weight | Error Weight |
|---------|-----------|----------------|------------|-------------|
| `balanced` | 0.4 | 0.2 | 0.2 | 0.2 |
| `cost` | 0.2 | 0.1 | 0.5 | 0.2 |
| `speed` | 0.2 | 0.5 | 0.1 | 0.2 |
| `steps` | 0.6 | 0.1 | 0.1 | 0.2 |

### Tier Assignment

- **EXEMPLAR** -- Top performer per site
- **EFFICIENT** -- Above median efficiency
- **BASELINE** -- Below median

### Key Functions

- `rank_golden_runs()` -- Grade, filter to GOLDEN/GOOD, group by site, rank by weighted efficiency
- `find_exemplars()` -- Return best EXEMPLAR per site
- `explain_efficiency_gap()` -- Compare two task analyses and explain WHY one is more efficient
- `capture_exemplar()` -- Save exemplar as fixture with metadata

---

## Task Bank

### Core Data Model (`task_bank.py`)

`TaskBankEntry` with fields: `id`, `category`, `user_text`, `success_criteria`, `difficulty`, `optimal_chain`, `test_command`

`TaskBank` supports filtering by `category` and `difficulty`.

### Evaluation Engine (`contrib/task_bank.py`)

14 criterion evaluators:

| Criterion | What It Checks |
|-----------|---------------|
| `must_answer_contains` | Final answer contains expected keywords |
| `answer_type` | Response type matches expectation |
| `must_reach_url` | Agent visited specific URL pattern |
| `must_fill_fields` | Form fill operations performed |
| `min_urls` | Minimum URL diversity |
| `max_steps` | Step count within limit |
| `payment_fields_visible` | Payment page was reached |
| `must_not_fill_payment` | Payment fields were NOT filled (safety) |
| `must_reach_cart` | Cart page was reached |
| `must_reach_checkout` | Checkout page was reached |
| `must_use_tools` | Specific tools were used |
| `no_browser_needed` | Task should complete without browser |
| `must_have_answer` | Task produced a final answer |
| `min_tool_count` | Minimum number of tool calls |

Fuzzy matching via `SequenceMatcher` + token overlap (threshold 0.45) matches task bank entries to actual traces.

`grade_with_task_bank()` first grades with the ruleset, then applies bank criteria. Critical criteria failures downgrade GOLDEN to GOOD.

---

## Flywheel (End-to-End Quality Loop)

`run_flywheel()` combines:
1. Load tasks from log directory
2. Build integrity locks on rules file, task bank files, grader.py source, replay.py source
3. Run signal detectors
4. Analyze all tasks
5. Grade all tasks
6. **Integrity check** -- verify no evaluation files changed during the run (prevents gaming the grader itself)
7. Classify failures with optional custom hooks
8. Build fix plan
9. Replay fixtures (if fixture_dir provided)
10. Compare against baseline (if baseline_path provided)
11. Output results

**Integrity Lock System:** SHA-256 hashes of evaluation files are captured at start and verified after grading. If any file changed during the run, raises `EVALUATION_DRIFT` error. This prevents the agent from gaming by modifying the grading rules mid-evaluation.

---

## Fixture Capture & Replay

### Capture (`capture.py`)

- PII sanitization: email, phone, card numbers, ZIP codes, addresses, URLs
- Milestone detection: FORM_FILL, CART, CHECKOUT, PAYMENT
- Output: JSON fixture with sanitized step sequence, milestones, expected result content

### Replay (`replay.py`)

- Fuzzy match fixture to current tasks by task_id or text similarity + site
- Milestone-depth comparison with step count delta
- Verdict: IMPROVED (deeper milestone or fewer steps), REGRESSION (shallower milestone or 5+ more steps), STABLE, UNMATCHED, EVALUATION_DRIFT

---

## Pricing System

### Layered Resolution

1. Custom path (explicit `--pricing` flag)
2. Environment variable (`AGENT_XRAY_PRICING`)
3. Local cache (`~/.agent-xray/pricing.json`, 7-day TTL)
4. Bundled fallback (shipped with package)

### Model Resolution

Exact match > alias > prefix match (longest key wins).

### Cost Calculation

`get_model_cost(model_name, input_tokens, output_tokens, cached_tokens, pricing_data)` returns USD cost.

### Management (CLI only)

- `pricing list` -- Show all model prices
- `pricing update` -- Fetch from GitHub
- `pricing path` -- Show cache file location

---

## Pytest Plugin

```python
# In tests:
def test_agent_behavior(xray):
    steps = [...] # list of step dicts
    report = xray.analyze(steps)
    assert report.grade in ("GOLDEN", "GOOD")
    assert report.error_rate < 0.1
    assert report.root_cause is None
```

`XrayFixture` provides `.analyze(steps)` returning `XrayReport` with grade, score, reasons, root_cause, and full TaskAnalysis.

CLI option: `--xray-rules <path>` to specify custom rules for the fixture.

---

## Extension Points & Plugin Hooks

| Extension Point | Mechanism | How to Use |
|----------------|-----------|-----------|
| **Signal detectors** | `entry_points("agent_xray.signals")` | Register custom detectors in pyproject.toml |
| **Target resolvers** | `register_target_resolver(name, resolver)` | Map root causes to project-specific file paths |
| **Format adapters** | `FORMATS` dict in `adapters/__init__.py` | Add new trace format parsers |
| **Task runner** | `TaskRunner` protocol | Custom HTTP/process task submission |
| **Prompt builder** | `PromptBuilder` protocol | Custom prompt reconstruction for surface analysis |
| **Tool registry** | `ToolRegistry` protocol | Custom tool surface description |
| **Step adapter** | `StepAdapter` protocol | Custom step normalization |
| **Flywheel hooks** | `DetectorHook` callable | Custom root-cause overrides during flywheel |
| **NOVVIOLA resolver** | `contrib/novviola.py:register()` | Pre-built resolver with 14 root-cause->file-path mappings |

---

## Dark Abilities (CLI-only, Not in MCP)

> **Status: Nearly all gaps CLOSED as of v1.15.0 (2026-03-29).**
> 41 MCP tools total. 13 new MCP tools + 3 report types + filtering params added across two audit rounds.

### Remaining CLI-Only (Low Priority for MCP)

1. **`enforce auto`** -- The autonomous enforce loop. Agents can manually orchestrate init/plan/check/guard/challenge. Auto mode is a convenience, not a blocker.
2. **`watch`** -- Live TUI monitoring. Agents don't need real-time streaming.
3. **`tui`** -- Interactive inspector with keybindings. Not meaningful for MCP.
4. **`record`** -- Subprocess stdout capture. Niche instrumentation use case.
5. **`quickstart`** -- Demo walkthrough for humans.
6. **`--markdown` output** -- MCP truncates to 20,000 chars. CLI can produce full Markdown for docs/PRs.
7. **`--outcome`, `--since`, `--pattern` filters** -- `days`, `site`, and `grade_filter` are now in MCP; remaining filters are lower priority.

### CLOSED Gaps (Now in MCP)

**Round 1 (commit 4a6bc19):**
- `replay` -- MCP tool added
- `validate-targets` -- MCP tool `validate_targets` added
- `baseline capture/list` -- MCP tools `baseline_capture`, `baseline_list` added
- `rules list/show/init` -- MCP tools `rules_list`, `rules_show`, `rules_init` added
- `golden best` -- MCP tool `golden_best` added
- `golden profiles` -- MCP tool `golden_profiles` added
- `--days`, `--site`, `--grade` filters -- Added to `analyze`, `grade`, `root_cause`, `diagnose`, `tree`, `search_tasks`, `report`
- `overhead`, `prompt-impact`, `compare` reports -- Added to `report()` tool

**Round 2 (v1.15.0, challenger audit):**
- `pricing list` -- MCP tool `pricing_list` added (full model pricing table)
- `baseline generate` -- MCP tool `baseline_generate` added (naked prompt generation)
- `task-bank show` -- MCP tool `task_bank_show` added (individual entry lookup)
- `format detect` -- MCP tool `format_detect` added (auto-detect trace format with confidence)

---

## Hidden/Underutilized Features

### 1. Evaluation Drift Detection
The flywheel's integrity lock system (SHA-256 hashing of grader.py, replay.py, rules files, and task bank files) prevents gaming the evaluator itself. This is a unique anti-gaming feature that most users don't know about.

### 2. Prompt Bug Pattern Matching
Root cause classification includes 12 regex patterns that match against LLM reasoning text to identify prompt-level confusion. This maps to specific prompt sections (research, tools, browser, payment, planning) and generates section-specific fix hints.

### 3. CommerceDetector Confidence Levels
Cart/checkout/payment detection has 4 confidence levels: `none`, `keyword_match`, `url_match`, `action_sequence`. The `action_sequence` level requires corroborating evidence (e.g., cart URL + add-to-cart action). The `browser_flow` ruleset uses these confidence levels directly in its scoring signals.

### 4. Contributing Factor Analysis
`_identify_contributing_factors()` in baseline.py identifies specific overhead causes: extra tools not in baseline, consecutive tool repetition, error delta, step delta, redundant web_search before navigate. This goes far beyond "X% overhead" to explain WHY.

### 5. Multi-Agent Delegation Tracking
The MultiAgentDetector tracks: delegation targets (extracted from tool inputs), delegation success (verified by checking subsequent steps for matching agent roles), and delegation depth (explicit `depth` fields or inferred from nested sub-agent calls).

### 6. Memory Overload Quality Degradation
Root cause classification for `memory_overload` checks for: context usage >85%, quality degradation in late steps vs early steps, reasoning mentioning context pressure, compaction/eviction events, and short final answers under pressure.

### 7. Suspiciously Short Detection
Both the commerce detector and the default grading rules detect "suspiciously short" tasks -- those that claim to reach payment in 1-2 steps, which is physically impossible for real commerce flows. This prevents false GOLDEN grades.

### 8. NOVVIOLA Target Resolver
`contrib/novviola.py` provides a pre-built `TargetResolver` that maps all 17 root causes to specific NOVVIOLA file paths, plus 14 prompt bug patterns to specific files with fix descriptions. It also includes `NOVVIOLA_VERIFY_COMMANDS` -- shell commands for each root cause. Activate with `register()`.

### 9. Task Bank Test Commands
Each `TaskBankEntry` can include a `test_command` -- a shell command that independently verifies the task was completed correctly. This is not currently surfaced in any MCP tool.

### 10. Adapter Confidence Scoring
`format_info(path)` returns not just the detected format but a confidence score (0.0-1.0) based on heuristic scoring + load verification. This could be exposed as an MCP tool for trace format debugging.

---

## Recommended Workflows

### Investigation Workflow (Recommended Order)

**START HERE — one call to get the full picture:**
```
triage(log_dir, days=1)  -> grades + worst failure surfaced + fix plan + next commands
```

**Then go deeper as needed:**
```
1. triage         -> START HERE. Grades all, surfaces worst, returns fix plan.
2. surface_task   -> Deep dive on a specific task from the fix plan.
3. reasoning      -> Read the model's reasoning chain for that task.
4. diff_tasks     -> Compare a GOLDEN vs BROKEN task to spot divergence.
5. root_cause     -> Get detailed root cause with evidence and confidence.
6. diagnose       -> Build comprehensive fix plan from all root causes.
7. compare_runs   -> After fixing, compare day-over-day improvement.
```

**Legacy flow (still works but triage replaces steps 1-4):**
```
completeness -> grade -> report health -> diagnose -> surface_task -> reasoning -> root_cause
```

### Quality Gate Workflow

```
1. flywheel       -> Grade + root cause + baseline comparison in one call.
2. task_bank_validate -> Check against curated expectations.
3. golden_rank    -> Rank the good runs to find exemplars.
4. capture_task   -> Save the best run as a fixture.
5. replay         -> (CLI only) Compare against saved fixture next time.
```

### A/B Testing Workflow

```
1. compare_runs   -> Compare left vs right log directories.
2. report overhead -> (CLI only) Measure prompt overhead vs baselines.
3. report prompt-impact -> (CLI only) Compare by prompt hash.
```

### Enforce Workflow (Code Change Discipline)

```
1. enforce init     -> Establish baseline test state.
2. enforce plan     -> Record your hypothesis.
3. (make changes)
4. enforce check    -> Verify improvement.
5. enforce guard    -> Detect gaming.
6. enforce challenge -> (every 5 iterations) Cross-iteration integrity.
7. enforce report   -> Summarize the session.
```

### MCP Agent Workflow (Minimal Token Path)

```
1. grade logs/structured/ --rules browser_flow
2. If BROKEN/WEAK exist: diagnose logs/structured/
3. For top fix-plan entry: surface_task <task_id> logs/structured/
4. For reasoning: reasoning <task_id> logs/structured/
5. After fixing: flywheel logs/structured/ (single call instead of grade+diagnose+compare)
```
