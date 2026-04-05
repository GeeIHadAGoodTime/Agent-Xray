# agent-xray Onboarding

`agent-xray` is a local-first debugger and referee for AI-agent behavior. It reconstructs what the agent actually saw, grades what happened, and, in enforce mode, tells you whether a proposed code change genuinely improved the task. Treat it as an evidence tool, not a vibes tool.

## Quick Start

```bash
pip install agent-xray
agent-xray quickstart          # generates demo traces and runs a full walkthrough
agent-xray analyze my-traces.jsonl
```

`analyze` loads a trace file, reconstructs the agent timeline, and prints a summary of what happened: tool calls, decisions, failures, and timing. No configuration, no task bank, no enforce setup. Start here.

---

## Complete Command Reference

### Core Analysis (the golden path)

Run these in order on the same trace file, stopping when you have enough evidence.

```
analyze  -->  "step 14 failed"
grade    -->  "browser_flow rule 3: no snapshot before click"
root-cause --> "agent skipped snapshot because tool was unavailable"
diagnose -->  "tool_available_count: 0 at step 12 due to approval block"
surface  -->  full detail confirms the approval bridge was not wired
compare  -->  after fix, day2 shows rule 3 passing
```

**1. `agent-xray analyze ./traces --rules browser_flow --task-bank task_bank.json --json`**
Reconstruct the full agent timeline: tool call sequence, step count, timing, and outcome. First command for any investigation.

**2. `agent-xray grade ./traces --rules browser_flow --task-bank task_bank.json --json`**
Apply a ruleset and produce pass/fail per rule. With `--task-bank`, grading matches each task to its bank entry and evaluates defined success criteria (must_reach_url, must_answer_contains, payment_fields_visible, max_steps). Without it, you only get generic signal-based grading.

**3. `agent-xray root-cause ./traces --rules browser_flow --task-bank task_bank.json --json`**
Identify the earliest divergence point: where did the agent first go wrong? Traces backward from the failure to the decision that caused it, filtering out downstream symptoms.

**4. `agent-xray diagnose ./traces --rules default --task-bank task_bank.json --project-root . --json`**
Produce structured diagnostics: tool availability gaps, approval blocks, prompt mismatches, timing anomalies, and a prioritized fix plan. Use when root-cause points at infrastructure rather than agent logic.

**5. `agent-xray surface task-123 ./traces --task-bank task_bank.json --json`**
Extract every detail for one task: full tool call chain, LLM inputs/outputs, intermediate state, and final outcome. Use when you need the complete picture for a single execution.

**6. `agent-xray compare ./runs/day1 ./runs/day2 --rules browser_flow --json`**
Side-by-side diff of two trace directories. Shows new failures, resolved failures, timing regressions, and tool call differences. This is how you measure whether a change actually helped.

### Investigation Tools

**`agent-xray search "pizza" ./traces --grade BROKEN,WEAK --json`**
Find tasks by user_text substring. Filter by grade to focus on failures. Use when you know what the user asked but not the task ID.

**`agent-xray tree ./traces --rules browser_flow --json`**
Show a day/site/task hierarchy. With `--rules`, enriches the tree with grades and scores. Use for a bird's-eye view of trace organization.

**`agent-xray reasoning task-123 ./traces --json`**
Extract the model's reasoning chain for a specific task. Use when you need to understand why the agent chose a particular path.

**`agent-xray diff task-123 task-456 ./traces --summary --json`**
Compare two tasks side by side: tool calls, timing, outcomes. Use `--summary` for a concise view. Use when two runs of the same task diverged.

**`agent-xray completeness ./traces --json`**
Check data completeness of agent traces: missing fields, sparse steps, format issues. Use to verify your logging pipeline before analysis.

### Live Monitoring

**`agent-xray watch ./traces/agent-steps-20260328.jsonl --rules browser_flow --poll 2 --json`**
Tail a JSONL log file and grade tasks as they complete in real-time. Use during live agent sessions to catch failures immediately.

**`agent-xray tui ./traces --task-id task-123`**
Open the interactive terminal decision-surface inspector. Browse steps, inspect tool inputs/outputs, and explore the agent's view interactively.

**`agent-xray record --output-dir ./traces -- python my_agent.py`**
Run a subprocess and capture tool calls from its stdout as JSONL steps. Use to create traces from any agent that prints tool calls to stdout.

### Quality Assurance

#### golden -- Golden exemplar management

Rank, inspect, and capture the best agent runs as reference fixtures.

- **`agent-xray golden rank ./traces --optimize balanced --rules browser_flow --json`** -- Rank golden/good runs by efficiency within each site. Profiles: `balanced`, `cost`, `speed`, `steps`.
- **`agent-xray golden best ./traces --optimize balanced --json`** -- Show the single top exemplar for each site.
- **`agent-xray golden capture ./traces --site dominos --out fixtures/dominos.json`** -- Capture the best run for a site as a reusable fixture.
- **`agent-xray golden compare ./traces --fixtures ./golden-fixtures/ --json`** -- Compare current runs against previously captured golden fixtures to detect regressions.
- **`agent-xray golden profiles`** -- List all available optimization profiles and their weight distributions.

#### enforce -- Controlled experiment loop

Disciplined, incremental agent improvement: one hypothesis, one change, one measurement.

Loop: `init` --> `plan` --> (make one change) --> `check` --> iterate or revert.

- **`agent-xray enforce init --test "pytest tests/ -x -q --tb=short"`** -- Capture the deterministic baseline. Use the same test command for the entire session.
- **`agent-xray enforce plan --hypothesis "checkout timeout caused by missing await" --expected-tests tests/test_checkout.py::test_flow`** -- Register one hypothesis and predicted test movement before editing code.
- **`agent-xray enforce check --json`** -- Run the test command against the changed tree and compare to baseline.
- **`agent-xray enforce diff --full --json`** -- Preview whether the current diff fits one-change-at-a-time limits before running check.
- **`agent-xray enforce status --json`** -- Show current session status, baseline context, and iteration history.
- **`agent-xray enforce challenge --json`** -- Run adversarial cross-iteration review: catches cumulative gaming, churn, and scope creep.
- **`agent-xray enforce report --format markdown`** -- Generate the full enforcement report (text, JSON, or Markdown).
- **`agent-xray enforce reset`** -- Discard the active session when you intentionally want a fresh baseline.
- **`agent-xray enforce auto --test "pytest tests/ -x" --agent-cmd "codex exec '{hypothesis}'" --max-iterations 50`** -- Run the full autonomous enforce loop. The agent iterates inside the enforce pipeline with guardrails.
- **`agent-xray enforce guard --json`** -- Check for unreviewed working-tree changes outside the tracked hypothesis.

#### task-bank -- Task bank management

- **`agent-xray task-bank list ./task_bank.json --json`** -- List all task entries in a bank.
- **`agent-xray task-bank show ./task_bank.json checkout-payment-gate --json`** -- Show one task entry with its success criteria.
- **`agent-xray task-bank validate ./task_bank.json --json`** -- Validate task bank schema and criterion names.

#### flywheel, replay, capture -- End-to-end and fixture workflows

**`agent-xray flywheel ./traces --rules browser_flow --baseline ./prev.json --out ./current.json --json`**
Run grading + root-cause + baseline comparison in one shot. Use for automated quality loops.

**`agent-xray replay ./fixtures/task-123.json --log-dir ./traces --json`**
Compare a captured fixture against current logs to detect behavioral drift.

**`agent-xray capture task-123 --log-dir ./traces --out ./fixtures/task-123.json --no-sanitize`**
Capture a task as a sanitized (or raw) fixture for replay and regression testing.

### Configuration

#### rules -- Ruleset management

- **`agent-xray rules list`** -- List available built-in rulesets (default, browser_flow, coding_agent, research_agent).
- **`agent-xray rules show browser_flow`** -- Show a ruleset's full JSON definition.
- **`agent-xray rules init --base browser_flow`** -- Scaffold a custom ruleset to stdout, extending an existing base.

#### pricing -- Model pricing data

- **`agent-xray pricing list`** -- Show all known models and their per-token prices.
- **`agent-xray pricing show gpt-4.1-nano`** -- Show pricing for a specific model.
- **`agent-xray pricing update`** -- Fetch the latest pricing data from GitHub.
- **`agent-xray pricing path`** -- Show where pricing data is loaded from.

#### baseline -- Overhead measurement baselines

- **`agent-xray baseline capture task-123 ./traces -o baselines/dominos.json`** -- Capture a task as a baseline for overhead measurement.
- **`agent-xray baseline generate task-123 ./traces`** -- Print the naked prompt for a task (useful for prompt engineering).
- **`agent-xray baseline list ./baselines/`** -- List all baselines in a directory.

#### Other configuration commands

**`agent-xray quickstart`**
Create a demo trace directory and run a full walkthrough. Use for first-time setup and verification.

**`agent-xray validate-targets --project-root /path/to/project`**
Validate that fix-plan target paths from diagnose actually exist in your project. Use after `diagnose` to confirm fix targets are real files.

### Reports (16 types)

```bash
agent-xray report ./traces <type> [--rules browser_flow] [--task-bank task_bank.json] [--json|--markdown]
```

| Type | What it shows |
|------|--------------|
| `health` | Overall trace health: task counts, pass rates, grade distribution |
| `golden` | Golden and good runs with scores and efficiency metrics |
| `broken` | All broken tasks with failure reasons and step counts |
| `tools` | Tool usage statistics: call counts, success rates, timing |
| `flows` | Common tool-call sequences and navigation patterns |
| `outcomes` | Outcome distribution by category and site |
| `actions` | Action-level breakdown: what the agent did at each step |
| `coding` | Coding-specific metrics: file edits, test runs, compilation |
| `research` | Research-specific metrics: queries, sources, synthesis |
| `cost` | Token usage and estimated cost per task and model |
| `fixes` | Fix attempts: what was tried, what worked, what regressed |
| `timeline` | Temporal view with configurable buckets (`--bucket 15m`) |
| `spins` | Spin detection: repeated tool calls, loops, stuck patterns |
| `compare` | Day-over-day comparison (`--day1 20260327 --day2 20260328`) |
| `overhead` | Agent overhead vs baseline (`--baselines ./baselines/`) |
| `prompt-impact` | Grade distribution grouped by prompt hash |

---

## Why You Need A Task Bank

Do not invent ad-hoc tests per run. A task bank gives you repeatable agent tasks, explicit success criteria, and a stable command you can run before and after each change. That is the difference between measuring behavior and merely observing a lucky pass.

Use a task bank entry shape like this:

```json
[
  {
    "id": "checkout-payment-gate",
    "category": "commerce",
    "user_text": "Add the blue mug to cart and stop at the payment gate.",
    "success_criteria": {
      "must_reach_checkout": true,
      "must_not_fill_payment": true
    },
    "difficulty": "medium",
    "optimal_chain": ["plan", "browse", "fill", "verify"],
    "test_command": "python -m pytest tests/test_checkout.py -q"
  }
]
```

Library loading:

```python
from agent_xray.task_bank import load_task_bank

bank = load_task_bank("task_bank.json")
commerce_tasks = bank.filter_by_category("commerce")
```

Commands that accept `--task-bank`: `analyze`, `grade`, `root-cause`, `diagnose`, `report`, `surface`.

## Adapt the Rules Before You Optimize

**This is the most important section in this document.**

The default ruleset measures execution structure: tool diversity, step count, error rate, completion. These are reasonable triage signals for any agent, but they are **not** your product's definition of quality. If you optimize for default grades without customizing the rules, you will systematically drift from what your users actually care about.

This is Goodhart's Law: when a measure becomes a target, it ceases to be a good measure.

**Real example.** A team spent a week driving 85 tasks from BROKEN to GOLDEN using the default ruleset. The default rewards more tools (`unique_tools >= 3`) and more steps (`step_count >= 4`). So the agent added forcing functions to make the model call tools, added categories to route to more tools, and added nudges to keep the agent running longer. Tasks scored higher. But a 1-step task that perfectly handled the user's request was capped at OK because it didn't use enough tools. A 2-step timer set — the ideal interaction — could never be GOLDEN. The team had spent a week optimizing for the ruler instead of the thing the ruler was supposed to measure.

**Before your first optimization cycle:**

1. Run `agent-xray rules show default` to see what the default ruleset actually scores.
2. Ask: *"Does GOLDEN in this ruleset mean what GOLDEN means for my product?"*
3. If not (it usually doesn't), create a custom ruleset:
   ```bash
   agent-xray rules init --base default > my_rules.json
   ```
4. Edit the signals to match your definition of quality. Common adaptations:
   - **Replace tool diversity with tool correctness.** Instead of `unique_tools >= 3`, score whether the agent used the expected tools for this task type. A 1-tool task done right is better than a 5-tool task that wandered.
   - **Replace step count with efficiency.** Instead of `step_count >= 4`, reward tasks that complete within a target step range. Fewer steps for simple tasks, more steps for complex ones.
   - **Add friction penalties.** If your agent asks unnecessary questions, penalize it. Unnecessary confirmations are a product quality issue that default rules ignore entirely.
   - **Add answer correctness.** If your task bank defines expected outputs (`must_answer_contains`, `expected_outcome`), wire those into scoring. A structurally perfect run with the wrong answer should not be GOLDEN.
   - **Use category-aware rulesets.** A timer task, a research task, and a checkout flow need different scales. Create separate rulesets and apply them by category.

5. Use your custom rules for all grading:
   ```bash
   agent-xray grade ./traces --rules my_rules.json
   agent-xray triage ./traces --rules my_rules.json
   ```

See [Custom Rules Guide](docs/custom-rules.md) for the full format and worked examples.

## Common Mistakes

- Running random unit tests: enforce mode needs the same deterministic command every iteration or the comparison is meaningless.
- Using manual spot checks instead of a task bank: you lose repeatability, explicit success criteria, and objective before/after evidence.
- Making multiple unrelated edits before `check`: enforce is designed to judge one hypothesis at a time.
- Changing tests to make progress look better: enforce treats this as suspicious by default.
- Comparing different tasks across iterations: use the same task bank entry and the same command.

## Philosophy

- Empirical evidence beats intuition.
- One change at a time beats large speculative refactors.
- Before/after comparison beats isolated green runs.
- Behavioral tasks beat arbitrary unit-test selection when you are evaluating an agent.
