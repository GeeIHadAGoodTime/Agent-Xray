# agent-xray Onboarding

`agent-xray` is a local-first debugger and referee for AI-agent behavior. It reconstructs what the agent actually saw, grades what happened, and, in enforce mode, tells you whether a proposed code change genuinely improved the task. Treat it as an evidence tool, not a vibes tool.

## Quick Start

The fastest path from zero to insight:

```bash
pip install agent-xray
agent-xray analyze my-traces.jsonl
```

That is it. `analyze` loads a trace file, reconstructs the agent timeline, and prints a summary of what happened: tool calls, decisions, failures, and timing. No configuration, no task bank, no enforce setup. Start here.

If you already have traces from a specific task and want to drill into one:

```bash
agent-xray surface my-traces.jsonl --task checkout-payment-gate
```

Once you are comfortable with these two commands, move to the full analysis workflow below.

## Analysis Workflow

Analysis is a progressive deepening pipeline. Each step reveals more detail than the last. Run them in order on the same trace file, stopping when you have enough evidence.

### Step 1: Load and summarize

```bash
agent-xray analyze traces.jsonl
```

Reconstructs the full agent timeline. Shows tool call sequence, step count, timing, and high-level outcome. This is your first look at what actually happened.

### Step 2: Grade against rules

```bash
agent-xray grade traces.jsonl --rules browser_flow
```

Applies a named rule set to the trace and produces a pass/fail verdict per rule. Use built-in rule sets (`browser_flow`, `payment_safety`, `tool_usage`) or point to your own. Grading turns a vague "it looked wrong" into a specific "rule X failed at step Y."

**High-value path — expectation-aware grading:**

```bash
agent-xray grade traces.jsonl --rules browser_flow --task-bank task_bank.json
```

When you provide a task bank, grading matches each logged task to its bank entry and evaluates the defined success criteria (must_reach_url, must_answer_contains, payment_fields_visible, max_steps, etc.). This is the difference between "did the agent navigate?" (generic) and "did the agent reach the Wisconsin WDFI site and discuss registered agents?" (expectation-aware). Without `--task-bank`, you only get generic signal-based grading.

### Step 3: Root-cause analysis

```bash
agent-xray root-cause traces.jsonl
```

Identifies the earliest divergence point: where did the agent first go wrong? The output traces backward from the failure to the decision that caused it, filtering out downstream symptoms.

### Step 4: Diagnose

```bash
agent-xray diagnose traces.jsonl
```

Produces structured diagnostics: tool availability gaps, approval blocks, prompt mismatches, and timing anomalies. Use `--json` for machine-readable output. This is the command to reach for when root-cause points at infrastructure rather than agent logic.

### Step 5: Surface a specific task

```bash
agent-xray surface traces.jsonl --task <task-id>
```

Extracts and displays every detail for one task: the full tool call chain, LLM inputs/outputs, intermediate state, and the final outcome. Use this when you need the complete picture for a single execution.

### Step 6: Compare across runs

```bash
agent-xray compare day1.jsonl day2.jsonl
```

Side-by-side diff of two trace files. Shows what changed between runs: new failures, resolved failures, timing regressions, and tool call differences. This is how you measure whether a change actually helped.

### Typical progression

Most investigations follow this pattern:

```
analyze  -->  "step 14 failed"
grade    -->  "browser_flow rule 3: no snapshot before click"
root-cause --> "agent skipped snapshot because tool was unavailable"
diagnose -->  "tool_available_count: 0 at step 12 due to approval block"
surface  -->  full detail confirms the approval bridge was not wired
compare  -->  after fix, day2 shows rule 3 passing
```

You do not need all six steps every time. Many issues are clear after `analyze` + `grade`. Reach for deeper commands only when the simpler ones leave questions open.

## The Enforce Workflow

Enforce mode is for iterative code improvement. Once you understand what went wrong (using the analysis workflow above), enforce mode helps you fix it with discipline: one hypothesis, one change, one measurement.

Use the same loop every time:

1. `init`: capture the baseline with one deterministic test command.
2. `plan`: write one hypothesis and the tests you expect to move.
3. `change`: make one small change that matches the hypothesis.
4. `check`: run the exact same command against the changed tree.
5. `iterate`: keep or revert based on the before/after evidence, then repeat.

Example:

```bash
agent-xray enforce init --test "python -m pytest tests/ -x -q --tb=short"
agent-xray enforce plan --hypothesis "checkout timeout is caused by a missing await" --expected-tests tests/test_checkout.py::test_checkout_flow
# make one code change
agent-xray enforce check
agent-xray enforce challenge
agent-xray enforce report --format markdown
```

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
