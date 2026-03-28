# agent-xray Onboarding

`agent-xray` is a local-first debugger and referee for AI-agent behavior. It reconstructs what the agent actually saw, grades what happened, and, in enforce mode, tells you whether a proposed code change genuinely improved the task. Treat it as an evidence tool, not a vibes tool.

## The Enforce Workflow

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

## Quick Start

For trace inspection:

```bash
agent-xray quickstart
agent-xray analyze ./traces
agent-xray surface broken-task --log-dir ./traces
agent-xray diagnose ./traces --json
```

For an agentic code-fix loop:

```bash
agent-xray enforce init --test "python -m pytest tests/ -x -q --tb=short"
agent-xray enforce plan --hypothesis "one sentence, one expected fix"
agent-xray enforce diff
agent-xray enforce check
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
