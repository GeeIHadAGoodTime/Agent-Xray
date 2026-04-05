# Custom Rules Guide

`agent-xray` rules are JSON files that score analyzer metrics and detector metrics. This guide covers the file format, every supported operator, `golden_requirements`, and a worked example for coding agents.

## Why You Must Customize

Default rules are generic triage signals, not product quality definitions. If you use them as optimization targets, agents (human or AI) will optimize for structure instead of outcomes:

- Default rewards `unique_tools >= 3` → agents add unnecessary tool calls to inflate scores
- Default rewards `step_count >= 4` → agents add steps that don't help the user
- Default ignores answer correctness → structurally clean runs with wrong answers score GOLDEN
- Default ignores friction → agents that ask unnecessary questions aren't penalized

**Every product has a different definition of GOLDEN.** A coding agent's GOLDEN is "tests pass, code is clean." A voice assistant's GOLDEN is "handled it like a friend in minimal steps." A research agent's GOLDEN is "correct answer from credible sources." None of these map to the default's "used 3+ tools in 4+ steps with no errors."

Create your custom rules before running any optimization campaign. The format below gives you full control over what GOLDEN means for your product.

## Ruleset File Format

A ruleset is a single JSON object.

```json
{
  "name": "my_rules",
  "description": "Rules for my agent",
  "signals": [
    {
      "label": "tool_diversity",
      "field": "unique_tools",
      "op": "gte",
      "value": 3,
      "points": 1,
      "reason": "used multiple tools"
    }
  ],
  "grade_thresholds": {
    "GOLDEN": 5,
    "GOOD": 3,
    "OK": 1,
    "WEAK": 0
  },
  "golden_requirements": []
}
```

Top-level keys:

| Key | Required | Meaning |
| --- | --- | --- |
| `name` | Yes | Ruleset name shown in CLI output |
| `description` | No | Human-readable summary |
| `signals` | Yes | List of scoring rules |
| `grade_thresholds` or `thresholds` | Yes | Cutoffs for `GOLDEN`, `GOOD`, `OK`, and `WEAK` |
| `golden_requirements` | No | Extra gates that must pass before a run can stay `GOLDEN` |

Anything below `WEAK` becomes `BROKEN` automatically.

## Rule Object Format

Each item in `signals` is one scoring rule.

```json
{
  "name": "checkout_reached",
  "field": "commerce.reached_checkout",
  "op": "equals",
  "value": true,
  "points": 3,
  "else_points": 0,
  "reason": "+3 checkout reached",
  "else_reason": ""
}
```

Supported rule keys:

| Key | Required | Meaning |
| --- | --- | --- |
| `name` or `label` | No | Stable identifier for the rule |
| `field` or `metric` | Yes | Metric path to score |
| `op` | Yes for modern format | Operator name |
| `value` | Usually | Comparison value used by `op` |
| `points` | Yes | Points added when the rule passes |
| `else_points` | No | Points added when the rule fails |
| `reason` | No | Message emitted when the rule passes |
| `else_reason` | No | Message emitted when the rule fails |

`field` values can use dotted paths such as:

- `error_rate`
- `unique_tools`
- `commerce.reached_checkout`
- `coding.test_runs`
- `research.source_diversity`

Detector metrics are exposed both as nested maps and as flattened top-level keys. Use dotted paths when possible because they avoid collisions across detectors.

## Supported Operators

`grader.py` supports the following normalized operators:

| Operator | Meaning | Example |
| --- | --- | --- |
| `gte` | actual >= value | `{"field":"step_count","op":"gte","value":4}` |
| `gt` | actual > value | `{"field":"error_rate","op":"gt","value":0.5}` |
| `lte` | actual <= value | `{"field":"errors","op":"lte","value":1}` |
| `lt` | actual < value | `{"field":"avg_cost_per_step","op":"lt","value":0.05}` |
| `equals` | actual == value | `{"field":"commerce.reached_payment","op":"equals","value":true}` |
| `in` | actual is one of the provided values | `{"field":"site_name","op":"in","value":["shop","checkout"]}` |
| `contains_any` | actual list overlaps the provided list, or scalar actual equals one of the provided values | `{"field":"site_name","op":"contains_any","value":["localhost","shop"]}` |
| `ne` | actual != value | `{"field":"max_repeat_tool","op":"ne","value":"browser_snapshot"}` |
| `not_in` | actual is not in the provided values | `{"field":"site_name","op":"not_in","value":["unknown","no-navigation"]}` |

Notes:

- `in` expects the rule's `value` to be a list, tuple, or set.
- `contains_any` also expects a list, tuple, or set. It is most useful when your custom detector returns a list metric.
- If a rule fails and you do not supply `else_points`, it contributes `0`.

## Legacy Comparator Syntax

Older rulesets can omit `op` and `value` and instead use shorthand keys:

```json
{
  "name": "healthy_step_count",
  "metric": "step_count",
  "gte": 4,
  "points": 1
}
```

The grader normalizes these shorthands internally:

- `gte`
- `gt`
- `lte`
- `lt`
- `equals`
- `in`
- `contains_any`
- `ne`
- `not_in`

New rules should prefer the explicit `field` + `op` + `value` form because it is easier to validate and document.

## How To Create A New Ruleset

1. Pick the metrics you care about.

   Common sources:

   - core analyzer metrics such as `step_count`, `errors`, `error_rate`, `unique_tools`
   - detector metrics such as `commerce.reached_checkout`, `coding.test_runs`, `research.source_diversity`

2. Write a JSON file.

   You can put it anywhere, but the built-in convention is `src/agent_xray/rules/<name>.json`.

3. Define grade thresholds.

   `GOLDEN`, `GOOD`, `OK`, and `WEAK` are required. Scores below `WEAK` become `BROKEN`.

4. Run it.

```bash
agent-xray grade ./traces --rules ./src/agent_xray/rules/my_rules.json
```

5. Adjust based on real traces.

   If the score bands feel too generous or too harsh, fix the thresholds first. If the signals are wrong, fix the rules or add better detector metrics.

## How `golden_requirements` Work

`golden_requirements` are extra checks applied only after a task already scores high enough to be `GOLDEN`.

If any requirement fails:

- the run is downgraded from `GOLDEN` to `GOOD`
- the failure reason is appended to the result

Two formats are supported.

### String requirements

String requirements reference a rule result by its effective rule name. The effective name is chosen in this order:

1. `name`
2. `label`
3. `field` or `metric`

Example:

```json
{
  "golden_requirements": ["tests_written", "edit_test_cycle"]
}
```

This is the pattern used by the bundled [`coding_agent.json`](../src/agent_xray/rules/coding_agent.json) ruleset.

### Dict requirements

Dict requirements look like regular rules but do not add points. They only gate `GOLDEN`.

Example:

```json
{
  "golden_requirements": [
    {
      "field": "commerce.payment_fields_confirmed",
      "op": "equals",
      "value": true,
      "reason": "payment details were not visibly confirmed"
    }
  ]
}
```

This is the pattern used by the bundled [`browser_flow.json`](../src/agent_xray/rules/browser_flow.json) ruleset.

## Example: Rules For A Coding Agent

The built-in coding ruleset is a good starting point because it already scores edit-test behavior, linting, error count, and tool diversity.

Example:

```json
{
  "name": "coding_strict",
  "description": "Strict grading rules for a code-editing agent",
  "signals": [
    {
      "label": "tests_written",
      "field": "coding.test_to_edit_ratio",
      "op": "gte",
      "value": 0.3,
      "points": 3,
      "reason": "ran tests relative to edits"
    },
    {
      "label": "edit_test_cycle",
      "field": "coding.has_test_verify_cycle",
      "op": "equals",
      "value": true,
      "points": 2,
      "reason": "completed an edit-and-test cycle"
    },
    {
      "label": "ran_linter",
      "field": "coding.lint_runs",
      "op": "gte",
      "value": 1,
      "points": 1,
      "reason": "ran linting"
    },
    {
      "label": "high_error_rate",
      "field": "error_rate",
      "op": "gte",
      "value": 0.5,
      "points": -3,
      "reason": "too many failing steps"
    },
    {
      "label": "no_files_touched",
      "field": "coding.file_operations",
      "op": "equals",
      "value": 0,
      "points": -3,
      "reason": "claimed coding work without touching files"
    }
  ],
  "grade_thresholds": {
    "GOLDEN": 7,
    "GOOD": 4,
    "OK": 1,
    "WEAK": -2
  },
  "golden_requirements": ["tests_written", "edit_test_cycle"]
}
```

Run it with:

```bash
agent-xray grade ./traces --rules ./coding_strict.json
```

## Recommendations

- Start with the bundled ruleset closest to your domain and edit that, rather than writing a new file from scratch.
- Prefer dotted metric paths like `coding.test_runs` over ambiguous flat names.
- Use `golden_requirements` for must-have behaviors such as "tests were actually run" or "payment was visibly confirmed."
- Add new detector metrics when the rules feel forced. The grader is intentionally simple; the analyzer and detectors are where better signals should live.
