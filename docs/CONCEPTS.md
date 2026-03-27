# Concepts

## Information Surfaces

An agent decision is made from a surface, not from one field. `agent-xray` treats these as first-class:

- prompt
- tools available
- reasoning
- conversation history
- tool result
- context window state
- injected corrections
- page URL and browser cues

## Grading

The grader does not assume one benchmark. It computes task metrics once, then applies any JSON rules file that references those metrics.

Useful metrics include:

- `step_count`
- `unique_urls`
- `unique_tools`
- `error_rate`
- `real_fill_count`
- `reached_cart`
- `reached_checkout`
- `reached_payment`
- `max_repeat_count`

## Root Causes

Root-cause labels are heuristics meant to answer: where should the engineer look first?

- operational: routing, approval, tool, runner, environment
- behavioral: tool selection, stuck loop, reasoning, prompt
- fundamental: model limit

## Golden Runs

A golden run is not just a successful outcome. It is a reusable structural trace:

- ordered milestones
- rough step budget
- expected page or result markers
- sanitized task and tool payloads
