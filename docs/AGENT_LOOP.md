# Agent Loop

Use this loop when you want to improve an agent with evidence from step logs instead of guesswork.

1. Grade recent runs.
2. Inspect the worst task with `surface` and `reasoning`.
3. Name the failing information surface.
4. Make one targeted fix.
5. Re-run the task.
6. Capture the new golden fixture if the fix holds.
7. Replay the fixture against later runs to catch regressions.

## Investigation Questions

- Did the model have the right tool available?
- Did the prompt point it toward the right action?
- Did the tool return a recoverable result?
- Did context pressure strip out something important?
- Was the agent spinning even after receiving corrective signals?

## Stop Conditions

- The task moved from `BROKEN` or `WEAK` into `GOOD` or `GOLDEN`.
- Repeated attempts do not improve the same root cause.
- The remaining issue is clearly a model or environment limit.
