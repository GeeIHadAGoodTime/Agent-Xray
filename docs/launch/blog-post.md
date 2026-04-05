# My AI Agent Was Failing 30% of Tasks. Here's How I Found Out Why.

LLM agents fail silently. There are no stack traces. No error logs. The agent just does the wrong thing and you're left reading raw traces trying to figure out what happened.

I was running a voice assistant with a browser agent handling 350+ tasks — shopping, scheduling, research, smart home. About 30% were failing, and I couldn't figure out why. The traces showed me what tools the agent called, but not why it made bad decisions.

Existing observability tools (LangSmith, Langfuse) show you **what happened** — traces, spans, latencies. I needed to know **why** — what was in the prompt, what tools were available, what the model was reasoning about, and where the decision went wrong.

So I built [agent-xray](https://github.com/GeeIHadAGoodTime/Agent-Xray).

## One command to find the worst failure

```bash
$ agent-xray triage ./traces

Tasks: 358 | GOLDEN: 9 | GOOD: 64 | OK: 198 | WEAK: 7 | BROKEN: 30

Worst task: c8ca3f576c61
  User: "Get directions from Milwaukee to Madison on Google Maps"
  Grade: BROKEN (score: -3)
  Root cause: spin
  Evidence: agent called web_search 4 times despite error messages
           explicitly saying "Use browser_navigate"

Fix plan:
  1. tool_bug (severity 4/5, 12 tasks, impact 72)
     → 4 broken tools at 100% failure rate
  2. early_abort (severity 3/5, 12 tasks, impact 36)
     → agent giving up after first error
  3. spin (severity 3/5, 5 tasks, impact 25)
     → repeating same action without progress
```

One command. 358 tasks triaged. Worst failure surfaced with root cause and a prioritized fix plan.

## Replaying the decision surface

The triage told me the agent was spinning. But why? I replayed the decision surface — what the agent actually saw at each step:

```bash
$ agent-xray surface c8ca3f576c61 --log-dir ./traces
```

Step by step, I could see: the agent had `browser_navigate` available in its tool set. The error message after each `web_search` call explicitly said "Use browser_navigate to go to the URL." But the agent kept calling `web_search` anyway.

The root cause wasn't a missing tool or a permission issue. The agent had everything it needed. It was a **reasoning bug** — the model pattern-matched "find directions" → "search for directions" instead of "navigate to Google Maps."

Without the decision surface reconstruction, I would have stared at raw JSON for an hour to figure that out.

## Structural grades, not output grades

agent-xray grades execution **structure** — tool diversity, loop resistance, error rate, completion — not whether the output was correct. This is a deliberate design choice:

- A `GOLDEN` task has clean execution but could still have the wrong answer
- A `BROKEN` task with spinning and errors might have eventually stumbled into the right result

The grades are triage signals, not verdicts. They tell you where to look. The decision surface tells you what actually happened.

## 22 ways an agent can fail

The root-cause classifier identifies 22 distinct failure modes:

| Common failures | What they mean |
|---|---|
| `spin` | Repeating the same action without progress |
| `tool_bug` | Right tool called, but it returned errors |
| `early_abort` | Gave up after first obstacle |
| `routing_bug` | Never got the right tools |
| `reasoning_bug` | Had the tools, made bad decisions |
| `stuck_loop` | Acting without meaningful state change |
| `approval_block` | Permission gates blocked progress |

Each classification comes with evidence from the trace and a fix hint pointing at the right subsystem.

## Quality gates in CI

The part that actually changed my workflow: a pytest plugin that asserts on agent execution quality in CI.

```python
def test_checkout_agent(xray):
    steps = run_my_agent("Buy the blue mug on demo-shop.example.test")
    report = xray.analyze(steps)
    assert report.grade in ("GOLDEN", "GOOD")
    assert report.error_rate < 0.1
```

Now when a code change degrades agent quality, CI catches it. No dashboard needed.

## Before and after

After fixing the top 3 root causes (tool bugs, early aborts, spins):

```bash
$ agent-xray compare ./traces-before ./traces-after

Grade shifts:
  BROKEN → OK:    12 tasks improved
  BROKEN → GOOD:   3 tasks improved
  OK → GOOD:       8 tasks improved
  GOOD → GOLDEN:   4 tasks improved

  GOOD → OK:       1 task regressed  ← caught this too
```

One command quantifies the impact across all tasks — not just the ones I fixed.

## Try it

```bash
pip install agent-xray
agent-xray quickstart
```

The quickstart ships sample traces and runs a full walkthrough: grade, surface, health report.

Works with OpenAI, Anthropic, LangChain, CrewAI, and OpenTelemetry traces. Zero dependencies for the core library. Fully offline — no account, no telemetry.

**GitHub**: https://github.com/GeeIHadAGoodTime/Agent-Xray
**PyPI**: https://pypi.org/project/agent-xray/

---

*agent-xray is MIT-licensed and open source. It's designed to complement production tracing tools like LangSmith and Langfuse — use them to collect traces at scale, use agent-xray to deeply debug why a specific decision went wrong.*
