# Hacker News Launch

## Submission

**Title** (pick one):

Option A (problem-first — historically best performing on HN):
```
Show HN: agent-xray – Debug AI agent failures that produce no error logs
```

Option B (technique-first):
```
Show HN: agent-xray – Structural grading and root-cause classification for AI agents
```

Option C (contrarian — matches the 102-point "you don't need new tools" pattern):
```
Show HN: Your AI agent tracing tool shows what happened, not why it failed
```

**URL**: https://github.com/GeeIHadAGoodTime/Agent-Xray

## First Comment (post immediately after submission)

```
Hi HN — I built agent-xray because my AI agent was failing 30% of tasks and I couldn't figure out why from the traces.

Existing tools (LangSmith, Langfuse, AgentOps) show you what happened — traces, spans, latencies. But when your agent just does the wrong thing, there's no stack trace. You're reading raw JSON trying to figure out what went wrong.

agent-xray reconstructs the decision surface at each step (what was in the prompt, what tools were available, what the model was reasoning about) and classifies failures into 22 root-cause categories (spin, tool_bug, early_abort, routing_bug, etc.).

One command triages all your tasks:

    pip install agent-xray
    agent-xray triage ./traces

It also has a pytest plugin so you can assert on agent execution quality in CI:

    def test_checkout_agent(xray):
        report = xray.analyze(steps)
        assert report.grade in ("GOLDEN", "GOOD")

Key design choices:
- Local-first, fully offline, zero required dependencies
- Framework-agnostic (OpenAI, Anthropic, LangChain, CrewAI, OpenTelemetry)
- 37 MCP tools so your AI coding assistant can debug your AI agent
- Grades measure execution structure, not output correctness (it's honest about this)
- MIT licensed

It's complementary to production tracing tools — use LangSmith to collect, use agent-xray to debug.

Happy to answer questions about the approach, the grading heuristics, or what I learned debugging 350+ agent tasks.
```

## Timing

Best HN submission times (US-centric audience):
- Tuesday-Thursday, 8-10am ET (highest front-page probability)
- Avoid weekends and Monday mornings

## If It Gets Traction

Be ready to answer:
1. "How is this different from LangSmith?" → Complementary. LangSmith collects at scale. agent-xray debugs locally. Think pytest vs Sauce Labs.
2. "Why not just use LLM-as-judge?" → Structural grades are deterministic and fast. LLM-as-judge is expensive and non-deterministic. Use both.
3. "22 root causes sounds like heuristics" → They are. But systematic heuristics across 350 tasks beat manual analysis. The tool is honest that grades measure structure, not correctness.
4. "Zero stars, should I trust this?" → Fair concern. The tool is 9 days old. Try `agent-xray quickstart` — it ships sample data. Judge by output, not stars.
