# Reddit Launch Posts

## r/MachineLearning

**Tag**: `[P]` (Project)

**Title**:
```
[P] agent-xray: Open-source structural grading and root-cause classification for AI agent traces
```

**Body**:
```
I built an open-source tool for debugging AI agent failures — the kind that produce no error logs, where the agent just does the wrong thing.

**Problem**: Existing observability tools (LangSmith, Langfuse, Arize Phoenix) show traces and latencies, but when an agent spins on the same action or gives up after one error, there's no automated way to detect and classify that failure pattern.

**Approach**: agent-xray reads agent trace logs (JSONL from any framework) and:
1. Assigns structural execution grades (GOLDEN/GOOD/OK/WEAK/BROKEN) using configurable JSON rulesets
2. Classifies failures into 22 root-cause categories (spin, tool_bug, early_abort, routing_bug, etc.) using a cascade heuristic classifier
3. Reconstructs the "decision surface" at each step — what the agent saw, what tools it had, what it was reasoning about
4. Compares runs before/after a fix to quantify impact across all tasks

**Design choices**:
- Grades measure execution structure (tool diversity, loop resistance, error rate), NOT output correctness. The tool is explicit about this distinction.
- The root-cause classifier is heuristic-based, not ML-based. It's a cascade of pattern checks. Simple but systematic.
- Local-first, zero required dependencies, fully offline. No account, no telemetry.
- Framework-agnostic: adapters for OpenAI, Anthropic, LangChain, CrewAI, OpenTelemetry.

**Limitations**:
- Heuristic grading can be wrong — it's a triage signal, not a verdict
- Root-cause classification is shallow pattern matching, not deep causal analysis
- Tested primarily on browser-based agent tasks; may need ruleset tuning for other domains

**Try it**:
```
pip install agent-xray
agent-xray quickstart
```

GitHub: https://github.com/GeeIHadAGoodTime/Agent-Xray
MIT licensed.

Happy to discuss the approach, limitations, or how it compares to other tools in the space.
```

---

## r/LangChain

**Title**:
```
Open-source tool for debugging LangChain/LangGraph agent failures — structural grading + root-cause classification
```

**Body**:
```
I built agent-xray to debug agent failures that don't produce error logs. It reads your existing trace files and:

- Grades each execution structurally (GOLDEN → BROKEN)
- Classifies failure mode (spin, tool_bug, early_abort, etc.)
- Reconstructs what the agent saw at each decision point
- Has a pytest plugin for CI quality gates

Works with LangChain/LangGraph traces out of the box (`--format langchain`). Also supports OpenAI, Anthropic, CrewAI, and OpenTelemetry.

Quick start:
```
pip install agent-xray
agent-xray quickstart
```

It's complementary to LangSmith — use LangSmith to collect traces, use agent-xray to debug specific failures locally.

GitHub: https://github.com/GeeIHadAGoodTime/Agent-Xray
```

---

## r/LocalLLaMA (good fit — local-first tool)

**Title**:
```
agent-xray: Debug your AI agents locally — structural grading, root-cause classification, zero dependencies, fully offline
```

**Body**:
```
Built an open-source agent debugging tool that runs entirely locally. No cloud, no account, no telemetry.

It reads agent trace logs and tells you: this task is BROKEN because the agent was spinning (calling the same tool 4 times with no progress). Here's what the agent saw. Here's what to fix.

- 22 root-cause categories
- Configurable grading rulesets
- pytest plugin for CI quality gates
- 37 MCP tools (Claude Code / Cursor can debug your agents)
- Zero required dependencies

```
pip install agent-xray
agent-xray quickstart
```

GitHub: https://github.com/GeeIHadAGoodTime/Agent-Xray
MIT licensed, Python 3.10+.
```

---

## Additional Subreddits to Consider

| Subreddit | Fit | Notes |
|-----------|-----|-------|
| r/SideProject | Good | Explicitly welcomes self-promotion |
| r/OpenSource | Good | Open-source project showcase |
| r/Python | Medium | Post in weekly "What's New" thread |
| r/coolgithubprojects | Good | Designed for GitHub project discovery |
| r/ArtificialIntelligence | Medium | Broader audience, less technical |

## Posting Order

1. r/MachineLearning (largest, most credible audience)
2. r/LangChain (direct user base)
3. r/LocalLLaMA (local-first angle)
4. r/SideProject + r/OpenSource + r/coolgithubprojects (discovery)

Space posts 1-2 days apart to avoid looking spammy.
