# agent-xray Roadmap

Last updated: 2026-04-05

## Current State (v1.25.4)

### Verified Inventory

| Component | Count | Status |
|-----------|-------|--------|
| MCP tools (`@server.tool`) | 37 | Verified by decorator count |
| CLI commands | 32 | Verified by subcommand audit |
| Root cause categories | 22 | Verified from `root_cause.py` |
| Grade levels | 5 | GOLDEN / GOOD / OK / WEAK / BROKEN |
| Signal detector packs | 6 | commerce, coding, research, planning, memory, multi_agent |
| Format adapters | 7 | generic, openai_sdk, openai_chat, anthropic, langchain, crewai, otel |
| Report types | 16 | health, golden, broken, tools, flows, outcomes, actions, coding, research, cost, fixes, timeline, spins, overhead, prompt-impact, compare |
| Built-in rulesets | 5 | default, simple, browser_flow, coding_agent, research_agent |
| Gaming detectors | 8 | test_modification, hardcoded_values, special_case, mock_insertion, assertion_weakening, exception_swallowing, early_return, import_removal |
| Enforce verbs | 9 | init, check, quick, plan, diff, guard, status, challenge, reset |
| Test files | 61 | 1,471 collected tests |
| Source LOC | ~23,300 | 56 modules |
| Test LOC | ~21,800 | 61 files |

### What Works

- **Core workflow**: triage → surface → grade → root_cause → inspect — verified producing correct output
- **Quickstart**: ships 5 real trace files + synthetic fallback, runs automatically
- **TUI**: Textual-based step-by-step inspector with keybindings
- **pytest plugin**: `xray` fixture registered via `pytest11` entry point
- **Examples**: 4 self-contained framework integration scripts
- **Zero dependencies**: core library has `dependencies = []`
- **CI**: lint (ruff) + typecheck (mypy --strict) + test (3 OS × 4 Python versions)
- **Onboarding**: ONBOARDING.md with quickstart → analyze → grade → surface progression

### Documentation Discrepancies Fixed (this audit)

| What | Was | Now |
|------|-----|-----|
| MCP tool count (README, CAPABILITIES, CLAUDE.md) | 49 / 48 | 37 |
| Root cause category count (README) | 19 | 22 |
| Root cause category count (CAPABILITIES) | 17 | 22 |
| CAPABILITIES version | 1.25.3 | 1.25.4 |
| Missing root causes in README table | context_overflow, rate_limit_cascade, timeout missing | Added |

---

## Phase 0: Distribution (1 afternoon — highest impact)

The tool is ready. Nobody knows it exists. This phase is the minimum viable experiment to test whether the audience cares.

### P0.1 — GitHub Discoverability

- [ ] Set topics: `ai`, `agents`, `debugging`, `observability`, `mcp`, `llm`, `evaluation`, `tracing`
- [ ] Create GitHub Release for v1.25.4 with release notes
- [ ] Set homepage URL (PyPI link or future docs site)
- [ ] Add social preview image (terminal screenshot of triage output)
- [ ] Update repo description to lead with the value prop, not the feature list

**Effort**: 30 minutes. **Impact**: Shows up in GitHub search/explore, release feeds.

### P0.2 — First Public Post

- [ ] Write one blog post / README-based walkthrough: "Debugging AI Agent Failures with Structural Grading"
  - Show a real before/after: BROKEN task → root cause → fix → GOLDEN
  - Include triage output, surface output, compare_runs output
  - End with `pip install agent-xray && agent-xray quickstart`
- [ ] Submit to Hacker News (Show HN)
- [ ] Post to r/MachineLearning and r/LangChain

**Effort**: 2-3 hours. **Impact**: This is the demand test. If it gets traction, the roadmap shifts. If it doesn't, we know the market signal.

### P0.3 — Lead with the pytest Plugin

- [ ] Move pytest plugin section higher in README (currently line 456 — should be in first 100 lines)
- [ ] Add a "CI Quality Gates" section showing how to fail CI on BROKEN agents
- [ ] Example GitHub Actions workflow snippet for agent quality CI

**Effort**: 1 hour. **Impact**: Lowest-friction adoption path — developers already run pytest.

---

## Phase 1: Quality & Trust (1-2 weeks)

These items fix real issues that would erode trust if someone evaluates the tool seriously.

### P1.1 — Version Hygiene

- [ ] Decide: reset to 0.x (honest about maturity) OR commit to semver discipline going forward
- [ ] Batch future releases to weekly cadence max
- [ ] Add `scripts/validate_release.py` to prevent version drift

**Rationale**: 49 versions in 9 days signals instability. This is the #1 trust barrier for external developers.

### P1.2 — Test Gaps (Critical)

| Gap | Current | Target | Impact |
|-----|---------|--------|--------|
| Enforce engine | 0 dedicated tests (973 lines of fixtures, 0 test functions) | 20+ tests | Enforce is a marquee feature with zero test coverage |
| TUI | 3 import tests only | 10+ functional tests | Users will open the TUI — it needs to work |
| Analyzer | No `test_analyzer.py` (60+ functions) | 15+ unit tests | Core analysis engine tested only indirectly |
| Watch mode | No tests (200+ lines) | 5+ tests | Live monitoring untested |
| contrib/ | No tests for novviola.py or task_bank.py | 10+ tests | Plugin system untested |

### P1.3 — V1 Roadmap Critical Path Items (from `docs/V1_ROADMAP.md`)

These block a credible v1.0 tag:

1. **Schema round-trip bug** — `TaskOutcome.from_dict()` double-nests metadata
2. **Adapter autodetect** — silently misclassifies real traces as `generic`
3. **Grading scores not comparable** — GOLDEN unreachable in `default.json`
4. **Coding signal false positives** — `FILE_PATH_RE` matches version strings
5. **CLI error handling** — Python tracebacks instead of user-friendly messages

### P1.4 — `simple.json` Ruleset Documentation

- [ ] Document in README alongside other rulesets
- [ ] Add to ONBOARDING.md quick reference

---

## Phase 2: Adoption Wedges (1-2 months)

Based on what we learned about the competitive landscape, these are the highest-leverage features for differentiation.

### P2.1 — GitHub Actions Integration

```yaml
# .github/workflows/agent-quality.yml
- run: pip install agent-xray
- run: agent-xray grade ./traces --rules my_rules.json --fail-on BROKEN
```

- [ ] Add `--fail-on` flag to CLI (exit code 1 if any task matches grade)
- [ ] Publish a reusable GitHub Action
- [ ] Write "Agent Quality CI" tutorial

**Rationale**: This is the pytest plugin's natural extension. If developers can add agent quality gates to CI in 5 lines, that's the lowest-friction adoption path.

### P2.2 — Interactive Demo (no install required)

- [ ] GitHub Codespace / Gitpod config with pre-installed agent-xray + sample traces
- [ ] One-click "try it" badge in README

**Rationale**: Removes the `pip install` barrier entirely for evaluation.

### P2.3 — Docs Site

- [ ] MkDocs with mkdocs-material
- [ ] Sections: Getting Started, Tutorials, Guides, Reference, Architecture
- [ ] 5-minute tutorial: install → quickstart → triage → surface → root_cause
- [ ] Set as GitHub Pages + homepage URL

### P2.4 — Broader Adapter Coverage

Priority based on framework popularity:

1. **OpenAI Agents SDK** (high — OpenAI's official agent framework)
2. **LangGraph v2** (high — increasingly used over raw LangChain)
3. **AutoGen** (medium — Microsoft's agent framework)
4. **Google ADK** (medium — growing)

---

## Phase 3: Competitive Moat (3-6 months)

Features that widen the gap vs. eventual platform competition.

### P3.1 — Web Dashboard (Optional SaaS Layer)

- [ ] Lightweight web UI for team trace viewing (complement to CLI/MCP)
- [ ] Share triage results, surface views, grade trends with team
- [ ] Self-hostable (Docker) with optional hosted tier

**Rationale**: This is what every competitor has that agent-xray doesn't. Not required for developer adoption, but required for team adoption. Also the only monetization path.

### P3.2 — LLM-Powered Root Cause (Beyond Heuristics)

- [ ] Optional LLM pass for ambiguous classifications (the 22-category heuristic handles 80%, LLM handles the 20% it can't)
- [ ] Use the decision surface as context for the LLM to reason about failure
- [ ] Keep fully optional — local-first principle preserved

**Rationale**: The current heuristic classifier is "correct but shallow" per our audit. LLM-powered analysis on the reconstructed decision surface could produce genuinely deeper insights than any competitor.

### P3.3 — Real-Time MCP Instrumentation

- [ ] MCP proxy that instruments agent traces in real-time and grades as they happen
- [ ] Live `watch` mode with grade updates as tasks complete
- [ ] Alert on BROKEN tasks via webhook

### P3.4 — Framework-Specific Rulesets Library

- [ ] Community-contributed rulesets for common agent patterns
- [ ] Published as separate package or ruleset registry
- [ ] `agent-xray rules install browser-commerce` pattern

---

## Success Metrics

### Phase 0 (Distribution Test)

| Metric | Target | Timeframe |
|--------|--------|-----------|
| HN upvotes | > 20 | 1 week after post |
| GitHub stars | > 100 | 2 weeks after post |
| PyPI downloads (excluding mirrors) | > 50/day sustained | 2 weeks after post |
| GitHub issues from external users | > 3 | 1 month |

If Phase 0 metrics are met → proceed to Phase 2 with confidence.
If Phase 0 metrics are NOT met → re-evaluate positioning, try different angle (pytest plugin first?), or accept niche-tool status.

### Phase 1 (Quality)

| Metric | Target |
|--------|--------|
| Test coverage | > 80% branch |
| V1 critical path items closed | 5/5 |
| Version cadence | < 2 releases/week |

### Phase 2 (Adoption)

| Metric | Target |
|--------|--------|
| GitHub stars | > 500 |
| Monthly active PyPI installs (non-mirror) | > 500 |
| External PRs | > 5 |
| Docs site page views | > 1K/month |

---

## What NOT to Build

Based on the competitive audit, these are traps:

1. **Full SaaS trace collection platform** — This is LangSmith/Langfuse territory. $500M+ in incumbent funding. Don't compete on ingestion at scale.
2. **400+ framework integrations** — AgentOps path. agent-xray's value is depth of analysis, not breadth of integration.
3. **Real-time production monitoring** — Wrong paradigm for a local-first tool. Stay in the "debug and CI" lane.
4. **Pricing/billing infrastructure** — Premature until Phase 0 validates demand.

agent-xray's wedge is: **the deepest forensic analysis of agent failures, runnable locally in one command.** Everything on this roadmap should sharpen that wedge, not dilute it.
