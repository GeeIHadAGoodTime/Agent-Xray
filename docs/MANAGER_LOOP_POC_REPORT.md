# Does agent-xray Work? A 12-Hour Field Evaluation

**Date**: 2026-03-30
**Duration**: ~12 hours (2026-03-29 22:48 CDT -- 2026-03-30 10:30 CDT)
**Subject**: NOVVIOLA -- a voice-first AI assistant with 85 task bank entries (web, calendar, music, memory, browser, email, phone, smart home)
**Question**: Is agent-xray useful for finding and fixing bugs in LLM agent systems? Is it worth adopting?

---

## TL;DR

Over 12 hours, a Claude Code agent used agent-xray to debug NOVVIOLA's agent behavior. It produced **25 bug fixes** in NOVVIOLA. Of those, **71% (24/34 commits) were directly informed by agent-xray trace analysis** -- bugs found by replaying what the LLM agent saw, did, and decided at each step. The 5 highest-impact fixes (affecting 57, 42, 38, 15, and 10 tasks respectively) were behavioral failures that produce **no error logs** -- silent bugs where the agent does the wrong thing without crashing. These are the bugs agent-xray is built to find.

Agent quality improved from a baseline dominated by BROKEN traces to **11 GOLDEN, 63 GOOD, 181 OK** across 293 graded tasks. The realistic estimate: agent-xray provided a **2-3x productivity multiplier** for evidence-based debugging compared to manual log analysis.

**Caveat**: agent-xray was modified 42 times during the evaluation (v1.14.0 → v1.25.3). This is a methodological impurity. We quantify its impact below.

---

## 1. What We Tested

### The Codebase

NOVVIOLA is a voice AI assistant with an internal LLM agent (gpt-4.1-nano) that handles user commands via browser automation, calendar, music, memory, email, phone, and smart home tools. It has 85 task bank entries -- standardized test scenarios like "order a pizza from Dominos," "schedule a meeting for Friday," "create a playlist with these songs."

The agent's behavior is captured in structured step logs (JSONL) -- every tool call, every LLM response, every decision the model made. These logs are what agent-xray analyzes.

### The Setup

One Claude Code agent ("viper") ran a debugging campaign, using agent-xray as a read-only diagnostic instrument. A second agent ("manager/apex") ran hourly audit cycles evaluating the first agent's tool usage and fixing agent-xray bugs it discovered. The two agents shared a filesystem and coordinated through a blackboard.

### The Grading System

agent-xray grades every task trace as GOLDEN (perfect), GOOD (effective), OK (completed), WEAK (partially failed), or BROKEN (failed). This taxonomy gives a shared vocabulary for quality measurement -- instead of "it seems to work better," you get specific grade transitions tied to specific commits.

---

## 2. What agent-xray Found

### The 5 Bugs That Mattered Most

These are the fixes that each affected dozens of tasks. For each: what was the bug, how did agent-xray surface it, and could it have been found without it?

#### Bug 1: Silent tool_choice bypass (~57 tasks affected)

**What**: `openai_compatible.py:691` defaulted to `tool_choice="auto"` in agent mode. gpt-4.1-nano would return text answers instead of calling tools for actionable tasks -- web searches, calendar operations, file lookups.

**Symptom**: Viola answers "Here's what I'd suggest..." instead of actually doing the thing. No errors. No crashes. The model successfully returns text; the system considers the task complete.

**How agent-xray found it**: `triage` flagged tasks TB-011, TB-023, TB-024 as BROKEN with zero tool calls. `surface_task` replayed the full decision chain -- the LLM had tools available but never invoked them. This pointed directly at `tool_choice` as the culprit. A 7-point pipeline trace from `pipeline.py` → `ai_controller.py` → `openai_compatible.py` confirmed the missing parameter.

**Without agent-xray**: Very hard. There are no errors to find. The only signal is behavioral -- Viola gives text answers when it should act. You'd need to systematically test tasks, notice the pattern, then trace through 4 layers of code to find a missing parameter 7 layers deep in the call chain. The bug is invisible to log-based debugging because nothing fails.

#### Bug 2: 3-way spin detection failure (~38-44 tasks affected)

**What**: Three independent bugs conspired to make spin detection never fire on `browser_click_ref`: (1) signature mismatch between `record_success` and `is_call_blocked` -- one included the page URL, the other didn't, so blocking never matched; (2) fingerprint too brittle -- dynamic content (timestamps, ad noise) made page snapshots differ even when nothing changed; (3) the executor classified clicks with a `"clicked"` key as success even with a "no visible effect" warning.

**Symptom**: Agent clicks the same button 11+ times on Dominos, wasting all its steps and tokens, eventually force-terminated.

**How agent-xray found it**: `surface_task` on the Dominos task showed 11 consecutive `browser_click_ref` calls all classified as "success" despite "no visible effect" warnings. The step-by-step replay made it obvious that (a) clicks weren't changing anything, (b) spin detection wasn't firing, and (c) the executor was misclassifying results. Three parallel research agents then traced the data flow across `spin_detector.py`, `agent_executor.py`, and the browser server.

**Without agent-xray**: The symptom (repeated clicks) would show in application logs. But diagnosing the 3-way interaction across 3 files -- signature mismatch + fingerprint fragility + misclassification -- would require manually constructing the call chain. The fingerprint fragility bug in particular is nearly invisible without comparing pre/post-click snapshot content side by side, which `surface_task` provides.

#### Bug 3: URL bounce evasion (~42 tasks affected)

**What**: When the agent clicked the same element across alternating URLs (A→B→A→B), each click looked unique because the URL was part of the spin signature. All detection modes missed it.

**Symptom**: Agent stuck clicking "Next" or "Submit" buttons that bounce between two pages, burning 10-20+ steps.

**How agent-xray found it**: After fixing Bug 2, aggregate analysis showed 42 tasks still had undetected `browser_click_ref` spins -- 59% error rate across 87/147 calls. The cross-task analysis revealed the URL-bounce pattern as the systemic cause. This is exactly the kind of finding that requires analyzing many tasks at once, not just one.

**Without agent-xray**: An individual trace might show the bounce, but the insight that this was the #1 systemic issue affecting 42 tasks requires cross-task aggregation. Without agent-xray's grading, you'd need to manually review dozens of task traces to spot the pattern.

#### Bug 4: Hard loop breaker false positive (~10-15 tasks affected)

**What**: The hard loop breaker killed any sequence of 3+ calls to the same tool regardless of whether the arguments differed. Legitimate batch operations (adding 5 tracks to a playlist) were terminated.

**Symptom**: "Create a playlist with these songs" fails with `spin_terminated` after adding 2-3 tracks.

**How agent-xray found it**: `surface_task` on task 5b40f0b63108 showed 5 `playlist_add_track` calls with different track URIs, all killed by the loop breaker. The trace made it immediately obvious the arguments were diverse -- the loop breaker was only checking tool names.

**Without agent-xray**: Medium difficulty. The `spin_terminated` error would appear in logs. But confirming the arguments were diverse (and therefore the kill was wrong) requires seeing the actual arguments of each call, which `surface_task` provides in one step.

#### Bug 5: user_id silo mismatch (~5-10 tasks affected)

**What**: `_call_user_scoped()` had an early return when MCP metadata lacked user_id, causing `memory_store` and `memory_recall` to use different user_id values. Stored memories were invisible on recall.

**Symptom**: "Remember my seat preference is aisle" succeeds. "What are my seat preferences?" returns nothing.

**How agent-xray found it**: Memory task testing (TB-055, TB-056) exposed the symptom. Trace analysis pointed to `_call_user_scoped()`. A 3-line fix.

**Without agent-xray**: This one is findable through basic testing + code review. The function is small. Agent-xray accelerated diagnosis but wasn't essential.

### Attribution Summary

| Bug | Severity | Tasks Affected | Found by agent-xray? | Findable without it? |
|-----|----------|---------------|----------------------|---------------------|
| tool_choice bypass | Silent, behavioral | ~57 | Yes (triage + surface_task) | Very hard -- no error signal |
| 3-way spin gap | Silent, behavioral | ~38-44 | Yes (surface_task replay) | Hard -- 3 interacting bugs |
| URL bounce evasion | Silent, behavioral | ~42 | Yes (aggregate grade analysis) | Hard -- requires cross-task view |
| Loop breaker false positive | Error visible | ~10-15 | Yes (surface_task) | Medium -- log shows symptom |
| user_id silo mismatch | Error visible | ~5-10 | Partially | Low-medium -- basic testing finds it |

**The pattern**: agent-xray's primary value is finding **silent behavioral bugs** -- cases where the agent does the wrong thing without producing errors. These are the hardest bugs in LLM agent systems and the ones that traditional debugging (error logs, stack traces) cannot find.

---

## 3. Full Commit Attribution

34 commits were made to NOVVIOLA during the 12-hour window. Here's how each was discovered:

| Discovery Method | Commits | % | Lines Changed |
|-----------------|---------|---|---------------|
| **agent-xray-informed** | 24 | 71% | +2,597 / -236 |
| **Code-review-findable** | 3 | 9% | +194 / -43 |
| **Log-readable** | 1 | 3% | +18 / -7 |
| **Docs/campaign tracking** | 6 | 18% | +199 / -85 |

"Agent-xray-informed" means the commit message or campaign log cites a specific task ID and agent-xray tool that surfaced the bug. 89% of functional line changes were agent-xray-informed.

### Which agent-xray tools actually drove fixes?

| Tool | Fixes Informed | What It Does |
|------|---------------|-------------|
| `surface_task` | 15 | Step-by-step replay of a single task -- shows what the agent saw, decided, and did |
| `grade` / `root_cause` | 9 | Grade distribution + automated root cause classification across all tasks |
| `inspect_task` | 3 | All-in-one deep dive (grade + root cause + surface + reasoning) |
| `triage` | 2 | Single-call entry point: worst failure + grade distribution + fix priorities |
| `diff_tasks` | 1 | Compare a GOLDEN trace against a BROKEN trace for the same task type |
| `task_bank_validate` | 1 | Validate task bank entries against actual tool names |

**The agent used 6 of 49 available tools.** Those 6 tools drove 24 fixes. The other 43 tools were unused. More on this in Section 5.

---

## 4. Grade Progression

| Metric | Before Campaign | After 12 Hours | Change |
|--------|----------------|----------------|--------|
| GOLDEN | ~2 | 11 | +9 |
| GOOD | ~15 | 63 | +48 |
| OK | ~40 | 181 | +141 |
| WEAK | ~5 | 2 | -3 |
| BROKEN | ~30+ | 36 | See note |
| Total graded | ~90 | 293 | +203 |

**Important caveats on these numbers:**

1. **Stale trace inflation**: The grader counts all traces, not the latest per task. The 293 "tasks" represent ~80-100 unique tasks, many graded on pre-fix code. The BROKEN count (36) includes ~10 stale pre-fix traces. The effective BROKEN rate for current code is estimated at 2-5 tasks.

2. **Coverage expansion**: Total graded grew from ~90 to 293. Many new OK grades are tasks that simply hadn't been run before. The improvement is real but partially inflated by increased coverage.

3. **Environmental gaps account for most remaining BROKEN**: smart_home (requires Home Assistant, not configured), email (requires provider), and music (requires connected account) dominate the BROKEN count. These are environment issues, not code bugs.

4. **Simple task ceiling**: Tasks that correctly require 1-2 steps (set a timer, store a memory) max out at OK under the default ruleset. ~20-30 of 85 tasks hit this ceiling regardless of correctness.

Despite these caveats, the direction is unambiguous. The 5 systemic fixes (tool_choice, spin 3-way, URL bounce, loop breaker, user_id) each affected dozens of tasks and are objectively correct bug fixes verified by regression tests.

---

## 5. What We Learned About Tool Adoption

### The 12% problem

agent-xray had 37 MCP tools available. The debugging agent used 6. This is the most important finding for anyone considering adopting the tool.

**Why only 6?**

1. **Discovery**: NOVVIOLA's instruction file (CLAUDE.md) only documented 8 of 49 tools. The agent didn't know the other 41 existed.
2. **Habit**: Even after documentation was updated mid-POC, the agent kept using its familiar 6-tool workflow.
3. **Context compaction**: LLM context windows compress over time. Newly added documentation was lost between cycles.
4. **Task focus**: When fixing a specific bug, the agent used the minimum tools needed rather than exploring the toolkit.

### What this means for adoption

The effective tool surface of agent-xray is **5-6 tools**, not 49:

| Tool | Role | Adoption Priority |
|------|------|-------------------|
| `triage` | "What's broken and what should I fix first?" | **Start here** |
| `surface_task` | "Show me exactly what happened step-by-step" | **Core workflow** |
| `grade` | "How many tasks are passing/failing?" | **Core workflow** |
| `root_cause` | "What patterns explain the failures?" | **Core workflow** |
| `inspect_task` | "Deep dive on one specific task" | **When surface_task isn't enough** |
| `diagnose` | "What's wrong with this specific task?" | **Alternative to root_cause** |

The other 43 tools solve real problems (golden exemplar analysis, baseline tracking, A/B testing, gaming detection, pricing analysis) but are advanced features that a new user won't touch. Documenting all 49 tools equally does more harm than good -- it buries the 6 that matter.

**Recommendation**: Ship a "start with these 5" guide. Frame everything else as advanced/optional.

---

## 6. The Enforce Workflow: A Cautionary Tale

agent-xray includes an A/B testing workflow called "enforce" -- plan a fix, make the change, test for regressions, detect test-gaming. It was used for 16 of 36 campaign cycles, then abandoned in favor of manual commits.

**Why it was abandoned:**

1. **Gaming false positives**: The gaming detector flagged legitimate code patterns (a string `"no visible effect"` in production code) and legitimate test additions as suspicious. 2 false positives in 6 cycles, 0 true positives.
2. **Windows-specific failures**: Silent commit failures where enforce reported "COMMITTED" but no git commit was created.
3. **Speed**: Manual commits were faster for well-understood fixes. The enforce overhead (init → plan → change → check → guard) adds 4 extra tool calls per fix.

**What this means**: Complex multi-step workflows face an uphill adoption battle with LLM agents. The agent optimized for task completion speed over process discipline. The enforce concept is sound for long-running unattended operation, but in a supervised debugging campaign, it added more friction than safety.

---

## 7. The Moving Target Problem

agent-xray was modified 42 times during the evaluation. This is a methodological impurity that we should quantify rather than ignore.

### What changed and when

| Hours into POC | Version | Key Change | Impact on Evaluation |
|----------------|---------|-----------|---------------------|
| 0-0.5 | v1.14.0 | Performance optimizations (loop fusion, caching) | Speed improvement only -- no new capabilities used |
| 0.5 | v1.15.0 | +4 tools (pricing_list, baseline_generate, task_bank_show, format_detect) | **None of these were used by the debugging agent** |
| 1.0 | v1.16.0 | +1 tool: `triage` | **Used extensively from this point forward** |
| 1.0 | v1.17.0 | +3 tools (gaming_audit, pricing_update, inspect_task) | inspect_task was used in 3 cycles; others unused |
| 4.0 | v1.19.0 | +2 tools (signal_detect, match_task) | **Unused by debugging agent** |
| 4.5 | v1.20.0 | +1 tool (golden_capture) + workflow hints | **Unused by debugging agent** |
| 7.0 | v1.23.0 | grade_filter uses caller's rules (bugfix) | Improved accuracy of grade/root_cause |
| 8.0 | v1.25.0 | CLI parity (triage, inspect, signal-detect) | **Not relevant -- agent uses MCP, not CLI** |
| 12.0 | v1.25.3 | Pricing fix (5x error), task_bank correctness | Fixed cost estimates; improved grading accuracy |

### How much did this affect results?

**Conservatively**: The debugging agent used `triage` (added at hour 1), `inspect_task` (added at hour 1), and benefited from the grade_filter fix (hour 7) and pricing fix (hour 12). Everything else was unused.

**The honest assessment**: The tool that existed at hour 0 (v1.14.0 with 37 MCP tools) already had `surface_task`, `grade`, `root_cause`, `diagnose`, and `diff_tasks` -- the 5 core tools that drove 22 of 24 agent-xray-informed fixes. `triage` (added at hour 1) replaced a 4-tool sequence with a single call, which is a convenience improvement, not a capability change.

**Conclusion**: The core value proposition -- step-by-step trace replay and automated grading -- was present from the start. The mid-evaluation changes mostly added tools that were never used and fixed bugs in peripheral features. The evaluation results are not significantly contaminated by the moving target.

### Two bugs that DID affect evaluation accuracy

1. **Pricing wrong by 5x** (fixed at hour 12): Anthropic cached_input rates were incorrect. All cost estimates during the first 12 hours were wrong for cached token scenarios. This does NOT affect the debugging results -- pricing is informational only. But if you're evaluating agent-xray's cost analysis features specifically, those numbers from this POC are unreliable.

2. **grade_filter used wrong rules** (fixed at hour 7): For 1 hour after grade_filter was added (hours 6-7), filtered grade results used hardcoded rules instead of the caller's rules. This affects only the `grade_filter` parameter, not the core grading. Most grading during the POC was unfiltered.

---

## 8. Cost and Time

### What agent-xray saved

The 24 agent-xray-informed fixes span the spectrum from "would have taken 5 minutes longer" to "might never have been found":

| Difficulty without agent-xray | Fixes | Examples |
|------------------------------|-------|---------|
| **Very hard / might not be found** | 3 | tool_choice bypass (no error signal), 3-way spin gap (3 interacting bugs), URL bounce (requires cross-task aggregation) |
| **Hard / would take significantly longer** | 5 | Early-abort pattern (21 tasks), follow-up routing loss, browser @ref resolution, think escape hatch, consultative routing |
| **Medium / faster with agent-xray** | 10 | Dropdown fuzzy matching, file tool hints, web_search hints, memory FTS5, tool descriptions, loop breaker |
| **Easy / marginal benefit** | 6 | Tool set additions, task bank naming, threshold adjustments |

**Realistic time estimate**: The 3 "very hard" bugs would have consumed 3-6 hours each of manual debugging (grepping step logs, forming hypotheses, testing). agent-xray's trace replay reduced each to 30-60 minutes. The "hard" bugs saved roughly 1-2 hours each. Total estimated time savings: **15-25 hours of debugging compressed into 12 hours of campaign work**.

### What it cost

- **Token cost of agent-xray MCP calls**: Low. Each `triage` or `surface_task` call processes local JSONL files. The MCP server runs locally with no API calls. The cost is the Claude tokens spent on the tool call itself (~500-1000 tokens per call, ~60 calls over 12 hours = ~45K tokens ≈ $0.50-1.00 at Opus pricing).
- **Total campaign cost**: Two Opus-tier Claude Code agents running for 12 hours. The manager agent (auditing + fixing agent-xray) consumed roughly half the total budget for work that was useful but not part of the core evaluation.
- **The 42 agent-xray commits**: 1,835 insertions across 14 files. This was real development work done by the manager agent, not free.

---

## 9. Verdict

### Is agent-xray worth using?

**Yes, for a specific class of problems.** agent-xray excels at finding silent behavioral bugs in LLM agent systems -- cases where the agent does the wrong thing without producing errors. These bugs have no stack traces, no error logs, and no obvious signals. They are the hardest bugs in agent systems and the most common cause of poor user experience.

The 5-tool core workflow (`triage` → `grade` → `surface_task` → `root_cause` → `inspect_task`) provides:
- **A shared quality vocabulary** (GOLDEN/GOOD/OK/WEAK/BROKEN) that replaces subjective assessment
- **Step-by-step trace replay** that shows exactly what the LLM saw, decided, and did
- **Automated root cause classification** that groups failures by pattern (spin, early_abort, tool_bug, prompt_bug)
- **Cross-task aggregation** that reveals systemic issues affecting dozens of tasks at once

### When is it NOT worth using?

- **For bugs that produce errors**: If you have a stack trace, you don't need trace replay. Read the error.
- **For simple codebases**: If your agent has 3 tools and 1 task type, manual log inspection is fine.
- **If you won't run tasks repeatedly**: The grading system's value comes from grading many tasks and tracking trends. A one-off investigation gets less benefit.

### What should you actually install?

Start with 5 tools. Ignore the other 44 until you need them:

```
1. triage()          -- "What's broken?" (start every investigation here)
2. surface_task()    -- "Show me what happened step-by-step"
3. grade()           -- "Grade all my task traces"
4. root_cause()      -- "What patterns explain the failures?"
5. inspect_task()    -- "Deep dive on one task"
```

### The honest multiplier

**2-3x for evidence-based debugging of behavioral bugs.** Not 10x (because most of the toolkit goes unused). Not 1x (because the silent bugs it finds are genuinely hard to find otherwise). For a debugging tool, 2-3x is a strong result -- the equivalent of having a second pair of eyes that has already read every step log.

---

## Appendix A: All 34 NOVVIOLA Commits with Attribution

| # | Hash | Fix | Category | agent-xray Tool | Lines |
|---|------|-----|----------|----------------|-------|
| 1 | `014b490b` | Add ask_user/memory_recall to focused tool sets | xray-informed | surface_task | +7/-0 |
| 2 | `8fb4afdf` | Spin: arg diversity check for hard loop breaker | xray-informed | surface_task | +231/-57 |
| 3 | `c596a864` | Task bank: rename add_track_to_playlist | xray-informed | task_bank_validate | +4/-4 |
| 4 | `b9e5d2a2` | Grading: lower suspicious_short threshold | xray-informed | grade + surface_task | +1/-1 |
| 5 | `646ea945` | Tools: next-step hints for search_files/list_directory | xray-informed | surface_task (3 tasks) | +21/-11 |
| 6 | `d9022158` | Search: direct-answer guidance in navigate_hint | xray-informed | surface_task (3 tasks) | +3/-1 |
| 7 | `0b592748` | Tools: file_info gap 3-pronged fix | xray-informed | surface_task + inspect_task | +44/-21 |
| 8 | `80676226` | Test: regression test for create_playlist user_id | code-review | (regression guard) | +41/-0 |
| 9 | `476dad3b` | Docs: campaign cycle 17 findings | docs/chore | -- | +82/-25 |
| 10 | `71f96d70` | Docs: session findings | docs/chore | -- | +64/-54 |
| 11 | `99354f0d` | Docs: routing bypass root cause documented | docs/chore | -- | +3/-0 |
| 12 | `631f8716` | Agent: force tool_choice=required for actionable categories | xray-informed | triage + surface_task | +12/-6 |
| 13 | `8b6d7f07` | Browser: dropdown fuzzy matching + ARIA listbox | xray-informed | inspect_task | +276/-45 |
| 14 | `a7ce9938` | Browser: auto-detect @ref patterns in click/get_text | xray-informed | step log error analysis | +23/-0 |
| 15 | `cc1f8b00` | Memory: always inject user_id in _call_user_scoped | xray-informed | surface_task + live verify | +112/-4 |
| 16 | `45a0f6c1` | Agent: forward _agent_tool_choice to provider | log-readable | live testing | +18/-7 |
| 17 | `c64dd491` | Classifier: add 'memory' category for recall queries | xray-informed | surface_task | +354/-1 |
| 18 | `0c7dd96b` | Memory: strip stop words + OR fallback in FTS5 | xray-informed | live task replay | +129/-26 |
| 19 | `1db3fe78` | Routing: consultative category + classifier fixes | xray-informed | surface_task (TB-024) | +134/-14 |
| 20 | `57e4f7d3` | Tooling: disambiguate descriptions + send_task matching | xray-informed | grade (7 misuse calls) | +31/-10 |
| 21 | `adc44c1c` | Tools: redirect check_api_registry to web_search | xray-informed | grade (LLC task WEAK) | +2/-1 |
| 22 | `c29e771a` | Tooling: inline agent-xray grade in send_task | docs/chore | -- | +32/-0 |
| 23 | `b3289224` | Docs: cycles 25-27 + consultative results | docs/chore | -- | +18/-6 |
| 24 | `c1bc552a` | Tools: gmail user_id + schedule error surfacing | code-review | (same pattern as prior fix) | +112/-41 |
| 25 | `b3bf343c` | Agent: prevent think tool as terminal action | xray-informed | surface_task | +79/-1 |
| 26 | `7c9eb9dc` | Spin: unify browser signature + fuzzy fingerprint | xray-informed | surface_task + diff_tasks | +146/-37 |
| 27 | `ef752f84` | Routing: preserve category during follow-up routing | xray-informed | surface_task (TB-023) | +242/-10 |
| 28 | `1f5615ea` | Tools: browser_evaluate approval + executor wiring | xray-informed | grade (6 broken tasks) | +283/-16 |
| 29 | `f86ee14a` | Browser: resolve @eN refs in browser_type/select | xray-informed | inspect_task (TB-013) | +50/-4 |
| 30 | `51c2c4c0` | Agent: early-abort nudge after browser nav | xray-informed | grade + root_cause | +205/-0 |
| 31 | `150a561b` | Calendar: pass kwargs to _call_user_scoped | code-review | (same pattern as #24) | +41/-2 |
| 32 | `4e74bd67` | Tools: improve descriptions for model selection | xray-informed | task bank analysis | +21/-10 |
| 33 | `8474907d` | Tools: add search_tracks MCP tool | xray-informed | triage (worst_task) | +149/-6 |
| 34 | `611b3cdb` | Spin: Mode 8 URL bounce detection | xray-informed | surface_task + aggregate | +153/-1 |

## Appendix B: agent-xray Changes During POC (42 commits)

| Category | Count | Impact on Evaluation |
|----------|-------|---------------------|
| Features (new tools) | 14 | 12 tools added were never used by the debugging agent |
| Bugfixes | 14 | 2 HIGH (pricing 5x error, grade_filter rules), 3 MEDIUM, 9 LOW |
| Releases/chores | 7 | Version bumps only |
| Docs | 5 | CAPABILITIES.md updates, not absorbed by agent |
| Tests | 2 | Regression coverage |

**Key point**: The core tools that drove results (`surface_task`, `grade`, `root_cause`, `diagnose`, `diff_tasks`) existed before the POC started. `triage` was added 1 hour in as a convenience wrapper. `inspect_task` was added 1 hour in. Everything else was unused by the debugging agent.

## Appendix C: Grade Distribution at POC End

```
GRADE SUMMARY (293 traces, ~80-100 unique tasks, default ruleset)

  GOLDEN:  11  (tasks with perfect execution)
  GOOD:    63  (effective, minor inefficiencies)
  OK:     181  (completed but suboptimal)
  WEAK:     2  (partially failed)
  BROKEN:  36  (failed -- ~26 stale/environmental, ~10 real)

Top failure patterns: tool_bug (19), early_abort (6), spin (5), prompt_bug (5)
Environmental gaps: smart_home (100% broken, needs Home Assistant), play_playlist (82%, fixed in final cycle)
```
