# Agent-Xray V1.0 Roadmap

## Summary

| Metric | Value |
|--------|-------|
| Current aggregate score | 90/120 (75%) |
| Target aggregate score | 120/120 (100%) |
| Subsystems audited | 12 |
| Total work items | ~95 |
| Estimated total changed lines | ~4,000-6,000 |

### Subsystem Score Table

| # | Subsystem | Current | Target | Gap |
|---|-----------|---------|--------|-----|
| 1 | Schema & Data Model | 8/10 | 10/10 | 2 |
| 2 | Adapters | 7/10 | 10/10 | 3 |
| 3 | Grading | 7/10 | 10/10 | 3 |
| 4 | Root Cause | 8/10 | 10/10 | 2 |
| 5 | Decision Surface | 9/10 | 10/10 | 1 |
| 6 | Reports | 7/10 | 10/10 | 3 |
| 7 | Diagnosis/Fix Plan | 7/10 | 10/10 | 3 |
| 8 | Signals | 7/10 | 10/10 | 3 |
| 9 | CLI & UX | 7/10 | 10/10 | 3 |
| 10 | Testing | 8/10 | 10/10 | 2 |
| 11 | Packaging | 9/10 | 10/10 | 1 |
| 12 | Documentation | 6/10 | 10/10 | 4 |

### Critical Path Items (block v1.0 tag)

1. **Schema round-trip bug** -- `TaskOutcome.from_dict()` double-nests metadata (`schema.py:724`)
2. **Adapter autodetect silently misclassifies** -- real OpenAI/LangGraph/CrewAI traces fall through to `generic` (`adapters/__init__.py:33`)
3. **Grading scores not comparable across rulesets** -- raw sums have different envelopes; GOLDEN is unreachable in `default.json`
4. **Coding signal false positives** -- `FILE_PATH_RE` matches version strings like `1.2.3` (`signals/coding.py:32`)
5. **CLI errors show Python tracebacks** -- no exception boundary in `main()` (`cli.py:422`)
6. **Coverage at 66%** -- no `fail_under` gate in CI; regressions ship silently
7. **Version source of truth is broken** -- `pyproject.toml` says 0.1.0, `CHANGELOG.md` says 0.2.0, no git tags
8. **Docstring coverage near zero** -- 88/94 public functions lack docstrings; no API docs site

---

## Phase 1: Foundation (must-do before v1.0 tag)

These are correctness bugs, silent data loss, and contract violations. Ship nothing without these.

### P1.1 -- Schema Correctness (Schema subsystem)
- **Fix `TaskOutcome.from_dict()` round-trip bug** at `src/agent_xray/schema.py:724`. Exclude `"metadata"` from unknown-key capture; merge `payload["metadata"]` with extra fields instead of nesting it.
- **Add `AgentTask.from_dict()`** companion at `schema.py:791` so the schema layer can round-trip its own canonical task representation. Signature: `AgentTask.from_dict(cls, payload, *, strict=False) -> AgentTask`.
- **Add `schema_version`** field to `AgentStep`, `TaskOutcome`, `AgentTask`. Constants: `SCHEMA_VERSION = "1.0"`, `LEGACY_SCHEMA_VERSION = "0"` in `schema.py`. Emit from all `to_dict()` methods. Export from `__init__.py`.
- Files: `schema.py` (220-300 lines), `capture.py` (8-15), `replay.py` (10-20), `__init__.py`
- Tests: `test_task_outcome_round_trip_preserves_metadata`, `test_agent_task_round_trip`, `test_to_dict_emits_schema_version`, `test_from_dict_rejects_unknown_schema_version` in `tests/test_schema.py`

### P1.2 -- Adapter Autodetect Fix (Adapters subsystem)
- **Rewrite ingestion in `adapters/__init__.py:33`**: split into `iter_jsonl_objects()`, `load_json_document()`, `iter_batched_events()`. Add explicit autodetect branches for `chat.completion`, `chat.completion.chunk`, LangGraph `type/ns/data`, CrewAI `events[]`, OTel `resourceSpans`.
- **Add `openai_chat.py` adapter**: parse raw Chat Completions responses (non-streaming `chat.completion` and streaming `chat.completion.chunk` with delta accumulation). Pair with `role:tool` result messages.
- **Fix `otel.py:22`**: remove the import gate for parsing exported OTLP JSON. Walk descendant tool spans, not just direct children. Use `gen_ai.operation.name=="execute_tool"`.
- Files: `adapters/__init__.py`, new `adapters/openai_chat.py`, `adapters/otel.py:285-315`
- Tests: at least one real fixture per adapter in CI

### P1.3 -- Grading Engine Fix (Grading subsystem)
- **Compile and validate rules on load** at `grader.py:62`. Canonicalize legacy JSON once: `metric -> field`, shorthand comparators -> `op/value`. Reject invalid rules with `ValueError` (bad op, missing `field`/`points`, duplicate signal names, wrong threshold order).
- **Add normalized scoring** to `GradeResult` at `grader.py:32`: add `normalized_score`, `positive_points`, `positive_max`, `penalty_points`, `penalty_max`. Formula: `positive_rate = positive_points / positive_max; penalty_rate = penalty_points / penalty_max; normalized_score = round(100 * positive_rate * (1 - penalty_rate))`. Default thresholds: GOLDEN 85, GOOD 65, OK 40, WEAK 20.
- **Fix `default.json`**: GOLDEN threshold (5) is above max raw score (3), making it unreachable. Fix with normalized model.
- Files: `grader.py` (~150 lines), `rules/default.json`, `rules/browser_flow.json`, `rules/coding_agent.json`, `rules/research_agent.json`

### P1.4 -- Signal Accuracy Fix (Signals subsystem)
- **Fix `FILE_PATH_RE`** in `signals/coding.py:32-34`. Add version-string negative lookahead and `VERSION_RE` guard. Add `_looks_like_version()` helper.
- **Fix `has_test_verify_cycle`** at `coding.py:68-69`: require at least one test step after the first file-op step (order-sensitive).
- **Raise `signals/__init__.py` coverage** from 42% to >85%. Add tests for `entry_points().select`, plugin load/error handling, `run_detection(task, detectors=[...])`.
- Files: `signals/coding.py:32-87`, `signals/__init__.py:27-72`

### P1.5 -- CLI Error Handling (CLI & UX subsystem)
- **Add exception boundary in `main()`** at `cli.py:422`. Catch `CliUserError`, `FileNotFoundError`, `KeyError`, `ValueError`, `ImportError`. Print concise stderr messages with hints, never tracebacks. If `--json`, emit `{"error": {...}}`.
- **Add `CliUserError` class** in new `cli_support.py` with `message`, `hint`, `exit_code`.
- **Detect empty inputs**: when `load_tasks()` returns `[]`, explain what went wrong instead of printing "Analyzed 0 task(s)".
- Files: `cli.py:422`, new `cli_support.py`, `analyzer.py:351-446`

### P1.6 -- Coverage Gate (Testing subsystem)
- **Fix coverage measurement**: prevent pre-import of `agent_xray` before coverage starts (fixes `module-not-measured` noise).
- **Add `fail_under = 80`** to `pyproject.toml` coverage config.
- **Priority test gaps**: `root_cause.py` (46% -> 90%), `grader.py` (59% -> 92%), `schema.py` (54% -> 90%).
- Files: `pyproject.toml:136`, `tests/test_root_cause.py`, `tests/test_grader_edge_cases.py`, `tests/test_schema.py`

### P1.7 -- Version Source of Truth (Packaging subsystem)
- **Adopt `setuptools-scm`**: add to `build-system.requires` in `pyproject.toml:2`. Replace static `version = "0.1.0"` with `dynamic = ["version"]`. Add `[tool.setuptools_scm]` block.
- **Fix release automation**: add `tags: ["v*"]` to CI push trigger at `ci.yml:4`. Add `fetch-depth: 0` at `ci.yml:65`.
- **Fix CHANGELOG**: sync top section version. Add `scripts/validate_release.py`.
- Files: `pyproject.toml`, `.github/workflows/ci.yml`, `CHANGELOG.md`, `src/agent_xray/__init__.py:51`

---

## Phase 2: Polish (should-do for quality v1.0)

These items improve usability, completeness, and contract quality. A v1.0 without these is shippable but rough.

### P2.1 -- Schema Declarative Normalization (Schema)
- Replace 4 manual context-hoisting blocks in `schema.py:451-610` with declarative `_extract_context_values()` and `_build_context()` helpers. Add `AgentStep.from_components()` constructor.
- Preserve unknown nested context fields (e.g., `model.reasoning_tokens`) in `extensions` as namespaced keys instead of silently dropping them.
- Split canonical output schemas (`additionalProperties: False`) from legacy-compatible parsers.
- Estimated: 200+ lines in `schema.py`, 160-230 lines of new tests

### P2.2 -- New Adapters (Adapters)
- **Add `langgraph.py`**: parse LangGraph v2 `StreamPart` records and `astream_events(version="v2")`.
- **Add `crewai_webhook.py`**: parse AMP webhook envelopes with `events[]` array and single-event realtime deliveries.
- **Fix `langchain.py:81`**: stop using `parent_run_id` as fallback task id; prefer `metadata.task_id`, then `parent_ids[0]`.
- **Fix `anthropic.py:42`**: map `tool_result.is_error=true` to `error`. Preserve `text`/`thinking` blocks as reasoning context.
- 14 new test fixtures needed (see Per-Subsystem section below)

### P2.3 -- Root Cause Scoring (Root Cause)
- **Refactor `classify_task()` at `root_cause.py:165`** from early-return cascade to candidate scoring. Add `_score_candidates()`, `_confidence_label()`, `_task_family()` helpers.
- **Extend taxonomy**: add `test_failure_loop`, `insufficient_sources`, `service_limit`. Broaden error-kind mappings.
- **Add numeric confidence**: `confidence_score: float` and `candidate_scores: dict` on `RootCauseResult`.
- **Fix prompt-bug specificity**: `_enrich_prompt_bug()` at line 254 should run for every final `prompt_bug`, not only fallback path.
- Coverage target: `root_cause.py` above 90%

### P2.4 -- Report Architecture (Reports)
- **Introduce `ReportDocument` model** with `ReportSection`, `ReportOptions` in new `report_models.py`.
- **Add renderer abstraction** in `report_renderers.py`: `render_text()`, `render_markdown()`, `render_html()`, `render_report(doc, fmt=...)`.
- **Add report registry** in `report_registry.py`: `register_report()`, `get_report_builder()`, `list_report_types()`.
- **Fix text/data parity**: `reports.py:513` emits 6 action categories but `:609` serializes only 3.
- **Fix compare UX**: add `resolve_compare_days()` with auto-pick and invalid-day validation.

### P2.5 -- Fix Plan Enrichment (Diagnosis/Fix Plan)
- **Add framework context** to `RootCauseResult` at `root_cause.py:75`: `framework`, `trace_format`, `task_category`.
- **Create `fix_targets.py`** with structured `FixTarget` (label, kind, path, line, url, framework, note) and `FixTargetResolver` protocol.
- **Upgrade `build_fix_plan()`** at `diagnose.py:100`: accept `resolver` and `severity_scorer`, group by `(root_cause, framework, prompt_section)`, add urgency scoring.
- **Add `fixplan` report type** to CLI at `cli.py:263`.

### P2.6 -- New Signal Detectors (Signals)
- **Add `signals/multi_agent.py`**: detect spawn/delegate/wait/close patterns. Emit `spawn_count`, `used_multi_agent`, `parallel_coordination`, `coordination_ratio`.
- **Add `signals/planning.py`**: detect plan/todo/checklist patterns. Emit `has_explicit_plan`, `has_plan_before_action`, `replan_count`.
- **Add `signals/memory.py`**: detect memory/RAG/retrieval patterns. Emit `retrieval_count`, `used_rag`, `retrieval_backed_answer`.
- **Add `signals/safety.py`**: detect approval/injection/secret patterns. Emit `approval_block_count`, `secret_exposure_risk`, `used_guardrails`.
- **Add cross-detector `correlate_signals()`** in `signals/__init__.py:62`.
- **Fix `analyzer.py:141`**: stop blind `metrics.update()`; namespace detector metrics.

### P2.7 -- CLI Polish (CLI & UX)
- **Parser/help rewrite**: add descriptions, epilogs with examples to all subparsers. Fix `report` arg order to `report <type> <log_dir>`.
- **Add global flags**: `--verbose`, `--quiet`, `--no-color` via parent parser and `CliContext`.
- **Add progress indicators**: `ProgressSink` protocol, wire through `load_tasks`, `analyze_tasks`, `grade_tasks`.
- **Output polish**: replace raw dict prints at `cli.py:112-113` and `:177-178` with stable text formatters.

### P2.8 -- Decision Surface Performance (Decision Surface)
- **Fix O(n^2) history** at `surface.py:37`: replace additive copies with windowed history + rolling summary. Add `history_mode`, `history_window`, `history_summary_chars` parameters.
- **Add sparse-metadata provenance**: `surface_coverage`, `missing_fields`, `field_sources` per step. Make unknown explicit instead of pretending empty.
- **Replace positional diff** at `surface.py:178` with alignment-aware diff using `difflib.SequenceMatcher`. Add `alignment_score`, `step_pairs`.

### P2.9 -- Docstrings & Quick Start (Documentation)
- **P0 docstrings**: add Google-style docstrings to the ~25 core exported symbols (`TaskAnalysis`, `analyze_task`, `load_tasks`, `GradeResult`, `load_rules`, `grade_task`, `RootCauseResult`, `classify_task`, `AgentStep.from_dict`, `AgentTask`, `surface_for_task`, etc.).
- **Ship bundled example traces**: move/duplicate fixtures into `src/agent_xray/examples/`. Update `pyproject.toml:110` to include them.
- **Fix README Quick Start**: start from bundled traces, not `./traces`. First command should work on fresh install.

---

## Phase 3: Excellence (nice-to-have for 10/10)

These take subsystems from good to exceptional. They can ship post-1.0 without embarrassment.

### P3.1 -- Schema Strict Mode & Extensions
- Tighten canonical JSON schemas with `additionalProperties: False` and `schema_version` required.
- Add `AGENT_STEP_COMPAT_JSON_SCHEMA` for legacy input.
- Version fixtures in `capture.py:119`; reject unsupported versions in `replay.py:15`.

### P3.2 -- Adapter Real-World Validation
- Create capture scripts per framework under `tests/captures/` with exact library versions.
- Replace all synthetic-only fixtures with captured traces. Require at least one real fixture per adapter in CI.
- Add golden assertion sets: step count, task grouping, tool order, timestamp fidelity.

### P3.3 -- Ruleset Inheritance & Cleanup
- Add `extends: "default"` mechanism so specialized rulesets inherit generic safety signals.
- Add `orchestrator_agent.json` weighted around `multi_agent.*`, `planning.*`.
- Reweight existing rulesets for new signal detectors.
- Add `docs/RULE_AUTHORING.md` with canonical schema, operators, scoring formula, calibration workflow.

### P3.4 -- Report Expansion
- **Add `costs` report**: group by day/mode/site/model using existing `analyzer.py:54-102` data.
- **Add `regression` report**: worst task regressions by grade drop, spin increase, cost increase.
- **Replace commerce-only `flows`** with domain-neutral `WorkflowDetector` in `signals/workflow.py`.
- **Add markdown/html CLI output**: `--output-format`, `--out` flags.

### P3.5 -- Root Cause Documentation
- Document `ClassificationConfig` in code docstring, `README.md:185-203`, and `docs/CONCEPTS.md`.
- Add API snippet examples for threshold tuning.

### P3.6 -- CLI Extras
- **Add `quickstart` command**: generate deterministic sample workspace with demo traces, run full pipeline, print next-steps.
- **Add `completion` command**: render shell completions for bash/zsh/fish.
- **Add color output**: ANSI styling for grades, headings, warnings. Respect `NO_COLOR`.

### P3.7 -- Decision Surface Typed Contexts
- Add `MemoryContext` and `DelegationContext` dataclasses in `schema.py:204-215`.
- Surface them per step with same missing/provenance contract.
- TUI lazy caching for large tasks instead of precomputing full surface blob.

### P3.8 -- Testing Excellence
- **Property-based tests**: Hypothesis for `AgentStep.from_dict` (never raises on JSON-like input), coercion totality, round-trip identity.
- **Performance benchmarks**: `test_bench_load_tasks_100k_lines`, `test_bench_surface_for_5k_step_task` using `pytest-benchmark`.
- **TUI functional tests**: Textual pilot tests for render, navigation, mode switching.
- **Mutation testing pilot**: `mutmut` on `schema.py`, `grader.py`, `root_cause.py`.
- Coverage targets: overall >= 85% line, >= 80% branch.

### P3.9 -- Packaging Hardening
- **Add package verification CI job**: `python -m build`, `twine check`, wheel + sdist install smoke tests.
- **Add security scanning**: `pip-audit`, `dependency-review-action`, SBOM generation with `cyclonedx-py`.
- **Fix classifier**: Alpha -> Production/Stable. Remove AsyncIO classifier.
- **Clean up extras**: `[all]` = runtime only; `[dev]` = tooling. Fix CI to install `.[dev]`.
- **Close platform gap**: remove Windows/macOS 3.10 CI exclusions or stop claiming support.
- **TestPyPI rehearsal** with `v1.0.0rc1` before GA.

### P3.10 -- Full Documentation Site
- **MkDocs site**: `mkdocs-material` + `mkdocstrings[python]`. Add `mkdocs.yml`.
- **Docs IA**: `getting-started/`, `tutorials/`, `guides/`, `reference/`, `architecture/`.
- **5-minute tutorial**: `docs/tutorials/first-root-cause.md` -- install, grade, surface, root-cause, capture, replay.
- **Architecture docs**: subsystem overview, data-flow diagram, extensibility guide.
- **CONTRIBUTING cleanup**: add workflow section, branch naming, fixture policy, PR checklist.
- **P1/P2 docstrings**: adapter entry points, replay/capture/flywheel, CLI handlers, TUI methods.
- Point `[project.urls].Documentation` to built docs site.

---

## Per-Subsystem Plans

### 1. Schema & Data Model (8 -> 10)

**Findings**:
- `TaskOutcome.from_dict()` round-trip bug at `schema.py:724` -- double-nests metadata
- `AgentStep` normalization has 5 sources of truth: reserved field sets (`:14`), constructor kwargs (`:237`), merge helpers (`:312`), `from_dict()` hoisting (`:451`), JSON schema (`:803`)
- Unknown nested context fields silently lost at `schema.py:538-610`
- No version field on payloads or fixtures
- `from_dict({})` returns valid blank step (bad default for durable schema)
- Coercion too permissive: `_coerce_optional_bool("maybe")` returns `True`
- No `AgentTask.from_dict()` companion at `schema.py:791`

**Action Items** (priority order):
1. Fix `TaskOutcome.from_dict()` metadata round-trip -- `schema.py:724`
2. Add `AgentTask.from_dict()` and `AgentStep.from_dict(strict=True)` -- `schema.py`
3. Replace manual context hoisting with declarative `_extract_context_values()`, `_build_context()` -- `schema.py:451-610`
4. Preserve unknown nested fields in extensions as namespaced keys -- `schema.py:538`
5. Add `SCHEMA_VERSION = "1.0"` to steps/tasks/fixtures -- `schema.py`, `capture.py:119`, `replay.py:15`
6. Tighten canonical JSON schemas, split from legacy-compatible parser -- `schema.py:803-922`

**Estimated**: 400-575 changed lines across `schema.py`, `capture.py`, `replay.py`, `otel.py`, tests

---

### 2. Adapters (7 -> 10)

**Findings**:
- Auto-detection is JSONL-only, silently falls back to `generic` -- `adapters/__init__.py:33-169`
- OpenAI coverage is Assistants-only; no raw Chat Completions -- `openai_sdk.py:40-72`
- LangChain misgrouping: falls back to `parent_run_id` -- `langchain.py:81`
- Anthropic drops `is_error`, `text`/`thinking` blocks -- `anthropic.py:19-42`
- CrewAI fixture-only, not real webhook format -- `crewai.py:11`
- OTel import gate unnecessary for static JSON; only direct child spans extracted -- `otel.py:22-315`
- Tests are all happy-path synthetic fixtures -- `test_adapters.py:31`

**Action Items** (priority order):
1. Rewrite ingestion/autodetect in `adapters/__init__.py:33` -- support `.json` docs, webhook envelopes, arrays
2. Add `adapters/openai_chat.py` -- raw Chat Completions + streaming chunk accumulation
3. Add `adapters/langgraph.py` -- LangGraph v2 StreamPart + astream_events; fix `langchain.py` root-correlation
4. Fix `otel.py` -- remove import gate, walk descendant tool spans, fix semantic mapping
5. Add `adapters/crewai_webhook.py` -- AMP webhook envelopes with event types
6. Extend `anthropic.py` -- `is_error` mapping, reasoning/text/thinking preservation, raw Messages envelopes
7. Replace synthetic fixtures with real captured traces; one real fixture per adapter in CI

**New Fixtures Needed**:
- `tests/fixtures/openai_chat_completion.jsonl`
- `tests/fixtures/openai_chat_completion_stream.jsonl`
- `tests/fixtures/anthropic_parallel_tool_use.jsonl`
- `tests/fixtures/anthropic_tool_error.jsonl`
- `tests/fixtures/langgraph_stream_updates_v2.jsonl`
- `tests/fixtures/langgraph_stream_tasks_subgraph_v2.jsonl`
- `tests/fixtures/langchain_astream_events_v2.jsonl`
- `tests/fixtures/crewai_webhook_batch.json`
- `tests/fixtures/crewai_webhook_realtime.jsonl`
- `tests/fixtures/otel_real_otlp.json`
- `tests/fixtures/otel_nested_tool_spans.json`
- `tests/fixtures/otel_execute_tool_error.json`

---

### 3. Grading (7 -> 10)

**Findings**:
- `_compare` at `grader.py:83` implements every comparator twice (op/value and shorthand forms)
- `load_rules` silently accepts malformed rules at `grader.py:62`; no validation for unknown ops, missing `points`, empty `field`
- Scoring not comparable across rulesets: `default` -19..3, `browser_flow` -24..18
- `default.json` GOLDEN threshold (5) > max raw score (3) -- unreachable
- Golden gating brittle with duplicate labels allowed -- `grader.py:232`
- Ruleset duplication high between `default.json` and domain-specific rules
- Tests at 59% coverage with 7 operators tested -- `test_grader_edge_cases.py:81`

**Action Items** (priority order):
1. Rule compilation/validation at `grader.py:62` -- canonicalize, validate, reject bad rules
2. Normalized scoring at `grader.py:185` -- add `normalized_score` with new thresholds
3. Ruleset cleanup -- `default.json`, `browser_flow.json`, `coding_agent.json`, `research_agent.json`; add `extends` mechanism
4. Coverage expansion -- 15 new tests targeting >90% branch coverage on `grader.py`
5. Add `docs/RULE_AUTHORING.md`; trim legacy syntax from `README.md:123`

---

### 4. Root Cause (8 -> 10)

**Findings**:
- `classify_task()` at `root_cause.py:165` is first-match cascade, not scorer -- browser-heavy taxonomy
- Signal metrics from coding/research detectors are ignored -- `root_cause.py:165-251`
- Confidence is not real confidence -- several branches silently inherit `"medium"` -- `root_cause.py:232-250`
- Error-kind coverage partial -- `analyzer.py:14-30` defines 9 kinds, `root_cause.py:201-215` uses 6
- Strong prompt_bug evidence loses targeting -- early return at `:222-226` bypasses `_enrich_prompt_bug()` at `:254-287`
- Coverage at 40% -- `root_cause.py`

**Action Items** (priority order):
1. Refactor `classify_task()` to candidate scoring with `_score_candidates()`, `_confidence_label()`, `_task_family()` -- `root_cause.py:165`
2. Add `test_failure_loop`, `insufficient_sources`, `service_limit`; broaden error mappings -- `root_cause.py:12-72`, `diagnose.py:9-64`
3. Add numeric `confidence_score: float` and `candidate_scores: dict` to `RootCauseResult` -- `root_cause.py:76`
4. Expand tests to >90% coverage -- `tests/test_root_cause.py`, `tests/test_diagnose.py`
5. Document `ClassificationConfig` -- `root_cause.py:152`, `README.md`, `docs/CONCEPTS.md`

---

### 5. Decision Surface (9 -> 10)

**Findings**:
- O(n^2) conversation history at `surface.py:37` -- full per-step copy at `:116`
- Missing metadata collapsed to `[]`/`0` instead of `None` -- `surface.py:54`, `:77`
- No distinction between "not captured" and "captured and empty" -- `surface.py:67`, `:231`
- Positional zip diff misaligns on retry steps -- `surface.py:178`
- `prior_conversation_turns` from `analyzer.py:440` never exposed at `surface.py:143`
- No memory/RAG or delegation surface types

**Action Items** (priority order):
1. Fix history: windowed mode + rolling summary at `surface.py:37`; add `history_mode`, `history_window`, `history_summary_chars`
2. Add sparse-metadata provenance: `surface_coverage`, `missing_fields`, `field_sources` per step
3. Replace positional diff with alignment-aware diff using `difflib.SequenceMatcher` at `surface.py:178`
4. Wire CLI/TUI/comparison to new surface contract
5. Add `MemoryContext` and `DelegationContext` typed surfaces -- `schema.py:204-215`

---

### 6. Reports (7 -> 10)

**Findings**:
- Docs and CLI help out of sync -- `reports.py:1` lists 8 types, `cli.py:263` dispatches 9+compare, help omits `coding`/`research`
- Flow report is commerce-specific -- `reports.py:367` hardcodes `Cart -> Checkout -> Payment`
- Output formats limited to text/JSON -- no markdown/html
- Text/data parity weak -- `reports.py:513` emits 6 action categories, `:609` serializes 3; compare text includes `payment_pct`, compare JSON drops it
- Compare UX brittle -- `cli.py:253` hard-fails without `--day1`/`--day2`; no auto-pick
- Tests shallow -- `test_reports.py:7` missing coding/research; assertions are header/key only

**Action Items** (priority order):
1. Canonical `ReportDocument` + `ReportSection` + `ReportOptions` model in `report_models.py`
2. Report registry in `report_registry.py`; renderer abstraction in `report_renderers.py`
3. Fix parity bugs in existing reports; align help text with actual choices
4. `resolve_compare_days()` with auto-pick and validation; normalized compare JSON
5. Replace commerce-only flows with `WorkflowDetector` in `signals/workflow.py`
6. Add markdown/html output and `--out` flag
7. Add `costs` and `regression` report types
8. Route `cmd_compare` through report renderer; add customization API

**New Test Files**: `tests/test_report_renderers.py`, `tests/test_report_compare_ux.py`, `tests/test_report_registry.py`

---

### 7. Diagnosis/Fix Plan (7 -> 10)

**Findings**:
- Fix plan is string-only -- `diagnose.py:9,67,77` hardcode `targets` as `list[str]`
- Grouping too lossy -- `diagnose.py:100` groups by `root_cause` only; merges OpenAI/LangChain/prompt-section failures
- Upstream missing framework context -- `root_cause.py:75` has no `framework`/`trace_format`
- `impact` is not urgency -- `diagnose.py:106` uses `len(items) * abs(worst.score)`
- No diagnosis report type -- `cli.py:263` has no `fixplan`

**Action Items** (priority order):
1. Add `framework`, `trace_format`, `task_category` to `RootCauseResult` at `root_cause.py:75`; write `task.metadata["trace_format"]` in `analyzer.py:372`
2. Create `fix_targets.py` with `FixTarget`, `FixResolutionContext`, `FixTargetResolver` protocol
3. Upgrade `build_fix_plan()` at `diagnose.py:100` -- structured targets, framework-aware grouping, urgency scoring
4. Add `FixSeverity` scoring: base by cause + frequency + worst_score + confidence + breadth
5. Add `report_fix_plan`/`report_fix_plan_data` in `reports.py`; wire `fixplan` to CLI at `cli.py:263`

---

### 8. Signals (7 -> 10)

**Findings**:
- `FILE_PATH_RE` matches version strings -- `coding.py:32-34`
- `has_test_verify_cycle` is order-insensitive -- `coding.py:68-69`
- `research.py:15-16` overcounts browsing as research; `is_synthesis` too narrow; `has_url_in_result` is just `"http" in result`
- `analyzer.py:141` blindly flattens detector outputs -- collision risk with more detectors
- `signals/__init__.py` at 42% coverage

**Action Items** (priority order):
1. P0: Fix `FILE_PATH_RE` with version guard and `__init__.py` coverage
2. P1: Add `multi_agent.py` and `planning.py` detectors
3. P2: Add `memory.py` and `safety.py` detectors
4. P3: Add `correlate_signals()` aggregation; namespace detector metrics in `analyzer.py:141`; reweight rulesets

**New Files**: `signals/multi_agent.py`, `signals/planning.py`, `signals/memory.py`, `signals/safety.py`

---

### 9. CLI & UX (7 -> 10)

**Findings**:
- Help/discoverability gap -- `build_parser()` at `cli.py:306` has no descriptions/examples
- `report` arg order wrong -- `report log_dir report_type` instead of `report <type> <log_dir>`
- Error handling not CLI-grade -- `main()` at `cli.py:422` has no exception boundary
- Empty inputs treated as success -- `cli.py:112-113` prints "Analyzed 0 task(s)"
- Output inconsistent -- raw dicts, pretty JSON, ad hoc renders
- No `--verbose`/`--quiet`/`--no-color`; no progress indicators; no shell completion

**Action Items** (priority order):
1. `main(argv=None)` exception boundary, `CliUserError`, stderr/error mapping, empty-dir detection
2. Parser/help rewrite: descriptions, examples, fix `report` order, fix help omissions
3. Output controls: `--verbose`, `--quiet`, `--no-color`, `CliContext`
4. Progress plumbing through `load_tasks`, `analyze_tasks`, `grade_tasks`, `compare_model_runs`, `run_flywheel`
5. `quickstart` command and sample workspace generation
6. `completion` command for bash/zsh/fish
7. Test expansion and README/install docs

**New Files**: `cli_support.py`, `demo.py`, `completion.py`

---

### 10. Testing (8 -> 10)

**Current State**: 211 tests, 66% overall coverage. Worst modules: `tui/app.py` (0%), `pytest_plugin.py` (33%), `signals/__init__.py` (42%), `root_cause.py` (46%), `schema.py` (54%), `grader.py` (59%).

**Action Items** (priority order):
1. Coverage job hygiene: fix pre-import issue, add `fail_under`
2. `root_cause.py` gaps: `_enrich_prompt_bug` branches, `classify_failures`, `summarize_root_causes`, `format_root_causes_text`, `ClassificationConfig`
3. `grader.py` gaps: `ne`/`not_in` operators, legacy keys, threshold boundaries, golden requirement downgrades
4. `schema.py` gaps: coercion edge cases, `from_dict` precedence, property accessors, `TaskOutcome`/`AgentTask` round-trips
5. TUI functional tests with Textual pilot -- `tests/test_tui_app.py`
6. Property-based tests with Hypothesis -- `tests/test_schema_property.py`
7. Performance benchmarks -- `tests/test_bench_large_traces.py`
8. Real adapter fixture corpus under `tests/fixtures/real/`
9. Mutation testing pilot on `schema.py`, `grader.py`, `root_cause.py`

**Coverage Targets**: `root_cause >= 90`, `schema >= 90`, `grader >= 92`, `tui/app >= 80`, `pytest_plugin >= 85`, `signals/__init__ >= 85`, `analyzer >= 85`, overall >= 85% line, >= 80% branch

---

### 11. Packaging (9 -> 10)

**Findings**:
- Version inconsistency: `pyproject.toml` = 0.1.0, `__init__.py` = 0.1.0, `CHANGELOG.md` = 0.2.0, no git tags
- Publish workflow not tag-triggered -- `ci.yml:3` only runs on push to main/PRs
- Classifiers wrong: still Alpha, declares AsyncIO
- No artifact verification before publish -- no `build`, `twine check`, install smoke test
- No security scanning or SBOM
- CI skips Windows/macOS on 3.10 while claiming support
- `[all]` extras include dev tooling

**Action Items** (priority order):
1. Adopt `setuptools-scm` for version source of truth
2. Repair tag-triggered release workflow in `ci.yml`
3. Add `package` CI job: build, twine check, wheel/sdist install smoke test
4. Add security scanning: `pip-audit`, `dependency-review-action`, SBOM generation
5. Fix classifiers for 1.0 (Production/Stable, drop AsyncIO)
6. Clean up extras: `[all]` = runtime, `[dev]` = tooling
7. Close platform matrix gap
8. TestPyPI rehearsal with `v1.0.0rc1`

---

### 12. Documentation (6 -> 10)

**Findings**:
- README Quick Start starts from `./traces` -- broken for fresh `pip install` users
- 88/94 public functions lack docstrings; 24/30 modules lack module docstrings
- No API docs site; no tutorials; no architecture docs
- Example traces exist in `examples/sample_traces` and `tests/fixtures` but not shipped in wheel
- `CONTRIBUTING.md` covers extension mechanics but not contribution workflow
- Version/docs mismatch between `pyproject.toml` and `CHANGELOG.md`

**Action Items** (priority order):
1. Ship bundled example traces in package data; fix `pyproject.toml:110` packaging config
2. Fix README Quick Start to use bundled traces; link to 5-minute tutorial
3. Create 5-minute tutorial at `docs/tutorials/first-root-cause.md`
4. Set up MkDocs site with `mkdocs-material` + `mkdocstrings[python]`
5. P0 docstrings: ~25 core exported symbols (Google-style)
6. Docs IA: `getting-started/`, `tutorials/`, `guides/`, `reference/`, `architecture/`
7. Architecture docs: subsystem overview, data-flow, extensibility
8. CONTRIBUTING cleanup: workflow section, branch naming, fixture policy, PR checklist
9. P1 docstrings: adapter entry points, replay/capture/flywheel/comparison
10. P2 docstrings: CLI handlers, TUI methods, schema properties
11. Point `[project.urls].Documentation` to docs site

**New File**: `mkdocs.yml`, `src/agent_xray/examples/tutorial/`, 16+ docs pages
