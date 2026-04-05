# Changelog

## [1.25.4] - 2026-04-05

### Fixed
- README MCP tool count corrected: 49 → 37 (actual `@server.tool` decorators in `mcp_server.py`)
- Root cause category count corrected: 19 → 22 across README, CAPABILITIES.md (added `context_overflow`, `rate_limit_cascade`, `timeout` to docs)
- CAPABILITIES.md version synced to 1.25.4
- CAPABILITIES.md root cause count corrected: 17 → 22
- CAPABILITIES.md MCP tool count corrected: 48 → 37

## [1.25.3] - 2026-03-30

### Fixed
- Anthropic cached_input pricing rates were wrong by ~5x (50% of input instead of 10%)
- `no_browser_needed` task bank criterion always returned PASS even when browser tools were used
- Unknown task bank criteria silently passed as `[PASS] (skipped)` — now correctly `[FAIL] (not implemented)`
- CAPABILITIES.md doc drift: `messages.stream` → `messages.create`, `xray_trace` context manager → decorator

### Added
- Claude 4.5/4.6, Codex 5.4 model pricing
- Manager-agent loop POC evaluation report (`docs/MANAGER_LOOP_POC_REPORT.md`)

## [1.25.2] - 2026-03-30

### Fixed
- OTel `.json` files now discovered in directory scans (previously only `.jsonl`)
- `format_detect` MCP tool crashed on directory input
- Dead `--task-bank` flag removed from `inspect` CLI parser
- README tool count corrected (was 28, actual 48+)

### Added
- `--site` filter on `search` CLI (parity with MCP `search_tasks`)

## [1.25.1] - 2026-03-30

### Changed
- Version bump for PyPI publish (v1.25.0 had packaging conflict)

## [1.25.0] - 2026-03-30

### Added
- `inspect` CLI subcommand — all-in-one single-task investigator (was MCP-only)
- `signal-detect` CLI subcommand — domain signal detectors (was MCP-only)
- `preflight_diff` MCP tool + `enforce preflight-diff` CLI — check git diff against project guardrails
- `min_steps` parameter on `report` golden data
- Per-criterion PASS/FAIL evaluation in `surface_task` with task bank matching

## [1.24.0] - 2026-03-30

### Added
- `triage` CLI subcommand — the #1 adoption entry point
- Next-step hints in `inspect_task` output for progressive disclosure

## [1.23.1] - 2026-03-30

### Fixed
- Empty-after-filter handling and input validation in `grade_filter`

## [1.23.0] - 2026-03-30

### Fixed
- `grade_filter` used hardcoded default rules instead of caller's rules — now passes through correctly

### Added
- `grade_filter` parameter on `triage`, `grade`, `root_cause`, `diagnose` MCP tools

## [1.20.0] - 2026-03-29

### Added
- `golden_capture` MCP tool — save exemplar task as fixture
- Workflow hints in tool outputs to drive adoption of related tools
- `outcome` filter on grade/root_cause/diagnose tools
- `gaming_audit` `allow_test_modification` config exposed in MCP

## [1.19.0] - 2026-03-29

### Added
- `signal_detect` MCP tool — run domain signal detectors (commerce, coding, research, multi-agent, memory, planning)
- `match_task` MCP tool — fuzzy-match task to task bank entries

## [1.17.0] - 2026-03-29

### Added
- `gaming_audit` MCP tool — detect test-gaming patterns in diffs (8 detectors)
- `pricing_update` MCP tool — fetch latest pricing from GitHub
- `inspect_task` MCP tool — all-in-one: grade + root cause + surface + reasoning

## [1.16.0] - 2026-03-29

### Added
- `triage()` MCP tool — single-call investigation entry point returning grade distribution, worst failure, and fix plan
- Tool count: 42 MCP tools

## [1.15.0] - 2026-03-29

### Added
- `pricing_list`, `baseline_generate`, `task_bank_show`, `format_detect` MCP tools
- Tool count: 41 MCP tools

## [1.14.0] - 2026-03-29

### Changed
- Performance: fused 10+ loops in `_compute_core_metrics` into single pass + cached `sorted_steps`

## [1.2.4] - 2026-03-28

### Fixed
- **CRITICAL**: `task_failed` field added to `_compute_core_metrics()` in v1.2.2 but never added to `TaskAnalysis` dataclass, causing `TypeError` crash on any call to `analyze_task()` (affects all CLI commands and API usage)
- `task_failed_penalty` rule in `default.json` was dead code (could never fire) due to the above crash; now operational
- Completeness dimension count updated from 12 to 13 in tests to match the step-data-loss dimension added in v1.2.2

## [0.2.0] - 2026-03-27

### Added
- Layered schema: core step fields plus typed extension contexts for model, tools, reasoning, and browser state
- Built-in adapters for generic JSONL, OpenAI, LangChain, Anthropic, CrewAI, and OTel-style traces
- Auto-detect trace format from file contents
- `--format` support across the CLI
- Pluggable signal detection with bundled commerce, coding, and research detector packs
- Domain-specific reports: `coding` (file ops, test/verify cycles) and `research` (search diversity, citations)
- JSON output (`--json`) for all 10 report types: health, golden, broken, tools, flows, outcomes, actions, coding, research, compare
- `ClassificationConfig` for tunable root-cause thresholds (spin, stall, abort, etc.)
- Diagnosis module: `build_fix_plan()` with priority-ranked fix entries, targets, and investigate-worst-task
- Interactive TUI for decision-surface replay
- GitHub Actions CI with lint, typecheck, test matrix (3 OS × 4 Python), and PyPI trusted-publisher publish on tag
- pytest plugin for agent trace assertions
- Flywheel baseline comparison and fixture replay
- Cost-per-decision tracking via `ModelContext`
- Model comparison CLI (`agent-xray compare`) for A/B testing different models
- Optional dependency extras for runner, OTel, TUI, and full dev installs
- `py.typed` marker for downstream type checkers
- Comprehensive test suite: 211 tests covering reports, protocols, diagnosis, integration, grading edge cases

### Changed
- Schema evolved from a flat step record toward typed contexts while preserving `AgentStep.from_dict()` compatibility
- Adapters moved from example-only usage into first-class package modules
- `resolve_task()` now shows available task IDs on KeyError for better DX
- Public docs now reflect the real CLI surface, optional extras, and extension points

### Fixed
- `mypy --strict` support for the core package configuration used in CI
- README install and quick-start examples are now platform-neutral
- All project URLs now point to correct GitHub repository
- Packaging metadata now includes classifiers, URLs, pytest entry points, coverage config, and typed package data

## [0.1.0] - 2026-03-26

### Added
- Initial extraction from an internal step-log-analysis workflow
- Agent step schema and task loading utilities
- Task analysis with metrics and grading
- Configurable JSON grading rules
- Root-cause classification
- Decision-surface reconstruction
- Golden fixture capture and replay
- CLI for analysis, grading, replay, and surface inspection
- MIT license
