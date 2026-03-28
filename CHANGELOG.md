# Changelog

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
