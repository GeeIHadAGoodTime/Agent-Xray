# Changelog

## [0.2.0] - Unreleased

### Added
- Layered schema: core step fields plus typed extension contexts for model, tools, reasoning, and browser state
- Built-in adapters for generic JSONL, OpenAI, LangChain, Anthropic, CrewAI, and OTel-style traces
- Auto-detect trace format from file contents
- `--format` support across the CLI
- Pluggable signal detection with bundled commerce, coding, and research detector packs
- Interactive TUI for decision-surface replay
- GitHub Actions CI with lint, typecheck, test matrix, and PyPI publish workflow
- pytest plugin for agent trace assertions
- Flywheel baseline comparison and fixture replay
- Cost-per-decision tracking via `ModelContext`
- Optional dependency extras for runner, OTel, TUI, and full dev installs
- `py.typed` marker for downstream type checkers

### Changed
- Schema evolved from a flat step record toward typed contexts while preserving `AgentStep.from_dict()` compatibility
- Adapters moved from example-only usage into first-class package modules
- Public docs now reflect the real CLI surface, optional extras, and extension points

### Fixed
- `mypy --strict` support for the core package configuration used in CI
- README install and quick-start examples are now platform-neutral
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
