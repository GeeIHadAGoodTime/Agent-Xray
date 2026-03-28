# Contributing to agent-xray

## The Core Principle

The project IS the documentation. The code is the implementation; the README is the product. An undocumented feature isn't a feature -- it's an easter egg that nobody finds.

Every user-visible feature MUST be documented in README.md before or at the time of merge. No exceptions. There is an automated test (`tests/test_readme_coverage.py`) that enforces this -- your PR will fail CI if you add a public feature without documenting it.

## What Counts as "Documented"

A feature must appear in the correct README section:

| You added... | It must appear in... |
|---|---|
| CLI command | **Full CLI Reference** tables |
| MCP tool | **MCP Server Tools** table |
| Signal detector | **Signal Detector Packs** table |
| Public API export | **Library Usage** section |
| Entry point / script | **Install** or **Quick Start** section |
| Adapter / framework support | **Supported Frameworks** table |
| Report type | **Reports, Watch Mode, and Completeness** section |
| Root cause category | **Root Cause Classification** table |

If you're unsure which section, read the README. It is comprehensive by design.

## Development Setup

```bash
git clone https://github.com/GeeIHadAGoodTime/Agent-Xray.git
cd Agent-Xray
python -m pip install -e ".[all]"
```

## Testing

All of these must pass before you open a PR:

```bash
pytest tests/ -q
ruff check src tests
ruff format --check src tests
mypy src/agent_xray --strict
```

The core library has zero runtime dependencies. Keep it that way. Optional extras (`[runner]`, `[tui]`, `[otel]`, etc.) exist for features that need third-party packages.

## Style

- Python 3.10+
- `ruff` for linting and formatting
- `mypy --strict` for type checking
- Normalize into `AgentStep` at the adapter boundary, not inside the analyzer
- Small explicit helpers over clever abstractions

## Enforce Mode for Bug Fixes

If your PR fixes a bug, use `agent-xray enforce` to prove it:

```bash
agent-xray enforce init --test "pytest tests/ -q"
# make your fix
agent-xray enforce check --hypothesis "description of the root cause"
```

This captures before/after evidence that the fix works and didn't break anything else. Include the enforce report in your PR description.

## PR Expectations

- Focused. One concern per PR.
- If you changed the CLI, the README tables are updated.
- If you added a detector, it has tests for both `detect_step()` and `summarize()`.
- If you added an adapter, it has a fixture file under `tests/fixtures/`.
- `pytest`, `ruff`, and `mypy` pass locally for the files you touched.
- No drive-by reformatting of unrelated files.
