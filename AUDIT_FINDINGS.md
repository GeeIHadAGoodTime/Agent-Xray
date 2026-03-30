# agent-xray v1.7.2 Audit Findings

Audit date: 2026-03-28  
Repo version audited: `1.7.2`

This file covers:

1. Reverse audit: capabilities present in code but absent from `README.md`
2. Claims vs reality: whether README claims match the shipped code

## Part 1: Reverse Audit

### Undocumented CLI and entrypoint surfaces

The README quick start covers `quickstart`, `analyze`, `surface`, `grade`, `diff`, `flywheel`, `compare`, and `tui`, but the CLI exposes substantially more surface area in [`src/agent_xray/cli.py`](src/agent_xray/cli.py):

- Hidden top-level commands:
  - `report` with 15 report types: `health`, `golden`, `broken`, `tools`, `flows`, `outcomes`, `actions`, `coding`, `research`, `cost`, `fixes`, `timeline`, `spins`, `compare`, `overhead`, `prompt-impact` (`cli.py:2443-2488`)
  - `completeness` (`cli.py:2490-2501`)
  - `diagnose` (`cli.py:2503-2521`)
  - `validate-targets` (`cli.py:2523-2537`)
  - `watch` (`cli.py:2578-2592`)
  - `record` (`cli.py:2651-2672`)
  - `rules` management: `list`, `show`, `init` (`cli.py:2674-2695`)
  - `pricing` management: `list`, `show`, `update`, `path` (`cli.py:2697-2729`)
  - `baseline` management: `capture`, `generate`, `list` (`cli.py:2594-2641`)
  - `golden` management: `rank`, `best`, `capture`, `compare`, `profiles` (`cli.py:2731-2846`)

- Hidden `report` capabilities:
  - grade/site/outcome/since filters (`cli.py:2477-2479`)
  - `--bucket` for timeline bucketing (`cli.py:2471-2475`)
  - `--day1` and `--day2` for day-over-day compare reports (`cli.py:2480-2481`)
  - `--baselines` for overhead reports (`cli.py:2482-2485`)
  - `--expected-rejection` to suppress specific `tool_rejection_mismatch` classifications (`cli.py:193-205`, `cli.py:2479`)

- Hidden `surface`/`diff`/`tree`/`capture`/`replay`/`search` affordances:
  - `diff --summary` (`cli.py:2318`)
  - `tree --rules` to enrich the tree with grades and scores (`cli.py:2355-2358`)
  - `capture --no-sanitize` (`cli.py:2378-2381`)
  - `search --grade` (`cli.py:2557-2560`)

- Hidden `enforce` capabilities relative to README:
  - `enforce diff` exists and is wired (`cli.py:2913-2921`)
  - `enforce init` and `enforce auto` both accept `--max-files-per-change`, `--max-diff-lines`, and `--rules-file` (`cli.py:3013-3024`, `cli.py:3060-3070`)
  - `enforce init` accepts `--stash-first` (`cli.py:2892-2895`)
  - `enforce plan` supports `--expected-tests` (`cli.py:3036-3040`)

- Hidden trace-format support:
  - the CLI supports `--format openai_chat`, but the README format table omits it (`cli.py:88-96`, `README.md:170-178`)

- Hidden published entrypoints:
  - `agent-xray-mcp` is packaged as a console script in [`pyproject.toml`](pyproject.toml) but not documented in the README (`pyproject.toml:127-128`)
  - the pytest plugin entrypoint is packaged in [`pyproject.toml`](pyproject.toml) but README only advertises the feature, not how to activate or use it (`pyproject.toml:130-131`)

### Undocumented MCP server surface

The package ships a full MCP server in [`src/agent_xray/mcp_server.py`](src/agent_xray/mcp_server.py), but the README has no MCP server section and no tool inventory.

Undocumented MCP tools:

- enforcement tools:
  - `enforce_init` (`mcp_server.py:43-72`)
  - `enforce_check` (`mcp_server.py:74-84`)
  - `enforce_diff` (`mcp_server.py:86-95`)
  - `enforce_plan` (`mcp_server.py:97-116`)
  - `enforce_guard` (`mcp_server.py:118-127`)
  - `enforce_status` (`mcp_server.py:129-138`)
  - `enforce_challenge` (`mcp_server.py:140-150`)
  - `enforce_reset` (`mcp_server.py:152-161`)
  - `enforce_report` (`mcp_server.py:163-187`)

- analysis tools:
  - `analyze` (`mcp_server.py:189-223`)
  - `grade` (`mcp_server.py:225-250`)
  - `root_cause` (`mcp_server.py:252-277`)
  - `completeness` (`mcp_server.py:279-298`)

### Undocumented library exports and module capabilities

#### `src/agent_xray/__init__.py`

The README documents a small subset of the library API. The package exports many more public symbols:

- completeness API:
  - `CompletenessReport`, `CompletenessWarning`, `check_completeness`
- diagnose / fix-plan API:
  - `DefaultTargetResolver`, `FIX_TARGETS`, `FixPlanEntry`, `INVESTIGATION_HINTS`, `TargetResolver`, `build_fix_plan`, `format_fix_plan_text`, `get_target_resolver`, `list_all_targets`, `register_target_resolver`, `validate_fix_targets`
- baseline / overhead API:
  - `Baseline`, `OverheadResult`, `PromptHashGroup`, `build_baseline`, `format_overhead_report`, `format_prompt_impact_report`, `generate_naked_prompt`, `group_by_prompt_hash`, `measure_overhead`
- comparison API:
  - `ModelComparisonResult`, `compare_model_runs`, `format_model_comparison`
- signals API:
  - `SignalDetector`, `discover_detectors`, `run_detection`
- enforce report API:
  - `format_enforce_json`, `format_enforce_markdown`, `generate_report`
- MCP exports:
  - `mcp_server`, `mcp_main`

#### `src/agent_xray/enforce_audit.py`

The README documents the 8 detector names, but the module ships more behavior than is described:

- `detect_rule_violations` adds project-rule enforcement on top of gaming detection (`enforce_audit.py:631-706`)
- `classify_diff_quality` classifies changes as `behavioral_improvement`, `bug_fix`, `refactor`, `test_improvement`, or `neutral` (`enforce_audit.py:567-628`)
- `analyze_successful_changes` summarizes what successful committed changes tend to look like (`enforce_audit.py:721-804`)
- `classify_change_quality` and `quality_distribution` grade change records into `EXCELLENT/GOOD/NEUTRAL/POOR/HARMFUL` (`enforce_audit.py:807-879`)

Also, `challenge_iterations()` runs more checks than the README lists:

- undocumented checks present in code:
  - consecutive suspicious/gaming streaks (`enforce_audit.py:350-359`)
  - net regression over the reviewed range (`enforce_audit.py:361-372`)
  - hot files modified repeatedly (`enforce_audit.py:374-382`)
  - vetoing unreverted gaming iterations (`enforce_audit.py:384-389`)
  - cumulative gaming signal density (`enforce_audit.py:462-471`)

#### `src/agent_xray/root_cause.py`

The README documents the 19 labels, but not these exposed features:

- numeric confidence scoring via `confidence_score` in `RootCauseResult` (`root_cause.py:268-293`)
- prompt attribution fields `prompt_section` and `prompt_fix_hint` (`root_cause.py:280-281`, `root_cause.py:1014-1066`)
- `ClassificationConfig.expected_rejections` to suppress expected policy rejections (`root_cause.py:490-494`)
- `format_root_causes_text()` (`root_cause.py:1112-1136`)

#### `src/agent_xray/signals/__init__.py`

The README says signal detection is pluggable, but it does not document:

- dynamic plugin discovery through the `agent_xray.signals` entry-point group (`signals/__init__.py:29-40`, `signals/__init__.py:42-61`)
- the built-in detector set now includes:
  - `commerce`
  - `coding`
  - `research`
  - `multi_agent`
  - `memory`
  - `planning`
  (`signals/__init__.py:85-94`)

The README only calls out commerce/coding/research detector packs in the architecture section.

#### `src/agent_xray/completeness.py`

The entire completeness subsystem is undocumented in README. It checks:

- `outcome_records`
- `tool_schemas`
- `model_name`
- `cache_tokens`
- `final_answer`
- `system_prompt`
- `rejected_tools`
- `approval_path`
- `conversation_history`
- `step_durations`
- `system_context`
- `llm_reasoning`
- `step_data_loss`
- optional `target_validity` when `project_root` is supplied

See [`completeness.py:16-441`](src/agent_xray/completeness.py).

#### `src/agent_xray/golden.py`

The README mentions golden replay and exemplar ranking, but not:

- optimization profiles `balanced`, `cost`, `speed`, `steps` (`golden.py:16-21`)
- `compute_efficiency()` as a public scoring primitive (`golden.py:117-147`)
- `find_exemplars()` (`golden.py:284-307`)
- `explain_efficiency_gap()` (`golden.py:309-410`)
- `capture_exemplar()` writing exemplar fixtures with `efficiency_metadata` (`golden.py:468-522`)

#### `src/agent_xray/baseline.py`

README mentions baseline deltas inside flywheel, but not the baseline module itself:

- persistence helpers:
  - `save_baseline`, `load_baseline`, `load_baselines`
- full overhead system:
  - `measure_overhead`, `measure_all_overhead`
  - overhead categories `efficient`, `acceptable`, `bloated`, `pathological`
- prompt-hash grouping:
  - `group_by_prompt_hash`
  - `PromptHashGroup`
  - `format_prompt_impact_report`
  - `prompt_impact_data`
- structured report payload helpers:
  - `overhead_report_data`

See [`baseline.py:15-836`](src/agent_xray/baseline.py).

#### `src/agent_xray/comparison.py`

The README mentions model comparison at the CLI level, but not these library-level structures:

- `DivergencePoint`
- `ModelCostSummary`
- `ModelComparisonResult`
- decision-surface divergence notes and per-side cost summaries

See [`comparison.py:15-254`](src/agent_xray/comparison.py).

#### `src/agent_xray/diagnose.py`

The diagnose / fix-plan subsystem is not documented in README:

- target resolver protocol and registration:
  - `TargetResolver`
  - `DefaultTargetResolver`
  - `register_target_resolver`
  - `get_target_resolver`
- fix-plan generation:
  - `FixPlanEntry`
  - `build_fix_plan`
  - `format_fix_plan_text`
- target inventory and validation:
  - `list_all_targets`
  - `validate_fix_targets`
  - `FIX_TARGETS`, `INVESTIGATION_HINTS`, `CODE_EXTENSIONS`

#### `src/agent_xray/surface.py`

The README documents `surface`, `reasoning`, `tree`, and `diff`, but not these extra capabilities:

- surface completeness scoring and `missing_surfaces` per step (`surface.py:149-248`, `surface.py:386-476`)
- memory / RAG fields in surfaces (`surface.py:108-146`, `surface.py:449-455`)
- aligned diffing with `step_alignment`, `similarity_score`, and `divergence_point` (`surface.py:276-327`, `surface.py:525-576`)
- prompt-only diffs via `format_prompt_diff()` (`surface.py:757-775`)
- enriched trees with grades and scores via `enriched_tree_for_tasks()` and `format_enriched_tree_text()` (`surface.py:790-882`)

#### `src/agent_xray/replay.py`

README mentions replay, but not these details:

- fixture matching by `task_id`, fuzzy text similarity, and site hint (`replay.py:45-64`)
- `EVALUATION_DRIFT` detection when integrity hashes change (`replay.py:67-95`)
- explicit verdicts:
  - `IMPROVED`
  - `REGRESSION`
  - `STABLE`
  - `UNMATCHED`
  - `EVALUATION_DRIFT`

#### `src/agent_xray/enforce_report.py`

README says reports can be text/json/markdown. It does not document that the report module also provides:

- project-rule loading and diff checking:
  - `load_project_rules`
  - `check_against_rules`
  - `format_rules_violations`
- letter grading for an entire enforce session (`grade_enforce_session`)
- prediction-accuracy reporting (`format_enforce_text` / `format_enforce_markdown`)
- detailed change maps and timelines in JSON/text/Markdown (`enforce_report.py:246-285`, `enforce_report.py:323-784`)

### Dead or stale README claims

- README says the enforce decision is `COMMIT`, `REVERT`, or `REJECTED`, but the code returns `COMMITTED`, `REVERTED`, or `REJECTED`.
  - README: `README.md:387`
  - code/tests: `tests/test_cli.py:319-381`, `src/agent_xray/enforce.py` change records consumed by `cli.py`

- README says `enforce challenge` runs “9 cross-iteration checks”, but the bullet list under that heading contains 8 items and the code currently performs more than 9 checks.
  - README: `README.md:328-336`
  - code: `enforce_audit.py:327-564`

- README’s “Supported Frameworks” table omits the shipped `openai_chat` adapter/format.
  - README: `README.md:170-178`
  - code: `cli.py:88-96`, `tests/test_adapters.py:31-95`

- README has no MCP server section even though `agent-xray-mcp` is a packaged entrypoint and `mcp_server.py` exposes 48 tools.
  - README: no mention of `agent-xray-mcp`
  - code: `pyproject.toml:127-128`, `mcp_server.py:43-300`

- README’s architecture section says `signals/` contains detector packs for commerce, coding, and research, but the built-in detector set also includes planning, memory, and multi-agent detectors.
  - README: `README.md:449`
  - code: `signals/__init__.py:85-94`

## Part 2: Claims vs Reality

### Documented CLI commands

All README-documented top-level commands do exist:

- `quickstart`
- `analyze`
- `surface`
- `grade`
- `diff`
- `flywheel`
- `compare`
- `tui`
- `enforce init`
- `enforce check`
- `enforce challenge`
- `enforce status`
- `enforce report`
- `enforce reset`
- `enforce plan`
- `enforce guard`
- `enforce auto`

### Parameter names, types, and defaults

Verified against [`src/agent_xray/cli.py`](src/agent_xray/cli.py):

- `grade --rules` defaults to the bundled default rules file, not the bare string `"default"` (`cli.py:2327-2338`). The effective ruleset name is still `default`, so user behavior matches the README.
- `enforce init` defaults:
  - `--project-root .`
  - `--max-iterations 50`
  - `--challenge-every 5`
  - `--max-files-per-change 5`
  - `--max-diff-lines 200`
- `enforce report --format` defaults to `text` and supports `text|json|markdown` (`cli.py:2943-2958`)
- `enforce auto` requires both `--test` and `--agent-cmd` (`cli.py:2969-3026`)

No broken parser wiring was found for documented commands.

### Example verification

Commands executed successfully on 2026-03-28 against the repo checkout:

- `python -m agent_xray --version`
  - returned `agent-xray 1.7.2`
- `python -m agent_xray quickstart`
  - succeeded and created a demo trace directory
- `python -m agent_xray analyze <quickstart-dir>`
  - succeeded
- `python -m agent_xray surface broken-task --log-dir <quickstart-dir>`
  - succeeded
- `python -m agent_xray grade <quickstart-dir>`
  - succeeded
- `python -m agent_xray report <quickstart-dir> health`
  - succeeded

Observations:

- The README’s concrete quick-start commands are valid.
- Several README examples use placeholder IDs or placeholder directories (`task-a`, `task-b`, `./runs-gpt4`, `./runs-gpt5`). Those are illustrative, not copy-paste runnable without matching local data.

### Version consistency

Versioning was consistent before modification:

- [`pyproject.toml`](pyproject.toml): `1.7.2`
- [`src/agent_xray/__init__.py`](src/agent_xray/__init__.py): `1.7.2`
- `python -m agent_xray --version`: `1.7.2`
