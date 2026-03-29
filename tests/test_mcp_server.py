from __future__ import annotations

import importlib
import json
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest


MCP_TOOL_NAMES = [
    "enforce_init",
    "enforce_check",
    "enforce_diff",
    "enforce_plan",
    "enforce_guard",
    "enforce_status",
    "enforce_challenge",
    "enforce_reset",
    "enforce_report",
    "analyze",
    "grade",
    "root_cause",
    "completeness",
    "surface_task",
    "search_tasks",
    "diagnose",
    "compare_runs",
    "report",
    "diff_tasks",
    "reasoning",
    "tree",
    "golden_rank",
    "golden_compare",
    "task_bank_validate",
    "task_bank_list",
    "flywheel",
    "capture_task",
    "pricing_show",
]

BANNED_DOCSTRING_MARKERS = (
    "High-value path:",
    "Next step:",
    "Common mistakes:",
    "Prerequisites:",
)


def _load_mcp_server_module():
    sys.modules.pop("agent_xray.mcp_server", None)
    return importlib.import_module("agent_xray.mcp_server")


def _create_git_repo(repo: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.name", "Agent Xray Tests"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "agent-xray-tests@example.com"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    (repo / "calc.py").write_text(
        "def add(a: int, b: int) -> int:\n"
        "    return a - b\n",
        encoding="utf-8",
    )
    tests_dir = repo / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_calc.py").write_text(
        "from calc import add\n\n\n"
        "def test_add() -> None:\n"
        "    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial broken fixture"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def _pytest_command() -> str:
    return subprocess.list2cmdline([sys.executable, "-m", "pytest", "tests", "-q"])


def test_server_module_imports_without_error() -> None:
    module = _load_mcp_server_module()
    assert module.server is not None
    assert callable(module.enforce_init)
    assert callable(module.analyze)


def test_server_module_can_import_with_mocked_fastmcp(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeFastMCP:
        def __init__(self, name: str) -> None:
            self.name = name
            self.tools: dict[str, object] = {}

        def tool(self, *args, **kwargs):
            def decorator(func):
                self.tools[func.__name__] = func
                return func

            return decorator

        def run(self, transport: str = "stdio") -> None:
            self.transport = transport

    fake_mcp = types.ModuleType("mcp")
    fake_server = types.ModuleType("mcp.server")
    fake_fastmcp = types.ModuleType("mcp.server.fastmcp")
    fake_fastmcp.FastMCP = FakeFastMCP

    monkeypatch.setitem(sys.modules, "mcp", fake_mcp)
    monkeypatch.setitem(sys.modules, "mcp.server", fake_server)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fake_fastmcp)
    monkeypatch.delitem(sys.modules, "agent_xray.mcp_server", raising=False)

    module = importlib.import_module("agent_xray.mcp_server")
    assert module.server.name == "agent-xray"
    assert "enforce_init" in module.server.tools
    assert "grade" in module.server.tools

    monkeypatch.delitem(sys.modules, "agent_xray.mcp_server", raising=False)
    importlib.import_module("agent_xray.mcp_server")


def test_main_runs_stdio_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_mcp_server_module()
    called: list[str] = []
    monkeypatch.setattr(module.server, "run", lambda transport="stdio": called.append(transport))
    module.main()
    assert called == ["stdio"]


def test_enforce_init_tool(tmp_path: Path) -> None:
    mcp_server = _load_mcp_server_module()

    _create_git_repo(tmp_path)

    payload = json.loads(
        mcp_server.enforce_init(
            test_command=_pytest_command(),
            project_root=str(tmp_path),
            max_iterations=5,
        )
    )

    assert payload["baseline"]["failed"] == 1
    assert Path(payload["session_dir"]).exists()


def test_enforce_check_tool(tmp_path: Path) -> None:
    mcp_server = _load_mcp_server_module()

    _create_git_repo(tmp_path)
    json.loads(
        mcp_server.enforce_init(
            test_command=_pytest_command(),
            project_root=str(tmp_path),
            max_iterations=5,
        )
    )

    (tmp_path / "calc.py").write_text(
        "def add(a: int, b: int) -> int:\n"
        "    return a + b\n",
        encoding="utf-8",
    )

    payload = json.loads(
        mcp_server.enforce_check(
            hypothesis="Fix add arithmetic bug",
            project_root=str(tmp_path),
        )
    )

    assert payload["decision"] == "RECOMMEND_COMMIT"
    assert payload["after"]["failed"] == 0
    assert payload["commit_hash"] is None  # Socratic: never auto-commits


def test_enforce_init_tool_passes_stash_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mcp_server = _load_mcp_server_module()
    from agent_xray.enforce import TestResult

    captured: dict[str, object] = {}

    def mock_enforce_init(config):
        captured["stash_first"] = config.stash_first
        return (
            TestResult(
                exit_code=0,
                passed=1,
                failed=0,
                errors=0,
                skipped=0,
                total=1,
                duration_seconds=0.1,
                output="ok",
            ),
            Path("session-dir"),
        )

    monkeypatch.setattr("agent_xray.enforce.enforce_init", mock_enforce_init)

    payload = json.loads(
        mcp_server.enforce_init(
            test_command="pytest",
            project_root=".",
            stash_first=True,
        )
    )

    assert captured["stash_first"] is True
    assert payload["session_dir"] == "session-dir"


def test_enforce_diff_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    mcp_server = _load_mcp_server_module()

    monkeypatch.setattr(
        "agent_xray.enforce.enforce_diff",
        lambda project_root=".": {
            "files": ["src/foo.py"],
            "file_count": 1,
            "diff_lines": ["+new line"],
            "diff_line_count": 1,
            "would_reject": False,
            "reject_reason": "",
        },
    )

    payload = json.loads(mcp_server.enforce_diff(project_root="."))

    assert payload["file_count"] == 1
    assert payload["would_reject"] is False
    assert payload["diff_lines"] == ["+new line"]


def test_analyze_tool(tmp_trace_dir: Path) -> None:
    mcp_server = _load_mcp_server_module()

    payload = json.loads(mcp_server.analyze(str(tmp_trace_dir)))

    assert payload["summary"]["tasks"] == 4
    assert payload["summary"]["grade_distribution"]["BROKEN"] >= 1
    assert "worst_tasks" in payload
    assert len(payload["worst_tasks"]) <= 10
    assert "grade" in payload["worst_tasks"][0]


def test_analyze_tool_uses_task_bank_when_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mcp_server = _load_mcp_server_module()

    tasks = [types.SimpleNamespace(task_id="task-1", task_text="Find docs", steps=[])]
    grade = types.SimpleNamespace(
        task_id="task-1",
        grade="BROKEN",
        score=1,
        reasons=["missing criterion"],
    )
    calls: dict[str, object] = {}

    monkeypatch.setattr(mcp_server, "_load_tasks", lambda log_dir, format: tasks)
    monkeypatch.setattr(
        "agent_xray.grader.load_rules",
        lambda rules=None: types.SimpleNamespace(name="fake-rules"),
    )

    def fail_grade_tasks(*args, **kwargs):
        raise AssertionError("grade_tasks should not run when task_bank is provided")

    def fake_grade_with_task_bank(tasks_arg, task_bank_arg, rule_set_arg):
        calls["tasks"] = tasks_arg
        calls["task_bank"] = task_bank_arg
        calls["rules"] = rule_set_arg.name
        return [grade]

    monkeypatch.setattr("agent_xray.grader.grade_tasks", fail_grade_tasks)
    monkeypatch.setattr(
        "agent_xray.contrib.task_bank.grade_with_task_bank",
        fake_grade_with_task_bank,
    )
    monkeypatch.setattr(
        "agent_xray.analyzer.analyze_task",
        lambda task: types.SimpleNamespace(site_name="docs.example.com"),
    )

    payload = json.loads(mcp_server.analyze("traces", task_bank="task_bank.json"))

    assert calls == {
        "tasks": tasks,
        "task_bank": "task_bank.json",
        "rules": "fake-rules",
    }
    assert payload["summary"]["grade_distribution"]["BROKEN"] == 1
    assert payload["worst_tasks"][0]["site"] == "docs.example.com"


def test_grade_tool(tmp_trace_dir: Path) -> None:
    mcp_server = _load_mcp_server_module()

    payload = json.loads(mcp_server.grade(str(tmp_trace_dir)))

    assert payload["summary"]["tasks"] == 4
    assert payload["summary"]["distribution"]["GOLDEN"] >= 1
    assert any(task["grade"] == "BROKEN" for task in payload["worst_tasks"])


def test_report_tools_skips_grading(monkeypatch: pytest.MonkeyPatch) -> None:
    mcp_server = _load_mcp_server_module()

    tasks = [types.SimpleNamespace(task_id="task-1")]
    analyses = {"task-1": types.SimpleNamespace(site_name="example.com")}

    monkeypatch.setattr(mcp_server, "_load_tasks", lambda log_dir, format: tasks)
    monkeypatch.setattr(
        "agent_xray.analyzer.analyze_tasks",
        lambda tasks_arg: analyses,
    )
    monkeypatch.setattr(
        "agent_xray.reports.report_tools_data",
        lambda tasks_arg, analyses_arg: {"tool_count": len(tasks_arg), "sites": list(analyses_arg)},
    )

    def fail_load_rules(*args, **kwargs):
        raise AssertionError("load_rules should not run for tools reports")

    def fail_grade_tasks(*args, **kwargs):
        raise AssertionError("grade_tasks should not run for tools reports")

    monkeypatch.setattr("agent_xray.grader.load_rules", fail_load_rules)
    monkeypatch.setattr("agent_xray.grader.grade_tasks", fail_grade_tasks)

    payload = json.loads(mcp_server.report("traces", report_type="tools"))

    assert payload["rules"] == "default"
    assert payload["data"] == {"tool_count": 1, "sites": ["task-1"]}


def test_search_tasks_caps_matches_and_uses_last_page_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mcp_server = _load_mcp_server_module()

    tasks = [
        types.SimpleNamespace(
            task_id=f"task-{index}",
            task_text="Checkout order for customer",
            steps=[
                types.SimpleNamespace(page_url=f"https://first-{index}.example.com/start", browser=None),
                types.SimpleNamespace(page_url=None, browser={"page_url": f"https://last-{index}.example.com/checkout"}),
            ],
            outcome=types.SimpleNamespace(status="success"),
        )
        for index in range(30)
    ]

    monkeypatch.setattr(mcp_server, "_load_tasks", lambda log_dir, format: tasks)

    payload = json.loads(mcp_server.search_tasks("traces", query="checkout"))

    assert payload["shown"] == 25
    assert payload["total_matches"] == 25
    assert len(payload["matches"]) == 25
    assert payload["matches"][0]["site"] == "last-0.example.com"
    assert "Stopped after 25 matches" in payload["note"]


def test_enforce_check_truncates_large_test_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mcp_server = _load_mcp_server_module()

    long_output = "x" * 800
    monkeypatch.setattr(
        "agent_xray.enforce.enforce_check",
        lambda hypothesis="", project_root=".": {
            "decision": "RECOMMEND_COMMIT",
            "before": {
                "passed": 1,
                "failed": 1,
                "errors": 0,
                "total": 2,
                "output": long_output,
                "test_names_failed": ["tests/test_calc.py::test_add"],
            },
            "after": {
                "passed": 2,
                "failed": 0,
                "errors": 0,
                "total": 2,
                "output": long_output,
                "test_names_passed": ["tests/test_calc.py::test_add"],
            },
        },
    )

    payload = json.loads(mcp_server.enforce_check(project_root="."))

    assert len(payload["before"]["output"]) == 500
    assert payload["before"]["test_names_failed"] == ["tests/test_calc.py::test_add"]
    assert len(payload["after"]["output"]) == 500
    assert payload["after"]["test_names_passed"] == ["tests/test_calc.py::test_add"]


def test_enforce_status_truncates_large_test_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mcp_server = _load_mcp_server_module()

    long_output = "y" * 900
    monkeypatch.setattr(
        "agent_xray.enforce.enforce_status",
        lambda project_root=".": {
            "session_active": True,
            "last_result": {
                "passed": 3,
                "failed": 1,
                "errors": 0,
                "total": 4,
                "output": long_output,
                "test_names_failed": ["tests/test_api.py::test_status"],
            },
        },
    )

    payload = json.loads(mcp_server.enforce_status(project_root="."))

    assert payload["session_active"] is True
    assert len(payload["last_result"]["output"]) == 500
    assert payload["last_result"]["test_names_failed"] == ["tests/test_api.py::test_status"]


def test_tree_includes_sample_task_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    mcp_server = _load_mcp_server_module()

    monkeypatch.setattr(
        mcp_server,
        "_load_tasks",
        lambda log_dir, format: [types.SimpleNamespace(task_id="task-1")],
    )
    monkeypatch.setattr(
        "agent_xray.surface.enriched_tree_for_tasks",
        lambda tasks, grades=None: {
            "20260329": {
                "shop.example.com": [
                    {"task_id": "t1", "outcome": "success"},
                    {"task_id": "t2", "outcome": "failed"},
                    {"task_id": "t3", "outcome": "failed"},
                    {"task_id": "t4", "outcome": "failed"},
                ]
            }
        },
    )

    payload = json.loads(mcp_server.tree("traces"))

    site = payload["tree"]["20260329"]["shop.example.com"]
    assert site["count"] == 4
    assert site["sample_task_ids"] == ["t1", "t2", "t3"]
    assert site["success"] == 1
    assert site["failed"] == 3


def test_mcp_tool_descriptions_are_trimmed_for_schema_efficiency() -> None:
    """Every MCP tool docstring should stay compact because it is sent in the MCP schema."""
    module = _load_mcp_server_module()

    for name in MCP_TOOL_NAMES:
        tool_func = getattr(module, name)
        doc = tool_func.__doc__
        assert doc is not None, f"{name} has no docstring"
        stripped = doc.strip()
        assert "\n" not in stripped, f"{name} docstring should be a single compact line"
        for marker in BANNED_DOCSTRING_MARKERS:
            assert marker not in stripped, f"{name} docstring should omit {marker!r}"
        sentence_count = len(re.findall(r"[.!?](?:\s|$)", stripped))
        assert 1 <= sentence_count <= 2, (
            f"{name} docstring should be 1-2 sentences, got {sentence_count}: {stripped!r}"
        )


def test_analyze_description_stays_purpose_focused() -> None:
    """The analyze tool docstring should keep the core purpose without workflow guidance."""
    module = _load_mcp_server_module()

    doc = module.analyze.__doc__
    assert doc is not None
    assert "Analyze agent traces" in doc
    assert "Start here" not in doc
