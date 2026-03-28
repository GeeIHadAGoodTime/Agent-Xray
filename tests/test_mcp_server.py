from __future__ import annotations

import importlib
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest


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
    module = importlib.import_module("agent_xray.mcp_server")
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
    module = importlib.import_module("agent_xray.mcp_server")
    called: list[str] = []
    monkeypatch.setattr(module.server, "run", lambda transport="stdio": called.append(transport))
    module.main()
    assert called == ["stdio"]


def test_enforce_init_tool(tmp_path: Path) -> None:
    import agent_xray.mcp_server as mcp_server

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
    import agent_xray.mcp_server as mcp_server

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

    assert payload["decision"] == "COMMITTED"
    assert payload["after"]["failed"] == 0
    assert payload["commit_hash"]


def test_enforce_init_tool_passes_stash_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_xray.mcp_server as mcp_server
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
    import agent_xray.mcp_server as mcp_server

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
    import agent_xray.mcp_server as mcp_server

    payload = json.loads(mcp_server.analyze(str(tmp_trace_dir)))

    assert payload["summary"]["tasks"] == 4
    assert payload["summary"]["grade_distribution"]["BROKEN"] >= 1
    assert len(payload["tasks"]) == 4
    assert "analysis" in payload["tasks"][0]


def test_grade_tool(tmp_trace_dir: Path) -> None:
    import agent_xray.mcp_server as mcp_server

    payload = json.loads(mcp_server.grade(str(tmp_trace_dir)))

    assert payload["summary"]["tasks"] == 4
    assert payload["summary"]["distribution"]["GOLDEN"] >= 1
    assert any(task["grade"] == "BROKEN" for task in payload["tasks"])


def test_mcp_tool_descriptions_include_flow_guidance() -> None:
    """Every MCP tool docstring should include a 'Next step:' hint to guide analysis flow."""
    module = importlib.import_module("agent_xray.mcp_server")

    tool_names = [
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
    for name in tool_names:
        tool_func = getattr(module, name)
        doc = tool_func.__doc__
        assert doc is not None, f"{name} has no docstring"
        assert "Next step:" in doc, (
            f"{name} docstring is missing 'Next step:' flow guidance"
        )


def test_analyze_description_says_start_here() -> None:
    """The analyze tool should clearly signal it is the entry point for trace triage."""
    module = importlib.import_module("agent_xray.mcp_server")

    doc = module.analyze.__doc__
    assert doc is not None
    assert "Start here" in doc
