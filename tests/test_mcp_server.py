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
    "replay",
    "validate_targets",
    "rules_list",
    "rules_show",
    "rules_init",
    "baseline_capture",
    "baseline_list",
    "golden_best",
    "golden_profiles",
    "pricing_list",
    "baseline_generate",
    "task_bank_show",
    "format_detect",
    "triage",
    "gaming_audit",
    "pricing_update",
    "inspect_task",
    "signal_detect",
    "match_task",
    "golden_capture",
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

    monkeypatch.setattr(mcp_server, "_load_tasks", lambda log_dir, format, **kw: tasks)
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

    monkeypatch.setattr(mcp_server, "_load_tasks", lambda log_dir, format, **kw: tasks)
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

    monkeypatch.setattr(mcp_server, "_load_tasks", lambda log_dir, format, **kw: tasks)

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
        lambda log_dir, format, **kw: [types.SimpleNamespace(task_id="task-1")],
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


def _make_step(
    task_id: str,
    step: int,
    tool_name: str,
    *,
    tool_input: dict[str, object] | None = None,
    tool_result: str | None = None,
    error: str | None = None,
    page_url: str | None = None,
    llm_reasoning: str | None = None,
    timestamp: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "task_id": task_id,
        "step": step,
        "tool_name": tool_name,
        "tool_input": tool_input or {},
        "timestamp": timestamp or f"2026-03-30T12:{step:02d}:00Z",
    }
    if tool_result is not None:
        payload["tool_result"] = tool_result
    if error is not None:
        payload["error"] = error
    if page_url is not None:
        payload["page_url"] = page_url
    if llm_reasoning is not None:
        payload["llm_reasoning"] = llm_reasoning
    return payload


def _make_task(
    task_id: str,
    task_text: str,
    steps: list[dict[str, object]],
    *,
    task_category: str,
    status: str,
    final_answer: str | None,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index, step in enumerate(steps):
        payload = dict(step)
        if index == 0:
            payload.setdefault("user_text", task_text)
            payload.setdefault("task_category", task_category)
        records.append(payload)
    records.append(
        {
            "event": "task_complete",
            "task_id": task_id,
            "status": status,
            "final_answer": final_answer,
            "total_steps": len(steps),
            "total_duration_s": round(len(steps) * 0.5, 2),
            "timestamp": "2026-03-30T12:59:00Z",
        }
    )
    return records


def _write_log(log_dir: Path, tasks: list[list[dict[str, object]]]) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "trace_20260330.jsonl"
    lines = [json.dumps(record, sort_keys=True) for task in tasks for record in task]
    log_path.write_text("\n".join(lines), encoding="utf-8")
    return log_path


def test_triage_returns_grade_distribution(tmp_path: Path) -> None:
    mcp_server = _load_mcp_server_module()

    good_steps = [
        _make_step(
            "checkout-task",
            1,
            "browser_navigate",
            tool_input={"url": "https://shop.example.test/"},
            tool_result="Homepage with wireless headset listing.",
            page_url="https://shop.example.test/",
            llm_reasoning="Open the storefront.",
        ),
        _make_step(
            "checkout-task",
            2,
            "browser_click",
            tool_input={"ref": "add-to-cart"},
            tool_result="Added to cart. Your cart subtotal is $129.",
            page_url="https://shop.example.test/cart",
            llm_reasoning="Add the item to the cart.",
        ),
        _make_step(
            "checkout-task",
            3,
            "browser_fill_ref",
            tool_input={"ref": "shipping-form", "fields": ["address", "zip"], "text": "123 Main St zip 60601"},
            tool_result="Shipping form accepted.",
            page_url="https://shop.example.test/checkout",
            llm_reasoning="Fill the shipping form.",
        ),
        _make_step(
            "checkout-task",
            4,
            "browser_fill_ref",
            tool_input={
                "ref": "payment-form",
                "fields": ["card number", "cvv", "expiration"],
                "text": "4111 1111 1111 1111 123 12/29",
            },
            tool_result="card number cvv expir payment method confirmed",
            page_url="https://shop.example.test/payment",
            llm_reasoning="Enter the payment details.",
        ),
        _make_step(
            "checkout-task",
            5,
            "browser_click",
            tool_input={"ref": "place-order"},
            tool_result="Order confirmation page loaded.",
            page_url="https://shop.example.test/order/confirmation",
            llm_reasoning="Submit the order.",
        ),
    ]
    broken_steps = [
        _make_step(
            "broken-task",
            1,
            "browser_snapshot",
            tool_result="Checkout spinner still visible.",
            error="Timed out waiting for checkout.",
            page_url="https://shop.example.test/checkout",
            llm_reasoning="Check whether checkout recovered.",
        ),
        _make_step(
            "broken-task",
            2,
            "browser_snapshot",
            tool_result="Checkout spinner still visible.",
            error="Timed out waiting for checkout.",
            page_url="https://shop.example.test/checkout",
            llm_reasoning="Retry the checkout snapshot.",
        ),
        _make_step(
            "broken-task",
            3,
            "browser_snapshot",
            tool_result="Checkout spinner still visible.",
            error="Timed out waiting for checkout.",
            page_url="https://shop.example.test/checkout",
            llm_reasoning="The flow still appears stuck.",
        ),
    ]
    log_dir = tmp_path / "triage-logs"
    _write_log(
        log_dir,
        [
            _make_task(
                "checkout-task",
                "Buy the wireless headset and complete checkout on shop.example.test.",
                good_steps,
                task_category="commerce",
                status="success",
                final_answer="Order placed.",
            ),
            _make_task(
                "broken-task",
                "Recover the stuck checkout flow on shop.example.test.",
                broken_steps,
                task_category="commerce",
                status="failed",
                final_answer=None,
            ),
        ],
    )

    payload = json.loads(mcp_server.triage(str(log_dir)))

    grades = payload.get("grades") or payload.get("summary", {}).get("grade_distribution")
    worst_failure = payload.get("worst_failure") or payload.get("worst_task")
    assert isinstance(grades, dict)
    assert grades
    assert worst_failure is not None
    assert worst_failure["task_id"] == "broken-task"


def test_triage_next_hints_use_correct_param_names(tmp_path: Path) -> None:
    """Workflow hints must reference real parameter names so agents can copy-paste them."""
    mcp_server = _load_mcp_server_module()
    log_dir = tmp_path / "hint-logs"
    _write_log(
        log_dir,
        [
            _make_task(
                "broken-task",
                "Buy headset on shop.example.test.",
                [
                    _make_step("broken-task", 1, "browser_snapshot",
                               tool_result="Spinner.", error="Timeout.",
                               page_url="https://shop.example.test/checkout",
                               llm_reasoning="Checking."),
                ],
                task_category="commerce", status="failed", final_answer=None,
            ),
        ],
    )
    payload = json.loads(mcp_server.triage(str(log_dir)))
    hints = payload.get("next", {})
    # compare_runs hint must use left_log_dir/right_log_dir, never old/new
    after_fix = hints.get("after_fix", "")
    assert "left_log_dir=" in after_fix, f"compare_runs hint should use left_log_dir=, got: {after_fix}"
    assert "right_log_dir=" in after_fix, f"compare_runs hint should use right_log_dir=, got: {after_fix}"
    assert "old_log_dir" not in after_fix
    assert "old_dir" not in after_fix
    # deep_dive should use log_dir= and task_id=
    deep = hints.get("deep_dive", "")
    assert "log_dir=" in deep
    assert "task_id=" in deep


def test_grade_next_hint_includes_log_dir(tmp_trace_dir: Path) -> None:
    """grade()'s next hint must include log_dir for diagnose() so agents can execute it."""
    mcp_server = _load_mcp_server_module()
    payload = json.loads(mcp_server.grade(str(tmp_trace_dir)))
    hint = payload.get("next", "")
    # diagnose hint must include log_dir=
    assert "diagnose(log_dir=" in hint, f"grade next hint should include diagnose(log_dir=), got: {hint}"
    # compare_runs must use left_log_dir/right_log_dir
    assert "left_log_dir=" in hint, f"grade next hint should use left_log_dir=, got: {hint}"


def test_diagnose_next_hint_shows_required_params(tmp_trace_dir: Path) -> None:
    """diagnose()'s next hint must show enforce_init's required test_command param."""
    mcp_server = _load_mcp_server_module()
    payload = json.loads(mcp_server.diagnose(str(tmp_trace_dir)))
    hint = payload.get("next", "")
    # enforce_init requires test_command
    assert "test_command=" in hint, f"diagnose next hint should include test_command=, got: {hint}"
    # enforce_plan requires hypothesis
    assert "hypothesis=" in hint, f"diagnose next hint should include hypothesis=, got: {hint}"
    # compare_runs should use correct param names
    assert "left_log_dir=" in hint, f"diagnose next hint should use left_log_dir=, got: {hint}"


def test_inspect_task_returns_comprehensive_report(tmp_path: Path) -> None:
    mcp_server = _load_mcp_server_module()

    log_dir = tmp_path / "inspect-logs"
    _write_log(
        log_dir,
        [
            _make_task(
                "broken-task",
                "Recover the stuck checkout flow on shop.example.test.",
                [
                    _make_step(
                        "broken-task",
                        1,
                        "browser_snapshot",
                        tool_result="Checkout spinner still visible.",
                        error="Timed out waiting for checkout.",
                        page_url="https://shop.example.test/checkout",
                        llm_reasoning="Inspect the checkout page state.",
                    ),
                    _make_step(
                        "broken-task",
                        2,
                        "browser_snapshot",
                        tool_result="Checkout spinner still visible.",
                        error="Timed out waiting for checkout.",
                        page_url="https://shop.example.test/checkout",
                        llm_reasoning="The task is still blocked at checkout.",
                    ),
                ],
                task_category="commerce",
                status="failed",
                final_answer=None,
            )
        ],
    )

    payload = json.loads(mcp_server.inspect_task(str(log_dir), "broken-task"))

    assert isinstance(payload["grade"], str)
    assert payload["root_cause"] is not None
    assert isinstance(payload["steps"], list)
    assert payload["steps"]
    assert isinstance(payload["reasoning_chain"], list)
    assert payload["reasoning_chain"]


def test_signal_detect_all_detectors(tmp_path: Path) -> None:
    mcp_server = _load_mcp_server_module()

    log_dir = tmp_path / "signal-all-logs"
    _write_log(
        log_dir,
        [
            _make_task(
                "checkout-task",
                "Buy the wireless headset and complete checkout on shop.example.test.",
                [
                    _make_step(
                        "checkout-task",
                        1,
                        "browser_navigate",
                        tool_input={"url": "https://shop.example.test/"},
                        tool_result="Homepage with wireless headset listing.",
                        page_url="https://shop.example.test/",
                        llm_reasoning="Open the storefront.",
                    ),
                    _make_step(
                        "checkout-task",
                        2,
                        "browser_click",
                        tool_input={"ref": "add-to-cart"},
                        tool_result="Added to cart. Your cart subtotal is $129.",
                        page_url="https://shop.example.test/cart",
                        llm_reasoning="Add the item to the cart.",
                    ),
                    _make_step(
                        "checkout-task",
                        3,
                        "browser_fill_ref",
                        tool_input={"ref": "payment-form", "fields": ["card number", "cvv"], "text": "4111 1111 1111 1111 123"},
                        tool_result="card number cvv payment method confirmed",
                        page_url="https://shop.example.test/payment",
                        llm_reasoning="Enter payment details.",
                    ),
                ],
                task_category="commerce",
                status="success",
                final_answer="Order placed.",
            )
        ],
    )

    payload = json.loads(mcp_server.signal_detect(str(log_dir), "checkout-task"))

    assert isinstance(payload["detectors_run"], list)
    assert payload["detectors_run"]
    assert "commerce" in payload["detectors_run"]
    assert set(payload["detectors_run"]) == set(payload["signals"])


def test_signal_detect_single_detector(tmp_path: Path) -> None:
    mcp_server = _load_mcp_server_module()

    log_dir = tmp_path / "signal-one-logs"
    _write_log(
        log_dir,
        [
            _make_task(
                "checkout-task",
                "Buy the wireless headset and complete checkout on shop.example.test.",
                [
                    _make_step(
                        "checkout-task",
                        1,
                        "browser_fill_ref",
                        tool_input={"ref": "payment-form", "fields": ["card number", "cvv"], "text": "4111 1111 1111 1111 123"},
                        tool_result="card number cvv payment method confirmed",
                        page_url="https://shop.example.test/payment",
                        llm_reasoning="Enter payment details.",
                    )
                ],
                task_category="commerce",
                status="success",
                final_answer="Order placed.",
            )
        ],
    )

    payload = json.loads(mcp_server.signal_detect(str(log_dir), "checkout-task", detector="commerce"))

    assert payload["detectors_run"] == ["commerce"]
    assert list(payload["signals"]) == ["commerce"]


def test_signal_detect_unknown_detector(tmp_path: Path) -> None:
    mcp_server = _load_mcp_server_module()

    log_dir = tmp_path / "signal-error-logs"
    _write_log(
        log_dir,
        [
            _make_task(
                "checkout-task",
                "Buy the wireless headset and complete checkout on shop.example.test.",
                [
                    _make_step(
                        "checkout-task",
                        1,
                        "browser_snapshot",
                        tool_result="Checkout page loaded.",
                        page_url="https://shop.example.test/checkout",
                        llm_reasoning="Inspect the checkout page.",
                    )
                ],
                task_category="commerce",
                status="success",
                final_answer="Order placed.",
            )
        ],
    )

    payload = json.loads(mcp_server.signal_detect(str(log_dir), "checkout-task", detector="nonexistent"))

    assert "error" in payload
    assert "nonexistent" in payload["error"]


def test_match_task_no_match(tmp_path: Path) -> None:
    mcp_server = _load_mcp_server_module()

    log_dir = tmp_path / "match-logs"
    _write_log(
        log_dir,
        [
            _make_task(
                "research-task",
                "Research agent observability best practices and summarize the findings.",
                [
                    _make_step(
                        "research-task",
                        1,
                        "read_url",
                        tool_input={"url": "https://docs.example.test/observability"},
                        tool_result="Observability guide loaded.",
                        page_url="https://docs.example.test/observability",
                        llm_reasoning="Read the documentation first.",
                    )
                ],
                task_category="research",
                status="success",
                final_answer="Summary delivered.",
            )
        ],
    )
    bank_path = tmp_path / "task-bank.json"
    bank_path.write_text(
        json.dumps(
            [
                {
                    "id": "test-1",
                    "user_text": "buy shoes on amazon",
                    "site": "amazon.com",
                    "category": "commerce",
                }
            ]
        ),
        encoding="utf-8",
    )

    payload = json.loads(mcp_server.match_task(str(log_dir), "research-task", str(bank_path)))

    assert payload["task_id"] == "research-task"
    assert payload["match"] is None


def test_golden_capture_returns_exemplar(tmp_path: Path) -> None:
    mcp_server = _load_mcp_server_module()

    log_dir = tmp_path / "golden-logs"
    output_path = tmp_path / "fixtures" / "checkout-exemplar.json"
    _write_log(
        log_dir,
        [
            _make_task(
                "checkout-task",
                "Buy the wireless headset and complete checkout on shop.example.test.",
                [
                    _make_step(
                        "checkout-task",
                        1,
                        "browser_navigate",
                        tool_input={"url": "https://shop.example.test/"},
                        tool_result="Homepage with wireless headset listing.",
                        page_url="https://shop.example.test/",
                        llm_reasoning="Open the storefront.",
                    ),
                    _make_step(
                        "checkout-task",
                        2,
                        "browser_click",
                        tool_input={"ref": "add-to-cart"},
                        tool_result="Added to cart. Your cart subtotal is $129.",
                        page_url="https://shop.example.test/cart",
                        llm_reasoning="Add the item to the cart.",
                    ),
                    _make_step(
                        "checkout-task",
                        3,
                        "browser_fill_ref",
                        tool_input={"ref": "shipping-form", "fields": ["address", "zip"], "text": "123 Main St zip 60601"},
                        tool_result="Shipping form accepted.",
                        page_url="https://shop.example.test/checkout",
                        llm_reasoning="Fill the shipping form.",
                    ),
                    _make_step(
                        "checkout-task",
                        4,
                        "browser_fill_ref",
                        tool_input={
                            "ref": "payment-form",
                            "fields": ["card number", "cvv", "expiration"],
                            "text": "4111 1111 1111 1111 123 12/29",
                        },
                        tool_result="card number cvv expir payment method confirmed",
                        page_url="https://shop.example.test/payment",
                        llm_reasoning="Enter the payment details.",
                    ),
                    _make_step(
                        "checkout-task",
                        5,
                        "browser_click",
                        tool_input={"ref": "place-order"},
                        tool_result="Order confirmation page loaded.",
                        page_url="https://shop.example.test/order/confirmation",
                        llm_reasoning="Submit the order.",
                    ),
                ],
                task_category="commerce",
                status="success",
                final_answer="Order placed.",
            )
        ],
    )

    payload = json.loads(
        mcp_server.golden_capture(str(log_dir), "checkout-task", output=str(output_path))
    )

    assert payload["saved_to"] == str(output_path)
    assert "exemplar" in payload
    assert payload["exemplar"]["task_id"] == "checkout-task"
    assert payload["exemplar"]["site"]
    assert payload["exemplar"]["step_sequence"]
    assert "efficiency_metadata" in payload["exemplar"]


def test_triage_empty_dir(tmp_path: Path) -> None:
    mcp_server = _load_mcp_server_module()

    log_dir = tmp_path / "empty-logs"
    log_dir.mkdir()

    payload = json.loads(mcp_server.triage(str(log_dir)))

    assert isinstance(payload, dict)
    assert payload["error"] == "No tasks found"
