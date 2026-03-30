from __future__ import annotations

import importlib
import json
import re
import sys
import types
from pathlib import Path

import pytest


MCP_TOOL_NAMES = [
    "enforce",
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


def test_server_module_imports_without_error() -> None:
    module = _load_mcp_server_module()
    assert module.server is not None
    assert callable(module.enforce)
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
    assert "enforce" in module.server.tools
    assert "grade" in module.server.tools

    monkeypatch.delitem(sys.modules, "agent_xray.mcp_server", raising=False)
    importlib.import_module("agent_xray.mcp_server")


def test_main_runs_stdio_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_mcp_server_module()
    called: list[str] = []
    monkeypatch.setattr(module.server, "run", lambda transport="stdio": called.append(transport))
    module.main()
    assert called == ["stdio"]


def test_enforce_tool_dispatches_verbs(monkeypatch: pytest.MonkeyPatch) -> None:
    mcp_server = _load_mcp_server_module()

    long_output = "x" * 800
    calls: list[tuple[str, str, str]] = []

    monkeypatch.setattr(
        "agent_xray.enforce.enforce_auto",
        lambda hypothesis="", project_root=".": (
            calls.append(("auto", hypothesis, project_root))
            or {"verb": "auto", "hypothesis": hypothesis, "project_root": project_root}
        ),
    )
    monkeypatch.setattr(
        "agent_xray.enforce.enforce_init",
        lambda hypothesis="", project_root=".": (
            calls.append(("init", hypothesis, project_root))
            or {"verb": "init", "hypothesis": hypothesis, "project_root": project_root}
        ),
    )
    monkeypatch.setattr(
        "agent_xray.enforce.enforce_check",
        lambda hypothesis="", project_root=".": (
            calls.append(("check", hypothesis, project_root))
            or {
                "decision": "RECOMMEND_COMMIT",
                "before": {
                    "passed": 1,
                    "failed": 1,
                    "errors": 0,
                    "total": 2,
                    "output": long_output,
                },
                "after": {
                    "passed": 2,
                    "failed": 0,
                    "errors": 0,
                    "total": 2,
                    "output": long_output,
                },
            }
        ),
    )
    monkeypatch.setattr(
        "agent_xray.enforce.enforce_plan",
        lambda hypothesis="", project_root=".": (
            calls.append(("plan", hypothesis, project_root))
            or {"status": "plan_registered", "hypothesis": hypothesis}
        ),
    )
    monkeypatch.setattr(
        "agent_xray.enforce.enforce_diff",
        lambda project_root=".": (
            calls.append(("diff", "", project_root))
            or {"file_count": 1, "diff_lines": ["+new line"], "would_reject": False}
        ),
    )
    monkeypatch.setattr(
        "agent_xray.enforce.enforce_guard",
        lambda project_root=".": (
            calls.append(("guard", "", project_root))
            or {"status": "clean", "warnings": []}
        ),
    )
    monkeypatch.setattr(
        "agent_xray.enforce.enforce_status",
        lambda project_root=".": (
            calls.append(("status", "", project_root))
            or {
                "session_active": True,
                "last_result": {
                    "passed": 3,
                    "failed": 1,
                    "errors": 0,
                    "total": 4,
                    "output": long_output,
                },
            }
        ),
    )
    monkeypatch.setattr(
        "agent_xray.enforce.enforce_challenge",
        lambda project_root=".": (
            calls.append(("challenge", "", project_root))
            or {"changes_reviewed": 1, "findings": ["challenge"]}
        ),
    )
    monkeypatch.setattr(
        "agent_xray.enforce.enforce_reset",
        lambda project_root=".": (
            calls.append(("reset", "", project_root))
            or True
        ),
    )
    monkeypatch.setattr(
        "agent_xray.enforce.build_enforce_report",
        lambda project_root=".": (
            calls.append(("report", "", project_root))
            or {"project_root": project_root, "summary": "ok"}
        ),
    )
    monkeypatch.setattr("agent_xray.enforce_report.format_enforce_markdown", lambda report: "# report")
    monkeypatch.setattr("agent_xray.enforce_report.format_enforce_text", lambda report, color=False: "report")
    monkeypatch.setattr(
        "agent_xray.enforce_report.format_enforce_json",
        lambda report: json.dumps({"formatted": report}),
    )

    auto_payload = json.loads(mcp_server.enforce(hypothesis="auto hypothesis", project_root="repo"))
    init_payload = json.loads(mcp_server.enforce(verb="init", hypothesis="init hypothesis", project_root="repo"))
    check_payload = json.loads(mcp_server.enforce(verb="check", hypothesis="check hypothesis", project_root="repo"))
    plan_payload = json.loads(mcp_server.enforce(verb="plan", hypothesis="plan hypothesis", project_root="repo"))
    diff_payload = json.loads(mcp_server.enforce(verb="diff", project_root="repo"))
    guard_payload = json.loads(mcp_server.enforce(verb="guard", project_root="repo"))
    status_payload = json.loads(mcp_server.enforce(verb="status", project_root="repo"))
    challenge_payload = json.loads(mcp_server.enforce(verb="challenge", project_root="repo"))
    reset_payload = json.loads(mcp_server.enforce(verb="reset", project_root="repo"))
    report_payload = json.loads(mcp_server.enforce(verb="report", project_root="repo", format="markdown"))

    assert auto_payload["verb"] == "auto"
    assert init_payload["verb"] == "init"
    assert check_payload["decision"] == "RECOMMEND_COMMIT"
    assert len(check_payload["before"]["output"]) == 500
    assert plan_payload["status"] == "plan_registered"
    assert diff_payload["diff_lines"] == ["+new line"]
    assert guard_payload["status"] == "clean"
    assert status_payload["session_active"] is True
    assert len(status_payload["last_result"]["output"]) == 500
    assert challenge_payload["changes_reviewed"] == 1
    assert reset_payload == {"success": True}
    assert report_payload == {"format": "markdown", "report": "# report"}
    assert calls == [
        ("auto", "auto hypothesis", "repo"),
        ("init", "init hypothesis", "repo"),
        ("check", "check hypothesis", "repo"),
        ("plan", "plan hypothesis", "repo"),
        ("diff", "", "repo"),
        ("guard", "", "repo"),
        ("status", "", "repo"),
        ("challenge", "", "repo"),
        ("reset", "", "repo"),
        ("report", "", "repo"),
    ]


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


def test_grade_with_grade_filter(tmp_trace_dir: Path) -> None:
    """grade_filter=BROKEN should return only BROKEN-graded tasks (not to be confused with outcome)."""
    mcp_server = _load_mcp_server_module()

    # Without filter: should have multiple grades
    all_payload = json.loads(mcp_server.grade(str(tmp_trace_dir)))
    assert all_payload["summary"]["tasks"] == 4

    # With grade_filter=BROKEN: should return only BROKEN tasks
    broken_payload = json.loads(mcp_server.grade(str(tmp_trace_dir), grade_filter="BROKEN"))
    if broken_payload["summary"]["tasks"] > 0:
        assert all(t["grade"] == "BROKEN" for t in broken_payload["worst_tasks"])

    # With grade_filter=GOLDEN: should return only GOLDEN tasks
    golden_payload = json.loads(mcp_server.grade(str(tmp_trace_dir), grade_filter="GOLDEN"))
    if golden_payload["summary"]["tasks"] > 0:
        assert all(t["grade"] == "GOLDEN" for t in golden_payload["worst_tasks"])


def test_triage_with_grade_filter(tmp_trace_dir: Path) -> None:
    """triage(grade_filter='BROKEN') should scope the investigation to BROKEN-graded tasks."""
    mcp_server = _load_mcp_server_module()

    payload = json.loads(mcp_server.triage(str(tmp_trace_dir), grade_filter="BROKEN"))
    # Should still return valid JSON (even if all tasks filtered out)
    assert isinstance(payload, dict)


def test_root_cause_with_grade_filter(tmp_trace_dir: Path) -> None:
    """root_cause(grade_filter='BROKEN') should scope classification to BROKEN-graded tasks."""
    mcp_server = _load_mcp_server_module()

    payload = json.loads(mcp_server.root_cause(str(tmp_trace_dir), grade_filter="BROKEN"))
    assert isinstance(payload, dict)
    assert payload["summary"]["tasks"] > 0, "Expected at least 1 BROKEN task in fixture"
    # All classified failures should come from BROKEN tasks
    if payload["tasks"]:
        for task_result in payload["tasks"]:
            if "grade" in task_result:
                assert task_result["grade"] == "BROKEN"


def test_diagnose_with_grade_filter(tmp_trace_dir: Path) -> None:
    """diagnose(grade_filter='BROKEN') should build fix plan only from BROKEN-graded tasks."""
    mcp_server = _load_mcp_server_module()

    payload = json.loads(mcp_server.diagnose(str(tmp_trace_dir), grade_filter="BROKEN"))
    assert isinstance(payload, dict)
    assert payload["summary"]["tasks"] > 0, "Expected at least 1 BROKEN task in fixture"
    assert "fix_plan" in payload


def test_grade_filter_empty_after_filter_returns_error(tmp_trace_dir: Path) -> None:
    """When grade_filter matches no tasks, all callers should return an explicit error."""
    mcp_server = _load_mcp_server_module()

    # GOLDEN is unlikely to be the grade for all tasks; use a nonexistent grade to be sure
    for fn in [mcp_server.grade, mcp_server.root_cause, mcp_server.diagnose]:
        payload = json.loads(fn(str(tmp_trace_dir), grade_filter="INVALID_GRADE"))
        assert "error" in payload, f"{fn.__name__} should return error for invalid grade_filter"

    # triage also handles it
    triage_payload = json.loads(mcp_server.triage(str(tmp_trace_dir), grade_filter="INVALID_GRADE"))
    assert "error" in triage_payload


def test_grade_filter_whitespace_stripped(tmp_trace_dir: Path) -> None:
    """grade_filter=' BROKEN ' (with spaces) should still match BROKEN tasks."""
    mcp_server = _load_mcp_server_module()

    # With whitespace: should still find BROKEN tasks
    payload = json.loads(mcp_server.grade(str(tmp_trace_dir), grade_filter=" BROKEN "))
    assert payload["summary"]["tasks"] > 0, "Whitespace-padded grade_filter should be stripped"
    for t in payload["worst_tasks"]:
        assert t["grade"] == "BROKEN"


def test_grade_filter_case_insensitive(tmp_trace_dir: Path) -> None:
    """grade_filter='broken' (lowercase) should match BROKEN tasks."""
    mcp_server = _load_mcp_server_module()

    payload = json.loads(mcp_server.grade(str(tmp_trace_dir), grade_filter="broken"))
    assert payload["summary"]["tasks"] > 0, "Lowercase grade_filter should match"
    for t in payload["worst_tasks"]:
        assert t["grade"] == "BROKEN"


def test_grade_filter_uses_caller_rules_not_default(tmp_trace_dir: Path) -> None:
    """grade_filter must filter by the caller's rules, not hardcoded default rules.

    This is a regression test for the bug where _load_tasks pre-filtered with
    default rules, but the caller used custom rules that assigned different grades.
    """
    mcp_server = _load_mcp_server_module()

    # Grade with default rules and grade_filter=BROKEN
    broken_default = json.loads(mcp_server.grade(str(tmp_trace_dir), rules="default", grade_filter="BROKEN"))
    # All returned tasks must actually be BROKEN per the output
    assert broken_default["summary"]["tasks"] > 0, "Expected BROKEN tasks in fixture"
    for t in broken_default["worst_tasks"]:
        assert t["grade"] == "BROKEN", f"grade_filter=BROKEN returned non-BROKEN task: {t}"


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
        if name == "enforce":
            assert "Run the enforce discipline workflow." in stripped
            assert "Most users should just call enforce()" in stripped
            continue
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
    suggestions = payload["suggested_next_tools"]
    assert any(s.startswith("diff_tasks(") for s in suggestions)
    assert any(s.startswith("inspect_task(") for s in suggestions)
    assert any(s.startswith("signal_detect(") for s in suggestions)
    assert any(s.startswith("compare_runs(") for s in suggestions)
    assert not any(s.startswith("golden_rank(") for s in suggestions)
    assert not any(s.startswith("tree(") for s in suggestions)


def test_triage_suggested_next_tools_use_real_param_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """triage() should emit directly callable suggestion strings with the real MCP parameter names."""
    mcp_server = _load_mcp_server_module()

    tasks = [
        types.SimpleNamespace(task_id=f"golden-{index}", task_text=f"Golden task {index}", steps=[types.SimpleNamespace(timestamp=f"2026-03-30T12:0{index}:00Z")])
        for index in range(3)
    ]
    tasks.extend(
        [
            types.SimpleNamespace(task_id="broken-1", task_text="Broken task", steps=[types.SimpleNamespace(timestamp="2026-03-30T12:10:00Z")]),
            types.SimpleNamespace(task_id="ok-1", task_text="OK task 1", steps=[types.SimpleNamespace(timestamp="2026-03-30T12:11:00Z")]),
        ]
    )
    tasks.extend(
        types.SimpleNamespace(task_id=f"ok-{index}", task_text=f"Other task {index}", steps=[types.SimpleNamespace(timestamp=f"2026-03-30T12:{index:02d}:30Z")])
        for index in range(2, 19)
    )
    grades = [
        types.SimpleNamespace(task_id="golden-0", grade="GOLDEN", score=95),
        types.SimpleNamespace(task_id="golden-1", grade="GOLDEN", score=90),
        types.SimpleNamespace(task_id="golden-2", grade="GOLDEN", score=85),
        types.SimpleNamespace(task_id="broken-1", grade="BROKEN", score=-4),
    ]
    grades.extend(
        types.SimpleNamespace(task_id=task.task_id, grade="OK", score=25)
        for task in tasks
        if task.task_id not in {"golden-0", "golden-1", "golden-2", "broken-1"}
    )

    monkeypatch.setattr(mcp_server, "_load_tasks", lambda log_dir, format, **kw: tasks)
    monkeypatch.setattr("agent_xray.grader.load_rules", lambda: types.SimpleNamespace(name="default"))
    monkeypatch.setattr("agent_xray.grader.grade_tasks", lambda tasks_arg, rules_arg: grades)
    monkeypatch.setattr("agent_xray.root_cause.classify_failures", lambda tasks_arg, grades_arg: [])
    monkeypatch.setattr("agent_xray.diagnose.build_fix_plan", lambda results: [])
    monkeypatch.setattr(
        "agent_xray.surface.surface_for_task",
        lambda task: {"steps": [{"step": 1, "tool_name": "browser_snapshot", "error": "Timeout"}]},
    )

    payload = json.loads(mcp_server.triage("traces"))

    assert payload["suggested_next_tools"] == [
        "diff_tasks(log_dir='traces', task_id_a='golden-0', task_id_b='broken-1')",
        "golden_rank(log_dir='traces')",
        "inspect_task(log_dir='traces', task_id='broken-1')",
        "signal_detect(log_dir='traces', task_id='broken-1')",
        "compare_runs(left_log_dir='traces', right_log_dir='traces_after')",
        "tree(log_dir='traces')",
    ]
    assert "next" not in payload


def test_dedupe_tasks_keeps_latest_trace_for_normalized_task_text() -> None:
    mcp_server = _load_mcp_server_module()

    older = types.SimpleNamespace(
        task_id="task-old",
        task_text="Buy   headset",
        steps=[types.SimpleNamespace(timestamp="2026-03-30T10:00:00Z")],
    )
    newer = types.SimpleNamespace(
        task_id="task-new",
        task_text="  buy headset  ",
        steps=[types.SimpleNamespace(timestamp="2026-03-30T11:00:00Z")],
    )
    distinct = types.SimpleNamespace(
        task_id="task-other",
        task_text="Research observability",
        steps=[types.SimpleNamespace(timestamp="2026-03-30T09:00:00Z")],
    )

    deduped = mcp_server._dedupe_tasks([older, newer, distinct])

    assert [task.task_id for task in deduped] == ["task-new", "task-other"]


def test_triage_loads_tasks_with_dedupe(monkeypatch: pytest.MonkeyPatch) -> None:
    mcp_server = _load_mcp_server_module()
    calls: dict[str, object] = {}
    task = types.SimpleNamespace(task_id="task-1", task_text="One task", steps=[types.SimpleNamespace(timestamp="2026-03-30T12:00:00Z")])
    grade = types.SimpleNamespace(task_id="task-1", grade="OK", score=20)

    def fake_load_tasks(log_dir, format, **kwargs):
        calls["kwargs"] = kwargs
        return [task]

    monkeypatch.setattr(mcp_server, "_load_tasks", fake_load_tasks)
    monkeypatch.setattr("agent_xray.grader.load_rules", lambda: types.SimpleNamespace(name="default"))
    monkeypatch.setattr("agent_xray.grader.grade_tasks", lambda tasks_arg, rules_arg: [grade])
    monkeypatch.setattr("agent_xray.root_cause.classify_failures", lambda tasks_arg, grades_arg: [])
    monkeypatch.setattr("agent_xray.diagnose.build_fix_plan", lambda results: [])
    monkeypatch.setattr("agent_xray.surface.surface_for_task", lambda task_arg: {"steps": []})

    payload = json.loads(mcp_server.triage("traces"))

    assert "error" not in payload
    assert calls["kwargs"]["dedupe"] is True


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
    """diagnose()'s next hint must reference the unified enforce MCP tool."""
    mcp_server = _load_mcp_server_module()
    payload = json.loads(mcp_server.diagnose(str(tmp_trace_dir)))
    hint = payload.get("next", "")
    assert "enforce(" in hint, f"diagnose next hint should reference enforce(...), got: {hint}"
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


def test_simple_ruleset_is_packaged_and_loadable() -> None:
    rules_path = Path(__file__).resolve().parents[1] / "src" / "agent_xray" / "rules" / "simple.json"
    payload = json.loads(rules_path.read_text(encoding="utf-8"))

    assert rules_path.exists()
    assert payload["name"] == "simple"
    assert payload["grade_thresholds"] == {
        "GOLDEN": 3,
        "GOOD": 2,
        "OK": 1,
        "WEAK": 0,
    }
    assert payload["signals"][0]["gte"] == 1
    assert payload["signals"][1]["gte"] == 1
