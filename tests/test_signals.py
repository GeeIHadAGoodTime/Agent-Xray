from __future__ import annotations

import json

from agent_xray.analyzer import analyze_task
from agent_xray.grader import grade_task, load_rules
from agent_xray.schema import AgentStep, AgentTask, TaskOutcome
from agent_xray.signals import BUILTIN_DETECTORS, discover_detectors, run_detection
from agent_xray.signals.coding import CodingDetector
from agent_xray.signals.commerce import CommerceDetector
from agent_xray.signals.research import ResearchDetector


def _step(
    step: int,
    tool_name: str,
    tool_input: dict[str, object] | None = None,
    *,
    tool_result: str | None = None,
    page_url: str | None = None,
    error: str | None = None,
) -> AgentStep:
    return AgentStep(
        task_id="task-1",
        step=step,
        tool_name=tool_name,
        tool_input=tool_input or {},
        tool_result=tool_result,
        page_url=page_url,
        error=error,
    )


def test_commerce_detector_payment_page() -> None:
    detector = CommerceDetector()
    signals = detector.detect_step(
        _step(
            1,
            "browser_snapshot",
            {},
            tool_result="card number cvv expir",
            page_url="https://shop.example.test/payment",
        )
    )
    assert signals["is_payment_page"] is True


def test_commerce_detector_checkout() -> None:
    detector = CommerceDetector()
    summary = detector.summarize(
        AgentTask(task_id="task-1", steps=[]),
        [
            detector.detect_step(
                _step(
                    1,
                    "browser_snapshot",
                    {},
                    tool_result="checkout",
                    page_url="https://shop.example.test/",
                )
            )
        ],
    )
    assert summary["reached_checkout"] is False
    assert summary["reached_checkout_confidence"] == "keyword_match"


def test_commerce_detector_cart() -> None:
    detector = CommerceDetector()
    summary = detector.summarize(
        AgentTask(task_id="task-1", steps=[]),
        [detector.detect_step(_step(1, "browser_snapshot", {}, tool_result="your cart"))],
    )
    assert summary["reached_cart"] is False
    assert summary["reached_cart_confidence"] == "keyword_match"


def test_commerce_detector_cart_action_sequence() -> None:
    detector = CommerceDetector()
    steps = [
        _step(
            1,
            "browser_click",
            {"ref": "add-to-cart"},
            tool_result="Added to cart. Your cart subtotal is $129.",
            page_url="https://shop.example.test/products/wireless-headset",
        )
    ]
    summary = detector.summarize(
        AgentTask(task_id="task-1", steps=steps),
        [detector.detect_step(step) for step in steps],
    )
    assert summary["reached_cart"] is True
    assert summary["reached_cart_confidence"] == "action_sequence"


def test_commerce_detector_checkout_requires_url_or_progression() -> None:
    detector = CommerceDetector()
    steps = [
        _step(
            1,
            "browser_snapshot",
            {},
            tool_result="Checkout is available in the top navigation.",
            page_url="https://shop.example.test/",
        ),
        _step(
            2,
            "browser_click",
            {"ref": "proceed-to-checkout"},
            tool_result="Address review loaded.",
            page_url="https://shop.example.test/checkout",
        ),
    ]
    summary = detector.summarize(
        AgentTask(task_id="task-1", steps=steps),
        [detector.detect_step(step) for step in steps],
    )
    assert summary["reached_checkout"] is True
    assert summary["reached_checkout_confidence"] == "action_sequence"


def test_commerce_detector_payment_requires_url_or_payment_fields() -> None:
    detector = CommerceDetector()
    weak_steps = [
        _step(
            1,
            "browser_snapshot",
            {},
            tool_result="Payment method options are listed in the footer.",
            page_url="https://shop.example.test/",
        )
    ]
    weak_summary = detector.summarize(
        AgentTask(task_id="task-1", steps=weak_steps),
        [detector.detect_step(step) for step in weak_steps],
    )
    assert weak_summary["reached_payment"] is False
    assert weak_summary["reached_payment_confidence"] == "keyword_match"

    strong_steps = [
        _step(
            1,
            "browser_fill_ref",
            {"ref": "payment-form", "fields": ["card number", "cvv"], "text": "4111 1111 1111 1111"},
            tool_result="Payment form completed.",
            page_url="https://shop.example.test/review",
        )
    ]
    strong_summary = detector.summarize(
        AgentTask(task_id="task-2", steps=strong_steps),
        [detector.detect_step(step) for step in strong_steps],
    )
    assert strong_summary["reached_payment"] is True
    assert strong_summary["reached_payment_confidence"] == "action_sequence"


def test_commerce_detector_ignores_stale_cart_keywords_without_progression() -> None:
    detector = CommerceDetector()
    steps = [
        _step(
            1,
            "browser_navigate",
            {"url": "https://pizza.example.test"},
            tool_result="Homepage banner. 1 ITEMS IN YOUR CART.",
            page_url="https://pizza.example.test/",
        ),
        _step(
            2,
            "browser_click",
            {"ref": "order-now"},
            tool_result="Menu page loaded.",
            page_url="https://pizza.example.test/menu",
        ),
        _step(
            3,
            "browser_click",
            {"ref": "pizzas"},
            tool_result="Pizza category page loaded.",
            page_url="https://pizza.example.test/menu/pizzas",
        ),
    ]
    summary = detector.summarize(
        AgentTask(task_id="task-1", steps=steps),
        [detector.detect_step(step) for step in steps],
    )
    assert summary["reached_cart"] is False
    assert summary["reached_checkout"] is False
    assert summary["reached_payment"] is False
    assert summary["milestone_confidence"] == {
        "cart": "keyword_match",
        "checkout": "none",
        "payment": "none",
    }


def test_commerce_detector_form_fill() -> None:
    detector = CommerceDetector()
    summary = detector.summarize(
        AgentTask(task_id="task-1", steps=[]),
        [
            detector.detect_step(
                _step(
                    1,
                    "browser_fill_ref",
                    {"ref": "e10", "text": "123 Main St", "fields": ["address"]},
                )
            )
        ],
    )
    assert summary["form_fill_count"] == 1
    assert summary["real_fill_count"] == 1


def test_coding_detector_file_ops() -> None:
    detector = CodingDetector()
    summary = detector.summarize(
        AgentTask(
            task_id="task-1",
            steps=[
                _step(1, "write_file", {"path": "src/app.py"}),
                _step(2, "edit_file", {"path": "tests/test_app.py"}),
            ],
        ),
        [
            detector.detect_step(_step(1, "write_file", {"path": "src/app.py"})),
            detector.detect_step(_step(2, "edit_file", {"path": "tests/test_app.py"})),
        ],
    )
    assert summary["file_operations"] == 2
    assert summary["unique_files_touched"] >= 2


def test_coding_detector_test_verify_cycle() -> None:
    detector = CodingDetector()
    steps = [
        _step(1, "edit_file", {"path": "src/app.py"}),
        _step(2, "pytest", {"path": "tests/test_app.py"}),
    ]
    summary = detector.summarize(
        AgentTask(task_id="task-1", steps=steps), [detector.detect_step(step) for step in steps]
    )
    assert summary["has_test_verify_cycle"] is True


def test_coding_detector_lint_run() -> None:
    detector = CodingDetector()
    summary = detector.summarize(
        AgentTask(task_id="task-1", steps=[]),
        [detector.detect_step(_step(1, "ruff", {"path": "src/app.py"}))],
    )
    assert summary["lint_runs"] == 1


def test_research_detector_search() -> None:
    detector = ResearchDetector()
    signals = detector.detect_step(_step(1, "web_search", {"query": "agent_xray"}))
    assert signals["is_search"] is True


def test_research_detector_source_diversity() -> None:
    detector = ResearchDetector()
    steps = [
        _step(1, "web_search", {"query": "x"}, tool_result="https://a.example.test/result"),
        _step(
            2,
            "read_url",
            {"url": "https://b.example.test/report"},
            tool_result="https://b.example.test/report",
        ),
        _step(3, "browser_snapshot", {}, page_url="https://c.example.test/page"),
    ]
    summary = detector.summarize(
        AgentTask(task_id="task-1", steps=steps), [detector.detect_step(step) for step in steps]
    )
    assert summary["source_diversity"] == 3


def test_research_detector_synthesis() -> None:
    detector = ResearchDetector()
    summary = detector.summarize(
        AgentTask(task_id="task-1", steps=[]),
        [
            detector.detect_step(
                _step(1, "summarize", {}, tool_result="According to https://a.example.test")
            )
        ],
    )
    assert summary["has_synthesis_step"] is True


def test_discover_detectors_finds_builtins() -> None:
    names = {detector.name for detector in discover_detectors()}
    assert {"commerce", "coding", "research", "multi_agent", "memory", "planning"} <= names


def test_builtin_detectors_registry_contains_all_signal_modules() -> None:
    names = {detector_cls().name for detector_cls in BUILTIN_DETECTORS}
    assert names == {"commerce", "coding", "research", "multi_agent", "memory", "planning"}


def test_run_detection_returns_all_detectors() -> None:
    task = AgentTask(
        task_id="task-1",
        steps=[
            _step(
                1,
                "browser_snapshot",
                {},
                tool_result="your cart",
                page_url="https://shop.example.test/cart",
            ),
            _step(2, "write_file", {"path": "src/app.py"}),
            _step(
                3,
                "web_search",
                {"query": "agent xray"},
                tool_result="https://docs.example.test/page",
            ),
            _step(4, "create_plan", {"plan_id": "plan-1", "steps": ["search", "write"]}),
        ],
    )
    results = run_detection(task)
    assert {"commerce", "coding", "research", "multi_agent", "memory", "planning"} <= set(results)


def test_grader_with_dotpath_signal_fields(tmp_path) -> None:
    rules_path = tmp_path / "dotpath_rules.json"
    rules_path.write_text(
        json.dumps(
            {
                "name": "dotpath",
                "description": "dotpath grading",
                "signals": [
                    {
                        "field": "commerce.reached_payment",
                        "op": "equals",
                        "value": True,
                        "points": 5,
                        "label": "payment_reached",
                    },
                    {
                        "field": "commerce.url_has_terminal",
                        "op": "equals",
                        "value": True,
                        "points": 2,
                        "label": "terminal_url",
                    },
                ],
                "thresholds": {"GOLDEN": 7, "GOOD": 4, "OK": 1, "WEAK": 0},
                "golden_requirements": ["payment_reached", "terminal_url"],
            }
        ),
        encoding="utf-8",
    )
    task = AgentTask(
        task_id="task-1",
        steps=[
            _step(
                1,
                "browser_fill_ref",
                {"ref": "e20", "text": "4111 1111 1111 1111"},
                tool_result="card number cvv expir",
                page_url="https://shop.example.test/payment",
            )
        ],
    )
    result = grade_task(task, load_rules(rules_path))
    assert result.grade == "GOLDEN"
    assert result.score == 7


def test_browser_flow_penalizes_failure_language_in_final_answer() -> None:
    clean_task = AgentTask(
        task_id="task-1",
        steps=[
            _step(
                1,
                "browser_navigate",
                {"url": "https://shop.example.test/cart"},
                tool_result="Cart loaded.",
                page_url="https://shop.example.test/cart",
            ),
            _step(
                2,
                "browser_snapshot",
                {},
                tool_result="card number cvv expir",
                page_url="https://shop.example.test/payment",
            ),
        ],
        outcome=TaskOutcome(
            task_id="task-1",
            status="completed",
            total_steps=2,
            total_duration_s=2.0,
            final_answer="Payment page loaded for checkout review.",
        ),
    )
    failed_task = AgentTask(
        task_id="task-1",
        steps=clean_task.steps,
        outcome=TaskOutcome(
            task_id="task-1",
            status="completed",
            total_steps=2,
            total_duration_s=2.0,
            final_answer="The checkout page is currently showing an error, and I cannot proceed further.",
        ),
    )
    clean_analysis = analyze_task(clean_task)
    failed_analysis = analyze_task(failed_task)
    clean_result = grade_task(clean_task, load_rules("browser_flow"), analysis=clean_analysis)
    failed_result = grade_task(failed_task, load_rules("browser_flow"), analysis=failed_analysis)
    signal_map = {signal.name: signal for signal in failed_result.signals}

    assert clean_analysis.final_answer_contains_failure_keywords is False
    assert failed_analysis.final_answer_contains_failure_keywords is True
    assert clean_result.grade == "GOOD"
    assert failed_result.grade == "OK"
    assert failed_result.score == clean_result.score - 3
    assert signal_map["final_answer_indicates_failure"].passed is True
    assert signal_map["final_answer_indicates_failure"].points == -3
