"""Tests for test output parsing."""

from __future__ import annotations

import pytest

from agent_xray.enforce import TestResult, parse_test_output


# ---------------------------------------------------------------------------
# pytest format
# ---------------------------------------------------------------------------

class TestParsePytestOutput:
    def test_basic_pytest_summary(self):
        output = """
tests/test_foo.py ..F.
tests/test_bar.py ...

============================= short test summary info =============================
FAILED tests/test_foo.py::test_baz - assert 1 == 2
========================= 1 failed, 6 passed in 2.34s =========================
"""
        result = parse_test_output(output, exit_code=1)
        assert result.passed == 6
        assert result.failed == 1
        assert result.total == 7
        assert result.duration_seconds == pytest.approx(2.34)
        assert result.exit_code == 1

    def test_all_passed(self):
        output = "========================= 10 passed in 1.50s ========================="
        result = parse_test_output(output, exit_code=0)
        assert result.passed == 10
        assert result.failed == 0
        assert result.total == 10
        assert result.duration_seconds == pytest.approx(1.5)

    def test_all_failed(self):
        output = "========================= 5 failed in 3.21s ========================="
        result = parse_test_output(output, exit_code=1)
        assert result.passed == 0
        assert result.failed == 5
        assert result.total == 5

    def test_mixed_with_errors(self):
        output = "=============== 3 failed, 5 passed, 2 errors in 4.10s ==============="
        result = parse_test_output(output, exit_code=1)
        assert result.passed == 5
        assert result.failed == 3
        assert result.errors == 2
        assert result.total == 10

    def test_with_skipped(self):
        output = "============= 8 passed, 2 skipped in 1.80s ============="
        result = parse_test_output(output, exit_code=0)
        assert result.passed == 8
        assert result.skipped == 2
        assert result.total == 10

    def test_verbose_output_names(self):
        output = """
tests/test_foo.py::test_alpha PASSED
tests/test_foo.py::test_beta PASSED
tests/test_foo.py::test_gamma FAILED
tests/test_bar.py::test_delta PASSED

========================= 1 failed, 3 passed in 1.20s =========================
"""
        result = parse_test_output(output, exit_code=1)
        assert result.passed == 3
        assert result.failed == 1
        assert "tests/test_foo.py::test_alpha" in result.test_names_passed
        assert "tests/test_foo.py::test_beta" in result.test_names_passed
        assert "tests/test_bar.py::test_delta" in result.test_names_passed
        assert "tests/test_foo.py::test_gamma" in result.test_names_failed

    def test_with_warnings(self):
        output = "=========== 5 passed, 2 warnings in 0.85s ==========="
        result = parse_test_output(output, exit_code=0)
        assert result.passed == 5
        assert result.total == 5

    def test_error_status(self):
        output = """
tests/test_foo.py::test_alpha ERROR

========================= 1 error in 0.50s =========================
"""
        result = parse_test_output(output, exit_code=1)
        assert result.errors == 1

    def test_deselected(self):
        output = "=========== 3 passed, 7 deselected in 0.40s ==========="
        result = parse_test_output(output, exit_code=0)
        assert result.passed == 3

    def test_quiet_output_parses_failed_names(self):
        output = """
....F.F

============================= short test summary info =============================
FAILED tests/test_foo.py::test_bar - AssertionError: boom
FAILED tests/test_baz.py::test_qux - ValueError: nope
========================= 2 failed, 5 passed in 1.23s =========================
"""
        result = parse_test_output(output, exit_code=1)
        assert result.passed == 5
        assert result.failed == 2
        assert result.test_names_passed == []
        assert result.test_names_failed == [
            "tests/test_foo.py::test_bar",
            "tests/test_baz.py::test_qux",
        ]

    def test_quiet_output_parses_error_names(self):
        output = """
.E.

============================= short test summary info =============================
ERROR tests/test_foo.py::test_bar - RuntimeError: boom
========================= 1 error, 2 passed in 0.45s =========================
"""
        result = parse_test_output(output, exit_code=1)
        assert result.errors == 1
        assert result.test_names_failed == ["tests/test_foo.py::test_bar"]


# ---------------------------------------------------------------------------
# Generic format
# ---------------------------------------------------------------------------

class TestParseGenericOutput:
    def test_generic_passed(self):
        output = "Results: 12 passed, 3 failed"
        result = parse_test_output(output, exit_code=1)
        assert result.passed == 12
        assert result.failed == 3
        assert result.total == 15

    def test_generic_success(self):
        output = "Test run: 8 ok, 0 fail"
        result = parse_test_output(output, exit_code=0)
        assert result.passed == 8
        assert result.failed == 0

    def test_generic_failures_only(self):
        output = "ERROR: 4 failures found"
        result = parse_test_output(output, exit_code=1)
        assert result.failed == 4

    def test_case_insensitive(self):
        output = "10 PASSED, 2 FAILED"
        result = parse_test_output(output, exit_code=1)
        assert result.passed == 10
        assert result.failed == 2


# ---------------------------------------------------------------------------
# Fallback (exit code only)
# ---------------------------------------------------------------------------

class TestParseFallback:
    def test_exit_zero(self):
        output = "Some unrecognizable output\nDone."
        result = parse_test_output(output, exit_code=0)
        assert result.passed == 1
        assert result.failed == 0
        assert result.total == 1

    def test_exit_nonzero(self):
        output = "Some unrecognizable output\nFailed."
        result = parse_test_output(output, exit_code=1)
        assert result.passed == 0
        assert result.failed == 1
        assert result.total == 1

    def test_empty_output(self):
        result = parse_test_output("", exit_code=0)
        assert result.passed == 1
        assert result.total == 1

    def test_empty_output_failure(self):
        result = parse_test_output("", exit_code=2)
        assert result.failed == 1
        assert result.total == 1


# ---------------------------------------------------------------------------
# TestResult round-trip
# ---------------------------------------------------------------------------

class TestTestResultRoundTrip:
    def test_from_parse_to_dict_and_back(self):
        output = "========================= 5 passed, 1 failed in 2.00s ========================="
        tr = parse_test_output(output, exit_code=1)
        d = tr.to_dict()
        tr2 = TestResult.from_dict(d)
        assert tr2.passed == tr.passed
        assert tr2.failed == tr.failed
        assert tr2.duration_seconds == tr.duration_seconds
        assert tr2.output == tr.output
        assert tr2.timestamp == tr.timestamp

    def test_timestamp_preserved(self):
        output = "3 passed in 1.00s"
        tr = parse_test_output(output, exit_code=0)
        assert tr.timestamp != ""
        d = tr.to_dict()
        tr2 = TestResult.from_dict(d)
        assert tr2.timestamp == tr.timestamp


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestParseEdgeCases:
    def test_multiline_pytest_with_noise(self):
        output = """
platform linux -- Python 3.12.0, pytest-8.0.0
rootdir: /home/user/project
configfile: pyproject.toml
plugins: cov-5.0
collected 42 items

tests/test_core.py ............................ [66%]
tests/test_cli.py ......FFFF........             [100%]

============================= short test summary info =============================
FAILED tests/test_cli.py::test_a - AssertionError
FAILED tests/test_cli.py::test_b - AssertionError
FAILED tests/test_cli.py::test_c - AssertionError
FAILED tests/test_cli.py::test_d - AssertionError
========================= 4 failed, 38 passed in 12.34s =========================
"""
        result = parse_test_output(output, exit_code=1)
        assert result.passed == 38
        assert result.failed == 4
        assert result.total == 42
        assert result.duration_seconds == pytest.approx(12.34)

    def test_no_duration(self):
        output = "5 passed, 1 failed"
        result = parse_test_output(output, exit_code=1)
        assert result.passed == 5
        assert result.failed == 1

    def test_only_numbers(self):
        """Output with just numbers shouldn't confuse the parser."""
        output = "Test #42 completed. Return code: 0"
        result = parse_test_output(output, exit_code=0)
        # Should fall back to exit code
        assert result.total >= 1

    def test_unicode_output(self):
        output = "Tests: 3 passed \u2714, 1 failed \u2718 in 1.00s"
        result = parse_test_output(output, exit_code=1)
        # Should handle unicode gracefully
        assert result.exit_code == 1
