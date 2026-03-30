"""Tests for gaming detection and audit heuristics."""

from __future__ import annotations

import pytest

from agent_xray.enforce import ChangeRecord, TestResult
from agent_xray.enforce_audit import (
    GamingSignal,
    analyze_successful_changes,
    audit_change,
    challenge_iterations,
    classify_change_quality,
    detect_assertion_weakening,
    detect_early_return,
    detect_exception_swallowing,
    detect_hardcoded_values,
    detect_import_removal,
    detect_mock_insertion,
    detect_special_case_branching,
    detect_test_file_modification,
    quality_distribution,
)

# ---------------------------------------------------------------------------
# detect_test_file_modification
# ---------------------------------------------------------------------------

class TestDetectTestFileModification:
    def test_detects_test_file(self):
        sig = detect_test_file_modification(["tests/test_foo.py"])
        assert sig is not None
        assert sig.name == "test_file_modification"
        assert sig.confidence == pytest.approx(0.3)

    def test_detects_test_prefix(self):
        sig = detect_test_file_modification(["src/test_bar.py"])
        assert sig is not None

    def test_detects_test_suffix(self):
        sig = detect_test_file_modification(["src/bar_test.py"])
        assert sig is not None

    def test_detects_conftest(self):
        sig = detect_test_file_modification(["tests/conftest.py"])
        assert sig is not None

    def test_no_test_files(self):
        sig = detect_test_file_modification(["src/main.py", "lib/utils.py"])
        assert sig is None

    def test_allowed_test_modification(self):
        sig = detect_test_file_modification(
            ["tests/test_foo.py"], allow_test_modification=True
        )
        assert sig is None

    def test_empty_list(self):
        sig = detect_test_file_modification([])
        assert sig is None

    def test_spec_directory(self):
        sig = detect_test_file_modification(["spec/test_thing.py"])
        assert sig is not None


# ---------------------------------------------------------------------------
# detect_hardcoded_values
# ---------------------------------------------------------------------------

class TestDetectHardcodedValues:
    def test_detects_return_42(self):
        diff = """\
+    return 42
"""
        sig = detect_hardcoded_values(diff)
        assert sig is not None
        assert sig.name == "hardcoded_value_insertion"

    def test_detects_return_string(self):
        diff = """\
+    return "expected_value"
"""
        sig = detect_hardcoded_values(diff)
        assert sig is not None

    def test_detects_return_true(self):
        diff = """\
+    return True
"""
        sig = detect_hardcoded_values(diff)
        assert sig is not None

    def test_detects_return_empty_list(self):
        diff = """\
+    return []
"""
        sig = detect_hardcoded_values(diff)
        assert sig is not None

    def test_ignores_normal_return(self):
        diff = """\
+    return self.process_data(input_value, config=config)
"""
        sig = detect_hardcoded_values(diff)
        assert sig is None

    def test_no_additions(self):
        diff = """\
-    return old_value
 context line
"""
        sig = detect_hardcoded_values(diff)
        assert sig is None

    @pytest.mark.parametrize(
        "line",
        [
            '+    return "no visible effect"\n',
            '+    return "not found"\n',
            '+    return "error"\n',
            '+    return "success"\n',
        ],
    )
    def test_allows_common_production_strings(self, line: str):
        sig = detect_hardcoded_values(line)
        assert sig is None

    def test_respects_project_allowlist(self):
        sig = detect_hardcoded_values('+    return "custom benign marker"\n', ["custom benign"])
        assert sig is None


# ---------------------------------------------------------------------------
# detect_special_case_branching
# ---------------------------------------------------------------------------

class TestDetectSpecialCaseBranching:
    def test_detects_test_mode_check(self):
        diff = """\
+    if test_mode:
+        return mock_data
"""
        sig = detect_special_case_branching(diff)
        assert sig is not None
        assert sig.name == "special_case_branching"

    def test_detects_TEST_MODE(self):
        diff = """\
+    if TEST_MODE:
"""
        sig = detect_special_case_branching(diff)
        assert sig is not None

    def test_detects_is_test(self):
        diff = """\
+    if is_test:
"""
        sig = detect_special_case_branching(diff)
        assert sig is not None

    def test_detects_mock_check(self):
        diff = """\
+    if MOCK:
"""
        sig = detect_special_case_branching(diff)
        assert sig is not None

    def test_ignores_normal_branch(self):
        diff = """\
+    if value > threshold:
+        process(value)
"""
        sig = detect_special_case_branching(diff)
        assert sig is None


# ---------------------------------------------------------------------------
# detect_mock_insertion
# ---------------------------------------------------------------------------

class TestDetectMockInsertion:
    def test_detects_mock(self):
        diff = """\
+    mock = Mock()
"""
        sig = detect_mock_insertion(diff)
        assert sig is not None
        assert sig.name == "mock_stub_insertion"
        assert sig.confidence == pytest.approx(0.2)

    def test_detects_magicmock(self):
        diff = """\
+    MagicMock(return_value=42)
"""
        sig = detect_mock_insertion(diff)
        assert sig is not None

    def test_detects_patch(self):
        diff = """\
+    @patch('module.func')
"""
        sig = detect_mock_insertion(diff)
        assert sig is not None

    def test_detects_monkeypatch(self):
        diff = """\
+    monkeypatch.setattr(module, 'func', lambda: 42)
"""
        sig = detect_mock_insertion(diff)
        assert sig is not None

    def test_ignores_normal_code(self):
        diff = """\
+    result = process_data(input)
"""
        sig = detect_mock_insertion(diff)
        assert sig is None


# ---------------------------------------------------------------------------
# detect_assertion_weakening
# ---------------------------------------------------------------------------

class TestDetectAssertionWeakening:
    def test_detects_assert_removal(self):
        diff = """\
-    assert result == expected
-    assert other == value
+    pass
"""
        sig = detect_assertion_weakening(diff)
        assert sig is not None
        assert sig.name == "assertion_weakening"

    def test_detects_assertEqual_to_assertIn(self):
        diff = """\
-    self.assertEqual(result, 42)
+    self.assertIn(result, [40, 41, 42, 43])
"""
        sig = detect_assertion_weakening(diff)
        assert sig is not None

    def test_no_weakening(self):
        diff = """\
+    assert result == expected
+    assert other == value
"""
        sig = detect_assertion_weakening(diff)
        assert sig is None

    def test_replacing_with_comment(self):
        diff = """\
-    assert result == expected
+    # TODO: re-enable assertion
"""
        sig = detect_assertion_weakening(diff)
        assert sig is not None


# ---------------------------------------------------------------------------
# detect_exception_swallowing
# ---------------------------------------------------------------------------

class TestDetectExceptionSwallowing:
    def test_detects_except_pass(self):
        diff = """\
+    except Exception: pass
"""
        sig = detect_exception_swallowing(diff)
        assert sig is not None
        assert sig.name == "exception_swallowing"

    def test_detects_bare_except_pass(self):
        diff = """\
+    except:
+        pass
"""
        sig = detect_exception_swallowing(diff)
        assert sig is not None

    def test_detects_except_as_pass(self):
        diff = """\
+    except ValueError as e: pass
"""
        sig = detect_exception_swallowing(diff)
        assert sig is not None

    def test_ignores_except_with_logging(self):
        diff = """\
+    except Exception as e:
+        logger.error("Failed: %s", e)
+        raise
"""
        sig = detect_exception_swallowing(diff)
        assert sig is None


# ---------------------------------------------------------------------------
# detect_early_return
# ---------------------------------------------------------------------------

class TestDetectEarlyReturn:
    def test_detects_return_none(self):
        diff = """\
+    return None
"""
        sig = detect_early_return(diff)
        assert sig is not None
        assert sig.name == "early_return"

    def test_detects_return_false(self):
        diff = """\
+    return False
"""
        sig = detect_early_return(diff)
        assert sig is not None

    def test_detects_bare_return(self):
        diff = """\
+    return
"""
        sig = detect_early_return(diff)
        assert sig is not None

    def test_detects_return_empty_list(self):
        diff = """\
+    return []
"""
        sig = detect_early_return(diff)
        assert sig is not None

    def test_ignores_return_with_computation(self):
        diff = """\
+    return self.compute(data, config=cfg)
"""
        sig = detect_early_return(diff)
        assert sig is None


# ---------------------------------------------------------------------------
# detect_import_removal
# ---------------------------------------------------------------------------

class TestDetectImportRemoval:
    def test_detects_net_removal(self):
        diff = """\
-import os
-from pathlib import Path
"""
        sig = detect_import_removal(diff)
        assert sig is not None
        assert sig.name == "import_removal"

    def test_no_net_removal(self):
        diff = """\
-import os
+import sys
"""
        sig = detect_import_removal(diff)
        assert sig is None

    def test_only_additions(self):
        diff = """\
+import json
+from pathlib import Path
"""
        sig = detect_import_removal(diff)
        assert sig is None


# ---------------------------------------------------------------------------
# audit_change (combined)
# ---------------------------------------------------------------------------

class TestAuditChange:
    def test_clean_diff(self):
        verdict, reasons, signals = audit_change(
            diff="+    result = compute(data)\n",
            files_modified=["src/main.py"],
        )
        assert verdict == "VALID"

    def test_gaming_detected(self):
        diff = """\
+    except Exception: pass
+    return 42
"""
        verdict, reasons, signals = audit_change(
            diff=diff,
            files_modified=["tests/test_foo.py"],
        )
        # Multiple signals: test file mod + exception swallowing + hardcoded + early return
        assert verdict in ("GAMING", "SUSPICIOUS")
        assert len(signals) > 0

    def test_suspicious_but_not_gaming(self):
        diff = """\
+    return None
"""
        verdict, reasons, signals = audit_change(
            diff=diff,
            files_modified=["src/main.py"],
        )
        # early_return has confidence 0.4, should be VALID or SUSPICIOUS
        assert verdict in ("VALID", "SUSPICIOUS")

    def test_empty_diff(self):
        verdict, reasons, signals = audit_change(
            diff="",
            files_modified=[],
        )
        assert verdict == "VALID"

    def test_test_files_with_allow(self):
        verdict, reasons, signals = audit_change(
            diff="+    result = compute()\n",
            files_modified=["tests/test_foo.py"],
            allow_test_modification=True,
        )
        assert verdict == "VALID"

    def test_project_allowlist_filters_signal(self):
        verdict, reasons, signals = audit_change(
            diff='+    return "custom benign marker"\n',
            files_modified=["src/main.py"],
            project_allowlist=["custom benign"],
        )
        assert verdict == "VALID"
        assert reasons == ["No gaming signals detected"]
        assert signals == []


# ---------------------------------------------------------------------------
# challenge_iterations
# ---------------------------------------------------------------------------

def _make_record(
    iteration: int,
    verdict: str = "VALID",
    decision: str = "COMMITTED",
    files: list[str] | None = None,
    net: int = 1,
    improved: list[str] | None = None,
    regressed: list[str] | None = None,
    gaming: list[str] | None = None,
    reasons: list[str] | None = None,
) -> ChangeRecord:
    before = TestResult(
        exit_code=1, passed=5, failed=3, errors=0, skipped=0,
        total=8, duration_seconds=1.0, output="",
    )
    after = TestResult(
        exit_code=1, passed=6, failed=2, errors=0, skipped=0,
        total=8, duration_seconds=1.0, output="",
    )
    return ChangeRecord(
        iteration=iteration,
        files_modified=files or ["src/main.py"],
        audit_verdict=verdict,
        decision=decision,
        before=before,
        after=after,
        net_improvement=net,
        tests_improved=improved or [],
        tests_regressed=regressed or [],
        gaming_signals=gaming or [],
        audit_reasons=reasons or [],
    )


class TestChallengeIterations:
    def test_clean_iterations(self):
        iterations = [
            _make_record(1, files=["src/a.py"]),
            _make_record(2, files=["src/b.py"]),
        ]
        result = challenge_iterations(iterations)
        assert result.changes_reviewed == 2
        assert len(result.vetoed) == 0
        assert any("clean" in f.lower() for f in result.findings)

    def test_detects_consecutive_suspicious(self):
        iterations = [
            _make_record(1, verdict="SUSPICIOUS"),
            _make_record(2, verdict="SUSPICIOUS"),
        ]
        result = challenge_iterations(iterations)
        assert any("consecutive" in f.lower() for f in result.findings)

    def test_detects_gaming_not_reverted(self):
        iterations = [
            _make_record(1, verdict="GAMING", decision="COMMITTED"),
        ]
        result = challenge_iterations(iterations)
        assert 1 in result.vetoed

    def test_detects_hot_files(self):
        iterations = [
            _make_record(1, files=["src/main.py"]),
            _make_record(2, files=["src/main.py"]),
            _make_record(3, files=["src/main.py"]),
        ]
        result = challenge_iterations(iterations)
        assert any("hot files" in f.lower() for f in result.findings)

    def test_empty_iterations(self):
        result = challenge_iterations([])
        assert result.changes_reviewed == 0

    def test_detects_net_regression(self):
        before = TestResult(
            exit_code=1, passed=8, failed=2, errors=0, skipped=0,
            total=10, duration_seconds=1.0, output="",
        )
        after = TestResult(
            exit_code=1, passed=5, failed=5, errors=0, skipped=0,
            total=10, duration_seconds=1.0, output="",
        )
        iterations = [
            ChangeRecord(iteration=1, before=before, after=after, decision="COMMITTED", audit_verdict="VALID"),
        ]
        result = challenge_iterations(iterations)
        assert any("regression" in f.lower() for f in result.findings)


# ---------------------------------------------------------------------------
# GamingSignal
# ---------------------------------------------------------------------------

class TestGamingSignal:
    def test_creation(self):
        sig = GamingSignal(
            name="test_signal",
            confidence=0.5,
            description="test",
        )
        assert sig.name == "test_signal"
        assert sig.confidence == 0.5
        assert sig.line_evidence == ""

    def test_with_evidence(self):
        sig = GamingSignal(
            name="test_signal",
            confidence=0.8,
            description="found issue",
            line_evidence="+ return 42",
        )
        assert sig.line_evidence == "+ return 42"


# ===========================================================================
# Gap 2 tests: analyze_successful_changes
# ===========================================================================


class TestAnalyzeSuccessfulBasic:
    """Basic happy-path for analyze_successful_changes."""

    def test_analyze_successful_basic(self):
        recs = [
            _make_record(1, files=["src/a.py"], net=2, decision="COMMITTED"),
            _make_record(2, files=["src/b.py", "src/c.py"], net=3, decision="COMMITTED"),
            _make_record(3, files=["src/d.py"], net=-1, decision="REVERTED"),
        ]
        result = analyze_successful_changes(recs)

        # 2 committed out of 3 total
        assert result["success_rate"] == round(2 / 3, 4)
        # Average files: (1 + 2) / 2 = 1.5
        assert result["avg_files_per_change"] == 1.5
        # Average net: (2 + 3) / 2 = 2.5
        assert result["avg_net_improvement"] == 2.5
        assert result["single_file_changes"] == 1
        assert result["multi_file_changes"] == 1
        assert isinstance(result["patterns"], list)
        assert isinstance(result["common_file_patterns"], dict)


class TestAnalyzeSuccessfulEmpty:
    """Edge case: no iterations or no committed iterations."""

    def test_analyze_successful_empty(self):
        result = analyze_successful_changes([])
        assert result["avg_files_per_change"] == 0.0
        assert result["avg_net_improvement"] == 0.0
        assert result["success_rate"] == 0.0
        assert result["single_file_changes"] == 0
        assert result["multi_file_changes"] == 0
        assert result["patterns"] == []

    def test_no_committed(self):
        recs = [
            _make_record(1, decision="REVERTED"),
            _make_record(2, decision="PENDING"),
        ]
        result = analyze_successful_changes(recs)
        assert result["success_rate"] == 0.0
        assert result["single_file_changes"] == 0


class TestAnalyzeSuccessfulPatterns:
    """Verify pattern strings are generated correctly."""

    def test_analyze_successful_patterns(self):
        recs = [
            _make_record(1, files=["src/foo.py"], net=1, decision="COMMITTED"),
            _make_record(2, files=["src/bar.py"], net=2, decision="COMMITTED"),
            _make_record(3, files=["src/baz.py"], net=1, decision="COMMITTED"),
        ]
        result = analyze_successful_changes(recs)
        patterns = result["patterns"]

        # Should mention single-file percentage
        assert any("single file" in p for p in patterns)
        # Should mention .py files
        assert any(".py" in p for p in patterns)

    def test_mixed_extensions(self):
        recs = [
            _make_record(1, files=["a.py", "b.js"], net=1, decision="COMMITTED"),
            _make_record(2, files=["c.py"], net=1, decision="COMMITTED"),
        ]
        result = analyze_successful_changes(recs)
        common = result["common_file_patterns"]
        assert ".py" in common
        assert common[".py"] >= 2


# ===========================================================================
# Gap 5 tests: classify_change_quality & quality_distribution
# ===========================================================================


class TestClassifyExcellent:
    """EXCELLENT: net>=3, no gaming, no regressions, COMMITTED."""

    def test_classify_excellent(self):
        r = _make_record(1, net=5, decision="COMMITTED")
        assert classify_change_quality(r) == "EXCELLENT"

    def test_excellent_boundary(self):
        r = _make_record(1, net=3, decision="COMMITTED")
        assert classify_change_quality(r) == "EXCELLENT"

    def test_not_excellent_with_gaming(self):
        r = _make_record(1, net=5, decision="COMMITTED", gaming=["test weakening"])
        assert classify_change_quality(r) != "EXCELLENT"

    def test_not_excellent_with_regressions(self):
        r = _make_record(1, net=5, decision="COMMITTED", regressed=["test_x"])
        assert classify_change_quality(r) != "EXCELLENT"


class TestClassifyGood:
    """GOOD: net>=1, no gaming, COMMITTED."""

    def test_classify_good(self):
        r = _make_record(1, net=2, decision="COMMITTED")
        assert classify_change_quality(r) == "GOOD"

    def test_good_with_regressions_still_good(self):
        # net>=1, no gaming, committed -- regressions don't block GOOD
        r = _make_record(1, net=2, decision="COMMITTED", regressed=["test_a"])
        assert classify_change_quality(r) == "GOOD"

    def test_good_boundary(self):
        r = _make_record(1, net=1, decision="COMMITTED")
        assert classify_change_quality(r) == "GOOD"


class TestClassifyNeutral:
    """NEUTRAL: net==0, no regressions, COMMITTED."""

    def test_classify_neutral(self):
        r = _make_record(1, net=0, decision="COMMITTED", files=["a.py"])
        assert classify_change_quality(r) == "NEUTRAL"

    def test_neutral_pending(self):
        # PENDING with zero net falls through to NEUTRAL
        r = _make_record(1, net=0, decision="PENDING")
        assert classify_change_quality(r) == "NEUTRAL"


class TestClassifyPoor:
    """POOR: (net==0 AND regressions) OR (REVERTED without gaming)."""

    def test_classify_poor_zero_with_regressions(self):
        r = _make_record(1, net=0, decision="COMMITTED", regressed=["test_x"])
        assert classify_change_quality(r) == "POOR"

    def test_classify_poor_reverted_no_gaming(self):
        r = _make_record(1, net=0, decision="REVERTED")
        assert classify_change_quality(r) == "POOR"

    def test_classify_poor_reverted_small_loss(self):
        # Reverted but net_improvement is not < -2 and no gaming -> POOR
        r = _make_record(1, net=-1, decision="REVERTED")
        assert classify_change_quality(r) == "POOR"


class TestClassifyHarmful:
    """HARMFUL: REVERTED AND (gaming OR net < -2)."""

    def test_classify_harmful_gaming(self):
        r = _make_record(1, net=0, decision="REVERTED", gaming=["test deletion"])
        assert classify_change_quality(r) == "HARMFUL"

    def test_classify_harmful_large_regression(self):
        r = _make_record(1, net=-3, decision="REVERTED")
        assert classify_change_quality(r) == "HARMFUL"

    def test_classify_harmful_both(self):
        r = _make_record(1, net=-5, decision="REVERTED", gaming=["assertion removal"])
        assert classify_change_quality(r) == "HARMFUL"


class TestQualityDistribution:
    """quality_distribution returns counts for every quality level."""

    def test_quality_distribution(self):
        recs = [
            _make_record(1, net=5, decision="COMMITTED"),         # EXCELLENT
            _make_record(2, net=2, decision="COMMITTED"),          # GOOD
            _make_record(3, net=0, decision="COMMITTED"),          # NEUTRAL
            _make_record(4, net=0, decision="COMMITTED",
                         regressed=["t1"]),                        # POOR
            _make_record(5, net=-3, decision="REVERTED"),          # HARMFUL
        ]
        dist = quality_distribution(recs)
        assert dist["EXCELLENT"] == 1
        assert dist["GOOD"] == 1
        assert dist["NEUTRAL"] == 1
        assert dist["POOR"] == 1
        assert dist["HARMFUL"] == 1

    def test_distribution_empty(self):
        dist = quality_distribution([])
        assert all(v == 0 for v in dist.values())
        assert set(dist.keys()) == {
            "EXCELLENT", "GOOD", "NEUTRAL", "POOR", "HARMFUL"
        }

    def test_distribution_all_same(self):
        recs = [
            _make_record(i, net=1, decision="COMMITTED") for i in range(5)
        ]
        dist = quality_distribution(recs)
        assert dist["GOOD"] == 5
        assert dist["EXCELLENT"] == 0


# ===========================================================================
# Gap 6 tests: Enhanced adversarial challenges (checks 5-9)
# ===========================================================================


class TestChallengeCumulativeGaming:
    """Check 5: cumulative gaming signal count > 5."""

    def test_challenge_cumulative_gaming(self):
        recs = [
            _make_record(1, gaming=["g1", "g2"]),
            _make_record(2, gaming=["g3", "g4"]),
            _make_record(3, gaming=["g5", "g6"]),
        ]
        result = challenge_iterations(recs)
        assert any("cumulative gaming signal density" in f.lower()
                    for f in result.findings)
        # Verify the count is reported correctly
        assert any("6 signals" in f for f in result.findings)

    def test_no_cumulative_gaming_under_threshold(self):
        recs = [
            _make_record(1, gaming=["g1"]),
            _make_record(2, gaming=["g2"]),
        ]
        result = challenge_iterations(recs)
        assert not any("cumulative gaming signal density" in f.lower()
                       for f in result.findings)

    def test_cumulative_gaming_exactly_at_threshold(self):
        # Exactly 5 should NOT trigger (threshold is > 5)
        recs = [
            _make_record(1, gaming=["g1", "g2", "g3"]),
            _make_record(2, gaming=["g4", "g5"]),
        ]
        result = challenge_iterations(recs)
        assert not any("cumulative gaming signal density" in f.lower()
                       for f in result.findings)


class TestChallengeAssertionErosion:
    """Check 6: assertion erosion (net assertions removed > 2)."""

    def test_challenge_assertion_erosion(self):
        recs = [
            _make_record(1, reasons=["assertion removed in test_a"]),
            _make_record(2, reasons=["assertion weakened in test_b"]),
            _make_record(3, reasons=["assertion deleted in test_c"]),
        ]
        result = challenge_iterations(recs)
        assert any("assertion erosion" in f.lower() for f in result.findings)
        assert any("3 assertions" in f for f in result.findings)

    def test_no_assertion_erosion_below_threshold(self):
        recs = [
            _make_record(1, reasons=["assertion removed in test_a"]),
            _make_record(2, reasons=["assertion removed in test_b"]),
        ]
        result = challenge_iterations(recs)
        assert not any("assertion erosion" in f.lower() for f in result.findings)

    def test_assertion_keyword_without_removal(self):
        # "assertion" present but no removal/weakened/deleted keyword
        recs = [
            _make_record(1, reasons=["assertion added in test_a"]),
            _make_record(2, reasons=["assertion added in test_b"]),
            _make_record(3, reasons=["assertion added in test_c"]),
        ]
        result = challenge_iterations(recs)
        assert not any("assertion erosion" in f.lower() for f in result.findings)


class TestChallengeDiminishingReturns:
    """Check 7: last N iterations all have net_improvement == 0."""

    def test_challenge_diminishing_returns(self):
        recs = [
            _make_record(1, net=3),
            _make_record(2, net=0),
            _make_record(3, net=0),
            _make_record(4, net=0),
        ]
        result = challenge_iterations(recs)
        assert any("diminishing returns" in f.lower() for f in result.findings)
        assert any("last 3" in f for f in result.findings)

    def test_no_diminishing_returns(self):
        recs = [
            _make_record(1, net=0),
            _make_record(2, net=0),
            _make_record(3, net=1),  # breaks the trailing streak
        ]
        result = challenge_iterations(recs)
        assert not any("diminishing returns" in f.lower() for f in result.findings)

    def test_all_zero(self):
        recs = [_make_record(i, net=0) for i in range(5)]
        result = challenge_iterations(recs)
        assert any("last 5" in f for f in result.findings)


class TestChallengeCoverageGap:
    """Check 8: persistent failures that appear in regressed but never improved."""

    def test_challenge_coverage_gap(self):
        recs = [
            _make_record(1, regressed=["test_login"], improved=["test_signup"]),
            _make_record(2, regressed=["test_login"]),
            _make_record(3, regressed=["test_login"], improved=["test_other"]),
        ]
        result = challenge_iterations(recs)
        assert any("persistent failures" in f.lower() for f in result.findings)
        assert any("test_login" in f for f in result.findings)

    def test_no_coverage_gap_when_addressed(self):
        recs = [
            _make_record(1, regressed=["test_login"]),
            _make_record(2, regressed=["test_login"], improved=["test_login"]),
        ]
        result = challenge_iterations(recs)
        assert not any("persistent failures" in f.lower() for f in result.findings)

    def test_no_gap_single_regression(self):
        # Only regressed once -- threshold is 2+
        recs = [_make_record(1, regressed=["test_foo"])]
        result = challenge_iterations(recs)
        assert not any("persistent failures" in f.lower() for f in result.findings)


class TestChallengeScopeCreep:
    """Check 9: avg files/change increasing over time."""

    def test_challenge_scope_creep(self):
        recs = [
            _make_record(1, files=["a.py"]),
            _make_record(2, files=["b.py"]),
            _make_record(3, files=["c.py", "d.py", "e.py"]),
            _make_record(4, files=["f.py", "g.py", "h.py", "i.py"]),
        ]
        result = challenge_iterations(recs)
        assert any("scope creep" in f.lower() for f in result.findings)

    def test_no_scope_creep_decreasing(self):
        recs = [
            _make_record(1, files=["a.py", "b.py", "c.py"]),
            _make_record(2, files=["d.py", "e.py", "f.py"]),
            _make_record(3, files=["g.py"]),
            _make_record(4, files=["h.py"]),
        ]
        result = challenge_iterations(recs)
        assert not any("scope creep" in f.lower() for f in result.findings)

    def test_no_scope_creep_too_few_iterations(self):
        # Need >= 4 iterations to detect scope creep
        recs = [
            _make_record(1, files=["a.py"]),
            _make_record(2, files=["b.py", "c.py", "d.py"]),
        ]
        result = challenge_iterations(recs)
        assert not any("scope creep" in f.lower() for f in result.findings)

    def test_scope_creep_reports_averages(self):
        recs = [
            _make_record(1, files=["a.py"]),
            _make_record(2, files=["b.py"]),
            _make_record(3, files=["c.py", "d.py", "e.py"]),
            _make_record(4, files=["f.py", "g.py", "h.py", "i.py"]),
        ]
        result = challenge_iterations(recs)
        scope_msgs = [f for f in result.findings if "scope creep" in f.lower()]
        assert len(scope_msgs) == 1
        # First half avg = 1.0, second half avg = 3.5
        assert "1.0" in scope_msgs[0]
        assert "3.5" in scope_msgs[0]
