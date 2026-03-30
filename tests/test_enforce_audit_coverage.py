from __future__ import annotations

from agent_xray.enforce_audit import audit_change, detect_test_file_modification


def test_test_file_modification_confidence_is_low() -> None:
    signal = detect_test_file_modification(["tests/test_checkout.py"])

    assert signal is not None
    assert signal.confidence == 0.3


def test_no_visible_effect_phrase_is_allowlisted() -> None:
    verdict, reasons, signals = audit_change(
        diff='+    note = "no visible effect"\n',
        files_modified=["src/notes.py"],
    )

    assert verdict == "VALID"
    assert signals == []
    assert reasons == ["No gaming signals detected"]


def test_project_allowlist_skips_allowed_test_paths() -> None:
    verdict, reasons, signals = audit_change(
        diff="+    helper = build_fixture()\n",
        files_modified=["tests/fixtures/test_helper.py"],
        project_allowlist=["tests/fixtures"],
    )

    assert verdict == "VALID"
    assert signals == []
    assert reasons == ["No gaming signals detected"]
