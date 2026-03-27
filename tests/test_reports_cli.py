from __future__ import annotations

import json
from argparse import Namespace

from agent_xray.cli import cmd_report


def _ns(report_type: str, log_dir, *, use_json: bool = False, markdown: bool = False, day1=None, day2=None):
    return Namespace(
        log_dir=log_dir,
        days=None,
        rules=None,
        format="auto",
        report_type=report_type,
        json=use_json,
        markdown=markdown,
        day1=day1,
        day2=day2,
    )


# ── Health ───────────────────────────────────────────────────────────


def test_report_health_text(tmp_trace_dir, capsys):
    assert cmd_report(_ns("health", tmp_trace_dir)) == 0
    assert "HEALTH DASHBOARD" in capsys.readouterr().out


def test_report_health_json(tmp_trace_dir, capsys):
    assert cmd_report(_ns("health", tmp_trace_dir, use_json=True)) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["total"] == 4
    assert "distribution" in data


def test_report_health_markdown(tmp_trace_dir, capsys):
    assert cmd_report(_ns("health", tmp_trace_dir, markdown=True)) == 0
    out = capsys.readouterr().out
    assert "## Health Dashboard" in out
    assert "| Grade | Count | Pct |" in out


# ── Golden ───────────────────────────────────────────────────────────


def test_report_golden_text(tmp_trace_dir, capsys):
    assert cmd_report(_ns("golden", tmp_trace_dir)) == 0
    assert "GOLDEN/GOOD RUNS" in capsys.readouterr().out


def test_report_golden_json(tmp_trace_dir, capsys):
    assert cmd_report(_ns("golden", tmp_trace_dir, use_json=True)) == 0
    data = json.loads(capsys.readouterr().out)
    assert "count" in data
    assert "tasks" in data


# ── Broken ───────────────────────────────────────────────────────────


def test_report_broken_text(tmp_trace_dir, capsys):
    assert cmd_report(_ns("broken", tmp_trace_dir)) == 0
    assert "BROKEN TASKS" in capsys.readouterr().out


def test_report_broken_json(tmp_trace_dir, capsys):
    assert cmd_report(_ns("broken", tmp_trace_dir, use_json=True)) == 0
    data = json.loads(capsys.readouterr().out)
    assert "count" in data
    assert "why" in data


# ── Tools ────────────────────────────────────────────────────────────


def test_report_tools_text(tmp_trace_dir, capsys):
    assert cmd_report(_ns("tools", tmp_trace_dir)) == 0
    assert "TOOL EFFECTIVENESS" in capsys.readouterr().out


def test_report_tools_json(tmp_trace_dir, capsys):
    assert cmd_report(_ns("tools", tmp_trace_dir, use_json=True)) == 0
    data = json.loads(capsys.readouterr().out)
    assert "tools" in data
    assert len(data["tools"]) > 0


# ── Flows ────────────────────────────────────────────────────────────


def test_report_flows_text(tmp_trace_dir, capsys):
    assert cmd_report(_ns("flows", tmp_trace_dir)) == 0
    assert "FLOW ANALYSIS" in capsys.readouterr().out


def test_report_flows_json(tmp_trace_dir, capsys):
    assert cmd_report(_ns("flows", tmp_trace_dir, use_json=True)) == 0
    data = json.loads(capsys.readouterr().out)
    assert "sites" in data


# ── Outcomes ─────────────────────────────────────────────────────────


def test_report_outcomes_text(tmp_trace_dir, capsys):
    assert cmd_report(_ns("outcomes", tmp_trace_dir)) == 0
    assert "OUTCOME DISTRIBUTION" in capsys.readouterr().out


def test_report_outcomes_json(tmp_trace_dir, capsys):
    assert cmd_report(_ns("outcomes", tmp_trace_dir, use_json=True)) == 0
    data = json.loads(capsys.readouterr().out)
    assert "outcomes" in data


# ── Actions ──────────────────────────────────────────────────────────


def test_report_actions_text(tmp_trace_dir, capsys):
    assert cmd_report(_ns("actions", tmp_trace_dir)) == 0
    assert "PRIORITIZED ACTION ITEMS" in capsys.readouterr().out


def test_report_actions_json(tmp_trace_dir, capsys):
    assert cmd_report(_ns("actions", tmp_trace_dir, use_json=True)) == 0
    data = json.loads(capsys.readouterr().out)
    assert "action_items" in data


# â”€â”€ Cost â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def test_report_cost_text(tmp_trace_dir, capsys):
    assert cmd_report(_ns("cost", tmp_trace_dir)) == 0
    assert "COST ANALYSIS" in capsys.readouterr().out


def test_report_cost_json(tmp_trace_dir, capsys):
    assert cmd_report(_ns("cost", tmp_trace_dir, use_json=True)) == 0
    data = json.loads(capsys.readouterr().out)
    assert "summary" in data
    assert "by_model" in data


def test_report_cost_markdown(tmp_trace_dir, capsys):
    assert cmd_report(_ns("cost", tmp_trace_dir, markdown=True)) == 0
    out = capsys.readouterr().out
    assert "## Cost Analysis" in out
    assert "| Metric | Value |" in out


# â”€â”€ Fixes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def test_report_fixes_text(tmp_trace_dir, capsys):
    assert cmd_report(_ns("fixes", tmp_trace_dir)) == 0
    assert "FIX PLAN REPORT" in capsys.readouterr().out


def test_report_fixes_json(tmp_trace_dir, capsys):
    assert cmd_report(_ns("fixes", tmp_trace_dir, use_json=True)) == 0
    data = json.loads(capsys.readouterr().out)
    assert "fixes" in data
    assert data["count"] >= 1


def test_report_fixes_markdown(tmp_trace_dir, capsys):
    assert cmd_report(_ns("fixes", tmp_trace_dir, markdown=True)) == 0
    out = capsys.readouterr().out
    assert "## Fix Plan Report" in out
    assert "| Priority | Root Cause | Affected Tasks | Impact Score | Investigate Task |" in out


# ── Compare ──────────────────────────────────────────────────────────


def test_report_compare_requires_days(tmp_trace_dir, capsys):
    result = cmd_report(_ns("compare", tmp_trace_dir))
    assert result == 1
    assert "--day1" in capsys.readouterr().out


def test_report_compare_text(tmp_trace_dir, capsys):
    assert cmd_report(_ns("compare", tmp_trace_dir, day1="20260326", day2="20260327")) == 0
    assert "DAY COMPARISON" in capsys.readouterr().out


def test_report_compare_json(tmp_trace_dir, capsys):
    assert cmd_report(_ns("compare", tmp_trace_dir, use_json=True, day1="20260326", day2="20260327")) == 0
    data = json.loads(capsys.readouterr().out)
    assert "day1" in data
    assert "day2" in data


def test_report_compare_markdown(tmp_trace_dir, capsys):
    assert cmd_report(_ns("compare", tmp_trace_dir, markdown=True, day1="20260326", day2="20260327")) == 0
    out = capsys.readouterr().out
    assert "## Day Comparison" in out
    assert "| Metric | 20260326 | 20260327 | Delta |" in out
