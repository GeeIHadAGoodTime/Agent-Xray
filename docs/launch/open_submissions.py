"""Open submission pages for HN and Reddit with pre-filled content.

Usage:
    python open_submissions.py hn          # Open HN submit page + copy first comment
    python open_submissions.py reddit-ml   # Open r/MachineLearning + copy body
    python open_submissions.py reddit-lc   # Open r/LangChain + copy body
    python open_submissions.py reddit-ll   # Open r/LocalLLaMA + copy body
    python open_submissions.py github      # Open GitHub repo settings for social preview
    python open_submissions.py all         # Open everything in sequence
"""

import subprocess
import sys
import urllib.parse
import webbrowser

REPO_URL = "https://github.com/GeeIHadAGoodTime/Agent-Xray"

# --- Hacker News ---

HN_TITLE = "Show HN: agent-xray \u2013 Debug AI agent failures that produce no error logs"

HN_FIRST_COMMENT = r"""Hi HN \u2014 I built agent-xray because my AI agent was failing 30% of tasks and I couldn't figure out why from the traces.

Existing tools (LangSmith, Langfuse, AgentOps) show you what happened \u2014 traces, spans, latencies. But when your agent just does the wrong thing, there's no stack trace. You're reading raw JSON trying to figure out what went wrong.

agent-xray reconstructs the decision surface at each step (what was in the prompt, what tools were available, what the model was reasoning about) and classifies failures into 22 root-cause categories (spin, tool_bug, early_abort, routing_bug, etc.).

One command triages all your tasks:

    pip install agent-xray
    agent-xray triage ./traces

It also has a pytest plugin so you can assert on agent execution quality in CI:

    def test_checkout_agent(xray):
        report = xray.analyze(steps)
        assert report.grade in ("GOLDEN", "GOOD")

Key design choices:
- Local-first, fully offline, zero required dependencies
- Framework-agnostic (OpenAI, Anthropic, LangChain, CrewAI, OpenTelemetry)
- 37 MCP tools so your AI coding assistant can debug your AI agent
- Grades measure execution structure, not output correctness (it's honest about this)
- MIT licensed

It's complementary to production tracing tools \u2014 use LangSmith to collect, use agent-xray to debug.

Happy to answer questions about the approach, the grading heuristics, or what I learned debugging 350+ agent tasks."""

# --- Reddit ---

REDDIT_ML_TITLE = "[P] agent-xray: Open-source structural grading and root-cause classification for AI agent traces"

REDDIT_ML_BODY = """I built an open-source tool for debugging AI agent failures \u2014 the kind that produce no error logs, where the agent just does the wrong thing.

**Problem**: Existing observability tools (LangSmith, Langfuse, Arize Phoenix) show traces and latencies, but when an agent spins on the same action or gives up after one error, there's no automated way to detect and classify that failure pattern.

**Approach**: agent-xray reads agent trace logs (JSONL from any framework) and:
1. Assigns structural execution grades (GOLDEN/GOOD/OK/WEAK/BROKEN) using configurable JSON rulesets
2. Classifies failures into 22 root-cause categories (spin, tool_bug, early_abort, routing_bug, etc.) using a cascade heuristic classifier
3. Reconstructs the "decision surface" at each step \u2014 what the agent saw, what tools it had, what it was reasoning about
4. Compares runs before/after a fix to quantify impact across all tasks

**Design choices**:
- Grades measure execution structure (tool diversity, loop resistance, error rate), NOT output correctness. The tool is explicit about this distinction.
- The root-cause classifier is heuristic-based, not ML-based. It's a cascade of pattern checks. Simple but systematic.
- Local-first, zero required dependencies, fully offline. No account, no telemetry.
- Framework-agnostic: adapters for OpenAI, Anthropic, LangChain, CrewAI, OpenTelemetry.

**Limitations**:
- Heuristic grading can be wrong \u2014 it's a triage signal, not a verdict
- Root-cause classification is shallow pattern matching, not deep causal analysis
- Tested primarily on browser-based agent tasks; may need ruleset tuning for other domains

**Try it**:

    pip install agent-xray
    agent-xray quickstart

GitHub: https://github.com/GeeIHadAGoodTime/Agent-Xray
MIT licensed.

Happy to discuss the approach, limitations, or how it compares to other tools in the space."""

REDDIT_LC_TITLE = "Open-source tool for debugging LangChain/LangGraph agent failures \u2014 structural grading + root-cause classification"

REDDIT_LC_BODY = """I built agent-xray to debug agent failures that don't produce error logs. It reads your existing trace files and:

- Grades each execution structurally (GOLDEN \u2192 BROKEN)
- Classifies failure mode (spin, tool_bug, early_abort, etc.)
- Reconstructs what the agent saw at each decision point
- Has a pytest plugin for CI quality gates

Works with LangChain/LangGraph traces out of the box (`--format langchain`). Also supports OpenAI, Anthropic, CrewAI, and OpenTelemetry.

Quick start:

    pip install agent-xray
    agent-xray quickstart

It's complementary to LangSmith \u2014 use LangSmith to collect traces, use agent-xray to debug specific failures locally.

GitHub: https://github.com/GeeIHadAGoodTime/Agent-Xray"""

REDDIT_LL_TITLE = "agent-xray: Debug your AI agents locally \u2014 structural grading, root-cause classification, zero dependencies, fully offline"

REDDIT_LL_BODY = """Built an open-source agent debugging tool that runs entirely locally. No cloud, no account, no telemetry.

It reads agent trace logs and tells you: this task is BROKEN because the agent was spinning (calling the same tool 4 times with no progress). Here's what the agent saw. Here's what to fix.

- 22 root-cause categories
- Configurable grading rulesets
- pytest plugin for CI quality gates
- 37 MCP tools (Claude Code / Cursor can debug your agents)
- Zero required dependencies

    pip install agent-xray
    agent-xray quickstart

GitHub: https://github.com/GeeIHadAGoodTime/Agent-Xray
MIT licensed, Python 3.10+."""


def copy_to_clipboard(text: str) -> None:
    """Copy text to Windows clipboard."""
    proc = subprocess.Popen(["clip"], stdin=subprocess.PIPE)
    proc.communicate(text.encode("utf-16-le"))


def open_hn() -> None:
    params = urllib.parse.urlencode({"u": REPO_URL, "t": HN_TITLE})
    url = f"https://news.ycombinator.com/submitlink?{params}"
    copy_to_clipboard(HN_FIRST_COMMENT)
    print(f"[HN] Opening submission page...")
    print(f"[HN] First comment COPIED TO CLIPBOARD - paste it after submitting")
    print(f"[HN] URL: {url}")
    webbrowser.open(url)


def open_reddit(subreddit: str, title: str, body: str) -> None:
    params = urllib.parse.urlencode({
        "type": "TEXT",
        "title": title,
    })
    url = f"https://www.reddit.com/r/{subreddit}/submit?{params}"
    copy_to_clipboard(body)
    print(f"[Reddit r/{subreddit}] Opening submission page...")
    print(f"[Reddit r/{subreddit}] Post body COPIED TO CLIPBOARD - paste into body field")
    print(f"[Reddit r/{subreddit}] URL: {url}")
    webbrowser.open(url)


def open_github_settings() -> None:
    url = f"{REPO_URL}/settings"
    print(f"[GitHub] Opening repo settings for social preview upload...")
    print(f"[GitHub] Scroll to 'Social preview' section and upload a terminal screenshot")
    webbrowser.open(url)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return

    target = sys.argv[1].lower()

    if target == "hn":
        open_hn()
    elif target == "reddit-ml":
        open_reddit("MachineLearning", REDDIT_ML_TITLE, REDDIT_ML_BODY)
    elif target == "reddit-lc":
        open_reddit("LangChain", REDDIT_LC_TITLE, REDDIT_LC_BODY)
    elif target == "reddit-ll":
        open_reddit("LocalLLaMA", REDDIT_LL_TITLE, REDDIT_LL_BODY)
    elif target == "github":
        open_github_settings()
    elif target == "all":
        print("=" * 60)
        print("LAUNCH SEQUENCE")
        print("=" * 60)
        print()
        print("Step 1: Hacker News")
        print("-" * 40)
        open_hn()
        input("\nPress Enter after submitting HN post and pasting first comment...")
        print()
        print("Step 2: Reddit r/MachineLearning")
        print("-" * 40)
        open_reddit("MachineLearning", REDDIT_ML_TITLE, REDDIT_ML_BODY)
        input("\nPress Enter after submitting r/MachineLearning post...")
        print()
        print("Step 3: GitHub Social Preview")
        print("-" * 40)
        open_github_settings()
        print()
        print("=" * 60)
        print("DONE - r/LangChain and r/LocalLLaMA posts should be")
        print("spaced 1-2 days apart. Run:")
        print("  python open_submissions.py reddit-lc")
        print("  python open_submissions.py reddit-ll")
        print("=" * 60)
    else:
        print(f"Unknown target: {target}")
        print(__doc__)


if __name__ == "__main__":
    main()
