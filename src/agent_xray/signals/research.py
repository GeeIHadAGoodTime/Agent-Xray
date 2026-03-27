"""Signals for search-heavy research and source synthesis tasks."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlparse

from ..schema import AgentStep, AgentTask


class ResearchDetector:
    """Detect searching, reading, and citation-backed synthesis behavior."""

    name = "research"

    SEARCH_TOOLS = {"web_search", "search", "google", "bing", "duckduckgo", "arxiv_search"}
    READ_TOOLS = {"read_url", "fetch_page", "scrape", "browser_navigate", "browser_snapshot"}
    CITE_PATTERNS = [r"https?://", r"\[\d+\]", r"according to", r"source:"]
    URL_RE = re.compile(r"https?://[^\s'\"`]+")

    def detect_step(self, step: AgentStep) -> dict[str, bool]:
        tool = step.tool_name.lower()
        result = step.tool_result or ""
        return {
            "is_search": tool in self.SEARCH_TOOLS or "search" in tool,
            "is_read": self._is_read_tool(tool),
            "is_synthesis": tool in {"respond", "answer", "summarize", "write"},
            "has_citation": any(
                re.search(pattern, result, re.IGNORECASE) for pattern in self.CITE_PATTERNS
            ),
            "has_url_in_result": "http" in result,
        }

    def summarize(
        self, task: AgentTask, step_signals: list[dict[str, bool]]
    ) -> dict[str, int | float | bool]:
        searches = sum(1 for signals in step_signals if signals["is_search"])
        reads = sum(1 for signals in step_signals if signals["is_read"])
        citations = sum(1 for signals in step_signals if signals["has_citation"])
        urls = sum(1 for signals in step_signals if signals["has_url_in_result"])
        return {
            "search_count": searches,
            "read_count": reads,
            "source_diversity": self._count_unique_domains(task),
            "citation_count": citations,
            "search_to_read_ratio": searches / max(reads, 1),
            "has_synthesis_step": any(signals["is_synthesis"] for signals in step_signals),
            "url_references": urls,
        }

    def _count_unique_domains(self, task: AgentTask) -> int:
        domains: set[str] = set()
        for step in task.sorted_steps:
            for candidate in self._iter_sources(step):
                for url in self.URL_RE.findall(candidate):
                    host = urlparse(url).netloc.lower()
                    if host:
                        domains.add(host)
        return len(domains)

    def _is_read_tool(self, tool: str) -> bool:
        return (
            tool in self.READ_TOOLS
            or tool.startswith("read_url")
            or "fetch" in tool
            or "scrape" in tool
        )

    def _iter_sources(self, step: AgentStep) -> Iterable[str]:
        if page_url := self._page_url(step):
            yield page_url
        if step.tool_result:
            yield step.tool_result
        yield from self._iter_strings(step.tool_input)

    def _iter_strings(self, value: Any) -> Iterable[str]:
        if isinstance(value, str):
            yield value
            return
        if isinstance(value, dict):
            for item in value.values():
                yield from self._iter_strings(item)
            return
        if isinstance(value, list):
            for item in value:
                yield from self._iter_strings(item)

    def _page_url(self, step: AgentStep) -> str | None:
        page_url = getattr(step, "page_url", None)
        if page_url:
            return str(page_url)
        browser = getattr(step, "browser", None)
        browser_page_url = getattr(browser, "page_url", None)
        return str(browser_page_url) if browser_page_url else None
