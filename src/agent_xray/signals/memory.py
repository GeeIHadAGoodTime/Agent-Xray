"""Signals for memory, retrieval, and context injection behavior."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from ..schema import AgentStep, AgentTask


class MemoryDetector:
    """Detect memory storage, recall, RAG retrieval, and context injection patterns."""

    name = "memory"

    STORE_TOOLS = {
        "cache_write",
        "memory_put",
        "memory_save",
        "memory_store",
        "remember",
        "save_memory",
        "store_memory",
        "vector_store_upsert",
    }
    RECALL_TOOLS = {
        "get_memory",
        "lookup_memory",
        "memory_get",
        "memory_query",
        "memory_recall",
        "recall_memory",
        "retrieve_memory",
        "search_memory",
        "vector_search",
    }
    FORGET_TOOLS = {"clear_memory", "delete_memory", "evict_memory", "forget", "forget_memory"}
    RAG_TOOLS = {
        "rag_query",
        "retrieve_context",
        "search_knowledge_base",
        "semantic_search",
        "vector_search",
    }
    CONTEXT_TOOLS = {"context_inject", "inject_context", "load_context", "prepend_context"}
    MEMORY_KEY_KEYS = ("memory_key", "key", "slot", "namespace", "memory_id", "name")
    MISS_PATTERNS = ("0 results", "memory miss", "no memory", "not found", "no relevant context")
    MEMORY_KEY_RE = re.compile(r"\b(?:memory[_ ]?key|key)\s*[:=]\s*([a-z0-9_.:/-]+)", re.IGNORECASE)

    def detect_step(self, step: AgentStep) -> dict[str, bool | str | None]:
        """Analyze one step for memory and retrieval signals."""

        tool = step.tool_name.lower()
        combined_text = " ".join(self._iter_step_strings(step)).lower()
        is_memory_store = (
            tool in self.STORE_TOOLS
            or "store_memory" in tool
            or ("remember" in tool and "forget" not in tool)
            or "save_memory" in tool
        )
        is_memory_recall = (
            tool in self.RECALL_TOOLS
            or tool in self.RAG_TOOLS
            or "memory" in tool
            and any(token in tool for token in ("get", "query", "recall", "retrieve", "search"))
            or "recall" in combined_text
        )
        is_rag_query = tool in self.RAG_TOOLS or any(
            token in combined_text
            for token in (
                "rag",
                "embedding",
                "knowledge base",
                "retriev",
                "vector",
                "semantic search",
                "top_k",
            )
        )
        return {
            "is_memory_store": is_memory_store,
            "is_memory_recall": is_memory_recall,
            "is_memory_forget": tool in self.FORGET_TOOLS or "forget" in tool or "evict" in tool,
            "is_rag_query": is_rag_query,
            "is_context_injection": tool in self.CONTEXT_TOOLS
            or "inject_context" in tool
            or "context injection" in combined_text
            or "injected context" in combined_text,
            "memory_key": self._extract_memory_key(step, combined_text),
        }

    def summarize(
        self, task: AgentTask, step_signals: list[dict[str, bool | str | None]]
    ) -> dict[str, int | float]:
        """Summarize memory and retrieval usage across a task."""

        recalls = 0
        recall_hits = 0
        keys = {
            str(signals["memory_key"]).strip()
            for signals in step_signals
            if isinstance(signals.get("memory_key"), str) and str(signals["memory_key"]).strip()
        }
        operations = 0
        rag_queries = 0
        context_injections = 0
        for step, signals in zip(task.sorted_steps, step_signals, strict=False):
            if (
                signals["is_memory_store"]
                or signals["is_memory_recall"]
                or signals["is_memory_forget"]
            ):
                operations += 1
            if signals["is_memory_recall"]:
                recalls += 1
                if self._recall_hit(step):
                    recall_hits += 1
            if signals["is_rag_query"]:
                rag_queries += 1
            if signals["is_context_injection"]:
                context_injections += 1
        return {
            "memory_operations": operations,
            "unique_keys": len(keys),
            "recall_hit_rate": recall_hits / max(recalls, 1),
            "rag_queries": rag_queries,
            "context_injections": context_injections,
        }

    def _extract_memory_key(self, step: AgentStep, combined_text: str) -> str | None:
        for key in self.MEMORY_KEY_KEYS:
            value = step.tool_input.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        match = self.MEMORY_KEY_RE.search(combined_text)
        if match:
            return match.group(1).strip()
        return None

    def _recall_hit(self, step: AgentStep) -> bool:
        if step.error:
            return False
        result = (step.tool_result or "").strip().lower()
        if not result:
            return False
        return not any(pattern in result for pattern in self.MISS_PATTERNS)

    def _iter_step_strings(self, step: AgentStep) -> Iterable[str]:
        yield step.tool_name
        if step.tool_result:
            yield step.tool_result
        if step.error:
            yield step.error
        yield from self._iter_strings(step.tool_input)
        yield from self._iter_strings(step.extensions)
        if step.llm_reasoning:
            yield step.llm_reasoning

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
