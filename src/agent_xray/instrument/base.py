"""Thread-safe JSONL step recorder for agent-xray instrumentation.

Provides :class:`StepRecorder`, the base class used by all SDK-specific
instrumentors to write JSONL traces that ``load_tasks()`` can read natively.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..schema import SCHEMA_VERSION

_DEFAULT_OUTPUT_DIR = "./traces"
_MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MiB before rotation


class StepRecorder:
    """Thread-safe JSONL writer that emits agent-xray native step records.

    Args:
        output_dir: Directory where ``.jsonl`` trace files are written.
        task_id: Default task identifier when none is supplied per-call.
        flush_every: Flush the output file after this many buffered lines.
        max_file_bytes: Rotate the output file when it exceeds this size.

    Example:
        >>> recorder = StepRecorder(output_dir="./traces")
        >>> recorder.start_task("task-1", "Buy headphones")
        >>> recorder.record_step("task-1", 1, "browser_navigate", {"url": "https://shop.test"})
        >>> recorder.end_task("task-1", "success")
        >>> recorder.close()
    """

    def __init__(
        self,
        output_dir: str | Path = _DEFAULT_OUTPUT_DIR,
        *,
        task_id: str | None = None,
        flush_every: int = 1,
        max_file_bytes: int = _MAX_FILE_BYTES,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._default_task_id = task_id or "default-task"
        self._flush_every = max(1, flush_every)
        self._max_file_bytes = max_file_bytes
        self._lock = threading.Lock()
        self._step_counters: dict[str, int] = {}
        self._buffered = 0
        self._handle: Any | None = None
        self._current_path: Path | None = None

    def _ensure_open(self) -> Any:
        if self._handle is not None and not self._handle.closed:
            if self._current_path is not None:
                try:
                    size = self._current_path.stat().st_size
                except OSError:
                    size = 0
                if size >= self._max_file_bytes:
                    self._handle.close()
                    self._handle = None
        if self._handle is None or self._handle.closed:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            day = datetime.now(timezone.utc).strftime("%Y%m%d")
            base = f"agent-steps-{day}.jsonl"
            path = self._output_dir / base
            suffix = 0
            while path.exists() and path.stat().st_size >= self._max_file_bytes:
                suffix += 1
                path = self._output_dir / f"agent-steps-{day}-{suffix}.jsonl"
            self._current_path = path
            self._handle = open(path, "a", encoding="utf-8")  # noqa: SIM115
        return self._handle

    def _write_line(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)
        with self._lock:
            handle = self._ensure_open()
            handle.write(line + "\n")
            self._buffered += 1
            if self._buffered >= self._flush_every:
                handle.flush()
                self._buffered = 0

    def _next_step(self, task_id: str) -> int:
        count = self._step_counters.get(task_id, 0) + 1
        self._step_counters[task_id] = count
        return count

    def start_task(
        self,
        task_id: str | None = None,
        task_text: str | None = None,
        *,
        task_category: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Record the start of a new task.

        Args:
            task_id: Unique task identifier. Uses the default when omitted.
            task_text: Human-readable task description.
            task_category: Optional category label.
            metadata: Extra metadata to attach to the start event.

        Returns:
            The resolved ``task_id``.
        """
        resolved_id = task_id or self._default_task_id
        payload: dict[str, Any] = {
            "event": "task_start",
            "task_id": resolved_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "schema_version": SCHEMA_VERSION,
        }
        if task_text is not None:
            payload["user_text"] = task_text
        if task_category is not None:
            payload["task_category"] = task_category
        if metadata:
            payload.update(metadata)
        self._write_line(payload)
        return resolved_id

    def record_step(
        self,
        task_id: str | None = None,
        step: int | None = None,
        tool_name: str = "",
        tool_input: dict[str, Any] | None = None,
        tool_result: str | None = None,
        error: str | None = None,
        *,
        duration_ms: int | None = None,
        model_name: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cost_usd: float | None = None,
        tools_available: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Record a single agent step.

        Args:
            task_id: Task this step belongs to.
            step: Explicit step index. Auto-incremented when omitted.
            tool_name: Name of the tool invoked.
            tool_input: Structured tool input.
            tool_result: Tool output text.
            error: Error message, if the step failed.
            duration_ms: Step duration in milliseconds.
            model_name: Model identifier.
            input_tokens: Input token count.
            output_tokens: Output token count.
            cost_usd: Estimated cost in USD.
            tools_available: Tool names exposed to the model.
            **kwargs: Additional fields written into the step payload.
        """
        resolved_id = task_id or self._default_task_id
        resolved_step = step if step is not None else self._next_step(resolved_id)

        payload: dict[str, Any] = {
            "task_id": resolved_id,
            "step": resolved_step,
            "tool_name": tool_name,
            "tool_input": tool_input or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "schema_version": SCHEMA_VERSION,
        }
        if tool_result is not None:
            payload["tool_result"] = tool_result
        if error is not None:
            payload["error"] = error
        if duration_ms is not None:
            payload["duration_ms"] = duration_ms
        if model_name is not None:
            payload["model_name"] = model_name
        if input_tokens is not None:
            payload["input_tokens"] = input_tokens
        if output_tokens is not None:
            payload["output_tokens"] = output_tokens
        if cost_usd is not None:
            payload["cost_usd"] = cost_usd
        if tools_available is not None:
            payload["tools_available"] = tools_available

        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value

        self._write_line(payload)

    def end_task(
        self,
        task_id: str | None = None,
        status: str = "success",
        *,
        final_answer: str | None = None,
        total_duration_s: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record the completion of a task.

        Args:
            task_id: Task identifier.
            status: Outcome label such as ``success`` or ``failed``.
            final_answer: Final answer text.
            total_duration_s: Total runtime in seconds.
            metadata: Extra metadata to attach to the completion event.
        """
        resolved_id = task_id or self._default_task_id
        step_count = self._step_counters.get(resolved_id, 0)
        payload: dict[str, Any] = {
            "event": "task_complete",
            "task_id": resolved_id,
            "outcome": status,
            "total_steps": step_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "schema_version": SCHEMA_VERSION,
        }
        if final_answer is not None:
            payload["final_answer"] = final_answer
        if total_duration_s is not None:
            payload["total_duration_s"] = total_duration_s
        if metadata:
            payload.update(metadata)
        self._write_line(payload)

    def close(self) -> None:
        """Flush and close the output file."""
        with self._lock:
            if self._handle is not None and not self._handle.closed:
                self._handle.flush()
                self._handle.close()
                self._handle = None

    @property
    def output_dir(self) -> Path:
        """Return the configured output directory."""
        return self._output_dir

    @property
    def current_path(self) -> Path | None:
        """Return the current trace file path, or ``None`` if not yet opened."""
        return self._current_path

    def __enter__(self) -> StepRecorder:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _monotonic_ms() -> float:
    """Return monotonic clock value in milliseconds."""
    return time.monotonic() * 1000


__all__ = [
    "StepRecorder",
]
