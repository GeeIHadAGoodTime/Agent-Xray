from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from ..schema import AgentStep, AgentTask


class CodingDetector:
    name = "coding"

    COMMON_TLDS = {
        "ai",
        "app",
        "com",
        "dev",
        "edu",
        "gov",
        "io",
        "net",
        "org",
        "test",
        "us",
    }
    FILE_TOOLS = {"write_file", "read_file", "edit_file", "create_file", "delete_file", "patch"}
    TEST_TOOLS = {"run_tests", "pytest", "test", "run_test"}
    BUILD_TOOLS = {"build", "compile", "make", "cargo_build", "npm_run"}
    LINT_TOOLS = {"lint", "ruff", "eslint", "mypy", "typecheck"}
    GIT_TOOLS = {"git_commit", "git_push", "git_diff", "git_status"}
    SHELL_TOOLS = {"bash", "shell", "terminal", "execute", "run_command"}
    FILE_PATH_RE = re.compile(
        r"(?:[A-Za-z]:)?[\\/][^\s'\"`]+|[A-Za-z0-9_.-]+(?:[\\/][A-Za-z0-9_.-]+)+|[A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,8}"
    )

    def detect_step(self, step: AgentStep) -> dict[str, bool]:
        tool = step.tool_name.lower()
        return {
            "is_file_op": tool in self.FILE_TOOLS
            or any(keyword in tool for keyword in ("file", "write", "read", "edit")),
            "is_test": tool in self.TEST_TOOLS or "test" in tool,
            "is_build": tool in self.BUILD_TOOLS or "build" in tool or "compile" in tool,
            "is_lint": tool in self.LINT_TOOLS or "lint" in tool or "type" in tool,
            "is_git": tool in self.GIT_TOOLS or "git" in tool,
            "is_shell": tool in self.SHELL_TOOLS or "bash" in tool or "shell" in tool,
            "has_error": step.error is not None,
            "has_file_path": self._has_file_path(step.tool_input),
        }

    def summarize(
        self, task: AgentTask, step_signals: list[dict[str, bool]]
    ) -> dict[str, int | float | bool]:
        file_ops = sum(1 for signals in step_signals if signals["is_file_op"])
        tests = sum(1 for signals in step_signals if signals["is_test"])
        builds = sum(1 for signals in step_signals if signals["is_build"])
        lints = sum(1 for signals in step_signals if signals["is_lint"])
        git_ops = sum(1 for signals in step_signals if signals["is_git"])
        shell_commands = sum(1 for signals in step_signals if signals["is_shell"])
        errors = sum(1 for signals in step_signals if signals["has_error"])
        return {
            "file_operations": file_ops,
            "test_runs": tests,
            "build_runs": builds,
            "lint_runs": lints,
            "git_operations": git_ops,
            "shell_commands": shell_commands,
            "error_count": errors,
            "test_to_edit_ratio": tests / max(file_ops, 1),
            "has_test_verify_cycle": tests > 0 and file_ops > 0,
            "unique_files_touched": self._count_unique_files(task),
        }

    def _has_file_path(self, value: Any) -> bool:
        return any(True for text in self._iter_strings(value) for _ in self._iter_file_matches(text))

    def _count_unique_files(self, task: AgentTask) -> int:
        files: set[str] = set()
        for step in task.sorted_steps:
            for text in self._iter_strings(step.tool_input):
                files.update(self._iter_file_matches(text))
        return len(files)

    def _iter_file_matches(self, text: str) -> Iterable[str]:
        for match in self.FILE_PATH_RE.finditer(text):
            candidate = match.group(0).rstrip(".,:;")
            if candidate and not self._is_likely_url(candidate):
                yield candidate

    def _is_likely_url(self, value: str) -> bool:
        text = value.strip().rstrip(".,:;")
        lowered = text.lower()
        if lowered.startswith(("http://", "https://")) or "://" in lowered:
            return True
        if re.fullmatch(r"\d{1,2}/\d{1,2}(?:/\d{2,4})?", text):
            return True
        head = lowered.lstrip("/").split("/", 1)[0]
        if head.count(".") != 1:
            return False
        name, suffix = head.rsplit(".", 1)
        return bool(name) and suffix in self.COMMON_TLDS and re.fullmatch(r"[a-z0-9_-]+", name)

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
