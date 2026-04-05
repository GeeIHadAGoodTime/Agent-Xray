from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_project_version(pyproject_text: str) -> str:
    project_match = re.search(
        r"^\[project\]\s*(?P<body>.*?)(?=^\[|\Z)",
        pyproject_text,
        re.MULTILINE | re.DOTALL,
    )
    if not project_match:
        raise ValueError("pyproject.toml is missing a [project] section")

    version_match = re.search(
        r'^version\s*=\s*"([^"]+)"\s*$',
        project_match.group("body"),
        re.MULTILINE,
    )
    if not version_match:
        raise ValueError('pyproject.toml [project] section is missing version = "..."')
    return version_match.group(1)


def _count_mcp_tools(mcp_server_text: str) -> int:
    return len(re.findall(r"^\s*@server\.tool\s*\(", mcp_server_text, re.MULTILINE))


def _count_root_causes(root_cause_text: str) -> int:
    root_causes_match = re.search(
        r"^ROOT_CAUSES\s*=\s*\{(?P<body>.*?)^\}\s*^\s*BASELINE_CONFIDENCE_SCORES\s*=",
        root_cause_text,
        re.MULTILINE | re.DOTALL,
    )
    if not root_causes_match:
        raise ValueError("Could not locate ROOT_CAUSES in src/agent_xray/root_cause.py")
    return len(
        re.findall(r'^\s{4}"[^"]+":\s*\{', root_causes_match.group("body"), re.MULTILINE)
    )


def _require_match(text: str, pattern: str, description: str, errors: list[str]) -> re.Match[str] | None:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        errors.append(f"{description}: expected pattern not found")
        return None
    return match


def _check_number(text: str, pattern: str, expected: int, description: str, errors: list[str]) -> None:
    match = _require_match(text, pattern, description, errors)
    if not match:
        return
    actual = int(match.group(1))
    if actual != expected:
        errors.append(f"{description}: expected {expected}, found {actual}")


def main() -> int:
    pyproject_path = ROOT / "pyproject.toml"
    changelog_path = ROOT / "CHANGELOG.md"
    capabilities_path = ROOT / "CAPABILITIES.md"
    readme_path = ROOT / "README.md"
    mcp_server_path = ROOT / "src" / "agent_xray" / "mcp_server.py"
    root_cause_path = ROOT / "src" / "agent_xray" / "root_cause.py"

    pyproject_text = _read(pyproject_path)
    changelog_text = _read(changelog_path)
    capabilities_text = _read(capabilities_path)
    readme_text = _read(readme_path)
    mcp_server_text = _read(mcp_server_path)
    root_cause_text = _read(root_cause_path)

    version = _extract_project_version(pyproject_text)
    actual_tool_count = _count_mcp_tools(mcp_server_text)
    actual_root_cause_count = _count_root_causes(root_cause_text)

    errors: list[str] = []

    if not re.search(rf"^##\s*\[{re.escape(version)}\](?:\s|-|$)", changelog_text, re.MULTILINE):
        errors.append(f"CHANGELOG.md: missing entry for version {version}")

    if not re.search(rf"^Version:\s*{re.escape(version)}\b", capabilities_text, re.MULTILINE):
        errors.append(f"CAPABILITIES.md: expected header version {version}")

    _check_number(
        capabilities_text,
        r"^\| `mcp_server\.py` \| .* \| FastMCP server with (\d+) tools \|$",
        actual_tool_count,
        "CAPABILITIES.md mcp_server.py summary count",
        errors,
    )
    _check_number(
        capabilities_text,
        r"^### MCP Tools \((\d+) exposed via `mcp_server\.py`\)$",
        actual_tool_count,
        "CAPABILITIES.md MCP Tools heading count",
        errors,
    )
    _check_number(
        capabilities_text,
        r"^> (\d+) MCP tools total\.",
        actual_tool_count,
        "CAPABILITIES.md MCP tools summary count",
        errors,
    )
    _check_number(
        capabilities_text,
        r"^\| `root_cause\.py` \| .* \| (\d+) root cause classifiers with configurable thresholds \|$",
        actual_root_cause_count,
        "CAPABILITIES.md root_cause.py summary count",
        errors,
    )

    _check_number(
        readme_text,
        r"^\*\*Total:\s*(\d+)\s+MCP tools\*\*",
        actual_tool_count,
        "README.md MCP tools total",
        errors,
    )
    _check_number(
        readme_text,
        r"^\| `root_cause` \| Classifies failure modes using a (\d+)-category cascade classifier with evidence \|$",
        actual_root_cause_count,
        "README.md root_cause tool description count",
        errors,
    )
    _check_number(
        readme_text,
        r"^\s+-> root_cause\.py\s+(\d+)-category failure classifier with cascade ordering$",
        actual_root_cause_count,
        "README.md architecture tree root cause count",
        errors,
    )

    if errors:
        print("Release validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(
        f"Release validation passed for {version} "
        f"({actual_tool_count} MCP tools, {actual_root_cause_count} root cause categories)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
