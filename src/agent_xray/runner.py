from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

_httpx: Any
try:
    import httpx as _httpx
except ImportError:  # pragma: no cover - exercised in dependency-missing environments
    _httpx = None

httpx: Any = _httpx


@runtime_checkable
class TaskRunner(Protocol):
    async def send(self, task_text: str) -> str: ...

    async def get_status(self, task_id: str) -> str: ...


@dataclass(slots=True)
class StaticRunner:
    task_id: str = "task-1"
    status: str = "completed"

    async def send(self, task_text: str) -> str:
        return self.task_id

    async def get_status(self, task_id: str) -> str:
        return self.status


@dataclass(slots=True)
class GenericHTTPRunner:
    base_url: str
    send_path: str = "/tasks"
    status_path_template: str = "/tasks/{task_id}"
    timeout_s: float = 30.0
    headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_httpx()

    async def send(self, task_text: str) -> str:
        client_module = _require_httpx()
        async with client_module.AsyncClient(
            timeout=self.timeout_s, headers=self.headers
        ) as client:
            response = await client.post(
                f"{self.base_url.rstrip('/')}{self.send_path}", json={"task_text": task_text}
            )
            response.raise_for_status()
            payload = _coerce_response_payload(response.json())
            task_id = payload.get("task_id") or payload.get("id")
            if not task_id:
                raise ValueError("runner response did not contain a task id")
            return str(task_id)

    async def get_status(self, task_id: str) -> str:
        client_module = _require_httpx()
        path = self.status_path_template.format(task_id=task_id)
        async with client_module.AsyncClient(
            timeout=self.timeout_s, headers=self.headers
        ) as client:
            response = await client.get(f"{self.base_url.rstrip('/')}{path}")
            response.raise_for_status()
            payload = _coerce_response_payload(response.json())
            return str(payload.get("status") or payload.get("outcome") or "unknown")


def _require_httpx() -> Any:
    if httpx is None:
        raise ImportError(
            "httpx is required for GenericHTTPRunner. Install with: pip install agent-xray[runner]"
        )
    return httpx


def _coerce_response_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return {str(key): value for key, value in payload.items()}
    raise ValueError("runner response body was not a JSON object")
