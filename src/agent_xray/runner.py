from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx


class TaskRunner(Protocol):
    async def send(self, task_text: str) -> str: ...

    async def get_status(self, task_id: str) -> str: ...


@dataclass(slots=True)
class GenericHTTPRunner:
    base_url: str
    send_path: str = "/tasks"
    status_path_template: str = "/tasks/{task_id}"
    timeout_s: float = 30.0
    headers: dict[str, str] = field(default_factory=dict)

    async def send(self, task_text: str) -> str:
        async with httpx.AsyncClient(timeout=self.timeout_s, headers=self.headers) as client:
            response = await client.post(
                f"{self.base_url.rstrip('/')}{self.send_path}", json={"task_text": task_text}
            )
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            task_id = payload.get("task_id") or payload.get("id")
            if not task_id:
                raise ValueError("runner response did not contain a task id")
            return str(task_id)

    async def get_status(self, task_id: str) -> str:
        path = self.status_path_template.format(task_id=task_id)
        async with httpx.AsyncClient(timeout=self.timeout_s, headers=self.headers) as client:
            response = await client.get(f"{self.base_url.rstrip('/')}{path}")
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            return str(payload.get("status") or payload.get("outcome") or "unknown")
