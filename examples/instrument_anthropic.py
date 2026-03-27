"""Trace an Anthropic SDK agent and produce JSONL for agent-xray."""

from __future__ import annotations

import json
import time
from pathlib import Path

import anthropic

TRACE_DIR = Path("traces")
TRACE_DIR.mkdir(exist_ok=True)
trace_file = TRACE_DIR / "anthropic_run.jsonl"

client = anthropic.Anthropic()
task_id = f"anth-{int(time.time())}"
tools = [
    {
        "name": "get_weather",
        "description": "Get weather for a city.",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
]

messages: list[dict] = [{"role": "user", "content": "What is the weather in Paris?"}]
step = 0

with trace_file.open("a") as f:
    while True:
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=256,
            tools=tools,
            messages=messages,
        )
        for block in resp.content:
            if block.type == "tool_use":
                step += 1
                result = '{"temp": "18C", "sky": "cloudy"}'  # simulated
                f.write(
                    json.dumps(
                        {
                            "task_id": task_id,
                            "step": step,
                            "tool_name": block.name,
                            "tool_input": block.input,
                            "tool_result": result,
                            "model_name": resp.model,
                            "input_tokens": resp.usage.input_tokens,
                            "output_tokens": resp.usage.output_tokens,
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        }
                    )
                    + "\n"
                )
                messages.append({"role": "assistant", "content": resp.content})
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": block.id, "content": result},
                        ],
                    }
                )
        if resp.stop_reason == "end_turn":
            f.write(
                json.dumps(
                    {
                        "event": "task_complete",
                        "task_id": task_id,
                        "status": "success",
                        "total_steps": step,
                    }
                )
                + "\n"
            )
            break

print(f"Trace written to {trace_file}")
print("Run:  agent-xray analyze ./traces")
