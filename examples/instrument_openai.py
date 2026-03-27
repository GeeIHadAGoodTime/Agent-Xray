"""Trace an OpenAI SDK agent loop and produce JSONL for agent-xray."""

from __future__ import annotations

import json
import time
from pathlib import Path

from openai import OpenAI

TRACE_DIR = Path("traces")
TRACE_DIR.mkdir(exist_ok=True)
trace_file = TRACE_DIR / "openai_run.jsonl"

client = OpenAI()
task_id = f"oai-{int(time.time())}"
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Look up a stock price.",
            "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}},
        },
    },
]

messages: list[dict] = [{"role": "user", "content": "What is the price of AAPL?"}]
step = 0

with trace_file.open("a") as f:
    while True:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools,
        )
        choice = resp.choices[0]
        if choice.finish_reason == "tool_calls":
            for tc in choice.message.tool_calls or []:
                step += 1
                result = '{"price": "189.52", "currency": "USD"}'  # simulated
                f.write(
                    json.dumps(
                        {
                            "task_id": task_id,
                            "step": step,
                            "tool_name": tc.function.name,
                            "tool_input": json.loads(tc.function.arguments),
                            "tool_result": result,
                            "model_name": resp.model,
                            "input_tokens": resp.usage.prompt_tokens if resp.usage else None,
                            "output_tokens": resp.usage.completion_tokens if resp.usage else None,
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        }
                    )
                    + "\n"
                )
                messages.append(choice.message)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        else:
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
