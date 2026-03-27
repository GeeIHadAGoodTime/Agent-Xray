# Integration Guide

This guide covers how to connect your agent to `agent-xray` so every tool call, model decision, and browser action is captured as a trace you can analyze, grade, and replay.

## Architecture

```text
Your Agent (Anthropic / OpenAI / LangChain / MCP / custom)
    |
    v
Instrumentor (writes one JSON line per tool call)
    |
    v
traces/*.jsonl (agent-xray native format)
    |
    v
agent-xray analyze / grade / surface / flywheel
```

The instrumentor is a thin logging layer. It does not modify your agent's behavior. It writes one JSONL line per tool call and one `task_complete` event per finished task.

## Getting Your First Trace in 5 Minutes

### Anthropic SDK

```python
import json, time
from pathlib import Path
import anthropic

client = anthropic.Anthropic()
trace = Path("traces/run.jsonl")
trace.parent.mkdir(exist_ok=True)
step = 0

# After each tool_use block in the response:
step += 1
with trace.open("a") as f:
    f.write(json.dumps({
        "task_id": "my-task",
        "step": step,
        "tool_name": block.name,
        "tool_input": block.input,
        "tool_result": result_text,
        "model_name": resp.model,
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
    }) + "\n")
```

Full working example: [`examples/instrument_anthropic.py`](../examples/instrument_anthropic.py)

### OpenAI SDK

```python
# After each tool_call in choice.message.tool_calls:
step += 1
with trace.open("a") as f:
    f.write(json.dumps({
        "task_id": "my-task",
        "step": step,
        "tool_name": tc.function.name,
        "tool_input": json.loads(tc.function.arguments),
        "tool_result": result_text,
        "model_name": resp.model,
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
    }) + "\n")
```

Full working example: [`examples/instrument_openai.py`](../examples/instrument_openai.py)

### MCP (Playwright, browser tools, etc.)

Wrap your MCP client so `call_tool` logs every invocation:

```python
class TracedMCPClient:
    def call_tool(self, name, arguments):
        self._step += 1
        start = time.monotonic()
        result = self._inner.call_tool(name, arguments)
        self._trace.write(json.dumps({
            "task_id": self._task_id, "step": self._step,
            "tool_name": name, "tool_input": arguments,
            "tool_result": str(result.content[0].text),
            "duration_ms": int((time.monotonic() - start) * 1000),
        }) + "\n")
        return result
```

Full working example: [`examples/instrument_mcp.py`](../examples/instrument_mcp.py)

### LangChain

Use a callback handler that writes JSONL on `on_tool_end`:

```python
from langchain_core.callbacks import BaseCallbackHandler

class AgentXrayCallback(BaseCallbackHandler):
    def on_tool_end(self, output, *, run_id, **kwargs):
        self.step += 1
        self.trace.write(json.dumps({
            "task_id": self.task_id, "step": self.step,
            "tool_name": kwargs.get("name", "unknown"),
            "tool_result": output[:500],
        }) + "\n")

# Pass to your agent:
executor.invoke({"input": "..."}, config={"callbacks": [cb]})
```

Full working example: [`examples/instrument_langchain.py`](../examples/instrument_langchain.py)

### Manual JSONL (no SDK)

If you have a custom agent loop, write JSONL directly. This works with any language or framework:

```bash
mkdir -p traces
cat > traces/manual.jsonl <<'JSONL'
{"task_id":"my-task","step":1,"tool_name":"web_search","tool_input":{"query":"test"},"tool_result":"Found results."}
{"task_id":"my-task","step":2,"tool_name":"summarize","tool_input":{},"tool_result":"Summary ready."}
{"event":"task_complete","task_id":"my-task","status":"success","total_steps":2}
JSONL

agent-xray analyze ./traces
```

## JSONL Format Reference

Each line is one JSON object. There are two kinds of records.

### Step records (one per tool call)

| Field | Required | Type | Description |
| --- | --- | --- | --- |
| `task_id` | Yes | string | Groups related steps into one task |
| `step` | Yes | int | 1-based step index within the task |
| `tool_name` | Yes | string | Name of the tool called |
| `tool_input` | Yes | object | Arguments passed to the tool |
| `tool_result` | No | string | Tool output text |
| `error` | No | string | Error message if the step failed |
| `duration_ms` | No | int | Step execution time in milliseconds |
| `timestamp` | No | string | ISO-8601 timestamp |
| `model_name` | No | string | Model used for the decision |
| `input_tokens` | No | int | Prompt tokens consumed |
| `output_tokens` | No | int | Completion tokens generated |
| `cost_usd` | No | float | Estimated step cost in USD |
| `tools_available` | No | list[string] | Tools visible to the model |
| `llm_reasoning` | No | string | Model's reasoning or scratchpad |
| `page_url` | No | string | Browser URL at the time of the step |
| `temperature` | No | float | Sampling temperature |
| `tool_choice` | No | string | Tool selection mode |

### Task completion records

| Field | Required | Type | Description |
| --- | --- | --- | --- |
| `event` | Yes | `"task_complete"` | Marks end of task |
| `task_id` | Yes | string | Must match the step records |
| `status` | Yes | string | `"success"` or `"failed"` |
| `total_steps` | No | int | Step count for the task |
| `final_answer` | No | string | Agent's final response |

### What fields matter most?

The four required fields (`task_id`, `step`, `tool_name`, `tool_input`) are enough to run `agent-xray analyze`. Everything else improves the quality of grading, surface replay, and root-cause classification:

- **`tool_result` + `error`**: Enable error-rate and error-kind analysis.
- **`model_name` + `input_tokens` + `output_tokens`**: Enable cost tracking and model comparison.
- **`tools_available`**: Enable tool-diversity scoring and routing-bug detection.
- **`llm_reasoning`**: Makes `agent-xray surface` and `agent-xray reasoning` useful.
- **`page_url`**: Enables site inference, URL progression tracking, and browser-flow signals.
- **`duration_ms`**: Enables latency analysis.

Start with the required four. Add more fields as you need deeper analysis.

## File Organization

Put all JSONL files in a single directory. `agent-xray` scans `*.jsonl` files recursively:

```text
traces/
  session_20260327.jsonl      # one file per session, day, or run
  checkout_flow.jsonl
  research_task.jsonl
```

Multiple tasks can live in one file. Tasks are grouped by `task_id`, not by file.

## Common Pitfalls

**Forgetting `task_id`**: Every line needs a `task_id`. Without it, steps cannot be grouped and will be silently skipped.

**Step numbering gaps**: Steps should be sequential starting at 1. Gaps work but may cause surface completeness warnings.

**Appending to a stale file**: If you append new runs to an old file, previous tasks are still loaded. Use timestamped filenames or clear the directory between runs.

**Missing `task_complete` event**: Analysis works without it, but grading and root-cause classification are more accurate when the terminal event is present.

**Large `tool_result` values**: Truncate screenshots, HTML dumps, or large payloads before writing. The analyzer only needs the first few hundred characters. Use `result[:500]` or similar.

## Analyzing the Output

Once you have a `traces/` directory with JSONL files:

```bash
# Summary and grade distribution
agent-xray analyze ./traces

# Decision surface for one task
agent-xray surface my-task --log-dir ./traces

# Grade with a specific ruleset
agent-xray grade ./traces --rules browser_flow

# Full quality loop
agent-xray flywheel ./traces --baseline ./baseline.json

# Interactive inspector
agent-xray tui ./traces
```

## Library Usage

You can also analyze traces programmatically:

```python
from agent_xray import AgentStep, AgentTask, grade_task, load_rules, surface_for_task

steps = [AgentStep.from_dict(record) for record in my_records]
task = AgentTask.from_steps(steps, task_text="Check the checkout flow")

grade = grade_task(task, load_rules("browser_flow"))
print(grade.grade, grade.score)

surface = surface_for_task(task)
print(surface["steps"][0]["tools_available_names"])
```
