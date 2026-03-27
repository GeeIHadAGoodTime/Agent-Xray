# agent-xray Tutorial

This tutorial takes about five minutes and uses one tiny JSONL trace so you can see the full `agent-xray` loop end to end.

## Step 1: Install `agent-xray`

```bash
pip install agent-xray
```

Expected output:

```text
Successfully installed agent-xray-0.1.0
```

Your exact version may differ.

## Step 2: Create a sample trace file

Create a directory named `traces`, then save the following content as `traces/demo.jsonl`:

```bash
mkdir -p traces
cat > traces/demo.jsonl <<'JSONL'
{"task_id":"demo-checkout","step":1,"tool_name":"browser_navigate","tool_input":{"url":"https://demo-shop.example.test"},"tool_result":"Homepage loaded.","timestamp":"2026-03-27T12:00:00Z","duration_ms":900,"user_text":"Buy the blue mug on demo-shop.example.test and stop once checkout is visible.","task_category":"commerce","model_name":"gpt-5-mini","temperature":0.0,"tool_choice":"auto","tools_available":["browser_navigate","browser_click","browser_fill_ref","browser_snapshot"],"message_count":1,"llm_reasoning":"Open the storefront first.","page_url":"https://demo-shop.example.test/"}
{"task_id":"demo-checkout","step":2,"tool_name":"browser_click","tool_input":{"ref":"product-blue-mug"},"tool_result":"Product page opened.","timestamp":"2026-03-27T12:00:04Z","duration_ms":420,"model_name":"gpt-5-mini","temperature":0.0,"tool_choice":"auto","tools_available":["browser_click","browser_fill_ref","browser_snapshot"],"message_count":2,"llm_reasoning":"Open the mug detail page.","page_url":"https://demo-shop.example.test/products/blue-mug"}
{"task_id":"demo-checkout","step":3,"tool_name":"browser_fill_ref","tool_input":{"ref":"shipping-form","fields":["email","zip"],"text":"alex@example.test 60601"},"tool_result":"Shipping details accepted.","timestamp":"2026-03-27T12:00:09Z","duration_ms":610,"model_name":"gpt-5-mini","temperature":0.0,"tool_choice":"auto","tools_available":["browser_click","browser_fill_ref","browser_snapshot"],"message_count":3,"llm_reasoning":"Provide the minimum details required to continue.","page_url":"https://demo-shop.example.test/checkout"}
{"task_id":"demo-checkout","step":4,"tool_name":"browser_snapshot","tool_input":{},"tool_result":"Checkout page is visible with the order summary.","timestamp":"2026-03-27T12:00:12Z","duration_ms":250,"model_name":"gpt-5-mini","temperature":0.0,"tool_choice":"auto","tools_available":["browser_click","browser_snapshot"],"message_count":4,"llm_reasoning":"Verify that checkout is now visible.","page_url":"https://demo-shop.example.test/checkout"}
{"event":"task_complete","task_id":"demo-checkout","status":"success","final_answer":"Checkout page is open for the blue mug.","total_steps":4,"total_duration_s":2.18,"timestamp":"2026-03-27T12:00:12Z"}
JSONL
```

Expected output:

```text
(no terminal output)
```

## Step 3: Analyze the trace directory

```bash
agent-xray analyze ./traces
```

Expected output:

```text
Analyzed 1 task(s) with rules=default
{'GOLDEN': 0, 'GOOD': 1, 'OK': 0, 'WEAK': 0, 'BROKEN': 0}
```

## Step 4: Inspect the decision surface

```bash
agent-xray surface demo-checkout --log-dir ./traces
```

Expected output:

```text
========================================================================
AGENT XRAY SURFACE: demo-checkout
========================================================================
user: Buy the blue mug on demo-shop.example.test and stop once checkout is visible.
------------------------------------------------------------------------
STEP 1 [2026-03-27T12:00:00Z]
model: model=gpt-5-mini temp=0.0 tool_choice=auto
tools: 4 available
context: 1 messages
surface: completeness=0.674 missing=15
  missing_surfaces: prompt_variant, rejected_tools, focused_set, conversation_turn_count, system_prompt_hash, context_usage_pct, context_window, compaction, correction_messages, intervention_signals, approval_path, screenshot_state, snapshot_compression, memory, rag
decision: browser_navigate {"url": "https://demo-shop.example.test"}
reasoning: Open the storefront first.
result: Homepage loaded.
------------------------------------------------------------------------
STEP 2 [2026-03-27T12:00:04Z]
model: model=gpt-5-mini temp=0.0 tool_choice=auto
tools: 3 available
context: 2 messages
surface: completeness=0.674 missing=15
  missing_surfaces: prompt_variant, rejected_tools, focused_set, conversation_turn_count, system_prompt_hash, context_usage_pct, context_window, compaction, correction_messages, intervention_signals, approval_path, screenshot_state, snapshot_compression, memory, rag
decision: browser_click {"ref": "product-blue-mug"}
reasoning: Open the mug detail page.
result: Product page opened.
------------------------------------------------------------------------
STEP 3 [2026-03-27T12:00:09Z]
model: model=gpt-5-mini temp=0.0 tool_choice=auto
tools: 3 available
context: 3 messages
surface: completeness=0.674 missing=15
  missing_surfaces: prompt_variant, rejected_tools, focused_set, conversation_turn_count, system_prompt_hash, context_usage_pct, context_window, compaction, correction_messages, intervention_signals, approval_path, screenshot_state, snapshot_compression, memory, rag
decision: browser_fill_ref {"fields": ["email", "zip"], "ref": "shipping-form", "text": "alex@example.test 60601"}
reasoning: Provide the minimum details required to continue.
result: Shipping details accepted.
------------------------------------------------------------------------
STEP 4 [2026-03-27T12:00:12Z]
model: model=gpt-5-mini temp=0.0 tool_choice=auto
tools: 2 available
context: 4 messages
surface: completeness=0.674 missing=15
  missing_surfaces: prompt_variant, rejected_tools, focused_set, conversation_turn_count, system_prompt_hash, context_usage_pct, context_window, compaction, correction_messages, intervention_signals, approval_path, screenshot_state, snapshot_compression, memory, rag
decision: browser_snapshot {}
reasoning: Verify that checkout is now visible.
result: Checkout page is visible with the order summary.
```

## Step 5: Grade the run

```bash
agent-xray grade ./traces
```

Expected output:

```text
GRADE SUMMARY
Tasks: 1
Rules: default

  GOLDEN: 0
  GOOD: 1
  OK: 0
  WEAK: 0
  BROKEN: 0
```

## Step 6: Interpret the output

This run grades as `GOOD` under the bundled `default` rules because it:

- uses three different tools
- makes a non-trivial four-step attempt
- records zero errors

The surface output shows the exact decision context at each step:

- which tools were available
- what the model said it was trying to do
- what page URL it was on
- how much optional instrumentation is still missing from the trace

The `missing_surfaces` line is useful when a trace feels thin. In this example, the run is still debuggable, but it does not include prompt variants, compaction metadata, correction messages, or memory/RAG context yet.
