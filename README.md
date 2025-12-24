# Aden

**LLM Observability & Cost Control SDK (Python)**

Aden automatically tracks every LLM API call in your application—usage, latency, costs—and gives you real-time controls to prevent budget overruns. Works with OpenAI, Anthropic, and Google Gemini.

```python
from aden import instrument, MeterOptions, create_console_emitter
from openai import OpenAI

# One line to start tracking everything
instrument(MeterOptions(emit_metric=create_console_emitter()))

# Use your SDK normally - metrics collected automatically
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

---

## Table of Contents

- [Why Aden?](#why-aden)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Sending Metrics to Your Backend](#sending-metrics-to-your-backend)
- [Cost Control](#cost-control)
- [Multi-Provider Support](#multi-provider-support)
- [What Metrics Are Collected?](#what-metrics-are-collected)
- [Metric Emitters](#metric-emitters)
- [Advanced Configuration](#advanced-configuration)
- [Sync vs Async Context](#sync-vs-async-context)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

---

## Why Aden?

Building with LLMs is expensive and unpredictable:

- **No visibility**: You don't know which features or users consume the most tokens
- **Runaway costs**: One bug or bad prompt can blow through your budget in minutes
- **No control**: Once a request is sent, you can't stop it

Aden solves these problems:

| Problem | Aden Solution |
|---------|---------------|
| No visibility into LLM usage | Automatic metric collection for every API call |
| Unpredictable costs | Real-time budget tracking and enforcement |
| No per-user limits | Context-based controls (per user, per feature, per tenant) |
| Expensive models used unnecessarily | Automatic model degradation when approaching limits |

---

## Installation

```bash
pip install aden
```

Install with specific provider support:

```bash
# Individual providers
pip install aden[openai]      # OpenAI/GPT models
pip install aden[anthropic]   # Anthropic/Claude models
pip install aden[gemini]      # Google Gemini models

# All providers
pip install aden[all]

# Framework support
pip install aden[pydantic-ai]  # PydanticAI integration
pip install aden[livekit]      # LiveKit voice agents
```

---

## Quick Start

### Step 1: Add Instrumentation

Add this **once** at your application startup (before creating any LLM clients):

```python
from aden import instrument, MeterOptions, create_console_emitter

instrument(MeterOptions(
    emit_metric=create_console_emitter(pretty=True),
))
```

### Step 2: Use Your SDK Normally

That's it! Every API call is now tracked:

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
)

# Console output:
# + [a1b2c3d4] openai gpt-4o 1234ms
#   tokens: 12 in / 247 out
```

### Step 3: Clean Up on Shutdown

```python
from aden import uninstrument

# In your shutdown handler
uninstrument()
```

---

## Sending Metrics to Your Backend

For production, send metrics to your backend instead of the console:

### Option A: Custom Handler

```python
import httpx

async def http_emitter(event):
    async with httpx.AsyncClient() as client:
        await client.post(
            "https://api.yourcompany.com/v1/metrics",
            json={
                "trace_id": event.trace_id,
                "model": event.model,
                "input_tokens": event.usage.input_tokens if event.usage else 0,
                "output_tokens": event.usage.output_tokens if event.usage else 0,
                "latency_ms": event.latency_ms,
                "error": event.error,
            },
            headers={"Authorization": f"Bearer {API_KEY}"},
        )

instrument(MeterOptions(emit_metric=http_emitter))
```

### Option B: Aden Control Server

For real-time cost control (budgets, throttling, model degradation), connect to an Aden control server:

```python
import os
from aden import instrument, MeterOptions

instrument(MeterOptions(
    api_key=os.environ["ADEN_API_KEY"],
    server_url=os.environ.get("ADEN_API_URL"),
))
```

This enables all the [Cost Control](#cost-control) features described below.

---

## Cost Control

Aden's cost control system lets you set budgets, throttle requests, and automatically downgrade to cheaper models—all in real-time.

### Control Actions

The control server can apply these actions to requests:

| Action | What It Does | Use Case |
|--------|--------------|----------|
| **allow** | Request proceeds normally | Default when within limits |
| **block** | Request is rejected with an error | Budget exhausted |
| **throttle** | Request is delayed before proceeding | Rate limiting |
| **degrade** | Request uses a cheaper model | Approaching budget limit |
| **alert** | Request proceeds, notification sent | Warning threshold reached |

### Local Cost Control (No Server)

For local development or testing, see the `cost_control_local.py` example which demonstrates implementing a policy engine locally. This pattern is useful for:

- Understanding how cost control decisions work
- Testing policy configurations before deploying a server
- Simple use cases that don't need a full control server

```python
# See examples/cost_control_local.py for a complete example
# that implements budget limits, throttling, and model degradation
# without requiring a control server.
```

### Control Server

For production cost control, connect to an Aden control server:

```python
from aden import instrument, MeterOptions, create_control_agent, ControlAgentOptions

agent = create_control_agent(ControlAgentOptions(
    server_url="https://your-control-server.com",
    api_key="your-api-key",
    on_alert=lambda alert: print(f"[{alert.level}] {alert.message}"),
))

instrument(MeterOptions(
    control_agent=agent,
))
```

---

## Multi-Provider Support

Aden works with all major LLM providers. Instrumentation automatically detects available SDKs:

```python
from aden import instrument, MeterOptions, create_console_emitter

# Instrument all available providers at once
result = instrument(MeterOptions(
    emit_metric=create_console_emitter(pretty=True),
))

print(f"OpenAI: {result.openai}")
print(f"Anthropic: {result.anthropic}")
print(f"Gemini: {result.gemini}")
```

### OpenAI

```python
from openai import OpenAI

client = OpenAI()

# Chat completions
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
)

# Streaming
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
)

for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
# Metrics emitted when stream completes
```

### Anthropic

```python
from anthropic import Anthropic

client = Anthropic()

response = client.messages.create(
    model="claude-3-5-sonnet-latest",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
)
```

### Google Gemini

```python
import google.generativeai as genai

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

model = genai.GenerativeModel("gemini-2.0-flash")
response = model.generate_content("Explain quantum computing")
```

---

## What Metrics Are Collected?

Every LLM API call generates a `MetricEvent`:

```python
@dataclass
class MetricEvent:
    # Identity
    trace_id: str           # Unique ID for this request
    span_id: str            # Span ID (OTel compatible)
    request_id: str | None  # Provider's request ID

    # Request details
    provider: str           # "openai", "anthropic", "gemini"
    model: str              # e.g., "gpt-4o", "claude-3-5-sonnet"
    stream: bool
    timestamp: str          # ISO timestamp

    # Performance
    latency_ms: float
    error: str | None

    # Token usage
    usage: NormalizedUsage | None
    # - input_tokens: int
    # - output_tokens: int
    # - total_tokens: int
    # - reasoning_tokens: int   # For o1/o3 models
    # - cached_tokens: int      # Prompt cache hits

    # Tool usage
    tool_calls: list[ToolCallMetric] | None

    # Custom metadata
    metadata: dict | None
```

---

## Metric Emitters

Emitters determine where metrics go. You can use built-in emitters or create custom ones.

### Built-in Emitters

```python
from aden import (
    create_console_emitter,     # Log to console (development)
    create_batch_emitter,       # Batch before sending
    create_multi_emitter,       # Send to multiple destinations
    create_filtered_emitter,    # Filter events
    create_transform_emitter,   # Transform events
    create_file_emitter,        # Write to JSON files
    create_memory_emitter,      # Store in memory (testing)
    create_noop_emitter,        # Discard all events
)
```

### Console Emitter (Development)

```python
instrument(MeterOptions(
    emit_metric=create_console_emitter(pretty=True),
))

# Output:
# + [a1b2c3d4] openai gpt-4o 1234ms
#   tokens: 12 in / 247 out
```

### Multiple Destinations

```python
instrument(MeterOptions(
    emit_metric=create_multi_emitter([
        create_console_emitter(pretty=True),  # Log locally
        my_backend_emitter,                    # Send to backend
    ]),
))
```

### Filtering Events

```python
instrument(MeterOptions(
    emit_metric=create_filtered_emitter(
        my_emitter,
        lambda event: event.usage and event.usage.total_tokens > 100  # Only large requests
    ),
))
```

### File Logging

```python
from aden import create_file_emitter

instrument(MeterOptions(
    emit_metric=create_file_emitter(log_dir="./logs"),
))
# Creates: ./logs/metrics-2024-01-15.jsonl
```

### Custom Emitter

```python
def my_emitter(event):
    # Store in your database
    db.llm_metrics.insert({
        "trace_id": event.trace_id,
        "model": event.model,
        "tokens": event.usage.total_tokens if event.usage else 0,
        "latency_ms": event.latency_ms,
    })

    # Check for anomalies
    if event.latency_ms > 30000:
        alert_ops(f"Slow LLM call: {event.latency_ms}ms")

instrument(MeterOptions(emit_metric=my_emitter))
```

---

## Advanced Configuration

### Full Options Reference

```python
instrument(MeterOptions(
    # === Metrics Destination ===
    emit_metric=my_emitter,           # Required unless api_key is set

    # === Control Server (enables cost control) ===
    api_key="aden_xxx",               # Your Aden API key
    server_url="https://...",         # Control server URL (optional)

    # === Context Tracking ===
    get_context_id=lambda: get_user_id(),  # For per-user budgets
    request_metadata={"env": "prod"},      # Custom metadata

    # === Pre-request Hook ===
    before_request=my_budget_checker,

    # === Local Control Agent ===
    control_agent=my_control_agent,
))
```

### beforeRequest Hook

Implement custom rate limiting or request modification:

```python
from aden import BeforeRequestResult

def budget_check(params, context):
    # Check your own rate limits
    if not check_rate_limit(context.metadata.get("user_id")):
        return BeforeRequestResult.cancel("Rate limit exceeded")

    # Optionally delay the request
    if should_throttle():
        return BeforeRequestResult.throttle(delay_ms=1000)

    # Optionally switch to a cheaper model
    if should_degrade():
        return BeforeRequestResult.degrade(
            to_model="gpt-4o-mini",
            reason="High load"
        )

    return BeforeRequestResult.proceed()

instrument(MeterOptions(
    emit_metric=my_emitter,
    before_request=budget_check,
    request_metadata={"user_id": get_current_user_id()},
))
```

### Legacy Per-Instance Wrapping

For backward compatibility, you can still wrap individual clients:

```python
from aden import make_metered_openai, MeterOptions
from openai import OpenAI

client = OpenAI()
metered = make_metered_openai(client, MeterOptions(
    emit_metric=my_emitter,
))
```

---

## Sync vs Async Context

**Understanding when to use sync vs async instrumentation is critical for proper operation**, especially when using the control server for budget enforcement.

### The Problem

Python applications can run in two contexts:

- **Sync context**: Regular Python code without an event loop (e.g., scripts, CLI tools, Flask)
- **Async context**: Code running inside an async event loop (e.g., `asyncio.run()`, FastAPI, async frameworks)

Aden's control agent uses WebSocket/HTTP connections that are inherently async. When you call `instrument()` with an API key from a sync context, it needs to establish these connections. When called from an async context, different handling is required.

### Quick Reference

| Your Context | Instrumentation | Uninstrumentation |
|--------------|-----------------|-------------------|
| Sync (no event loop) | `instrument()` | `uninstrument()` |
| Async (inside event loop) | `await instrument_async()` | `await uninstrument_async()` |

### Sync Context (Scripts, CLI, Flask)

Use `instrument()` and `uninstrument()` when you're **not** inside an async event loop:

```python
from aden import instrument, uninstrument, MeterOptions

# Works correctly - no event loop running
instrument(MeterOptions(
    api_key="your-api-key",
    emit_metric=my_emitter,
))

# Use LLM SDKs normally
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(...)

# Clean up
uninstrument()
```

**How it works**: When an API key is provided, `instrument()` internally creates an event loop to connect to the control server. Metrics are queued and sent via a background thread.

### Async Context (FastAPI, asyncio.run, PydanticAI)

Use `instrument_async()` and `uninstrument_async()` when you're **inside** an async event loop:

```python
import asyncio
from aden import instrument_async, uninstrument_async, MeterOptions

async def main():
    # Must use async version inside event loop
    result = await instrument_async(MeterOptions(
        api_key="your-api-key",
        emit_metric=my_emitter,
    ))

    # Use LLM SDKs normally
    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    response = await client.chat.completions.create(...)

    # Clean up with async version
    await uninstrument_async()

asyncio.run(main())
```

**How it works**: The async version uses the existing event loop for all operations, avoiding the need for background threads.

### Common Mistakes

#### ❌ Wrong: Using sync instrument inside asyncio.run()

```python
import asyncio
from aden import instrument, MeterOptions

async def main():
    # WRONG: This is inside an event loop!
    instrument(MeterOptions(api_key="..."))  # Will log a warning
    # Control agent won't be created properly

asyncio.run(main())
```

You'll see this warning:
```
[aden] API key provided but called from async context. Use instrument_async() for control agent support.
```

#### ✅ Correct: Using async instrument inside asyncio.run()

```python
import asyncio
from aden import instrument_async, uninstrument_async, MeterOptions

async def main():
    # CORRECT: Using async version
    await instrument_async(MeterOptions(api_key="..."))
    # ... your code ...
    await uninstrument_async()

asyncio.run(main())
```

### Framework-Specific Examples

#### FastAPI

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from aden import instrument_async, uninstrument_async, MeterOptions

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: inside async context
    await instrument_async(MeterOptions(api_key="..."))
    yield
    # Shutdown
    await uninstrument_async()

app = FastAPI(lifespan=lifespan)
```

#### Flask

```python
from flask import Flask
from aden import instrument, uninstrument, MeterOptions

app = Flask(__name__)

# Sync context at module level
instrument(MeterOptions(api_key="..."))

@app.teardown_appcontext
def cleanup(exception):
    uninstrument()
```

#### PydanticAI

PydanticAI uses async internally, so always use async instrumentation:

```python
import asyncio
from pydantic_ai import Agent
from aden import instrument_async, uninstrument_async, MeterOptions

async def main():
    await instrument_async(MeterOptions(api_key="..."))

    agent = Agent("openai:gpt-4o-mini")
    result = await agent.run("Hello!")

    await uninstrument_async()

asyncio.run(main())
```

#### LangGraph / LangChain

LangGraph can run in both sync and async modes. Match your instrumentation:

```python
# Sync LangGraph
from aden import instrument, uninstrument, MeterOptions

instrument(MeterOptions(api_key="..."))
# graph.invoke(...)  # Sync
uninstrument()

# Async LangGraph
import asyncio
from aden import instrument_async, uninstrument_async, MeterOptions

async def main():
    await instrument_async(MeterOptions(api_key="..."))
    # await graph.ainvoke(...)  # Async
    await uninstrument_async()

asyncio.run(main())
```

### Without API Key (Local Only)

If you're **not** using the control server (no `api_key`), you can use `instrument()` from any context:

```python
from aden import instrument, uninstrument, MeterOptions, create_console_emitter

# Works from anywhere - no server connection needed
instrument(MeterOptions(
    emit_metric=create_console_emitter(pretty=True),
))

# Later...
uninstrument()
```

This is because without an API key, no async connections need to be established.

### How Metrics Are Sent

| Context | With API Key | Without API Key |
|---------|--------------|-----------------|
| Sync | Background thread flushes every 1s | Emitter called synchronously |
| Async | Event loop sends immediately | Emitter called (sync or async) |

The sync context uses a background thread that:
- Queues metrics as they're generated
- Flushes to the server every 1 second
- Also flushes when batch size (10 events) is reached
- Performs a final flush on `uninstrument()`

---

## API Reference

### Core Functions

| Function | Description |
|----------|-------------|
| `instrument(options)` | Instrument all SDKs (sync context) |
| `instrument_async(options)` | Instrument all SDKs (async context) |
| `uninstrument()` | Remove instrumentation (sync context) |
| `uninstrument_async()` | Remove instrumentation (async context) |
| `is_instrumented()` | Check if instrumented |
| `get_instrumented_sdks()` | Get which SDKs are instrumented |

### Provider-Specific Functions

| Function | Description |
|----------|-------------|
| `instrument_openai(options)` | Instrument OpenAI only |
| `instrument_anthropic(options)` | Instrument Anthropic only |
| `instrument_gemini(options)` | Instrument Gemini only |
| `uninstrument_openai()` | Remove OpenAI instrumentation |
| `uninstrument_anthropic()` | Remove Anthropic instrumentation |
| `uninstrument_gemini()` | Remove Gemini instrumentation |

### Emitter Factories

| Function | Description |
|----------|-------------|
| `create_console_emitter(pretty=False)` | Log to console |
| `create_batch_emitter(handler, batch_size, flush_interval)` | Batch events |
| `create_multi_emitter(emitters)` | Multiple destinations |
| `create_filtered_emitter(emitter, filter_fn)` | Filter events |
| `create_transform_emitter(emitter, transform_fn)` | Transform events |
| `create_file_emitter(log_dir)` | Write to JSON files |
| `create_memory_emitter()` | Store in memory |
| `create_noop_emitter()` | Discard events |

### Control Agent

| Function | Description |
|----------|-------------|
| `create_control_agent(options)` | Create local control agent |
| `create_control_agent_emitter(agent)` | Create emitter from agent |

### Types

```python
from aden import (
    MetricEvent,
    MeterOptions,
    NormalizedUsage,
    ToolCallMetric,
    BeforeRequestResult,
    BeforeRequestContext,
    ControlPolicy,
    ControlDecision,
    AlertEvent,
    RequestCancelledError,
    BudgetExceededError,
)
```

---

## Examples

Run examples with `python examples/<name>.py`:

| Example | Description |
|---------|-------------|
| `openai_basic.py` | Basic OpenAI instrumentation |
| `anthropic_basic.py` | Basic Anthropic instrumentation |
| `gemini_basic.py` | Basic Gemini instrumentation |
| `cost_control_local.py` | Cost control without a server |
| `pydantic_ai_example.py` | PydanticAI framework integration |

---

## Troubleshooting

### Metrics not appearing

1. **Check instrumentation order**: Call `instrument()` before creating SDK clients
   ```python
   # Correct
   instrument(MeterOptions(...))
   client = OpenAI()

   # Wrong - client created before instrumentation
   client = OpenAI()
   instrument(MeterOptions(...))
   ```

2. **Check SDK is installed**: Aden only instruments SDKs that are importable
   ```bash
   pip install openai anthropic google-generativeai
   ```

3. **Verify emitter is working**: Test with console emitter first
   ```python
   instrument(MeterOptions(
       emit_metric=create_console_emitter(pretty=True),
   ))
   ```

### Budget not enforcing

1. **Check control agent is connected**: Budget enforcement requires a control agent connected to your server
   ```python
   agent = create_control_agent(ControlAgentOptions(
       server_url="https://your-server.com",
       api_key="your-api-key",
   ))
   instrument(MeterOptions(
       control_agent=agent,  # Required!
   ))
   ```

2. **Verify server policy is configured**: Check your control server has the budget configured for your context

### Streaming not tracked

1. **Consume the stream**: Metrics are emitted when the stream completes
   ```python
   stream = client.chat.completions.create(..., stream=True)
   for chunk in stream:  # Must iterate through stream
       print(chunk.choices[0].delta.content or "", end="")
   # Metrics emitted here
   ```

### Control agent not working / Metrics not sent to server

1. **Check you're using the right function for your context**:
   ```python
   # If you see this warning:
   # [aden] API key provided but called from async context. Use instrument_async()

   # You're calling instrument() from inside asyncio.run() or similar
   # Solution: use instrument_async() instead

   async def main():
       await instrument_async(MeterOptions(api_key="..."))  # Correct
       # ...
       await uninstrument_async()

   asyncio.run(main())
   ```

2. **Ensure uninstrument is called**: The sync context uses a background thread that flushes on shutdown
   ```python
   # Always call uninstrument() to flush remaining metrics
   uninstrument()  # or await uninstrument_async()
   ```

3. **Check aiohttp is installed**: The control agent requires aiohttp
   ```bash
   pip install aiohttp
   ```

### Async coroutine warnings

If you see warnings like:
```
RuntimeWarning: coroutine '...' was never awaited
```

This usually means you're mixing sync and async contexts incorrectly. See [Sync vs Async Context](#sync-vs-async-context) for the correct patterns.

---

## License

MIT
