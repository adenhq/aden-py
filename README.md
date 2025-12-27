# Aden

**LLM Observability & Cost Control SDK (Python)**

Aden automatically tracks every LLM API call in your application—usage, latency, costs—and gives you real-time controls to prevent budget overruns. Works with OpenAI, Anthropic, and Google Gemini.

```python
import os
from aden import instrument, MeterOptions
from openai import OpenAI

# One line to start tracking everything
instrument(MeterOptions(api_key=os.environ["ADEN_API_KEY"]))

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
- [Local Testing (Console Emitter)](#local-testing-console-emitter)
- [Custom Metric Handlers](#custom-metric-handlers)
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

### Step 1: Set Your API Key

Add the following to your shell configuration file (`~/.bashrc` for bash or `~/.zshrc` for zsh):

```bash
export ADEN_API_KEY="your-api-key"
export ADEN_API_URL="https://your-aden-server.com"  # Optional: defaults to Aden cloud
```

Then reload your shell or run `source ~/.bashrc` (or `source ~/.zshrc`).

### Step 2: Add Instrumentation

Add this **once** at your application startup (before creating any LLM clients):

```python
import os
from aden import instrument, MeterOptions

instrument(MeterOptions(api_key=os.environ["ADEN_API_KEY"]))
```

### Step 3: Use Your SDK Normally

That's it! Every API call is now tracked and sent to the Aden server:

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
)
# Metrics automatically sent: model, tokens, latency, cost
```

### Step 4: Clean Up on Shutdown

```python
from aden import uninstrument

# In your shutdown handler
uninstrument()
```

---

## Local Testing (Console Emitter)

For local development and testing without a server connection, use the console emitter:

```python
from aden import instrument, MeterOptions, create_console_emitter

instrument(MeterOptions(
    emit_metric=create_console_emitter(pretty=True),
))

# Console output:
# + [a1b2c3d4] openai gpt-4o 1234ms
#   tokens: 12 in / 247 out
```

---

## Custom Metric Handlers

For advanced use cases, you can create custom metric handlers:

```python
import httpx

async def http_emitter(event):
    async with httpx.AsyncClient() as client:
        await client.post(
            "https://api.yourcompany.com/v1/metrics",
            json={
                "trace_id": event.trace_id,
                "model": event.model,
                "input_tokens": event.input_tokens,
                "output_tokens": event.output_tokens,
                "latency_ms": event.latency_ms,
                "error": event.error,
            },
            headers={"Authorization": f"Bearer {API_KEY}"},
        )

instrument(MeterOptions(emit_metric=http_emitter))
```

---

## Cost Control

Aden's cost control system lets you set budgets, throttle requests, and automatically downgrade to cheaper models—all in real-time. When you provide an API key, the SDK automatically connects to the control server and enforces budgets configured on the server.

### How It Works

1. **SDK connects to control server** via WebSocket on startup
2. **Server sends policy** with budgets, thresholds, and pricing
3. **SDK enforces locally** using cached policy (no per-request latency)
4. **Metrics sent to server** for spend tracking and visibility
5. **Control events reported** when actions are taken (block, throttle, degrade, alert)

### Control Actions

| Action | What It Does | Use Case |
|--------|--------------|----------|
| **allow** | Request proceeds normally | Default when within limits |
| **block** | Request is rejected with `RequestCancelledError` | Budget exhausted |
| **throttle** | Request is delayed before proceeding | Rate limiting |
| **degrade** | Request uses a cheaper model | Approaching budget limit |
| **alert** | Request proceeds, notification sent | Warning threshold reached |

### Basic Setup

```python
import asyncio
from aden import instrument_async, uninstrument_async, MeterOptions, RequestCancelledError

async def main():
    # Connects to control server and enables budget enforcement
    await instrument_async(MeterOptions(
        api_key="your-api-key",
        server_url="https://your-aden-server.com",  # Optional
        on_alert=lambda alert: print(f"[{alert.level}] {alert.message}"),
    ))

    # Use LLM SDKs normally - budgets enforced automatically
    from openai import AsyncOpenAI
    client = AsyncOpenAI()

    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello!"}],
        )
    except RequestCancelledError as e:
        print(f"Request blocked: {e}")  # Budget exceeded

    await uninstrument_async()

asyncio.run(main())
```

### Multi-Budget Validation

Aden supports multiple budget types that can be matched based on request metadata. When a request matches multiple budgets, **all matching budgets are validated** and the most restrictive decision is returned.

| Budget Type | Matches On | Use Case |
|-------------|------------|----------|
| **global** | All requests | Organization-wide spend limit |
| **agent** | `metadata.agent` | Per-agent budgets (e.g., "enterprise", "basic") |
| **tenant** | `metadata.tenant` | Multi-tenant applications |
| **customer** | `metadata.customer` | Per-customer spend limits |
| **feature** | `metadata.feature` | Feature-specific budgets |
| **tag** | `metadata.tag` | Custom groupings |

#### Passing Metadata for Budget Matching

Use `extra_body.metadata` to pass context that matches your budget configuration:

```python
response = await client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_body={
        "metadata": {
            "agent": "enterprise",      # Matches agent-type budget
            "tenant": "acme-corp",      # Matches tenant-type budget
            "customer": "customer-123", # Matches customer-type budget
        }
    },
)
# This request will be validated against:
# 1. Global budget (always)
# 2. Enterprise agent budget (if configured)
# 3. Acme-corp tenant budget (if configured)
# 4. Customer-123 budget (if configured)
# Most restrictive action wins (e.g., if any budget blocks, request is blocked)
```

### Budget Thresholds

Configure thresholds on the server to trigger actions at different usage levels:

```
0% ────────── 50% ─────────── 80% ─────────── 95% ─────────── 100%
   [ALLOW]      [ALERT]         [DEGRADE]       [THROTTLE]      [BLOCK]
```

Example server-side budget configuration:
```json
{
  "type": "global",
  "limit": 100.00,
  "thresholds": [
    {"percent": 50, "action": "alert"},
    {"percent": 80, "action": "degrade", "degradeTo": "gpt-4o-mini"},
    {"percent": 95, "action": "throttle", "delayMs": 2000}
  ],
  "limitAction": "block"
}
```

### Handling Budget Errors

```python
from aden import RequestCancelledError

try:
    response = await client.chat.completions.create(...)
except RequestCancelledError as e:
    # Budget exceeded - request was blocked before being sent
    print(f"Request blocked: {e}")
    # Handle gracefully (show user message, use fallback, etc.)
```

---

## Multi-Provider Support

Aden works with all major LLM providers. Instrumentation automatically detects available SDKs:

```python
import os
from aden import instrument, MeterOptions

# Instrument all available providers at once
result = instrument(MeterOptions(api_key=os.environ["ADEN_API_KEY"]))

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

### Console Emitter (Local Testing Only)

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
    # === Primary: Aden Server (recommended for production) ===
    api_key="aden_xxx",               # Your Aden API key (enables metrics + cost control)
    server_url="https://...",         # Control server URL (optional, uses default if not set)

    # === Alternative: Custom Emitter (local testing or custom backends) ===
    emit_metric=my_emitter,           # Custom handler for metrics

    # === Context Tracking ===
    get_context_id=lambda: get_user_id(),  # For per-user budgets
    request_metadata={"env": "prod"},      # Static metadata for all requests

    # === Callbacks ===
    on_alert=lambda alert: print(f"[{alert.level}] {alert.message}"),  # Alert handler
    before_request=my_budget_checker,  # Custom pre-request hook (runs after budget check)

    # === Reliability ===
    fail_open=True,                   # Allow requests if control server is unreachable
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

### Logging Configuration

Aden uses Python's logging module with two loggers:

| Logger | Purpose | Default Level |
|--------|---------|---------------|
| `aden` | Connection status, warnings, errors | INFO |
| `aden.metrics` | Detailed metric output | DEBUG |

**Environment Variable (Recommended):**

Set the `ADEN_LOG_LEVEL` environment variable to control log verbosity:

```bash
# Show only warnings and errors (production)
export ADEN_LOG_LEVEL=warning

# Show debug info (development)
export ADEN_LOG_LEVEL=debug
```

**Programmatic Configuration:**

Use `configure_logging()` to configure logging in code:

```python
from aden import configure_logging, instrument, MeterOptions, create_console_emitter

# Show all logs including debug info
configure_logging(level="debug")

# Or show only warnings and errors (quiet mode)
configure_logging(level="warning")

# Use logging-based console emitter (metrics at DEBUG level)
instrument(MeterOptions(
    emit_metric=create_console_emitter(use_logging=True),
))
```

**Log Level Reference:**

| Level | What You'll See |
|-------|-----------------|
| `debug` | Everything: flush events, metric details, connection state |
| `info` | Connection status, SDK instrumentation, control agent start/stop |
| `warning` | HTTP failures, connection issues, config problems |
| `error` | Critical errors only |

**Manual Configuration:**

If you prefer to configure logging yourself:

```python
import logging

# Configure aden logger
logging.getLogger("aden").setLevel(logging.DEBUG)

# Configure metrics logger separately
logging.getLogger("aden.metrics").setLevel(logging.DEBUG)
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

### Without API Key (Local Testing Only)

For local testing without a server connection, you can use `instrument()` from any context:

```python
from aden import instrument, uninstrument, MeterOptions, create_console_emitter

# Works from anywhere - no server connection needed
instrument(MeterOptions(
    emit_metric=create_console_emitter(pretty=True),
))

# Later...
uninstrument()
```

This is because without an API key, no async connections need to be established. For production use, always configure an API key to send metrics to the server.

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
| `get_control_agent()` | Get the current control agent instance (created automatically with api_key) |

### Types

```python
from aden import (
    # Metrics
    MetricEvent,
    MeterOptions,
    NormalizedUsage,
    ToolCallMetric,

    # Before request hooks
    BeforeRequestResult,
    BeforeRequestContext,

    # Alerts and errors
    AlertEvent,
    RequestCancelledError,
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
| `control_actions.py` | Budget control with server (multi-budget validation) |
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

3. **Verify instrumentation is working**: Test with console emitter for local debugging
   ```python
   from aden import create_console_emitter

   instrument(MeterOptions(
       emit_metric=create_console_emitter(pretty=True),
   ))
   ```

### Budget not enforcing

1. **Check you're using `instrument_async` with an API key**: Budget enforcement requires connecting to the control server
   ```python
   # Must use instrument_async for budget enforcement
   await instrument_async(MeterOptions(
       api_key="your-api-key",  # Required for budget enforcement
       server_url="https://your-server.com",
   ))
   ```

2. **Verify server policy is configured**: Check your control server has budgets configured

3. **Check metadata is being passed**: For multi-budget matching, ensure you're passing the right metadata
   ```python
   response = await client.chat.completions.create(
       model="gpt-4o",
       messages=[...],
       extra_body={"metadata": {"agent": "enterprise"}},  # For agent-type budgets
   )
   ```

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
