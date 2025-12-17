# OpenAI Meter (Python)

SDK metering for AI Cost ERP - usage tracking, budget enforcement, and cost control for OpenAI API calls.

Designed for integration with LiveKit voice agents and other OpenAI SDK consumers.

## Installation

```bash
pip install openai-meter

# With LiveKit support
pip install openai-meter[livekit]
```

## Quick Start

```python
from openai import OpenAI
from openai_meter import make_metered_openai, MeterOptions, create_console_emitter

# Create an OpenAI client
client = OpenAI()

# Wrap with metering
metered = make_metered_openai(client, MeterOptions(
    emit_metric=create_console_emitter(),
))

# Use normally - metrics are collected automatically
response = metered.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
)

# Output:
# + [a1b2c3d4] gpt-4o-mini 245ms
#   tokens: 12 in / 8 out
```

## Features

### Core Metering

- **Zero SDK Modification**: Monkey-patches the OpenAI client without modifying the SDK
- **Streaming Support**: Full support for streaming responses with accurate token counting
- **Async/Sync**: Works with both `OpenAI` and `AsyncOpenAI` clients
- **Usage Normalization**: Handles both Responses API and Chat Completions API shapes

### Budget Enforcement

```python
from openai_meter import MeterOptions, BeforeRequestResult

def budget_check(params, context):
    remaining_budget = get_remaining_budget(context.metadata["tenant_id"])
    if remaining_budget <= 0:
        return BeforeRequestResult.cancel("Budget exceeded")
    if remaining_budget < 10:
        return BeforeRequestResult.throttle(delay_ms=2000)
    return BeforeRequestResult.proceed()

metered = make_metered_openai(client, MeterOptions(
    emit_metric=my_emitter,
    before_request=budget_check,
    request_metadata={"tenant_id": "acme-corp"},
))
```

### Emitters

```python
from openai_meter import (
    create_console_emitter,      # Development/debugging
    create_batch_emitter,        # Batch for efficiency
    create_multi_emitter,        # Multiple destinations
    create_filtered_emitter,     # Filter events
    create_memory_emitter,       # Testing
    create_noop_emitter,         # Disable metrics
)

# Send to multiple destinations
emitter = create_multi_emitter([
    create_console_emitter(),
    create_batch_emitter(lambda events: send_to_backend(events)),
])
```

### Custom Emitter

```python
import httpx

async def http_emitter(event):
    async with httpx.AsyncClient() as client:
        await client.post(
            "https://api.yourerp.com/v1/ingest",
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
```

## LiveKit Integration

For voice agents using LiveKit with OpenAI:

```python
from livekit.agents import JobContext
from livekit.plugins import openai as lk_openai
from openai_meter.livekit import create_livekit_meter, create_console_emitter

# Create meter with session budget
meter = create_livekit_meter(
    emit_metric=create_console_emitter(),
    max_cost_per_session=0.50,  # 50 cents max per session
)

async def entrypoint(ctx: JobContext):
    # Start session tracking
    meter.start_session(ctx.room.name)

    # Get the underlying OpenAI client from LiveKit's LLM
    llm = lk_openai.LLM()

    # If you have direct access to the OpenAI client:
    # metered_client = meter.meter_openai_client(openai_client, session_id=ctx.room.name)

    # ... use the LLM in your agent

    # End session and get final metrics
    final_metrics = meter.end_session(ctx.room.name)
    print(f"Session cost: ${final_metrics.estimated_cost_usd:.4f}")
```

### Session Metrics

The LiveKit integration tracks per-session metrics:

- Total tokens (input, output, reasoning, cached)
- Request counts by type (LLM, TTS, STT)
- Latency breakdown
- Estimated cost
- Error count

## Types

### MetricEvent

```python
@dataclass
class MetricEvent:
    trace_id: str              # Unique trace ID
    model: str                 # Model used
    stream: bool               # Whether streaming
    request_id: str | None     # OpenAI request ID
    latency_ms: float          # Request latency
    usage: NormalizedUsage | None
    error: str | None          # Error message if failed
    tool_calls: list[ToolCallMetric] | None
    metadata: dict | None      # Custom metadata
```

### NormalizedUsage

```python
@dataclass
class NormalizedUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    reasoning_tokens: int      # For o1/o3 models
    cached_tokens: int
    accepted_prediction_tokens: int
    rejected_prediction_tokens: int
```

## Performance

The metering layer adds minimal overhead:

- **Metering overhead**: ~10-50μs per request
- **API latency**: ~500-10,000ms per request
- **Relative overhead**: 0.0001% - 0.01%

Performance options:

```python
MeterOptions(
    emit_metric=my_emitter,
    async_emit=True,      # Fire-and-forget (default)
    sample_rate=0.1,      # Only meter 10% of requests
)
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check .

# Type check
mypy src/openai_meter
```

## License

MIT
