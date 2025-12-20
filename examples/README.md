# Aden Examples

Test matrix covering vendor SDKs and the PydanticAI agent framework.

## Test Matrix

| Example | OpenAI | Anthropic | Gemini | Streaming | Tools | Multi-step |
|---------|:------:|:---------:|:------:|:---------:|:-----:|:----------:|
| **Vendor SDKs** |
| [openai_basic.py](./openai_basic.py) | X | - | - | X | X | - |
| [anthropic_basic.py](./anthropic_basic.py) | - | X | - | X | X | - |
| [gemini_basic.py](./gemini_basic.py) | - | - | X | X | - | - |
| **Agent Frameworks** |
| [pydantic_ai_example.py](./pydantic_ai_example.py) | X | X | - | X | X | X |
| **Cost Control** |
| [control_actions.py](./control_actions.py) | X | - | - | - | - | - |
| [cost_control_local.py](./cost_control_local.py) | X | - | - | - | - | - |

## Quick Start

```bash
# Set API keys
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=...

# Run a specific example
python examples/openai_basic.py
python examples/pydantic_ai_example.py
python examples/cost_control_local.py
```

## Vendor SDK Examples

### OpenAI (`openai_basic.py`)
- Chat Completions API (streaming & non-streaming)
- Tool/function calls
- Metrics captured for all requests

### Anthropic (`anthropic_basic.py`)
- Messages API (streaming & non-streaming)
- Tool use
- Prompt caching

### Gemini (`gemini_basic.py`)
- generateContent (streaming & non-streaming)
- Chat sessions
- System instructions
- Different model tiers (flash-lite, flash, pro)

## Agent Framework Examples

### PydanticAI (`pydantic_ai_example.py`)
- Basic agents with system prompts
- Structured output with Pydantic models
- Tool/function calling
- Multi-provider agents (OpenAI + Anthropic)
- Streaming responses
- Multi-agent workflows
- Dependency injection

Since PydanticAI uses OpenAI/Anthropic SDKs internally, Aden automatically captures all LLM calls without any framework-specific integration.

## Cost Control Examples

### Control Actions (`control_actions.py`)
Demonstrates server-based control actions:
- `allow` - Request proceeds normally
- `block` - Request rejected (budget exceeded)
- `throttle` - Request delayed before proceeding
- `degrade` - Request uses a cheaper model
- `alert` - Request proceeds but triggers notification

**Requires:** Running control server with `ADEN_API_KEY` and `ADEN_API_URL`

### Local Cost Control (`cost_control_local.py`)
Demonstrates policy evaluation locally (no server needed):
- Budget limits per user/context
- Model blocking rules
- Rate limiting / throttling
- Automatic model degradation
- Live demo with actual OpenAI calls

## What Aden Captures

Each example demonstrates Aden capturing:

```
[a1b2c3d4] openai gpt-4o-mini (stream) 1234ms
  tokens: 150 in / 89 out
  tools: 1 calls (get_weather)
```

Metrics include:
- **trace_id**: Groups related calls
- **span_id**: Unique per LLM call
- **Tokens**: input, output, cached, reasoning
- **Latency**: Time to complete
- **Tools**: Tool calls made
- **Model**: Actual model used (may differ if degraded)

## Dependencies

### Vendor SDKs
```bash
pip install openai anthropic google-generativeai
```

### PydanticAI
```bash
pip install pydantic-ai
```

### Development (from source)
```bash
cd arp-ingress-exp-py
pip install -e ".[all,dev]"
```

## Viewing Metrics

Examples write to JSONL files. View with:

```bash
# Pretty print
cat ./openai-metrics.jsonl | python -m json.tool

# Count requests
wc -l ./openai-metrics.jsonl
```

Example output:
```json
{
  "trace_id": "abc123",
  "span_id": "def456",
  "provider": "openai",
  "model": "gpt-4o-mini",
  "latency_ms": 1234,
  "stream": true,
  "usage": {
    "input_tokens": 150,
    "output_tokens": 89,
    "total_tokens": 239
  },
  "tool_calls": [
    {"name": "get_weather", "arguments": {"location": "Tokyo"}}
  ]
}
```
