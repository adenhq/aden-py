"""
Content Capture Example (Layer 0 & Layer 6)

Demonstrates deep telemetry features:
- Layer 0: Raw content capture (system prompts, messages, responses)
- Layer 6: Tool call deep inspection with schema validation

Run: python examples/content_capture.py
"""

import json
import os
import sys

# Add parent directory to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from aden import (
    instrument,
    uninstrument,
    MeterOptions,
    MetricEvent,
    ContentCaptureOptions,
)


def create_capture_emitter():
    """Create an emitter that displays captured content."""

    def emit(event: MetricEvent) -> None:
        print("\n" + "=" * 60)
        print(f"METRIC EVENT: {event.provider} / {event.model}")
        print(f"  Trace: {event.trace_id[:8]}... | Latency: {event.latency_ms:.0f}ms")
        print(f"  Tokens: {event.input_tokens} in / {event.output_tokens} out")

        # Layer 0: Content Capture
        if event.content_capture:
            cc = event.content_capture
            print("\n  [Layer 0: Content Capture]")

            if cc.system_prompt:
                prompt = cc.system_prompt if isinstance(cc.system_prompt, str) else f"<ref:{cc.system_prompt.content_id}>"
                print(f"    System Prompt: {prompt[:80]}{'...' if len(str(prompt)) > 80 else ''}")

            if cc.messages:
                if isinstance(cc.messages, list):
                    print(f"    Messages: {len(cc.messages)} messages")
                    for msg in cc.messages[:3]:  # Show first 3
                        content = msg.content if isinstance(msg.content, str) else "<large>"
                        preview = content[:50] if content else ""
                        print(f"      [{msg.role}]: {preview}{'...' if content and len(content) > 50 else ''}")
                else:
                    print(f"    Messages: <ref:{cc.messages.content_id}>")

            if cc.tools:
                if isinstance(cc.tools, list):
                    print(f"    Tools: {[t.name for t in cc.tools]}")
                else:
                    print(f"    Tools: <ref:{cc.tools.content_id}>")

            if cc.params:
                params = []
                if cc.params.temperature is not None:
                    params.append(f"temp={cc.params.temperature}")
                if cc.params.max_tokens is not None:
                    params.append(f"max_tokens={cc.params.max_tokens}")
                if params:
                    print(f"    Params: {', '.join(params)}")

            if cc.response_content:
                response = cc.response_content if isinstance(cc.response_content, str) else f"<ref:{cc.response_content.content_id}>"
                print(f"    Response: {response[:80]}{'...' if len(str(response)) > 80 else ''}")

            if cc.finish_reason:
                print(f"    Finish Reason: {cc.finish_reason}")

        # Layer 6: Tool Call Deep Inspection
        if event.tool_calls_captured:
            print("\n  [Layer 6: Tool Call Deep Inspection]")
            for tc in event.tool_calls_captured:
                print(f"    Tool: {tc.name} (id={tc.id[:12]}...)")

                if tc.arguments:
                    args = tc.arguments if isinstance(tc.arguments, dict) else "<large>"
                    print(f"      Arguments: {json.dumps(args)[:60]}...")

                print(f"      Valid: {tc.is_valid}")

                if tc.validation_errors:
                    print(f"      Validation Errors:")
                    for err in tc.validation_errors:
                        print(f"        - {err.path}: {err.message}")

        if event.tool_validation_errors_count:
            print(f"\n  Tool Validation Errors: {event.tool_validation_errors_count}")

        print("=" * 60)

    return emit


def test_openai_content_capture() -> None:
    """Test content capture with OpenAI."""
    try:
        from openai import OpenAI
    except ImportError:
        print("OpenAI SDK not installed, skipping...")
        return

    print("\n### OpenAI Content Capture Test ###")

    client = OpenAI()

    # Non-streaming with system prompt
    print("\n--- Non-streaming with system prompt ---")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": "What is Python?"},
        ],
        temperature=0.7,
        max_tokens=100,
    )
    print(f"Model response: {response.choices[0].message.content}")

    # Streaming
    print("\n--- Streaming response ---")
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Count 1 to 3"}],
        stream=True,
    )
    print("Streaming: ", end="", flush=True)
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()

    # With tool calls
    print("\n--- With tool calls ---")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What's the weather in Paris and London?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["city"],
                    },
                },
            },
        ],
    )
    tool_calls = response.choices[0].message.tool_calls or []
    print(f"Tool calls made: {len(tool_calls)}")
    for tc in tool_calls:
        print(f"  - {tc.function.name}: {tc.function.arguments}")


def test_anthropic_content_capture() -> None:
    """Test content capture with Anthropic."""
    try:
        from anthropic import Anthropic
    except ImportError:
        print("Anthropic SDK not installed, skipping...")
        return

    print("\n### Anthropic Content Capture Test ###")

    client = Anthropic()

    # Non-streaming with system prompt
    print("\n--- Non-streaming with system prompt ---")
    response = client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=100,
        system="You are a helpful assistant. Be very concise.",
        messages=[{"role": "user", "content": "What is Python?"}],
    )
    print(f"Model response: {response.content[0].text}")

    # Streaming
    print("\n--- Streaming response ---")
    with client.messages.stream(
        model="claude-3-5-haiku-latest",
        max_tokens=50,
        messages=[{"role": "user", "content": "Count 1 to 3"}],
    ) as stream:
        print("Streaming: ", end="", flush=True)
        for text in stream.text_stream:
            print(text, end="", flush=True)
    print()

    # With tool calls
    print("\n--- With tool calls ---")
    response = client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=200,
        messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
        tools=[
            {
                "name": "get_weather",
                "description": "Get weather for a city",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["city"],
                },
            },
        ],
    )
    for block in response.content:
        if block.type == "tool_use":
            print(f"  Tool call: {block.name}({json.dumps(block.input)})")


def test_gemini_content_capture() -> None:
    """Test content capture with Gemini (google-generativeai)."""
    try:
        import google.generativeai as genai
    except ImportError:
        print("Gemini SDK (google-generativeai) not installed, skipping...")
        return

    print("\n### Gemini Content Capture Test ###")

    model = genai.GenerativeModel("gemini-2.0-flash")

    # Non-streaming with system instruction
    print("\n--- Non-streaming with system instruction ---")
    model_with_system = genai.GenerativeModel(
        "gemini-2.0-flash",
        system_instruction="You are a helpful assistant. Be very concise.",
    )
    response = model_with_system.generate_content("What is Python?")
    print(f"Model response: {response.text[:150]}...")

    # Streaming
    print("\n--- Streaming response ---")
    response = model.generate_content("Count 1 to 3", stream=True)
    print("Streaming: ", end="", flush=True)
    for chunk in response:
        if chunk.text:
            print(chunk.text, end="", flush=True)
    print()


def test_genai_content_capture() -> None:
    """Test content capture with google-genai SDK."""
    try:
        from google import genai
    except ImportError:
        print("GenAI SDK (google-genai) not installed, skipping...")
        return

    print("\n### GenAI (google-genai) Content Capture Test ###")

    client = genai.Client()

    # Non-streaming
    print("\n--- Non-streaming ---")
    response = client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents="What is Python? Answer in one sentence.",
    )
    print(f"Model response: {response.text[:150]}...")


def test_validation_errors() -> None:
    """Test schema validation error detection."""
    try:
        from openai import OpenAI
    except ImportError:
        print("OpenAI SDK not installed, skipping validation test...")
        return

    print("\n### Schema Validation Test ###")
    print("(Asking model to use tool - validation will check arguments)")

    client = OpenAI()

    # Create a tool with strict schema
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Calculate 15 * 23"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Perform a calculation",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "enum": ["add", "subtract", "multiply", "divide"],
                            },
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                        "required": ["operation", "a", "b"],
                    },
                },
            },
        ],
    )

    tool_calls = response.choices[0].message.tool_calls or []
    print(f"Tool calls: {len(tool_calls)}")
    for tc in tool_calls:
        print(f"  - {tc.function.name}: {tc.function.arguments}")


def main() -> None:
    """Run content capture examples."""
    print("=" * 60)
    print("Content Capture Example (Layer 0 & Layer 6)")
    print("=" * 60)

    # Initialize instrumentation with content capture enabled
    result = instrument(
        MeterOptions(
            emit_metric=create_capture_emitter(),
            # Enable Layer 0: Raw Content Capture
            capture_content=True,
            content_capture_options=ContentCaptureOptions(
                max_content_bytes=4096,
                capture_system_prompt=True,
                capture_messages=True,
                capture_tools_schema=True,
                capture_response=True,
            ),
            # Enable Layer 6: Tool Call Deep Inspection
            capture_tool_calls=True,
            validate_tool_schemas=True,
        )
    )
    print(f"\nInstrumented: openai={result.openai}, anthropic={result.anthropic}, gemini={result.gemini}, genai={result.genai}")

    try:
        test_openai_content_capture()
        test_anthropic_content_capture()
        test_gemini_content_capture()
        test_genai_content_capture()
        test_validation_errors()
    finally:
        uninstrument()

    print("\n" + "=" * 60)
    print("All content capture tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
