"""
Anthropic SDK Basic Example

Tests: Messages API, streaming, non-streaming, tool use

Run: python examples/anthropic_basic.py
"""

import os
import sys

# Add parent directory to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use shell environment

from anthropic import Anthropic

from aden import (
    instrument,
    uninstrument,
    create_console_emitter,
    MeterOptions,
)


def test_messages(client: Anthropic) -> None:
    """Test non-streaming messages."""
    print("\n=== Anthropic Messages (non-streaming) ===")
    response = client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=100,
        messages=[{"role": "user", "content": "Say hello in 5 words"}],
    )

    for block in response.content:
        if block.type == "text":
            print(f"Response: {block.text}")
            break


def test_messages_streaming(client: Anthropic) -> None:
    """Test streaming messages."""
    print("\n=== Anthropic Messages (streaming) ===")

    with client.messages.stream(
        model="claude-3-5-haiku-latest",
        max_tokens=100,
        messages=[{"role": "user", "content": "Count from 1 to 5"}],
    ) as stream:
        print("Response: ", end="", flush=True)
        for text in stream.text_stream:
            print(text, end="", flush=True)
        print()


def test_with_tools(client: Anthropic) -> None:
    """Test messages with tool use."""
    print("\n=== Anthropic with Tool Use ===")
    response = client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=200,
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        tools=[
            {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                    },
                    "required": ["location"],
                },
            },
        ],
    )

    tool_uses = [b for b in response.content if b.type == "tool_use"]
    print(f"Tool calls: {len(tool_uses)}")
    for tool_use in tool_uses:
        print(f"  - {tool_use.name}({tool_use.input})")


def test_with_cache(client: Anthropic) -> None:
    """Test messages with prompt caching."""
    print("\n=== Anthropic with Prompt Caching ===")
    # Long system prompt to trigger caching
    system_prompt = "You are a helpful assistant. " * 100

    response = client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=50,
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            },
        ],
        messages=[{"role": "user", "content": "Hi!"}],
    )

    cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
    print(f"Cache tokens: {cache_read}")


def main() -> None:
    """Run all Anthropic tests."""
    print("Starting Anthropic SDK tests...\n")

    # Initialize instrumentation
    result = instrument(
        MeterOptions(
            emit_metric=create_console_emitter(pretty=True),
        )
    )
    print(f"Instrumented: anthropic={result.anthropic}")

    # Create client AFTER instrumentation
    client = Anthropic()

    try:
        test_messages(client)
        test_messages_streaming(client)
        test_with_tools(client)
        # test_with_cache(client)  # Uncomment if you have cache-enabled model
    finally:
        uninstrument()

    print("\n=== All Anthropic tests complete ===\n")


if __name__ == "__main__":
    main()
