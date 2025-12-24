"""
OpenAI SDK Basic Example

Tests: Chat Completions API, streaming, non-streaming, tool calls

Run: python examples/openai_basic.py
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

from openai import OpenAI

from aden import (
    instrument,
    uninstrument,
    create_console_emitter,
    create_control_agent,
    create_control_agent_emitter,
    create_multi_emitter,
    ControlAgentOptions,
    MeterOptions,
)


def test_chat_completion(client: OpenAI) -> None:
    """Test non-streaming chat completion."""
    print("\n=== OpenAI Chat Completion (non-streaming) ===")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello in 5 words"}],
    )
    print(f"Response: {response.choices[0].message.content}")


def test_chat_completion_streaming(client: OpenAI) -> None:
    """Test streaming chat completion."""
    print("\n=== OpenAI Chat Completion (streaming) ===")
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Count from 1 to 5"}],
        stream=True,
    )

    print("Response: ", end="", flush=True)
    for chunk in stream:
        content = chunk.choices[0].delta.content or ""
        print(content, end="", flush=True)
    print()


def test_with_tools(client: OpenAI) -> None:
    """Test chat completion with tool calls."""
    print("\n=== OpenAI with Tool Calls ===")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                        },
                        "required": ["location"],
                    },
                },
            },
        ],
    )
    tool_calls = response.choices[0].message.tool_calls or []
    print(f"Tool calls: {len(tool_calls)}")
    for call in tool_calls:
        print(f"  - {call.function.name}({call.function.arguments})")


def main() -> None:
    """Run all OpenAI tests."""
    print("Starting OpenAI SDK tests...\n")

    # Create control agent that connects to the server
    agent = create_control_agent(ControlAgentOptions(
        server_url=os.environ.get("ADEN_API_URL", "http://localhost:8888"),
        api_key=os.environ.get("ADEN_API_KEY", ""),
    ))

    # Initialize instrumentation with both console AND server emitters
    result = instrument(
        MeterOptions(
            emit_metric=create_multi_emitter([
                create_console_emitter(pretty=True),
                create_control_agent_emitter(agent),
            ]),
        )
    )
    print(f"Instrumented: openai={result.openai}")

    # Create client AFTER instrumentation
    client = OpenAI()

    try:
        test_chat_completion(client)
        test_chat_completion_streaming(client)
        test_with_tools(client)
    finally:
        uninstrument()
        agent.disconnect_sync()

    print("\n=== All OpenAI tests complete ===\n")


if __name__ == "__main__":
    main()
