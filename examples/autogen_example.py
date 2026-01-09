"""
AutoGen Basic Example

Tests: Whether Aden's existing instrumentation captures AutoGen's LLM calls
for OpenAI, Anthropic, and Gemini.

AutoGen uses its own model clients, not the SDKs directly.
This example tests if existing aden instrumentation catches the calls.

Prerequisites:
    pip install autogen-agentchat "autogen-ext[openai,anthropic,gemini]"

Run: uv run python examples/autogen_example.py
"""

import asyncio
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

from aden import (
    instrument_async,
    uninstrument_async,
    create_console_emitter,
    MeterOptions,
)


async def test_openai():
    """Test AutoGen with OpenAI."""
    print("\n=== AutoGen + OpenAI ===")

    if not os.environ.get("OPENAI_API_KEY"):
        print("Skipped: OPENAI_API_KEY not set")
        return

    try:
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        from autogen_core.models import UserMessage
    except ImportError as e:
        print(f"Not installed: {e}")
        return

    client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    result = await client.create(
        messages=[UserMessage(content="Say hello in 3 words", source="user")],
    )

    print(f"Response: {result.content}")
    print(f"Usage: {result.usage}")


async def test_anthropic():
    """Test AutoGen with Anthropic."""
    print("\n=== AutoGen + Anthropic ===")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Skipped: ANTHROPIC_API_KEY not set")
        return

    try:
        from autogen_ext.models.anthropic import AnthropicChatCompletionClient
        from autogen_core.models import UserMessage
    except ImportError as e:
        print(f"Not installed: {e}")
        return

    client = AnthropicChatCompletionClient(
        model="claude-3-5-haiku-latest",
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )

    result = await client.create(
        messages=[UserMessage(content="Say hello in 3 words", source="user")],
    )

    print(f"Response: {result.content}")
    print(f"Usage: {result.usage}")


async def test_gemini():
    """Test AutoGen with Gemini via OpenAI-compatible API."""
    print("\n=== AutoGen + Gemini (OpenAI-compatible) ===")

    if not os.environ.get("GOOGLE_API_KEY"):
        print("Skipped: GOOGLE_API_KEY not set")
        return

    try:
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        from autogen_core.models import UserMessage
    except ImportError as e:
        print(f"Not installed: {e}")
        return

    # Gemini uses OpenAI-compatible API
    client = OpenAIChatCompletionClient(
        model="gemini-2.0-flash",
        api_key=os.environ.get("GOOGLE_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    result = await client.create(
        messages=[UserMessage(content="Say hello in 3 words", source="user")],
    )

    print(f"Response: {result.content}")
    print(f"Usage: {result.usage}")


async def main():
    """Run all AutoGen tests."""
    print("Starting AutoGen tests...")
    print("Testing if existing Aden instrumentation captures AutoGen calls...\n")

    # Initialize instrumentation BEFORE importing/using AutoGen
    # Use instrument_async to connect to control server
    result = await instrument_async(
        MeterOptions(
            api_key=os.environ.get("ADEN_API_KEY"),
            server_url=os.environ.get("ADEN_API_URL"),
            emit_metric=create_console_emitter(pretty=True),
        )
    )
    print(f"Instrumented: openai={result.openai}, anthropic={result.anthropic}, gemini={result.gemini}")

    try:
        await test_openai()
    except Exception as e:
        print(f"OpenAI Error: {e}")

    try:
        await test_anthropic()
    except Exception as e:
        print(f"Anthropic Error: {e}")

    try:
        await test_gemini()
    except Exception as e:
        print(f"Gemini Error: {e}")

    await uninstrument_async()

    print("\n=== AutoGen tests complete ===")
    print("\nCheck which providers showed metric events above.")
    print("If a provider didn't show metrics, we may need dedicated instrumentation.")


if __name__ == "__main__":
    asyncio.run(main())
