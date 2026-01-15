"""
Haystack Basic Example

Tests: Whether Aden's existing instrumentation captures Haystack's LLM calls.

Haystack uses actual provider SDKs internally:
- OpenAI: built-in (openai>=1.99.2)
- Anthropic: anthropic-haystack package
- Gemini: google-ai-haystack package

This example tests if existing aden instrumentation catches the calls.

Prerequisites:
    pip install haystack-ai anthropic-haystack google-ai-haystack

Run: uv run python examples/haystack_example.py
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


def test_haystack_openai():
    """Test Haystack with OpenAI (built-in)."""
    print("\n=== Haystack + OpenAI ===")

    if not os.environ.get("OPENAI_API_KEY"):
        print("Skipped: OPENAI_API_KEY not set")
        return

    try:
        from haystack.components.generators import OpenAIGenerator
    except ImportError as e:
        print(f"Not installed: {e}")
        return

    # Create generator
    generator = OpenAIGenerator(model="gpt-4o-mini")

    # Run generation
    result = generator.run(prompt="Say hello in exactly 3 words.")
    response = result["replies"][0]
    print(f"Response: {response}")


def test_haystack_anthropic():
    """Test Haystack with Anthropic."""
    print("\n=== Haystack + Anthropic ===")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Skipped: ANTHROPIC_API_KEY not set")
        return

    try:
        from haystack_integrations.components.generators.anthropic import AnthropicGenerator
    except ImportError as e:
        print(f"Not installed: {e}")
        print("Install with: pip install anthropic-haystack")
        return

    # Create generator
    generator = AnthropicGenerator(model="claude-3-5-haiku-latest")

    # Run generation
    result = generator.run(prompt="Say hello in exactly 3 words.")
    response = result["replies"][0]
    print(f"Response: {response}")


def test_haystack_gemini():
    """Test Haystack with Gemini."""
    print("\n=== Haystack + Gemini ===")

    if not os.environ.get("GOOGLE_API_KEY"):
        print("Skipped: GOOGLE_API_KEY not set")
        return

    try:
        from haystack_integrations.components.generators.google_ai import GoogleAIGeminiGenerator
    except ImportError as e:
        print(f"Not installed: {e}")
        print("Install with: pip install google-ai-haystack")
        return

    # Create generator
    generator = GoogleAIGeminiGenerator(model="gemini-2.0-flash")

    # Run generation (Gemini uses 'parts' instead of 'prompt')
    result = generator.run(parts=["Say hello in exactly 3 words."])
    response = result["replies"][0]
    print(f"Response: {response}")


async def main():
    """Run all Haystack tests."""
    print("Starting Haystack tests...")
    print("Testing if existing Aden instrumentation captures Haystack calls...")
    print("(Haystack uses actual provider SDKs internally)\n")

    # Initialize instrumentation BEFORE importing/using Haystack
    result = await instrument_async(
        MeterOptions(
            api_key=os.environ.get("ADEN_API_KEY"),
            server_url=os.environ.get("ADEN_API_URL"),
            emit_metric=create_console_emitter(pretty=True),
        )
    )
    print(f"Instrumented: openai={result.openai}, anthropic={result.anthropic}, gemini={result.gemini}")

    try:
        test_haystack_openai()
    except Exception as e:
        print(f"OpenAI Error: {e}")

    try:
        test_haystack_anthropic()
    except Exception as e:
        print(f"Anthropic Error: {e}")

    try:
        test_haystack_gemini()
    except Exception as e:
        print(f"Gemini Error: {e}")

    await uninstrument_async()

    print("\n=== Haystack tests complete ===")
    print("\nCheck which providers showed metric events above.")
    print("If a provider didn't show metrics, we may need dedicated instrumentation.")


if __name__ == "__main__":
    asyncio.run(main())
