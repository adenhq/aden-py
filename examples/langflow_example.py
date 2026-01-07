"""
LangFlow Example

Demonstrates Aden instrumentation with LangFlow.
LangFlow uses LangChain under the hood, with Google Gemini using
the low-level gRPC client (GenerativeServiceClient).

Features demonstrated:
- LangFlow with OpenAI provider
- LangFlow with Google Gemini provider (requires gemini_grpc instrumentation)

Run: python examples/langflow_example.py

Requirements:
    pip install langflow python-dotenv
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

try:
    from langflow.components.models import LanguageModelComponent
except ImportError:
    print("LangFlow not installed. Run: pip install langflow")
    sys.exit(1)

from aden import (
    instrument,
    uninstrument,
    create_console_emitter,
    MeterOptions,
)


# =============================================================================
# LangFlow + OpenAI Example
# =============================================================================

def test_langflow_openai() -> None:
    """Test LangFlow with OpenAI provider."""
    print("\n=== LangFlow + OpenAI ===")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set, skipping OpenAI test")
        return

    comp = LanguageModelComponent()
    comp.set_attributes({
        "provider": "OpenAI",
        "model_name": "gpt-4o-mini",
        "api_key": api_key,
        "temperature": 0.1,
        "stream": False,
    })

    model = comp.build_model()
    response = model.invoke("What is 2 + 2? Reply with just the number.")
    print(f"Response: {response.content}")


# =============================================================================
# LangFlow + Google Gemini Example
# =============================================================================

def test_langflow_gemini() -> None:
    """Test LangFlow with Google Gemini provider.

    Note: LangFlow uses LangChain's ChatGoogleGenerativeAI, which internally
    uses the low-level gRPC client (GenerativeServiceClient). Aden's
    gemini_grpc instrumentation captures these calls automatically.
    """
    print("\n=== LangFlow + Google Gemini ===")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("GOOGLE_API_KEY not set, skipping Gemini test")
        return

    comp = LanguageModelComponent()
    comp.set_attributes({
        "provider": "Google",
        "model_name": "gemini-2.0-flash",
        "api_key": api_key,
        "temperature": 0.1,
        "stream": False,
    })

    model = comp.build_model()
    response = model.invoke("What is 3 + 3? Reply with just the number.")
    print(f"Response: {response.content}")


# =============================================================================
# LangFlow + Anthropic Example
# =============================================================================

def test_langflow_anthropic() -> None:
    """Test LangFlow with Anthropic provider."""
    print("\n=== LangFlow + Anthropic ===")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set, skipping Anthropic test")
        return

    comp = LanguageModelComponent()
    comp.set_attributes({
        "provider": "Anthropic",
        "model_name": "claude-3-haiku-20240307",
        "api_key": api_key,
        "temperature": 0.1,
        "stream": False,
    })

    model = comp.build_model()
    response = model.invoke("What is 4 + 4? Reply with just the number.")
    print(f"Response: {response.content}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Run all LangFlow examples."""
    print("=" * 60)
    print("LangFlow + Aden Instrumentation Example")
    print("=" * 60)

    # Initialize Aden instrumentation
    result = instrument(MeterOptions(
        emit_metric=create_console_emitter(pretty=True),
    ))

    print(f"\nInstrumented SDKs: {result}")
    print("-" * 60)

    try:
        # Run tests for each provider
        test_langflow_openai()
        test_langflow_gemini()
        test_langflow_anthropic()

    finally:
        # Clean up
        uninstrument()
        print("\n" + "=" * 60)
        print("Done!")


if __name__ == "__main__":
    main()
