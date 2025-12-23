"""
Google Gemini SDK Basic Example

Tests: generateContent, generateContentStream, chat sessions

Run: python examples/gemini_basic.py
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

import google.generativeai as genai

from aden import (
    instrument,
    uninstrument,
    create_console_emitter,
    MeterOptions,
)

# Configure with API key
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))


def test_generate_content() -> None:
    """Test non-streaming content generation."""
    print("\n=== Gemini generateContent (non-streaming) ===")
    model = genai.GenerativeModel("gemini-2.0-flash")

    result = model.generate_content("Say hello in 5 words")
    print(f"Response: {result.text}")


def test_generate_content_streaming() -> None:
    """Test streaming content generation."""
    print("\n=== Gemini generateContentStream (streaming) ===")
    model = genai.GenerativeModel("gemini-2.0-flash")

    response = model.generate_content("Count from 1 to 5", stream=True)

    print("Response: ", end="", flush=True)
    for chunk in response:
        print(chunk.text, end="", flush=True)
    print()


def test_chat_session() -> None:
    """Test chat session."""
    print("\n=== Gemini Chat Session ===")
    model = genai.GenerativeModel("gemini-2.0-flash")
    chat = model.start_chat(
        history=[
            {"role": "user", "parts": "Hi, I'm Alice"},
            {"role": "model", "parts": "Hello Alice! Nice to meet you."},
        ]
    )

    result = chat.send_message("What's my name?")
    print(f"Response: {result.text}")


def test_chat_session_streaming() -> None:
    """Test chat session with streaming."""
    print("\n=== Gemini Chat Session (streaming) ===")
    model = genai.GenerativeModel("gemini-2.0-flash")
    chat = model.start_chat()

    response = chat.send_message("Tell me a short joke", stream=True)

    print("Response: ", end="", flush=True)
    for chunk in response:
        print(chunk.text, end="", flush=True)
    print()


def test_with_system_instruction() -> None:
    """Test with system instruction."""
    print("\n=== Gemini with System Instruction ===")
    model = genai.GenerativeModel(
        "gemini-2.0-flash",
        system_instruction="You are a pirate. Always respond like a pirate.",
    )

    result = model.generate_content("How are you today?")
    print(f"Response: {result.text}")


def test_different_models() -> None:
    """Test different model tiers."""
    print("\n=== Gemini Different Models ===")

    # Flash model (fast)
    flash_model = genai.GenerativeModel("gemini-2.5-flash")
    flash_result = flash_model.generate_content("Say hi")
    print(f"Flash response: {flash_result.text[:50]}...")

    # Pro model (powerful)
    pro_model = genai.GenerativeModel("gemini-2.5-pro")
    pro_result = pro_model.generate_content("Say hello")
    print(f"Pro response: {pro_result.text[:50]}...")


def main() -> None:
    """Run all Gemini tests."""
    print("Starting Gemini SDK tests...\n")

    # Initialize instrumentation
    result = instrument(
        MeterOptions(
            emit_metric=create_console_emitter(pretty=True),
        )
    )
    print(f"Instrumented: gemini={result.gemini}")

    try:
        test_generate_content()
        test_generate_content_streaming()
        test_chat_session()
        test_chat_session_streaming()
        test_with_system_instruction()
        test_different_models()
    finally:
        uninstrument()

    print("\n=== All Gemini tests complete ===\n")


if __name__ == "__main__":
    main()
