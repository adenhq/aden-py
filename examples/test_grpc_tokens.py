"""
Test gRPC token capture with local emitter.

Run: ADEN_LOG_LEVEL=debug python examples/test_grpc_tokens.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from aden import instrument, uninstrument, MeterOptions, ContentCaptureOptions
from aden.emitters import create_console_emitter, create_memory_emitter


def test_gemini_grpc_tokens():
    """Test gRPC token capture."""
    try:
        import google.generativeai as genai
    except ImportError:
        print("google-generativeai not installed")
        return

    print("\n### Testing Gemini gRPC Token Capture ###")

    # Simple request
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content("Say hello in one word.")

    print(f"\nResponse: {response.text}")

    # Check usage_metadata directly
    print(f"\n--- Direct usage_metadata inspection ---")
    if hasattr(response, 'usage_metadata'):
        um = response.usage_metadata
        print(f"  prompt_token_count: {getattr(um, 'prompt_token_count', 'N/A')}")
        print(f"  candidates_token_count: {getattr(um, 'candidates_token_count', 'N/A')}")
        print(f"  total_token_count: {getattr(um, 'total_token_count', 'N/A')}")
        print(f"  cached_content_token_count: {getattr(um, 'cached_content_token_count', 'N/A')}")
    else:
        print("  No usage_metadata on response")


def main():
    print("=" * 70)
    print("Testing gRPC Token Capture")
    print("=" * 70)

    # Use console emitter to see metrics printed to stdout
    console_emitter = create_console_emitter()

    # Also use memory emitter to inspect the raw events
    memory_emitter = create_memory_emitter()

    def combined_emitter(event):
        console_emitter(event)
        memory_emitter(event)

    result = instrument(
        MeterOptions(
            emit_metric=combined_emitter,
            agent_name="test-grpc-agent",
            capture_content=True,
            content_capture_options=ContentCaptureOptions(
                max_content_bytes=4096,
                capture_system_prompt=True,
                capture_messages=True,
                capture_response=True,
            ),
        )
    )
    print(f"\nInstrumented: gemini={result.gemini}, gemini_grpc={result.gemini_grpc}")

    try:
        test_gemini_grpc_tokens()
    finally:
        uninstrument()

    # Print captured events from memory emitter
    print("\n" + "=" * 70)
    print("CAPTURED EVENTS (from memory emitter):")
    print("=" * 70)
    for i, event in enumerate(memory_emitter.events):
        print(f"\nEvent {i + 1}:")
        print(f"  Provider: {event.provider}")
        print(f"  Model: {event.model}")
        print(f"  Agent Name: {event.agent_name}")
        print(f"  Input tokens: {event.input_tokens}")
        print(f"  Output tokens: {event.output_tokens}")
        print(f"  Total tokens: {event.total_tokens}")
        print(f"  Cached tokens: {event.cached_tokens}")
        print(f"  Latency: {event.latency_ms:.0f}ms")

    print("\nDone!")


if __name__ == "__main__":
    main()
