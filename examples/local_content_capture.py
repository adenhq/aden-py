"""
Local Content Capture Example

Captures LLM content (prompts, responses, tool calls) to a local JSONL file.
Useful for debugging, auditing, and offline analysis.

Run: python examples/local_content_capture.py
Output: ./metrics.jsonl
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
    pass

from aden import (
    instrument,
    uninstrument,
    MeterOptions,
    ContentCaptureOptions,
    create_jsonl_emitter,
)


def main() -> None:
    """Capture LLM content to a local JSONL file."""
    output_file = "./metrics.jsonl"

    print("=" * 60)
    print("Local Content Capture Example")
    print(f"Output file: {output_file}")
    print("=" * 60)

    # Create JSONL emitter
    emitter = create_jsonl_emitter(output_file, append=False)

    # Initialize instrumentation with content capture
    result = instrument(
        MeterOptions(
            emit_metric=emitter,
            capture_content=True,
            content_capture_options=ContentCaptureOptions(
                capture_system_prompt=True,
                capture_messages=True,
                capture_tools_schema=True,
                capture_response=True,
            ),
            capture_tool_calls=True,
        )
    )
    print(f"\nInstrumented: openai={result.openai}, anthropic={result.anthropic}, gemini={result.gemini}")

    try:
        # Test with OpenAI
        try:
            from openai import OpenAI
            client = OpenAI()

            print("\n--- OpenAI request ---")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is 2 + 2?"},
                ],
                max_tokens=50,
            )
            print(f"Response: {response.choices[0].message.content}")
        except ImportError:
            print("OpenAI SDK not installed, skipping...")
        except Exception as e:
            print(f"OpenAI error: {e}")

        # Test with Anthropic
        try:
            from anthropic import Anthropic
            client = Anthropic()

            print("\n--- Anthropic request ---")
            response = client.messages.create(
                model="claude-3-5-haiku-latest",
                max_tokens=50,
                system="You are a helpful assistant.",
                messages=[{"role": "user", "content": "What is 3 + 3?"}],
            )
            print(f"Response: {response.content[0].text}")
        except ImportError:
            print("Anthropic SDK not installed, skipping...")
        except Exception as e:
            print(f"Anthropic error: {e}")

    finally:
        uninstrument()
        emitter.close()

    print("\n" + "=" * 60)
    print(f"Content captured to: {output_file}")
    print("View with: cat metrics.jsonl | jq .")
    print("=" * 60)


if __name__ == "__main__":
    main()
