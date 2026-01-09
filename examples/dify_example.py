"""
Dify with Aden Instrumentation Example

This example demonstrates how to use Aden to monitor LLM usage
through Dify, an open-source LLM application development platform.

Dify runs as a separate server and handles LLM calls on the backend.
Aden automatically instruments the dify-client SDK to capture usage
metrics from the API responses.

Prerequisites:
1. A running Dify server (see: https://docs.dify.ai/getting-started/install-self-hosted)
2. A Dify application with an API key
3. pip install dify-client aden-py python-dotenv
"""

import os
from dotenv import load_dotenv

load_dotenv()

from aden import (
    instrument,
    uninstrument,
    MeterOptions,
    create_console_emitter,
    BeforeRequestResult,
    RequestCancelledError,
)


def budget_check(params, context):
    """
    Enforce budget limits before each Dify request.

    Note: Since Dify handles LLM calls on the server, budget checking
    happens before the Dify API call, not the underlying LLM call.
    """
    budget_info = getattr(context, 'budget', None)

    if budget_info and budget_info.get('exhausted', False):
        return BeforeRequestResult.cancel("Budget exhausted")

    if budget_info and budget_info.get('percent_used', 0) >= 95:
        return BeforeRequestResult.throttle(delay_ms=2000)

    return BeforeRequestResult.proceed()


# Initialize Aden instrumentation
instrument(MeterOptions(
    api_key=os.environ.get("ADEN_API_KEY"),
    server_url=os.environ.get("ADEN_API_URL"),
    emit_metric=create_console_emitter(pretty=True),
    on_alert=lambda alert: print(f"[Aden {alert.level}] {alert.message}"),
    before_request=budget_check,
))


def test_chat():
    """Test Dify ChatClient with Aden instrumentation."""
    from dify_client import ChatClient

    # Get Dify API key from environment
    dify_api_key = os.environ.get("DIFY_API_KEY")
    dify_api_url = os.environ.get("DIFY_API_URL", "http://localhost/v1")

    if not dify_api_key:
        print("⚠️  DIFY_API_KEY not set, skipping Dify test")
        return

    print("\n" + "=" * 50)
    print("Testing Dify ChatClient")
    print("=" * 50)

    try:
        # Create Dify ChatClient
        # Note: You may need to set api_base if using self-hosted Dify
        client = ChatClient(api_key=dify_api_key)

        # If using Dify Cloud or self-hosted, update the base URL
        if dify_api_url != "http://localhost/v1":
            client.base_url = dify_api_url

        # Send a chat message
        response = client.create_chat_message(
            inputs={},
            query="What is 2 + 2?",
            user="aden-test-user",
            response_mode="blocking",
        )

        # Check response
        response.raise_for_status()
        result = response.json()

        print(f"\n✅ Response: {result.get('answer', 'No answer')[:100]}...")

        # Print usage info from Dify
        metadata = result.get("metadata", {})
        usage = metadata.get("usage", {})
        if usage:
            print(f"\n📊 Usage from Dify:")
            print(f"   Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
            print(f"   Completion tokens: {usage.get('completion_tokens', 'N/A')}")
            print(f"   Total tokens: {usage.get('total_tokens', 'N/A')}")
            print(f"   Total cost: {usage.get('total_price', 'N/A')} {usage.get('currency', '')}")

    except RequestCancelledError as e:
        print(f"\n❌ Request cancelled (budget exceeded): {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def test_completion():
    """Test Dify CompletionClient with Aden instrumentation."""
    from dify_client import CompletionClient

    dify_api_key = os.environ.get("DIFY_COMPLETION_API_KEY")

    if not dify_api_key:
        print("\n⚠️  DIFY_COMPLETION_API_KEY not set, skipping completion test")
        return

    print("\n" + "=" * 50)
    print("Testing Dify CompletionClient")
    print("=" * 50)

    try:
        client = CompletionClient(api_key=dify_api_key)

        response = client.create_completion_message(
            inputs={"topic": "Python programming"},
            query="Write a haiku about the given topic.",
            user="aden-test-user",
            response_mode="blocking",
        )

        response.raise_for_status()
        result = response.json()

        print(f"\n✅ Response: {result.get('answer', 'No answer')}")

    except RequestCancelledError as e:
        print(f"\n❌ Request cancelled: {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    print("🚀 Dify + Aden Integration Example")
    print("=" * 50)
    print("\nRequired environment variables:")
    print("  DIFY_API_KEY         - Dify Chat App API Key")
    print("  DIFY_API_URL         - Dify API URL (default: http://localhost/v1)")
    print("  ADEN_API_KEY         - Aden API Key (optional)")
    print("  ADEN_API_URL         - Aden Server URL (optional)")
    print()

    try:
        test_chat()
        test_completion()
    finally:
        print("\n" + "=" * 50)
        print("Cleaning up...")
        uninstrument()
        print("Done!")
