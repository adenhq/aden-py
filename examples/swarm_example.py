"""
OpenAI Swarm Basic Example

Tests: Whether Aden's existing instrumentation captures OpenAI Swarm's LLM calls.

OpenAI Swarm internally uses the OpenAI Python SDK directly:
"internally just instantiates an OpenAI client"

Note: Swarm only supports OpenAI (no Anthropic, Gemini).
Note: Swarm is experimental/educational. OpenAI recommends using Agents SDK for production.

Prerequisites:
    pip install git+https://github.com/openai/swarm.git

Run: uv run python examples/swarm_example.py
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


def test_swarm_basic():
    """Test basic Swarm agent."""
    print("\n=== Swarm Basic Agent ===")

    if not os.environ.get("OPENAI_API_KEY"):
        print("Skipped: OPENAI_API_KEY not set")
        return

    try:
        from swarm import Swarm, Agent
    except ImportError as e:
        print(f"Not installed: {e}")
        print("Install with: pip install git+https://github.com/openai/swarm.git")
        return

    # Create a simple agent
    agent = Agent(
        name="Greeter",
        instructions="You are a friendly greeter. Keep responses very short (3 words max).",
    )

    # Create Swarm client and run
    client = Swarm()
    response = client.run(
        agent=agent,
        messages=[{"role": "user", "content": "Say hello"}],
    )

    print(f"Response: {response.messages[-1]['content']}")


def test_swarm_with_function():
    """Test Swarm agent with function calling."""
    print("\n=== Swarm Agent with Function ===")

    if not os.environ.get("OPENAI_API_KEY"):
        print("Skipped: OPENAI_API_KEY not set")
        return

    try:
        from swarm import Swarm, Agent
    except ImportError as e:
        print(f"Not installed: {e}")
        return

    def get_weather(location: str) -> str:
        """Get weather for a location."""
        return f"The weather in {location} is sunny, 25°C."

    # Create agent with function
    agent = Agent(
        name="WeatherBot",
        instructions="You help users check the weather. Use the get_weather function.",
        functions=[get_weather],
    )

    client = Swarm()
    response = client.run(
        agent=agent,
        messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    )

    print(f"Response: {response.messages[-1]['content']}")


def test_swarm_handoff():
    """Test Swarm agent handoff."""
    print("\n=== Swarm Agent Handoff ===")

    if not os.environ.get("OPENAI_API_KEY"):
        print("Skipped: OPENAI_API_KEY not set")
        return

    try:
        from swarm import Swarm, Agent
    except ImportError as e:
        print(f"Not installed: {e}")
        return

    # Create two agents
    sales_agent = Agent(
        name="Sales",
        instructions="You are a sales agent. Keep responses brief.",
    )

    def transfer_to_sales():
        """Transfer to sales agent."""
        return sales_agent

    triage_agent = Agent(
        name="Triage",
        instructions="You route users. If they want to buy something, transfer to sales.",
        functions=[transfer_to_sales],
    )

    client = Swarm()
    response = client.run(
        agent=triage_agent,
        messages=[{"role": "user", "content": "I want to buy a product"}],
    )

    print(f"Final Agent: {response.agent.name}")
    print(f"Response: {response.messages[-1]['content']}")


async def main():
    """Run all Swarm tests."""
    print("Starting OpenAI Swarm tests...")
    print("Testing if existing Aden instrumentation captures Swarm calls...")
    print("(Swarm uses OpenAI SDK directly)")
    print("(Note: Swarm only supports OpenAI, no Anthropic/Gemini)\n")

    # Initialize instrumentation BEFORE using Swarm
    result = await instrument_async(
        MeterOptions(
            api_key=os.environ.get("ADEN_API_KEY"),
            server_url=os.environ.get("ADEN_API_URL"),
            emit_metric=create_console_emitter(pretty=True),
        )
    )
    print(f"Instrumented: openai={result.openai}, anthropic={result.anthropic}, gemini={result.gemini}")

    try:
        test_swarm_basic()
    except Exception as e:
        print(f"Basic Error: {e}")

    try:
        test_swarm_with_function()
    except Exception as e:
        print(f"Function Error: {e}")

    try:
        test_swarm_handoff()
    except Exception as e:
        print(f"Handoff Error: {e}")

    await uninstrument_async()

    print("\n=== Swarm tests complete ===")
    print("\nCheck if OpenAI metric events appeared above.")
    print("If metrics appeared, existing instrumentation works for Swarm.")


if __name__ == "__main__":
    asyncio.run(main())
