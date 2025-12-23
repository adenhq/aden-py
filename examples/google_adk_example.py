"""
Google ADK (Agent Development Kit) Example

Demonstrates using Aden SDK with Google ADK agents.
ADK is Google's open-source framework for building AI agents with Gemini.

Prerequisites:
    pip install google-adk

Run: python examples/google_adk_example.py

Documentation:
    - ADK Docs: https://google.github.io/adk-docs/
    - ADK GitHub: https://github.com/google/adk-python
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

try:
    from google.adk.agents import Agent
    from google.adk.runners import InMemoryRunner
    from google.genai import types
except ImportError:
    print("Google ADK not installed. Run: pip install google-adk")
    sys.exit(1)

from aden import (
    instrument_async,
    uninstrument_async,
    create_console_emitter,
    MeterOptions,
)


# =============================================================================
# Tool Definitions
# =============================================================================

def get_weather(city: str) -> dict:
    """Get the current weather for a city.

    Args:
        city: The name of the city to get weather for.

    Returns:
        Weather information including temperature and conditions.
    """
    # Simulated weather data
    weather_data = {
        "Tokyo": {"temp": 22, "condition": "Partly Cloudy", "humidity": 65},
        "New York": {"temp": 18, "condition": "Sunny", "humidity": 45},
        "London": {"temp": 14, "condition": "Rainy", "humidity": 80},
        "Paris": {"temp": 16, "condition": "Cloudy", "humidity": 70},
        "Sydney": {"temp": 25, "condition": "Sunny", "humidity": 55},
    }

    if city in weather_data:
        data = weather_data[city]
        return {
            "status": "success",
            "city": city,
            "temperature_celsius": data["temp"],
            "condition": data["condition"],
            "humidity_percent": data["humidity"],
        }
    else:
        return {
            "status": "success",
            "city": city,
            "temperature_celsius": 20,
            "condition": "Unknown",
            "humidity_percent": 50,
            "note": "Weather data not available, using defaults",
        }


def calculate(expression: str) -> dict:
    """Evaluate a mathematical expression.

    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 2", "10 * 5").

    Returns:
        The result of the calculation.
    """
    try:
        # Safe evaluation of simple math expressions
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return {"status": "error", "message": "Invalid characters in expression"}

        result = eval(expression)
        return {
            "status": "success",
            "expression": expression,
            "result": result,
        }
    except Exception as e:
        return {
            "status": "error",
            "expression": expression,
            "message": str(e),
        }


# =============================================================================
# Agent Definition
# =============================================================================

# Create the ADK agent with Gemini model
# Available models: gemini-2.5-flash, gemini-2.5-pro, gemini-2.0-flash
weather_agent = Agent(
    name="weather_assistant",
    model="gemini-2.0-flash",  # Use Gemini 2.0 Flash
    description="A helpful assistant that provides weather information and can do calculations.",
    instruction="""You are a helpful weather assistant. You can:
1. Get current weather for cities using the get_weather tool
2. Perform calculations using the calculate tool

When asked about weather, always use the get_weather tool.
When asked to calculate something, use the calculate tool.
Be concise but friendly in your responses.""",
    tools=[get_weather, calculate],
)


# =============================================================================
# Runner Setup
# =============================================================================

async def run_agent_query(runner: InMemoryRunner, session_id: str, query: str) -> str:
    """Run a single query through the agent and return the response."""
    print(f"\n> User: {query}")

    # Create the user message
    content = types.Content(
        role="user",
        parts=[types.Part(text=query)]
    )

    # Run the agent and collect the response
    final_response = ""
    events = runner.run_async(
        user_id="demo_user",
        session_id=session_id,
        new_message=content,
    )

    async for event in events:
        # Check for tool calls (for visibility)
        if hasattr(event, 'tool_calls') and event.tool_calls:
            for tool_call in event.tool_calls:
                print(f"  [Tool Call] {tool_call.name}({tool_call.args})")

        # Check for final response
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response = event.content.parts[0].text
                print(f"< Agent: {final_response}")

    return final_response


async def main() -> None:
    """Run the ADK agent demo."""
    print("=" * 60)
    print("Google ADK + Aden SDK Demo")
    print("=" * 60)

    # Check for API key
    if not os.environ.get("GOOGLE_API_KEY"):
        print("\nGOOGLE_API_KEY not set. Please set it in .env or environment.")
        print("Get an API key from: https://aistudio.google.com/app/apikey")
        return

    # Initialize Aden instrumentation
    # If ADEN_API_KEY is set, metrics will be sent to the control server
    print("\nInitializing Aden instrumentation...")
    aden_api_key = os.environ.get("ADEN_API_KEY")
    aden_server_url = os.environ.get("ADEN_API_URL")

    result = await instrument_async(
        MeterOptions(
            api_key=aden_api_key,
            server_url=aden_server_url,
            emit_metric=create_console_emitter(pretty=True),
        )
    )
    print(f"Instrumented: gemini={result.gemini}, genai={result.genai}")
    if aden_api_key:
        print(f"Control server: {aden_server_url or 'default'}")

    # Create the InMemoryRunner
    print("\nCreating ADK agent runner...")
    runner = InMemoryRunner(
        agent=weather_agent,
        app_name="weather_demo",
    )

    # Create a session
    session = await runner.session_service.create_session(
        app_name="weather_demo",
        user_id="demo_user",
    )
    session_id = session.id
    print(f"Session created: {session_id[:8]}...")

    print("\n" + "=" * 60)
    print("Running Agent Queries")
    print("=" * 60)

    try:
        # Query 1: Weather request
        await run_agent_query(
            runner, session_id,
            "What's the weather like in Tokyo?"
        )

        # Query 2: Another city
        await run_agent_query(
            runner, session_id,
            "How about New York?"
        )

        # Query 3: Calculation
        await run_agent_query(
            runner, session_id,
            "Can you calculate 25 * 4 + 10?"
        )

        # Query 4: Combined request
        await run_agent_query(
            runner, session_id,
            "What's the weather in London? And what's 100 divided by 4?"
        )

    finally:
        await uninstrument_async()

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
