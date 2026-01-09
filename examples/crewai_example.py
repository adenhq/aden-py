"""
CrewAI Basic Example

Tests: Whether Aden's existing instrumentation captures CrewAI's LLM calls.

CrewAI uses LiteLLM internally, which makes direct HTTP calls via httpx
instead of using provider SDKs (OpenAI, Anthropic).

This example tests if existing aden instrumentation catches the calls.

Prerequisites:
    pip install crewai

Run: uv run python examples/crewai_example.py
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


def test_crewai_openai():
    """Test CrewAI with OpenAI (default)."""
    print("\n=== CrewAI + OpenAI ===")

    if not os.environ.get("OPENAI_API_KEY"):
        print("Skipped: OPENAI_API_KEY not set")
        return

    try:
        from crewai import Agent, Task, Crew
    except ImportError as e:
        print(f"Not installed: {e}")
        return

    # Create a simple agent
    agent = Agent(
        role="Greeter",
        goal="Say hello briefly",
        backstory="You are a friendly greeter who gives short responses.",
        llm="gpt-4o-mini",  # Uses OpenAI via LiteLLM
        verbose=False,
    )

    # Create a simple task
    task = Task(
        description="Say hello in exactly 3 words.",
        expected_output="A 3-word greeting",
        agent=agent,
    )

    # Create and run crew
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=False,
    )

    result = crew.kickoff()
    print(f"Response: {result.raw}")


def test_crewai_anthropic():
    """Test CrewAI with Anthropic."""
    print("\n=== CrewAI + Anthropic ===")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Skipped: ANTHROPIC_API_KEY not set")
        return

    try:
        from crewai import Agent, Task, Crew
    except ImportError as e:
        print(f"Not installed: {e}")
        return

    # Create agent with Anthropic model
    agent = Agent(
        role="Greeter",
        goal="Say hello briefly",
        backstory="You are a friendly greeter who gives short responses.",
        llm="anthropic/claude-3-5-haiku-latest",  # Anthropic via LiteLLM
        verbose=False,
    )

    task = Task(
        description="Say hello in exactly 3 words.",
        expected_output="A 3-word greeting",
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=False,
    )

    result = crew.kickoff()
    print(f"Response: {result.raw}")


def test_crewai_gemini():
    """Test CrewAI with Gemini."""
    print("\n=== CrewAI + Gemini ===")

    if not os.environ.get("GEMINI_API_KEY"):
        print("Skipped: GEMINI_API_KEY not set")
        return

    try:
        from crewai import Agent, Task, Crew
    except ImportError as e:
        print(f"Not installed: {e}")
        return

    # Create agent with Gemini model
    agent = Agent(
        role="Greeter",
        goal="Say hello briefly",
        backstory="You are a friendly greeter who gives short responses.",
        llm="gemini/gemini-2.0-flash",  # Gemini via LiteLLM
        verbose=False,
    )

    task = Task(
        description="Say hello in exactly 3 words.",
        expected_output="A 3-word greeting",
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=False,
    )

    result = crew.kickoff()
    print(f"Response: {result.raw}")


async def main():
    """Run all CrewAI tests."""
    print("Starting CrewAI tests...")
    print("Testing if existing Aden instrumentation captures CrewAI calls...")
    print("(CrewAI uses LiteLLM which uses httpx, not provider SDKs)\n")

    # Initialize instrumentation BEFORE importing/using CrewAI
    result = await instrument_async(
        MeterOptions(
            api_key=os.environ.get("ADEN_API_KEY"),
            server_url=os.environ.get("ADEN_API_URL"),
            emit_metric=create_console_emitter(pretty=True),
        )
    )
    print(f"Instrumented: openai={result.openai}, anthropic={result.anthropic}, gemini={result.gemini}")

    try:
        test_crewai_openai()
    except Exception as e:
        print(f"OpenAI Error: {e}")

    try:
        test_crewai_anthropic()
    except Exception as e:
        print(f"Anthropic Error: {e}")

    try:
        test_crewai_gemini()
    except Exception as e:
        print(f"Gemini Error: {e}")

    await uninstrument_async()

    print("\n=== CrewAI tests complete ===")
    print("\nCheck which providers showed metric events above.")
    print("If NO metrics appeared, CrewAI/LiteLLM bypasses our SDK patches.")
    print("We would need dedicated LiteLLM instrumentation.")


if __name__ == "__main__":
    asyncio.run(main())
