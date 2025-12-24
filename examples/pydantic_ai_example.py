"""
PydanticAI Framework Example

Demonstrates Aden instrumentation with PydanticAI agents.
PydanticAI uses OpenAI/Anthropic/Gemini SDKs under the hood,
so Aden's global instrumentation automatically captures all LLM calls.

Features demonstrated:
- Basic agent with system prompt
- Structured output with Pydantic models
- Tool/function calling
- Multi-agent workflows
- Streaming responses

Run: python examples/pydantic_ai_example.py

Requirements:
    pip install pydantic-ai openai anthropic
"""

import asyncio
import os
import sys
from typing import Any

# Add parent directory to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use shell environment

try:
    from pydantic import BaseModel
    from pydantic_ai import Agent, RunContext
except ImportError:
    print("PydanticAI not installed. Run: pip install pydantic-ai")
    sys.exit(1)

from aden import (
    instrument_async,
    uninstrument_async,
    create_console_emitter,
    MeterOptions,
)


# =============================================================================
# Pydantic Models for Structured Output
# =============================================================================

class WeatherResponse(BaseModel):
    """Structured weather response."""
    location: str
    temperature: float
    unit: str
    conditions: str
    humidity: int


class TaskAnalysis(BaseModel):
    """Structured task analysis."""
    task: str
    complexity: str  # low, medium, high
    estimated_steps: int
    required_tools: list[str]
    recommendation: str


# =============================================================================
# Basic Agent Example
# =============================================================================

async def test_basic_agent() -> None:
    """Test basic agent with system prompt."""
    print("\n=== PydanticAI Basic Agent ===")

    agent = Agent(
        "openai:gpt-4o-mini",
        system_prompt="You are a helpful assistant. Keep responses brief.",
    )

    result = await agent.run("What is the capital of France?")
    print(f"Response: {result.output}")


# =============================================================================
# Structured Output Example
# =============================================================================

async def test_structured_output() -> None:
    """Test agent with structured Pydantic output."""
    print("\n=== PydanticAI Structured Output ===")

    agent = Agent(
        "openai:gpt-4o-mini",
        output_type=TaskAnalysis,
        system_prompt="Analyze the given task and provide a structured analysis.",
    )

    result = await agent.run("Build a REST API for user authentication")
    analysis = result.output

    print(f"Task: {analysis.task}")
    print(f"Complexity: {analysis.complexity}")
    print(f"Steps: {analysis.estimated_steps}")
    print(f"Tools: {', '.join(analysis.required_tools)}")
    print(f"Recommendation: {analysis.recommendation}")


# =============================================================================
# Tool Calling Example
# =============================================================================

async def test_with_tools() -> None:
    """Test agent with tool/function calling."""
    print("\n=== PydanticAI with Tools ===")

    agent = Agent(
        "openai:gpt-4o-mini",
        system_prompt="You are a helpful assistant with access to weather data.",
    )

    @agent.tool
    async def get_weather(ctx: RunContext[None], location: str) -> str:
        """Get current weather for a location."""
        # Simulated weather data
        return f"Weather in {location}: 72°F, Sunny, Humidity: 45%"

    @agent.tool
    async def get_forecast(ctx: RunContext[None], location: str, days: int = 3) -> str:
        """Get weather forecast for a location."""
        return f"{days}-day forecast for {location}: Mostly sunny with occasional clouds."

    result = await agent.run("What's the weather like in San Francisco?")
    print(f"Response: {result.output}")


# =============================================================================
# Multi-Provider Example
# =============================================================================

async def test_multi_provider() -> None:
    """Test agents with different providers."""
    print("\n=== PydanticAI Multi-Provider ===")

    # OpenAI agent
    openai_agent = Agent(
        "openai:gpt-4o-mini",
        system_prompt="You are a creative writer. Be concise.",
    )

    # Anthropic agent (if available)
    try:
        anthropic_agent = Agent(
            "anthropic:claude-3-5-haiku-latest",
            system_prompt="You are a technical analyst. Be precise.",
        )
        has_anthropic = True
    except Exception:
        has_anthropic = False
        print("(Anthropic not available, skipping)")

    # Get creative response from OpenAI
    creative_result = await openai_agent.run("Describe a sunset in one sentence.")
    print(f"OpenAI (creative): {creative_result.output}")

    # Get technical response from Anthropic
    if has_anthropic:
        technical_result = await anthropic_agent.run(
            "Explain why the sky appears red during sunset in one sentence."
        )
        print(f"Anthropic (technical): {technical_result.output}")


# =============================================================================
# Streaming Example
# =============================================================================

async def test_streaming() -> None:
    """Test agent with streaming response."""
    print("\n=== PydanticAI Streaming ===")

    agent = Agent(
        "openai:gpt-4o-mini",
        system_prompt="You are a storyteller.",
    )

    print("Response: ", end="", flush=True)
    async with agent.run_stream("Tell me a very short story about a robot.") as result:
        async for text in result.stream_text():
            print(text, end="", flush=True)
    print()


# =============================================================================
# Multi-Agent Workflow Example
# =============================================================================

async def test_multi_agent_workflow() -> None:
    """Test multi-agent workflow with handoffs."""
    print("\n=== PydanticAI Multi-Agent Workflow ===")

    # Research agent
    researcher = Agent(
        "openai:gpt-4o-mini",
        system_prompt="You are a researcher. Gather key facts briefly.",
    )

    # Writer agent
    writer = Agent(
        "openai:gpt-4o-mini",
        system_prompt="You are a writer. Create content from research notes.",
    )

    # Step 1: Research
    print("Step 1: Researching...")
    research_result = await researcher.run(
        "Research the key benefits of renewable energy. List 3 bullet points."
    )
    print(f"Research: {research_result.output[:100]}...")

    # Step 2: Write based on research
    print("\nStep 2: Writing...")
    write_result = await writer.run(
        f"Write a brief paragraph based on these research notes:\n{research_result.output}"
    )
    print(f"Article: {write_result.output}")


# =============================================================================
# Dependencies Example
# =============================================================================

async def test_with_dependencies() -> None:
    """Test agent with dependency injection."""
    print("\n=== PydanticAI with Dependencies ===")

    class UserContext:
        def __init__(self, user_id: str, preferences: dict[str, Any]) -> None:
            self.user_id = user_id
            self.preferences = preferences

    agent = Agent(
        "openai:gpt-4o-mini",
        deps_type=UserContext,
        system_prompt="You are a personalized assistant. Use the user's preferences.",
    )

    @agent.system_prompt
    async def add_user_context(ctx: RunContext[UserContext]) -> str:
        prefs = ctx.deps.preferences
        return f"User preferences: {prefs}. Tailor your response accordingly."

    user_ctx = UserContext(
        user_id="user_123",
        preferences={"style": "casual", "detail_level": "brief"},
    )

    result = await agent.run(
        "Recommend a good book to read.",
        deps=user_ctx,
    )
    print(f"Response: {result.output}")


# =============================================================================
# Main
# =============================================================================

async def main() -> None:
    """Run all PydanticAI examples."""
    print("=" * 60)
    print("PydanticAI + Aden SDK Example")
    print("=" * 60)

    # Initialize Aden instrumentation BEFORE creating agents
    # This patches OpenAI/Anthropic SDKs that PydanticAI uses internally
    # Use instrument_async() since we're in an async context
    result = await instrument_async(
        MeterOptions(
            emit_metric=create_console_emitter(pretty=True),
            track_tool_calls=True,
        )
    )
    print(f"\nInstrumented: openai={result.openai}, anthropic={result.anthropic}")

    try:
        await test_basic_agent()
        await test_structured_output()
        await test_with_tools()
        await test_multi_provider()
        await test_streaming()
        await test_multi_agent_workflow()
        await test_with_dependencies()
    finally:
        await uninstrument_async()

    print("\n" + "=" * 60)
    print("All PydanticAI tests complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
