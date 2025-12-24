"""
LangChain / LangGraph Example

Demonstrates Aden instrumentation with LangChain and LangGraph.
These frameworks use OpenAI/Anthropic SDKs under the hood,
so Aden's global instrumentation automatically captures all LLM calls.

Features demonstrated:
- Basic LangChain chat
- LangChain with tools
- LangGraph agent with state management
- Multi-step reasoning workflow

Run: python examples/langchain_example.py

Requirements:
    pip install langchain langchain-openai langgraph
"""

import asyncio
import os
import sys
from typing import Annotated, TypedDict

# Add parent directory to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use shell environment

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    from langchain_core.tools import tool
except ImportError:
    print("LangChain not installed. Run: pip install langchain langchain-openai")
    sys.exit(1)

try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    print("(LangGraph not installed, skipping graph examples)")

from aden import (
    instrument,
    uninstrument,
    create_console_emitter,
    create_control_agent,
    create_control_agent_emitter,
    create_multi_emitter,
    ControlAgentOptions,
    MeterOptions,
)


# =============================================================================
# Basic LangChain Chat Example
# =============================================================================

def test_basic_chat() -> None:
    """Test basic LangChain chat."""
    print("\n=== LangChain Basic Chat ===")

    llm = ChatOpenAI(model="gpt-4o-mini")

    messages = [
        SystemMessage(content="You are a helpful assistant. Keep responses brief."),
        HumanMessage(content="What is the capital of Japan?"),
    ]

    response = llm.invoke(messages)
    print(f"Response: {response.content}")


# =============================================================================
# LangChain Streaming Example
# =============================================================================

def test_streaming() -> None:
    """Test LangChain streaming."""
    print("\n=== LangChain Streaming ===")

    llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Count from 1 to 5."),
    ]

    print("Response: ", end="", flush=True)
    for chunk in llm.stream(messages):
        print(chunk.content, end="", flush=True)
    print()


# =============================================================================
# LangChain with Tools Example
# =============================================================================

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    # Simulated weather data
    weather_data = {
        "tokyo": "72°F, Sunny",
        "london": "58°F, Cloudy",
        "new york": "65°F, Partly Cloudy",
    }
    return weather_data.get(location.lower(), f"Weather data not available for {location}")


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)  # Note: only for demo, use safer eval in production
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


def test_with_tools() -> None:
    """Test LangChain with tool calling."""
    print("\n=== LangChain with Tools ===")

    llm = ChatOpenAI(model="gpt-4o-mini")
    llm_with_tools = llm.bind_tools([get_weather, calculate])

    messages = [
        SystemMessage(content="You are a helpful assistant with access to tools."),
        HumanMessage(content="What's the weather in Tokyo?"),
    ]

    response = llm_with_tools.invoke(messages)

    if response.tool_calls:
        print(f"Tool calls: {len(response.tool_calls)}")
        for call in response.tool_calls:
            print(f"  - {call['name']}({call['args']})")
            # Execute the tool
            if call['name'] == 'get_weather':
                result = get_weather.invoke(call['args'])
                print(f"    Result: {result}")
    else:
        print(f"Response: {response.content}")


# =============================================================================
# LangChain Chain Example
# =============================================================================

def test_chain() -> None:
    """Test LangChain with chained prompts."""
    print("\n=== LangChain Chain ===")

    llm = ChatOpenAI(model="gpt-4o-mini")

    # Step 1: Generate a topic
    topic_response = llm.invoke([
        SystemMessage(content="You generate single-word topics."),
        HumanMessage(content="Give me a random science topic in one word."),
    ])
    topic = topic_response.content.strip()
    print(f"Topic: {topic}")

    # Step 2: Generate a fact about the topic
    fact_response = llm.invoke([
        SystemMessage(content="You are a science educator. Be concise."),
        HumanMessage(content=f"Tell me one interesting fact about {topic}."),
    ])
    print(f"Fact: {fact_response.content}")


# =============================================================================
# LangGraph Agent Example
# =============================================================================

if HAS_LANGGRAPH:
    class AgentState(TypedDict):
        """State for the LangGraph agent."""
        messages: Annotated[list, add_messages]

    def test_langgraph_agent() -> None:
        """Test LangGraph agent with tools."""
        print("\n=== LangGraph Agent ===")

        llm = ChatOpenAI(model="gpt-4o-mini")
        tools = [get_weather, calculate]
        llm_with_tools = llm.bind_tools(tools)

        def agent_node(state: AgentState) -> dict:
            """The agent node that calls the LLM."""
            response = llm_with_tools.invoke(state["messages"])
            return {"messages": [response]}

        def should_continue(state: AgentState) -> str:
            """Determine if we should continue to tools or end."""
            last_message = state["messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return END

        # Build the graph
        graph = StateGraph(AgentState)
        graph.add_node("agent", agent_node)
        graph.add_node("tools", ToolNode(tools))

        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
        graph.add_edge("tools", "agent")

        app = graph.compile()

        # Run the agent
        initial_state = {
            "messages": [
                SystemMessage(content="You are a helpful assistant with access to weather and calculator tools."),
                HumanMessage(content="What's the weather in London?"),
            ]
        }

        result = app.invoke(initial_state)
        final_message = result["messages"][-1]
        print(f"Final response: {final_message.content}")

    def test_langgraph_multi_step() -> None:
        """Test LangGraph with multi-step reasoning."""
        print("\n=== LangGraph Multi-Step Workflow ===")

        llm = ChatOpenAI(model="gpt-4o-mini")

        class WorkflowState(TypedDict):
            """State for multi-step workflow."""
            task: str
            research: str
            draft: str
            final: str

        def research_node(state: WorkflowState) -> dict:
            """Research step."""
            print("Step 1: Researching...")
            response = llm.invoke([
                SystemMessage(content="You are a researcher. Provide 3 key bullet points."),
                HumanMessage(content=f"Research: {state['task']}"),
            ])
            return {"research": response.content}

        def draft_node(state: WorkflowState) -> dict:
            """Draft step."""
            print("Step 2: Drafting...")
            response = llm.invoke([
                SystemMessage(content="You are a writer. Write a brief paragraph."),
                HumanMessage(content=f"Write about this research:\n{state['research']}"),
            ])
            return {"draft": response.content}

        def review_node(state: WorkflowState) -> dict:
            """Review and finalize step."""
            print("Step 3: Reviewing...")
            response = llm.invoke([
                SystemMessage(content="You are an editor. Polish this text briefly."),
                HumanMessage(content=f"Edit this draft:\n{state['draft']}"),
            ])
            return {"final": response.content}

        # Build the workflow
        workflow = StateGraph(WorkflowState)
        workflow.add_node("research", research_node)
        workflow.add_node("draft", draft_node)
        workflow.add_node("review", review_node)

        workflow.add_edge(START, "research")
        workflow.add_edge("research", "draft")
        workflow.add_edge("draft", "review")
        workflow.add_edge("review", END)

        app = workflow.compile()

        # Run the workflow
        result = app.invoke({
            "task": "Benefits of solar energy",
            "research": "",
            "draft": "",
            "final": "",
        })

        print(f"\nFinal output:\n{result['final']}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Run all LangChain/LangGraph examples."""
    print("=" * 60)
    print("LangChain / LangGraph + Aden SDK Example")
    print("=" * 60)

    # Create control agent that connects to the server
    agent = create_control_agent(ControlAgentOptions(
        server_url=os.environ.get("ADEN_API_URL", "http://localhost:8888"),
        api_key=os.environ.get("ADEN_API_KEY", ""),
    ))

    # Initialize Aden instrumentation BEFORE creating LLMs
    # This patches OpenAI/Anthropic SDKs that LangChain uses internally
    result = instrument(
        MeterOptions(
            emit_metric=create_multi_emitter([
                create_console_emitter(pretty=True),
                create_control_agent_emitter(agent),
            ]),
            track_tool_calls=True,
        )
    )
    print(f"\nInstrumented: openai={result.openai}, anthropic={result.anthropic}")

    try:
        # LangChain examples
        test_basic_chat()
        test_streaming()
        test_with_tools()
        test_chain()

        # LangGraph examples (if available)
        if HAS_LANGGRAPH:
            test_langgraph_agent()
            test_langgraph_multi_step()
    finally:
        uninstrument()
        agent.disconnect_sync()

    print("\n" + "=" * 60)
    print("All LangChain/LangGraph tests complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
