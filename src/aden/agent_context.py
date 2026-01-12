"""
Agent context management using contextvars.

Provides reliable agent identification without per-request stack inspection.
Supports three modes:
1. Explicit context manager: `with aden.agent("name"): ...`
2. Framework integration: `aden.set_current_agent("name")`
3. Fallback: Derive from agent_stack (existing heuristics)
"""

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generator

from .call_stack import CallStackInfo


@dataclass
class AgentInfo:
    """Information about the current agent."""

    name: str
    """Human-readable agent name."""

    metadata: dict[str, Any] | None = None
    """Optional metadata about the agent (type, version, etc.)."""


# Context variable for tracking current agent
_current_agent: contextvars.ContextVar[AgentInfo | None] = contextvars.ContextVar(
    "aden_current_agent", default=None
)


@contextmanager
def agent(name: str, **metadata: Any) -> Generator[None, None, None]:
    """
    Context manager to explicitly scope LLM calls to an agent.

    All LLM calls within this context will be tagged with the agent name.
    Supports nesting - inner contexts override outer ones.

    Args:
        name: Human-readable agent name
        **metadata: Optional metadata (type, version, etc.)

    Example:
        with aden.agent("research-agent", type="researcher"):
            response = client.chat.completions.create(...)
            # This call is tagged as "research-agent"
    """
    info = AgentInfo(name=name, metadata=metadata if metadata else None)
    token = _current_agent.set(info)
    try:
        yield
    finally:
        _current_agent.reset(token)


def set_current_agent(name: str, **metadata: Any) -> contextvars.Token[AgentInfo | None]:
    """
    Set the current agent for this context.

    For framework integrations - call this when an agent starts executing.
    Returns a token that can be used to reset the context.

    Args:
        name: Human-readable agent name
        **metadata: Optional metadata

    Returns:
        Token to reset the agent context

    Example (framework integration):
        class MyAgent:
            def run(self):
                token = aden.set_current_agent(self.__class__.__name__)
                try:
                    # agent execution
                finally:
                    aden.reset_agent(token)
    """
    info = AgentInfo(name=name, metadata=metadata if metadata else None)
    return _current_agent.set(info)


def reset_agent(token: contextvars.Token[AgentInfo | None]) -> None:
    """
    Reset agent context to previous state.

    Args:
        token: Token returned from set_current_agent()
    """
    _current_agent.reset(token)


def get_current_agent_info() -> AgentInfo | None:
    """
    Get the current agent info if set explicitly.

    Returns:
        AgentInfo if agent context is set, None otherwise
    """
    return _current_agent.get()


def get_current_agent(
    stack_info: CallStackInfo | None = None,
    fallback_name: str | None = None,
) -> str | None:
    """
    Get the current agent name using the priority chain:
    1. Explicit context (set via `agent()` or `set_current_agent()`)
    2. Derived from stack_info.agent_stack (heuristic fallback)
    3. fallback_name (usually from MeterOptions.agent_name)

    Args:
        stack_info: Optional CallStackInfo with agent_stack from heuristics
        fallback_name: Optional fallback name (e.g., from MeterOptions)

    Returns:
        Agent name or None if no agent identified
    """
    # Priority 1: Explicit context
    info = _current_agent.get()
    if info is not None:
        return info.name

    # Priority 2: Derived from stack heuristics
    if stack_info and stack_info.agent_stack:
        return stack_info.agent_stack[0]

    # Priority 3: Explicit fallback
    return fallback_name


def get_current_agent_metadata() -> dict[str, Any] | None:
    """
    Get metadata for the current agent if available.

    Returns:
        Metadata dict or None
    """
    info = _current_agent.get()
    return info.metadata if info else None
