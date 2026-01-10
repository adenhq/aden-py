"""
Call stack capture utilities for LLM instrumentation.

Captures the call stack at the time of an LLM API call to provide
context about which agent/handler triggered the request.
"""

import inspect
from dataclasses import dataclass
from typing import Any


# Known framework patterns to identify as agents
AGENT_PATTERNS = [
    # Google ADK
    ("google.adk", "Agent"),
    ("google.adk", "Runner"),
    ("google.adk", "InMemoryRunner"),
    # PydanticAI
    ("pydantic_ai", "Agent"),
    ("pydantic_ai", "Tool"),
    # LangChain
    ("langchain", "Agent"),
    ("langchain", "Chain"),
    ("langchain", "Tool"),
    # CrewAI
    ("crewai", "Agent"),
    ("crewai", "Crew"),
    ("crewai", "Task"),
    # AutoGen
    ("autogen", "Agent"),
    ("autogen", "AssistantAgent"),
    # Generic patterns
    ("agent", ""),
    ("handler", ""),
    ("runner", ""),
]

# Packages to skip in call stack (internal SDK packages)
SKIP_PACKAGES = {
    "aden",
    "openai",
    "anthropic",
    "google.generativeai",
    "google.genai",
    "httpx",
    "httpcore",
    "anyio",
    "asyncio",
    "threading",
}


@dataclass
class CallStackInfo:
    """Captured call stack information."""

    call_site_file: str | None = None
    """File path where the LLM call originated."""

    call_site_line: int | None = None
    """Line number where the call originated."""

    call_site_function: str | None = None
    """Function name where the call originated."""

    call_stack: list[str] | None = None
    """Full call stack (file:line:function format)."""

    agent_stack: list[str] | None = None
    """Stack of agent/handler names (framework-specific)."""


def _should_skip_frame(frame: Any) -> bool:
    """Check if a frame should be skipped (internal SDK frame)."""
    filename = frame.f_code.co_filename

    # Skip frames from known internal packages
    for pkg in SKIP_PACKAGES:
        if f"/{pkg}/" in filename or f"\\{pkg}\\" in filename:
            return True
        if filename.endswith(f"/{pkg}.py") or filename.endswith(f"\\{pkg}.py"):
            return True

    # Skip <frozen> and built-in frames
    if filename.startswith("<"):
        return True

    return False


def _is_agent_frame(frame: Any) -> str | None:
    """Check if a frame is from an agent framework, return agent name if so."""
    filename = frame.f_code.co_filename.lower()
    func_name = frame.f_code.co_name

    # Check module/package patterns
    for pattern_pkg, pattern_cls in AGENT_PATTERNS:
        if pattern_pkg in filename:
            # Try to get class name from local variables
            local_self = frame.f_locals.get("self")
            if local_self is not None:
                cls_name = type(local_self).__name__
                if pattern_cls and pattern_cls.lower() in cls_name.lower():
                    return f"{cls_name}.{func_name}"
                elif not pattern_cls:
                    # Generic pattern match
                    return f"{cls_name}.{func_name}"

    # Check function name patterns
    func_lower = func_name.lower()
    if any(p in func_lower for p in ["agent", "handler", "runner", "execute", "invoke"]):
        local_self = frame.f_locals.get("self")
        if local_self is not None:
            cls_name = type(local_self).__name__
            return f"{cls_name}.{func_name}"

    return None


def capture_call_stack(skip_frames: int = 2, max_depth: int = 20) -> CallStackInfo:
    """
    Capture the current call stack for LLM request attribution.

    Args:
        skip_frames: Number of frames to skip (default 2 for this function + caller)
        max_depth: Maximum stack depth to capture

    Returns:
        CallStackInfo with call site and stack information
    """
    result = CallStackInfo()
    call_stack: list[str] = []
    agent_stack: list[str] = []
    first_user_frame = None

    try:
        # Get the current stack
        stack = inspect.stack()

        for i, frame_info in enumerate(stack):
            if i < skip_frames:
                continue
            if i >= skip_frames + max_depth:
                break

            frame = frame_info.frame

            # Skip internal SDK frames
            if _should_skip_frame(frame):
                continue

            filename = frame_info.filename
            lineno = frame_info.lineno
            func_name = frame_info.function

            # Record first user frame as call site
            if first_user_frame is None:
                first_user_frame = frame_info
                result.call_site_file = filename
                result.call_site_line = lineno
                result.call_site_function = func_name

            # Add to call stack
            call_stack.append(f"{filename}:{lineno}:{func_name}")

            # Check if this is an agent frame
            agent_name = _is_agent_frame(frame)
            if agent_name and agent_name not in agent_stack:
                agent_stack.append(agent_name)

    except Exception:
        # Don't let stack capture errors break instrumentation
        pass

    result.call_stack = call_stack if call_stack else None
    result.agent_stack = agent_stack if agent_stack else None

    return result


def format_call_stack(info: CallStackInfo, max_entries: int = 10) -> str:
    """Format call stack info for display."""
    lines = []

    if info.call_site_file:
        lines.append(f"Call site: {info.call_site_file}:{info.call_site_line} in {info.call_site_function}")

    if info.agent_stack:
        lines.append(f"Agent stack: {' -> '.join(info.agent_stack)}")

    if info.call_stack:
        lines.append("Call stack:")
        for entry in info.call_stack[:max_entries]:
            lines.append(f"  {entry}")
        if len(info.call_stack) > max_entries:
            lines.append(f"  ... and {len(info.call_stack) - max_entries} more")

    return "\n".join(lines)
