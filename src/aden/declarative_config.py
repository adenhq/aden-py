"""
Declarative configuration for prompt and tool modifications.

This module provides a declarative way to configure prompt injections, tool injections,
and context compaction without writing custom hook functions.

Example usage:

    from aden import MeterOptions
    from aden.declarative_config import (
        DeclarativeConfig,
        SystemInjection,
        ToolInjection,
        CompactionConfig,
        build_modify_params_hook,
    )

    config = DeclarativeConfig(
        injections=[
            SystemInjection(
                content="Always respond in JSON format",
                position="prepend",
                conditions={"models": ["gpt-4o", "gpt-4o-mini"]},
            ),
            SystemInjection(
                content="Be concise and direct",
                position="append",
            ),
        ],
        tool_injections=[
            ToolInjection(
                name="audit_log",
                description="Log an audit event",
                parameters={"type": "object", "properties": {"event": {"type": "string"}}},
            ),
        ],
        compaction=CompactionConfig(
            max_messages=50,
            strategy="truncate",
            preserve_recent=10,
        ),
    )

    instrument_openai(MeterOptions(
        emit_metric=emitter,
        modify_params=build_modify_params_hook(config),
    ))
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from .types import BeforeRequestContext, ModifyParamsResult


@dataclass
class SystemInjection:
    """Configuration for injecting a system message.

    Attributes:
        content: The system message content to inject
        position: Where to inject - "prepend" (start) or "append" (end)
        conditions: Optional conditions for when to inject
            - models: List of model names/patterns to match
            - providers: List of providers to match (openai, anthropic, gemini)
    """
    content: str
    position: Literal["prepend", "append"] = "prepend"
    conditions: dict[str, Any] | None = None


@dataclass
class MessageInjection:
    """Configuration for injecting a custom message.

    Attributes:
        role: The message role (system, user, assistant)
        content: The message content
        position: Where to inject - "prepend", "append", or index
        conditions: Optional conditions for when to inject
    """
    role: Literal["system", "user", "assistant"]
    content: str
    position: Literal["prepend", "append"] | int = "prepend"
    conditions: dict[str, Any] | None = None


@dataclass
class ToolInjection:
    """Configuration for injecting a tool.

    Attributes:
        name: Tool function name
        description: Tool description
        parameters: JSON schema for tool parameters
        conditions: Optional conditions for when to inject
    """
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    conditions: dict[str, Any] | None = None


@dataclass
class CompactionConfig:
    """Configuration for context compaction.

    Attributes:
        max_messages: Maximum number of messages to keep
        max_tokens: Maximum token count (requires token counter)
        strategy: Compaction strategy - "truncate" or "summarize"
        preserve_recent: Number of recent messages to always preserve
        preserve_system: Whether to always preserve system messages
        token_counter: Optional function to count tokens in messages
    """
    max_messages: int | None = None
    max_tokens: int | None = None
    strategy: Literal["truncate", "summarize"] = "truncate"
    preserve_recent: int = 5
    preserve_system: bool = True
    token_counter: Callable[[list[dict]], int] | None = None


@dataclass
class ToolInstructionConfig:
    """Configuration for automatic tool usage instructions.

    Attributes:
        enabled: Whether to add tool instructions
        template: Template for the instruction (uses {tool_names} placeholder)
        position: Where to add the instruction
        force_tool_use: Whether to require tool use
    """
    enabled: bool = True
    template: str = "You should use the available tools when appropriate: {tool_names}"
    position: Literal["prepend", "append"] = "append"
    force_tool_use: bool = False


@dataclass
class DeclarativeConfig:
    """Complete declarative configuration for request modifications.

    Attributes:
        injections: List of system/message injections
        tool_injections: List of tools to inject
        compaction: Context compaction configuration
        tool_instruction: Tool usage instruction configuration
        track_diff: Whether to track original vs modified params
    """
    injections: list[SystemInjection | MessageInjection] = field(default_factory=list)
    tool_injections: list[ToolInjection] = field(default_factory=list)
    compaction: CompactionConfig | None = None
    tool_instruction: ToolInstructionConfig | None = None
    track_diff: bool = False


def _matches_conditions(conditions: dict[str, Any] | None, context: BeforeRequestContext, params: dict) -> bool:
    """Check if conditions match the current request context.

    Model matching supports:
    - Exact match: "gpt-4o" matches only "gpt-4o"
    - Wildcard prefix: "gpt-4o*" matches "gpt-4o", "gpt-4o-mini", etc.
    - Wildcard suffix: "*-mini" matches "gpt-4o-mini", etc.
    """
    if conditions is None:
        return True

    # Check model conditions
    if "models" in conditions:
        model = params.get("model", context.model)
        model_patterns = conditions["models"]

        def matches_pattern(pattern: str, value: str) -> bool:
            if pattern.startswith("*") and pattern.endswith("*"):
                # Contains match: *text*
                return pattern[1:-1] in value
            elif pattern.startswith("*"):
                # Suffix match: *-mini
                return value.endswith(pattern[1:])
            elif pattern.endswith("*"):
                # Prefix match: gpt-4o*
                return value.startswith(pattern[:-1])
            else:
                # Exact match
                return pattern == value

        if not any(matches_pattern(pattern, model) for pattern in model_patterns):
            return False

    # Check provider conditions
    if "providers" in conditions:
        # Provider info would need to be passed in context or inferred
        # For now, we can check based on model naming conventions
        pass

    # Check metadata conditions
    if "metadata" in conditions:
        request_metadata = context.metadata or {}
        for key, value in conditions["metadata"].items():
            if request_metadata.get(key) != value:
                return False

    return True


def _apply_injections(
    params: dict,
    context: BeforeRequestContext,
    injections: list[SystemInjection | MessageInjection],
) -> dict:
    """Apply message injections to params."""
    if not injections:
        return params

    messages = list(params.get("messages", []))

    for injection in injections:
        if not _matches_conditions(injection.conditions, context, params):
            continue

        if isinstance(injection, SystemInjection):
            msg = {"role": "system", "content": injection.content}
        else:
            msg = {"role": injection.role, "content": injection.content}

        position = injection.position
        if position == "prepend":
            messages.insert(0, msg)
        elif position == "append":
            messages.append(msg)
        elif isinstance(position, int):
            messages.insert(position, msg)

    return {**params, "messages": messages}


def _apply_tool_injections(
    params: dict,
    context: BeforeRequestContext,
    tool_injections: list[ToolInjection],
) -> dict:
    """Apply tool injections to params."""
    if not tool_injections:
        return params

    tools = list(params.get("tools") or [])

    for tool_config in tool_injections:
        if not _matches_conditions(tool_config.conditions, context, params):
            continue

        # Check if tool already exists
        existing_names = [t.get("function", {}).get("name") for t in tools]
        if tool_config.name in existing_names:
            continue

        tool = {
            "type": "function",
            "function": {
                "name": tool_config.name,
                "description": tool_config.description,
                "parameters": tool_config.parameters,
            },
        }
        tools.append(tool)

    if tools:
        return {**params, "tools": tools}
    return params


def _apply_compaction(
    params: dict,
    context: BeforeRequestContext,
    config: CompactionConfig,
) -> dict:
    """Apply context compaction to params."""
    messages = params.get("messages", [])
    if not messages:
        return params

    # Check if compaction is needed
    needs_compaction = False

    if config.max_messages and len(messages) > config.max_messages:
        needs_compaction = True

    if config.max_tokens and config.token_counter:
        token_count = config.token_counter(messages)
        if token_count > config.max_tokens:
            needs_compaction = True

    if not needs_compaction:
        return params

    # Apply compaction based on strategy
    if config.strategy == "truncate":
        # Separate system messages and other messages
        system_msgs = []
        other_msgs = []

        for msg in messages:
            if config.preserve_system and msg.get("role") == "system":
                system_msgs.append(msg)
            else:
                other_msgs.append(msg)

        # Keep recent messages
        preserved = other_msgs[-config.preserve_recent:] if config.preserve_recent else []

        # Calculate how many more we can keep
        target_count = (config.max_messages or len(messages)) - len(system_msgs) - len(preserved)

        if target_count > 0 and len(other_msgs) > config.preserve_recent:
            # Keep some older messages too
            older_to_keep = min(target_count, len(other_msgs) - config.preserve_recent)
            start_idx = len(other_msgs) - config.preserve_recent - older_to_keep
            middle_msgs = other_msgs[start_idx:start_idx + older_to_keep] if older_to_keep > 0 else []
        else:
            middle_msgs = []

        compacted = system_msgs + middle_msgs + preserved
        return {**params, "messages": compacted}

    # Summarize strategy would require LLM call - not implemented in declarative
    return params


def _apply_tool_instruction(
    params: dict,
    context: BeforeRequestContext,
    config: ToolInstructionConfig,
) -> dict:
    """Apply tool usage instruction to params."""
    if not config.enabled:
        return params

    tools = params.get("tools")
    if not tools:
        return params

    # Extract tool names
    tool_names = []
    for tool in tools:
        if isinstance(tool, dict):
            name = tool.get("function", {}).get("name") or tool.get("name")
            if name:
                tool_names.append(name)

    if not tool_names:
        return params

    # Generate instruction
    instruction = config.template.format(tool_names=", ".join(tool_names))

    if config.force_tool_use:
        instruction = f"IMPORTANT: You MUST use one of these tools: {', '.join(tool_names)}. Do not respond without using a tool."

    messages = list(params.get("messages", []))
    msg = {"role": "system", "content": instruction}

    if config.position == "prepend":
        messages.insert(0, msg)
    else:
        messages.append(msg)

    return {**params, "messages": messages}


def build_modify_params_hook(
    config: DeclarativeConfig,
) -> Callable[[dict, BeforeRequestContext], dict | ModifyParamsResult]:
    """Build a modify_params hook from declarative configuration.

    Args:
        config: The declarative configuration

    Returns:
        A function suitable for use as MeterOptions.modify_params
    """

    def modify_params_hook(params: dict, context: BeforeRequestContext) -> dict | ModifyParamsResult:
        import copy

        # Track original if requested
        original = copy.deepcopy(params) if config.track_diff else None

        modified = params

        # Apply injections
        if config.injections:
            modified = _apply_injections(modified, context, config.injections)

        # Apply tool injections
        if config.tool_injections:
            modified = _apply_tool_injections(modified, context, config.tool_injections)

        # Apply compaction
        if config.compaction:
            modified = _apply_compaction(modified, context, config.compaction)

        # Apply tool instruction
        if config.tool_instruction:
            modified = _apply_tool_instruction(modified, context, config.tool_instruction)

        # Return with diff tracking if enabled
        if config.track_diff:
            return ModifyParamsResult(params=modified, original_params=original)

        return modified

    return modify_params_hook


# Convenience functions for common patterns

def system_injection(
    content: str,
    position: Literal["prepend", "append"] = "prepend",
    models: list[str] | None = None,
) -> DeclarativeConfig:
    """Create a simple system injection config.

    Example:
        config = system_injection("Be concise", position="prepend")
        instrument_openai(MeterOptions(modify_params=build_modify_params_hook(config)))
    """
    conditions = {"models": models} if models else None
    return DeclarativeConfig(
        injections=[SystemInjection(content=content, position=position, conditions=conditions)]
    )


def tool_injection(
    name: str,
    description: str,
    parameters: dict[str, Any] | None = None,
) -> DeclarativeConfig:
    """Create a simple tool injection config.

    Example:
        config = tool_injection("audit_log", "Log audit events")
        instrument_openai(MeterOptions(modify_params=build_modify_params_hook(config)))
    """
    return DeclarativeConfig(
        tool_injections=[ToolInjection(
            name=name,
            description=description,
            parameters=parameters or {"type": "object", "properties": {}},
        )]
    )


def context_compaction(
    max_messages: int = 50,
    preserve_recent: int = 10,
) -> DeclarativeConfig:
    """Create a simple context compaction config.

    Example:
        config = context_compaction(max_messages=50, preserve_recent=10)
        instrument_openai(MeterOptions(modify_params=build_modify_params_hook(config)))
    """
    return DeclarativeConfig(
        compaction=CompactionConfig(
            max_messages=max_messages,
            preserve_recent=preserve_recent,
            strategy="truncate",
        )
    )


def enforce_tool_use() -> DeclarativeConfig:
    """Create a config that enforces tool use when tools are present.

    Example:
        config = enforce_tool_use()
        instrument_openai(MeterOptions(modify_params=build_modify_params_hook(config)))
    """
    return DeclarativeConfig(
        tool_instruction=ToolInstructionConfig(
            enabled=True,
            force_tool_use=True,
            position="append",
        )
    )


# JSON/dict-based configuration support

def from_dict(config_dict: dict[str, Any]) -> DeclarativeConfig:
    """Create DeclarativeConfig from a dictionary (e.g., loaded from JSON/YAML).

    Example:
        config_dict = {
            "injections": [
                {"content": "Be concise", "position": "prepend"}
            ],
            "tool_injections": [
                {"name": "audit_log", "description": "Log events"}
            ],
            "compaction": {
                "max_messages": 50,
                "preserve_recent": 10
            }
        }
        config = from_dict(config_dict)
    """
    injections = []
    for inj in config_dict.get("injections", []):
        if "role" in inj and inj["role"] != "system":
            injections.append(MessageInjection(**inj))
        else:
            # Remove 'role' if present for SystemInjection
            inj_copy = {k: v for k, v in inj.items() if k != "role"}
            injections.append(SystemInjection(**inj_copy))

    tool_injections = [
        ToolInjection(**ti) for ti in config_dict.get("tool_injections", [])
    ]

    compaction = None
    if "compaction" in config_dict:
        compaction = CompactionConfig(**config_dict["compaction"])

    tool_instruction = None
    if "tool_instruction" in config_dict:
        tool_instruction = ToolInstructionConfig(**config_dict["tool_instruction"])

    return DeclarativeConfig(
        injections=injections,
        tool_injections=tool_injections,
        compaction=compaction,
        tool_instruction=tool_instruction,
        track_diff=config_dict.get("track_diff", False),
    )
