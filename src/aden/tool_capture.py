"""
Tool call capture extraction for Layer 6.

Extracts detailed tool call information with argument parsing and validation
for OpenAI and Anthropic APIs.
"""

import json
from typing import Any

from .content_storage import process_content, process_string_content
from .schema_validator import find_tool_schema, validate_against_schema
from .types import (
    ContentCaptureOptions,
    ToolCallCapture,
    ToolCallValidationError,
)


def extract_openai_tool_calls(
    response: Any,
    tools_schema: list[dict[str, Any]] | None,
    options: ContentCaptureOptions,
    validate: bool = True,
) -> tuple[list[ToolCallCapture], list[dict[str, Any]]]:
    """Extract tool calls from OpenAI response.

    Handles both Chat Completions API and Responses API formats.

    Args:
        response: Response object from OpenAI
        tools_schema: Tool definitions from the request (for validation)
        options: Content capture configuration
        validate: Whether to validate arguments against schema

    Returns:
        Tuple of:
        - List of ToolCallCapture objects
        - List of large content payloads for server storage
    """
    captures: list[ToolCallCapture] = []
    large_content_payloads: list[dict[str, Any]] = []

    # Convert response to dict
    data: dict[str, Any] = {}
    if hasattr(response, "model_dump"):
        data = response.model_dump()
    elif hasattr(response, "__dict__"):
        data = response.__dict__
    elif isinstance(response, dict):
        data = response

    # Handle Chat Completions API format
    choices = data.get("choices", [])
    if choices:
        first_choice = choices[0] if isinstance(choices[0], dict) else {}
        message = first_choice.get("message", {})
        tool_calls = message.get("tool_calls", []) if isinstance(message, dict) else []

        for idx, tc in enumerate(tool_calls):
            if not isinstance(tc, dict):
                continue

            tc_id = tc.get("id", "")
            func = tc.get("function", {})
            if not isinstance(func, dict):
                continue

            name = func.get("name", "")
            arguments_raw = func.get("arguments", "{}")

            capture, payloads = _process_tool_call(
                tc_id=tc_id,
                name=name,
                arguments_raw=arguments_raw,
                index=idx,
                tools_schema=tools_schema,
                options=options,
                validate=validate,
            )
            captures.append(capture)
            large_content_payloads.extend(payloads)

    # Handle Responses API format (output array with function_call items)
    output = data.get("output", [])
    if output and not choices:
        idx = 0
        for item in output:
            if not isinstance(item, dict):
                continue

            item_type = item.get("type", "")
            if item_type == "function_call":
                tc_id = item.get("call_id", item.get("id", ""))
                name = item.get("name", "")
                arguments_raw = item.get("arguments", "{}")

                capture, payloads = _process_tool_call(
                    tc_id=tc_id,
                    name=name,
                    arguments_raw=arguments_raw,
                    index=idx,
                    tools_schema=tools_schema,
                    options=options,
                    validate=validate,
                )
                captures.append(capture)
                large_content_payloads.extend(payloads)
                idx += 1

    return captures, large_content_payloads


def extract_anthropic_tool_calls(
    response: Any,
    tools_schema: list[dict[str, Any]] | None,
    options: ContentCaptureOptions,
    validate: bool = True,
) -> tuple[list[ToolCallCapture], list[dict[str, Any]]]:
    """Extract tool calls from Anthropic response.

    Args:
        response: Response object from Anthropic
        tools_schema: Tool definitions from the request (for validation)
        options: Content capture configuration
        validate: Whether to validate arguments against schema

    Returns:
        Tuple of:
        - List of ToolCallCapture objects
        - List of large content payloads for server storage
    """
    captures: list[ToolCallCapture] = []
    large_content_payloads: list[dict[str, Any]] = []

    # Get content blocks from response
    content_blocks = getattr(response, "content", [])

    idx = 0
    for block in content_blocks:
        block_type = getattr(block, "type", None)
        if block_type != "tool_use":
            continue

        tc_id = getattr(block, "id", "")
        name = getattr(block, "name", "")
        arguments = getattr(block, "input", {})

        # For Anthropic, arguments come as dict directly
        arguments_raw = json.dumps(arguments, ensure_ascii=False) if arguments else "{}"

        capture, payloads = _process_tool_call(
            tc_id=tc_id,
            name=name,
            arguments_raw=arguments_raw,
            arguments_parsed=arguments,  # Already parsed for Anthropic
            index=idx,
            tools_schema=tools_schema,
            options=options,
            validate=validate,
        )
        captures.append(capture)
        large_content_payloads.extend(payloads)
        idx += 1

    return captures, large_content_payloads


def _process_tool_call(
    tc_id: str,
    name: str,
    arguments_raw: str,
    index: int,
    tools_schema: list[dict[str, Any]] | None,
    options: ContentCaptureOptions,
    validate: bool,
    arguments_parsed: dict[str, Any] | None = None,
) -> tuple[ToolCallCapture, list[dict[str, Any]]]:
    """Process a single tool call into ToolCallCapture.

    Args:
        tc_id: Tool call ID
        name: Tool/function name
        arguments_raw: Raw JSON string of arguments
        index: Index in tool_calls array
        tools_schema: Tool definitions for validation
        options: Content capture configuration
        validate: Whether to validate against schema
        arguments_parsed: Pre-parsed arguments (for Anthropic)

    Returns:
        Tuple of (ToolCallCapture, list of large content payloads)
    """
    large_content_payloads: list[dict[str, Any]] = []

    # Parse arguments if not already parsed
    arguments: dict[str, Any] | None = arguments_parsed
    if arguments is None:
        try:
            arguments = json.loads(arguments_raw) if arguments_raw else None
        except json.JSONDecodeError:
            arguments = None

    # Validate against schema
    validation_errors: list[ToolCallValidationError] = []
    if validate and arguments is not None and tools_schema:
        schema = find_tool_schema(name, tools_schema)
        if schema:
            validation_errors = validate_against_schema(arguments, schema)

    # Process arguments for storage
    processed_args = None
    if arguments is not None:
        processed, payload = process_content(arguments, options)
        processed_args = processed
        if payload:
            large_content_payloads.append(payload)

    # Process raw arguments for storage
    processed_raw, raw_payload = process_string_content(arguments_raw, options)
    if raw_payload:
        large_content_payloads.append(raw_payload)

    return ToolCallCapture(
        id=tc_id,
        name=name,
        arguments=processed_args,  # type: ignore
        arguments_raw=processed_raw,
        validation_errors=validation_errors if validation_errors else None,
        is_valid=len(validation_errors) == 0,
        index=index,
    ), large_content_payloads


class ToolCallStreamAccumulator:
    """Accumulates tool call data from streaming chunks.

    Used by MeteredAsyncStream and MeteredSyncStream to reconstruct
    full tool call information from streaming deltas.
    """

    def __init__(self) -> None:
        """Initialize the accumulator."""
        # Dict of tool call index -> partial data
        self._tool_calls: dict[int, dict[str, Any]] = {}

    def process_openai_chunk(self, chunk: Any) -> None:
        """Process an OpenAI streaming chunk for tool calls.

        Handles both Chat Completions streaming and Responses API streaming.

        Args:
            chunk: Streaming chunk from OpenAI
        """
        # Handle Chat Completions streaming format
        choices = getattr(chunk, "choices", [])
        if choices:
            delta = getattr(choices[0], "delta", None)
            if delta:
                tool_calls = getattr(delta, "tool_calls", None)
                if tool_calls:
                    for tc in tool_calls:
                        index = getattr(tc, "index", 0)

                        if index not in self._tool_calls:
                            self._tool_calls[index] = {
                                "id": "",
                                "name": "",
                                "arguments": "",
                            }

                        # Accumulate ID
                        tc_id = getattr(tc, "id", None)
                        if tc_id:
                            self._tool_calls[index]["id"] = tc_id

                        # Accumulate function info
                        func = getattr(tc, "function", None)
                        if func:
                            name = getattr(func, "name", None)
                            if name:
                                self._tool_calls[index]["name"] = name
                            args = getattr(func, "arguments", None)
                            if args:
                                self._tool_calls[index]["arguments"] += args

        # Handle Responses API streaming format
        chunk_type = getattr(chunk, "type", "")
        if chunk_type == "response.function_call_arguments.delta":
            # Get call_id to identify which tool call
            call_id = getattr(chunk, "call_id", "")
            delta_args = getattr(chunk, "delta", "")

            # Find or create entry by call_id
            found = False
            for idx, tc_data in self._tool_calls.items():
                if tc_data.get("id") == call_id:
                    tc_data["arguments"] += delta_args
                    found = True
                    break

            if not found and call_id:
                # New tool call
                idx = len(self._tool_calls)
                self._tool_calls[idx] = {
                    "id": call_id,
                    "name": getattr(chunk, "name", ""),
                    "arguments": delta_args,
                }

        elif chunk_type == "response.function_call_arguments.done":
            # Final arguments for a tool call
            call_id = getattr(chunk, "call_id", "")
            name = getattr(chunk, "name", "")
            arguments = getattr(chunk, "arguments", "")

            # Find or create entry
            found = False
            for idx, tc_data in self._tool_calls.items():
                if tc_data.get("id") == call_id:
                    if name:
                        tc_data["name"] = name
                    # Arguments might already be accumulated from deltas
                    if not tc_data["arguments"]:
                        tc_data["arguments"] = arguments
                    found = True
                    break

            if not found and call_id:
                idx = len(self._tool_calls)
                self._tool_calls[idx] = {
                    "id": call_id,
                    "name": name,
                    "arguments": arguments,
                }

    def process_anthropic_chunk(self, chunk: Any) -> None:
        """Process an Anthropic streaming chunk for tool calls.

        Args:
            chunk: Streaming chunk from Anthropic
        """
        chunk_type = getattr(chunk, "type", "")

        if chunk_type == "content_block_start":
            content_block = getattr(chunk, "content_block", None)
            if content_block and getattr(content_block, "type", "") == "tool_use":
                index = getattr(chunk, "index", len(self._tool_calls))
                self._tool_calls[index] = {
                    "id": getattr(content_block, "id", ""),
                    "name": getattr(content_block, "name", ""),
                    "arguments": "",
                }

        elif chunk_type == "content_block_delta":
            delta = getattr(chunk, "delta", None)
            if delta and getattr(delta, "type", "") == "input_json_delta":
                index = getattr(chunk, "index", 0)
                if index in self._tool_calls:
                    partial_json = getattr(delta, "partial_json", "")
                    self._tool_calls[index]["arguments"] += partial_json

    def finalize(
        self,
        tools_schema: list[dict[str, Any]] | None,
        options: ContentCaptureOptions,
        validate: bool = True,
    ) -> tuple[list[ToolCallCapture], list[dict[str, Any]]]:
        """Finalize accumulated tool calls into ToolCallCapture objects.

        Args:
            tools_schema: Tool definitions for validation
            options: Content capture configuration
            validate: Whether to validate against schema

        Returns:
            Tuple of:
            - List of ToolCallCapture objects
            - List of large content payloads for server storage
        """
        captures: list[ToolCallCapture] = []
        large_content_payloads: list[dict[str, Any]] = []

        for index in sorted(self._tool_calls.keys()):
            tc_data = self._tool_calls[index]

            capture, payloads = _process_tool_call(
                tc_id=tc_data["id"],
                name=tc_data["name"],
                arguments_raw=tc_data["arguments"],
                index=index,
                tools_schema=tools_schema,
                options=options,
                validate=validate,
            )
            captures.append(capture)
            large_content_payloads.extend(payloads)

        return captures, large_content_payloads

    @property
    def has_tool_calls(self) -> bool:
        """Whether any tool calls have been accumulated."""
        return len(self._tool_calls) > 0

    def clear(self) -> None:
        """Clear accumulated tool calls."""
        self._tool_calls.clear()
