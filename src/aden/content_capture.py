"""
Content capture extraction for Layer 0.

Extracts request and response content from provider-specific formats
for OpenAI and Anthropic APIs.
"""

from typing import Any

from .content_storage import process_content, process_string_content
from .types import (
    ContentCapture,
    ContentCaptureOptions,
    ContentReference,
    MessageCapture,
    RequestParamsCapture,
    ToolSchemaCapture,
)


def extract_openai_request_content(
    params: dict[str, Any],
    options: ContentCaptureOptions,
) -> tuple[ContentCapture, list[dict[str, Any]]]:
    """Extract content from OpenAI request params.

    Handles both Chat Completions API and Responses API formats.

    Args:
        params: Request parameters passed to create()
        options: Content capture configuration

    Returns:
        Tuple of:
        - ContentCapture with extracted request content
        - List of large content payloads for server storage
    """
    large_content_payloads: list[dict[str, Any]] = []
    capture = ContentCapture()

    # Extract system prompt (from system param or first system message)
    if options.capture_system_prompt:
        # Check for explicit system param (Responses API style)
        system = params.get("system")
        if system:
            processed, payload = process_string_content(system, options)
            capture.system_prompt = processed
            if payload:
                large_content_payloads.append(payload)

    # Extract messages
    if options.capture_messages:
        messages = params.get("messages", [])
        if messages:
            captured_messages: list[MessageCapture] = []

            for msg in messages:
                if not isinstance(msg, dict):
                    continue

                role = msg.get("role", "")
                content = msg.get("content", "")

                # Handle system messages
                if role == "system" and options.capture_system_prompt:
                    if isinstance(content, str):
                        processed, payload = process_string_content(content, options)
                        capture.system_prompt = processed
                        if payload:
                            large_content_payloads.append(payload)
                    continue  # Don't add to messages list

                # Handle multimodal content (list of content parts)
                if isinstance(content, list):
                    text_parts: list[str] = []
                    for part in content:
                        if isinstance(part, dict):
                            part_type = part.get("type", "")
                            if part_type == "text":
                                text_parts.append(part.get("text", ""))
                            elif part_type == "image_url":
                                capture.has_images = True
                                url_obj = part.get("image_url", {})
                                url = url_obj.get("url", "") if isinstance(url_obj, dict) else ""
                                # Only capture URLs, not base64 data URIs
                                if url and not url.startswith("data:"):
                                    if capture.image_urls is None:
                                        capture.image_urls = []
                                    capture.image_urls.append(url)
                    content = "\n".join(text_parts)

                # Process message content
                if isinstance(content, str):
                    processed, payload = process_string_content(content, options)
                    if payload:
                        large_content_payloads.append(payload)
                else:
                    processed = None

                captured_messages.append(
                    MessageCapture(
                        role=role,
                        content=processed,
                        name=msg.get("name"),
                        tool_call_id=msg.get("tool_call_id"),
                    )
                )

            if captured_messages:
                capture.messages = captured_messages

    # Extract tools schema
    if options.capture_tools_schema:
        tools = params.get("tools", [])
        if tools:
            captured_tools: list[ToolSchemaCapture] = []

            for tool in tools:
                if not isinstance(tool, dict):
                    continue

                # OpenAI format: { type: "function", function: { name, description, parameters } }
                if tool.get("type") == "function":
                    func = tool.get("function", {})
                    if isinstance(func, dict):
                        schema = func.get("parameters", {})
                        processed, payload = process_content(schema, options)
                        if payload:
                            large_content_payloads.append(payload)

                        captured_tools.append(
                            ToolSchemaCapture(
                                name=func.get("name", ""),
                                description=func.get("description"),
                                parameters_schema=processed,  # type: ignore
                            )
                        )

            if captured_tools:
                capture.tools = captured_tools

    # Extract request parameters
    capture.params = RequestParamsCapture(
        temperature=params.get("temperature"),
        max_tokens=params.get("max_tokens") or params.get("max_completion_tokens"),
        top_p=params.get("top_p"),
        frequency_penalty=params.get("frequency_penalty"),
        presence_penalty=params.get("presence_penalty"),
        stop=params.get("stop"),
        seed=params.get("seed"),
    )

    return capture, large_content_payloads


def extract_openai_response_content(
    response: Any,
    capture: ContentCapture,
    options: ContentCaptureOptions,
) -> list[dict[str, Any]]:
    """Extract content from OpenAI response.

    Handles both Chat Completions API and Responses API formats.

    Args:
        response: Response object from OpenAI
        capture: ContentCapture to populate
        options: Content capture configuration

    Returns:
        List of large content payloads for server storage
    """
    large_content_payloads: list[dict[str, Any]] = []

    if not options.capture_response:
        return large_content_payloads

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
    capture.choice_count = len(choices) if choices else None

    if choices:
        first_choice = choices[0] if isinstance(choices[0], dict) else {}
        message = first_choice.get("message", {})
        content = message.get("content", "") if isinstance(message, dict) else ""

        capture.finish_reason = first_choice.get("finish_reason")

        if content:
            processed, payload = process_string_content(content, options)
            capture.response_content = processed
            if payload:
                large_content_payloads.append(payload)

    # Handle Responses API format (output array)
    output = data.get("output", [])
    if output and not choices:
        text_parts: list[str] = []
        for item in output:
            if isinstance(item, dict):
                item_type = item.get("type", "")
                if item_type == "message":
                    msg_content = item.get("content", [])
                    if isinstance(msg_content, list):
                        for part in msg_content:
                            if isinstance(part, dict) and part.get("type") == "output_text":
                                text_parts.append(part.get("text", ""))

        if text_parts:
            full_content = "\n".join(text_parts)
            processed, payload = process_string_content(full_content, options)
            capture.response_content = processed
            if payload:
                large_content_payloads.append(payload)

        # Get finish reason from status
        capture.finish_reason = data.get("status")

    return large_content_payloads


def extract_anthropic_request_content(
    params: dict[str, Any],
    options: ContentCaptureOptions,
) -> tuple[ContentCapture, list[dict[str, Any]]]:
    """Extract content from Anthropic request params.

    Args:
        params: Request parameters passed to messages.create()
        options: Content capture configuration

    Returns:
        Tuple of:
        - ContentCapture with extracted request content
        - List of large content payloads for server storage
    """
    large_content_payloads: list[dict[str, Any]] = []
    capture = ContentCapture()

    # Extract system prompt
    if options.capture_system_prompt:
        system = params.get("system")
        if system:
            # Anthropic system can be string or list of content blocks
            if isinstance(system, str):
                processed, payload = process_string_content(system, options)
                capture.system_prompt = processed
                if payload:
                    large_content_payloads.append(payload)
            elif isinstance(system, list):
                text_parts: list[str] = []
                for block in system:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                if text_parts:
                    system_text = "\n".join(text_parts)
                    processed, payload = process_string_content(system_text, options)
                    capture.system_prompt = processed
                    if payload:
                        large_content_payloads.append(payload)

    # Extract messages
    if options.capture_messages:
        messages = params.get("messages", [])
        if messages:
            captured_messages: list[MessageCapture] = []

            for msg in messages:
                if not isinstance(msg, dict):
                    continue

                role = msg.get("role", "")
                content = msg.get("content", "")

                # Handle Anthropic content blocks (list format)
                if isinstance(content, list):
                    text_parts: list[str] = []
                    for block in content:
                        if isinstance(block, dict):
                            block_type = block.get("type", "")
                            if block_type == "text":
                                text_parts.append(block.get("text", ""))
                            elif block_type == "image":
                                capture.has_images = True
                                source = block.get("source", {})
                                if isinstance(source, dict) and source.get("type") == "url":
                                    url = source.get("url", "")
                                    if url:
                                        if capture.image_urls is None:
                                            capture.image_urls = []
                                        capture.image_urls.append(url)
                            elif block_type == "tool_result":
                                # Tool result content
                                tool_content = block.get("content", "")
                                if isinstance(tool_content, str):
                                    text_parts.append(f"[tool_result: {tool_content}]")
                    content = "\n".join(text_parts)

                # Process message content
                if isinstance(content, str):
                    processed, payload = process_string_content(content, options)
                    if payload:
                        large_content_payloads.append(payload)
                else:
                    processed = None

                captured_messages.append(
                    MessageCapture(
                        role=role,
                        content=processed,
                    )
                )

            if captured_messages:
                capture.messages = captured_messages

    # Extract tools schema
    if options.capture_tools_schema:
        tools = params.get("tools", [])
        if tools:
            captured_tools: list[ToolSchemaCapture] = []

            for tool in tools:
                if not isinstance(tool, dict):
                    continue

                # Anthropic format: { name, description, input_schema }
                schema = tool.get("input_schema", {})
                processed, payload = process_content(schema, options)
                if payload:
                    large_content_payloads.append(payload)

                captured_tools.append(
                    ToolSchemaCapture(
                        name=tool.get("name", ""),
                        description=tool.get("description"),
                        parameters_schema=processed,  # type: ignore
                    )
                )

            if captured_tools:
                capture.tools = captured_tools

    # Extract request parameters
    capture.params = RequestParamsCapture(
        temperature=params.get("temperature"),
        max_tokens=params.get("max_tokens"),
        top_p=params.get("top_p"),
        top_k=params.get("top_k"),
        stop=params.get("stop_sequences"),
    )

    return capture, large_content_payloads


def extract_anthropic_response_content(
    response: Any,
    capture: ContentCapture,
    options: ContentCaptureOptions,
) -> list[dict[str, Any]]:
    """Extract content from Anthropic response.

    Args:
        response: Response object from Anthropic
        capture: ContentCapture to populate
        options: Content capture configuration

    Returns:
        List of large content payloads for server storage
    """
    large_content_payloads: list[dict[str, Any]] = []

    if not options.capture_response:
        return large_content_payloads

    # Get content blocks from response
    content_blocks = getattr(response, "content", [])
    capture.finish_reason = getattr(response, "stop_reason", None)

    # Extract text from content blocks
    text_parts: list[str] = []
    for block in content_blocks:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text = getattr(block, "text", "")
            if text:
                text_parts.append(text)

    if text_parts:
        full_content = "\n".join(text_parts)
        processed, payload = process_string_content(full_content, options)
        capture.response_content = processed
        if payload:
            large_content_payloads.append(payload)

    return large_content_payloads


class StreamContentAccumulator:
    """Accumulates response content from streaming chunks.

    Used by MeteredAsyncStream and MeteredSyncStream to reconstruct
    full response content from streaming deltas.
    """

    def __init__(self, max_bytes: int = 1024 * 1024):
        """Initialize the accumulator.

        Args:
            max_bytes: Maximum bytes to accumulate (default 1MB)
        """
        self._parts: list[str] = []
        self._max_bytes = max_bytes
        self._current_bytes = 0
        self._truncated = False

    def add(self, content: str) -> None:
        """Add a content chunk.

        Args:
            content: Content string to add
        """
        if self._truncated:
            return

        content_bytes = len(content.encode("utf-8"))
        if self._current_bytes + content_bytes > self._max_bytes:
            # Truncate to fit within limit
            remaining = self._max_bytes - self._current_bytes
            if remaining > 0:
                # Approximate truncation (may cut mid-character)
                self._parts.append(content[:remaining])
            self._truncated = True
            return

        self._parts.append(content)
        self._current_bytes += content_bytes

    def get_content(self) -> str:
        """Get accumulated content.

        Returns:
            Full accumulated content string
        """
        content = "".join(self._parts)
        if self._truncated:
            content += "...[truncated]"
        return content

    @property
    def is_truncated(self) -> bool:
        """Whether content was truncated due to size limit."""
        return self._truncated

    def finalize(
        self,
        capture: ContentCapture,
        options: ContentCaptureOptions,
    ) -> list[dict[str, Any]]:
        """Finalize accumulated content into ContentCapture.

        Args:
            capture: ContentCapture to populate
            options: Content capture configuration

        Returns:
            List of large content payloads for server storage
        """
        large_content_payloads: list[dict[str, Any]] = []

        content = self.get_content()
        if content:
            processed, payload = process_string_content(content, options)
            capture.response_content = processed
            if payload:
                large_content_payloads.append(payload)

        return large_content_payloads


# =============================================================================
# Gemini Content Extraction
# =============================================================================


def _extract_gemini_content_text(content: Any) -> str:
    """Extract text from Gemini content (string, Content, or list).

    Gemini accepts various content formats:
    - str: Simple text
    - Content: Protobuf-like object with parts
    - list: List of Content or str
    - dict: Dict with 'parts' key
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts = []
        for item in content:
            texts.append(_extract_gemini_content_text(item))
        return "\n".join(texts)

    if isinstance(content, dict):
        # Dict format: {"role": "user", "parts": [{"text": "..."}]}
        parts = content.get("parts", [])
        texts = []
        for part in parts:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict) and "text" in part:
                texts.append(part["text"])
        return "\n".join(texts)

    # Content object with parts attribute
    if hasattr(content, "parts"):
        texts = []
        for part in content.parts:
            if hasattr(part, "text"):
                texts.append(part.text)
            elif isinstance(part, str):
                texts.append(part)
        return "\n".join(texts)

    # Try text attribute directly
    if hasattr(content, "text"):
        return content.text

    return str(content)


def extract_gemini_request_content(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    options: ContentCaptureOptions,
) -> tuple[ContentCapture, list[dict[str, Any]]]:
    """Extract content from Gemini request.

    Gemini generate_content signature:
        generate_content(contents, generation_config=None, safety_settings=None,
                        tools=None, tool_config=None, system_instruction=None, ...)

    Args:
        args: Positional arguments to generate_content
        kwargs: Keyword arguments to generate_content
        options: Content capture configuration

    Returns:
        Tuple of:
        - ContentCapture with extracted request content
        - List of large content payloads for server storage
    """
    large_content_payloads: list[dict[str, Any]] = []
    capture = ContentCapture()

    # Extract contents (first positional arg or 'contents' kwarg)
    contents = args[0] if args else kwargs.get("contents")

    # Extract system instruction
    if options.capture_system_prompt:
        system_instruction = kwargs.get("system_instruction")
        if system_instruction:
            system_text = _extract_gemini_content_text(system_instruction)
            if system_text:
                processed, payload = process_string_content(system_text, options)
                capture.system_prompt = processed
                if payload:
                    large_content_payloads.append(payload)

    # Extract messages from contents
    if options.capture_messages and contents:
        captured_messages: list[MessageCapture] = []

        if isinstance(contents, str):
            # Simple string content
            processed, payload = process_string_content(contents, options)
            if payload:
                large_content_payloads.append(payload)
            captured_messages.append(MessageCapture(role="user", content=processed))

        elif isinstance(contents, list):
            # List of content items
            for item in contents:
                if isinstance(item, str):
                    processed, payload = process_string_content(item, options)
                    if payload:
                        large_content_payloads.append(payload)
                    captured_messages.append(MessageCapture(role="user", content=processed))
                elif isinstance(item, dict):
                    role = item.get("role", "user")
                    text = _extract_gemini_content_text(item)
                    processed, payload = process_string_content(text, options)
                    if payload:
                        large_content_payloads.append(payload)
                    captured_messages.append(MessageCapture(role=role, content=processed))
                elif hasattr(item, "role") and hasattr(item, "parts"):
                    # Content object
                    role = item.role if hasattr(item, "role") else "user"
                    text = _extract_gemini_content_text(item)
                    processed, payload = process_string_content(text, options)
                    if payload:
                        large_content_payloads.append(payload)
                    captured_messages.append(MessageCapture(role=role, content=processed))
        else:
            # Single Content object
            text = _extract_gemini_content_text(contents)
            processed, payload = process_string_content(text, options)
            if payload:
                large_content_payloads.append(payload)
            captured_messages.append(MessageCapture(role="user", content=processed))

        if captured_messages:
            capture.messages = captured_messages

    # Extract tools schema
    if options.capture_tools_schema:
        tools = kwargs.get("tools")
        if tools:
            captured_tools: list[ToolSchemaCapture] = []

            tool_list = tools if isinstance(tools, list) else [tools]
            for tool in tool_list:
                # Gemini tools can be function declarations or Tool objects
                if hasattr(tool, "function_declarations"):
                    for func_decl in tool.function_declarations:
                        name = getattr(func_decl, "name", "unknown")
                        description = getattr(func_decl, "description", None)
                        params = getattr(func_decl, "parameters", None)

                        schema = None
                        if params:
                            # Convert to dict if possible
                            if hasattr(params, "to_dict"):
                                schema = params.to_dict()
                            elif hasattr(params, "__dict__"):
                                schema = params.__dict__
                            else:
                                schema = {"type": "object"}

                        captured_tools.append(
                            ToolSchemaCapture(
                                name=name,
                                description=description,
                                parameters_schema=schema,
                            )
                        )
                elif isinstance(tool, dict):
                    # Dict format
                    func_decls = tool.get("function_declarations", [])
                    for func_decl in func_decls:
                        captured_tools.append(
                            ToolSchemaCapture(
                                name=func_decl.get("name", "unknown"),
                                description=func_decl.get("description"),
                                parameters_schema=func_decl.get("parameters"),
                            )
                        )

            if captured_tools:
                capture.tools = captured_tools

    # Extract generation config parameters
    generation_config = kwargs.get("generation_config")
    if generation_config:
        if isinstance(generation_config, dict):
            capture.params = RequestParamsCapture(
                temperature=generation_config.get("temperature"),
                max_tokens=generation_config.get("max_output_tokens"),
                top_p=generation_config.get("top_p"),
                top_k=generation_config.get("top_k"),
                stop=generation_config.get("stop_sequences"),
            )
        elif hasattr(generation_config, "temperature"):
            capture.params = RequestParamsCapture(
                temperature=getattr(generation_config, "temperature", None),
                max_tokens=getattr(generation_config, "max_output_tokens", None),
                top_p=getattr(generation_config, "top_p", None),
                top_k=getattr(generation_config, "top_k", None),
                stop=getattr(generation_config, "stop_sequences", None),
            )

    return capture, large_content_payloads


def extract_gemini_response_content(
    response: Any,
    capture: ContentCapture,
    options: ContentCaptureOptions,
) -> list[dict[str, Any]]:
    """Extract content from Gemini response.

    Args:
        response: Response object from Gemini
        capture: ContentCapture to populate
        options: Content capture configuration

    Returns:
        List of large content payloads for server storage
    """
    large_content_payloads: list[dict[str, Any]] = []

    if not options.capture_response:
        return large_content_payloads

    # Try to get text from response
    text = None

    # GenerateContentResponse has .text property
    if hasattr(response, "text"):
        try:
            text = response.text
        except Exception:
            pass

    # Or extract from candidates
    if not text and hasattr(response, "candidates"):
        candidates = response.candidates
        if candidates:
            first_candidate = candidates[0]
            if hasattr(first_candidate, "content"):
                text = _extract_gemini_content_text(first_candidate.content)

            # Get finish reason
            if hasattr(first_candidate, "finish_reason"):
                finish_reason = first_candidate.finish_reason
                # Convert enum to string if needed
                if hasattr(finish_reason, "name"):
                    capture.finish_reason = finish_reason.name.lower()
                else:
                    capture.finish_reason = str(finish_reason)

        capture.choice_count = len(candidates) if candidates else None

    if text:
        processed, payload = process_string_content(text, options)
        capture.response_content = processed
        if payload:
            large_content_payloads.append(payload)

    return large_content_payloads


# -----------------------------------------------------------------------------
# Gemini gRPC Content Capture
# -----------------------------------------------------------------------------


def _extract_grpc_content_text(content: Any) -> str:
    """Extract text from a gRPC Content protobuf message."""
    if content is None:
        return ""

    # Content has 'parts' field
    parts = getattr(content, "parts", [])
    text_parts = []

    for part in parts:
        # Part has 'text' field
        if hasattr(part, "text"):
            text_parts.append(part.text)

    return " ".join(text_parts).strip()


def extract_gemini_grpc_request_content(
    request: Any,
    options: ContentCaptureOptions,
) -> tuple[ContentCapture, list[dict[str, Any]]]:
    """Extract content from Gemini gRPC GenerateContentRequest.

    The gRPC client uses protobuf request objects with fields:
        - model: str
        - contents: list of Content protobuf
        - system_instruction: Content protobuf
        - generation_config: GenerationConfig protobuf
        - tools: list of Tool protobuf

    Args:
        request: GenerateContentRequest protobuf
        options: Content capture configuration

    Returns:
        Tuple of:
        - ContentCapture with extracted request content
        - List of large content payloads for server storage
    """
    large_content_payloads: list[dict[str, Any]] = []
    capture = ContentCapture()

    if request is None:
        return capture, large_content_payloads

    # Extract system instruction
    if options.capture_system_prompt:
        system_instruction = getattr(request, "system_instruction", None)
        if system_instruction:
            system_text = _extract_grpc_content_text(system_instruction)
            if system_text:
                processed, payload = process_string_content(system_text, options)
                capture.system_prompt = processed
                if payload:
                    large_content_payloads.append(payload)

    # Extract messages from contents
    if options.capture_messages:
        contents = getattr(request, "contents", [])
        if contents:
            captured_messages: list[MessageCapture] = []

            for content in contents:
                role = getattr(content, "role", "user") or "user"
                text = _extract_grpc_content_text(content)
                processed, payload = process_string_content(text, options)
                if payload:
                    large_content_payloads.append(payload)
                captured_messages.append(MessageCapture(role=role, content=processed))

            if captured_messages:
                capture.messages = captured_messages

    # Extract tools schema
    if options.capture_tools_schema:
        tools = getattr(request, "tools", [])
        if tools:
            captured_tools: list[ToolSchemaCapture] = []

            for tool in tools:
                # Tool has function_declarations field
                func_declarations = getattr(tool, "function_declarations", [])
                for func_decl in func_declarations:
                    name = getattr(func_decl, "name", "unknown")
                    description = getattr(func_decl, "description", None)
                    params = getattr(func_decl, "parameters", None)

                    schema = None
                    if params:
                        # Convert protobuf to dict if possible
                        if hasattr(params, "to_dict"):
                            schema = params.to_dict()
                        elif hasattr(params, "_pb"):
                            # Some protobuf wrappers have _pb attribute
                            try:
                                from google.protobuf.json_format import MessageToDict
                                schema = MessageToDict(params._pb)
                            except Exception:
                                schema = {"type": "object"}
                        else:
                            schema = {"type": "object"}

                    captured_tools.append(
                        ToolSchemaCapture(
                            name=name,
                            description=description,
                            parameters_schema=schema,
                        )
                    )

            if captured_tools:
                capture.tools = captured_tools

    # Extract generation config parameters
    generation_config = getattr(request, "generation_config", None)
    if generation_config:
        capture.params = RequestParamsCapture(
            temperature=getattr(generation_config, "temperature", None),
            max_tokens=getattr(generation_config, "max_output_tokens", None),
            top_p=getattr(generation_config, "top_p", None),
            top_k=getattr(generation_config, "top_k", None),
            stop=list(getattr(generation_config, "stop_sequences", [])) or None,
        )

    return capture, large_content_payloads


def extract_gemini_grpc_response_content(
    response: Any,
    capture: ContentCapture,
    options: ContentCaptureOptions,
) -> list[dict[str, Any]]:
    """Extract content from Gemini gRPC GenerateContentResponse.

    Args:
        response: GenerateContentResponse protobuf
        capture: ContentCapture to populate
        options: Content capture configuration

    Returns:
        List of large content payloads for server storage
    """
    large_content_payloads: list[dict[str, Any]] = []

    if not options.capture_response:
        return large_content_payloads

    if response is None:
        return large_content_payloads

    # Try to get text from response
    text = None

    # GenerateContentResponse has candidates
    candidates = getattr(response, "candidates", [])
    if candidates:
        first_candidate = candidates[0]
        content = getattr(first_candidate, "content", None)
        if content:
            text = _extract_grpc_content_text(content)

        # Get finish reason
        finish_reason = getattr(first_candidate, "finish_reason", None)
        if finish_reason:
            # Convert enum to string
            if hasattr(finish_reason, "name"):
                capture.finish_reason = finish_reason.name.lower()
            else:
                capture.finish_reason = str(finish_reason)

        capture.choice_count = len(candidates)

    if text:
        processed, payload = process_string_content(text, options)
        capture.response_content = processed
        if payload:
            large_content_payloads.append(payload)

    return large_content_payloads
