"""
Anthropic SDK instrumentation.

This module provides global instrumentation for the Anthropic SDK by patching
the client prototypes, so all instances are automatically metered.
"""

import asyncio
import logging
import time
from functools import wraps
from typing import Any, AsyncIterator, Callable, Iterator
from uuid import uuid4

from datetime import datetime

from .call_stack import CallStackInfo, capture_call_stack
from .content_capture import (
    StreamContentAccumulator,
    extract_anthropic_request_content,
    extract_anthropic_response_content,
)
from .large_content import store_large_content
from .normalize import normalize_anthropic_usage
from .tool_capture import (
    ToolCallStreamAccumulator,
    extract_anthropic_tool_calls,
)
from .types import (
    BeforeRequestAction,
    BeforeRequestContext,
    BeforeRequestResult,
    ContentCapture,
    ContentCaptureOptions,
    MeterOptions,
    MetricEvent,
    NormalizedUsage,
    RequestCancelledError,
    ToolCallCapture,
)

logger = logging.getLogger("aden")

# Module-level state
_is_instrumented = False
_global_options: MeterOptions | None = None

# Store original methods for uninstrumentation
_original_messages_create: Callable[..., Any] | None = None
_original_async_messages_create: Callable[..., Any] | None = None


def _get_anthropic_classes(options: MeterOptions) -> tuple[Any, Any] | None:
    """Get Anthropic and AsyncAnthropic classes from options or auto-import."""
    if options.sdks:
        Anthropic = options.sdks.Anthropic
        AsyncAnthropic = options.sdks.AsyncAnthropic
        if Anthropic or AsyncAnthropic:
            return (Anthropic, AsyncAnthropic)

    # Try auto-import
    try:
        from anthropic import Anthropic, AsyncAnthropic
        return (Anthropic, AsyncAnthropic)
    except ImportError:
        return None


def _extract_request_id(response: Any) -> str | None:
    """Extracts request ID from Anthropic response (uses 'id' field)."""
    if response is None:
        return None
    # Anthropic uses 'id' for message ID
    if hasattr(response, "id"):
        return response.id
    if isinstance(response, dict):
        return response.get("id")
    return None


def _extract_tool_calls(response: Any) -> tuple[int, str | None]:
    """Extracts tool call count and names from Anthropic response (tool_use content blocks).

    Returns:
        Tuple of (tool_call_count, tool_names_comma_separated)
    """
    tool_names: list[str] = []

    if response is None:
        return (0, None)

    # Get content array
    content = None
    if hasattr(response, "content"):
        content = response.content
    elif isinstance(response, dict):
        content = response.get("content")

    if not isinstance(content, list):
        return (0, None)

    # Anthropic uses content array with type: "tool_use"
    for item in content:
        if isinstance(item, dict):
            if item.get("type") == "tool_use":
                name = item.get("name")
                if name:
                    tool_names.append(name)
        elif hasattr(item, "type") and item.type == "tool_use":
            name = getattr(item, "name", None)
            if name:
                tool_names.append(name)

    count = len(tool_names)
    names = ",".join(tool_names) if tool_names else None
    return (count, names)


def _build_metric_event(
    trace_id: str,
    span_id: str,
    model: str,
    stream: bool,
    latency_ms: float,
    usage: NormalizedUsage | None,
    request_id: str | None = None,
    tool_call_count: int | None = None,
    tool_names: str | None = None,
    error: str | None = None,
    metadata: dict[str, Any] | None = None,
    stack_info: CallStackInfo | None = None,
    content_capture: ContentCapture | None = None,
    tool_calls_captured: list[ToolCallCapture] | None = None,
) -> MetricEvent:
    """Builds a MetricEvent for Anthropic with flat fields."""
    # Calculate tool validation errors count
    tool_validation_errors_count = None
    if tool_calls_captured:
        errors = sum(1 for tc in tool_calls_captured if not tc.is_valid)
        tool_validation_errors_count = errors if errors > 0 else None

    return MetricEvent(
        trace_id=trace_id,
        span_id=span_id,
        provider="anthropic",
        model=model,
        stream=stream,
        timestamp=datetime.now().isoformat(),
        latency_ms=latency_ms,
        request_id=request_id,
        error=error,
        # Flatten usage
        input_tokens=usage.input_tokens if usage else 0,
        output_tokens=usage.output_tokens if usage else 0,
        total_tokens=usage.total_tokens if usage else 0,
        cached_tokens=usage.cached_tokens if usage else 0,
        reasoning_tokens=usage.reasoning_tokens if usage else 0,
        # Flatten tool calls (summary)
        tool_call_count=tool_call_count if tool_call_count and tool_call_count > 0 else None,
        tool_names=tool_names,
        metadata=metadata,
        # Call stack info
        call_site_file=stack_info.call_site_file if stack_info else None,
        call_site_line=stack_info.call_site_line if stack_info else None,
        call_site_function=stack_info.call_site_function if stack_info else None,
        call_stack=stack_info.call_stack if stack_info else None,
        agent_stack=stack_info.agent_stack if stack_info else None,
        # Layer 0: Content Capture
        content_capture=content_capture,
        # Layer 6: Tool Call Deep Inspection
        tool_calls_captured=tool_calls_captured,
        tool_validation_errors_count=tool_validation_errors_count,
    )


async def _emit_metric(event: MetricEvent, options: MeterOptions) -> None:
    """Emits a metric, handling async/sync emitters."""
    try:
        result = options.emit_metric(event)
        if asyncio.iscoroutine(result):
            await result
    except Exception as e:
        if options.on_emit_error:
            options.on_emit_error(event, e)
        else:
            logger.error(f"Error emitting metric (trace_id={event.trace_id}): {e}")


def _emit_metric_sync(event: MetricEvent, options: MeterOptions) -> None:
    """Emits a metric synchronously."""
    try:
        result = options.emit_metric(event)
        if asyncio.iscoroutine(result):
            # Close the unawaited coroutine to prevent warnings
            result.close()
    except Exception as e:
        if options.on_emit_error:
            options.on_emit_error(event, e)
        else:
            logger.error(f"Error emitting metric (trace_id={event.trace_id}): {e}")


async def _execute_before_request_hook(
    params: dict[str, Any],
    context: BeforeRequestContext,
    options: MeterOptions,
) -> BeforeRequestResult:
    """Executes the beforeRequest hook if provided."""
    if options.before_request is None:
        return BeforeRequestResult.proceed()

    result = options.before_request(params, context)
    if asyncio.iscoroutine(result):
        result = await result

    return result


def _execute_before_request_hook_sync(
    params: dict[str, Any],
    context: BeforeRequestContext,
    options: MeterOptions,
) -> BeforeRequestResult:
    """Executes the beforeRequest hook synchronously."""
    if options.before_request is None:
        return BeforeRequestResult.proceed()

    result = options.before_request(params, context)
    if asyncio.iscoroutine(result):
        # Close the unawaited coroutine to prevent warnings
        result.close()
        return BeforeRequestResult.proceed()

    return result


async def _handle_before_request_result(
    result: BeforeRequestResult,
    params: dict[str, Any],
    context: BeforeRequestContext,
) -> dict[str, Any]:
    """Handle the before request result, returning potentially modified params."""
    if result.action == BeforeRequestAction.CANCEL:
        raise RequestCancelledError(result.reason, context)

    if result.action == BeforeRequestAction.THROTTLE:
        await asyncio.sleep(result.delay_ms / 1000)
        return params

    if result.action == BeforeRequestAction.DEGRADE:
        if result.delay_ms > 0:
            await asyncio.sleep(result.delay_ms / 1000)
        return {**params, "model": result.to_model}

    if result.action == BeforeRequestAction.ALERT:
        if result.delay_ms > 0:
            await asyncio.sleep(result.delay_ms / 1000)
        # Alert was already triggered by the hook
        return params

    return params


def _handle_before_request_result_sync(
    result: BeforeRequestResult,
    params: dict[str, Any],
    context: BeforeRequestContext,
) -> dict[str, Any]:
    """Handle the before request result synchronously."""
    if result.action == BeforeRequestAction.CANCEL:
        raise RequestCancelledError(result.reason, context)

    if result.action == BeforeRequestAction.THROTTLE:
        time.sleep(result.delay_ms / 1000)
        return params

    if result.action == BeforeRequestAction.DEGRADE:
        if result.delay_ms > 0:
            time.sleep(result.delay_ms / 1000)
        return {**params, "model": result.to_model}

    if result.action == BeforeRequestAction.ALERT:
        if result.delay_ms > 0:
            time.sleep(result.delay_ms / 1000)
        return params

    return params


class MeteredAsyncStream:
    """Wraps an async Anthropic stream to meter it.

    Anthropic streaming events:
    - message_start: Contains initial message with ID
    - content_block_start: Start of content block (may include tool_use)
    - content_block_delta: Streaming content delta
    - content_block_stop: End of content block
    - message_delta: Final usage statistics
    - message_stop: Stream complete
    """

    def __init__(
        self,
        stream: AsyncIterator[Any],
        trace_id: str,
        span_id: str,
        model: str,
        t0: float,
        options: MeterOptions,
        stack_info: CallStackInfo | None = None,
        content_capture: ContentCapture | None = None,
        tools_schema: list[dict[str, Any]] | None = None,
    ):
        self._stream = stream
        self._trace_id = trace_id
        self._span_id = span_id
        self._model = model
        self._t0 = t0
        self._options = options
        self._stack_info = stack_info
        self._final_usage: NormalizedUsage | None = None
        self._input_tokens: int = 0  # Track input tokens from message_start
        self._request_id: str | None = None
        self._tool_names: list[str] = []
        self._done = False
        self._error: str | None = None
        # Layer 0: Content Capture
        self._content_capture = content_capture
        self._content_accumulator: StreamContentAccumulator | None = None
        if options.capture_content and content_capture:
            capture_options = options.content_capture_options or ContentCaptureOptions()
            self._content_accumulator = StreamContentAccumulator(
                max_bytes=capture_options.max_content_bytes * 4
            )
        # Layer 6: Tool Call Deep Inspection
        self._tools_schema = tools_schema
        self._tool_call_accumulator: ToolCallStreamAccumulator | None = None
        if options.capture_tool_calls:
            self._tool_call_accumulator = ToolCallStreamAccumulator()

    def __aiter__(self) -> "MeteredAsyncStream":
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._stream.__anext__()

            if hasattr(chunk, "type"):
                chunk_type = chunk.type

                # message_start: Contains the initial message object
                if chunk_type == "message_start":
                    message = getattr(chunk, "message", None)
                    if message:
                        self._request_id = _extract_request_id(message)
                        # Capture input_tokens from message_start
                        usage = getattr(message, "usage", None)
                        if usage:
                            self._input_tokens = getattr(usage, "input_tokens", 0) or 0

                # content_block_start: May contain tool_use
                elif chunk_type == "content_block_start":
                    if self._options.track_tool_calls:
                        content_block = getattr(chunk, "content_block", None)
                        if content_block:
                            block_type = getattr(content_block, "type", None)
                            if block_type == "tool_use":
                                name = getattr(content_block, "name", None)
                                if name:
                                    self._tool_names.append(name)
                    # Layer 6: Accumulate tool calls
                    if self._tool_call_accumulator:
                        self._tool_call_accumulator.process_anthropic_chunk(chunk)

                # content_block_delta: Streaming content
                elif chunk_type == "content_block_delta":
                    # Layer 0: Accumulate text content
                    if self._content_accumulator:
                        delta = getattr(chunk, "delta", None)
                        if delta and getattr(delta, "type", "") == "text_delta":
                            text = getattr(delta, "text", "")
                            if text:
                                self._content_accumulator.add(text)
                    # Layer 6: Accumulate tool call arguments
                    if self._tool_call_accumulator:
                        self._tool_call_accumulator.process_anthropic_chunk(chunk)

                # message_delta: Contains final usage (output_tokens)
                elif chunk_type == "message_delta":
                    usage = getattr(chunk, "usage", None)
                    if usage:
                        output_tokens = getattr(usage, "output_tokens", 0) or 0
                        self._final_usage = NormalizedUsage(
                            input_tokens=self._input_tokens,
                            output_tokens=output_tokens,
                            total_tokens=self._input_tokens + output_tokens,
                            cached_tokens=0,  # Cache info comes from message_start if present
                            reasoning_tokens=0,
                            accepted_prediction_tokens=0,
                            rejected_prediction_tokens=0,
                        )

            return chunk

        except StopAsyncIteration:
            if not self._done:
                self._done = True
                await self._emit_final_metric()
            raise
        except Exception as e:
            if not self._done:
                self._done = True
                self._error = str(e)
                await self._emit_final_metric()
            raise

    async def _emit_final_metric(self) -> None:
        """Emit the final metric when stream ends."""
        tool_count = len(self._tool_names) if self._tool_names else None
        tool_names_str = ",".join(self._tool_names) if self._tool_names else None

        # Finalize Layer 0: Content Capture
        content_capture = self._content_capture
        if content_capture and self._content_accumulator:
            capture_options = self._options.content_capture_options or ContentCaptureOptions()
            self._content_accumulator.finalize(content_capture, capture_options)

        # Finalize Layer 6: Tool Call Deep Inspection
        tool_calls_captured: list[ToolCallCapture] | None = None
        if self._tool_call_accumulator and self._tool_call_accumulator.has_tool_calls:
            capture_options = self._options.content_capture_options or ContentCaptureOptions()
            tool_calls_captured, _ = self._tool_call_accumulator.finalize(
                self._tools_schema,
                capture_options,
                validate=self._options.validate_tool_schemas,
            )

        event = _build_metric_event(
            trace_id=self._trace_id,
            span_id=self._span_id,
            model=self._model,
            stream=True,
            latency_ms=(time.time() - self._t0) * 1000,
            usage=self._final_usage,
            request_id=self._request_id,
            tool_call_count=tool_count,
            tool_names=tool_names_str,
            error=self._error,
            stack_info=self._stack_info,
            content_capture=content_capture,
            tool_calls_captured=tool_calls_captured,
        )
        await _emit_metric(event, self._options)


class MeteredSyncStream:
    """Wraps a sync Anthropic stream to meter it."""

    def __init__(
        self,
        stream: Iterator[Any],
        trace_id: str,
        span_id: str,
        model: str,
        t0: float,
        options: MeterOptions,
        stack_info: CallStackInfo | None = None,
        content_capture: ContentCapture | None = None,
        tools_schema: list[dict[str, Any]] | None = None,
    ):
        self._stream = stream
        self._trace_id = trace_id
        self._span_id = span_id
        self._model = model
        self._t0 = t0
        self._options = options
        self._stack_info = stack_info
        self._final_usage: NormalizedUsage | None = None
        self._input_tokens: int = 0
        self._request_id: str | None = None
        self._tool_names: list[str] = []
        self._done = False
        self._error: str | None = None
        # Layer 0: Content Capture
        self._content_capture = content_capture
        self._content_accumulator: StreamContentAccumulator | None = None
        if options.capture_content and content_capture:
            capture_options = options.content_capture_options or ContentCaptureOptions()
            self._content_accumulator = StreamContentAccumulator(
                max_bytes=capture_options.max_content_bytes * 4
            )
        # Layer 6: Tool Call Deep Inspection
        self._tools_schema = tools_schema
        self._tool_call_accumulator: ToolCallStreamAccumulator | None = None
        if options.capture_tool_calls:
            self._tool_call_accumulator = ToolCallStreamAccumulator()

    def __iter__(self) -> "MeteredSyncStream":
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._stream)

            if hasattr(chunk, "type"):
                chunk_type = chunk.type

                if chunk_type == "message_start":
                    message = getattr(chunk, "message", None)
                    if message:
                        self._request_id = _extract_request_id(message)
                        usage = getattr(message, "usage", None)
                        if usage:
                            self._input_tokens = getattr(usage, "input_tokens", 0) or 0

                elif chunk_type == "content_block_start":
                    if self._options.track_tool_calls:
                        content_block = getattr(chunk, "content_block", None)
                        if content_block:
                            block_type = getattr(content_block, "type", None)
                            if block_type == "tool_use":
                                name = getattr(content_block, "name", None)
                                if name:
                                    self._tool_names.append(name)
                    # Layer 6: Accumulate tool calls
                    if self._tool_call_accumulator:
                        self._tool_call_accumulator.process_anthropic_chunk(chunk)

                elif chunk_type == "content_block_delta":
                    # Layer 0: Accumulate text content
                    if self._content_accumulator:
                        delta = getattr(chunk, "delta", None)
                        if delta and getattr(delta, "type", "") == "text_delta":
                            text = getattr(delta, "text", "")
                            if text:
                                self._content_accumulator.add(text)
                    # Layer 6: Accumulate tool call arguments
                    if self._tool_call_accumulator:
                        self._tool_call_accumulator.process_anthropic_chunk(chunk)

                elif chunk_type == "message_delta":
                    usage = getattr(chunk, "usage", None)
                    if usage:
                        output_tokens = getattr(usage, "output_tokens", 0) or 0
                        self._final_usage = NormalizedUsage(
                            input_tokens=self._input_tokens,
                            output_tokens=output_tokens,
                            total_tokens=self._input_tokens + output_tokens,
                            cached_tokens=0,
                            reasoning_tokens=0,
                            accepted_prediction_tokens=0,
                            rejected_prediction_tokens=0,
                        )

            return chunk

        except StopIteration:
            if not self._done:
                self._done = True
                self._emit_final_metric()
            raise
        except Exception as e:
            if not self._done:
                self._done = True
                self._error = str(e)
                self._emit_final_metric()
            raise

    def _emit_final_metric(self) -> None:
        """Emit the final metric when stream ends."""
        tool_count = len(self._tool_names) if self._tool_names else None
        tool_names_str = ",".join(self._tool_names) if self._tool_names else None

        # Finalize Layer 0: Content Capture
        content_capture = self._content_capture
        if content_capture and self._content_accumulator:
            capture_options = self._options.content_capture_options or ContentCaptureOptions()
            self._content_accumulator.finalize(content_capture, capture_options)

        # Finalize Layer 6: Tool Call Deep Inspection
        tool_calls_captured: list[ToolCallCapture] | None = None
        if self._tool_call_accumulator and self._tool_call_accumulator.has_tool_calls:
            capture_options = self._options.content_capture_options or ContentCaptureOptions()
            tool_calls_captured, _ = self._tool_call_accumulator.finalize(
                self._tools_schema,
                capture_options,
                validate=self._options.validate_tool_schemas,
            )

        event = _build_metric_event(
            trace_id=self._trace_id,
            span_id=self._span_id,
            model=self._model,
            stream=True,
            latency_ms=(time.time() - self._t0) * 1000,
            usage=self._final_usage,
            request_id=self._request_id,
            tool_call_count=tool_count,
            tool_names=tool_names_str,
            error=self._error,
            stack_info=self._stack_info,
            content_capture=content_capture,
            tool_calls_captured=tool_calls_captured,
        )
        _emit_metric_sync(event, self._options)


def _create_async_wrapper(
    original_fn: Callable[..., Any],
    get_options: Callable[[], MeterOptions | None],
) -> Callable[..., Any]:
    """Creates an async wrapper for Anthropic messages.create method."""

    @wraps(original_fn)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        options = get_options()
        if options is None:
            return await original_fn(self, *args, **kwargs)

        # Capture call stack before any async operations
        stack_info = capture_call_stack(skip_frames=3)

        # Extract params - Anthropic messages.create uses kwargs
        params = kwargs.copy()

        trace_id = options.generate_trace_id() if options.generate_trace_id else str(uuid4())
        span_id = options.generate_span_id() if options.generate_span_id else str(uuid4())
        model = params.get("model", "unknown")
        t0 = time.time()

        # Layer 0: Content Capture - extract request content before API call
        content_capture: ContentCapture | None = None
        large_content_payloads: list[dict[str, Any]] = []
        if options.capture_content:
            capture_options = options.content_capture_options or ContentCaptureOptions()
            content_capture, request_payloads = extract_anthropic_request_content(params, capture_options)
            if request_payloads:
                large_content_payloads.extend(request_payloads)

        # Store tools schema for Layer 6 validation
        tools_schema = params.get("tools") if options.capture_tool_calls else None

        # Execute beforeRequest hook
        context = BeforeRequestContext(
            model=model,
            stream=bool(params.get("stream")),
            span_id=span_id,
            trace_id=trace_id,
            timestamp=datetime.now(),
            metadata=options.request_metadata,
        )

        result = await _execute_before_request_hook(params, context, options)
        final_params = await _handle_before_request_result(result, params, context)

        # Update model if degraded
        model = final_params.get("model", model)

        try:
            response = await original_fn(self, **final_params)

            # Handle streaming
            if final_params.get("stream") and hasattr(response, "__aiter__"):
                # Store large content from request before streaming
                if large_content_payloads:
                    store_large_content(large_content_payloads)
                return MeteredAsyncStream(
                    response.__aiter__(), trace_id, span_id, model, t0, options,
                    stack_info=stack_info,
                    content_capture=content_capture,
                    tools_schema=tools_schema,
                )

            # Non-streaming response - extract tool calls (summary)
            tool_count, tool_names = (
                _extract_tool_calls(response) if options.track_tool_calls else (None, None)
            )

            # Layer 0: Extract response content
            if options.capture_content and content_capture:
                capture_options = options.content_capture_options or ContentCaptureOptions()
                response_payloads = extract_anthropic_response_content(response, content_capture, capture_options)
                if response_payloads:
                    large_content_payloads.extend(response_payloads)

            # Layer 6: Tool Call Deep Inspection
            tool_calls_captured: list[ToolCallCapture] | None = None
            if options.capture_tool_calls:
                capture_options = options.content_capture_options or ContentCaptureOptions()
                tool_calls_captured, tool_payloads = extract_anthropic_tool_calls(
                    response, tools_schema, capture_options,
                    validate=options.validate_tool_schemas,
                )
                if tool_payloads:
                    large_content_payloads.extend(tool_payloads)

            # Store all large content payloads
            if large_content_payloads:
                store_large_content(large_content_payloads)

            event = _build_metric_event(
                trace_id=trace_id,
                span_id=span_id,
                model=model,
                stream=False,
                latency_ms=(time.time() - t0) * 1000,
                usage=normalize_anthropic_usage(getattr(response, "usage", None)),
                request_id=_extract_request_id(response),
                tool_call_count=tool_count,
                tool_names=tool_names,
                stack_info=stack_info,
                content_capture=content_capture,
                tool_calls_captured=tool_calls_captured,
            )
            await _emit_metric(event, options)
            return response

        except Exception as e:
            event = _build_metric_event(
                trace_id=trace_id,
                span_id=span_id,
                model=model,
                stream=bool(params.get("stream")),
                latency_ms=(time.time() - t0) * 1000,
                usage=None,
                error=str(e),
                stack_info=stack_info,
                content_capture=content_capture,
            )
            await _emit_metric(event, options)
            raise

    return wrapper


def _create_sync_wrapper(
    original_fn: Callable[..., Any],
    get_options: Callable[[], MeterOptions | None],
) -> Callable[..., Any]:
    """Creates a sync wrapper for Anthropic messages.create method."""

    @wraps(original_fn)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        options = get_options()
        if options is None:
            return original_fn(self, *args, **kwargs)

        # Capture call stack
        stack_info = capture_call_stack(skip_frames=3)

        params = kwargs.copy()

        trace_id = options.generate_trace_id() if options.generate_trace_id else str(uuid4())
        span_id = options.generate_span_id() if options.generate_span_id else str(uuid4())
        model = params.get("model", "unknown")
        t0 = time.time()

        # Layer 0: Content Capture - extract request content before API call
        content_capture: ContentCapture | None = None
        large_content_payloads: list[dict[str, Any]] = []
        if options.capture_content:
            capture_options = options.content_capture_options or ContentCaptureOptions()
            content_capture, request_payloads = extract_anthropic_request_content(params, capture_options)
            if request_payloads:
                large_content_payloads.extend(request_payloads)

        # Store tools schema for Layer 6 validation
        tools_schema = params.get("tools") if options.capture_tool_calls else None

        context = BeforeRequestContext(
            model=model,
            stream=bool(params.get("stream")),
            span_id=span_id,
            trace_id=trace_id,
            timestamp=datetime.now(),
            metadata=options.request_metadata,
        )

        result = _execute_before_request_hook_sync(params, context, options)
        final_params = _handle_before_request_result_sync(result, params, context)
        model = final_params.get("model", model)

        try:
            response = original_fn(self, **final_params)

            if final_params.get("stream") and hasattr(response, "__iter__"):
                # Store large content from request before streaming
                if large_content_payloads:
                    store_large_content(large_content_payloads)
                return MeteredSyncStream(
                    iter(response), trace_id, span_id, model, t0, options,
                    stack_info=stack_info,
                    content_capture=content_capture,
                    tools_schema=tools_schema,
                )

            # Extract tool calls (summary for basic tracking)
            tool_count, tool_names = (
                _extract_tool_calls(response) if options.track_tool_calls else (None, None)
            )

            # Layer 0: Extract response content
            if options.capture_content and content_capture:
                capture_options = options.content_capture_options or ContentCaptureOptions()
                response_payloads = extract_anthropic_response_content(response, content_capture, capture_options)
                if response_payloads:
                    large_content_payloads.extend(response_payloads)

            # Layer 6: Tool Call Deep Inspection
            tool_calls_captured: list[ToolCallCapture] | None = None
            if options.capture_tool_calls:
                capture_options = options.content_capture_options or ContentCaptureOptions()
                tool_calls_captured, tool_payloads = extract_anthropic_tool_calls(
                    response, tools_schema, capture_options,
                    validate=options.validate_tool_schemas,
                )
                if tool_payloads:
                    large_content_payloads.extend(tool_payloads)

            # Store all large content payloads
            if large_content_payloads:
                store_large_content(large_content_payloads)

            event = _build_metric_event(
                trace_id=trace_id,
                span_id=span_id,
                model=model,
                stream=False,
                latency_ms=(time.time() - t0) * 1000,
                usage=normalize_anthropic_usage(getattr(response, "usage", None)),
                request_id=_extract_request_id(response),
                tool_call_count=tool_count,
                tool_names=tool_names,
                stack_info=stack_info,
                content_capture=content_capture,
                tool_calls_captured=tool_calls_captured,
            )
            _emit_metric_sync(event, options)
            return response

        except Exception as e:
            event = _build_metric_event(
                trace_id=trace_id,
                span_id=span_id,
                model=model,
                stream=bool(params.get("stream")),
                latency_ms=(time.time() - t0) * 1000,
                usage=None,
                error=str(e),
                stack_info=stack_info,
                content_capture=content_capture,
            )
            _emit_metric_sync(event, options)
            raise

    return wrapper


def instrument_anthropic(options: MeterOptions) -> bool:
    """
    Instrument the Anthropic SDK globally.

    Patches the Messages and AsyncMessages classes directly so all
    client instances are automatically metered.

    Args:
        options: Metering options including the metric emitter

    Returns:
        True if instrumentation succeeded, False if Anthropic SDK not available
    """
    global _is_instrumented, _global_options
    global _original_messages_create, _original_async_messages_create

    if _is_instrumented:
        return True

    # Check if Anthropic SDK is available and import the Messages classes
    try:
        from anthropic.resources import AsyncMessages, Messages
    except ImportError:
        logger.debug("Anthropic SDK not available, skipping instrumentation")
        return False

    _global_options = options

    def get_options() -> MeterOptions | None:
        return _global_options

    # Patch sync Messages.create
    try:
        _original_messages_create = Messages.create
        Messages.create = _create_sync_wrapper(_original_messages_create, get_options)
    except Exception as e:
        logger.warning(f"Failed to instrument sync Messages: {e}")

    # Patch async AsyncMessages.create
    try:
        _original_async_messages_create = AsyncMessages.create
        AsyncMessages.create = _create_async_wrapper(_original_async_messages_create, get_options)
    except Exception as e:
        logger.warning(f"Failed to instrument async Messages: {e}")

    _is_instrumented = True
    logger.info("[aden] Anthropic SDK instrumented")
    return True


def uninstrument_anthropic() -> None:
    """
    Remove Anthropic SDK instrumentation.

    Restores original methods on the Messages classes.
    """
    global _is_instrumented, _global_options
    global _original_messages_create, _original_async_messages_create

    if not _is_instrumented:
        return

    # Try to restore original methods
    try:
        from anthropic.resources import AsyncMessages, Messages

        if _original_messages_create:
            Messages.create = _original_messages_create

        if _original_async_messages_create:
            AsyncMessages.create = _original_async_messages_create

    except ImportError:
        pass

    _is_instrumented = False
    _global_options = None
    _original_messages_create = None
    _original_async_messages_create = None

    logger.info("[aden] Anthropic SDK uninstrumented")


def is_anthropic_instrumented() -> bool:
    """Check if Anthropic SDK is currently instrumented."""
    return _is_instrumented


def get_anthropic_options() -> MeterOptions | None:
    """Get current Anthropic instrumentation options."""
    return _global_options
