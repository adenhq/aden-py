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

from .normalize import normalize_anthropic_usage
from .types import (
    BeforeRequestAction,
    BeforeRequestContext,
    BeforeRequestResult,
    MeterOptions,
    MetricEvent,
    NormalizedUsage,
    RequestCancelledError,
    ToolCallMetric,
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


def _extract_tool_calls(response: Any) -> list[ToolCallMetric]:
    """Extracts tool call metrics from Anthropic response (tool_use content blocks)."""
    tool_calls: list[ToolCallMetric] = []

    if response is None:
        return tool_calls

    # Get content array
    content = None
    if hasattr(response, "content"):
        content = response.content
    elif isinstance(response, dict):
        content = response.get("content")

    if not isinstance(content, list):
        return tool_calls

    # Anthropic uses content array with type: "tool_use"
    for item in content:
        if isinstance(item, dict):
            if item.get("type") == "tool_use":
                tool_calls.append(
                    ToolCallMetric(type="function", name=item.get("name"))
                )
        elif hasattr(item, "type") and item.type == "tool_use":
            tool_calls.append(
                ToolCallMetric(type="function", name=getattr(item, "name", None))
            )

    return tool_calls


def _build_metric_event(
    trace_id: str,
    span_id: str,
    model: str,
    stream: bool,
    latency_ms: float,
    usage: NormalizedUsage | None,
    request_id: str | None = None,
    tool_calls: list[ToolCallMetric] | None = None,
    error: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> MetricEvent:
    """Builds a MetricEvent for Anthropic."""
    return MetricEvent(
        trace_id=trace_id,
        span_id=span_id,
        provider="anthropic",
        model=model,
        stream=stream,
        latency_ms=latency_ms,
        usage=usage,
        request_id=request_id,
        tool_calls=tool_calls if tool_calls else None,
        error=error,
        metadata=metadata,
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
            # Can't run async in sync context - just skip
            logger.warning("Async emitter used in sync context - metric may be lost")
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
        # Can't run async hook in sync context
        logger.warning("Async before_request hook used in sync context - skipping")
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
    ):
        self._stream = stream
        self._trace_id = trace_id
        self._span_id = span_id
        self._model = model
        self._t0 = t0
        self._options = options
        self._final_usage: NormalizedUsage | None = None
        self._input_tokens: int = 0  # Track input tokens from message_start
        self._request_id: str | None = None
        self._tool_calls: list[ToolCallMetric] = []
        self._done = False
        self._error: str | None = None

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
                elif chunk_type == "content_block_start" and self._options.track_tool_calls:
                    content_block = getattr(chunk, "content_block", None)
                    if content_block:
                        block_type = getattr(content_block, "type", None)
                        if block_type == "tool_use":
                            self._tool_calls.append(
                                ToolCallMetric(
                                    type="function",
                                    name=getattr(content_block, "name", None),
                                )
                            )

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
        event = _build_metric_event(
            trace_id=self._trace_id,
            span_id=self._span_id,
            model=self._model,
            stream=True,
            latency_ms=(time.time() - self._t0) * 1000,
            usage=self._final_usage,
            request_id=self._request_id,
            tool_calls=self._tool_calls if self._tool_calls else None,
            error=self._error,
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
    ):
        self._stream = stream
        self._trace_id = trace_id
        self._span_id = span_id
        self._model = model
        self._t0 = t0
        self._options = options
        self._final_usage: NormalizedUsage | None = None
        self._input_tokens: int = 0
        self._request_id: str | None = None
        self._tool_calls: list[ToolCallMetric] = []
        self._done = False
        self._error: str | None = None

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

                elif chunk_type == "content_block_start" and self._options.track_tool_calls:
                    content_block = getattr(chunk, "content_block", None)
                    if content_block:
                        block_type = getattr(content_block, "type", None)
                        if block_type == "tool_use":
                            self._tool_calls.append(
                                ToolCallMetric(
                                    type="function",
                                    name=getattr(content_block, "name", None),
                                )
                            )

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
        event = _build_metric_event(
            trace_id=self._trace_id,
            span_id=self._span_id,
            model=self._model,
            stream=True,
            latency_ms=(time.time() - self._t0) * 1000,
            usage=self._final_usage,
            request_id=self._request_id,
            tool_calls=self._tool_calls if self._tool_calls else None,
            error=self._error,
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

        # Extract params - Anthropic messages.create uses kwargs
        params = kwargs.copy()

        trace_id = options.generate_trace_id() if options.generate_trace_id else str(uuid4())
        span_id = options.generate_span_id() if options.generate_span_id else str(uuid4())
        model = params.get("model", "unknown")
        t0 = time.time()

        # Execute beforeRequest hook
        context = BeforeRequestContext(
            model=model,
            stream=bool(params.get("stream")),
            trace_id=trace_id,
            timestamp=__import__("datetime").datetime.now(),
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
                return MeteredAsyncStream(
                    response.__aiter__(), trace_id, span_id, model, t0, options
                )

            # Non-streaming response
            event = _build_metric_event(
                trace_id=trace_id,
                span_id=span_id,
                model=model,
                stream=False,
                latency_ms=(time.time() - t0) * 1000,
                usage=normalize_anthropic_usage(getattr(response, "usage", None)),
                request_id=_extract_request_id(response),
                tool_calls=_extract_tool_calls(response) if options.track_tool_calls else None,
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

        params = kwargs.copy()

        trace_id = options.generate_trace_id() if options.generate_trace_id else str(uuid4())
        span_id = options.generate_span_id() if options.generate_span_id else str(uuid4())
        model = params.get("model", "unknown")
        t0 = time.time()

        context = BeforeRequestContext(
            model=model,
            stream=bool(params.get("stream")),
            trace_id=trace_id,
            timestamp=__import__("datetime").datetime.now(),
            metadata=options.request_metadata,
        )

        result = _execute_before_request_hook_sync(params, context, options)
        final_params = _handle_before_request_result_sync(result, params, context)
        model = final_params.get("model", model)

        try:
            response = original_fn(self, **final_params)

            if final_params.get("stream") and hasattr(response, "__iter__"):
                return MeteredSyncStream(
                    iter(response), trace_id, span_id, model, t0, options
                )

            event = _build_metric_event(
                trace_id=trace_id,
                span_id=span_id,
                model=model,
                stream=False,
                latency_ms=(time.time() - t0) * 1000,
                usage=normalize_anthropic_usage(getattr(response, "usage", None)),
                request_id=_extract_request_id(response),
                tool_calls=_extract_tool_calls(response) if options.track_tool_calls else None,
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
