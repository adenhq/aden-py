"""
Core metering functionality.

This module provides the monkey-patching mechanism to intercept OpenAI API calls
and emit metrics without modifying the SDK.
"""

import asyncio
import logging
import random
import time
from datetime import datetime
from functools import wraps
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator, TypeVar
from uuid import uuid4

if TYPE_CHECKING:
    from openai import AsyncOpenAI, OpenAI

from .normalize import normalize_usage
from .types import (
    BeforeRequestAction,
    BeforeRequestContext,
    BeforeRequestResult,
    MeterOptions,
    MetricEvent,
    NormalizedUsage,
    RequestCancelledError,
    RequestMetadata,
    ToolCallMetric,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Marker attribute for metered clients
_METERED_ATTR = "__openai_meter__"
_METER_OPTIONS_ATTR = "__meter_options__"


def _extract_request_id(response: Any) -> str | None:
    """Extracts request ID from various response object shapes."""
    if response is None:
        return None

    # Try different attribute names
    for attr in ("request_id", "requestId", "_request_id"):
        if hasattr(response, attr):
            return getattr(response, attr)

    # Try dict access
    if isinstance(response, dict):
        return response.get("request_id") or response.get("requestId")

    return None


def _extract_tool_calls(response: Any) -> list[ToolCallMetric]:
    """Extracts tool call metrics from a response."""
    tool_calls: list[ToolCallMetric] = []

    if response is None:
        return tool_calls

    # Convert to dict if needed
    if hasattr(response, "model_dump"):
        data = response.model_dump()
    elif hasattr(response, "__dict__"):
        data = response.__dict__
    elif isinstance(response, dict):
        data = response
    else:
        return tool_calls

    # Handle output array (Responses API)
    output = data.get("output", [])
    if isinstance(output, list):
        for item in output:
            if isinstance(item, dict):
                item_type = item.get("type", "")
                if item_type == "function_call":
                    tool_calls.append(ToolCallMetric(type="function", name=item.get("name")))
                elif item_type in ("web_search_call", "code_interpreter_call", "file_search_call"):
                    tool_calls.append(ToolCallMetric(type=item_type.replace("_call", "")))

    # Handle choices array (Chat Completions API)
    choices = data.get("choices", [])
    if isinstance(choices, list):
        for choice in choices:
            if isinstance(choice, dict):
                message = choice.get("message", {})
                if isinstance(message, dict):
                    tc_list = message.get("tool_calls", [])
                    if isinstance(tc_list, list):
                        for tc in tc_list:
                            if isinstance(tc, dict):
                                fn = tc.get("function", {})
                                tool_calls.append(
                                    ToolCallMetric(
                                        type=tc.get("type", "function"),
                                        name=fn.get("name") if isinstance(fn, dict) else None,
                                    )
                                )

    return tool_calls


def _build_request_metadata(params: dict[str, Any], trace_id: str) -> RequestMetadata:
    """Builds request metadata from params."""
    return RequestMetadata(
        trace_id=trace_id,
        model=params.get("model", "unknown"),
        stream=bool(params.get("stream")),
        service_tier=params.get("service_tier"),
        max_output_tokens=params.get("max_output_tokens"),
        max_tool_calls=params.get("max_tool_calls"),
        prompt_cache_key=params.get("prompt_cache_key"),
        prompt_cache_retention=params.get("prompt_cache_retention"),
    )


def _build_before_request_context(
    params: dict[str, Any],
    trace_id: str,
    options: MeterOptions,
) -> BeforeRequestContext:
    """Builds the beforeRequest context from params and options."""
    return BeforeRequestContext(
        model=params.get("model", "unknown"),
        stream=bool(params.get("stream")),
        trace_id=trace_id,
        timestamp=datetime.now(),
        metadata=options.request_metadata,
    )


async def _execute_before_request_hook(
    params: dict[str, Any],
    context: BeforeRequestContext,
    options: MeterOptions,
) -> None:
    """Executes the beforeRequest hook if provided."""
    if options.before_request is None:
        return

    result = options.before_request(params, context)
    if asyncio.iscoroutine(result):
        result = await result

    if result.action == BeforeRequestAction.CANCEL:
        raise RequestCancelledError(result.reason, context)

    if result.action == BeforeRequestAction.THROTTLE:
        await asyncio.sleep(result.delay_ms / 1000)


def _execute_before_request_hook_sync(
    params: dict[str, Any],
    context: BeforeRequestContext,
    options: MeterOptions,
) -> None:
    """Executes the beforeRequest hook synchronously."""
    if options.before_request is None:
        return

    result = options.before_request(params, context)
    if asyncio.iscoroutine(result):
        # Close the unawaited coroutine to prevent warnings
        result.close()
        return

    if result.action == BeforeRequestAction.CANCEL:
        raise RequestCancelledError(result.reason, context)

    if result.action == BeforeRequestAction.THROTTLE:
        time.sleep(result.delay_ms / 1000)


def _create_metric_event(
    req_meta: RequestMetadata,
    request_id: str | None,
    latency_ms: float,
    usage: NormalizedUsage | None,
    tool_calls: list[ToolCallMetric] | None,
    error: str | None = None,
) -> MetricEvent:
    """Creates a MetricEvent from components."""
    return MetricEvent(
        trace_id=req_meta.trace_id,
        model=req_meta.model,
        stream=req_meta.stream,
        service_tier=req_meta.service_tier,
        max_output_tokens=req_meta.max_output_tokens,
        max_tool_calls=req_meta.max_tool_calls,
        prompt_cache_key=req_meta.prompt_cache_key,
        prompt_cache_retention=req_meta.prompt_cache_retention,
        request_id=request_id,
        latency_ms=latency_ms,
        usage=usage,
        tool_calls=tool_calls if tool_calls else None,
        error=error,
    )


def _handle_emit_error(event: MetricEvent, error: Exception, options: MeterOptions) -> None:
    """Handle emission error - call callback or log."""
    if options.on_emit_error:
        try:
            options.on_emit_error(event, error)
        except Exception as callback_error:
            logger.error(f"Error in on_emit_error callback: {callback_error}")
    else:
        logger.error(f"Error emitting metric (trace_id={event.trace_id}): {error}")


async def _emit_metric(event: MetricEvent, options: MeterOptions) -> None:
    """Emits a metric, handling async/sync emitters."""
    if options.async_emit:
        # Fire-and-forget with error handling
        async def emit_with_error_handling() -> None:
            try:
                result = options.emit_metric(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                _handle_emit_error(event, e, options)

        try:
            asyncio.create_task(emit_with_error_handling())
        except Exception as e:
            _handle_emit_error(event, e, options)
    else:
        try:
            result = options.emit_metric(event)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            _handle_emit_error(event, e, options)
            raise  # Re-raise in sync mode so caller knows emission failed


def _emit_metric_sync(event: MetricEvent, options: MeterOptions) -> None:
    """Emits a metric synchronously."""
    try:
        result = options.emit_metric(event)
        if asyncio.iscoroutine(result):
            # Close the unawaited coroutine to prevent warnings
            result.close()
    except Exception as e:
        _handle_emit_error(event, e, options)


def _should_meter(options: MeterOptions) -> bool:
    """Determines if this request should be metered based on sample rate."""
    if options.sample_rate >= 1.0:
        return True
    return random.random() < options.sample_rate


class MeteredAsyncStream:
    """Wraps an async stream to meter it."""

    def __init__(
        self,
        stream: AsyncIterator[Any],
        req_meta: RequestMetadata,
        t0: float,
        options: MeterOptions,
    ):
        self._stream = stream
        self._req_meta = req_meta
        self._t0 = t0
        self._options = options
        self._final_usage: NormalizedUsage | None = None
        self._request_id: str | None = None
        self._tool_calls: list[ToolCallMetric] = []
        self._done = False
        self._error: str | None = None

    def __aiter__(self) -> "MeteredAsyncStream":
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._stream.__anext__()

            # Extract request ID from any chunk that has it
            if self._request_id is None:
                self._request_id = _extract_request_id(chunk)

            # Check for completed event with usage
            if hasattr(chunk, "type"):
                chunk_type = chunk.type
                if chunk_type in ("response.completed", "message_stop"):
                    response = getattr(chunk, "response", chunk)
                    if hasattr(response, "usage"):
                        self._final_usage = normalize_usage(response.usage)
                    if self._options.track_tool_calls:
                        self._tool_calls.extend(_extract_tool_calls(response))

                # Track tool call events during streaming
                if self._options.track_tool_calls:
                    if chunk_type == "response.function_call_arguments.done":
                        self._tool_calls.append(
                            ToolCallMetric(type="function", name=getattr(chunk, "name", None))
                        )

            return chunk

        except StopAsyncIteration:
            if not self._done:
                self._done = True
                await self._emit_final_metric()
            raise
        except Exception as e:
            # Stream error - emit partial metrics before re-raising
            if not self._done:
                self._done = True
                self._error = str(e)
                await self._emit_final_metric()
            raise

    async def _emit_final_metric(self) -> None:
        """Emit the final metric when stream ends (normally or with error)."""
        event = _create_metric_event(
            self._req_meta,
            self._request_id,
            (time.time() - self._t0) * 1000,
            self._final_usage,
            self._tool_calls if self._tool_calls else None,
            error=self._error,
        )
        await _emit_metric(event, self._options)


class MeteredSyncStream:
    """Wraps a sync stream to meter it."""

    def __init__(
        self,
        stream: Iterator[Any],
        req_meta: RequestMetadata,
        t0: float,
        options: MeterOptions,
    ):
        self._stream = stream
        self._req_meta = req_meta
        self._t0 = t0
        self._options = options
        self._final_usage: NormalizedUsage | None = None
        self._request_id: str | None = None
        self._tool_calls: list[ToolCallMetric] = []
        self._done = False
        self._error: str | None = None

    def __iter__(self) -> "MeteredSyncStream":
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._stream)

            # Extract request ID from any chunk that has it
            if self._request_id is None:
                self._request_id = _extract_request_id(chunk)

            # Check for completed event with usage
            if hasattr(chunk, "type"):
                chunk_type = chunk.type
                if chunk_type in ("response.completed", "message_stop"):
                    response = getattr(chunk, "response", chunk)
                    if hasattr(response, "usage"):
                        self._final_usage = normalize_usage(response.usage)
                    if self._options.track_tool_calls:
                        self._tool_calls.extend(_extract_tool_calls(response))

                # Track tool call events during streaming
                if self._options.track_tool_calls:
                    if chunk_type == "response.function_call_arguments.done":
                        self._tool_calls.append(
                            ToolCallMetric(type="function", name=getattr(chunk, "name", None))
                        )

            return chunk

        except StopIteration:
            if not self._done:
                self._done = True
                self._emit_final_metric()
            raise
        except Exception as e:
            # Stream error - emit partial metrics before re-raising
            if not self._done:
                self._done = True
                self._error = str(e)
                self._emit_final_metric()
            raise

    def _emit_final_metric(self) -> None:
        """Emit the final metric when stream ends (normally or with error)."""
        event = _create_metric_event(
            self._req_meta,
            self._request_id,
            (time.time() - self._t0) * 1000,
            self._final_usage,
            self._tool_calls if self._tool_calls else None,
            error=self._error,
        )
        _emit_metric_sync(event, self._options)


def _wrap_async_create(original_fn: Any, options: MeterOptions) -> Any:
    """Wraps an async create method with metering."""

    @wraps(original_fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Handle both positional and keyword params
        params = kwargs if kwargs else (args[0] if args else {})
        if not isinstance(params, dict):
            params = {}

        # Check sample rate
        if not _should_meter(options):
            return await original_fn(*args, **kwargs)

        trace_id = options.generate_trace_id() if options.generate_trace_id else str(uuid4())
        t0 = time.time()
        req_meta = _build_request_metadata(params, trace_id)

        # Execute beforeRequest hook
        before_ctx = _build_before_request_context(params, trace_id, options)
        await _execute_before_request_hook(params, before_ctx, options)

        try:
            result = await original_fn(*args, **kwargs)

            # Handle streaming responses
            if params.get("stream") and hasattr(result, "__aiter__"):
                return MeteredAsyncStream(result.__aiter__(), req_meta, t0, options)

            # Non-streaming response
            event = _create_metric_event(
                req_meta,
                _extract_request_id(result),
                (time.time() - t0) * 1000,
                normalize_usage(getattr(result, "usage", None)),
                _extract_tool_calls(result) if options.track_tool_calls else None,
            )
            await _emit_metric(event, options)
            return result

        except Exception as e:
            event = _create_metric_event(
                req_meta,
                None,
                (time.time() - t0) * 1000,
                None,
                None,
                error=str(e),
            )
            await _emit_metric(event, options)
            raise

    return wrapper


def _wrap_sync_create(original_fn: Any, options: MeterOptions) -> Any:
    """Wraps a sync create method with metering."""

    @wraps(original_fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        params = kwargs if kwargs else (args[0] if args else {})
        if not isinstance(params, dict):
            params = {}

        if not _should_meter(options):
            return original_fn(*args, **kwargs)

        trace_id = options.generate_trace_id() if options.generate_trace_id else str(uuid4())
        t0 = time.time()
        req_meta = _build_request_metadata(params, trace_id)

        before_ctx = _build_before_request_context(params, trace_id, options)
        _execute_before_request_hook_sync(params, before_ctx, options)

        try:
            result = original_fn(*args, **kwargs)

            # Handle streaming responses - wrap iterator for proper metering
            if params.get("stream") and hasattr(result, "__iter__"):
                return MeteredSyncStream(iter(result), req_meta, t0, options)

            # Non-streaming response
            event = _create_metric_event(
                req_meta,
                _extract_request_id(result),
                (time.time() - t0) * 1000,
                normalize_usage(getattr(result, "usage", None)),
                _extract_tool_calls(result) if options.track_tool_calls else None,
            )
            _emit_metric_sync(event, options)
            return result

        except Exception as e:
            event = _create_metric_event(
                req_meta,
                None,
                (time.time() - t0) * 1000,
                None,
                None,
                error=str(e),
            )
            _emit_metric_sync(event, options)
            raise

    return wrapper


# Type alias for metered clients (resolved at runtime only if openai is installed)
MeteredOpenAI: Any = None
MeteredAsyncOpenAI: Any = None

try:
    from openai import AsyncOpenAI as _AsyncOpenAI
    from openai import OpenAI as _OpenAI
    MeteredOpenAI = _OpenAI
    MeteredAsyncOpenAI = _AsyncOpenAI
except ImportError:
    pass


def make_metered_openai(
    client: "OpenAI | AsyncOpenAI",
    options: MeterOptions,
) -> "OpenAI | AsyncOpenAI":
    """
    Wraps an OpenAI client with metering capabilities.

    This function injects metering into the client without modifying the SDK,
    allowing you to track usage metrics, billing data, and request metadata
    for every API call.

    Args:
        client: The OpenAI client instance to wrap
        options: Metering options including the metric emitter

    Returns:
        The same client with metering injected

    Example:
        ```python
        from openai import OpenAI
        from openai_meter import make_metered_openai, create_console_emitter

        client = OpenAI()
        metered = make_metered_openai(client, MeterOptions(
            emit_metric=create_console_emitter(),
        ))

        # Use normally - metrics are collected automatically
        response = metered.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        ```
    """
    # Check if client is async - use the imported class if available
    is_async = MeteredAsyncOpenAI is not None and isinstance(client, MeteredAsyncOpenAI)
    wrap_fn = _wrap_async_create if is_async else _wrap_sync_create

    # Wrap chat.completions.create
    if hasattr(client, "chat") and hasattr(client.chat, "completions"):
        original_chat_create = client.chat.completions.create
        client.chat.completions.create = wrap_fn(original_chat_create, options)  # type: ignore

    # Wrap responses.create (if available - newer API)
    if hasattr(client, "responses"):
        original_responses_create = client.responses.create
        client.responses.create = wrap_fn(original_responses_create, options)  # type: ignore

    # Mark client as metered
    setattr(client, _METERED_ATTR, True)
    setattr(client, _METER_OPTIONS_ATTR, options)

    return client


def is_metered(client: "OpenAI | AsyncOpenAI") -> bool:
    """
    Check if a client has already been wrapped with metering.

    Args:
        client: The OpenAI client to check

    Returns:
        True if the client is metered, False otherwise
    """
    return getattr(client, _METERED_ATTR, False)
