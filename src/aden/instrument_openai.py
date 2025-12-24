"""
OpenAI SDK instrumentation.

This module provides global instrumentation for the OpenAI SDK by patching
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
from .normalize import normalize_openai_usage
from .types import (
    BeforeRequestAction,
    BeforeRequestContext,
    BeforeRequestResult,
    MeterOptions,
    MetricEvent,
    NormalizedUsage,
    RequestCancelledError,
)

logger = logging.getLogger("aden")

# Module-level state
_is_instrumented = False
_global_options: MeterOptions | None = None

# Store original methods for uninstrumentation
_original_chat_create: Callable[..., Any] | None = None
_original_async_chat_create: Callable[..., Any] | None = None
_original_responses_create: Callable[..., Any] | None = None
_original_async_responses_create: Callable[..., Any] | None = None


def _get_openai_classes(options: MeterOptions) -> tuple[Any, Any] | None:
    """Get OpenAI and AsyncOpenAI classes from options or auto-import."""
    if options.sdks:
        OpenAI = options.sdks.OpenAI
        AsyncOpenAI = options.sdks.AsyncOpenAI
        if OpenAI or AsyncOpenAI:
            return (OpenAI, AsyncOpenAI)

    # Try auto-import
    try:
        from openai import AsyncOpenAI, OpenAI
        return (OpenAI, AsyncOpenAI)
    except ImportError:
        return None


def _extract_request_id(response: Any) -> str | None:
    """Extracts request ID from various response object shapes."""
    if response is None:
        return None
    for attr in ("request_id", "requestId", "_request_id"):
        if hasattr(response, attr):
            return getattr(response, attr)
    if isinstance(response, dict):
        return response.get("request_id") or response.get("requestId")
    return None


def _extract_tool_calls(response: Any) -> tuple[int, str | None]:
    """Extracts tool call count and names from a response.

    Returns:
        Tuple of (tool_call_count, tool_names_comma_separated)
    """
    tool_names: list[str] = []

    if response is None:
        return (0, None)

    # Convert to dict if needed
    if hasattr(response, "model_dump"):
        data = response.model_dump()
    elif hasattr(response, "__dict__"):
        data = response.__dict__
    elif isinstance(response, dict):
        data = response
    else:
        return (0, None)

    # Handle output array (Responses API)
    output = data.get("output", [])
    if isinstance(output, list):
        for item in output:
            if isinstance(item, dict):
                item_type = item.get("type", "")
                if item_type == "function_call":
                    name = item.get("name")
                    if name:
                        tool_names.append(name)
                elif item_type in ("web_search_call", "code_interpreter_call", "file_search_call"):
                    tool_names.append(item_type.replace("_call", ""))

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
                                name = fn.get("name") if isinstance(fn, dict) else None
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
    service_tier: str | None = None,
    metadata: dict[str, Any] | None = None,
    stack_info: CallStackInfo | None = None,
) -> MetricEvent:
    """Builds a MetricEvent for OpenAI with flat fields."""
    return MetricEvent(
        trace_id=trace_id,
        span_id=span_id,
        provider="openai",
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
        # Flatten tool calls
        tool_call_count=tool_call_count if tool_call_count and tool_call_count > 0 else None,
        tool_names=tool_names,
        service_tier=service_tier,
        metadata=metadata,
        # Call stack info
        call_site_file=stack_info.call_site_file if stack_info else None,
        call_site_line=stack_info.call_site_line if stack_info else None,
        call_site_function=stack_info.call_site_function if stack_info else None,
        call_stack=stack_info.call_stack if stack_info else None,
        agent_stack=stack_info.agent_stack if stack_info else None,
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
    """Wraps an async OpenAI stream to meter it."""

    def __init__(
        self,
        stream: AsyncIterator[Any],
        trace_id: str,
        span_id: str,
        model: str,
        t0: float,
        options: MeterOptions,
        stack_info: CallStackInfo | None = None,
    ):
        self._stream = stream
        self._trace_id = trace_id
        self._span_id = span_id
        self._model = model
        self._t0 = t0
        self._options = options
        self._stack_info = stack_info
        self._final_usage: NormalizedUsage | None = None
        self._request_id: str | None = None
        self._tool_names: list[str] = []
        self._done = False
        self._error: str | None = None

    def __aiter__(self) -> "MeteredAsyncStream":
        return self

    async def __aenter__(self) -> "MeteredAsyncStream":
        """Support async context manager protocol."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Cleanup on context manager exit."""
        if not self._done:
            self._done = True
            if exc_val:
                self._error = str(exc_val)
            await self._emit_final_metric()

    async def __anext__(self) -> Any:
        try:
            chunk = await self._stream.__anext__()

            if self._request_id is None:
                self._request_id = _extract_request_id(chunk)

            # Check for completed event with usage
            if hasattr(chunk, "type"):
                chunk_type = chunk.type
                if chunk_type in ("response.completed", "message_stop"):
                    response = getattr(chunk, "response", chunk)
                    if hasattr(response, "usage"):
                        self._final_usage = normalize_openai_usage(response.usage)
                    if self._options.track_tool_calls:
                        _, names = _extract_tool_calls(response)
                        if names:
                            self._tool_names.extend(names.split(","))

                if self._options.track_tool_calls:
                    if chunk_type == "response.function_call_arguments.done":
                        name = getattr(chunk, "name", None)
                        if name:
                            self._tool_names.append(name)

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
        )
        await _emit_metric(event, self._options)


class MeteredSyncStream:
    """Wraps a sync OpenAI stream to meter it."""

    def __init__(
        self,
        stream: Iterator[Any],
        trace_id: str,
        span_id: str,
        model: str,
        t0: float,
        options: MeterOptions,
        stack_info: CallStackInfo | None = None,
    ):
        self._stream = stream
        self._trace_id = trace_id
        self._span_id = span_id
        self._model = model
        self._t0 = t0
        self._options = options
        self._stack_info = stack_info
        self._final_usage: NormalizedUsage | None = None
        self._request_id: str | None = None
        self._tool_names: list[str] = []
        self._done = False
        self._error: str | None = None

    def __iter__(self) -> "MeteredSyncStream":
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._stream)

            if self._request_id is None:
                self._request_id = _extract_request_id(chunk)

            if hasattr(chunk, "type"):
                chunk_type = chunk.type
                if chunk_type in ("response.completed", "message_stop"):
                    response = getattr(chunk, "response", chunk)
                    if hasattr(response, "usage"):
                        self._final_usage = normalize_openai_usage(response.usage)
                    if self._options.track_tool_calls:
                        _, names = _extract_tool_calls(response)
                        if names:
                            self._tool_names.extend(names.split(","))

                if self._options.track_tool_calls:
                    if chunk_type == "response.function_call_arguments.done":
                        name = getattr(chunk, "name", None)
                        if name:
                            self._tool_names.append(name)

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
        )
        _emit_metric_sync(event, self._options)


def _create_async_wrapper(
    original_fn: Callable[..., Any], get_options: Callable[[], MeterOptions | None]
) -> Callable[..., Any]:
    """Creates an async wrapper for OpenAI methods."""

    @wraps(original_fn)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        options = get_options()
        if options is None:
            return await original_fn(self, *args, **kwargs)

        # Extract params
        params = kwargs if kwargs else (args[0] if args else {})
        if not isinstance(params, dict):
            params = {}

        trace_id = options.generate_trace_id() if options.generate_trace_id else str(uuid4())
        span_id = options.generate_span_id() if options.generate_span_id else str(uuid4())
        model = params.get("model", "unknown")
        t0 = time.time()

        # Capture call stack before making the request
        stack_info = capture_call_stack(skip_frames=3)

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
            response = await original_fn(self, **final_params) if kwargs else await original_fn(self, final_params, *args[1:])

            # Handle streaming
            if final_params.get("stream") and hasattr(response, "__aiter__"):
                return MeteredAsyncStream(response.__aiter__(), trace_id, span_id, model, t0, options, stack_info)

            # Non-streaming response - extract tool calls
            tool_count, tool_names = (
                _extract_tool_calls(response) if options.track_tool_calls else (None, None)
            )
            event = _build_metric_event(
                trace_id=trace_id,
                span_id=span_id,
                model=model,
                stream=False,
                latency_ms=(time.time() - t0) * 1000,
                usage=normalize_openai_usage(getattr(response, "usage", None)),
                request_id=_extract_request_id(response),
                tool_call_count=tool_count,
                tool_names=tool_names,
                stack_info=stack_info,
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
            )
            await _emit_metric(event, options)
            raise

    return wrapper


def _create_sync_wrapper(
    original_fn: Callable[..., Any], get_options: Callable[[], MeterOptions | None]
) -> Callable[..., Any]:
    """Creates a sync wrapper for OpenAI methods."""

    @wraps(original_fn)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        options = get_options()
        if options is None:
            return original_fn(self, *args, **kwargs)

        params = kwargs if kwargs else (args[0] if args else {})
        if not isinstance(params, dict):
            params = {}

        trace_id = options.generate_trace_id() if options.generate_trace_id else str(uuid4())
        span_id = options.generate_span_id() if options.generate_span_id else str(uuid4())
        model = params.get("model", "unknown")
        t0 = time.time()

        # Capture call stack before making the request
        stack_info = capture_call_stack(skip_frames=3)

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
            response = original_fn(self, **final_params) if kwargs else original_fn(self, final_params, *args[1:])

            if final_params.get("stream") and hasattr(response, "__iter__"):
                return MeteredSyncStream(iter(response), trace_id, span_id, model, t0, options, stack_info)

            # Extract tool calls
            tool_count, tool_names = (
                _extract_tool_calls(response) if options.track_tool_calls else (None, None)
            )
            event = _build_metric_event(
                trace_id=trace_id,
                span_id=span_id,
                model=model,
                stream=False,
                latency_ms=(time.time() - t0) * 1000,
                usage=normalize_openai_usage(getattr(response, "usage", None)),
                request_id=_extract_request_id(response),
                tool_call_count=tool_count,
                tool_names=tool_names,
                stack_info=stack_info,
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
            )
            _emit_metric_sync(event, options)
            raise

    return wrapper


def instrument_openai(options: MeterOptions) -> bool:
    """
    Instrument the OpenAI SDK globally.

    Patches the Completions and AsyncCompletions classes directly so all
    client instances are automatically metered.

    Args:
        options: Metering options including the metric emitter

    Returns:
        True if instrumentation succeeded, False if OpenAI SDK not available
    """
    global _is_instrumented, _global_options
    global _original_chat_create, _original_async_chat_create
    global _original_responses_create, _original_async_responses_create

    if _is_instrumented:
        return True

    # Check if OpenAI SDK is available
    try:
        from openai.resources.chat.completions import AsyncCompletions, Completions
    except ImportError:
        logger.debug("OpenAI SDK not available, skipping instrumentation")
        return False

    _global_options = options

    def get_options() -> MeterOptions | None:
        return _global_options

    # Patch sync Completions.create
    try:
        _original_chat_create = Completions.create
        Completions.create = _create_sync_wrapper(_original_chat_create, get_options)
    except Exception as e:
        logger.warning(f"Failed to instrument sync Completions: {e}")

    # Patch async AsyncCompletions.create
    try:
        _original_async_chat_create = AsyncCompletions.create
        AsyncCompletions.create = _create_async_wrapper(_original_async_chat_create, get_options)
    except Exception as e:
        logger.warning(f"Failed to instrument async Completions: {e}")

    # Try to patch responses.create (if available in newer SDK versions)
    try:
        from openai.resources.responses import AsyncResponses, Responses

        _original_responses_create = Responses.create
        Responses.create = _create_sync_wrapper(_original_responses_create, get_options)

        _original_async_responses_create = AsyncResponses.create
        AsyncResponses.create = _create_async_wrapper(_original_async_responses_create, get_options)
    except ImportError:
        # Responses API not available in this SDK version
        pass
    except Exception as e:
        logger.warning(f"Failed to instrument Responses: {e}")

    _is_instrumented = True
    logger.info("[aden] OpenAI SDK instrumented")
    return True


def uninstrument_openai() -> None:
    """
    Remove OpenAI SDK instrumentation.

    Restores original methods on the Completions classes.
    """
    global _is_instrumented, _global_options
    global _original_chat_create, _original_async_chat_create
    global _original_responses_create, _original_async_responses_create

    if not _is_instrumented:
        return

    # Try to restore original methods
    try:
        from openai.resources.chat.completions import AsyncCompletions, Completions

        if _original_chat_create:
            Completions.create = _original_chat_create
        if _original_async_chat_create:
            AsyncCompletions.create = _original_async_chat_create
    except ImportError:
        pass

    try:
        from openai.resources.responses import AsyncResponses, Responses

        if _original_responses_create:
            Responses.create = _original_responses_create
        if _original_async_responses_create:
            AsyncResponses.create = _original_async_responses_create
    except ImportError:
        pass

    _is_instrumented = False
    _global_options = None
    _original_chat_create = None
    _original_async_chat_create = None
    _original_responses_create = None
    _original_async_responses_create = None

    logger.info("[aden] OpenAI SDK uninstrumented")


def is_openai_instrumented() -> bool:
    """Check if OpenAI SDK is currently instrumented."""
    return _is_instrumented


def get_openai_options() -> MeterOptions | None:
    """Get current OpenAI instrumentation options."""
    return _global_options
