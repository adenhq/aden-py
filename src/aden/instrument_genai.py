"""
Google GenAI SDK instrumentation (new SDK).

This module provides global instrumentation for the new Google GenAI SDK
(google-genai package) used by Google ADK and other modern Google AI tools.

The new SDK uses:
    from google import genai
    client = genai.Client(api_key='...')
    client.models.generate_content(model='...', contents='...')
"""

import asyncio
import logging
import time
from functools import wraps
from typing import Any, Callable
from uuid import uuid4
from datetime import datetime

from .call_stack import CallStackInfo, capture_call_stack
from .content_capture import (
    extract_gemini_request_content,
    extract_gemini_response_content,
)
from .large_content import store_large_content
from .normalize import NormalizedUsage
from .types import (
    BeforeRequestAction,
    BeforeRequestContext,
    BeforeRequestResult,
    ContentCapture,
    ContentCaptureOptions,
    MeterOptions,
    MetricEvent,
    RequestCancelledError,
)

logger = logging.getLogger("aden")

# Module-level state
_is_instrumented = False
_global_options: MeterOptions | None = None

# Store original methods for uninstrumentation
_original_generate_content: Callable[..., Any] | None = None
_original_generate_content_async: Callable[..., Any] | None = None
_original_generate_content_stream: Callable[..., Any] | None = None
_original_generate_content_stream_async: Callable[..., Any] | None = None


def _normalize_genai_usage(usage: Any) -> NormalizedUsage | None:
    """Normalize usage metadata from google-genai response."""
    if usage is None:
        return None

    input_tokens = 0
    output_tokens = 0
    cached_tokens = 0

    if hasattr(usage, "prompt_token_count"):
        input_tokens = usage.prompt_token_count or 0
    elif hasattr(usage, "input_tokens"):
        input_tokens = usage.input_tokens or 0

    if hasattr(usage, "candidates_token_count"):
        output_tokens = usage.candidates_token_count or 0
    elif hasattr(usage, "output_tokens"):
        output_tokens = usage.output_tokens or 0

    if hasattr(usage, "cached_content_token_count"):
        cached_tokens = usage.cached_content_token_count or 0

    return NormalizedUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cached_tokens=cached_tokens,
        reasoning_tokens=0,
    )


def _build_metric_event(
    trace_id: str,
    span_id: str,
    model: str,
    stream: bool,
    latency_ms: float,
    usage: NormalizedUsage | None,
    error: str | None = None,
    metadata: dict[str, Any] | None = None,
    stack_info: CallStackInfo | None = None,
    content_capture: ContentCapture | None = None,
) -> MetricEvent:
    """Builds a MetricEvent for GenAI with flat fields."""
    return MetricEvent(
        trace_id=trace_id,
        span_id=span_id,
        provider="gemini",
        model=model,
        stream=stream,
        timestamp=datetime.now().isoformat(),
        latency_ms=latency_ms,
        request_id=None,
        error=error,
        input_tokens=usage.input_tokens if usage else 0,
        output_tokens=usage.output_tokens if usage else 0,
        total_tokens=usage.total_tokens if usage else 0,
        cached_tokens=usage.cached_tokens if usage else 0,
        reasoning_tokens=usage.reasoning_tokens if usage else 0,
        metadata=metadata,
        # Call stack info
        call_site_file=stack_info.call_site_file if stack_info else None,
        call_site_line=stack_info.call_site_line if stack_info else None,
        call_site_function=stack_info.call_site_function if stack_info else None,
        call_stack=stack_info.call_stack if stack_info else None,
        agent_stack=stack_info.agent_stack if stack_info else None,
        # Content capture
        content_capture=content_capture,
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


def _extract_model_from_kwargs(kwargs: dict[str, Any]) -> str:
    """Extract model name from kwargs."""
    model = kwargs.get("model", "unknown")
    if hasattr(model, "name"):
        return model.name
    return str(model) if model else "unknown"


def _extract_usage_from_response(response: Any) -> NormalizedUsage | None:
    """Extract usage from a genai response."""
    if response is None:
        return None

    # Try usage_metadata first (standard location)
    if hasattr(response, "usage_metadata"):
        return _normalize_genai_usage(response.usage_metadata)

    # Try direct usage attribute
    if hasattr(response, "usage"):
        return _normalize_genai_usage(response.usage)

    return None


def _create_sync_wrapper(
    original_fn: Callable[..., Any],
    get_options: Callable[[], MeterOptions | None],
    is_stream: bool = False,
) -> Callable[..., Any]:
    """Creates a sync wrapper for generate_content methods."""

    @wraps(original_fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        options = get_options()
        if options is None:
            return original_fn(*args, **kwargs)

        trace_id = options.generate_trace_id() if options.generate_trace_id else str(uuid4())
        span_id = options.generate_span_id() if options.generate_span_id else str(uuid4())
        model = _extract_model_from_kwargs(kwargs)
        t0 = time.time()

        # Capture call stack before making the request
        stack_info = capture_call_stack(skip_frames=3)

        # Layer 0: Content Capture - extract request content
        content_capture: ContentCapture | None = None
        large_content_payloads: list[dict[str, Any]] = []
        if options.capture_content:
            capture_options = options.content_capture_options or ContentCaptureOptions()
            # GenAI uses kwargs with 'contents' key
            content_capture, request_payloads = extract_gemini_request_content((), kwargs, capture_options)
            if request_payloads:
                large_content_payloads.extend(request_payloads)

        # Execute beforeRequest hook if present
        if options.before_request:
            context = BeforeRequestContext(
                model=model,
                stream=is_stream,
                span_id=span_id,
                trace_id=trace_id,
                timestamp=datetime.now(),
                metadata=options.request_metadata,
            )
            try:
                result = options.before_request(kwargs, context)
                if asyncio.iscoroutine(result):
                    # Close the unawaited coroutine to prevent warnings
                    result.close()
                    result = None
                if result:
                    if result.action == BeforeRequestAction.CANCEL:
                        raise RequestCancelledError(result.reason or "Request cancelled", context)
                    elif result.action == BeforeRequestAction.DEGRADE and result.to_model:
                        kwargs["model"] = result.to_model
                        model = result.to_model
            except RequestCancelledError:
                raise
            except Exception as e:
                logger.warning(f"Error in before_request hook: {e}")

        try:
            response = original_fn(*args, **kwargs)

            # For streaming, wrap the iterator
            if is_stream and hasattr(response, "__iter__"):
                # Store large content from request before streaming
                if large_content_payloads:
                    store_large_content(large_content_payloads)
                return _MeteredSyncStream(response, trace_id, span_id, model, t0, options, stack_info, content_capture)

            # Layer 0: Extract response content
            if options.capture_content and content_capture:
                capture_options = options.content_capture_options or ContentCaptureOptions()
                response_payloads = extract_gemini_response_content(response, content_capture, capture_options)
                if response_payloads:
                    large_content_payloads.extend(response_payloads)

            # Store all large content payloads
            if large_content_payloads:
                store_large_content(large_content_payloads)

            # Non-streaming: emit metric immediately
            usage = _extract_usage_from_response(response)
            event = _build_metric_event(
                trace_id=trace_id,
                span_id=span_id,
                model=model,
                stream=False,
                latency_ms=(time.time() - t0) * 1000,
                usage=usage,
                stack_info=stack_info,
                content_capture=content_capture,
            )
            _emit_metric_sync(event, options)
            return response

        except RequestCancelledError:
            raise
        except Exception as e:
            event = _build_metric_event(
                trace_id=trace_id,
                span_id=span_id,
                model=model,
                stream=is_stream,
                latency_ms=(time.time() - t0) * 1000,
                usage=None,
                error=str(e),
                stack_info=stack_info,
                content_capture=content_capture,
            )
            _emit_metric_sync(event, options)
            raise

    return wrapper


def _create_async_wrapper(
    original_fn: Callable[..., Any],
    get_options: Callable[[], MeterOptions | None],
    is_stream: bool = False,
) -> Callable[..., Any]:
    """Creates an async wrapper for generate_content methods."""

    @wraps(original_fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        options = get_options()
        if options is None:
            return await original_fn(*args, **kwargs)

        trace_id = options.generate_trace_id() if options.generate_trace_id else str(uuid4())
        span_id = options.generate_span_id() if options.generate_span_id else str(uuid4())
        model = _extract_model_from_kwargs(kwargs)
        t0 = time.time()

        # Capture call stack before making the request
        stack_info = capture_call_stack(skip_frames=3)

        # Execute beforeRequest hook if present
        if options.before_request:
            context = BeforeRequestContext(
                model=model,
                stream=is_stream,
                span_id=span_id,
                trace_id=trace_id,
                timestamp=datetime.now(),
                metadata=options.request_metadata,
            )
            try:
                result = options.before_request(kwargs, context)
                if asyncio.iscoroutine(result):
                    result = await result
                if result:
                    if result.action == BeforeRequestAction.CANCEL:
                        raise RequestCancelledError(result.reason or "Request cancelled", context)
                    elif result.action == BeforeRequestAction.DEGRADE and result.to_model:
                        kwargs["model"] = result.to_model
                        model = result.to_model
            except RequestCancelledError:
                raise
            except Exception as e:
                logger.warning(f"Error in before_request hook: {e}")

        try:
            response = await original_fn(*args, **kwargs)

            # For streaming, wrap the async iterator
            if is_stream and hasattr(response, "__aiter__"):
                return _MeteredAsyncStream(response, trace_id, span_id, model, t0, options, stack_info)

            # Non-streaming: emit metric immediately
            usage = _extract_usage_from_response(response)
            event = _build_metric_event(
                trace_id=trace_id,
                span_id=span_id,
                model=model,
                stream=False,
                latency_ms=(time.time() - t0) * 1000,
                usage=usage,
                stack_info=stack_info,
            )
            await _emit_metric(event, options)
            return response

        except RequestCancelledError:
            raise
        except Exception as e:
            event = _build_metric_event(
                trace_id=trace_id,
                span_id=span_id,
                model=model,
                stream=is_stream,
                latency_ms=(time.time() - t0) * 1000,
                usage=None,
                error=str(e),
                stack_info=stack_info,
            )
            await _emit_metric(event, options)
            raise

    return wrapper


class _MeteredSyncStream:
    """Wraps a sync genai stream to meter it."""

    def __init__(
        self,
        stream: Any,
        trace_id: str,
        span_id: str,
        model: str,
        t0: float,
        options: MeterOptions,
        stack_info: CallStackInfo | None = None,
    ):
        self._stream = iter(stream)
        self._trace_id = trace_id
        self._span_id = span_id
        self._model = model
        self._t0 = t0
        self._stack_info = stack_info
        self._options = options
        self._final_usage: NormalizedUsage | None = None
        self._done = False
        self._error: str | None = None

    def __iter__(self) -> "_MeteredSyncStream":
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._stream)

            # Try to extract usage from chunk
            usage = _extract_usage_from_response(chunk)
            if usage:
                self._final_usage = usage

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
        event = _build_metric_event(
            trace_id=self._trace_id,
            span_id=self._span_id,
            model=self._model,
            stream=True,
            latency_ms=(time.time() - self._t0) * 1000,
            usage=self._final_usage,
            error=self._error,
            stack_info=self._stack_info,
        )
        _emit_metric_sync(event, self._options)


class _MeteredAsyncStream:
    """Wraps an async genai stream to meter it."""

    def __init__(
        self,
        stream: Any,
        trace_id: str,
        span_id: str,
        model: str,
        t0: float,
        options: MeterOptions,
        stack_info: CallStackInfo | None = None,
    ):
        self._stream = stream.__aiter__()
        self._trace_id = trace_id
        self._span_id = span_id
        self._model = model
        self._t0 = t0
        self._options = options
        self._stack_info = stack_info
        self._final_usage: NormalizedUsage | None = None
        self._done = False
        self._error: str | None = None

    def __aiter__(self) -> "_MeteredAsyncStream":
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._stream.__anext__()

            # Try to extract usage from chunk
            usage = _extract_usage_from_response(chunk)
            if usage:
                self._final_usage = usage

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
        event = _build_metric_event(
            trace_id=self._trace_id,
            span_id=self._span_id,
            model=self._model,
            stream=True,
            latency_ms=(time.time() - self._t0) * 1000,
            usage=self._final_usage,
            error=self._error,
            stack_info=self._stack_info,
        )
        await _emit_metric(event, self._options)


def instrument_genai(options: MeterOptions) -> bool:
    """
    Instrument the Google GenAI SDK (google-genai package).

    This is the new SDK used by Google ADK and replaces google-generativeai.

    Args:
        options: Metering options including the metric emitter

    Returns:
        True if instrumentation succeeded, False if SDK not available
    """
    global _is_instrumented, _global_options
    global _original_generate_content, _original_generate_content_async
    global _original_generate_content_stream, _original_generate_content_stream_async

    if _is_instrumented:
        return True

    try:
        from google.genai import models
    except ImportError:
        logger.debug("Google GenAI SDK (google-genai) not available, skipping")
        return False

    _global_options = options

    def get_options() -> MeterOptions | None:
        return _global_options

    try:
        # Get the Models class
        Models = models.Models

        # Store and wrap sync generate_content
        if hasattr(Models, "generate_content"):
            _original_generate_content = Models.generate_content
            Models.generate_content = _create_sync_wrapper(
                _original_generate_content, get_options, is_stream=False
            )

        # Store and wrap sync generate_content_stream
        if hasattr(Models, "generate_content_stream"):
            _original_generate_content_stream = Models.generate_content_stream
            Models.generate_content_stream = _create_sync_wrapper(
                _original_generate_content_stream, get_options, is_stream=True
            )

        # For async, we need to wrap AsyncModels
        try:
            from google.genai import models as async_models_module
            AsyncModels = getattr(async_models_module, "AsyncModels", None)

            if AsyncModels:
                if hasattr(AsyncModels, "generate_content"):
                    _original_generate_content_async = AsyncModels.generate_content
                    AsyncModels.generate_content = _create_async_wrapper(
                        _original_generate_content_async, get_options, is_stream=False
                    )

                if hasattr(AsyncModels, "generate_content_stream"):
                    _original_generate_content_stream_async = AsyncModels.generate_content_stream
                    AsyncModels.generate_content_stream = _create_async_wrapper(
                        _original_generate_content_stream_async, get_options, is_stream=True
                    )
        except Exception as e:
            logger.debug(f"Could not instrument async GenAI methods: {e}")

        _is_instrumented = True
        logger.info("[aden] Google GenAI SDK (google-genai) instrumented")
        return True

    except Exception as e:
        logger.warning(f"Failed to instrument Google GenAI SDK: {e}")
        return False


def uninstrument_genai() -> None:
    """Remove Google GenAI SDK instrumentation."""
    global _is_instrumented, _global_options
    global _original_generate_content, _original_generate_content_async
    global _original_generate_content_stream, _original_generate_content_stream_async

    if not _is_instrumented:
        return

    try:
        from google.genai import models

        Models = models.Models

        if _original_generate_content:
            Models.generate_content = _original_generate_content
        if _original_generate_content_stream:
            Models.generate_content_stream = _original_generate_content_stream

        # Restore async methods
        try:
            AsyncModels = getattr(models, "AsyncModels", None)
            if AsyncModels:
                if _original_generate_content_async:
                    AsyncModels.generate_content = _original_generate_content_async
                if _original_generate_content_stream_async:
                    AsyncModels.generate_content_stream = _original_generate_content_stream_async
        except Exception:
            pass

    except ImportError:
        pass

    _is_instrumented = False
    _global_options = None
    _original_generate_content = None
    _original_generate_content_async = None
    _original_generate_content_stream = None
    _original_generate_content_stream_async = None

    logger.info("[aden] Google GenAI SDK uninstrumented")


def is_genai_instrumented() -> bool:
    """Check if Google GenAI SDK is currently instrumented."""
    return _is_instrumented


def get_genai_options() -> MeterOptions | None:
    """Get current GenAI instrumentation options."""
    return _global_options
