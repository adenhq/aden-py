"""
Google Gemini SDK instrumentation.

This module provides global instrumentation for the Google Generative AI (Gemini) SDK
by wrapping GenerativeModel methods, so all instances are automatically metered.
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
    extract_gemini_request_content,
    extract_gemini_response_content,
)
from .large_content import store_large_content
from .normalize import normalize_gemini_usage
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
)

logger = logging.getLogger("aden")

# Module-level state
_is_instrumented = False
_global_options: MeterOptions | None = None

# Store original methods for uninstrumentation
_original_get_generative_model: Callable[..., Any] | None = None
_original_generate_content: Callable[..., Any] | None = None
_original_generate_content_async: Callable[..., Any] | None = None


def _get_gemini_class(options: MeterOptions) -> Any | None:
    """Get GenerativeModel or GoogleGenerativeAI class from options or auto-import."""
    if options.sdks and options.sdks.GenerativeModel:
        return options.sdks.GenerativeModel

    # Try auto-import - for Python Gemini SDK, we use google.generativeai
    try:
        import google.generativeai as genai
        return genai
    except ImportError:
        return None


def _extract_model_name(model: Any) -> str:
    """Extract model name from a GenerativeModel instance."""
    if model is None:
        return "unknown"
    if hasattr(model, "model_name"):
        return model.model_name
    if hasattr(model, "_model_name"):
        return model._model_name
    if hasattr(model, "model"):
        return model.model
    return "gemini"


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
    """Builds a MetricEvent for Gemini with flat fields."""
    return MetricEvent(
        trace_id=trace_id,
        span_id=span_id,
        provider="gemini",
        model=model,
        stream=stream,
        timestamp=datetime.now().isoformat(),
        latency_ms=latency_ms,
        request_id=None,  # Gemini doesn't provide request IDs
        error=error,
        # Flatten usage
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


async def _execute_before_request_hook(
    model_name: str,
    context: BeforeRequestContext,
    options: MeterOptions,
) -> BeforeRequestResult:
    """Executes the beforeRequest hook if provided."""
    if options.before_request is None:
        return BeforeRequestResult.proceed()

    # For Gemini, we pass model in params since it's set at model creation
    params = {"model": model_name}
    result = options.before_request(params, context)
    if asyncio.iscoroutine(result):
        result = await result

    return result


def _execute_before_request_hook_sync(
    model_name: str,
    context: BeforeRequestContext,
    options: MeterOptions,
) -> BeforeRequestResult:
    """Executes the beforeRequest hook synchronously."""
    if options.before_request is None:
        return BeforeRequestResult.proceed()

    params = {"model": model_name}
    result = options.before_request(params, context)
    if asyncio.iscoroutine(result):
        # Close the unawaited coroutine to prevent warnings
        result.close()
        return BeforeRequestResult.proceed()

    return result


async def _handle_before_request_result(
    result: BeforeRequestResult,
    context: BeforeRequestContext,
) -> None:
    """Handle the before request result.

    Note: Unlike OpenAI/Anthropic, Gemini's model is set at creation time,
    so the 'degrade' action cannot change the model. It can only apply throttling.
    """
    if result.action == BeforeRequestAction.CANCEL:
        raise RequestCancelledError(result.reason, context)

    if result.action == BeforeRequestAction.THROTTLE:
        await asyncio.sleep(result.delay_ms / 1000)
        return

    if result.action == BeforeRequestAction.DEGRADE:
        # Can't change model for Gemini (set at creation), but can apply throttle
        if result.delay_ms > 0:
            await asyncio.sleep(result.delay_ms / 1000)
        return

    if result.action == BeforeRequestAction.ALERT:
        if result.delay_ms > 0:
            await asyncio.sleep(result.delay_ms / 1000)
        return


def _handle_before_request_result_sync(
    result: BeforeRequestResult,
    context: BeforeRequestContext,
) -> None:
    """Handle the before request result synchronously."""
    if result.action == BeforeRequestAction.CANCEL:
        raise RequestCancelledError(result.reason, context)

    if result.action == BeforeRequestAction.THROTTLE:
        time.sleep(result.delay_ms / 1000)
        return

    if result.action == BeforeRequestAction.DEGRADE:
        if result.delay_ms > 0:
            time.sleep(result.delay_ms / 1000)
        return

    if result.action == BeforeRequestAction.ALERT:
        if result.delay_ms > 0:
            time.sleep(result.delay_ms / 1000)
        return


class MeteredAsyncStreamResponse:
    """Wraps a Gemini async stream response to meter it.

    Gemini streaming returns an iterator of GenerateContentResponse chunks.
    Each chunk may have partial content and the final chunk contains usage_metadata.
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
    ):
        self._stream = stream
        self._trace_id = trace_id
        self._span_id = span_id
        self._model = model
        self._t0 = t0
        self._options = options
        self._stack_info = stack_info
        self._content_capture = content_capture
        self._content_accumulator = StreamContentAccumulator() if content_capture else None
        self._final_usage: NormalizedUsage | None = None
        self._done = False
        self._error: str | None = None

    def __aiter__(self) -> "MeteredAsyncStreamResponse":
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._stream.__anext__()

            # Extract usage_metadata from chunk if present
            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                self._final_usage = normalize_gemini_usage(chunk.usage_metadata)

            # Accumulate content for capture
            if self._content_accumulator:
                try:
                    if hasattr(chunk, "text"):
                        self._content_accumulator.add(chunk.text)
                except Exception:
                    pass

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
        # Finalize content capture
        large_content_payloads: list[dict[str, Any]] = []
        if self._content_capture and self._content_accumulator:
            capture_options = self._options.content_capture_options or ContentCaptureOptions()
            response_payloads = self._content_accumulator.finalize(self._content_capture, capture_options)
            if response_payloads:
                large_content_payloads.extend(response_payloads)

        # Store large content if any
        if large_content_payloads:
            store_large_content(large_content_payloads)

        event = _build_metric_event(
            trace_id=self._trace_id,
            span_id=self._span_id,
            model=self._model,
            stream=True,
            latency_ms=(time.time() - self._t0) * 1000,
            usage=self._final_usage,
            error=self._error,
            stack_info=self._stack_info,
            content_capture=self._content_capture,
        )
        await _emit_metric(event, self._options)


class MeteredSyncStreamResponse:
    """Wraps a Gemini sync stream response to meter it."""

    def __init__(
        self,
        response: Any,
        trace_id: str,
        span_id: str,
        model: str,
        t0: float,
        options: MeterOptions,
        stack_info: CallStackInfo | None = None,
        content_capture: ContentCapture | None = None,
    ):
        self._response = response
        self._iterator: Iterator[Any] | None = None
        self._trace_id = trace_id
        self._span_id = span_id
        self._model = model
        self._t0 = t0
        self._options = options
        self._stack_info = stack_info
        self._content_capture = content_capture
        self._content_accumulator = StreamContentAccumulator() if content_capture else None
        self._final_usage: NormalizedUsage | None = None
        self._done = False
        self._error: str | None = None

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to the underlying response."""
        return getattr(self._response, name)

    def __iter__(self) -> "MeteredSyncStreamResponse":
        self._iterator = iter(self._response)
        return self

    def __next__(self) -> Any:
        if self._iterator is None:
            self._iterator = iter(self._response)
        try:
            chunk = next(self._iterator)

            # Extract usage_metadata from chunk if present
            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                self._final_usage = normalize_gemini_usage(chunk.usage_metadata)

            # Accumulate content for capture
            if self._content_accumulator:
                try:
                    if hasattr(chunk, "text"):
                        self._content_accumulator.add(chunk.text)
                except Exception:
                    pass

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
        # Finalize content capture
        large_content_payloads: list[dict[str, Any]] = []
        if self._content_capture and self._content_accumulator:
            capture_options = self._options.content_capture_options or ContentCaptureOptions()
            response_payloads = self._content_accumulator.finalize(self._content_capture, capture_options)
            if response_payloads:
                large_content_payloads.extend(response_payloads)

        # Store large content if any
        if large_content_payloads:
            store_large_content(large_content_payloads)

        event = _build_metric_event(
            trace_id=self._trace_id,
            span_id=self._span_id,
            model=self._model,
            stream=True,
            latency_ms=(time.time() - self._t0) * 1000,
            usage=self._final_usage,
            error=self._error,
            stack_info=self._stack_info,
            content_capture=self._content_capture,
        )
        _emit_metric_sync(event, self._options)


def _wrap_generate_content(
    original_fn: Callable[..., Any],
    model_instance: Any,
    get_options: Callable[[], MeterOptions | None],
) -> Callable[..., Any]:
    """Wraps GenerativeModel.generate_content for metering."""

    @wraps(original_fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        options = get_options()
        if options is None:
            return original_fn(*args, **kwargs)

        # Capture call stack
        stack_info = capture_call_stack(skip_frames=3)

        model_name = _extract_model_name(model_instance)
        trace_id = options.generate_trace_id() if options.generate_trace_id else str(uuid4())
        span_id = options.generate_span_id() if options.generate_span_id else str(uuid4())
        t0 = time.time()

        # Check for streaming
        stream = kwargs.get("stream", False)

        # Layer 0: Content Capture - extract request content
        content_capture: ContentCapture | None = None
        large_content_payloads: list[dict[str, Any]] = []
        if options.capture_content:
            capture_options = options.content_capture_options or ContentCaptureOptions()
            content_capture, request_payloads = extract_gemini_request_content(args, kwargs, capture_options)
            if request_payloads:
                large_content_payloads.extend(request_payloads)

        context = BeforeRequestContext(
            model=model_name,
            stream=stream,
            span_id=span_id,
            trace_id=trace_id,
            timestamp=datetime.now(),
            metadata=options.request_metadata,
        )

        result = _execute_before_request_hook_sync(model_name, context, options)
        _handle_before_request_result_sync(result, context)

        try:
            response = original_fn(*args, **kwargs)

            # Handle streaming response
            if stream and hasattr(response, "__iter__"):
                # Store large content from request before streaming
                if large_content_payloads:
                    store_large_content(large_content_payloads)
                return MeteredSyncStreamResponse(
                    response, trace_id, span_id, model_name, t0, options,
                    stack_info=stack_info,
                    content_capture=content_capture,
                )

            # Non-streaming response - extract usage
            usage = None
            if hasattr(response, "usage_metadata"):
                usage = normalize_gemini_usage(response.usage_metadata)

            # Layer 0: Extract response content
            if options.capture_content and content_capture:
                capture_options = options.content_capture_options or ContentCaptureOptions()
                response_payloads = extract_gemini_response_content(response, content_capture, capture_options)
                if response_payloads:
                    large_content_payloads.extend(response_payloads)

            # Store all large content payloads
            if large_content_payloads:
                store_large_content(large_content_payloads)

            event = _build_metric_event(
                trace_id=trace_id,
                span_id=span_id,
                model=model_name,
                stream=False,
                latency_ms=(time.time() - t0) * 1000,
                usage=usage,
                stack_info=stack_info,
                content_capture=content_capture,
            )
            _emit_metric_sync(event, options)
            return response

        except Exception as e:
            event = _build_metric_event(
                trace_id=trace_id,
                span_id=span_id,
                model=model_name,
                stream=stream,
                latency_ms=(time.time() - t0) * 1000,
                usage=None,
                error=str(e),
                stack_info=stack_info,
                content_capture=content_capture,
            )
            _emit_metric_sync(event, options)
            raise

    return wrapper


def _wrap_generate_content_async(
    original_fn: Callable[..., Any],
    model_instance: Any,
    get_options: Callable[[], MeterOptions | None],
) -> Callable[..., Any]:
    """Wraps GenerativeModel.generate_content_async for metering."""

    @wraps(original_fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        options = get_options()
        if options is None:
            return await original_fn(*args, **kwargs)

        # Capture call stack before any async operations
        stack_info = capture_call_stack(skip_frames=3)

        model_name = _extract_model_name(model_instance)
        trace_id = options.generate_trace_id() if options.generate_trace_id else str(uuid4())
        span_id = options.generate_span_id() if options.generate_span_id else str(uuid4())
        t0 = time.time()

        stream = kwargs.get("stream", False)

        # Layer 0: Content Capture - extract request content
        content_capture: ContentCapture | None = None
        large_content_payloads: list[dict[str, Any]] = []
        if options.capture_content:
            capture_options = options.content_capture_options or ContentCaptureOptions()
            content_capture, request_payloads = extract_gemini_request_content(args, kwargs, capture_options)
            if request_payloads:
                large_content_payloads.extend(request_payloads)

        context = BeforeRequestContext(
            model=model_name,
            stream=stream,
            span_id=span_id,
            trace_id=trace_id,
            timestamp=datetime.now(),
            metadata=options.request_metadata,
        )

        result = await _execute_before_request_hook(model_name, context, options)
        await _handle_before_request_result(result, context)

        try:
            response = await original_fn(*args, **kwargs)

            # Handle streaming response
            if stream and hasattr(response, "__aiter__"):
                # Store large content from request before streaming
                if large_content_payloads:
                    store_large_content(large_content_payloads)
                return MeteredAsyncStreamResponse(
                    response.__aiter__(), trace_id, span_id, model_name, t0, options,
                    stack_info=stack_info,
                    content_capture=content_capture,
                )

            # Non-streaming response
            usage = None
            if hasattr(response, "usage_metadata"):
                usage = normalize_gemini_usage(response.usage_metadata)

            # Layer 0: Extract response content
            if options.capture_content and content_capture:
                capture_options = options.content_capture_options or ContentCaptureOptions()
                response_payloads = extract_gemini_response_content(response, content_capture, capture_options)
                if response_payloads:
                    large_content_payloads.extend(response_payloads)

            # Store all large content payloads
            if large_content_payloads:
                store_large_content(large_content_payloads)

            event = _build_metric_event(
                trace_id=trace_id,
                span_id=span_id,
                model=model_name,
                stream=False,
                latency_ms=(time.time() - t0) * 1000,
                usage=usage,
                stack_info=stack_info,
                content_capture=content_capture,
            )
            await _emit_metric(event, options)
            return response

        except Exception as e:
            event = _build_metric_event(
                trace_id=trace_id,
                span_id=span_id,
                model=model_name,
                stream=stream,
                latency_ms=(time.time() - t0) * 1000,
                usage=None,
                error=str(e),
                stack_info=stack_info,
                content_capture=content_capture,
            )
            await _emit_metric(event, options)
            raise

    return wrapper


def _wrap_send_message(
    original_fn: Callable[..., Any],
    model_instance: Any,
    get_options: Callable[[], MeterOptions | None],
) -> Callable[..., Any]:
    """Wraps ChatSession.send_message for metering."""

    @wraps(original_fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        options = get_options()
        if options is None:
            return original_fn(*args, **kwargs)

        # Capture call stack
        stack_info = capture_call_stack(skip_frames=3)

        model_name = _extract_model_name(model_instance)
        trace_id = options.generate_trace_id() if options.generate_trace_id else str(uuid4())
        span_id = options.generate_span_id() if options.generate_span_id else str(uuid4())
        t0 = time.time()

        stream = kwargs.get("stream", False)

        try:
            response = original_fn(*args, **kwargs)

            if stream and hasattr(response, "__iter__"):
                return MeteredSyncStreamResponse(
                    response, trace_id, span_id, model_name, t0, options,
                    stack_info=stack_info,
                )

            usage = None
            if hasattr(response, "usage_metadata"):
                usage = normalize_gemini_usage(response.usage_metadata)

            event = _build_metric_event(
                trace_id=trace_id,
                span_id=span_id,
                model=model_name,
                stream=False,
                latency_ms=(time.time() - t0) * 1000,
                usage=usage,
                stack_info=stack_info,
            )
            _emit_metric_sync(event, options)
            return response

        except Exception as e:
            event = _build_metric_event(
                trace_id=trace_id,
                span_id=span_id,
                model=model_name,
                stream=stream,
                latency_ms=(time.time() - t0) * 1000,
                usage=None,
                error=str(e),
                stack_info=stack_info,
            )
            _emit_metric_sync(event, options)
            raise

    return wrapper


def _wrap_send_message_async(
    original_fn: Callable[..., Any],
    model_instance: Any,
    get_options: Callable[[], MeterOptions | None],
) -> Callable[..., Any]:
    """Wraps ChatSession.send_message_async for metering."""

    @wraps(original_fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        options = get_options()
        if options is None:
            return await original_fn(*args, **kwargs)

        # Capture call stack before any async operations
        stack_info = capture_call_stack(skip_frames=3)

        model_name = _extract_model_name(model_instance)
        trace_id = options.generate_trace_id() if options.generate_trace_id else str(uuid4())
        span_id = options.generate_span_id() if options.generate_span_id else str(uuid4())
        t0 = time.time()

        stream = kwargs.get("stream", False)

        try:
            response = await original_fn(*args, **kwargs)

            if stream and hasattr(response, "__aiter__"):
                return MeteredAsyncStreamResponse(
                    response.__aiter__(), trace_id, span_id, model_name, t0, options,
                    stack_info=stack_info,
                )

            usage = None
            if hasattr(response, "usage_metadata"):
                usage = normalize_gemini_usage(response.usage_metadata)

            event = _build_metric_event(
                trace_id=trace_id,
                span_id=span_id,
                model=model_name,
                stream=False,
                latency_ms=(time.time() - t0) * 1000,
                usage=usage,
                stack_info=stack_info,
            )
            await _emit_metric(event, options)
            return response

        except Exception as e:
            event = _build_metric_event(
                trace_id=trace_id,
                span_id=span_id,
                model=model_name,
                stream=stream,
                latency_ms=(time.time() - t0) * 1000,
                usage=None,
                error=str(e),
                stack_info=stack_info,
            )
            await _emit_metric(event, options)
            raise

    return wrapper


def _wrap_chat_session(
    chat: Any,
    model_instance: Any,
    get_options: Callable[[], MeterOptions | None],
) -> Any:
    """Wrap a ChatSession to instrument send_message and send_message_async."""
    # Wrap sync send_message
    if hasattr(chat, "send_message"):
        original_send = chat.send_message
        chat.send_message = _wrap_send_message(original_send, model_instance, get_options)

    # Wrap async send_message_async
    if hasattr(chat, "send_message_async"):
        original_send_async = chat.send_message_async
        chat.send_message_async = _wrap_send_message_async(
            original_send_async, model_instance, get_options
        )

    return chat


def _wrap_start_chat(
    original_fn: Callable[..., Any],
    model_instance: Any,
    get_options: Callable[[], MeterOptions | None],
) -> Callable[..., Any]:
    """Wraps GenerativeModel.start_chat to wrap the returned ChatSession."""

    @wraps(original_fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        chat = original_fn(*args, **kwargs)
        return _wrap_chat_session(chat, model_instance, get_options)

    return wrapper


def _instrument_model_instance(model: Any, get_options: Callable[[], MeterOptions | None]) -> Any:
    """Instrument a GenerativeModel instance with metering wrappers."""
    # Wrap generate_content (sync)
    if hasattr(model, "generate_content"):
        original = model.generate_content
        model.generate_content = _wrap_generate_content(original, model, get_options)

    # Wrap generate_content_async
    if hasattr(model, "generate_content_async"):
        original_async = model.generate_content_async
        model.generate_content_async = _wrap_generate_content_async(
            original_async, model, get_options
        )

    # Wrap start_chat to instrument chat sessions
    if hasattr(model, "start_chat"):
        original_start_chat = model.start_chat
        model.start_chat = _wrap_start_chat(original_start_chat, model, get_options)

    return model


def instrument_gemini(options: MeterOptions) -> bool:
    """
    Instrument the Google Generative AI (Gemini) SDK globally.

    Since the Python Gemini SDK uses `genai.GenerativeModel()` to create models
    directly (not via a client instance), we need to wrap the GenerativeModel class.

    Args:
        options: Metering options including the metric emitter

    Returns:
        True if instrumentation succeeded, False if Gemini SDK not available
    """
    global _is_instrumented, _global_options, _original_get_generative_model

    if _is_instrumented:
        return True

    genai = _get_gemini_class(options)
    if genai is None:
        logger.debug("Google Generative AI SDK not available, skipping instrumentation")
        return False

    _global_options = options

    def get_options() -> MeterOptions | None:
        return _global_options

    # The Python SDK uses genai.GenerativeModel() directly
    # We need to wrap the class to intercept model creation
    try:
        if hasattr(genai, "GenerativeModel"):
            OriginalGenerativeModel = genai.GenerativeModel
            _original_get_generative_model = OriginalGenerativeModel

            class MeteredGenerativeModel(OriginalGenerativeModel):
                """GenerativeModel wrapper that automatically meters all calls."""

                def __init__(self, *args: Any, **kwargs: Any):
                    super().__init__(*args, **kwargs)
                    _instrument_model_instance(self, get_options)

            # Replace the GenerativeModel class
            genai.GenerativeModel = MeteredGenerativeModel
            _is_instrumented = True
            logger.info("[aden] Gemini SDK instrumented")
            return True
        else:
            logger.warning("GenerativeModel not found in google.generativeai")
            return False

    except Exception as e:
        logger.warning(f"Failed to instrument Gemini: {e}")
        return False


def uninstrument_gemini() -> None:
    """
    Remove Gemini SDK instrumentation.

    Restores original GenerativeModel class. Note that existing model instances
    that were created while instrumented will remain instrumented.
    """
    global _is_instrumented, _global_options, _original_get_generative_model

    if not _is_instrumented:
        return

    # Try to restore original class
    try:
        import google.generativeai as genai

        if _original_get_generative_model:
            genai.GenerativeModel = _original_get_generative_model

    except ImportError:
        pass

    _is_instrumented = False
    _global_options = None
    _original_get_generative_model = None

    logger.info("[aden] Gemini SDK uninstrumented")


def is_gemini_instrumented() -> bool:
    """Check if Gemini SDK is currently instrumented."""
    return _is_instrumented


def get_gemini_options() -> MeterOptions | None:
    """Get current Gemini instrumentation options."""
    return _global_options
