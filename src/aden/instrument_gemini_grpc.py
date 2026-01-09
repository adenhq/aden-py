"""
Google Gemini gRPC Client instrumentation.

This module patches the low-level GenerativeServiceClient used by LangChain
and other frameworks that don't use the high-level google.generativeai SDK.

Patches:
- google.ai.generativelanguage_v1beta.services.generative_service.GenerativeServiceClient
- google.ai.generativelanguage_v1beta.services.generative_service.GenerativeServiceAsyncClient
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
    extract_gemini_grpc_request_content,
    extract_gemini_grpc_response_content,
)
from .large_content import store_large_content
from .normalize import normalize_gemini_grpc_usage
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

# Store original methods
_original_generate_content: Callable[..., Any] | None = None
_original_generate_content_async: Callable[..., Any] | None = None
_original_stream_generate_content: Callable[..., Any] | None = None
_original_stream_generate_content_async: Callable[..., Any] | None = None


def _extract_model_name(request: Any) -> str:
    """Extract model name from request."""
    if request is None:
        return "unknown"

    # Try dict-style access
    if isinstance(request, dict):
        return request.get('model', 'gemini')

    # Try attribute access
    if hasattr(request, 'model'):
        model = request.model
        # Strip 'models/' prefix if present
        if isinstance(model, str) and model.startswith('models/'):
            return model[7:]
        return model or 'gemini'

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
    """Build a MetricEvent for Gemini gRPC calls."""
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
        call_site_file=stack_info.call_site_file if stack_info else None,
        call_site_line=stack_info.call_site_line if stack_info else None,
        call_site_function=stack_info.call_site_function if stack_info else None,
        call_stack=stack_info.call_stack if stack_info else None,
        agent_stack=stack_info.agent_stack if stack_info else None,
        content_capture=content_capture,
    )


def _emit_metric_sync(event: MetricEvent, options: MeterOptions) -> None:
    """Emit metric synchronously."""
    try:
        result = options.emit_metric(event)
        if asyncio.iscoroutine(result):
            result.close()
    except Exception as e:
        if options.on_emit_error:
            options.on_emit_error(event, e)
        else:
            logger.error(f"Error emitting metric (trace_id={event.trace_id}): {e}")


async def _emit_metric_async(event: MetricEvent, options: MeterOptions) -> None:
    """Emit metric asynchronously."""
    try:
        result = options.emit_metric(event)
        if asyncio.iscoroutine(result):
            await result
    except Exception as e:
        if options.on_emit_error:
            options.on_emit_error(event, e)
        else:
            logger.error(f"Error emitting metric (trace_id={event.trace_id}): {e}")


def _wrap_generate_content_sync(original_fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap GenerativeServiceClient.generate_content."""

    @wraps(original_fn)
    def wrapper(self, request=None, *args, **kwargs):
        options = _global_options
        if options is None:
            return original_fn(self, request, *args, **kwargs)

        stack_info = capture_call_stack(skip_frames=3)
        model_name = _extract_model_name(request)
        trace_id = options.generate_trace_id() if options.generate_trace_id else str(uuid4())
        span_id = options.generate_span_id() if options.generate_span_id else str(uuid4())
        t0 = time.time()

        # Layer 0: Content Capture - extract request content before API call
        content_capture: ContentCapture | None = None
        large_content_payloads: list[dict[str, Any]] = []
        if options.capture_content:
            capture_options = options.content_capture_options or ContentCaptureOptions()
            content_capture, request_payloads = extract_gemini_grpc_request_content(
                request, capture_options
            )
            if request_payloads:
                large_content_payloads.extend(request_payloads)

        try:
            response = original_fn(self, request, *args, **kwargs)

            usage = None
            if hasattr(response, 'usage_metadata'):
                usage_metadata = response.usage_metadata
                logger.debug(
                    f"[aden] gRPC usage_metadata: prompt_token_count={getattr(usage_metadata, 'prompt_token_count', None)}, "
                    f"candidates_token_count={getattr(usage_metadata, 'candidates_token_count', None)}, "
                    f"total_token_count={getattr(usage_metadata, 'total_token_count', None)}"
                )
                usage = normalize_gemini_grpc_usage(usage_metadata)
            else:
                logger.debug(f"[aden] gRPC response has no usage_metadata. Response attrs: {[a for a in dir(response) if not a.startswith('_')][:15]}")

            # Layer 0: Extract response content
            if options.capture_content and content_capture:
                capture_options = options.content_capture_options or ContentCaptureOptions()
                response_payloads = extract_gemini_grpc_response_content(
                    response, content_capture, capture_options
                )
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
                stream=False,
                latency_ms=(time.time() - t0) * 1000,
                usage=None,
                error=str(e),
                stack_info=stack_info,
                content_capture=content_capture,
            )
            _emit_metric_sync(event, options)
            raise

    return wrapper


def _wrap_generate_content_async(original_fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap GenerativeServiceAsyncClient.generate_content."""

    @wraps(original_fn)
    async def wrapper(self, request=None, *args, **kwargs):
        options = _global_options
        if options is None:
            return await original_fn(self, request, *args, **kwargs)

        stack_info = capture_call_stack(skip_frames=3)
        model_name = _extract_model_name(request)
        trace_id = options.generate_trace_id() if options.generate_trace_id else str(uuid4())
        span_id = options.generate_span_id() if options.generate_span_id else str(uuid4())
        t0 = time.time()

        # Layer 0: Content Capture - extract request content before API call
        content_capture: ContentCapture | None = None
        large_content_payloads: list[dict[str, Any]] = []
        if options.capture_content:
            capture_options = options.content_capture_options or ContentCaptureOptions()
            content_capture, request_payloads = extract_gemini_grpc_request_content(
                request, capture_options
            )
            if request_payloads:
                large_content_payloads.extend(request_payloads)

        try:
            response = await original_fn(self, request, *args, **kwargs)

            usage = None
            if hasattr(response, 'usage_metadata'):
                usage = normalize_gemini_grpc_usage(response.usage_metadata)

            # Layer 0: Extract response content
            if options.capture_content and content_capture:
                capture_options = options.content_capture_options or ContentCaptureOptions()
                response_payloads = extract_gemini_grpc_response_content(
                    response, content_capture, capture_options
                )
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
            await _emit_metric_async(event, options)
            return response

        except Exception as e:
            event = _build_metric_event(
                trace_id=trace_id,
                span_id=span_id,
                model=model_name,
                stream=False,
                latency_ms=(time.time() - t0) * 1000,
                usage=None,
                error=str(e),
                stack_info=stack_info,
                content_capture=content_capture,
            )
            await _emit_metric_async(event, options)
            raise

    return wrapper


def instrument_gemini_grpc(options: MeterOptions) -> bool:
    """
    Instrument the low-level Gemini gRPC client.

    This patches GenerativeServiceClient and GenerativeServiceAsyncClient
    which are used by LangChain and other frameworks.

    Args:
        options: Metering options including the metric emitter

    Returns:
        True if instrumentation succeeded, False if SDK not available
    """
    global _is_instrumented, _global_options
    global _original_generate_content, _original_generate_content_async

    if _is_instrumented:
        return True

    try:
        from google.ai.generativelanguage_v1beta.services.generative_service import (
            GenerativeServiceClient,
            GenerativeServiceAsyncClient,
        )
    except ImportError:
        logger.debug("Gemini gRPC client not available, skipping instrumentation")
        return False

    _global_options = options

    # Patch sync client
    _original_generate_content = GenerativeServiceClient.generate_content
    GenerativeServiceClient.generate_content = _wrap_generate_content_sync(
        _original_generate_content
    )

    # Patch async client
    _original_generate_content_async = GenerativeServiceAsyncClient.generate_content
    GenerativeServiceAsyncClient.generate_content = _wrap_generate_content_async(
        _original_generate_content_async
    )

    _is_instrumented = True
    logger.info("[aden] Gemini gRPC client instrumented")
    return True


def uninstrument_gemini_grpc() -> None:
    """Remove Gemini gRPC client instrumentation."""
    global _is_instrumented, _global_options
    global _original_generate_content, _original_generate_content_async

    if not _is_instrumented:
        return

    try:
        from google.ai.generativelanguage_v1beta.services.generative_service import (
            GenerativeServiceClient,
            GenerativeServiceAsyncClient,
        )

        if _original_generate_content:
            GenerativeServiceClient.generate_content = _original_generate_content
        if _original_generate_content_async:
            GenerativeServiceAsyncClient.generate_content = _original_generate_content_async

    except ImportError:
        pass

    _is_instrumented = False
    _global_options = None
    _original_generate_content = None
    _original_generate_content_async = None

    logger.info("[aden] Gemini gRPC client uninstrumented")


def is_gemini_grpc_instrumented() -> bool:
    """Check if Gemini gRPC client is instrumented."""
    return _is_instrumented
