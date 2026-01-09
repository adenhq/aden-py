"""
Dify SDK instrumentation.

This module provides global instrumentation for the Dify SDK by patching
the client classes, so all instances are automatically metered.

Dify is a platform where LLM calls happen on the server side. The SDK
calls Dify's API, and the response includes usage metadata from the server.
"""

import logging
import time
from functools import wraps
from typing import Any, Callable, Iterator
from uuid import uuid4

from datetime import datetime, timezone

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
_original_completion_create: Callable[..., Any] | None = None
_original_workflow_run: Callable[..., Any] | None = None


def _get_dify_classes() -> tuple[Any, Any, Any] | None:
    """Get Dify client classes from auto-import."""
    try:
        from dify_client import ChatClient, CompletionClient
        # WorkflowClient might not exist in all versions
        try:
            from dify_client import WorkflowClient
        except ImportError:
            WorkflowClient = None
        return (ChatClient, CompletionClient, WorkflowClient)
    except ImportError:
        return None


def _extract_usage_from_response(response: Any) -> dict[str, Any] | None:
    """
    Extract usage information from Dify API response.

    Dify response format:
    {
        "metadata": {
            "usage": {
                "prompt_tokens": 1033,
                "completion_tokens": 128,
                "total_tokens": 1161,
                "prompt_price": "0.0010330",
                "completion_price": "0.0002560",
                "total_price": "0.0012890",
                "currency": "USD",
                "latency": 0.768
            }
        }
    }
    """
    if response is None:
        return None

    # Handle requests.Response object
    if hasattr(response, 'json'):
        try:
            data = response.json()
        except Exception:
            return None
    elif isinstance(response, dict):
        data = response
    else:
        return None

    metadata = data.get("metadata", {})
    usage = metadata.get("usage", {})

    if not usage:
        return None

    return usage


def _normalize_dify_usage(usage: dict[str, Any]) -> NormalizedUsage:
    """
    Normalize Dify usage to standard format.

    Dify already provides calculated prices, which is convenient.
    """
    return NormalizedUsage(
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        total_tokens=usage.get("total_tokens", 0),
    )


def _extract_response_info(response: Any) -> dict[str, Any]:
    """Extract relevant info from Dify response."""
    info = {
        "message_id": None,
        "conversation_id": None,
        "answer": None,
        "model": None,
    }

    if response is None:
        return info

    # Handle requests.Response object
    if hasattr(response, 'json'):
        try:
            data = response.json()
        except Exception:
            return info
    elif isinstance(response, dict):
        data = response
    else:
        return info

    info["message_id"] = data.get("message_id") or data.get("id")
    info["conversation_id"] = data.get("conversation_id")
    info["answer"] = data.get("answer")

    # Model might be in metadata
    metadata = data.get("metadata", {})
    info["model"] = metadata.get("model") or metadata.get("model_id")

    return info


def _create_wrapper(
    original_method: Callable[..., Any],
    method_type: str,  # "chat", "completion", "workflow"
    options: MeterOptions,
) -> Callable[..., Any]:
    """Create a wrapper for Dify client methods."""

    @wraps(original_method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        trace_id = (
            options.generate_trace_id()
            if options.generate_trace_id
            else str(uuid4())
        )
        span_id = (
            options.generate_span_id()
            if options.generate_span_id
            else str(uuid4())
        )

        # Extract request info
        query = kwargs.get("query", args[1] if len(args) > 1 else "")
        user = kwargs.get("user", args[2] if len(args) > 2 else "unknown")
        response_mode = kwargs.get("response_mode", "blocking")

        # Create before-request context
        request_params = {
            "query": query,
            "user": user,
            "response_mode": response_mode,
            "method_type": method_type,
            **kwargs,
        }

        context = BeforeRequestContext(
            model=f"dify-{method_type}",  # We don't know the actual model until response
            stream=(response_mode == "streaming"),
            trace_id=trace_id,
            span_id=span_id,
            timestamp=datetime.now(timezone.utc),
        )

        # Call before_request hook if provided
        if options.before_request:
            try:
                result = options.before_request(request_params, context)
                if result and result.action == BeforeRequestAction.CANCEL:
                    raise RequestCancelledError(
                        result.reason or "Request cancelled by before_request hook"
                    )
            except RequestCancelledError:
                raise
            except Exception as e:
                logger.warning(f"[aden] before_request hook error: {e}")

        # Make the actual request
        start_time = time.perf_counter()
        error_info = None
        response = None

        try:
            response = original_method(self, *args, **kwargs)
        except Exception as e:
            error_info = {
                "type": type(e).__name__,
                "message": str(e),
            }
            raise
        finally:
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000

            # Extract usage and response info
            usage_data = _extract_usage_from_response(response)
            response_info = _extract_response_info(response)

            # Build metric event
            if usage_data:
                normalized = _normalize_dify_usage(usage_data)

                # Extract cost info from Dify (they provide it as strings)
                input_cost = None
                output_cost = None
                total_cost = None
                try:
                    if usage_data.get("prompt_price"):
                        input_cost = float(usage_data["prompt_price"])
                    if usage_data.get("completion_price"):
                        output_cost = float(usage_data["completion_price"])
                    if usage_data.get("total_price"):
                        total_cost = float(usage_data["total_price"])
                except (ValueError, TypeError):
                    pass

                event = MetricEvent(
                    trace_id=trace_id,
                    span_id=span_id,
                    provider="dify",
                    model=response_info.get("model") or f"dify-{method_type}",
                    stream=(response_mode == "streaming"),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    input_tokens=normalized.input_tokens,
                    output_tokens=normalized.output_tokens,
                    total_tokens=normalized.total_tokens,
                    input_cost=input_cost,
                    output_cost=output_cost,
                    total_cost=total_cost,
                    currency=usage_data.get("currency"),
                    latency_ms=latency_ms,
                    request_id=response_info.get("message_id"),
                    error=str(error_info) if error_info else None,
                )

                # Emit metric
                if options.emit_metric:
                    try:
                        options.emit_metric(event)
                    except Exception as e:
                        if options.on_emit_error:
                            options.on_emit_error(e, event)
                        else:
                            logger.warning(f"[aden] emit_metric error: {e}")
            elif error_info:
                # Still emit on error even without usage
                event = MetricEvent(
                    trace_id=trace_id,
                    span_id=span_id,
                    provider="dify",
                    model=f"dify-{method_type}",
                    stream=(response_mode == "streaming"),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    latency_ms=latency_ms,
                    error=str(error_info),
                )

                if options.emit_metric:
                    try:
                        options.emit_metric(event)
                    except Exception as e:
                        if options.on_emit_error:
                            options.on_emit_error(e, event)

        return response

    return wrapper


def _create_streaming_wrapper(
    original_method: Callable[..., Any],
    method_type: str,
    options: MeterOptions,
) -> Callable[..., Any]:
    """
    Create a wrapper for streaming Dify responses.

    For streaming, we need to accumulate the response and extract
    usage from the final message.
    """

    @wraps(original_method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        response_mode = kwargs.get("response_mode", "blocking")

        # If not streaming, use regular wrapper logic
        if response_mode != "streaming":
            return _create_wrapper(original_method, method_type, options)(
                self, *args, **kwargs
            )

        trace_id = (
            options.generate_trace_id()
            if options.generate_trace_id
            else str(uuid4())
        )
        span_id = (
            options.generate_span_id()
            if options.generate_span_id
            else str(uuid4())
        )

        query = kwargs.get("query", args[1] if len(args) > 1 else "")
        user = kwargs.get("user", args[2] if len(args) > 2 else "unknown")

        start_time = time.perf_counter()

        # Get the streaming response
        response = original_method(self, *args, **kwargs)

        # Wrap the iterator to capture the final message with usage
        def stream_wrapper() -> Iterator[Any]:
            final_usage = None
            final_response_info = {}
            error_info = None

            try:
                for chunk in response.iter_lines():
                    if chunk:
                        # Yield the chunk to the caller
                        yield chunk

                        # Try to parse for usage info (usually in the last message)
                        try:
                            import json
                            # Remove "data: " prefix if present
                            chunk_str = chunk.decode('utf-8') if isinstance(chunk, bytes) else chunk
                            if chunk_str.startswith("data: "):
                                chunk_str = chunk_str[6:]

                            data = json.loads(chunk_str)

                            # Check for usage in this chunk
                            if "metadata" in data and "usage" in data["metadata"]:
                                final_usage = data["metadata"]["usage"]
                                final_response_info = {
                                    "message_id": data.get("message_id"),
                                    "conversation_id": data.get("conversation_id"),
                                    "model": data.get("metadata", {}).get("model"),
                                }
                        except Exception:
                            pass
            except Exception as e:
                error_info = {
                    "type": type(e).__name__,
                    "message": str(e),
                }
                raise
            finally:
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000

                if final_usage:
                    normalized = _normalize_dify_usage(final_usage)

                    event = MetricEvent(
                        trace_id=trace_id,
                        span_id=span_id,
                        provider="dify",
                        model=final_response_info.get("model") or f"dify-{method_type}",
                        operation=f"{method_type}.create",
                        prompt_tokens=normalized.prompt_tokens,
                        completion_tokens=normalized.completion_tokens,
                        total_tokens=normalized.total_tokens,
                        prompt_cost=normalized.prompt_cost,
                        completion_cost=normalized.completion_cost,
                        total_cost=normalized.total_cost,
                        latency_ms=latency_ms,
                        request_id=final_response_info.get("message_id"),
                        error=error_info,
                        metadata={
                            "conversation_id": final_response_info.get("conversation_id"),
                            "user": user,
                            "response_mode": "streaming",
                            "currency": final_usage.get("currency", "USD"),
                        },
                    )

                    if options.emit_metric:
                        try:
                            options.emit_metric(event)
                        except Exception as e:
                            if options.on_emit_error:
                                options.on_emit_error(e, event)

        return stream_wrapper()

    return wrapper


def instrument_dify(options: MeterOptions) -> bool:
    """
    Instrument the Dify SDK globally.

    Args:
        options: Metering options including the metric emitter

    Returns:
        True if instrumentation was successful, False if SDK not found
    """
    global _is_instrumented, _global_options
    global _original_chat_create, _original_completion_create, _original_workflow_run

    if _is_instrumented:
        logger.debug("[aden] Dify already instrumented")
        return True

    classes = _get_dify_classes()
    if classes is None:
        logger.debug("[aden] dify_client not installed, skipping instrumentation")
        return False

    ChatClient, CompletionClient, WorkflowClient = classes
    _global_options = options

    # Patch ChatClient.create_chat_message
    if ChatClient and hasattr(ChatClient, 'create_chat_message'):
        _original_chat_create = ChatClient.create_chat_message
        ChatClient.create_chat_message = _create_wrapper(
            _original_chat_create, "chat", options
        )
        logger.debug("[aden] Patched ChatClient.create_chat_message")

    # Patch CompletionClient.create_completion_message
    if CompletionClient and hasattr(CompletionClient, 'create_completion_message'):
        _original_completion_create = CompletionClient.create_completion_message
        CompletionClient.create_completion_message = _create_wrapper(
            _original_completion_create, "completion", options
        )
        logger.debug("[aden] Patched CompletionClient.create_completion_message")

    # Patch WorkflowClient.run if available
    if WorkflowClient and hasattr(WorkflowClient, 'run'):
        _original_workflow_run = WorkflowClient.run
        WorkflowClient.run = _create_wrapper(
            _original_workflow_run, "workflow", options
        )
        logger.debug("[aden] Patched WorkflowClient.run")

    _is_instrumented = True
    logger.info("[aden] Dify SDK instrumented")
    return True


def uninstrument_dify() -> None:
    """Remove Dify SDK instrumentation."""
    global _is_instrumented, _global_options
    global _original_chat_create, _original_completion_create, _original_workflow_run

    if not _is_instrumented:
        return

    classes = _get_dify_classes()
    if classes is None:
        return

    ChatClient, CompletionClient, WorkflowClient = classes

    # Restore original methods
    if _original_chat_create and ChatClient:
        ChatClient.create_chat_message = _original_chat_create
        _original_chat_create = None

    if _original_completion_create and CompletionClient:
        CompletionClient.create_completion_message = _original_completion_create
        _original_completion_create = None

    if _original_workflow_run and WorkflowClient:
        WorkflowClient.run = _original_workflow_run
        _original_workflow_run = None

    _is_instrumented = False
    _global_options = None
    logger.info("[aden] Dify SDK uninstrumented")


def is_dify_instrumented() -> bool:
    """Check if Dify SDK is currently instrumented."""
    return _is_instrumented
