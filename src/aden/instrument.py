"""
Unified instrumentation for LLM SDKs.

Call `instrument()` once at startup, and all available LLM client instances
(OpenAI, Gemini, Anthropic) are automatically detected and metered.
"""

import asyncio
import atexit
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from uuid import uuid4

from .control_agent import ControlAgent, create_control_agent
from .large_content import set_store_func
from .control_types import (
    ControlAction,
    ControlAgentOptions,
    ControlEvent,
    ControlRequest,
    IControlAgent,
)
from .instrument_anthropic import (
    instrument_anthropic,
    is_anthropic_instrumented,
    uninstrument_anthropic,
)
from .instrument_gemini import (
    instrument_gemini,
    is_gemini_instrumented,
    uninstrument_gemini,
)
from .instrument_gemini_grpc import (
    instrument_gemini_grpc,
    is_gemini_grpc_instrumented,
    uninstrument_gemini_grpc,
)
from .instrument_genai import (
    instrument_genai,
    is_genai_instrumented,
    uninstrument_genai,
)
from .instrument_openai import (
    instrument_openai,
    is_openai_instrumented,
    uninstrument_openai,
)
from .types import (
    BeforeRequestAction,
    BeforeRequestContext,
    BeforeRequestResult,
    InstrumentationResult,
    MeterOptions,
    MetricEvent,
    get_control_server_url,
)

logger = logging.getLogger("aden")

# Track global options and control agent
_global_options: MeterOptions | None = None
_global_control_agent: IControlAgent | None = None

# Background event loop for control agent (keeps Socket.IO alive)
_background_loop: asyncio.AbstractEventLoop | None = None
_background_thread: threading.Thread | None = None
_background_loop_ready = threading.Event()


def _run_background_loop() -> None:
    """Run the background event loop in a dedicated thread."""
    global _background_loop
    _background_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_background_loop)
    _background_loop_ready.set()
    logger.debug("[aden] Background event loop started")
    _background_loop.run_forever()
    logger.debug("[aden] Background event loop stopped")


def _start_background_loop() -> asyncio.AbstractEventLoop:
    """Start the background event loop thread if not already running."""
    global _background_thread, _background_loop

    if _background_thread is not None and _background_thread.is_alive():
        assert _background_loop is not None
        return _background_loop

    _background_loop_ready.clear()
    _background_thread = threading.Thread(
        target=_run_background_loop,
        daemon=True,
        name="aden-control-loop",
    )
    _background_thread.start()
    _background_loop_ready.wait(timeout=5.0)

    if _background_loop is None:
        raise RuntimeError("Failed to start background event loop")

    return _background_loop


def _stop_background_loop() -> None:
    """Stop the background event loop and thread."""
    global _background_loop, _background_thread

    if _background_loop is not None:
        _background_loop.call_soon_threadsafe(_background_loop.stop)

    if _background_thread is not None:
        _background_thread.join(timeout=2.0)
        _background_thread = None

    _background_loop = None


def _run_in_background_loop(coro: Awaitable[Any]) -> Any:
    """Run a coroutine in the background event loop and wait for result."""
    loop = _start_background_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=30.0)


# Register cleanup on process exit
atexit.register(_stop_background_loop)


@dataclass
class InstrumentationResultWithAgent(InstrumentationResult):
    """Extended instrumentation result that includes control agent."""

    control_agent: IControlAgent | None = None
    """The control agent instance if one was created."""


def _create_control_before_request_hook(
    control_agent: "ControlAgent",
    get_context_id: Callable[[], str | None] | None = None,
    user_hook: Callable[..., Any] | None = None,
) -> Callable[[dict[str, Any], BeforeRequestContext], BeforeRequestResult]:
    """
    Creates a beforeRequest hook that integrates with the control agent
    to enforce budget limits, throttling, and model degradation.

    This hook is sync-compatible - it uses the sync version of get_decision
    which just reads from the cached policy (no network calls).
    """

    def hook(
        params: dict[str, Any], context: BeforeRequestContext
    ) -> BeforeRequestResult:
        # First, call user's hook if provided
        if user_hook:
            user_result = user_hook(params, context)
            # If user provided an async hook, we can't handle it here
            if asyncio.iscoroutine(user_result):
                logger.warning("[aden] Async user hook not supported with control agent - skipping")
                user_result = None

            if user_result:
                if user_result.action == BeforeRequestAction.CANCEL:
                    return user_result
                if user_result.action == BeforeRequestAction.DEGRADE:
                    # User's degrade takes precedence
                    return user_result

        # Get decision from control agent (sync - just reads cached policy)
        context_id = get_context_id() if get_context_id else None

        # Extract metadata from request params (e.g., extra_body.metadata)
        # This enables multi-budget matching based on agent, tenant, etc.
        request_metadata = params.get("metadata") or {}
        extra_body = params.get("extra_body") or {}
        if isinstance(extra_body, dict) and "metadata" in extra_body:
            request_metadata = {**request_metadata, **extra_body["metadata"]}

        model = params.get("model", "unknown")
        provider = "openai"  # TODO: detect provider from context

        decision = control_agent.get_decision_sync(
            ControlRequest(
                context_id=context_id,
                provider=provider,
                model=model,
                metadata=request_metadata if request_metadata else None,
            )
        )

        # Report control event to server for non-allow decisions
        if decision.action != ControlAction.ALLOW:
            control_event = ControlEvent(
                trace_id=str(uuid4()),
                span_id=str(uuid4()),
                provider=provider,
                original_model=model,
                action=decision.action,
                context_id=context_id,
                reason=decision.reason,
                degraded_to=decision.degrade_to_model,
                throttle_delay_ms=decision.throttle_delay_ms,
                budget_id=decision.budget_id,
            )
            control_agent.report_control_event_sync(control_event)

        # Map control decision to beforeRequest result
        if decision.action == ControlAction.BLOCK:
            return BeforeRequestResult.cancel(decision.reason or "Budget exceeded")

        if decision.action == ControlAction.THROTTLE:
            logger.info(f"[aden] Request throttled: {decision.reason}")
            return BeforeRequestResult.throttle(decision.throttle_delay_ms or 1000)

        if decision.action == ControlAction.DEGRADE:
            logger.info(
                f"[aden] Model degraded: {model} → {decision.degrade_to_model} ({decision.reason})"
            )
            return BeforeRequestResult.degrade(
                to_model=decision.degrade_to_model or model,
                reason=decision.reason or "",
                delay_ms=decision.throttle_delay_ms or 0,
            )

        if decision.action == ControlAction.ALERT:
            logger.info(f"[aden] Alert [{decision.alert_level}]: {decision.reason}")
            return BeforeRequestResult.alert(
                message=decision.reason or "Alert triggered",
                level=decision.alert_level or "warning",
                delay_ms=decision.throttle_delay_ms or 0,
            )

        # Allow
        return BeforeRequestResult.proceed()

    return hook


def _create_sync_compatible_emitter(
    control_agent: "ControlAgent",
    original_emitter: Callable[..., Any] | None = None,
) -> Callable[[MetricEvent], None]:
    """
    Creates a sync-compatible emitter that works in both sync and async contexts.

    - In sync context: queues metrics for background sending
    - In async context: sends metrics immediately
    """

    def emitter(event: MetricEvent) -> None:
        # Send to control agent (sync - just queues)
        control_agent.report_metric_sync(event)

        # Call original emitter if provided
        if original_emitter:
            result = original_emitter(event)
            if asyncio.iscoroutine(result):
                # Try to schedule in existing loop, otherwise just close it
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(result)
                except RuntimeError:
                    # No running loop - close the coroutine
                    result.close()

    return emitter


async def _resolve_options(options: MeterOptions) -> MeterOptions:
    """
    Resolve options by setting up control agent when api_key is provided.
    Returns a new MeterOptions with control agent integration.
    """
    global _global_control_agent

    # Check for API key (explicit or from environment)
    api_key = options.api_key or os.environ.get("ADEN_API_KEY")

    if api_key:
        # Create control agent
        control_agent = create_control_agent(
            ControlAgentOptions(
                server_url=get_control_server_url(options.server_url),
                api_key=api_key,
                fail_open=options.fail_open,
                get_context_id=options.get_context_id,
                on_alert=options.on_alert,
            )
        )

        # Connect the control agent
        await control_agent.connect()
        _global_control_agent = control_agent

        # Set up large content storage function
        set_store_func(control_agent.store_large_content_sync)

        # Create sync-compatible emitter
        emit_metric = _create_sync_compatible_emitter(control_agent, options.emit_metric)

        # Create beforeRequest hook that checks with control agent (sync)
        before_request = _create_control_before_request_hook(
            control_agent,
            options.get_context_id,
            options.before_request,
        )

        # Create new options with control agent integration
        return MeterOptions(
            emit_metric=emit_metric,
            track_tool_calls=options.track_tool_calls,
            generate_trace_id=options.generate_trace_id,
            generate_span_id=options.generate_span_id,
            before_request=before_request,
            request_metadata=options.request_metadata,
            sdks=options.sdks,
            async_emit=options.async_emit,
            sample_rate=options.sample_rate,
            on_emit_error=options.on_emit_error,
            on_alert=options.on_alert,
            get_context_id=options.get_context_id,
            api_key=api_key,
            server_url=options.server_url,
            fail_open=options.fail_open,
            # Content capture options
            capture_content=options.capture_content,
            content_capture_options=options.content_capture_options,
            capture_tool_calls=options.capture_tool_calls,
            validate_tool_schemas=options.validate_tool_schemas,
        )

    # No API key - just return options as-is
    return options


async def instrument_async(options: MeterOptions) -> InstrumentationResultWithAgent:
    """
    Instrument all available LLM SDKs globally (async version).

    This version should be used when you have an API key and want to
    connect to the control server for budget enforcement.

    Args:
        options: Metering options including the metric emitter

    Returns:
        InstrumentationResultWithAgent showing which SDKs were instrumented
        and the control agent if one was created

    Example:
        ```python
        import asyncio
        from aden import instrument_async, MeterOptions

        async def main():
            result = await instrument_async(MeterOptions(
                api_key="your-api-key",
                emit_metric=create_console_emitter(),
            ))
            print(f"Instrumented: {result}")
            print(f"Control agent connected: {result.control_agent is not None}")

        asyncio.run(main())
        ```
    """
    global _global_options

    # Resolve options (creates control agent if api_key provided)
    resolved_options = await _resolve_options(options)
    _global_options = resolved_options

    # Run all instrumentations
    openai_result = instrument_openai(resolved_options)
    anthropic_result = instrument_anthropic(resolved_options)
    gemini_result = instrument_gemini(resolved_options)
    genai_result = instrument_genai(resolved_options)
    gemini_grpc_result = instrument_gemini_grpc(resolved_options)

    result = InstrumentationResultWithAgent(
        openai=openai_result,
        anthropic=anthropic_result,
        gemini=gemini_result,
        genai=genai_result,
        gemini_grpc=gemini_grpc_result,
        control_agent=_global_control_agent,
    )

    # Log which SDKs were instrumented
    instrumented = []
    if result.openai:
        instrumented.append("openai")
    if result.anthropic:
        instrumented.append("anthropic")
    if result.gemini:
        instrumented.append("gemini")
    if result.genai:
        instrumented.append("genai")
    if result.gemini_grpc:
        instrumented.append("gemini_grpc")

    if instrumented:
        control_status = " + control agent" if _global_control_agent else ""
        logger.info(f"[aden] Instrumented: {', '.join(instrumented)}{control_status}")
    else:
        logger.warning("[aden] No LLM SDKs found to instrument")

    return result


def instrument(options: MeterOptions) -> InstrumentationResult:
    """
    Instrument all available LLM SDKs globally.

    Call once at application startup. All detected LLM client instances
    (OpenAI, Gemini, Anthropic) will automatically be metered.

    The function auto-detects which SDKs are installed and instruments them.
    SDKs that aren't installed are silently skipped.

    NOTE: If you provide an api_key and want to use the control agent,
    use `instrument_async()` instead, or run this in an async context.

    Args:
        options: Metering options including the metric emitter

    Returns:
        InstrumentationResult showing which SDKs were instrumented

    Example:
        ```python
        from aden import instrument, create_console_emitter
        from aden.types import MeterOptions

        # Simple setup with console logging
        result = instrument(MeterOptions(
            emit_metric=create_console_emitter(),
        ))

        # Use any LLM SDK normally - metrics collected automatically
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        ```
    """
    global _global_options

    # Check if we're in an async context and have an api_key
    api_key = options.api_key or os.environ.get("ADEN_API_KEY")

    if api_key:
        # Try to run async version in background loop
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context - can't block, warn user
            logger.warning(
                "[aden] API key provided but called from async context. "
                "Use instrument_async() for control agent support."
            )
        except RuntimeError:
            # No running loop - use background thread for persistent event loop
            # This keeps the Socket.IO connection alive for hot reload
            result = _run_in_background_loop(instrument_async(options))
            return InstrumentationResult(
                openai=result.openai,
                anthropic=result.anthropic,
                gemini=result.gemini,
            )

    # No API key - simple sync instrumentation
    _global_options = options

    # Run all instrumentations
    openai_result = instrument_openai(options)
    anthropic_result = instrument_anthropic(options)
    gemini_result = instrument_gemini(options)
    genai_result = instrument_genai(options)
    gemini_grpc_result = instrument_gemini_grpc(options)

    result = InstrumentationResult(
        openai=openai_result,
        anthropic=anthropic_result,
        gemini=gemini_result,
        genai=genai_result,
        gemini_grpc=gemini_grpc_result,
    )

    # Log which SDKs were instrumented
    instrumented = []
    if result.openai:
        instrumented.append("openai")
    if result.anthropic:
        instrumented.append("anthropic")
    if result.gemini:
        instrumented.append("gemini")
    if result.genai:
        instrumented.append("genai")
    if result.gemini_grpc:
        instrumented.append("gemini_grpc")

    if instrumented:
        logger.info(f"[aden] Instrumented: {', '.join(instrumented)}")
    else:
        logger.warning("[aden] No LLM SDKs found to instrument")

    return result


async def uninstrument_async() -> None:
    """
    Remove instrumentation from all LLM SDKs (async version).

    Disconnects from control server and restores original behavior for all clients.
    """
    global _global_options, _global_control_agent

    # Disconnect control agent if connected
    if _global_control_agent:
        await _global_control_agent.disconnect()
        _global_control_agent = None

    # Clear large content storage function
    set_store_func(None)

    uninstrument_openai()
    uninstrument_anthropic()
    uninstrument_gemini()
    uninstrument_genai()
    uninstrument_gemini_grpc()

    _global_options = None
    logger.info("[aden] All SDKs uninstrumented")


def uninstrument() -> None:
    """
    Remove instrumentation from all LLM SDKs.

    Restores original behavior for all clients.
    """
    global _global_options, _global_control_agent

    # Disconnect control agent if connected
    if _global_control_agent:
        # Use background loop for proper async disconnect if available
        if _background_loop is not None and _background_loop.is_running():
            try:
                _run_in_background_loop(_global_control_agent.disconnect())
            except Exception as e:
                logger.debug(f"[aden] Async disconnect error (ignored): {e}")
        else:
            _global_control_agent.disconnect_sync()
        _global_control_agent = None

    # Clear large content storage function
    set_store_func(None)

    # Stop the background event loop
    _stop_background_loop()

    uninstrument_openai()
    uninstrument_anthropic()
    uninstrument_gemini()
    uninstrument_genai()
    uninstrument_gemini_grpc()

    _global_options = None
    logger.info("[aden] All SDKs uninstrumented")


def get_instrumented_sdks() -> InstrumentationResult:
    """
    Check which SDKs are currently instrumented.

    Returns:
        InstrumentationResult with status of each SDK
    """
    return InstrumentationResult(
        openai=is_openai_instrumented(),
        anthropic=is_anthropic_instrumented(),
        gemini=is_gemini_instrumented(),
        genai=is_genai_instrumented(),
        gemini_grpc=is_gemini_grpc_instrumented(),
    )


def is_instrumented() -> bool:
    """
    Check if any SDK is currently instrumented.

    Returns:
        True if at least one SDK is instrumented
    """
    return (
        is_openai_instrumented()
        or is_anthropic_instrumented()
        or is_gemini_instrumented()
        or is_genai_instrumented()
        or is_gemini_grpc_instrumented()
    )


def get_instrumentation_options() -> MeterOptions | None:
    """
    Get the current instrumentation options.

    Returns:
        The options passed to instrument(), or None if not instrumented
    """
    return _global_options


def get_control_agent() -> IControlAgent | None:
    """
    Get the current control agent instance.

    Returns:
        The control agent if one was created, or None
    """
    return _global_control_agent




def update_instrumentation_options(
    emit_metric: Callable[..., Any] | None = None,
    before_request: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> None:
    """
    Update instrumentation options without re-instrumenting.

    Useful for changing emitters or settings at runtime.

    Args:
        emit_metric: New metric emitter function
        before_request: New before request hook
        **kwargs: Other options to update

    Raises:
        RuntimeError: If no SDK is instrumented
    """
    global _global_options

    if _global_options is None:
        raise RuntimeError(
            "Cannot update options: No LLM SDK is instrumented. Call instrument() first."
        )

    # Update the options object
    if emit_metric is not None:
        _global_options.emit_metric = emit_metric
    if before_request is not None:
        _global_options.before_request = before_request

    # Update any other provided options
    for key, value in kwargs.items():
        if hasattr(_global_options, key):
            setattr(_global_options, key, value)


# Re-export provider-specific functions for advanced use cases
__all__ = [
    # Unified API
    "instrument",
    "instrument_async",
    "uninstrument",
    "uninstrument_async",
    "get_instrumented_sdks",
    "is_instrumented",
    "get_instrumentation_options",
    "get_control_agent",
    "update_instrumentation_options",
    # OpenAI
    "instrument_openai",
    "uninstrument_openai",
    "is_openai_instrumented",
    # Anthropic
    "instrument_anthropic",
    "uninstrument_anthropic",
    "is_anthropic_instrumented",
    # Gemini (google-generativeai)
    "instrument_gemini",
    "uninstrument_gemini",
    "is_gemini_instrumented",
    # GenAI (google-genai)
    "instrument_genai",
    "uninstrument_genai",
    "is_genai_instrumented",
    # Gemini gRPC
    "instrument_gemini_grpc",
    "uninstrument_gemini_grpc",
    "is_gemini_grpc_instrumented",
]
