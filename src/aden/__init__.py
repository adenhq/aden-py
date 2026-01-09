"""
Aden - Multi-provider LLM SDK metering

Usage tracking, budget enforcement, and cost control for LLM API calls.
Supports OpenAI, Anthropic (Claude), and Google Gemini.
Designed for integration with LiveKit voice agents and other LLM SDK consumers.
"""

# Types
from .types import (
    NormalizedUsage,
    RequestMetadata,
    MetricEvent,
    RateLimitInfo,
    ToolCallMetric,
    MetricEmitter,
    MeterOptions,
    BeforeRequestContext,
    BeforeRequestResult,
    BeforeRequestAction,
    BudgetConfig,
    BudgetExceededInfo,
    RequestCancelledError,
    BudgetExceededError,
    SDKClasses,
    InstrumentationResult,
    Provider,
    DEFAULT_CONTROL_SERVER,
    get_control_server_url,
    # Layer 0: Content Capture
    ContentReference,
    MessageCapture,
    ToolSchemaCapture,
    RequestParamsCapture,
    ContentCapture,
    ContentCaptureOptions,
    # Layer 6: Tool Call Deep Inspection
    ToolCallCapture,
    ToolCallValidationError,
)

# Unified instrumentation API
from .instrument import (
    instrument,
    instrument_async,
    uninstrument,
    uninstrument_async,
    get_instrumented_sdks,
    is_instrumented,
    get_instrumentation_options,
    get_control_agent,
    update_instrumentation_options,
)

# Provider-specific instrumentation
from .instrument_openai import (
    instrument_openai,
    uninstrument_openai,
    is_openai_instrumented,
)
from .instrument_anthropic import (
    instrument_anthropic,
    uninstrument_anthropic,
    is_anthropic_instrumented,
)
from .instrument_gemini import (
    instrument_gemini,
    uninstrument_gemini,
    is_gemini_instrumented,
)
from .instrument_genai import (
    instrument_genai,
    uninstrument_genai,
    is_genai_instrumented,
)

# Control Agent
from .control_types import (
    ControlAction,
    ControlDecision,
    ControlRequest,
    ControlPolicy,
    ControlAgentOptions,
    AlertEvent,
    IControlAgent,
)
from .control_agent import (
    ControlAgent,
    create_control_agent,
    create_control_agent_emitter,
)

# Legacy OpenAI metering (backward compatibility)
from .meter import make_metered_openai, is_metered

# Usage normalization
from .normalize import (
    normalize_usage,
    normalize_openai_usage,
    normalize_anthropic_usage,
    normalize_gemini_usage,
    empty_usage,
    merge_usage,
)

# Emitters
from .emitters import (
    configure_logging,
    create_console_emitter,
    create_batch_emitter,
    create_multi_emitter,
    create_filtered_emitter,
    create_transform_emitter,
    create_noop_emitter,
    create_memory_emitter,
    create_jsonl_emitter,
    JsonlEmitter,
)

# File logging
from .file_logger import MetricFileLogger, create_file_emitter, DEFAULT_LOG_DIR

__version__ = "0.2.0"

__all__ = [
    # Types
    "NormalizedUsage",
    "RequestMetadata",
    "MetricEvent",
    "RateLimitInfo",
    "ToolCallMetric",
    "MetricEmitter",
    "MeterOptions",
    "BeforeRequestContext",
    "BeforeRequestResult",
    "BeforeRequestAction",
    "BudgetConfig",
    "BudgetExceededInfo",
    "SDKClasses",
    "InstrumentationResult",
    "Provider",
    "DEFAULT_CONTROL_SERVER",
    "get_control_server_url",
    # Layer 0: Content Capture
    "ContentReference",
    "MessageCapture",
    "ToolSchemaCapture",
    "RequestParamsCapture",
    "ContentCapture",
    "ContentCaptureOptions",
    # Layer 6: Tool Call Deep Inspection
    "ToolCallCapture",
    "ToolCallValidationError",
    # Errors
    "RequestCancelledError",
    "BudgetExceededError",
    # Unified instrumentation API
    "instrument",
    "instrument_async",
    "uninstrument",
    "uninstrument_async",
    "get_instrumented_sdks",
    "is_instrumented",
    "get_instrumentation_options",
    "get_control_agent",
    "update_instrumentation_options",
    # Provider-specific instrumentation
    "instrument_openai",
    "uninstrument_openai",
    "is_openai_instrumented",
    "instrument_anthropic",
    "uninstrument_anthropic",
    "is_anthropic_instrumented",
    "instrument_gemini",
    "uninstrument_gemini",
    "is_gemini_instrumented",
    "instrument_genai",
    "uninstrument_genai",
    "is_genai_instrumented",
    # Control Agent
    "ControlAction",
    "ControlDecision",
    "ControlRequest",
    "ControlPolicy",
    "ControlAgentOptions",
    "AlertEvent",
    "IControlAgent",
    "ControlAgent",
    "create_control_agent",
    "create_control_agent_emitter",
    # Legacy metering (backward compatibility)
    "make_metered_openai",
    "is_metered",
    # Usage normalization
    "normalize_usage",
    "normalize_openai_usage",
    "normalize_anthropic_usage",
    "normalize_gemini_usage",
    "empty_usage",
    "merge_usage",
    # Logging
    "configure_logging",
    # Emitters
    "create_console_emitter",
    "create_batch_emitter",
    "create_multi_emitter",
    "create_filtered_emitter",
    "create_transform_emitter",
    "create_noop_emitter",
    "create_memory_emitter",
    "create_jsonl_emitter",
    "JsonlEmitter",
    # File logging
    "MetricFileLogger",
    "create_file_emitter",
    "DEFAULT_LOG_DIR",
]
