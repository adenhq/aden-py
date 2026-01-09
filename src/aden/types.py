"""
Type definitions for Aden SDK.

These types mirror the TypeScript definitions and provide a consistent
interface for metering LLM API calls across multiple providers
(OpenAI, Anthropic, Gemini).
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable, Literal

# Provider type
Provider = Literal["openai", "anthropic", "gemini"]

# Default control server URL
DEFAULT_CONTROL_SERVER = "https://kube.acho.io"


def get_control_server_url(server_url: str | None = None) -> str:
    """
    Get the control server URL with priority:
    1. Explicit server_url option
    2. ADEN_API_URL environment variable
    3. DEFAULT_CONTROL_SERVER constant
    """
    import os
    return server_url or os.environ.get("ADEN_API_URL") or DEFAULT_CONTROL_SERVER


@dataclass
class NormalizedUsage:
    """
    Normalized usage metrics that work across both API response shapes
    (Responses API vs Chat Completions API).
    """

    input_tokens: int = 0
    """Input/prompt tokens consumed."""

    output_tokens: int = 0
    """Output/completion tokens consumed."""

    total_tokens: int = 0
    """Total tokens (input + output)."""

    reasoning_tokens: int = 0
    """Reasoning tokens used (for o1/o3 models)."""

    cached_tokens: int = 0
    """Tokens served from prompt cache (reduces cost)."""

    accepted_prediction_tokens: int = 0
    """Prediction tokens that were accepted."""

    rejected_prediction_tokens: int = 0
    """Prediction tokens that were rejected."""


@dataclass
class RateLimitInfo:
    """Rate limit information from response headers."""

    remaining_requests: int | None = None
    """Remaining requests in current window."""

    remaining_tokens: int | None = None
    """Remaining tokens in current window."""

    reset_requests: float | None = None
    """Time until request limit resets (seconds)."""

    reset_tokens: float | None = None
    """Time until token limit resets (seconds)."""


# =============================================================================
# Layer 0: Content Capture Types
# =============================================================================


@dataclass
class ContentReference:
    """Reference to large content stored separately via control agent.

    When content exceeds max_content_bytes, it is stored on the server
    and a reference is kept in the MetricEvent for retrieval.
    """

    content_id: str
    """Unique ID for retrieving the full content from server."""

    content_hash: str
    """SHA-256 hash of the full content for verification."""

    byte_size: int
    """Size of the full content in bytes."""

    truncated_preview: str | None = None
    """Truncated preview of the content (for quick inspection)."""


@dataclass
class MessageCapture:
    """Captured message from request or response."""

    role: str
    """Message role: system, user, assistant, tool."""

    content: str | ContentReference | None = None
    """Text content or reference to large content."""

    name: str | None = None
    """Name (for named messages or tool results)."""

    tool_call_id: str | None = None
    """Tool call ID (for tool result messages)."""


@dataclass
class ToolSchemaCapture:
    """Captured tool schema from request."""

    name: str
    """Tool/function name."""

    description: str | None = None
    """Tool description."""

    parameters_schema: dict[str, Any] | ContentReference | None = None
    """JSON Schema for parameters (or reference if large)."""


@dataclass
class RequestParamsCapture:
    """Captured request parameters."""

    temperature: float | None = None
    """Sampling temperature."""

    max_tokens: int | None = None
    """Maximum tokens to generate."""

    top_p: float | None = None
    """Nucleus sampling parameter."""

    frequency_penalty: float | None = None
    """Frequency penalty (OpenAI)."""

    presence_penalty: float | None = None
    """Presence penalty (OpenAI)."""

    stop: list[str] | None = None
    """Stop sequences."""

    seed: int | None = None
    """Random seed for reproducibility."""

    top_k: int | None = None
    """Top-k sampling (Anthropic/Gemini)."""


@dataclass
class ContentCapture:
    """Layer 0: Full content capture for request and response.

    Captures the actual text content flowing through LLM calls,
    enabling governance, debugging, and compliance use cases.
    """

    # === Request Content ===
    system_prompt: str | ContentReference | None = None
    """System prompt (or reference if large)."""

    messages: list[MessageCapture] | ContentReference | None = None
    """Messages array (or reference if large)."""

    tools: list[ToolSchemaCapture] | ContentReference | None = None
    """Tools schema array (or reference if large)."""

    params: RequestParamsCapture | None = None
    """Request parameters (temperature, max_tokens, etc.)."""

    # === Response Content ===
    response_content: str | ContentReference | None = None
    """Response text content (or reference if large)."""

    finish_reason: str | None = None
    """Why the response ended: stop, length, tool_calls, content_filter, etc."""

    choice_count: int | None = None
    """Number of choices in response (for n > 1)."""

    # === Multimodal ===
    has_images: bool = False
    """Whether request contained images."""

    image_urls: list[str] | None = None
    """Image URLs from request (not base64 data)."""


@dataclass
class ContentCaptureOptions:
    """Configuration for content capture behavior."""

    max_content_bytes: int = 4096
    """Maximum bytes before content is stored separately (default 4KB)."""

    capture_system_prompt: bool = True
    """Whether to capture system prompts."""

    capture_messages: bool = True
    """Whether to capture message history."""

    capture_tools_schema: bool = True
    """Whether to capture tools schema."""

    capture_response: bool = True
    """Whether to capture response content."""

    capture_images: bool = True
    """Whether to capture image URLs (never base64)."""

    redact_patterns: list[str] | None = None
    """Regex patterns to redact from captured content."""


# =============================================================================
# Layer 6: Tool Call Deep Inspection Types
# =============================================================================


@dataclass
class ToolCallValidationError:
    """Validation error for a tool call argument."""

    path: str
    """JSON path to the invalid field (e.g., 'properties.name')."""

    message: str
    """Validation error message."""

    expected_type: str | None = None
    """Expected type from schema."""

    actual_type: str | None = None
    """Actual type received."""


@dataclass
class ToolCallCapture:
    """Layer 6: Detailed tool call capture with validation.

    Captures full tool call information including arguments,
    IDs for correlation, and schema validation results.
    """

    id: str
    """Tool call ID for correlation with tool results."""

    name: str
    """Tool/function name."""

    arguments: dict[str, Any] | ContentReference | None = None
    """Parsed arguments (or reference if large)."""

    arguments_raw: str | ContentReference | None = None
    """Raw arguments JSON string (for debugging parse errors)."""

    validation_errors: list[ToolCallValidationError] | None = None
    """Schema validation errors (if any)."""

    is_valid: bool = True
    """Whether arguments passed schema validation."""

    index: int | None = None
    """Index in the tool_calls array (for ordering)."""


@dataclass
class ToolCallMetric:
    """Metric for individual tool calls."""

    type: str
    """Tool type (function, web_search, code_interpreter, etc.)."""

    name: str | None = None
    """Tool/function name."""

    duration_ms: float | None = None
    """Duration of tool execution in ms (if available)."""


@dataclass
class RequestMetadata:
    """Request metadata that affects billing/cost."""

    trace_id: str
    """Unique trace ID for this request."""

    model: str
    """Model used for the request."""

    stream: bool = False
    """Whether streaming was enabled."""

    service_tier: str | None = None
    """Service tier (affects pricing/performance)."""

    max_output_tokens: int | None = None
    """Maximum output tokens cap."""

    max_tool_calls: int | None = None
    """Maximum tool calls allowed."""

    prompt_cache_key: str | None = None
    """Prompt cache key for improved cache hits."""

    prompt_cache_retention: str | None = None
    """Prompt cache retention policy."""


@dataclass
class MetricEvent:
    """
    Complete metric event emitted after each API call.

    All fields are flat (not nested) for consistent cross-provider analytics.
    Uses OpenTelemetry-compatible naming: trace_id groups operations,
    span_id identifies each operation.
    """

    # === Identity (OTel-compatible) ===
    trace_id: str
    """Trace ID grouping related operations (OTel standard)."""

    span_id: str = ""
    """Unique span ID for this specific operation (OTel standard)."""

    parent_span_id: str | None = None
    """Parent span ID for nested/hierarchical calls (OTel standard)."""

    request_id: str | None = None
    """Provider-specific request ID (if available)."""

    provider: Provider = "openai"
    """LLM provider: openai, gemini, anthropic."""

    model: str = ""
    """Model used for the request."""

    stream: bool = False
    """Whether streaming was enabled."""

    timestamp: str = ""
    """ISO timestamp when the request started."""

    # === Performance ===
    latency_ms: float = 0
    """Request latency in milliseconds."""

    status_code: int | None = None
    """HTTP status code (if available)."""

    error: str | None = None
    """Error message if request failed."""

    # === Token Usage (flat, consistent across providers) ===
    input_tokens: int = 0
    """Input/prompt tokens consumed."""

    output_tokens: int = 0
    """Output/completion tokens consumed."""

    total_tokens: int = 0
    """Total tokens (input + output)."""

    cached_tokens: int = 0
    """Tokens served from cache (reduces cost)."""

    reasoning_tokens: int = 0
    """Reasoning tokens used (for o1/o3 models)."""

    # === Rate Limits (flat) ===
    rate_limit_remaining_requests: int | None = None
    """Remaining requests in current window."""

    rate_limit_remaining_tokens: int | None = None
    """Remaining tokens in current window."""

    rate_limit_reset_requests: float | None = None
    """Time until request limit resets (seconds)."""

    rate_limit_reset_tokens: float | None = None
    """Time until token limit resets (seconds)."""

    # === Call Relationship Tracking ===
    call_sequence: int | None = None
    """Sequence number within the trace."""

    agent_stack: list[str] | None = None
    """Stack of agent/handler names leading to this call."""

    # === Call Site (flat) ===
    call_site_file: str | None = None
    """File path where the call originated (immediate caller)."""

    call_site_line: int | None = None
    """Line number where the call originated."""

    call_site_column: int | None = None
    """Column number where the call originated."""

    call_site_function: str | None = None
    """Function name where the call originated."""

    call_stack: list[str] | None = None
    """Full call stack for detailed tracing (file:line:function)."""

    # === Tool Usage ===
    tool_call_count: int | None = None
    """Number of tool calls made."""

    tool_names: str | None = None
    """Tool names that were called (comma-separated)."""

    # === Provider-specific (optional) ===
    service_tier: str | None = None
    """Service tier (OpenAI: auto, default, flex, priority)."""

    metadata: dict[str, str] | None = None
    """Custom metadata attached to the request."""

    # === Layer 0: Content Capture (optional) ===
    content_capture: ContentCapture | None = None
    """Raw content capture (when capture_content=True).

    Contains system prompt, messages, tools schema, and response content.
    Large content is stored on server with ContentReference.
    """

    # === Layer 6: Tool Call Deep Inspection (optional) ===
    tool_calls_captured: list[ToolCallCapture] | None = None
    """Detailed tool call captures (when capture_tool_calls=True).

    Contains full arguments, IDs, and schema validation results.
    Different from tool_call_count/tool_names which are just summaries.
    """

    tool_validation_errors_count: int | None = None
    """Count of tool calls with validation errors."""


@dataclass
class BeforeRequestContext:
    """Context passed to the beforeRequest hook."""

    model: str
    """The model being used for this request."""

    stream: bool
    """Whether this is a streaming request."""

    span_id: str
    """Generated span ID for this request (OTel standard)."""

    trace_id: str
    """Trace ID grouping related operations (OTel standard)."""

    timestamp: datetime
    """Timestamp when the request was initiated."""

    metadata: dict[str, Any] | None = None
    """Custom metadata that can be passed through."""


class BeforeRequestAction(str, Enum):
    """Actions that can be returned from beforeRequest hook."""

    PROCEED = "proceed"
    THROTTLE = "throttle"
    CANCEL = "cancel"
    DEGRADE = "degrade"
    ALERT = "alert"


@dataclass
class BeforeRequestResult:
    """Result from the beforeRequest hook."""

    action: BeforeRequestAction
    """The action to take."""

    delay_ms: int = 0
    """Delay in milliseconds (for throttle action, or combined with degrade/alert)."""

    reason: str = ""
    """Reason for the action (for cancel/degrade actions)."""

    to_model: str = ""
    """Model to degrade to (for degrade action)."""

    level: Literal["info", "warning", "critical"] = "warning"
    """Alert level (for alert action)."""

    message: str = ""
    """Alert message (for alert action)."""

    @classmethod
    def proceed(cls) -> "BeforeRequestResult":
        """Create a proceed result."""
        return cls(action=BeforeRequestAction.PROCEED)

    @classmethod
    def throttle(cls, delay_ms: int) -> "BeforeRequestResult":
        """Create a throttle result."""
        return cls(action=BeforeRequestAction.THROTTLE, delay_ms=delay_ms)

    @classmethod
    def cancel(cls, reason: str) -> "BeforeRequestResult":
        """Create a cancel result."""
        return cls(action=BeforeRequestAction.CANCEL, reason=reason)

    @classmethod
    def degrade(cls, to_model: str, reason: str = "", delay_ms: int = 0) -> "BeforeRequestResult":
        """Create a degrade result to switch to a cheaper model."""
        return cls(action=BeforeRequestAction.DEGRADE, to_model=to_model, reason=reason, delay_ms=delay_ms)

    @classmethod
    def alert(
        cls, message: str, level: Literal["info", "warning", "critical"] = "warning", delay_ms: int = 0
    ) -> "BeforeRequestResult":
        """Create an alert result (proceeds but triggers notification)."""
        return cls(action=BeforeRequestAction.ALERT, message=message, level=level, delay_ms=delay_ms)


# Type aliases for callbacks
MetricEmitter = Callable[[MetricEvent], None | Awaitable[None]]
"""Callback function for emitting metrics."""

BeforeRequestHook = Callable[
    [dict[str, Any], BeforeRequestContext],
    BeforeRequestResult | Awaitable[BeforeRequestResult],
]
"""Hook called before each API request, allowing user-defined rate limiting."""


# Error callback type
EmitErrorHandler = Callable[["MetricEvent", Exception], None]
"""Callback when metric emission fails."""

# Alert callback type
AlertHandler = Callable[
    [dict[str, Any]],  # AlertEvent-like dict with level, message, provider, model, etc.
    None | Awaitable[None],
]
"""Callback when an alert is triggered."""


@dataclass
class SDKClasses:
    """
    SDK classes that can be passed for instrumentation.

    When provided, these classes are used instead of auto-importing.
    This ensures the correct SDK instances are patched, especially
    when multiple versions or custom wrappers are in use.

    Example:
        ```python
        from openai import OpenAI, AsyncOpenAI
        from anthropic import Anthropic, AsyncAnthropic

        await instrument(MeterOptions(
            emit_metric=create_console_emitter(),
            sdks=SDKClasses(
                OpenAI=OpenAI,
                AsyncOpenAI=AsyncOpenAI,
                Anthropic=Anthropic,
                AsyncAnthropic=AsyncAnthropic,
            ),
        ))
        ```
    """

    # OpenAI SDK classes
    OpenAI: Any | None = None
    """The OpenAI client class."""

    AsyncOpenAI: Any | None = None
    """The AsyncOpenAI client class."""

    # Anthropic SDK classes
    Anthropic: Any | None = None
    """The Anthropic client class."""

    AsyncAnthropic: Any | None = None
    """The AsyncAnthropic client class."""

    # Google Generative AI (Gemini) SDK classes
    GenerativeModel: Any | None = None
    """The google.generativeai.GenerativeModel class."""


@dataclass
class MeterOptions:
    """Options for the metered LLM client."""

    emit_metric: MetricEmitter
    """Custom metric emitter function."""

    track_tool_calls: bool = True
    """Whether to include tool call metrics (count and names)."""

    # Layer 0: Content Capture
    capture_content: bool = False
    """Enable Layer 0 raw content capture. Disabled by default for privacy.

    When enabled, captures system prompts, messages, tools schema, and response
    content. Large content is stored on server with ContentReference.
    """

    content_capture_options: ContentCaptureOptions | None = None
    """Configuration for content capture behavior (max bytes, redaction, etc.)."""

    # Layer 6: Tool Call Deep Inspection
    capture_tool_calls: bool = False
    """Enable Layer 6 tool call deep inspection. Disabled by default.

    When enabled, captures full tool call arguments with IDs and
    validates them against the tool schema if validate_tool_schemas=True.
    """

    validate_tool_schemas: bool = True
    """When capture_tool_calls=True, validate arguments against tool schemas.

    Validation errors are reported in ToolCallCapture.validation_errors.
    """

    generate_trace_id: Callable[[], str] | None = None
    """Custom trace ID generator (default: uuid4)."""

    generate_span_id: Callable[[], str] | None = None
    """Custom span ID generator (default: uuid4)."""

    before_request: BeforeRequestHook | None = None
    """Hook called before each request for user-defined rate limiting."""

    request_metadata: dict[str, Any] | None = None
    """Custom metadata to pass to beforeRequest hook."""

    # SDK injection
    sdks: SDKClasses | None = None
    """SDK classes to instrument. If not provided, auto-imports are attempted."""

    # Performance options
    async_emit: bool = False
    """Whether to emit metrics asynchronously (fire-and-forget).

    WARNING: When True, emission errors are silently logged and metrics may be lost.
    Set to True only if you accept potential data loss for lower latency.
    Default changed to False in v0.2.0 for reliability.
    """

    sample_rate: float = 1.0
    """Sampling rate for metrics (0.0-1.0). Values < 1.0 will randomly drop metrics."""

    on_emit_error: EmitErrorHandler | None = None
    """Callback when metric emission fails. Receives the event and exception.

    If not set, errors are logged to the 'aden' logger.
    Use this to implement custom error handling, alerting, or fallback storage.
    """

    on_alert: AlertHandler | None = None
    """Callback when an alert is triggered by the control agent.

    Alerts do NOT block requests - they are notifications only.
    Use this for logging, notifications, or monitoring.
    """

    # Context ID for control agent
    get_context_id: Callable[[], str | None] | None = None
    """Function to get current context ID (user ID, session ID, etc.)."""

    # Control Agent options
    api_key: str | None = None
    """API key for the control server.

    When provided, automatically creates a control agent and emitter.
    If not provided, checks ADEN_API_KEY environment variable.

    Example:
        ```python
        await instrument(MeterOptions(
            api_key=os.environ.get("ADEN_API_KEY"),
            emit_metric=create_console_emitter(),
        ))
        ```
    """

    server_url: str | None = None
    """Control server URL.

    Priority: server_url option > ADEN_API_URL env var > https://kube.acho.io
    Only used when api_key is provided.
    """

    fail_open: bool = True
    """Whether to allow requests when control server is unreachable.

    Default: True (fail open - requests proceed if server is down)
    Set to False for strict control (fail closed - block if server unreachable)
    """


@dataclass
class BudgetConfig:
    """Budget configuration for guardrails.

    NOTE: This is a planned feature and is not yet implemented in the metering logic.
    It is exported for API stability. Use `before_request` hook for budget enforcement.
    """

    max_input_tokens: int | None = None
    """Maximum input tokens allowed per request."""

    max_total_tokens: int | None = None
    """Maximum total tokens allowed per request."""

    on_exceeded: Literal["throw", "truncate", "warn"] = "throw"
    """Action to take when budget is exceeded."""

    on_exceeded_handler: Callable[["BudgetExceededInfo"], None | Awaitable[None]] | None = None
    """Custom handler when budget is exceeded."""


@dataclass
class BudgetExceededInfo:
    """Information about a budget violation."""

    estimated_input_tokens: int
    """Estimated input tokens."""

    max_input_tokens: int
    """Configured maximum."""

    model: str
    """Model being used."""

    input: Any
    """Original input that exceeded budget."""


class RequestCancelledError(Exception):
    """Error thrown when a request is cancelled by the beforeRequest hook."""

    def __init__(self, reason: str, context: BeforeRequestContext):
        super().__init__(f"Request cancelled: {reason}")
        self.reason = reason
        self.context = context


class BudgetExceededError(Exception):
    """Error thrown when a request exceeds the configured budget."""

    def __init__(self, info: BudgetExceededInfo):
        super().__init__(
            f"Budget exceeded: estimated {info.estimated_input_tokens} input tokens, "
            f"max allowed is {info.max_input_tokens} for model {info.model}"
        )
        self.estimated_input_tokens = info.estimated_input_tokens
        self.max_input_tokens = info.max_input_tokens
        self.model = info.model


@dataclass
class InstrumentationResult:
    """Result of calling instrument() - shows which SDKs were instrumented."""

    openai: bool = False
    """Whether OpenAI SDK was instrumented."""

    anthropic: bool = False
    """Whether Anthropic SDK was instrumented."""

    gemini: bool = False
    """Whether Google Gemini SDK (google-generativeai) was instrumented."""

    genai: bool = False
    """Whether Google GenAI SDK (google-genai) was instrumented."""

    gemini_grpc: bool = False
    """Whether Google Gemini gRPC client was instrumented."""

    def __str__(self) -> str:
        instrumented = [name for name, success in [
            ("openai", self.openai),
            ("anthropic", self.anthropic),
            ("gemini", self.gemini),
            ("genai", self.genai),
            ("gemini_grpc", self.gemini_grpc),
        ] if success]
        if instrumented:
            return f"Instrumented: {', '.join(instrumented)}"
        return "No SDKs instrumented"
