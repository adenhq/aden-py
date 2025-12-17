"""
Type definitions for OpenAI Meter.

These types mirror the TypeScript definitions and provide a consistent
interface for metering OpenAI API calls.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable, Literal, Protocol, TypedDict


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
    """Complete metric event emitted after each API call."""

    # From RequestMetadata
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

    # MetricEvent specific fields
    request_id: str | None = None
    """OpenAI request ID for correlation."""

    latency_ms: float = 0
    """Request latency in milliseconds."""

    usage: NormalizedUsage | None = None
    """Normalized usage metrics."""

    status_code: int | None = None
    """HTTP status code (if available)."""

    error: str | None = None
    """Error message if request failed."""

    rate_limit: RateLimitInfo | None = None
    """Rate limit information."""

    tool_calls: list[ToolCallMetric] | None = None
    """Tool calls made during the request."""

    metadata: dict[str, Any] | None = None
    """Custom metadata attached to the request."""


@dataclass
class BeforeRequestContext:
    """Context passed to the beforeRequest hook."""

    model: str
    """The model being used for this request."""

    stream: bool
    """Whether this is a streaming request."""

    trace_id: str
    """Generated trace ID for this request."""

    timestamp: datetime
    """Timestamp when the request was initiated."""

    metadata: dict[str, Any] | None = None
    """Custom metadata that can be passed through."""


class BeforeRequestAction(str, Enum):
    """Actions that can be returned from beforeRequest hook."""

    PROCEED = "proceed"
    THROTTLE = "throttle"
    CANCEL = "cancel"


@dataclass
class BeforeRequestResult:
    """Result from the beforeRequest hook."""

    action: BeforeRequestAction
    """The action to take."""

    delay_ms: int = 0
    """Delay in milliseconds (for throttle action)."""

    reason: str = ""
    """Reason for the action (for cancel action)."""

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


# Type aliases for callbacks
MetricEmitter = Callable[[MetricEvent], None | Awaitable[None]]
"""Callback function for emitting metrics."""

BeforeRequestHook = Callable[
    [dict[str, Any], BeforeRequestContext],
    BeforeRequestResult | Awaitable[BeforeRequestResult],
]
"""Hook called before each API request, allowing user-defined rate limiting."""


@dataclass
class MeterOptions:
    """Options for the metered OpenAI client."""

    emit_metric: MetricEmitter
    """Custom metric emitter function."""

    track_tool_calls: bool = True
    """Whether to include tool call metrics."""

    generate_trace_id: Callable[[], str] | None = None
    """Custom trace ID generator (default: uuid4)."""

    before_request: BeforeRequestHook | None = None
    """Hook called before each request for user-defined rate limiting."""

    request_metadata: dict[str, Any] | None = None
    """Custom metadata to pass to beforeRequest hook."""

    # Performance options
    async_emit: bool = True
    """Whether to emit metrics asynchronously (fire-and-forget)."""

    sample_rate: float = 1.0
    """Sampling rate for metrics (0.0-1.0)."""


@dataclass
class BudgetConfig:
    """Budget configuration for guardrails."""

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
