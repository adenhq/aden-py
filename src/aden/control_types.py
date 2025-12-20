"""
Control Types - Types for the Control Agent.

Defines control actions, events, and policies for bidirectional
communication with the control server.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable, Literal, Protocol

from .types import MetricEvent


# =============================================================================
# Control Actions
# =============================================================================

class ControlAction(str, Enum):
    """
    Control actions that can be applied to requests.
    - allow: Request proceeds normally
    - block: Request is rejected
    - throttle: Request is delayed before proceeding
    - degrade: Request uses a cheaper/fallback model
    - alert: Request proceeds but triggers an alert notification
    """
    ALLOW = "allow"
    BLOCK = "block"
    THROTTLE = "throttle"
    DEGRADE = "degrade"
    ALERT = "alert"


@dataclass
class ControlDecision:
    """Control decision - what action to take for a request."""

    action: ControlAction
    """The action to take."""

    reason: str | None = None
    """Human-readable reason for the decision."""

    degrade_to_model: str | None = None
    """If action is 'degrade', switch to this model."""

    throttle_delay_ms: int | None = None
    """If action is 'throttle', delay by this many milliseconds."""

    alert_level: Literal["info", "warning", "critical"] | None = None
    """If action is 'alert', the severity level."""


# =============================================================================
# Control Events (SDK → Server)
# =============================================================================

@dataclass
class ControlEvent:
    """Control event - emitted when a control action is taken."""

    trace_id: str
    """Trace ID for correlation."""

    span_id: str
    """Span ID of the affected request."""

    provider: str
    """Provider (openai, anthropic, gemini)."""

    original_model: str
    """Original model that was requested."""

    action: ControlAction
    """Action that was taken."""

    event_type: str = "control"
    """Event type discriminator."""

    timestamp: str = ""
    """ISO timestamp of the event."""

    sdk_instance_id: str = ""
    """SDK instance ID for tracking."""

    context_id: str | None = None
    """Context ID (user, session, deal, etc.)."""

    reason: str | None = None
    """Reason for the action."""

    degraded_to: str | None = None
    """If degraded, what model was used instead."""

    throttle_delay_ms: int | None = None
    """If throttled, how long was the delay in ms."""

    estimated_cost: float | None = None
    """Estimated cost that triggered the decision."""


@dataclass
class MetricEventWrapper:
    """Metric event wrapper for server emission."""

    data: MetricEvent
    """The actual metric data."""

    event_type: str = "metric"
    """Event type discriminator."""

    timestamp: str = ""
    """ISO timestamp of the event."""

    sdk_instance_id: str = ""
    """SDK instance ID for tracking."""


@dataclass
class HeartbeatEvent:
    """Heartbeat event - periodic health check."""

    status: Literal["healthy", "degraded", "reconnecting"]
    """Connection status."""

    requests_since_last: int
    """Requests processed since last heartbeat."""

    errors_since_last: int
    """Errors since last heartbeat."""

    policy_cache_age_seconds: int
    """Current policy cache age in seconds."""

    websocket_connected: bool
    """Whether WebSocket is connected."""

    sdk_version: str
    """SDK version."""

    event_type: str = "heartbeat"
    """Event type discriminator."""

    timestamp: str = ""
    """ISO timestamp of the event."""

    sdk_instance_id: str = ""
    """SDK instance ID for tracking."""


@dataclass
class ErrorEvent:
    """Error event - emitted when an error occurs."""

    message: str
    """Error message."""

    event_type: str = "error"
    """Event type discriminator."""

    timestamp: str = ""
    """ISO timestamp of the event."""

    sdk_instance_id: str = ""
    """SDK instance ID for tracking."""

    code: str | None = None
    """Error code (if available)."""

    stack: str | None = None
    """Stack trace (if available)."""

    trace_id: str | None = None
    """Related trace ID (if applicable)."""


# Union type for all events
ServerEvent = ControlEvent | MetricEventWrapper | HeartbeatEvent | ErrorEvent


# =============================================================================
# Control Policies (Server → SDK)
# =============================================================================

@dataclass
class BudgetRule:
    """Budget rule - limits spend per context."""

    context_id: str
    """Context ID this rule applies to (e.g., user_id, session_id)."""

    limit_usd: float
    """Budget limit in USD."""

    current_spend_usd: float = 0.0
    """Current spend in USD (server tracks this)."""

    action_on_exceed: ControlAction = ControlAction.BLOCK
    """Action to take when budget is exceeded."""

    degrade_to_model: str | None = None
    """If action is 'degrade', switch to this model."""


@dataclass
class ThrottleRule:
    """Throttle rule - rate limiting."""

    context_id: str | None = None
    """Context ID this rule applies to (omit for global)."""

    provider: str | None = None
    """Provider this rule applies to (omit for all)."""

    requests_per_minute: int | None = None
    """Maximum requests per minute."""

    delay_ms: int | None = None
    """Fixed delay to apply to each request (ms)."""


@dataclass
class BlockRule:
    """Block rule - hard block on certain requests."""

    reason: str
    """Reason shown to caller."""

    context_id: str | None = None
    """Context ID to block (omit for pattern match)."""

    provider: str | None = None
    """Provider to block (omit for all)."""

    model_pattern: str | None = None
    """Model pattern to block (e.g., 'gpt-4*')."""


@dataclass
class DegradeRule:
    """Degrade rule - automatic model downgrade."""

    from_model: str
    """Model to downgrade from."""

    to_model: str
    """Model to downgrade to."""

    trigger: Literal["budget_threshold", "rate_limit", "always"]
    """When to trigger the downgrade."""

    threshold_percent: float | None = None
    """For budget_threshold: percentage at which to trigger (0-100)."""

    context_id: str | None = None
    """Context ID this rule applies to (omit for all)."""


@dataclass
class AlertRule:
    """Alert rule - trigger notifications without blocking."""

    trigger: Literal["budget_threshold", "model_usage", "always"]
    """When to trigger the alert."""

    level: Literal["info", "warning", "critical"]
    """Alert severity level."""

    message: str
    """Message to include in the alert."""

    context_id: str | None = None
    """Context ID this rule applies to (omit for global)."""

    provider: str | None = None
    """Provider this rule applies to (omit for all)."""

    model_pattern: str | None = None
    """Model pattern to alert on (e.g., 'gpt-4*' for expensive models)."""

    threshold_percent: float | None = None
    """For budget_threshold: percentage at which to trigger (0-100)."""


@dataclass
class ControlPolicy:
    """Complete control policy from server."""

    version: str
    """Policy version for cache invalidation."""

    updated_at: str
    """When this policy was last updated."""

    budgets: list[BudgetRule] = field(default_factory=list)
    """Budget rules."""

    throttles: list[ThrottleRule] = field(default_factory=list)
    """Throttle rules."""

    blocks: list[BlockRule] = field(default_factory=list)
    """Block rules."""

    degradations: list[DegradeRule] = field(default_factory=list)
    """Degrade rules."""

    alerts: list[AlertRule] = field(default_factory=list)
    """Alert rules."""


# =============================================================================
# Control Request (for getting decisions)
# =============================================================================

@dataclass
class ControlRequest:
    """Request context for getting a control decision."""

    provider: str
    """Provider being called."""

    model: str
    """Model being requested."""

    context_id: str | None = None
    """Context ID (user, session, deal, etc.)."""

    estimated_cost: float | None = None
    """Estimated cost of this request in USD."""

    estimated_input_tokens: int | None = None
    """Estimated input tokens."""

    metadata: dict[str, Any] | None = None
    """Custom metadata."""


# =============================================================================
# Control Agent Options
# =============================================================================

@dataclass
class AlertEvent:
    """Alert event passed to onAlert callback."""

    level: Literal["info", "warning", "critical"]
    """Alert severity level."""

    message: str
    """Alert message."""

    reason: str
    """Reason the alert was triggered."""

    provider: str
    """Provider that triggered the alert."""

    model: str
    """Model that triggered the alert."""

    timestamp: datetime
    """Timestamp of the alert."""

    context_id: str | None = None
    """Context ID that triggered the alert."""


# Callback type for alerts
AlertCallback = Callable[[AlertEvent], None | Awaitable[None]]


@dataclass
class ControlAgentOptions:
    """Options for creating a control agent."""

    server_url: str
    """Server URL (wss:// for WebSocket, https:// for HTTP-only)."""

    api_key: str
    """API key for authentication."""

    polling_interval_ms: int = 30000
    """Polling interval for HTTP fallback (ms)."""

    heartbeat_interval_ms: int = 10000
    """Heartbeat interval (ms)."""

    timeout_ms: int = 5000
    """Request timeout (ms)."""

    fail_open: bool = True
    """Fail open (allow) if server is unreachable."""

    get_context_id: Callable[[], str | None] | None = None
    """Custom context ID extractor."""

    instance_id: str | None = None
    """SDK instance identifier (auto-generated if not provided)."""

    on_alert: AlertCallback | None = None
    """Callback invoked when an alert is triggered."""


# =============================================================================
# Control Agent Interface
# =============================================================================

class IControlAgent(Protocol):
    """Control Agent interface - the public API."""

    async def connect(self) -> None:
        """Connect to the control server."""
        ...

    async def disconnect(self) -> None:
        """Disconnect from the control server."""
        ...

    async def get_decision(self, request: ControlRequest) -> ControlDecision:
        """Get a control decision for a request."""
        ...

    async def report_metric(self, event: MetricEvent) -> None:
        """Report a metric event to the server."""
        ...

    async def report_control_event(self, event: ControlEvent) -> None:
        """Report a control event to the server."""
        ...

    def is_connected(self) -> bool:
        """Check if connected to server."""
        ...

    def get_policy(self) -> ControlPolicy | None:
        """Get current cached policy."""
        ...
