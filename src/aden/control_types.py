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

    degrade_to_provider: str | None = None
    """If action is 'degrade', the provider of the degraded model."""

    throttle_delay_ms: int | None = None
    """If action is 'throttle', delay by this many milliseconds."""

    alert_level: Literal["info", "warning", "critical"] | None = None
    """If action is 'alert', the severity level."""

    budget_id: str | None = None
    """ID of the budget that triggered this decision."""


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

    policy_id: str = "default"
    """Policy ID for routing notifications on the server."""

    budget_id: str | None = None
    """ID of the budget that triggered this action."""


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
    """Budget rule - limits spend per context.

    The server returns budgets with type-based matching:
    - global: Applies to all requests
    - agent: Matches request.metadata.agent
    - tenant: Matches request.metadata.tenant_id
    - customer: Matches request.metadata.customer_id
    - feature: Matches request.metadata.feature
    - tag: Matches if any request.metadata.tags intersect with budget tags
    """

    id: str
    """Unique budget ID from server."""

    budget_type: str
    """Budget type: global, agent, tenant, customer, feature, tag."""

    limit_usd: float
    """Budget limit in USD."""

    current_spend_usd: float = 0.0
    """Current spend in USD (server tracks this)."""

    action_on_exceed: ControlAction = ControlAction.BLOCK
    """Action to take when budget is exceeded."""

    name: str | None = None
    """Budget name (used for matching agent/tenant/customer/feature)."""

    degrade_to_model: str | None = None
    """If action is 'degrade', switch to this model."""

    degrade_to_provider: str | None = None
    """If action is 'degrade', the provider of the degraded model."""

    tags: list[str] | None = None
    """Tags for tag-type budgets."""

    # Legacy field for backwards compatibility
    context_id: str | None = None
    """Deprecated: Use budget_type and name for matching."""


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

    # Hybrid enforcement options
    server_validation_threshold: float = 5.0
    """Budget usage percentage at which to START considering server validation (0-100).
    This is the base threshold - actual validation probability scales from here.
    Set to 100 to disable server validation entirely. Default: 80%."""

    server_validation_timeout_ms: int = 2000
    """Timeout for server validation requests (ms). If validation times out,
    behavior depends on fail_open setting. Default: 2000ms."""

    enable_hybrid_enforcement: bool = True
    """Enable hybrid enforcement mode. When enabled, requests approaching
    budget limits will be validated with the server. Default: True."""

    # Adaptive enforcement options (reduces latency impact at high budgets)
    adaptive_threshold_enabled: bool = True
    """Enable adaptive threshold calculation. When enabled, the validation
    threshold adapts based on remaining budget and request volume.
    This significantly reduces latency impact for high-budget scenarios.
    Default: True."""

    adaptive_min_remaining_usd: float = 5.0
    """Minimum remaining budget (in USD) before forcing 100% validation rate.
    When remaining budget drops below this, all requests are validated.
    This provides a safety net regardless of percentage. Default: $5."""

    sampling_enabled: bool = True
    """Enable probabilistic sampling for server validation.
    Instead of validating every request above threshold, only validate
    a percentage of requests based on budget usage. This dramatically
    reduces latency impact while maintaining statistical enforcement.
    Default: True."""

    sampling_base_rate: float = 0.1
    """Base sampling rate at the validation threshold (0.0-1.0).
    At 80% usage (default threshold), only 10% of requests are validated.
    Rate increases as usage approaches 100%. Default: 0.1 (10%)."""

    sampling_full_validation_percent: float = 95.0
    """Usage percentage at which to validate 100% of requests.
    Between threshold and this value, sampling rate interpolates.
    Default: 95%."""

    max_expected_overspend_percent: float = 5.0
    """Maximum expected overspend as percentage of budget limit.
    Used to calculate soft/hard limit boundaries. Requests are blocked
    at (100 + this value)% to provide a hard stop. Default: 5%."""


@dataclass
class BudgetValidationRequest:
    """Request payload for server-side budget validation."""

    budget_id: str
    """The budget ID to validate."""

    estimated_cost: float
    """Estimated cost of the pending request in USD."""

    context_type: str | None = None
    """Budget type: global, agent, tenant, customer, feature, tag."""

    context_value: str | None = None
    """Context value (agent name, tenant_id, etc.)."""

    tags: list[str] | None = None
    """Tags for tag-type budgets."""


@dataclass
class BudgetValidationResponse:
    """Response from server-side budget validation."""

    allowed: bool
    """Whether the request should proceed."""

    action: str
    """Action to take: allow, block, degrade, throttle."""

    authoritative_spend: float
    """Server's authoritative spend value."""

    budget_limit: float
    """Budget limit for reference."""

    usage_percent: float
    """Current usage percentage."""

    policy_version: str
    """Current policy version for cache sync."""

    updated_spend: float
    """Updated spend to cache locally."""

    reason: str | None = None
    """Reason for the decision."""

    projected_percent: float | None = None
    """Projected usage percentage after this request."""

    degrade_to_model: str | None = None
    """Model to degrade to (if action is degrade)."""

    degrade_to_provider: str | None = None
    """Provider of the degraded model (if action is degrade)."""


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

    async def store_large_content(self, content_payloads: list[dict[str, Any]]) -> None:
        """Store large content items on the control server.

        Used by Layer 0 content capture for storing content that exceeds
        max_content_bytes threshold. Content is referenced via ContentReference.

        Args:
            content_payloads: List of content items to store, each containing:
                - content_id: Unique ID for the content
                - content_hash: SHA-256 hash of the content
                - content: The actual content string or serialized data
                - byte_size: Size of the content in bytes
        """
        ...
