"""
Control Agent - Bidirectional communication with control server.

Emits: metrics, control events, heartbeat
Receives: control policies (budgets, throttle, block, degrade)

Uses Socket.IO for real-time communication with HTTP polling fallback.
Supports hot reload of policy updates including budget policies.
"""

import asyncio
import json
import logging
import random
import re
import threading
import time
from dataclasses import asdict
from datetime import datetime
from typing import Any, Awaitable, Callable
from uuid import uuid4

from .control_types import (
    AlertEvent,
    AlertRule,
    BlockRule,
    BudgetRule,
    BudgetValidationRequest,
    BudgetValidationResponse,
    ControlAction,
    ControlAgentOptions,
    ControlDecision,
    ControlEvent,
    ControlPolicy,
    ControlRequest,
    ErrorEvent,
    HeartbeatEvent,
    IControlAgent,
    MetricEventWrapper,
    ServerEvent,
)
from .types import MetricEvent

logger = logging.getLogger("aden")

# Package version
SDK_VERSION = "0.2.0"


class ControlAgent(IControlAgent):
    """Control Agent implementation."""

    def __init__(self, options: ControlAgentOptions):
        self.options = ControlAgentOptions(
            server_url=options.server_url.rstrip("/"),
            api_key=options.api_key,
            polling_interval_ms=options.polling_interval_ms,
            heartbeat_interval_ms=options.heartbeat_interval_ms,
            timeout_ms=options.timeout_ms,
            fail_open=options.fail_open,
            get_context_id=options.get_context_id or (lambda: None),
            instance_id=options.instance_id or str(uuid4()),
            on_alert=options.on_alert or (lambda _: None),
            # Hybrid enforcement options
            server_validation_threshold=options.server_validation_threshold,
            server_validation_timeout_ms=options.server_validation_timeout_ms,
            enable_hybrid_enforcement=options.enable_hybrid_enforcement,
            adaptive_threshold_enabled=options.adaptive_threshold_enabled,
            adaptive_min_remaining_usd=options.adaptive_min_remaining_usd,
            sampling_enabled=options.sampling_enabled,
            sampling_base_rate=options.sampling_base_rate,
            sampling_full_validation_percent=options.sampling_full_validation_percent,
            max_expected_overspend_percent=options.max_expected_overspend_percent,
        )

        # Socket.IO / WebSocket state
        self._ws: Any = None  # Legacy websocket (deprecated)
        self._sio: Any = None  # Socket.IO client
        self._connected = False
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10

        # Policy cache
        self._cached_policy: ControlPolicy | None = None
        self._last_policy_fetch = 0

        # Timers/tasks
        self._polling_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._reconnect_task: asyncio.Task | None = None

        # Event queue for offline buffering
        self._event_queue: list[ServerEvent] = []
        self._max_queue_size = 1000
        self._sync_batch_size = 10  # Flush sync events after this many are queued
        self._sync_flush_interval = 1.0  # Flush sync events every second

        # Background thread for sync flushing
        self._sync_flush_thread: threading.Thread | None = None
        self._sync_flush_stop = threading.Event()

        # Stats
        self._requests_since_heartbeat = 0
        self._errors_since_heartbeat = 0

        # Rate limiting tracking
        self._request_counts: dict[str, dict[str, Any]] = {}

        # Pricing cache (loaded from server on connect)
        self._pricing_cache: dict[str, dict[str, float]] = {}
        self._pricing_aliases: dict[str, str] = {}

    async def connect(self) -> None:
        """Connect to the control server.

        This method is designed to fail-through gracefully. If the server
        is unreachable, the SDK will still operate with fail_open behavior
        (allowing requests if fail_open=True, the default).
        """
        url = self.options.server_url
        logger.info(f"[aden] Connecting to control server: {url}")

        try:
            # Determine transport based on URL scheme
            if url.startswith("wss://") or url.startswith("ws://"):
                await self._connect_websocket()
            else:
                # HTTP-only mode: just use polling
                logger.debug("[aden] Using HTTP polling mode")
                await self._start_polling()
        except Exception as e:
            logger.warning(f"[aden] Failed to connect to control server: {e}")
            if not self.options.fail_open:
                raise
            # Continue with fail-open mode - no policy cached, requests allowed

        try:
            # Fetch pricing table for accurate cost estimation
            await self._fetch_pricing()
        except Exception as e:
            logger.warning(f"[aden] Failed to fetch pricing table: {e}")
            # Continue with fallback pricing

        # Start heartbeat (never fails, just creates a background task)
        self._start_heartbeat()

        if self._cached_policy:
            logger.info("[aden] Control agent started")
        else:
            logger.warning(
                "[aden] Control agent started without policy cache "
                f"(fail_open={'enabled' if self.options.fail_open else 'disabled'})"
            )

    async def _connect_websocket(self) -> None:
        """Connect via Socket.IO."""
        try:
            # Try to import socketio library
            import socketio
        except ImportError:
            logger.warning("[aden] python-socketio library not installed, using HTTP polling only")
            await self._start_polling()
            return

        try:
            # Build Socket.IO URL (remove wss:// prefix and add http(s)://)
            base_url = self.options.server_url
            if base_url.startswith("wss://"):
                base_url = "https://" + base_url[6:]
            elif base_url.startswith("ws://"):
                base_url = "http://" + base_url[5:]

            # Create Socket.IO client
            sio = socketio.AsyncClient(
                reconnection=True,
                reconnection_attempts=self._max_reconnect_attempts,
                reconnection_delay=1,
                reconnection_delay_max=30,
                logger=False,
                engineio_logger=False,
            )

            # Store the client
            self._sio = sio

            # Set up event handlers
            @sio.on("connect", namespace="/v1/control/ws")
            async def on_connect():
                self._connected = True
                self._reconnect_attempts = 0
                logger.info("[aden] Socket.IO connected to control server")
                # Flush queued events
                await self._flush_event_queue()

            @sio.on("disconnect", namespace="/v1/control/ws")
            async def on_disconnect():
                logger.warning("[aden] Socket.IO disconnected")
                self._connected = False

            @sio.on("message", namespace="/v1/control/ws")
            async def on_message(data):
                await self._handle_socketio_message(data)

            @sio.on("connect_error", namespace="/v1/control/ws")
            async def on_connect_error(error):
                logger.warning(f"[aden] Socket.IO connection error: {error}")
                self._connected = False

            # Connect with auth headers
            auth = {
                "token": self.options.api_key,
            }
            headers = {
                "Authorization": f"Bearer {self.options.api_key}",
                "X-SDK-Instance-ID": self.options.instance_id,
            }

            try:
                await asyncio.wait_for(
                    sio.connect(
                        base_url,
                        namespaces=["/v1/control/ws"],
                        auth=auth,
                        headers=headers,
                        transports=["websocket", "polling"],
                    ),
                    timeout=self.options.timeout_ms / 1000,
                )
            except asyncio.TimeoutError:
                logger.warning("[aden] Socket.IO connection timeout, using polling")
                await self._start_polling()
            except Exception as e:
                logger.warning(f"[aden] Socket.IO connection failed: {e}, using polling")
                await self._start_polling()

        except Exception as e:
            logger.warning(f"[aden] Socket.IO setup failed: {e}, using polling")
            await self._start_polling()

    async def _handle_socketio_message(self, data: dict[str, Any]) -> None:
        """Handle incoming Socket.IO messages."""
        try:
            msg_type = data.get("type")
            if msg_type == "policy":
                self._cached_policy = self._parse_policy(data.get("policy", {}))
                self._last_policy_fetch = time.time()
                logger.info(f"[aden] Policy updated: {self._cached_policy.version}")
            elif msg_type == "budget_policy":
                # Handle budget-specific policy updates
                budget_data = data.get("budgets", [])
                if self._cached_policy and budget_data:
                    self._update_budgets_from_server(budget_data)
                    self._last_policy_fetch = time.time()
                    logger.info("[aden] Budget policy updated")
            elif msg_type == "command":
                logger.info(f"[aden] Command received: {data}")
            elif msg_type == "alert":
                logger.info(f"[aden] Alert received: {data.get('alert', {})}")
        except Exception as e:
            logger.warning(f"[aden] Failed to handle Socket.IO message: {e}")

    def _update_budgets_from_server(self, budget_data: list[dict[str, Any]]) -> None:
        """Update budgets in the cached policy from server data."""
        if not self._cached_policy:
            return

        # Parse and update budgets
        updated_budgets = []
        for b in budget_data:
            updated_budgets.append(self._parse_budget(b))

        self._cached_policy.budgets = updated_budgets
        logger.debug(f"[aden] Updated {len(updated_budgets)} budgets from server")

    def _parse_budget(self, b: dict[str, Any]) -> BudgetRule:
        """Parse a budget from server or legacy format."""
        from .control_types import BudgetRule

        # Map server limitAction to ControlAction
        limit_action = b.get("limitAction") or b.get("action_on_exceed", "block")
        action_map = {
            "kill": ControlAction.BLOCK,
            "block": ControlAction.BLOCK,
            "degrade": ControlAction.DEGRADE,
            "throttle": ControlAction.THROTTLE,
            "alert": ControlAction.ALERT,
        }
        action = action_map.get(limit_action, ControlAction.BLOCK)

        return BudgetRule(
            # Server format uses 'id', legacy uses 'context_id' as ID
            id=b.get("id") or b.get("context_id", ""),
            # Server format uses 'type', default to 'global' if not specified
            budget_type=b.get("type", "global"),
            # Server uses 'limit', legacy uses 'limit_usd'
            limit_usd=b.get("limit") or b.get("limit_usd", 0),
            # Server uses 'spent', legacy uses 'current_spend_usd'
            current_spend_usd=b.get("spent") or b.get("current_spend_usd", 0),
            action_on_exceed=action,
            # Server uses 'name' for matching, also used as context identifier
            name=b.get("name"),
            # Server uses 'degradeToModel', legacy uses 'degrade_to_model'
            degrade_to_model=b.get("degradeToModel") or b.get("degrade_to_model"),
            # Server uses 'degradeToProvider', legacy uses 'degrade_to_provider'
            degrade_to_provider=b.get("degradeToProvider") or b.get("degrade_to_provider"),
            # Tags for tag-type budgets
            tags=b.get("tags"),
            # Legacy context_id for backwards compatibility
            context_id=b.get("context_id"),
        )

    def _parse_policy(self, data: dict[str, Any]) -> ControlPolicy:
        """Parse policy JSON into ControlPolicy object.

        Handles both server format (id, type, limit, spent, limitAction) and
        legacy format (context_id, limit_usd, current_spend_usd, action_on_exceed).
        """
        from .control_types import (
            AlertRule,
            BlockRule,
            DegradeRule,
            ThrottleRule,
        )

        return ControlPolicy(
            version=data.get("version", "unknown"),
            updated_at=data.get("updated_at", ""),
            budgets=[self._parse_budget(b) for b in data.get("budgets", [])],
            throttles=[
                ThrottleRule(
                    context_id=t.get("context_id"),
                    provider=t.get("provider"),
                    requests_per_minute=t.get("requests_per_minute"),
                    delay_ms=t.get("delay_ms"),
                )
                for t in data.get("throttles", [])
            ],
            blocks=[
                BlockRule(
                    reason=b.get("reason", ""),
                    context_id=b.get("context_id"),
                    provider=b.get("provider"),
                    model_pattern=b.get("model_pattern"),
                )
                for b in data.get("blocks", [])
            ],
            degradations=[
                DegradeRule(
                    from_model=d.get("from_model", ""),
                    to_model=d.get("to_model", ""),
                    trigger=d.get("trigger", "always"),
                    threshold_percent=d.get("threshold_percent"),
                    context_id=d.get("context_id"),
                )
                for d in data.get("degradations", [])
            ],
            alerts=[
                AlertRule(
                    trigger=a.get("trigger", "always"),
                    level=a.get("level", "info"),
                    message=a.get("message", ""),
                    context_id=a.get("context_id"),
                    provider=a.get("provider"),
                    model_pattern=a.get("model_pattern"),
                    threshold_percent=a.get("threshold_percent"),
                )
                for a in data.get("alerts", [])
            ],
        )

    def _schedule_reconnect(self) -> None:
        """Schedule WebSocket reconnection with exponential backoff."""
        if self._reconnect_task is not None:
            return
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            logger.warning("[aden] Max reconnect attempts reached, using polling only")
            return

        # Exponential backoff: 1s, 2s, 4s, 8s, ... up to 30s
        delay = min(1.0 * (2 ** self._reconnect_attempts), 30.0)
        self._reconnect_attempts += 1

        async def reconnect():
            await asyncio.sleep(delay)
            self._reconnect_task = None
            if not self._connected:
                await self._connect_websocket()

        self._reconnect_task = asyncio.create_task(reconnect())

    async def _start_polling(self) -> None:
        """Start HTTP polling for policy updates."""
        if self._polling_task is not None:
            return

        # Fetch immediately
        await self._fetch_policy()

        async def poll_loop():
            while True:
                await asyncio.sleep(self.options.polling_interval_ms / 1000)
                if not self._connected:
                    await self._fetch_policy()

        self._polling_task = asyncio.create_task(poll_loop())

    def _stop_polling(self) -> None:
        """Stop HTTP polling."""
        if self._polling_task:
            self._polling_task.cancel()
            self._polling_task = None

    async def _fetch_policy(self) -> None:
        """Fetch policy via HTTP."""
        try:
            response = await self._http_request("/v1/control/policy", "GET")
            if response.get("ok"):
                self._cached_policy = self._parse_policy(response.get("data", {}))
                self._last_policy_fetch = time.time()
        except Exception as e:
            logger.warning(f"[aden] Failed to fetch policy: {e}")

    def _start_heartbeat(self) -> None:
        """Start heartbeat timer."""
        if self._heartbeat_task is not None:
            return

        async def heartbeat_loop():
            while True:
                await asyncio.sleep(self.options.heartbeat_interval_ms / 1000)
                await self._send_heartbeat()

        self._heartbeat_task = asyncio.create_task(heartbeat_loop())

    def _stop_heartbeat(self) -> None:
        """Stop heartbeat timer."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None

    async def _send_heartbeat(self) -> None:
        """Send heartbeat event."""
        event = HeartbeatEvent(
            event_type="heartbeat",
            timestamp=datetime.now().isoformat(),
            sdk_instance_id=self.options.instance_id or "",
            status="healthy" if self._connected else "degraded",
            requests_since_last=self._requests_since_heartbeat,
            errors_since_last=self._errors_since_heartbeat,
            policy_cache_age_seconds=int(time.time() - self._last_policy_fetch)
            if self._last_policy_fetch > 0
            else -1,
            websocket_connected=self._connected,
            sdk_version=SDK_VERSION,
        )

        await self._send_event(event)

        # Reset counters
        self._requests_since_heartbeat = 0
        self._errors_since_heartbeat = 0

    def disconnect_sync(self) -> None:
        """Disconnect from the control server (sync version)."""
        logger.debug("[aden] Disconnecting control agent (sync)")
        # Stop the background flush thread (which also does final flush)
        self._stop_sync_flush_thread()

        self._stop_polling()
        self._stop_heartbeat()

        if self._reconnect_task:
            self._reconnect_task.cancel()
            self._reconnect_task = None

        # Socket.IO disconnect needs to be done in async context
        # We mark as disconnected here; async cleanup happens in disconnect()
        self._connected = False
        logger.info("[aden] Control agent disconnected")

    async def disconnect(self) -> None:
        """Disconnect from the control server."""
        logger.debug("[aden] Disconnecting control agent (async)")
        # Flush any remaining events
        await self._flush_event_queue()

        self._stop_polling()
        self._stop_heartbeat()

        if self._reconnect_task:
            self._reconnect_task.cancel()
            self._reconnect_task = None

        # Disconnect Socket.IO client
        if self._sio:
            try:
                await self._sio.disconnect()
            except Exception as e:
                logger.debug(f"[aden] Socket.IO disconnect error (ignored): {e}")
            self._sio = None

        # Legacy websocket cleanup (if any)
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        self._connected = False
        logger.info("[aden] Control agent disconnected")

    def get_decision_sync(self, request: ControlRequest) -> ControlDecision:
        """Get a control decision for a request (sync version)."""
        self._requests_since_heartbeat += 1

        # If no policy, use default based on fail_open
        if not self._cached_policy:
            return (
                ControlDecision(action=ControlAction.ALLOW)
                if self.options.fail_open
                else ControlDecision(
                    action=ControlAction.BLOCK,
                    reason="No policy available and fail_open is False",
                )
            )

        return self._evaluate_policy(request, self._cached_policy)

    async def get_decision(self, request: ControlRequest) -> ControlDecision:
        """Get a control decision for a request (async version)."""
        return self.get_decision_sync(request)

    def _evaluate_policy(
        self, request: ControlRequest, policy: ControlPolicy
    ) -> ControlDecision:
        """
        Evaluate policy rules against a request.
        Priority order: block > budget/degrade > throttle > alert > allow
        """
        # Track throttle info separately
        throttle_info: dict[str, Any] | None = None

        # 1. Check block rules first (highest priority)
        for block in policy.blocks:
            if self._matches_block_rule(request, block):
                return ControlDecision(action=ControlAction.BLOCK, reason=block.reason)

        # 2. Check throttle rules
        for throttle in policy.throttles:
            if not throttle.context_id or throttle.context_id == request.context_id:
                if not throttle.provider or throttle.provider == request.provider:
                    if throttle.requests_per_minute:
                        key = f"{throttle.context_id or 'global'}:{throttle.provider or 'all'}"
                        rate_info = self._check_rate_limit(
                            key, throttle.requests_per_minute
                        )
                        if rate_info["exceeded"]:
                            throttle_info = {
                                "delay_ms": throttle.delay_ms or 1000,
                                "reason": f"Rate limit: {rate_info['count']}/{throttle.requests_per_minute}/min",
                            }
                            break

                    if throttle.delay_ms and not throttle.requests_per_minute:
                        throttle_info = {
                            "delay_ms": throttle.delay_ms,
                            "reason": "Fixed throttle delay",
                        }
                        break

        # 3. Check budget limits (with hybrid enforcement + adaptive sampling)
        # Collect ALL matching budgets and evaluate each one
        # Return the MOST RESTRICTIVE decision across all matching budgets
        matching_budgets = [
            budget for budget in policy.budgets
            if self._matches_budget(request, budget)
        ]

        if matching_budgets:
            logger.debug(
                f"[aden] Found {len(matching_budgets)} matching budget(s): "
                f"{[b.id for b in matching_budgets]}"
            )
            # Action priority (higher = more restrictive)
            action_priority = {
                ControlAction.ALLOW: 0,
                ControlAction.ALERT: 1,
                ControlAction.THROTTLE: 2,
                ControlAction.DEGRADE: 3,
                ControlAction.BLOCK: 4,
            }

            most_restrictive_decision: ControlDecision | None = None
            most_restrictive_priority = -1

            for budget in matching_budgets:
                decision = self._evaluate_single_budget(
                    request, budget, policy, throttle_info
                )

                if decision:
                    priority = action_priority.get(decision.action, 0)
                    if priority > most_restrictive_priority:
                        most_restrictive_decision = decision
                        most_restrictive_priority = priority

                    # Short-circuit: BLOCK is highest priority, no need to continue
                    if decision.action == ControlAction.BLOCK:
                        break

            if most_restrictive_decision:
                logger.debug(
                    f"[aden] Most restrictive decision across {len(matching_budgets)} budget(s): "
                    f"{most_restrictive_decision.action.value} - {most_restrictive_decision.reason}"
                )
                # Add throttle info if present and not already blocking
                if throttle_info and most_restrictive_decision.action != ControlAction.BLOCK:
                    most_restrictive_decision.throttle_delay_ms = throttle_info["delay_ms"]
                return most_restrictive_decision

        # 4. Check always-degrade rules
        for degrade in policy.degradations:
            if degrade.from_model == request.model and degrade.trigger == "always":
                if not degrade.context_id or degrade.context_id == request.context_id:
                    return ControlDecision(
                        action=ControlAction.DEGRADE,
                        reason="Model degradation rule (always)",
                        degrade_to_model=degrade.to_model,
                        throttle_delay_ms=throttle_info["delay_ms"]
                        if throttle_info
                        else None,
                    )

        # 5. Check alert rules
        for alert in policy.alerts:
            if self._matches_alert_rule(request, alert, policy):
                # Trigger the on_alert callback asynchronously
                alert_event = AlertEvent(
                    level=alert.level,
                    message=alert.message,
                    reason=f"Triggered by {alert.trigger}",
                    context_id=request.context_id,
                    provider=request.provider,
                    model=request.model,
                    timestamp=datetime.now(),
                )

                # Fire and forget
                if self.options.on_alert:
                    try:
                        result = self.options.on_alert(alert_event)
                        if asyncio.iscoroutine(result):
                            asyncio.create_task(result)
                    except Exception as e:
                        logger.warning(f"[aden] Alert callback error: {e}")

                return ControlDecision(
                    action=ControlAction.ALERT,
                    reason=alert.message,
                    alert_level=alert.level,
                    throttle_delay_ms=throttle_info["delay_ms"]
                    if throttle_info
                    else None,
                )

        # 6. If throttle is active but no other action, return throttle
        if throttle_info:
            return ControlDecision(
                action=ControlAction.THROTTLE,
                reason=throttle_info["reason"],
                throttle_delay_ms=throttle_info["delay_ms"],
            )

        return ControlDecision(action=ControlAction.ALLOW)

    def _matches_block_rule(self, request: ControlRequest, block: BlockRule) -> bool:
        """Check if request matches a block rule."""
        if block.context_id and block.context_id != request.context_id:
            return False
        if block.provider and block.provider != request.provider:
            return False
        if block.model_pattern:
            pattern = "^" + block.model_pattern.replace("*", ".*") + "$"
            if not re.match(pattern, request.model):
                return False
        return True

    def _matches_alert_rule(
        self, request: ControlRequest, alert: AlertRule, policy: ControlPolicy
    ) -> bool:
        """Check if request matches an alert rule."""
        if alert.context_id and alert.context_id != request.context_id:
            return False
        if alert.provider and alert.provider != request.provider:
            return False
        if alert.model_pattern:
            pattern = "^" + alert.model_pattern.replace("*", ".*") + "$"
            if not re.match(pattern, request.model):
                return False

        if alert.trigger == "always":
            return True
        elif alert.trigger == "model_usage":
            return True  # Model pattern already matched above
        elif alert.trigger == "budget_threshold":
            if alert.threshold_percent and request.context_id:
                for budget in policy.budgets:
                    if budget.context_id == request.context_id:
                        usage_percent = (
                            budget.current_spend_usd / budget.limit_usd
                        ) * 100
                        return usage_percent >= alert.threshold_percent
            return False

        return False

    def _check_rate_limit(
        self, key: str, limit: int
    ) -> dict[str, Any]:
        """Check rate limit for a key."""
        now = time.time()
        window_seconds = 60  # 1 minute window

        info = self._request_counts.get(key)
        if not info or now - info["window_start"] > window_seconds:
            info = {"count": 0, "window_start": now}

        info["count"] += 1
        self._request_counts[key] = info

        return {"exceeded": info["count"] > limit, "count": info["count"]}

    def _matches_budget(
        self, request: ControlRequest, budget: Any  # BudgetRule
    ) -> bool:
        """
        Check if a budget applies to the given request.

        Matching logic based on budget type:
        - global: Matches ALL requests
        - agent: Matches if request.metadata.agent == budget.name or budget.id
        - tenant: Matches if request.metadata.tenant_id == budget.name or budget.id
        - customer: Matches if request.metadata.customer_id == budget.name or budget.id
        - feature: Matches if request.metadata.feature == budget.name or budget.id
        - tag: Matches if any request.metadata.tags intersect with budget.tags
        - legacy (context_id): Matches if request.context_id == budget.context_id
        """
        budget_type = getattr(budget, "budget_type", "global")
        budget_name = getattr(budget, "name", None)
        budget_id = getattr(budget, "id", None)
        budget_context_id = getattr(budget, "context_id", None)
        budget_tags = getattr(budget, "tags", None) or []

        # Get metadata from request
        metadata = request.metadata or {}

        # Global budgets match all requests
        if budget_type == "global":
            return True

        # Legacy context_id matching (for backwards compatibility)
        if budget_context_id and request.context_id:
            if budget_context_id == request.context_id:
                return True

        # Type-based matching
        if budget_type == "agent":
            agent = metadata.get("agent")
            return agent and (agent == budget_name or agent == budget_id)

        elif budget_type == "tenant":
            tenant_id = metadata.get("tenant_id")
            return tenant_id and (tenant_id == budget_name or tenant_id == budget_id)

        elif budget_type == "customer":
            customer_id = metadata.get("customer_id")
            return customer_id and (customer_id == budget_name or customer_id == budget_id)

        elif budget_type == "feature":
            feature = metadata.get("feature")
            return feature and (feature == budget_name or feature == budget_id)

        elif budget_type == "tag":
            request_tags = metadata.get("tags", [])
            if not request_tags or not budget_tags:
                return False
            # Check for any intersection
            return bool(set(request_tags) & set(budget_tags))

        # Unknown budget type - don't match
        return False

    def _evaluate_single_budget(
        self,
        request: ControlRequest,
        budget: BudgetRule,
        policy: ControlPolicy,
        throttle_info: dict[str, Any] | None,
    ) -> ControlDecision | None:
        """
        Evaluate a single budget and return a decision.

        Returns None if the budget doesn't trigger any action (i.e., ALLOW).
        This allows the caller to find the most restrictive decision across
        all matching budgets.
        """
        estimated_cost = request.estimated_cost or 0
        current_spend = budget.current_spend_usd
        limit = budget.limit_usd

        # Calculate local usage and projected percentages
        usage_percent = (current_spend / limit * 100) if limit > 0 else 0
        projected_spend = current_spend + estimated_cost
        projected_percent = (projected_spend / limit * 100) if limit > 0 else 0
        remaining = max(0, limit - current_spend)

        # HARD LIMIT CHECK: Block if exceeding max allowed overspend
        # Only applies when budget action is BLOCK - degrade/throttle budgets don't need hard limits
        if budget.action_on_exceed == ControlAction.BLOCK:
            hard_limit_decision = self._check_hard_limit(usage_percent, projected_percent)
            if hard_limit_decision:
                hard_limit_decision.budget_id = budget.id
                return hard_limit_decision

        # HYBRID ENFORCEMENT: Check if we should validate with server
        # Uses adaptive thresholds + probabilistic sampling
        if self._should_validate_with_server(
            budget_usage_percent=usage_percent,
            remaining_budget_usd=remaining,
            budget_limit_usd=limit,
        ):
            logger.debug(
                f"[aden] Budget '{budget.id}' at {usage_percent:.1f}% "
                f"(${current_spend:.2f}/${limit:.2f}), validating with server"
            )

            # Sync validation (blocking but fast with 2s timeout)
            # Send local spend so server can use max(local, server) for accuracy
            validation = self._validate_budget_with_server_sync(
                budget_id=budget.id,
                estimated_cost=estimated_cost,
                local_spend=current_spend,
            )

            if validation:
                # Server validation succeeded - use authoritative decision
                decision = self._apply_server_validation_result(validation, budget.id)
                return decision
            else:
                # Server validation failed - fall back to local enforcement
                logger.warning(
                    f"[aden] Server validation failed for budget '{budget.id}', using local enforcement"
                )
                if not self.options.fail_open:
                    return ControlDecision(
                        action=ControlAction.BLOCK,
                        reason=f"Server validation failed for budget '{budget.id}' and fail_open is False",
                        budget_id=budget.id,
                    )
                # Continue with local enforcement below

        # LOCAL ENFORCEMENT: Check if budget would be exceeded (soft limit)
        if projected_percent >= 100:
            if (
                budget.action_on_exceed == ControlAction.DEGRADE
                and budget.degrade_to_model
            ):
                return ControlDecision(
                    action=ControlAction.DEGRADE,
                    reason=f"Budget '{budget.id}' exceeded: ${projected_spend:.4f} > ${limit} ({projected_percent:.1f}%)",
                    degrade_to_model=budget.degrade_to_model,
                    degrade_to_provider=budget.degrade_to_provider,
                    budget_id=budget.id,
                )
            return ControlDecision(
                action=budget.action_on_exceed,
                reason=f"Budget '{budget.id}' exceeded: ${projected_spend:.4f} > ${limit} ({projected_percent:.1f}%)",
                budget_id=budget.id,
            )

        # Check degradation rules based on budget threshold
        for degrade in policy.degradations:
            if (
                degrade.from_model == request.model
                and degrade.trigger == "budget_threshold"
                and degrade.threshold_percent
            ):
                if usage_percent >= degrade.threshold_percent:
                    return ControlDecision(
                        action=ControlAction.DEGRADE,
                        reason=f"Budget '{budget.id}' at {usage_percent:.1f}% (threshold: {degrade.threshold_percent}%)",
                        degrade_to_model=degrade.to_model,
                        budget_id=budget.id,
                    )

        # No restrictive action needed for this budget
        return None

    # =========================================================================
    # Hybrid Enforcement - Server-Side Budget Validation
    # =========================================================================

    def _calculate_sampling_rate(self, usage_percent: float) -> float:
        """
        Calculate the sampling rate for server validation based on usage percentage.

        The rate interpolates from sampling_base_rate at threshold to 1.0 at
        sampling_full_validation_percent.

        Example with defaults (threshold=80%, base_rate=0.1, full=95%):
        - At 80%: 10% of requests validated
        - At 87.5%: 55% of requests validated
        - At 95%+: 100% of requests validated
        """
        threshold = self.options.server_validation_threshold
        full_percent = self.options.sampling_full_validation_percent
        base_rate = self.options.sampling_base_rate

        if usage_percent >= full_percent:
            return 1.0

        if usage_percent < threshold:
            return 0.0

        # Linear interpolation from base_rate to 1.0
        range_size = full_percent - threshold
        progress = (usage_percent - threshold) / range_size
        return base_rate + (1.0 - base_rate) * progress

    def _should_validate_with_server(
        self,
        budget_usage_percent: float,
        remaining_budget_usd: float,
        budget_limit_usd: float,
    ) -> bool:
        """
        Determine if we should validate this request with the server.

        Uses adaptive thresholds and probabilistic sampling to minimize
        latency impact while maintaining enforcement accuracy.

        Returns True if:
        1. Hybrid enforcement is enabled
        2. Budget usage is at or above the validation threshold
        3. Either:
           a. Remaining budget is below adaptive_min_remaining_usd (always validate)
           b. Sampling dice roll succeeds based on current usage level
        """
        if not self.options.enable_hybrid_enforcement:
            return False

        # Below threshold - no validation needed
        if budget_usage_percent < self.options.server_validation_threshold:
            return False

        # ADAPTIVE: Force validation if remaining budget is critically low
        if self.options.adaptive_threshold_enabled:
            if remaining_budget_usd <= self.options.adaptive_min_remaining_usd:
                logger.debug(
                    f"[aden] Remaining budget ${remaining_budget_usd:.2f} <= "
                    f"${self.options.adaptive_min_remaining_usd:.2f}, forcing validation"
                )
                return True

        # SAMPLING: Probabilistic validation based on usage level
        if self.options.sampling_enabled:
            sampling_rate = self._calculate_sampling_rate(budget_usage_percent)
            should_sample = random.random() < sampling_rate

            if not should_sample:
                logger.debug(
                    f"[aden] Skipping validation (sampling rate: {sampling_rate:.1%}, "
                    f"usage: {budget_usage_percent:.1f}%)"
                )
                return False

            logger.debug(
                f"[aden] Sampled for validation (rate: {sampling_rate:.1%}, "
                f"usage: {budget_usage_percent:.1f}%)"
            )
            return True

        # No sampling - validate all requests above threshold
        return True

    def _check_hard_limit(
        self, usage_percent: float, projected_percent: float
    ) -> ControlDecision | None:
        """
        Check if request exceeds the hard limit (soft limit + max overspend buffer).

        This provides a safety net to prevent runaway spending even under
        concurrency race conditions.

        Returns a BLOCK decision if hard limit exceeded, None otherwise.
        """
        hard_limit = 100.0 + self.options.max_expected_overspend_percent

        if projected_percent >= hard_limit:
            return ControlDecision(
                action=ControlAction.BLOCK,
                reason=f"Hard limit exceeded: {projected_percent:.1f}% >= {hard_limit:.1f}%",
            )

        return None

    def _validate_budget_with_server_sync(
        self,
        budget_id: str,
        estimated_cost: float,
        local_spend: float | None = None,
        context_type: str | None = None,
        context_value: str | None = None,
        tags: list[str] | None = None,
    ) -> BudgetValidationResponse | None:
        """
        Validate budget with server synchronously.

        Returns BudgetValidationResponse if successful, None if validation failed.
        On failure, caller should fall back to local enforcement based on fail_open.

        Args:
            budget_id: The budget ID to validate
            estimated_cost: Estimated cost of the pending request
            local_spend: SDK's local spend tracking (server uses max of local vs TSDB)
            context_type: Budget type for matching
            context_value: Context value (agent name, tenant_id, etc.)
            tags: Tags for tag-type budgets
        """
        import urllib.request
        import urllib.error

        http_url = (
            self.options.server_url.replace("wss://", "https://").replace(
                "ws://", "http://"
            )
        )

        try:
            body = {
                "budget_id": budget_id,
                "estimated_cost": estimated_cost,
            }
            # Include local spend so server can use max(local, TSDB) for accuracy
            if local_spend is not None:
                body["local_spend"] = local_spend
            if context_type:
                body["context"] = {
                    "type": context_type,
                    "value": context_value,
                    "tags": tags,
                }

            data = json.dumps(body).encode("utf-8")
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.options.api_key}",
                "X-SDK-Instance-ID": self.options.instance_id or "",
            }

            req = urllib.request.Request(
                f"{http_url}/v1/control/budget/validate",
                data=data,
                headers=headers,
                method="POST",
            )

            timeout_sec = self.options.server_validation_timeout_ms / 1000
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                if resp.status == 200:
                    response_data = json.loads(resp.read().decode())
                    logger.debug(
                        f"[aden] Server validation response: allowed={response_data.get('allowed')}, "
                        f"action={response_data.get('action')}, reason={response_data.get('reason')}"
                    )
                    return BudgetValidationResponse(
                        allowed=response_data.get("allowed", True),
                        action=response_data.get("action", "allow"),
                        authoritative_spend=response_data.get("authoritative_spend", 0),
                        budget_limit=response_data.get("budget_limit", 0),
                        usage_percent=response_data.get("usage_percent", 0),
                        policy_version=response_data.get("policy_version", ""),
                        updated_spend=response_data.get("updated_spend", 0),
                        reason=response_data.get("reason"),
                        projected_percent=response_data.get("projected_percent"),
                        degrade_to_model=response_data.get("degrade_to_model"),
                        degrade_to_provider=response_data.get("degrade_to_provider"),
                    )
                else:
                    logger.warning(
                        f"[aden] Server validation returned status {resp.status}"
                    )
                    return None

        except urllib.error.HTTPError as e:
            logger.warning(f"[aden] Server validation HTTP error: {e.code}")
            return None
        except Exception as e:
            logger.warning(f"[aden] Server validation failed: {e}")
            return None

    async def _validate_budget_with_server(
        self,
        budget_id: str,
        estimated_cost: float,
        context_type: str | None = None,
        context_value: str | None = None,
        tags: list[str] | None = None,
    ) -> BudgetValidationResponse | None:
        """
        Validate budget with server asynchronously.

        Returns BudgetValidationResponse if successful, None if validation failed.
        """
        try:
            import aiohttp
        except ImportError:
            # Fall back to sync version
            return self._validate_budget_with_server_sync(
                budget_id, estimated_cost, context_type, context_value, tags
            )

        http_url = (
            self.options.server_url.replace("wss://", "https://").replace(
                "ws://", "http://"
            )
        )

        try:
            body = {
                "budget_id": budget_id,
                "estimated_cost": estimated_cost,
            }
            if context_type:
                body["context"] = {
                    "type": context_type,
                    "value": context_value,
                    "tags": tags,
                }

            timeout = aiohttp.ClientTimeout(
                total=self.options.server_validation_timeout_ms / 1000
            )
            async with aiohttp.ClientSession(timeout=timeout) as session:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.options.api_key}",
                    "X-SDK-Instance-ID": self.options.instance_id or "",
                }

                async with session.post(
                    f"{http_url}/v1/control/budget/validate",
                    headers=headers,
                    json=body,
                ) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        return BudgetValidationResponse(
                            allowed=response_data.get("allowed", True),
                            action=response_data.get("action", "allow"),
                            authoritative_spend=response_data.get("authoritative_spend", 0),
                            budget_limit=response_data.get("budget_limit", 0),
                            usage_percent=response_data.get("usage_percent", 0),
                            policy_version=response_data.get("policy_version", ""),
                            updated_spend=response_data.get("updated_spend", 0),
                            reason=response_data.get("reason"),
                            projected_percent=response_data.get("projected_percent"),
                            degrade_to_model=response_data.get("degrade_to_model"),
                            degrade_to_provider=response_data.get("degrade_to_provider"),
                        )
                    else:
                        logger.warning(
                            f"[aden] Server validation returned status {response.status}"
                        )
                        return None

        except Exception as e:
            logger.warning(f"[aden] Server validation failed: {e}")
            return None

    def _apply_server_validation_result(
        self,
        validation: BudgetValidationResponse,
        budget_id: str,
    ) -> ControlDecision:
        """Convert server validation response to a ControlDecision."""
        # Update local budget cache with authoritative spend
        if self._cached_policy and self._cached_policy.budgets:
            for budget in self._cached_policy.budgets:
                # Match by budget ID (primary) or legacy context_id
                if budget.id == budget_id or budget.context_id == budget_id:
                    budget.current_spend_usd = validation.updated_spend
                    break

        action_map = {
            "allow": ControlAction.ALLOW,
            "block": ControlAction.BLOCK,
            "degrade": ControlAction.DEGRADE,
            "throttle": ControlAction.THROTTLE,
        }

        action = action_map.get(validation.action, ControlAction.ALLOW)

        return ControlDecision(
            action=action,
            reason=validation.reason or f"Server validation: {validation.action}",
            degrade_to_model=validation.degrade_to_model,
            degrade_to_provider=validation.degrade_to_provider,
            budget_id=budget_id,
        )

    def _http_request_sync(
        self, path: str, method: str, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Make synchronous HTTP request to server using urllib."""
        import urllib.request
        import urllib.error

        http_url = (
            self.options.server_url.replace("wss://", "https://").replace(
                "ws://", "http://"
            )
        )

        try:
            data = json.dumps(body).encode("utf-8") if body else None

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.options.api_key}",
                "X-SDK-Instance-ID": self.options.instance_id or "",
            }

            req = urllib.request.Request(
                f"{http_url}{path}",
                data=data,
                headers=headers,
                method=method,
            )

            with urllib.request.urlopen(req, timeout=self.options.timeout_ms / 1000) as resp:
                if resp.status == 200:
                    return {"ok": True, "data": json.loads(resp.read().decode())}
                return {"ok": False, "status": resp.status}

        except urllib.error.HTTPError as e:
            logger.warning(f"[aden] HTTP error {e.code} on {path}")
            return {"ok": False, "status": e.code}
        except Exception as e:
            logger.warning(f"[aden] HTTP request failed: {e}")
            return {"ok": False, "error": str(e)}

    def _flush_event_queue_sync(self) -> None:
        """Flush queued events synchronously via HTTP."""
        if not self._event_queue:
            return

        events = self._event_queue.copy()
        self._event_queue.clear()
        event_count = len(events)
        logger.debug(f"[aden] Flushing {event_count} events to server")

        # Debug: log content_capture presence
        for e in events:
            if hasattr(e, 'data') and hasattr(e.data, 'content_capture'):
                cc = e.data.content_capture
                if cc:
                    logger.debug(f"[aden] Event has content_capture: system_prompt={cc.system_prompt is not None}, messages={cc.messages is not None}, response={cc.response_content is not None}")
                else:
                    logger.debug("[aden] Event content_capture is None")

        try:
            event_dicts = [asdict(e) for e in events]
            # Debug: log serialized content_capture
            for ed in event_dicts:
                if 'data' in ed and ed['data'].get('content_capture'):
                    logger.debug(f"[aden] Serialized content_capture keys: {list(ed['data']['content_capture'].keys())}")
            result = self._http_request_sync(
                "/v1/control/events",
                "POST",
                {"events": event_dicts},
            )
            if result.get("ok"):
                logger.debug(f"[aden] Successfully sent {event_count} events")
            else:
                logger.warning(
                    f"[aden] Failed to send {event_count} events to server: "
                    f"status={result.get('status', 'unknown')}"
                )
        except Exception as e:
            logger.warning(f"[aden] Failed to flush {event_count} events: {e}")

    def _start_sync_flush_thread(self) -> None:
        """Start the background thread for periodic sync flushing."""
        if self._sync_flush_thread is not None:
            return

        def flush_loop() -> None:
            while not self._sync_flush_stop.wait(timeout=self._sync_flush_interval):
                self._flush_event_queue_sync()
            # Final flush on stop
            self._flush_event_queue_sync()

        self._sync_flush_thread = threading.Thread(target=flush_loop, daemon=True)
        self._sync_flush_thread.start()

    def _stop_sync_flush_thread(self) -> None:
        """Stop the background flush thread."""
        if self._sync_flush_thread is None:
            return

        self._sync_flush_stop.set()
        self._sync_flush_thread.join(timeout=2.0)
        self._sync_flush_thread = None
        self._sync_flush_stop.clear()

    def report_metric_sync(self, event: MetricEvent) -> None:
        """Report a metric event to the server (sync version).

        Uses background thread for periodic flushing + batch size threshold.
        """
        # Start background flush thread on first use
        self._start_sync_flush_thread()

        # Inject context_id into metadata
        context_id = (
            self.options.get_context_id() if self.options.get_context_id else None
        )
        if context_id and event.metadata is None:
            event.metadata = {"context_id": context_id}
        elif context_id:
            event.metadata["context_id"] = context_id

        wrapper = MetricEventWrapper(
            event_type="metric",
            timestamp=datetime.now().isoformat(),
            sdk_instance_id=self.options.instance_id or "",
            data=event,
        )

        # Queue event - background thread flushes every second
        # Also flush immediately if batch size is reached
        self._queue_event(wrapper)
        if len(self._event_queue) >= self._sync_batch_size:
            self._flush_event_queue_sync()

        # Update local budget tracking for all matching budgets
        if self._cached_policy and self._cached_policy.budgets:
            if event.total_tokens > 0:
                estimated_cost = self._estimate_cost(event)
                metadata = event.metadata or {}

                for budget in self._cached_policy.budgets:
                    # Check if this budget applies to this event
                    should_update = False

                    # Global budgets always apply
                    if budget.budget_type == "global":
                        should_update = True
                    # Legacy context_id matching
                    elif budget.context_id and context_id:
                        should_update = budget.context_id == context_id
                    # Type-based matching
                    elif budget.budget_type == "agent":
                        agent = metadata.get("agent")
                        should_update = agent and (agent == budget.name or agent == budget.id)
                    elif budget.budget_type == "tenant":
                        tenant_id = metadata.get("tenant_id")
                        should_update = tenant_id and (tenant_id == budget.name or tenant_id == budget.id)
                    elif budget.budget_type == "customer":
                        customer_id = metadata.get("customer_id")
                        should_update = customer_id and (customer_id == budget.name or customer_id == budget.id)
                    elif budget.budget_type == "feature":
                        feature = metadata.get("feature")
                        should_update = feature and (feature == budget.name or feature == budget.id)
                    elif budget.budget_type == "tag":
                        request_tags = metadata.get("tags", [])
                        budget_tags = budget.tags or []
                        should_update = bool(set(request_tags) & set(budget_tags)) if request_tags and budget_tags else False

                    if should_update:
                        budget.current_spend_usd += estimated_cost

    async def report_metric(self, event: MetricEvent) -> None:
        """Report a metric event to the server."""
        # Inject context_id into metadata
        context_id = (
            self.options.get_context_id() if self.options.get_context_id else None
        )
        if context_id and event.metadata is None:
            event.metadata = {"context_id": context_id}
        elif context_id:
            event.metadata["context_id"] = context_id

        wrapper = MetricEventWrapper(
            event_type="metric",
            timestamp=datetime.now().isoformat(),
            sdk_instance_id=self.options.instance_id or "",
            data=event,
        )

        await self._send_event(wrapper)

        # Update local budget tracking for all matching budgets
        if self._cached_policy and self._cached_policy.budgets:
            if event.total_tokens > 0:
                estimated_cost = self._estimate_cost(event)
                metadata = event.metadata or {}

                for budget in self._cached_policy.budgets:
                    # Check if this budget applies to this event
                    should_update = False

                    # Global budgets always apply
                    if budget.budget_type == "global":
                        should_update = True
                    # Legacy context_id matching
                    elif budget.context_id and context_id:
                        should_update = budget.context_id == context_id
                    # Type-based matching
                    elif budget.budget_type == "agent":
                        agent = metadata.get("agent")
                        should_update = agent and (agent == budget.name or agent == budget.id)
                    elif budget.budget_type == "tenant":
                        tenant_id = metadata.get("tenant_id")
                        should_update = tenant_id and (tenant_id == budget.name or tenant_id == budget.id)
                    elif budget.budget_type == "customer":
                        customer_id = metadata.get("customer_id")
                        should_update = customer_id and (customer_id == budget.name or customer_id == budget.id)
                    elif budget.budget_type == "feature":
                        feature = metadata.get("feature")
                        should_update = feature and (feature == budget.name or feature == budget.id)
                    elif budget.budget_type == "tag":
                        request_tags = metadata.get("tags", [])
                        budget_tags = budget.tags or []
                        should_update = bool(set(request_tags) & set(budget_tags)) if request_tags and budget_tags else False

                    if should_update:
                        budget.current_spend_usd += estimated_cost

    def _estimate_cost(self, event: MetricEvent) -> float:
        """Estimate cost from a metric event using server pricing table."""
        if event.total_tokens == 0:
            return 0.0

        # Get pricing for this model (fetched from server on connect)
        rates = self._get_model_pricing(event.model)

        # Calculate cost (pricing is per 1M tokens)
        # Use cached_input rate for cached tokens if available
        cached_tokens = getattr(event, "cached_tokens", 0) or 0
        regular_input = max(0, event.input_tokens - cached_tokens)

        input_cost = regular_input * rates["input"] / 1_000_000
        cached_cost = cached_tokens * rates.get("cached_input", rates["input"] * 0.25) / 1_000_000
        output_cost = event.output_tokens * rates["output"] / 1_000_000

        return input_cost + cached_cost + output_cost

    def _get_model_pricing(self, model: str) -> dict[str, float]:
        """Get pricing for a model from cached pricing table."""
        if not model:
            return {"input": 1.0, "output": 3.0, "cached_input": 0.25}

        model_lower = model.lower()

        # Check direct match first
        if model_lower in self._pricing_cache:
            return self._pricing_cache[model_lower]

        # Check aliases
        if model_lower in self._pricing_aliases:
            canonical = self._pricing_aliases[model_lower]
            if canonical in self._pricing_cache:
                return self._pricing_cache[canonical]

        # Try prefix matching for versioned models
        for cached_model in self._pricing_cache:
            if model_lower.startswith(cached_model) or cached_model.startswith(model_lower):
                return self._pricing_cache[cached_model]

        # Fallback pricing for unknown models
        return {"input": 1.0, "output": 3.0, "cached_input": 0.25}

    async def _fetch_pricing(self) -> None:
        """Fetch pricing table from server and cache it."""
        try:
            import aiohttp
        except ImportError:
            logger.warning("[aden] aiohttp not installed, using fallback pricing")
            return

        try:
            http_url = (
                self.options.server_url.replace("wss://", "https://").replace(
                    "ws://", "http://"
                )
            )

            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.options.api_key}",
                }
                async with session.get(
                    f"{http_url}/tsdb/pricing",
                    headers=headers,
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        pricing = data.get("pricing", {})

                        # Build cache and alias map
                        for model, rates in pricing.items():
                            model_lower = model.lower()
                            self._pricing_cache[model_lower] = {
                                "input": rates.get("input", 1.0),
                                "output": rates.get("output", 3.0),
                                "cached_input": rates.get("cached_input", rates.get("input", 1.0) * 0.25),
                            }
                            # Index aliases
                            for alias in rates.get("aliases", []):
                                self._pricing_aliases[alias.lower()] = model_lower

                        logger.debug(f"[aden] Loaded pricing for {len(self._pricing_cache)} models")
                    else:
                        logger.warning(f"[aden] Failed to fetch pricing: {response.status}")
        except Exception as e:
            logger.warning(f"[aden] Failed to fetch pricing: {e}")

    async def report_control_event(self, event: ControlEvent) -> None:
        """Report a control event to the server."""
        event.event_type = "control"
        event.timestamp = datetime.now().isoformat()
        event.sdk_instance_id = self.options.instance_id or ""
        await self._send_event(event)

    def report_control_event_sync(self, event: ControlEvent) -> None:
        """Report a control event to the server (sync version).

        Uses background thread for batched sending, same as report_metric_sync.
        """
        # Start background flush thread on first use
        self._start_sync_flush_thread()

        event.event_type = "control"
        event.timestamp = datetime.now().isoformat()
        event.sdk_instance_id = self.options.instance_id or ""

        # Queue event - background thread flushes periodically
        self._queue_event(event)
        if len(self._event_queue) >= self._sync_batch_size:
            self._flush_event_queue_sync()

    async def report_error(
        self, message: str, error: Exception | None = None, trace_id: str | None = None
    ) -> None:
        """Report an error event."""
        self._errors_since_heartbeat += 1

        event = ErrorEvent(
            event_type="error",
            timestamp=datetime.now().isoformat(),
            sdk_instance_id=self.options.instance_id or "",
            message=message,
            code=type(error).__name__ if error else None,
            stack=str(error) if error else None,
            trace_id=trace_id,
        )

        await self._send_event(event)

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
        if not content_payloads:
            return

        try:
            result = await self._http_request(
                "/v1/control/content",
                "POST",
                {"items": content_payloads},
            )
            if result.get("ok"):
                logger.debug(f"[aden] Stored {len(content_payloads)} large content items")
            else:
                logger.warning(
                    f"[aden] Failed to store large content: status={result.get('status', 'unknown')}"
                )
        except Exception as e:
            logger.warning(f"[aden] Failed to store large content: {e}")

    def store_large_content_sync(self, content_payloads: list[dict[str, Any]]) -> None:
        """Store large content items on the control server (sync version).

        Args:
            content_payloads: List of content items to store
        """
        if not content_payloads:
            return

        try:
            import requests
        except ImportError:
            logger.warning("[aden] requests not installed, sync content storage disabled")
            return

        http_url = (
            self.options.server_url.replace("wss://", "https://").replace(
                "ws://", "http://"
            )
        )

        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.options.api_key}",
                "X-SDK-Instance-ID": self.options.instance_id or "",
            }
            response = requests.post(
                f"{http_url}/v1/control/content",
                headers=headers,
                json={"items": content_payloads},
                timeout=self.options.timeout_ms / 1000,
            )
            if response.status_code == 200:
                logger.debug(f"[aden] Stored {len(content_payloads)} large content items")
            else:
                logger.warning(
                    f"[aden] Failed to store large content: status={response.status_code}"
                )
        except Exception as e:
            logger.warning(f"[aden] Failed to store large content: {e}")

    async def _send_event(self, event: ServerEvent) -> None:
        """Send an event to the server."""
        # If Socket.IO is connected, send via Socket.IO
        if self._connected and self._sio:
            try:
                await self._sio.emit("message", asdict(event), namespace="/v1/control/ws")
                return
            except Exception:
                logger.warning("[aden] Socket.IO send failed, queuing event")

        # Otherwise queue for HTTP batch
        self._queue_event(event)

        # If not connected via Socket.IO, flush via HTTP
        if not self._connected:
            await self._flush_event_queue()

    def _queue_event(self, event: ServerEvent) -> None:
        """Queue an event for later sending."""
        if len(self._event_queue) >= self._max_queue_size:
            self._event_queue.pop(0)  # Drop oldest
        self._event_queue.append(event)

    async def _flush_event_queue(self) -> None:
        """Flush queued events."""
        if not self._event_queue:
            return

        events = self._event_queue.copy()
        self._event_queue.clear()
        event_count = len(events)
        logger.debug(f"[aden] Flushing {event_count} events to server")

        # If Socket.IO connected, send there
        if self._connected and self._sio:
            for event in events:
                try:
                    await self._sio.emit("message", asdict(event), namespace="/v1/control/ws")
                except Exception:
                    self._queue_event(event)
            logger.debug(f"[aden] Sent {event_count} events via Socket.IO")
            return

        # Otherwise send via HTTP batch
        try:
            result = await self._http_request(
                "/v1/control/events",
                "POST",
                {"events": [asdict(e) for e in events]},
            )
            if result.get("ok"):
                logger.debug(f"[aden] Successfully sent {event_count} events")
            else:
                logger.warning(
                    f"[aden] Failed to send {event_count} events to server: "
                    f"status={result.get('status', 'unknown')}"
                )
        except Exception as e:
            logger.warning(f"[aden] Failed to flush {event_count} events: {e}")

    async def _http_request(
        self, path: str, method: str, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Make HTTP request to server."""
        try:
            import aiohttp
        except ImportError:
            logger.warning("[aden] aiohttp not installed, HTTP requests disabled")
            return {"ok": False}

        http_url = (
            self.options.server_url.replace("wss://", "https://").replace(
                "ws://", "http://"
            )
        )

        try:
            timeout = aiohttp.ClientTimeout(total=self.options.timeout_ms / 1000)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.options.api_key}",
                    "X-SDK-Instance-ID": self.options.instance_id or "",
                }

                async with session.request(
                    method,
                    f"{http_url}{path}",
                    headers=headers,
                    json=body,
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {"ok": True, "data": data}
                    return {"ok": False, "status": response.status}
        except Exception as e:
            logger.warning(f"[aden] HTTP request failed: {e}")
            return {"ok": False, "error": str(e)}

    def is_connected(self) -> bool:
        """Check if connected to server (WebSocket)."""
        return self._connected

    def get_policy(self) -> ControlPolicy | None:
        """Get current cached policy."""
        return self._cached_policy


def create_control_agent(options: ControlAgentOptions) -> ControlAgent:
    """Create a control agent."""
    return ControlAgent(options)


def create_control_agent_emitter(
    agent: IControlAgent,
) -> Callable[[MetricEvent], Awaitable[None]]:
    """
    Create a metric emitter that sends to the control agent.

    This allows the control agent to work alongside other emitters:

    ```python
    agent = create_control_agent(ControlAgentOptions(...))

    await instrument(MeterOptions(
        emit_metric=create_multi_emitter([
            create_console_emitter(),
            create_control_agent_emitter(agent),
        ]),
    ))
    ```
    """

    async def emitter(event: MetricEvent) -> None:
        await agent.report_metric(event)

    return emitter
