"""
Control Agent - Bidirectional communication with control server.

Emits: metrics, control events, heartbeat
Receives: control policies (budgets, throttle, block, degrade)

Uses WebSocket for real-time communication with HTTP polling fallback.
"""

import asyncio
import json
import logging
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
        )

        # WebSocket state
        self._ws: Any = None
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

    async def connect(self) -> None:
        """Connect to the control server."""
        url = self.options.server_url

        # Determine transport based on URL scheme
        if url.startswith("wss://") or url.startswith("ws://"):
            await self._connect_websocket()
        else:
            # HTTP-only mode: just use polling
            await self._start_polling()

        # Start heartbeat
        self._start_heartbeat()

    async def _connect_websocket(self) -> None:
        """Connect via WebSocket."""
        try:
            # Try to import websockets library
            import websockets
        except ImportError:
            logger.warning("[aden] websockets library not installed, using HTTP polling only")
            await self._start_polling()
            return

        try:
            ws_url = f"{self.options.server_url}/v1/control/ws"
            headers = {
                "Authorization": f"Bearer {self.options.api_key}",
                "X-SDK-Instance-ID": self.options.instance_id,
            }

            try:
                self._ws = await asyncio.wait_for(
                    websockets.connect(ws_url, extra_headers=headers),
                    timeout=self.options.timeout_ms / 1000,
                )
                self._connected = True
                self._reconnect_attempts = 0
                logger.info("[aden] WebSocket connected to control server")

                # Start message handler
                asyncio.create_task(self._handle_websocket_messages())

                # Flush queued events
                await self._flush_event_queue()

            except asyncio.TimeoutError:
                logger.warning("[aden] WebSocket connection timeout, using polling")
                await self._start_polling()
            except Exception as e:
                logger.warning(f"[aden] WebSocket connection failed: {e}, using polling")
                await self._start_polling()

        except Exception as e:
            logger.warning(f"[aden] WebSocket setup failed: {e}, using polling")
            await self._start_polling()

    async def _handle_websocket_messages(self) -> None:
        """Handle incoming WebSocket messages."""
        if self._ws is None:
            return

        try:
            async for message in self._ws:
                try:
                    data = json.loads(message)
                    if data.get("type") == "policy":
                        self._cached_policy = self._parse_policy(data.get("policy", {}))
                        self._last_policy_fetch = time.time()
                        logger.info(f"[aden] Policy updated: {self._cached_policy.version}")
                    elif data.get("type") == "command":
                        logger.info(f"[aden] Command received: {data}")
                except json.JSONDecodeError:
                    logger.warning("[aden] Failed to parse WebSocket message")
        except Exception as e:
            logger.warning(f"[aden] WebSocket error: {e}")
            self._connected = False
            self._schedule_reconnect()
            await self._start_polling()

    def _parse_policy(self, data: dict[str, Any]) -> ControlPolicy:
        """Parse policy JSON into ControlPolicy object."""
        from .control_types import (
            AlertRule,
            BlockRule,
            BudgetRule,
            DegradeRule,
            ThrottleRule,
        )

        return ControlPolicy(
            version=data.get("version", "unknown"),
            updated_at=data.get("updated_at", ""),
            budgets=[
                BudgetRule(
                    context_id=b.get("context_id", ""),
                    limit_usd=b.get("limit_usd", 0),
                    current_spend_usd=b.get("current_spend_usd", 0),
                    action_on_exceed=ControlAction(b.get("action_on_exceed", "block")),
                    degrade_to_model=b.get("degrade_to_model"),
                )
                for b in data.get("budgets", [])
            ],
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
        # Stop the background flush thread (which also does final flush)
        self._stop_sync_flush_thread()

        self._stop_polling()
        self._stop_heartbeat()

        if self._reconnect_task:
            self._reconnect_task.cancel()
            self._reconnect_task = None

        self._connected = False

    async def disconnect(self) -> None:
        """Disconnect from the control server."""
        # Flush any remaining events
        await self._flush_event_queue()

        self._stop_polling()
        self._stop_heartbeat()

        if self._reconnect_task:
            self._reconnect_task.cancel()
            self._reconnect_task = None

        if self._ws:
            await self._ws.close()
            self._ws = None

        self._connected = False

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

        # 3. Check budget limits
        if request.context_id:
            for budget in policy.budgets:
                if budget.context_id == request.context_id:
                    projected_spend = budget.current_spend_usd + (
                        request.estimated_cost or 0
                    )
                    if projected_spend > budget.limit_usd:
                        if (
                            budget.action_on_exceed == ControlAction.DEGRADE
                            and budget.degrade_to_model
                        ):
                            return ControlDecision(
                                action=ControlAction.DEGRADE,
                                reason=f"Budget exceeded: ${projected_spend:.4f} > ${budget.limit_usd}",
                                degrade_to_model=budget.degrade_to_model,
                                throttle_delay_ms=throttle_info["delay_ms"]
                                if throttle_info
                                else None,
                            )
                        return ControlDecision(
                            action=budget.action_on_exceed,
                            reason=f"Budget exceeded: ${projected_spend:.4f} > ${budget.limit_usd}",
                        )

                    # Check degradation rules based on budget threshold
                    for degrade in policy.degradations:
                        if (
                            degrade.from_model == request.model
                            and degrade.trigger == "budget_threshold"
                            and degrade.threshold_percent
                        ):
                            usage_percent = (
                                budget.current_spend_usd / budget.limit_usd
                            ) * 100
                            if usage_percent >= degrade.threshold_percent:
                                return ControlDecision(
                                    action=ControlAction.DEGRADE,
                                    reason=f"Budget at {usage_percent:.1f}% (threshold: {degrade.threshold_percent}%)",
                                    degrade_to_model=degrade.to_model,
                                    throttle_delay_ms=throttle_info["delay_ms"]
                                    if throttle_info
                                    else None,
                                )

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

        try:
            events = self._event_queue.copy()
            self._event_queue.clear()
            self._http_request_sync(
                "/v1/control/events",
                "POST",
                {"events": [asdict(e) for e in events]},
            )
        except Exception as e:
            logger.warning(f"[aden] Failed to flush event queue sync: {e}")

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

        # Update local budget tracking
        if self._cached_policy and self._cached_policy.budgets:
            if event.total_tokens > 0:
                estimated_cost = self._estimate_cost(event)
                if context_id:
                    for budget in self._cached_policy.budgets:
                        if budget.context_id == context_id:
                            budget.current_spend_usd += estimated_cost
                            break

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

        # Update local budget tracking
        if self._cached_policy and self._cached_policy.budgets:
            if event.total_tokens > 0:
                estimated_cost = self._estimate_cost(event)
                context_id = (
                    self.options.get_context_id() if self.options.get_context_id else None
                )
                if context_id:
                    for budget in self._cached_policy.budgets:
                        if budget.context_id == context_id:
                            budget.current_spend_usd += estimated_cost
                            break

    def _estimate_cost(self, event: MetricEvent) -> float:
        """Estimate cost from a metric event."""
        if event.total_tokens == 0:
            return 0.0
        # Simplified cost estimation
        input_cost = event.input_tokens * 0.00001  # $0.01 per 1K tokens
        output_cost = event.output_tokens * 0.00003  # $0.03 per 1K tokens
        return input_cost + output_cost

    async def report_control_event(self, event: ControlEvent) -> None:
        """Report a control event to the server."""
        event.event_type = "control"
        event.timestamp = datetime.now().isoformat()
        event.sdk_instance_id = self.options.instance_id or ""
        await self._send_event(event)

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

    async def _send_event(self, event: ServerEvent) -> None:
        """Send an event to the server."""
        # If WebSocket is connected, send via WebSocket
        if self._connected and self._ws:
            try:
                await self._ws.send(json.dumps(asdict(event)))
                return
            except Exception:
                logger.warning("[aden] WebSocket send failed, queuing event")

        # Otherwise queue for HTTP batch
        self._queue_event(event)

        # If not connected via WebSocket, flush via HTTP
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

        # If WebSocket connected, send there
        if self._connected and self._ws:
            events = self._event_queue.copy()
            self._event_queue.clear()
            for event in events:
                try:
                    await self._ws.send(json.dumps(asdict(event)))
                except Exception:
                    self._queue_event(event)
            return

        # Otherwise send via HTTP batch
        try:
            events = self._event_queue.copy()
            self._event_queue.clear()
            await self._http_request(
                "/v1/control/events",
                "POST",
                {"events": [asdict(e) for e in events]},
            )
        except Exception as e:
            logger.warning(f"[aden] Failed to flush event queue: {e}")

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
