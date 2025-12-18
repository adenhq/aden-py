"""
LiveKit Voice Agent integration for openai-meter.

Provides simple one-liner instrumentation for LiveKit voice agents with:
- Session-level cost tracking
- Budget enforcement with auto-disconnect
- Local file logging (JSONL)
- HTTP API transport (recommended for production)
- Custom emitter support

Usage:
    ```python
    from openai_meter.livekit import instrument

    async def entrypoint(ctx: JobContext):
        await ctx.connect()

        # One line - that's it!
        instrument(ctx, max_cost=0.50)

        # ... rest of agent code
    ```

Environment variables (all optional):
    METER_ENABLED: Set to "false" to disable metering entirely
    METER_MAX_COST: Default max cost per session in USD
    METER_LOG_DIR: Directory for metric log files
    METER_LOG_TO_FILE: Set to "false" to disable file logging
    METER_API_URL: API endpoint for metrics (recommended for production)
    METER_API_KEY: API key for authentication
    TIMESCALE_DSN: Direct DB connection (fallback, not recommended for production)
"""

import asyncio
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

from .file_logger import MetricFileLogger, DEFAULT_LOG_DIR
from .types import MetricEvent, NormalizedUsage

logger = logging.getLogger(__name__)

# Thread lock for global state
_global_lock = threading.Lock()

# Type alias for callbacks
BudgetCallback = Callable[[str, float, float], None]


# =============================================================================
# Cost Tracking
# =============================================================================

# Pricing tables (USD per 1M tokens/characters/minutes)
LLM_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
}

TTS_PRICING: dict[str, float] = {
    "tts-1": 15.0,      # per 1M characters
    "tts-1-hd": 30.0,   # per 1M characters
}

STT_PRICING: dict[str, float] = {
    "whisper-1": 0.006,     # per minute
    "google": 0.016,        # per minute (approximate)
}


@dataclass
class SessionCostTracker:
    """Tracks costs for a single voice session."""

    session_id: str
    room_name: str
    started_at: datetime = field(default_factory=datetime.now)

    # Token counts
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_requests: int = 0

    # TTS metrics
    tts_characters: int = 0
    tts_requests: int = 0

    # STT metrics
    stt_audio_seconds: float = 0.0
    stt_requests: int = 0

    # Cost tracking (USD)
    estimated_cost_usd: float = 0.0

    # Budget
    max_cost_usd: Optional[float] = None
    budget_warning_threshold: float = 0.8  # Warn at 80%
    budget_exceeded_triggered: bool = False  # Prevent callback from firing multiple times

    def add_llm_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "gpt-4o-mini",
    ) -> None:
        """Add LLM usage and estimate cost."""
        self.llm_input_tokens += input_tokens
        self.llm_output_tokens += output_tokens
        self.llm_requests += 1

        rates = LLM_PRICING.get(model, {"input": 0.5, "output": 1.5})
        input_cost = (input_tokens / 1_000_000) * rates["input"]
        output_cost = (output_tokens / 1_000_000) * rates["output"]
        self.estimated_cost_usd += input_cost + output_cost

    def add_tts_usage(self, characters: int, model: str = "tts-1") -> None:
        """Add TTS usage and estimate cost."""
        self.tts_characters += characters
        self.tts_requests += 1

        rate = TTS_PRICING.get(model, 15.0)
        self.estimated_cost_usd += (characters / 1_000_000) * rate

    def add_stt_usage(self, audio_seconds: float, model: str = "whisper-1") -> None:
        """Add STT usage and estimate cost."""
        self.stt_audio_seconds += audio_seconds
        self.stt_requests += 1

        # Determine rate based on model
        if "google" in model.lower() or "telephony" in model.lower():
            rate = STT_PRICING.get("google", 0.016)
        else:
            rate = STT_PRICING.get("whisper-1", 0.006)

        self.estimated_cost_usd += (audio_seconds / 60) * rate

    def check_budget(self) -> tuple[bool, Optional[str]]:
        """
        Check if session is within budget.

        Returns:
            Tuple of (is_ok, warning_message)
        """
        if self.max_cost_usd is None:
            return True, None

        if self.estimated_cost_usd >= self.max_cost_usd:
            return False, (
                f"Budget exceeded: ${self.estimated_cost_usd:.4f} >= "
                f"${self.max_cost_usd:.4f}"
            )

        if self.estimated_cost_usd >= self.max_cost_usd * self.budget_warning_threshold:
            pct = (self.estimated_cost_usd / self.max_cost_usd) * 100
            return True, (
                f"Budget warning: ${self.estimated_cost_usd:.4f} / "
                f"${self.max_cost_usd:.4f} ({pct:.0f}%)"
            )

        return True, None

    def get_summary(self) -> dict[str, Any]:
        """Get session cost summary."""
        return {
            "session_id": self.session_id,
            "room_name": self.room_name,
            "started_at": self.started_at.isoformat(),
            "duration_seconds": (datetime.now() - self.started_at).total_seconds(),
            "llm": {
                "requests": self.llm_requests,
                "input_tokens": self.llm_input_tokens,
                "output_tokens": self.llm_output_tokens,
                "total_tokens": self.llm_input_tokens + self.llm_output_tokens,
            },
            "tts": {
                "requests": self.tts_requests,
                "characters": self.tts_characters,
            },
            "stt": {
                "requests": self.stt_requests,
                "audio_seconds": self.stt_audio_seconds,
            },
            "estimated_cost_usd": self.estimated_cost_usd,
            "max_cost_usd": self.max_cost_usd,
        }


# =============================================================================
# LiveKit Meter
# =============================================================================


class LiveKitMeter:
    """
    Cost metering for LiveKit voice agents.

    Processes LiveKit's native MetricsCollectedEvent to track costs,
    enforce budgets, and log metrics.

    Example:
        ```python
        from openai_meter.livekit import LiveKitMeter

        meter = LiveKitMeter(
            max_cost_per_session=0.50,
            log_to_file=True,
        )

        # Start session
        tracker = meter.start_session("session_123", "room_name")

        # Process metrics (called from MetricsCollectedEvent handler)
        meter.process_metrics(metrics_event.metrics, "session_123")

        # End session
        summary = meter.end_session("session_123")
        ```
    """

    def __init__(
        self,
        max_cost_per_session: Optional[float] = None,
        log_to_file: bool = True,
        log_dir: str = DEFAULT_LOG_DIR,
        custom_emitter: Optional[Callable[[MetricEvent], None]] = None,
        on_budget_warning: Optional[BudgetCallback] = None,
        on_budget_exceeded: Optional[BudgetCallback] = None,
    ):
        """
        Initialize the cost meter.

        Args:
            max_cost_per_session: Maximum cost in USD per session (None = unlimited)
            log_to_file: Whether to write raw metrics to local JSONL files
            log_dir: Directory for metric log files
            custom_emitter: Custom function to receive MetricEvent objects
            on_budget_warning: Called when approaching budget (session_id, current, max)
            on_budget_exceeded: Called when budget exceeded (session_id, current, max)
        """
        self.max_cost_per_session = max_cost_per_session
        self.log_to_file = log_to_file
        self.custom_emitter = custom_emitter
        self.on_budget_warning = on_budget_warning
        self.on_budget_exceeded = on_budget_exceeded

        self._sessions: dict[str, SessionCostTracker] = {}
        self._file_logger: Optional[MetricFileLogger] = None

        if log_to_file:
            self._file_logger = MetricFileLogger(log_dir)

        logger.info(
            f"LiveKitMeter initialized "
            f"(max_cost=${max_cost_per_session or 'unlimited'}, "
            f"log_to_file={log_to_file})"
        )

    def start_session(
        self,
        session_id: str,
        room_name: str,
        max_cost_usd: Optional[float] = None,
    ) -> SessionCostTracker:
        """
        Start tracking a new session.

        Args:
            session_id: Unique session identifier (e.g., room.sid)
            room_name: LiveKit room name
            max_cost_usd: Override default max cost for this session

        Returns:
            SessionCostTracker instance
        """
        tracker = SessionCostTracker(
            session_id=session_id,
            room_name=room_name,
            max_cost_usd=max_cost_usd or self.max_cost_per_session,
        )
        self._sessions[session_id] = tracker

        logger.info(f"Started cost tracking for session {session_id} (room: {room_name})")

        if self._file_logger:
            self._file_logger.write_session_start(session_id, room_name)

        return tracker

    def end_session(self, session_id: str) -> Optional[dict[str, Any]]:
        """
        End a session and return final summary.

        Args:
            session_id: Session identifier

        Returns:
            Session summary dict or None if session not found
        """
        tracker = self._sessions.pop(session_id, None)
        if not tracker:
            logger.warning(f"Session {session_id} not found for ending")
            return None

        summary = tracker.get_summary()

        logger.info(
            f"Session {session_id} ended: "
            f"${summary['estimated_cost_usd']:.4f} "
            f"({summary['llm']['total_tokens']} tokens, "
            f"{summary['tts']['characters']} TTS chars, "
            f"{summary['stt']['audio_seconds']:.1f}s STT)"
        )

        if self._file_logger:
            self._file_logger.write_session_end(session_id, summary)

        return summary

    def get_session(self, session_id: str) -> Optional[SessionCostTracker]:
        """Get tracker for an active session."""
        return self._sessions.get(session_id)

    def process_metrics(
        self,
        metrics: Any,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Process metrics from LiveKit's MetricsCollectedEvent.

        Args:
            metrics: The metrics object from LiveKit
            session_id: Session ID to associate metrics with
        """
        # Find the session tracker
        tracker = None
        if session_id:
            tracker = self._sessions.get(session_id)
        elif len(self._sessions) == 1:
            tracker = list(self._sessions.values())[0]
            session_id = tracker.session_id

        # Process LLM metrics
        if hasattr(metrics, 'prompt_tokens') and hasattr(metrics, 'completion_tokens'):
            input_tokens = metrics.prompt_tokens or 0
            output_tokens = metrics.completion_tokens or 0
            model = getattr(metrics, 'model', 'gpt-4o-mini')
            latency_ms = getattr(metrics, 'duration', 0) * 1000 if hasattr(metrics, 'duration') else 0

            logger.debug(f"LLM metrics: {input_tokens} in, {output_tokens} out, model={model}")

            if tracker:
                tracker.add_llm_usage(input_tokens, output_tokens, model)
                self._check_and_report_budget(tracker)

            if self._file_logger and session_id:
                self._file_logger.write_llm_event(
                    session_id=session_id,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=model,
                    latency_ms=latency_ms,
                )

            if self.custom_emitter:
                event = MetricEvent(
                    trace_id=session_id or "unknown",
                    model=model,
                    stream=False,
                    latency_ms=latency_ms,
                    usage=NormalizedUsage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=input_tokens + output_tokens,
                    ),
                )
                self.custom_emitter(event)

        # Process TTS metrics
        elif hasattr(metrics, 'characters_count'):
            characters = metrics.characters_count or 0
            model = getattr(metrics, 'model', 'tts-1')
            ttfb = getattr(metrics, 'ttfb', 0) or 0
            duration = getattr(metrics, 'duration', 0) or 0

            logger.debug(f"TTS metrics: {characters} characters, model={model}")

            if tracker:
                tracker.add_tts_usage(characters, model)
                self._check_and_report_budget(tracker)

            if self._file_logger and session_id:
                self._file_logger.write_tts_event(
                    session_id=session_id,
                    characters=characters,
                    model=model,
                )

            if self.custom_emitter:
                event = MetricEvent(
                    trace_id=session_id or "unknown",
                    model=model,
                    stream=False,
                    latency_ms=ttfb * 1000,  # ttfb is in seconds
                    metadata={"type": "tts", "characters": characters, "duration": duration},
                )
                self.custom_emitter(event)

        # Process STT metrics
        elif hasattr(metrics, 'audio_duration'):
            audio_duration = metrics.audio_duration or 0
            model = getattr(metrics, 'model', 'whisper-1')
            duration = getattr(metrics, 'duration', 0) or 0

            logger.debug(f"STT metrics: {audio_duration:.2f}s audio, model={model}")

            if tracker:
                tracker.add_stt_usage(audio_duration, model)
                self._check_and_report_budget(tracker)

            if self._file_logger and session_id:
                self._file_logger.write_stt_event(
                    session_id=session_id,
                    audio_seconds=audio_duration,
                    model=model,
                )

            if self.custom_emitter:
                event = MetricEvent(
                    trace_id=session_id or "unknown",
                    model=model,
                    stream=False,
                    latency_ms=duration * 1000,  # duration is in seconds
                    metadata={"type": "stt", "audio_duration": audio_duration},
                )
                self.custom_emitter(event)

    def _check_and_report_budget(self, tracker: SessionCostTracker) -> None:
        """Check budget and report warnings/exceeded."""
        is_ok, message = tracker.check_budget()

        if message:
            if is_ok:
                # Warning
                logger.warning(message)
                if self.on_budget_warning:
                    self.on_budget_warning(
                        tracker.session_id,
                        tracker.estimated_cost_usd,
                        tracker.max_cost_usd or 0,
                    )
            else:
                # Exceeded - only trigger callback once to prevent infinite loops
                # (the goodbye TTS would generate more metrics, triggering this again)
                if tracker.budget_exceeded_triggered:
                    return
                tracker.budget_exceeded_triggered = True

                logger.error(message)
                if self.on_budget_exceeded:
                    self.on_budget_exceeded(
                        tracker.session_id,
                        tracker.estimated_cost_usd,
                        tracker.max_cost_usd or 0,
                    )


# =============================================================================
# Simple One-Liner Instrumentation
# =============================================================================

# Global meter instance (protected by _global_lock)
_meter: Optional[LiveKitMeter] = None

# Track instrumented sessions to prevent double-instrumentation (protected by _global_lock)
_instrumented_sessions: set[str] = set()

# Global transport emitter (created lazily from METER_API_URL or TIMESCALE_DSN)
# Protected by _global_lock
_transport_emitter: Optional[Any] = None
_transport_init_attempted: bool = False
_transport_type: Optional[str] = None  # "http" or "timescale"


def _get_transport_emitter() -> Optional[Any]:
    """Get or create the transport emitter from environment variables.

    Thread-safe: uses _global_lock for initialization.

    Priority:
    1. METER_API_URL - HTTP API transport (recommended)
    2. TIMESCALE_DSN - Direct database connection (fallback)
    """
    global _transport_emitter, _transport_init_attempted, _transport_type

    # Fast path: already initialized
    if _transport_init_attempted:
        return _transport_emitter

    with _global_lock:
        # Double-check after acquiring lock
        if _transport_init_attempted:
            return _transport_emitter

        _transport_init_attempted = True

        # Check for HTTP API transport first (recommended)
        api_url = os.environ.get('METER_API_URL')
        if api_url:
            try:
                from .http_transport import HttpTransport
                api_key = os.environ.get('METER_API_KEY')
                _transport_emitter = HttpTransport(
                    api_url=api_url,
                    api_key=api_key,
                    batch_size=50,
                    flush_interval=5.0,
                )
                _transport_type = "http"
                logger.info(f"HTTP transport initialized: {api_url}")
                return _transport_emitter
            except Exception as e:
                logger.warning(f"Failed to initialize HTTP transport: {e}")

        # Fallback to TimescaleDB direct connection
        dsn = os.environ.get('TIMESCALE_DSN')
        if dsn:
            try:
                from .timescale import SyncTimescaleEmitter
                _transport_emitter = SyncTimescaleEmitter(
                    dsn=dsn,
                    batch_size=50,
                    flush_interval=5.0,
                )
                _transport_type = "timescale"
                logger.info("TimescaleDB transport initialized (direct connection)")
                return _transport_emitter
            except ImportError:
                logger.warning(
                    "TIMESCALE_DSN is set but psycopg2 is not installed. "
                    "Install with: pip install openai-meter[timescale-sync]"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize TimescaleDB transport: {e}")

        return None


def get_meter() -> Optional[LiveKitMeter]:
    """Get the global meter instance."""
    return _meter


def instrument(
    ctx: Any,
    max_cost: Optional[float] = None,
    on_budget_exceeded: Optional[BudgetCallback] = None,
    on_budget_warning: Optional[BudgetCallback] = None,
    log_to_file: bool = True,
    log_dir: str = DEFAULT_LOG_DIR,
    custom_emitter: Optional[Callable[[MetricEvent], None]] = None,
) -> SessionCostTracker:
    """
    One-liner to instrument a LiveKit agent with cost metering.

    This function:
    1. Initializes the meter (if not already done)
    2. Starts a session for this room
    3. Auto-registers the metrics callback
    4. Auto-registers cleanup on disconnect
    5. Provides default budget enforcement (graceful disconnect)

    Usage:
        ```python
        from openai_meter.livekit import instrument

        async def entrypoint(ctx: JobContext):
            await ctx.connect()

            # One line - that's it!
            instrument(ctx)

            # Or with options:
            instrument(ctx, max_cost=0.50, log_dir="./my_logs")

            # ... rest of agent code
        ```

    Environment variables (all optional):
        METER_ENABLED: Set to "false" to disable metering entirely
        METER_MAX_COST: Default max cost per session in USD
        METER_LOG_DIR: Directory for metric log files
        METER_LOG_TO_FILE: Set to "false" to disable file logging
        TIMESCALE_DSN: PostgreSQL/TimescaleDB connection string (auto-enables DB persistence)

    Args:
        ctx: LiveKit JobContext
        max_cost: Maximum cost in USD for this session (default: from env or None)
        on_budget_exceeded: Custom callback when budget exceeded (default: disconnect)
        on_budget_warning: Custom callback when approaching budget (default: log warning)
        log_to_file: Whether to write metrics to local files
        log_dir: Directory for metric log files
        custom_emitter: Optional custom emitter for metrics

    Returns:
        SessionCostTracker for this session
    """
    global _meter

    # Check if metering is disabled via environment
    if os.environ.get('METER_ENABLED', 'true').lower() == 'false':
        logger.info("Metering disabled via METER_ENABLED=false")
        return SessionCostTracker(session_id="disabled", room_name="disabled")

    # Get session ID and room name
    # Note: ctx.room.sid may be a coroutine in some versions, use room.name instead
    session_id = ctx.room.name
    room_name = ctx.room.name

    # Thread-safe check and registration of session
    with _global_lock:
        # Prevent double-instrumentation
        if session_id in _instrumented_sessions:
            logger.warning(f"Session {session_id} already instrumented, skipping")
            if _meter:
                tracker = _meter.get_session(session_id)
                if tracker:
                    return tracker
            return SessionCostTracker(session_id=session_id, room_name=room_name)

        _instrumented_sessions.add(session_id)

    # Get configuration from environment with parameter overrides
    effective_max_cost = max_cost
    if effective_max_cost is None:
        env_max_cost = os.environ.get('METER_MAX_COST')
        if env_max_cost:
            try:
                effective_max_cost = float(env_max_cost)
            except ValueError:
                logger.warning(f"Invalid METER_MAX_COST value: {env_max_cost}")

    effective_log_to_file = log_to_file
    if os.environ.get('METER_LOG_TO_FILE', 'true').lower() == 'false':
        effective_log_to_file = False

    effective_log_dir = os.environ.get('METER_LOG_DIR', log_dir)

    # Create default budget exceeded handler
    async def default_budget_exceeded(sid: str, current: float, max_budget: float) -> None:
        """Default handler: log and disconnect gracefully."""
        logger.error(
            f"Budget exceeded for session {sid}: "
            f"${current:.4f} >= ${max_budget:.4f}. Disconnecting."
        )
        try:
            await ctx.room.disconnect()
        except Exception as e:
            logger.error(f"Failed to disconnect after budget exceeded: {e}")

    # Wrapper to handle sync/async callbacks
    def budget_exceeded_wrapper(sid: str, current: float, max_budget: float) -> None:
        callback = on_budget_exceeded or default_budget_exceeded
        if asyncio.iscoroutinefunction(callback):
            asyncio.create_task(callback(sid, current, max_budget))
        else:
            callback(sid, current, max_budget)

    def budget_warning_wrapper(sid: str, current: float, max_budget: float) -> None:
        if on_budget_warning:
            if asyncio.iscoroutinefunction(on_budget_warning):
                asyncio.create_task(on_budget_warning(sid, current, max_budget))
            else:
                on_budget_warning(sid, current, max_budget)

    # Build effective emitter (combine transport + custom if both present)
    transport_emitter = _get_transport_emitter()
    effective_emitter = custom_emitter

    if transport_emitter and custom_emitter:
        # Combine both emitters
        def combined_emitter(event: MetricEvent) -> None:
            transport_emitter(event)
            custom_emitter(event)
        effective_emitter = combined_emitter
    elif transport_emitter:
        effective_emitter = transport_emitter

    # Initialize meter if not already done
    if _meter is None:
        _meter = LiveKitMeter(
            max_cost_per_session=effective_max_cost,
            log_to_file=effective_log_to_file,
            log_dir=effective_log_dir,
            custom_emitter=effective_emitter,
            on_budget_exceeded=budget_exceeded_wrapper,
            on_budget_warning=budget_warning_wrapper,
        )
    else:
        # Update callbacks on existing meter
        _meter.on_budget_exceeded = budget_exceeded_wrapper
        _meter.on_budget_warning = budget_warning_wrapper

    # Start session
    tracker = _meter.start_session(
        session_id=session_id,
        room_name=room_name,
        max_cost_usd=effective_max_cost,
    )

    # Auto-register metrics callback on room
    @ctx.room.on("metrics_collected")
    def _on_metrics_collected(event: Any) -> None:
        if _meter:
            _meter.process_metrics(event.metrics, session_id)

    # Auto-register cleanup on disconnect
    @ctx.room.on("disconnected")
    def _on_disconnected() -> None:
        if _meter:
            _meter.end_session(session_id)
        # Thread-safe removal from instrumented sessions
        with _global_lock:
            _instrumented_sessions.discard(session_id)
        # Flush transport emitter to ensure metrics are persisted
        if transport_emitter and hasattr(transport_emitter, 'flush'):
            try:
                transport_emitter.flush()
            except Exception as e:
                logger.warning(f"Failed to flush transport emitter: {e}")

    logger.info(
        f"Instrumented session {session_id} "
        f"(max_cost=${effective_max_cost or 'unlimited'}, "
        f"log_to_file={effective_log_to_file}, "
        f"transport={_transport_type or 'none'})"
    )

    return tracker
