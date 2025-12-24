"""
Metric emitters for various use cases.

Emitters are functions that receive MetricEvent objects and handle them
(logging, sending to backends, batching, etc.).
"""

import asyncio
import logging
import os
import sys
from collections.abc import Callable
from datetime import datetime
from threading import Lock, Timer
from typing import Any, Literal, TypeVar

from .types import MetricEmitter, MetricEvent

logger = logging.getLogger(__name__)

# Log level mapping
_LEVEL_MAP = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


def _get_log_level_from_env() -> int:
    """Get log level from ADEN_LOG_LEVEL environment variable."""
    env_level = os.environ.get("ADEN_LOG_LEVEL", "info").lower()
    return _LEVEL_MAP.get(env_level, logging.INFO)


def _setup_aden_logger() -> None:
    """Set up the aden logger with default configuration based on env var."""
    aden_logger = logging.getLogger("aden")

    # Only set up if not already configured
    if not aden_logger.handlers:
        log_level = _get_log_level_from_env()
        aden_logger.setLevel(log_level)

        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
        aden_logger.addHandler(handler)

        # Prevent propagation to root logger (avoids duplicate logs)
        aden_logger.propagate = False


# Auto-configure on import
_setup_aden_logger()


def configure_logging(
    level: Literal["debug", "info", "warning", "error"] | None = None,
    metrics_level: Literal["debug", "info", "warning", "error"] | None = None,
    format: str | None = None,
) -> None:
    """
    Configure Aden logging with appropriate log levels.

    Log levels by logger:
    - `aden`: Main logger for connection status, warnings, errors
    - `aden.metrics`: Metric output (when using `use_logging=True` in console emitter)

    Args:
        level: Log level for the main `aden` logger.
               If not specified, reads from ADEN_LOG_LEVEL env var (default: "info").
               "debug" - All logs including detailed flush info
               "info" - Connection status, SDK instrumentation (default)
               "warning" - Only warnings and errors
               "error" - Only errors
        metrics_level: Log level for the `aden.metrics` logger.
                       Defaults to same as `level`.
                       Use "debug" to see detailed metric output.
        format: Custom log format. Defaults to "[%(levelname)s] %(name)s: %(message)s"

    Environment Variables:
        ADEN_LOG_LEVEL: Set default log level ("debug", "info", "warning", "error")

    Example:
        ```python
        # Using environment variable (recommended for production)
        # export ADEN_LOG_LEVEL=warning

        # Or programmatically:
        from aden import configure_logging, instrument, MeterOptions, create_console_emitter

        # Enable verbose logging for debugging
        configure_logging(level="debug", metrics_level="debug")

        # Or just show warnings and errors
        configure_logging(level="warning")

        # Use logging-based console emitter
        instrument(MeterOptions(
            emit_metric=create_console_emitter(use_logging=True),
        ))
        ```

    Note:
        This function configures Python's logging module. If you have your own
        logging configuration, you can skip this and configure the `aden` and
        `aden.metrics` loggers directly.
    """
    # Get level from param or env var
    if level is None:
        log_level = _get_log_level_from_env()
    else:
        log_level = _LEVEL_MAP.get(level, logging.INFO)

    metrics_log_level = _LEVEL_MAP.get(metrics_level, log_level) if metrics_level else log_level
    log_format = format or "[%(levelname)s] %(name)s: %(message)s"

    # Configure main aden logger
    aden_logger = logging.getLogger("aden")
    aden_logger.setLevel(log_level)

    # Configure metrics logger
    metrics_logger = logging.getLogger("aden.metrics")
    metrics_logger.setLevel(metrics_log_level)

    # Update or add handler
    if aden_logger.handlers:
        aden_logger.handlers[0].setFormatter(logging.Formatter(log_format))
    else:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(log_format))
        aden_logger.addHandler(handler)

    # Prevent propagation to root logger (avoids duplicate logs)
    aden_logger.propagate = False
    metrics_logger.propagate = True  # Propagate to aden logger

T = TypeVar("T")


def create_console_emitter(
    level: str = "info",
    pretty: bool = True,
    use_logging: bool = False,
) -> MetricEmitter:
    """
    A simple console emitter for development/debugging.

    Args:
        level: Log level - "info" logs all events, "warn" logs only errors
        pretty: Whether to pretty-print the output
        use_logging: If True, use Python logging instead of print().
                     Metrics are logged at DEBUG level, errors at WARNING.

    Returns:
        A metric emitter function
    """
    metric_logger = logging.getLogger("aden.metrics")

    def emit(event: MetricEvent) -> None:
        if level == "warn" and not event.error:
            return

        prefix = "X" if event.error else "+"
        summary_parts = [
            f"{prefix} [{event.trace_id[:8]}]",
            event.provider,
            event.model,
            "(stream)" if event.stream else "",
            f"{event.latency_ms:.0f}ms",
        ]
        summary = " ".join(filter(None, summary_parts))

        if pretty and event.total_tokens > 0:
            lines = [summary]
            lines.append(f"  tokens: {event.input_tokens} in / {event.output_tokens} out")
            if event.cached_tokens > 0:
                lines.append(f"  cached: {event.cached_tokens}")
            if event.reasoning_tokens > 0:
                lines.append(f"  reasoning: {event.reasoning_tokens}")
            if event.tool_names:
                lines.append(f"  tools: {event.tool_names}")
            if event.agent_stack:
                lines.append(f"  agent: {' > '.join(event.agent_stack)}")
            if event.call_site_file and event.call_site_line:
                lines.append(f"  call_site: {event.call_site_file}:{event.call_site_line}")
            if event.error:
                lines.append(f"  error: {event.error}")

            output = "\n".join(lines)
        else:
            output = summary

        if use_logging:
            if event.error:
                metric_logger.warning(output)
            else:
                metric_logger.debug(output)
        else:
            print(output)

    return emit


class BatchEmitter:
    """
    An emitter that batches metrics and flushes periodically.

    Attributes:
        flush: Manually flush the current batch
        stop: Stop the emitter and flush remaining events
    """

    def __init__(
        self,
        flush_callback: Callable[[list[MetricEvent]], Any],
        max_batch_size: int = 100,
        flush_interval: float = 5.0,
    ):
        self._flush_callback = flush_callback
        self._max_batch_size = max_batch_size
        self._flush_interval = flush_interval
        self._batch: list[MetricEvent] = []
        self._lock = Lock()
        self._timer: Timer | None = None
        self._start_timer()

    def _start_timer(self) -> None:
        """Start the periodic flush timer."""
        self._timer = Timer(self._flush_interval, self._timer_flush)
        self._timer.daemon = True
        self._timer.start()

    def _timer_flush(self) -> None:
        """Called by timer to flush and restart."""
        self.flush()
        self._start_timer()

    def __call__(self, event: MetricEvent) -> None:
        """Add an event to the batch."""
        with self._lock:
            self._batch.append(event)
            if len(self._batch) >= self._max_batch_size:
                self._do_flush()

    def _do_flush(self) -> None:
        """Internal flush (assumes lock is held)."""
        if not self._batch:
            return
        to_flush = self._batch
        self._batch = []
        try:
            result = self._flush_callback(to_flush)
            if asyncio.iscoroutine(result):
                asyncio.create_task(result)
        except Exception as e:
            logger.error(f"Error flushing metrics batch: {e}")

    def flush(self) -> None:
        """Manually flush the current batch."""
        with self._lock:
            self._do_flush()

    def stop(self) -> None:
        """Stop the emitter and flush remaining events."""
        if self._timer:
            self._timer.cancel()
            self._timer = None
        self.flush()


def create_batch_emitter(
    flush: Callable[[list[MetricEvent]], Any],
    max_batch_size: int = 100,
    flush_interval: float = 5.0,
) -> BatchEmitter:
    """
    Creates an emitter that batches metrics and flushes periodically.

    Args:
        flush: Callback to handle batched events
        max_batch_size: Maximum batch size before auto-flush
        flush_interval: Maximum time (seconds) to wait before flushing

    Returns:
        A BatchEmitter instance with flush() and stop() methods
    """
    return BatchEmitter(flush, max_batch_size, flush_interval)


def create_multi_emitter(emitters: list[MetricEmitter]) -> MetricEmitter:
    """
    Creates an emitter that writes to multiple destinations.

    Args:
        emitters: List of emitters to forward events to

    Returns:
        A metric emitter that forwards to all destinations
    """

    async def emit(event: MetricEvent) -> None:
        tasks = []
        for emitter in emitters:
            result = emitter(event)
            if asyncio.iscoroutine(result):
                tasks.append(result)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    return emit


def create_filtered_emitter(
    emitter: MetricEmitter,
    filter_fn: Callable[[MetricEvent], bool],
) -> MetricEmitter:
    """
    Creates an emitter that filters events before passing to another emitter.

    Args:
        emitter: The downstream emitter
        filter_fn: Function that returns True for events to pass through

    Returns:
        A filtered metric emitter
    """

    def emit(event: MetricEvent) -> Any:
        if filter_fn(event):
            return emitter(event)
        return None

    return emit


def create_transform_emitter(
    emitter: Callable[[T], Any],
    transform: Callable[[MetricEvent], T],
) -> MetricEmitter:
    """
    Creates an emitter that transforms events before passing to another emitter.

    Args:
        emitter: The downstream handler
        transform: Function to transform MetricEvent to another type

    Returns:
        A transforming metric emitter
    """

    def emit(event: MetricEvent) -> Any:
        return emitter(transform(event))

    return emit


def create_noop_emitter() -> MetricEmitter:
    """
    Creates a no-op emitter (useful for testing or disabling metrics).

    Returns:
        A metric emitter that does nothing
    """

    def emit(event: MetricEvent) -> None:
        pass

    return emit


class MemoryEmitter:
    """
    An emitter that collects metrics in memory (useful for testing).

    Attributes:
        events: List of collected events
        clear: Clear collected events
    """

    def __init__(self) -> None:
        self.events: list[MetricEvent] = []

    def __call__(self, event: MetricEvent) -> None:
        self.events.append(event)

    def clear(self) -> None:
        """Clear collected events."""
        self.events.clear()


def create_memory_emitter() -> MemoryEmitter:
    """
    Helper to collect metrics in memory (useful for testing).

    Returns:
        A MemoryEmitter instance with events list and clear() method
    """
    return MemoryEmitter()
