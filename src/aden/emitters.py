"""
Metric emitters for various use cases.

Emitters are functions that receive MetricEvent objects and handle them
(logging, sending to backends, batching, etc.).
"""

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime
from threading import Lock, Timer
from typing import Any, TypeVar

from .types import MetricEmitter, MetricEvent

logger = logging.getLogger(__name__)

T = TypeVar("T")


def create_console_emitter(
    level: str = "info",
    pretty: bool = True,
) -> MetricEmitter:
    """
    A simple console emitter for development/debugging.

    Args:
        level: Log level - "info" logs all events, "warn" logs only errors
        pretty: Whether to pretty-print the output

    Returns:
        A metric emitter function
    """

    def emit(event: MetricEvent) -> None:
        if level == "warn" and not event.error:
            return

        prefix = "X" if event.error else "+"
        summary_parts = [
            f"{prefix} [{event.trace_id[:8]}]",
            event.model,
            "(stream)" if event.stream else "",
            f"{event.latency_ms:.0f}ms",
        ]
        summary = " ".join(filter(None, summary_parts))

        if pretty and event.total_tokens > 0:
            print(summary)
            print(f"  tokens: {event.input_tokens} in / {event.output_tokens} out")
            if event.cached_tokens > 0:
                print(f"  cached: {event.cached_tokens}")
            if event.reasoning_tokens > 0:
                print(f"  reasoning: {event.reasoning_tokens}")
            if event.tool_names:
                print(f"  tools: {event.tool_names}")
            if event.agent_stack:
                print(f"  agent: {' > '.join(event.agent_stack)}")
            if event.call_site_file and event.call_site_line:
                print(f"  call_site: {event.call_site_file}:{event.call_site_line}")
            if event.error:
                print(f"  error: {event.error}")
        else:
            print(summary)

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
