"""
HTTP transport for sending metrics to a central API endpoint.

This is the recommended approach for production - clients send metrics
to your API, which handles storage, aggregation, and multi-tenancy.
"""

import json
import logging
import os
import queue
import threading
import time
from dataclasses import asdict
from typing import Any, Callable, Optional

from .types import MetricEvent

logger = logging.getLogger(__name__)


# Callback type for queue overflow events
QueueOverflowHandler = Callable[[int], None]
"""Called when metrics are dropped due to queue overflow. Receives count of total dropped."""


class HttpTransport:
    """
    HTTP transport that batches and sends metrics to an API endpoint.

    Features:
    - Batched sending for efficiency
    - Background thread for non-blocking sends
    - Retry with exponential backoff
    - Configurable via environment variables
    - Queue overflow tracking and callbacks

    Environment variables:
        METER_API_URL: API endpoint URL (required)
        METER_API_KEY: API key for authentication (optional)
        METER_BATCH_SIZE: Batch size before sending (default: 50)
        METER_FLUSH_INTERVAL: Seconds between flushes (default: 5.0)

    Example:
        ```python
        transport = HttpTransport(
            api_url="https://api.example.com/v1/metrics",
            api_key="your-api-key",
        )

        # Use as emitter
        instrument(ctx, max_cost=1.0)  # Auto-detects METER_API_URL
        ```
    """

    def __init__(
        self,
        api_url: str,
        api_key: Optional[str] = None,
        batch_size: int = 50,
        flush_interval: float = 5.0,
        timeout: float = 10.0,
        max_retries: int = 3,
        max_queue_size: int = 10000,
        headers: Optional[dict[str, str]] = None,
        on_queue_overflow: Optional[QueueOverflowHandler] = None,
    ):
        self._api_url = api_url
        self._api_key = api_key
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._timeout = timeout
        self._max_retries = max_retries
        self._extra_headers = headers or {}
        self._on_queue_overflow = on_queue_overflow

        # Bounded queue to prevent memory issues
        self._queue: queue.Queue[MetricEvent] = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        # Tracking for observability
        self._dropped_count = 0
        self._sent_count = 0
        self._error_count = 0

        self._start_worker()

    def _start_worker(self) -> None:
        """Start the background worker thread."""
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        """Background worker that batches and sends metrics."""
        batch: list[MetricEvent] = []
        last_flush = time.time()

        while not self._stop_event.is_set():
            try:
                # Get event with timeout
                try:
                    event = self._queue.get(timeout=0.5)
                    batch.append(event)
                except queue.Empty:
                    pass

                # Check if we should flush
                should_flush = (
                    len(batch) >= self._batch_size
                    or (batch and time.time() - last_flush >= self._flush_interval)
                )

                if should_flush and batch:
                    self._send_batch(batch)
                    batch = []
                    last_flush = time.time()

            except Exception as e:
                logger.error(f"Error in HTTP transport worker: {e}")

        # Final flush on shutdown
        if batch:
            self._send_batch(batch)

    def _send_batch(self, batch: list[MetricEvent]) -> None:
        """Send a batch of metrics to the API."""
        try:
            import urllib.request
            import urllib.error

            # Convert events to JSON
            payload = {
                "metrics": [self._event_to_dict(e) for e in batch],
                "timestamp": time.time(),
            }

            data = json.dumps(payload).encode("utf-8")

            # Build headers
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "openai-meter/0.1.0",
                **self._extra_headers,
            }
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

            # Send with retries
            for attempt in range(self._max_retries):
                try:
                    req = urllib.request.Request(
                        self._api_url,
                        data=data,
                        headers=headers,
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                        if resp.status == 200:
                            self._sent_count += len(batch)
                            logger.debug(f"Sent {len(batch)} metrics to API (total: {self._sent_count})")
                            return
                        else:
                            logger.warning(f"API returned status {resp.status}")

                except urllib.error.HTTPError as e:
                    logger.warning(f"HTTP error {e.code} sending metrics (attempt {attempt + 1})")
                    if e.code >= 500:
                        # Server error - retry
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        # Client error - don't retry
                        logger.error(f"Client error sending metrics: {e}")
                        return

                except urllib.error.URLError as e:
                    logger.warning(f"URL error sending metrics (attempt {attempt + 1}): {e}")
                    time.sleep(2 ** attempt)
                    continue

            self._error_count += len(batch)
            logger.error(f"Failed to send {len(batch)} metrics after {self._max_retries} retries")

        except Exception as e:
            self._error_count += len(batch)
            logger.error(f"Error sending metrics batch: {e}")

    def _event_to_dict(self, event: MetricEvent) -> dict[str, Any]:
        """Convert MetricEvent to JSON-serializable dict."""
        # MetricEvent uses flat fields, so asdict works directly
        return asdict(event)

    def __call__(self, event: MetricEvent) -> None:
        """Add an event to the send queue."""
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self._dropped_count += 1
            logger.warning(
                f"Metric queue full, dropping event (total dropped: {self._dropped_count})"
            )
            if self._on_queue_overflow:
                try:
                    self._on_queue_overflow(self._dropped_count)
                except Exception as e:
                    logger.error(f"Error in queue overflow callback: {e}")

    @property
    def stats(self) -> dict[str, int]:
        """Get transport statistics for observability."""
        return {
            "sent": self._sent_count,
            "dropped": self._dropped_count,
            "errors": self._error_count,
            "queued": self._queue.qsize(),
        }

    def flush(self) -> None:
        """Manually trigger a flush (waits for current batch to send)."""
        # Put a marker and wait for queue to empty
        while not self._queue.empty():
            time.sleep(0.1)

    def close(self) -> None:
        """Stop the transport and flush remaining metrics."""
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=10.0)


def create_http_transport(
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    batch_size: int = 50,
    flush_interval: float = 5.0,
) -> HttpTransport:
    """
    Create an HTTP transport for sending metrics to an API.

    Args:
        api_url: API endpoint URL (or use METER_API_URL env var)
        api_key: API key (or use METER_API_KEY env var)
        batch_size: Events to batch before sending (or METER_BATCH_SIZE)
        flush_interval: Seconds between sends (or METER_FLUSH_INTERVAL)

    Returns:
        HttpTransport instance

    Raises:
        ValueError: If no API URL is provided or found in environment
    """
    url = api_url or os.environ.get("METER_API_URL")
    if not url:
        raise ValueError(
            "API URL required. Pass api_url or set METER_API_URL environment variable."
        )

    key = api_key or os.environ.get("METER_API_KEY")
    batch = int(os.environ.get("METER_BATCH_SIZE", batch_size))
    interval = float(os.environ.get("METER_FLUSH_INTERVAL", flush_interval))

    return HttpTransport(
        api_url=url,
        api_key=key,
        batch_size=batch,
        flush_interval=interval,
    )


# Global transport instance (created lazily)
_http_transport: Optional[HttpTransport] = None
_http_transport_init_attempted: bool = False


def _get_http_transport() -> Optional[HttpTransport]:
    """Get or create the HTTP transport from METER_API_URL."""
    global _http_transport, _http_transport_init_attempted

    if _http_transport_init_attempted:
        return _http_transport

    _http_transport_init_attempted = True
    api_url = os.environ.get("METER_API_URL")

    if not api_url:
        return None

    try:
        _http_transport = create_http_transport(api_url=api_url)
        logger.info(f"HTTP transport initialized: {api_url}")
        return _http_transport
    except Exception as e:
        logger.warning(f"Failed to initialize HTTP transport: {e}")
        return None
