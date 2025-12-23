"""
TimescaleDB emitter for persisting metrics to a time-series database.

TimescaleDB is PostgreSQL optimized for time-series data, making it ideal
for storing LLM usage metrics with automatic partitioning and compression.
"""

import asyncio
import json
import logging
import re
from dataclasses import asdict
from datetime import datetime, timezone
from threading import Lock, Timer
from typing import Any

from .types import MetricEvent

logger = logging.getLogger(__name__)

# SQL for creating the metrics table and hypertable
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS llm_metrics (
    time TIMESTAMPTZ NOT NULL,
    trace_id TEXT NOT NULL,
    request_id TEXT,
    model TEXT NOT NULL,

    -- Token usage
    input_tokens INTEGER,
    output_tokens INTEGER,
    total_tokens INTEGER,
    cached_tokens INTEGER,
    reasoning_tokens INTEGER,

    -- Performance
    latency_ms DOUBLE PRECISION,
    stream BOOLEAN DEFAULT FALSE,

    -- Metadata
    service_tier TEXT,
    error TEXT,

    -- Tool calls (JSONB for flexibility)
    tool_calls JSONB,

    -- Custom metadata
    metadata JSONB,

    -- Rate limit info
    rate_limit_remaining_requests INTEGER,
    rate_limit_remaining_tokens INTEGER
);
"""

CREATE_HYPERTABLE_SQL = """
SELECT create_hypertable('llm_metrics', 'time', if_not_exists => TRUE);
"""

# Optional: Create useful indexes
CREATE_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_llm_metrics_model ON llm_metrics (model, time DESC);
CREATE INDEX IF NOT EXISTS idx_llm_metrics_trace ON llm_metrics (trace_id);
CREATE INDEX IF NOT EXISTS idx_llm_metrics_error ON llm_metrics (time DESC) WHERE error IS NOT NULL;
"""

INSERT_SQL = """
INSERT INTO llm_metrics (
    time, trace_id, request_id, model,
    input_tokens, output_tokens, total_tokens, cached_tokens, reasoning_tokens,
    latency_ms, stream, service_tier, error, tool_calls, metadata,
    rate_limit_remaining_requests, rate_limit_remaining_tokens
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17
)
"""


def _event_to_row(event: MetricEvent) -> tuple[Any, ...]:
    """Convert a MetricEvent to a database row tuple."""
    # Build tool_calls JSON from flat fields
    tool_calls_json = None
    if event.tool_names:
        # Convert comma-separated names to array format
        names = event.tool_names.split(",")
        tool_calls_json = json.dumps([{"name": name} for name in names])

    metadata_json = None
    if event.metadata:
        metadata_json = json.dumps(event.metadata)

    return (
        datetime.now(timezone.utc),  # time
        event.trace_id,
        event.request_id,
        event.model,
        event.input_tokens if event.input_tokens else None,
        event.output_tokens if event.output_tokens else None,
        event.total_tokens if event.total_tokens else None,
        event.cached_tokens if event.cached_tokens else None,
        event.reasoning_tokens if event.reasoning_tokens else None,
        event.latency_ms,
        event.stream,
        event.service_tier,
        event.error,
        tool_calls_json,
        metadata_json,
        event.rate_limit_remaining_requests,
        event.rate_limit_remaining_tokens,
    )


class TimescaleEmitter:
    """
    Async emitter that batches and writes metrics to TimescaleDB.

    Features:
    - Automatic table/hypertable creation
    - Connection pooling via asyncpg
    - Batched inserts for efficiency
    - Fire-and-forget or awaitable modes

    Example:
        ```python
        emitter = await TimescaleEmitter.create(
            dsn="postgresql://user:pass@localhost/metrics",
            batch_size=100,
            flush_interval=5.0,
        )

        metered = make_metered_openai(client, MeterOptions(
            emit_metric=emitter,
        ))

        # Later: clean shutdown
        await emitter.close()
        ```
    """

    def __init__(
        self,
        pool: Any,  # asyncpg.Pool
        batch_size: int = 100,
        flush_interval: float = 5.0,
    ):
        self._pool = pool
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._batch: list[MetricEvent] = []
        self._lock = Lock()
        self._flush_task: asyncio.Task[None] | None = None
        self._closed = False

    @classmethod
    async def create(
        cls,
        dsn: str,
        batch_size: int = 100,
        flush_interval: float = 5.0,
        min_pool_size: int = 2,
        max_pool_size: int = 10,
        create_table: bool = True,
    ) -> "TimescaleEmitter":
        """
        Create a TimescaleEmitter with connection pool.

        Args:
            dsn: PostgreSQL connection string (e.g., postgresql://user:pass@host/db)
            batch_size: Number of events to batch before flushing
            flush_interval: Seconds between automatic flushes
            min_pool_size: Minimum connection pool size
            max_pool_size: Maximum connection pool size
            create_table: Whether to auto-create table and hypertable

        Returns:
            Configured TimescaleEmitter instance
        """
        try:
            import asyncpg
        except ImportError:
            raise ImportError(
                "asyncpg is required for TimescaleDB support. "
                "Install with: pip install openai-meter[timescale]"
            )

        pool = await asyncpg.create_pool(
            dsn,
            min_size=min_pool_size,
            max_size=max_pool_size,
        )

        if create_table:
            async with pool.acquire() as conn:
                await conn.execute(CREATE_TABLE_SQL)
                try:
                    await conn.execute(CREATE_HYPERTABLE_SQL)
                except Exception as e:
                    # Hypertable might already exist or TimescaleDB not installed
                    logger.warning(f"Could not create hypertable (may already exist): {e}")
                try:
                    await conn.execute(CREATE_INDEXES_SQL)
                except Exception as e:
                    logger.warning(f"Could not create indexes: {e}")

        emitter = cls(pool, batch_size, flush_interval)
        emitter._start_flush_loop()
        return emitter

    def _start_flush_loop(self) -> None:
        """Start the background flush loop."""
        async def flush_loop() -> None:
            while not self._closed:
                await asyncio.sleep(self._flush_interval)
                await self.flush()

        try:
            loop = asyncio.get_running_loop()
            self._flush_task = loop.create_task(flush_loop())
        except RuntimeError:
            # No running loop, will flush on close or manual calls
            pass

    def __call__(self, event: MetricEvent) -> None:
        """Add an event to the batch (sync interface for compatibility)."""
        with self._lock:
            self._batch.append(event)
            if len(self._batch) >= self._batch_size:
                # Schedule async flush
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._do_flush())
                except RuntimeError:
                    # No running loop, batch will flush on interval or close
                    pass

    async def emit(self, event: MetricEvent) -> None:
        """Async emit interface."""
        with self._lock:
            self._batch.append(event)
            should_flush = len(self._batch) >= self._batch_size

        if should_flush:
            await self._do_flush()

    async def _do_flush(self) -> None:
        """Internal flush implementation."""
        with self._lock:
            if not self._batch:
                return
            to_flush = self._batch
            self._batch = []

        if not to_flush:
            return

        try:
            rows = [_event_to_row(event) for event in to_flush]
            async with self._pool.acquire() as conn:
                await conn.executemany(INSERT_SQL, rows)
            logger.debug(f"Flushed {len(rows)} metrics to TimescaleDB")
        except Exception as e:
            logger.error(f"Error flushing metrics to TimescaleDB: {e}")
            # Re-add failed events to batch for retry
            with self._lock:
                self._batch = to_flush + self._batch

    async def flush(self) -> None:
        """Manually flush pending events."""
        await self._do_flush()

    async def close(self) -> None:
        """Close the emitter and flush remaining events."""
        self._closed = True
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        await self.flush()
        await self._pool.close()


async def create_timescale_emitter(
    dsn: str,
    batch_size: int = 100,
    flush_interval: float = 5.0,
    create_table: bool = True,
) -> TimescaleEmitter:
    """
    Create a TimescaleDB emitter.

    Args:
        dsn: PostgreSQL/TimescaleDB connection string
        batch_size: Events to batch before flushing (default: 100)
        flush_interval: Seconds between auto-flushes (default: 5.0)
        create_table: Auto-create table and hypertable (default: True)

    Returns:
        TimescaleEmitter instance

    Example:
        ```python
        from openai_meter.timescale import create_timescale_emitter

        emitter = await create_timescale_emitter(
            dsn="postgresql://localhost/metrics",
        )

        metered = make_metered_openai(client, MeterOptions(
            emit_metric=emitter,
        ))
        ```
    """
    return await TimescaleEmitter.create(
        dsn=dsn,
        batch_size=batch_size,
        flush_interval=flush_interval,
        create_table=create_table,
    )


# Convenience: sync wrapper for simpler setup
class SyncTimescaleEmitter:
    """
    Sync-friendly wrapper around TimescaleEmitter.

    Uses a background thread to handle async operations.
    """

    def __init__(
        self,
        dsn: str,
        batch_size: int = 100,
        flush_interval: float = 5.0,
    ):
        self._dsn = dsn
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._batch: list[MetricEvent] = []
        self._lock = Lock()
        self._timer: Timer | None = None
        self._conn: Any = None  # psycopg2 connection
        self._start_timer()

    def _get_connection(self) -> Any:
        """Get or create psycopg2 connection."""
        if self._conn is None:
            try:
                import psycopg2
            except ImportError:
                raise ImportError(
                    "psycopg2 is required for sync TimescaleDB support. "
                    "Install with: pip install psycopg2-binary"
                )
            self._conn = psycopg2.connect(self._dsn)
            self._ensure_table()
        return self._conn

    def _ensure_table(self) -> None:
        """Create table if needed."""
        conn = self._conn
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
            try:
                cur.execute(CREATE_HYPERTABLE_SQL)
            except Exception:
                pass  # May already exist
            conn.commit()

    def _start_timer(self) -> None:
        """Start periodic flush timer."""
        self._timer = Timer(self._flush_interval, self._timer_flush)
        self._timer.daemon = True
        self._timer.start()

    def _timer_flush(self) -> None:
        """Timer callback."""
        self.flush()
        self._start_timer()

    def __call__(self, event: MetricEvent) -> None:
        """Add event to batch."""
        with self._lock:
            self._batch.append(event)
            if len(self._batch) >= self._batch_size:
                self._do_flush()

    def _do_flush(self) -> None:
        """Internal flush."""
        if not self._batch:
            return
        to_flush = self._batch
        self._batch = []

        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                # Convert INSERT_SQL from asyncpg ($1, $2, ...) to psycopg2 (%s, %s, ...) format
                # Use regex to replace all $N placeholders with %s
                insert_sql = re.sub(r'\$\d+', '%s', INSERT_SQL)

                rows = [_event_to_row(event) for event in to_flush]
                cur.executemany(insert_sql, rows)
                conn.commit()
            logger.debug(f"Flushed {len(rows)} metrics to TimescaleDB")
        except Exception as e:
            logger.error(f"Error flushing to TimescaleDB: {e}")
            with self._lock:
                self._batch = to_flush + self._batch

    def flush(self) -> None:
        """Manual flush."""
        with self._lock:
            self._do_flush()

    def close(self) -> None:
        """Close emitter."""
        if self._timer:
            self._timer.cancel()
        self.flush()
        if self._conn:
            self._conn.close()
