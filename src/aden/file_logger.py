"""
File-based metric logging for local storage and analysis.

Writes raw metric data to JSONL files organized by date and session,
enabling offline analysis, debugging, and compliance auditing.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .types import MetricEvent

logger = logging.getLogger(__name__)

# Default log directory (can be overridden via environment or parameter)
DEFAULT_LOG_DIR = os.environ.get('METER_LOG_DIR', './meter_logs')


class MetricFileLogger:
    """
    Writes raw metric data to local JSONL files for analysis.

    Files are organized by date and session:
        meter_logs/
            2024-01-15/
                session_abc123.jsonl
                session_def456.jsonl
            2024-01-16/
                ...

    Each line in the JSONL file is a complete JSON object representing
    one metric event (LLM request, TTS synthesis, STT transcription).

    Example:
        ```python
        from openai_meter.file_logger import MetricFileLogger

        logger = MetricFileLogger("./my_logs")
        logger.write_llm_event(
            session_id="session_123",
            input_tokens=100,
            output_tokens=50,
            model="gpt-4o-mini",
        )
        ```
    """

    def __init__(self, log_dir: str = DEFAULT_LOG_DIR):
        """
        Initialize the file logger.

        Args:
            log_dir: Base directory for log files
        """
        self.log_dir = Path(log_dir)
        self._ensure_dir_exists()
        logger.info(f"Metric file logger initialized: {self.log_dir.absolute()}")

    def _ensure_dir_exists(self) -> None:
        """Create the log directory if it doesn't exist."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _get_session_file(self, session_id: str) -> Path:
        """Get the log file path for a session."""
        date_str = datetime.now().strftime('%Y-%m-%d')
        date_dir = self.log_dir / date_str
        date_dir.mkdir(exist_ok=True)

        # Sanitize session_id for filename
        safe_session_id = "".join(
            c if c.isalnum() or c in '-_' else '_' for c in session_id
        )
        return date_dir / f"session_{safe_session_id}.jsonl"

    def write_event(
        self,
        session_id: str,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        """
        Write a metric event to the session's log file.

        Args:
            session_id: Session identifier
            event_type: Type of event (llm, tts, stt, session_start, session_end)
            data: Event data to log
        """
        file_path = self._get_session_file(session_id)

        event = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "event_type": event_type,
            **data,
        }

        try:
            with open(file_path, 'a') as f:
                f.write(json.dumps(event) + '\n')
        except Exception as e:
            logger.error(f"Failed to write metric to file: {e}")

    def write_metric_event(self, event: MetricEvent) -> None:
        """
        Write a MetricEvent to the log file.

        Args:
            event: The MetricEvent to log
        """
        session_id = "unknown"
        if event.metadata and "session_id" in event.metadata:
            session_id = event.metadata["session_id"]

        data: dict[str, Any] = {
            "trace_id": event.trace_id,
            "model": event.model,
            "stream": event.stream,
            "latency_ms": event.latency_ms,
        }

        if event.total_tokens > 0:
            data["usage"] = {
                "input_tokens": event.input_tokens,
                "output_tokens": event.output_tokens,
                "total_tokens": event.total_tokens,
                "reasoning_tokens": event.reasoning_tokens,
                "cached_tokens": event.cached_tokens,
            }

        if event.error:
            data["error"] = event.error

        if event.metadata:
            data["metadata"] = event.metadata

        self.write_event(session_id, "metric", data)

    def write_llm_event(
        self,
        session_id: str,
        input_tokens: int,
        output_tokens: int,
        model: str,
        latency_ms: float = 0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Write an LLM metric event."""
        self.write_event(session_id, "llm", {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "latency_ms": latency_ms,
            **(metadata or {}),
        })

    def write_tts_event(
        self,
        session_id: str,
        characters: int,
        model: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Write a TTS metric event."""
        self.write_event(session_id, "tts", {
            "model": model,
            "characters": characters,
            **(metadata or {}),
        })

    def write_stt_event(
        self,
        session_id: str,
        audio_seconds: float,
        model: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Write an STT metric event."""
        self.write_event(session_id, "stt", {
            "model": model,
            "audio_seconds": audio_seconds,
            **(metadata or {}),
        })

    def write_session_start(
        self,
        session_id: str,
        room_name: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Write session start event."""
        self.write_event(session_id, "session_start", {
            "room_name": room_name,
            **(metadata or {}),
        })

    def write_session_end(
        self,
        session_id: str,
        summary: dict[str, Any],
    ) -> None:
        """Write session end event with final summary."""
        self.write_event(session_id, "session_end", {
            "summary": summary,
        })


def create_file_emitter(
    log_dir: str = DEFAULT_LOG_DIR,
) -> "MetricFileLogger":
    """
    Create a file-based metric emitter.

    Args:
        log_dir: Directory for log files

    Returns:
        MetricFileLogger instance that can be used as an emitter
    """
    return MetricFileLogger(log_dir)
