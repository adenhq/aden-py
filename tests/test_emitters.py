"""Tests for metric emitters."""

import pytest

from openai_meter.emitters import (
    create_console_emitter,
    create_filtered_emitter,
    create_memory_emitter,
    create_noop_emitter,
)
from openai_meter.types import MetricEvent, NormalizedUsage


def create_test_event(**overrides) -> MetricEvent:
    """Create a test MetricEvent with optional overrides."""
    defaults = {
        "trace_id": "test-trace-123",
        "model": "gpt-4o-mini",
        "stream": False,
        "request_id": "req-123",
        "latency_ms": 100.0,
        "usage": NormalizedUsage(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
        ),
    }
    defaults.update(overrides)
    return MetricEvent(**defaults)


class TestMemoryEmitter:
    """Tests for memory emitter."""

    def test_collects_events(self):
        emitter = create_memory_emitter()

        event1 = create_test_event(trace_id="1")
        event2 = create_test_event(trace_id="2")

        emitter(event1)
        emitter(event2)

        assert len(emitter.events) == 2
        assert emitter.events[0].trace_id == "1"
        assert emitter.events[1].trace_id == "2"

    def test_clear_removes_events(self):
        emitter = create_memory_emitter()
        emitter(create_test_event())
        emitter(create_test_event())

        emitter.clear()

        assert len(emitter.events) == 0


class TestFilteredEmitter:
    """Tests for filtered emitter."""

    def test_passes_matching_events(self):
        memory = create_memory_emitter()
        filtered = create_filtered_emitter(memory, lambda e: e.model == "gpt-4")

        filtered(create_test_event(model="gpt-4"))
        filtered(create_test_event(model="gpt-3.5-turbo"))
        filtered(create_test_event(model="gpt-4"))

        assert len(memory.events) == 2

    def test_filters_errors_only(self):
        memory = create_memory_emitter()
        filtered = create_filtered_emitter(memory, lambda e: e.error is not None)

        filtered(create_test_event())
        filtered(create_test_event(error="API error"))
        filtered(create_test_event())

        assert len(memory.events) == 1
        assert memory.events[0].error == "API error"


class TestNoopEmitter:
    """Tests for noop emitter."""

    def test_does_nothing(self):
        emitter = create_noop_emitter()
        # Should not raise
        emitter(create_test_event())
        emitter(create_test_event())


class TestConsoleEmitter:
    """Tests for console emitter."""

    def test_creates_emitter(self):
        emitter = create_console_emitter()
        assert callable(emitter)

    def test_warn_level_filters_success(self, capsys):
        emitter = create_console_emitter(level="warn")

        # Success event should not print
        emitter(create_test_event())
        captured = capsys.readouterr()
        assert captured.out == ""

        # Error event should print
        emitter(create_test_event(error="Test error"))
        captured = capsys.readouterr()
        assert "Test error" in captured.out or "X" in captured.out
