"""Tests for usage normalization."""

import pytest

from openai_meter.normalize import empty_usage, merge_usage, normalize_usage
from openai_meter.types import NormalizedUsage


class TestNormalizeUsage:
    """Tests for normalize_usage function."""

    def test_returns_none_for_none_input(self):
        assert normalize_usage(None) is None

    def test_normalizes_responses_api_shape(self):
        """Test normalization of Responses API usage format."""
        raw = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "input_tokens_details": {"cached_tokens": 20},
            "output_tokens_details": {"reasoning_tokens": 10},
        }

        result = normalize_usage(raw)

        assert result is not None
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.total_tokens == 150
        assert result.cached_tokens == 20
        assert result.reasoning_tokens == 10

    def test_normalizes_chat_completions_api_shape(self):
        """Test normalization of Chat Completions API usage format."""
        raw = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_tokens_details": {"cached_tokens": 20},
            "completion_tokens_details": {
                "reasoning_tokens": 10,
                "accepted_prediction_tokens": 5,
                "rejected_prediction_tokens": 2,
            },
        }

        result = normalize_usage(raw)

        assert result is not None
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.total_tokens == 150
        assert result.cached_tokens == 20
        assert result.reasoning_tokens == 10
        assert result.accepted_prediction_tokens == 5
        assert result.rejected_prediction_tokens == 2

    def test_handles_missing_details(self):
        """Test that missing detail fields default to 0."""
        raw = {"input_tokens": 100, "output_tokens": 50}

        result = normalize_usage(raw)

        assert result is not None
        assert result.reasoning_tokens == 0
        assert result.cached_tokens == 0
        assert result.accepted_prediction_tokens == 0
        assert result.rejected_prediction_tokens == 0

    def test_calculates_total_if_missing(self):
        """Test that total_tokens is calculated if not provided."""
        raw = {"input_tokens": 100, "output_tokens": 50}

        result = normalize_usage(raw)

        assert result is not None
        assert result.total_tokens == 150


class TestEmptyUsage:
    """Tests for empty_usage function."""

    def test_returns_zero_values(self):
        result = empty_usage()

        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.total_tokens == 0
        assert result.reasoning_tokens == 0
        assert result.cached_tokens == 0


class TestMergeUsage:
    """Tests for merge_usage function."""

    def test_sums_all_fields(self):
        a = NormalizedUsage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            reasoning_tokens=10,
            cached_tokens=20,
            accepted_prediction_tokens=5,
            rejected_prediction_tokens=2,
        )
        b = NormalizedUsage(
            input_tokens=200,
            output_tokens=100,
            total_tokens=300,
            reasoning_tokens=20,
            cached_tokens=40,
            accepted_prediction_tokens=10,
            rejected_prediction_tokens=4,
        )

        result = merge_usage(a, b)

        assert result.input_tokens == 300
        assert result.output_tokens == 150
        assert result.total_tokens == 450
        assert result.reasoning_tokens == 30
        assert result.cached_tokens == 60
        assert result.accepted_prediction_tokens == 15
        assert result.rejected_prediction_tokens == 6
