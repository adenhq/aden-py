"""
Usage normalization utilities.

The OpenAI API returns usage in two different shapes depending on the endpoint:
- Responses API: uses `input_tokens` / `output_tokens`
- Chat Completions API: uses `prompt_tokens` / `completion_tokens`

This module handles both and normalizes into a standard schema.
"""

from typing import Any

from .types import NormalizedUsage


def normalize_usage(usage: Any) -> NormalizedUsage | None:
    """
    Normalizes usage data from both OpenAI API response shapes into a consistent format.

    Args:
        usage: Raw usage object from OpenAI API response

    Returns:
        Normalized usage metrics, or None if no usage data provided

    Example:
        >>> response = client.chat.completions.create(...)
        >>> normalized = normalize_usage(response.usage)
        >>> print(f"Input: {normalized.input_tokens}, Output: {normalized.output_tokens}")
    """
    if usage is None:
        return None

    # Handle dict-like objects
    if isinstance(usage, dict):
        raw = usage
    elif hasattr(usage, "__dict__"):
        # Handle Pydantic models or dataclasses
        raw = usage.__dict__ if not hasattr(usage, "model_dump") else usage.model_dump()
    elif hasattr(usage, "model_dump"):
        raw = usage.model_dump()
    else:
        return None

    # Check if this is Responses API shape (input_tokens/output_tokens)
    if "input_tokens" in raw or "output_tokens" in raw:
        input_tokens = raw.get("input_tokens", 0) or 0
        output_tokens = raw.get("output_tokens", 0) or 0

        # Extract nested details
        input_details = raw.get("input_tokens_details", {}) or {}
        output_details = raw.get("output_tokens_details", {}) or {}

        return NormalizedUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=raw.get("total_tokens") or (input_tokens + output_tokens),
            reasoning_tokens=output_details.get("reasoning_tokens", 0) or 0,
            cached_tokens=input_details.get("cached_tokens", 0) or 0,
            accepted_prediction_tokens=0,
            rejected_prediction_tokens=0,
        )

    # Chat Completions API shape (prompt_tokens/completion_tokens)
    prompt_tokens = raw.get("prompt_tokens", 0) or 0
    completion_tokens = raw.get("completion_tokens", 0) or 0

    # Extract nested details
    prompt_details = raw.get("prompt_tokens_details", {}) or {}
    completion_details = raw.get("completion_tokens_details", {}) or {}

    return NormalizedUsage(
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        total_tokens=raw.get("total_tokens") or (prompt_tokens + completion_tokens),
        reasoning_tokens=completion_details.get("reasoning_tokens", 0) or 0,
        cached_tokens=prompt_details.get("cached_tokens", 0) or 0,
        accepted_prediction_tokens=completion_details.get("accepted_prediction_tokens", 0) or 0,
        rejected_prediction_tokens=completion_details.get("rejected_prediction_tokens", 0) or 0,
    )


def empty_usage() -> NormalizedUsage:
    """Creates an empty/zero usage object."""
    return NormalizedUsage()


def merge_usage(a: NormalizedUsage, b: NormalizedUsage) -> NormalizedUsage:
    """
    Merges two usage objects (useful for accumulating streaming deltas).

    Args:
        a: First usage object
        b: Second usage object

    Returns:
        Combined usage with summed values
    """
    return NormalizedUsage(
        input_tokens=a.input_tokens + b.input_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
        total_tokens=a.total_tokens + b.total_tokens,
        reasoning_tokens=a.reasoning_tokens + b.reasoning_tokens,
        cached_tokens=a.cached_tokens + b.cached_tokens,
        accepted_prediction_tokens=a.accepted_prediction_tokens + b.accepted_prediction_tokens,
        rejected_prediction_tokens=a.rejected_prediction_tokens + b.rejected_prediction_tokens,
    )
