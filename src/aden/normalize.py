"""
Usage normalization utilities.

Handles usage normalization across multiple LLM providers:
- OpenAI: Responses API (input_tokens/output_tokens) and Chat Completions (prompt_tokens/completion_tokens)
- Anthropic: Messages API (input_tokens/output_tokens with cache_read_input_tokens)
- Gemini: GenerateContent API (promptTokenCount/candidatesTokenCount)

All providers are normalized into a standard NormalizedUsage schema.
"""

from typing import Any, Literal

from .types import NormalizedUsage


def _to_dict(obj: Any) -> dict[str, Any]:
    """Convert an object to a dictionary for easier field access."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return {}


def normalize_openai_usage(usage: Any) -> NormalizedUsage | None:
    """
    Normalizes usage data from OpenAI API responses.

    Handles both API shapes:
    - Responses API: uses `input_tokens` / `output_tokens`
    - Chat Completions API: uses `prompt_tokens` / `completion_tokens`

    Args:
        usage: Raw usage object from OpenAI API response

    Returns:
        Normalized usage metrics, or None if no usage data provided
    """
    if usage is None:
        return None

    raw = _to_dict(usage)
    if not raw:
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


def normalize_anthropic_usage(usage: Any) -> NormalizedUsage | None:
    """
    Normalizes usage data from Anthropic API responses.

    Anthropic Messages API usage fields:
    - input_tokens: Input tokens consumed
    - output_tokens: Output tokens generated
    - cache_read_input_tokens: Tokens served from cache (optional)
    - cache_creation_input_tokens: Tokens used to create cache (optional)

    Args:
        usage: Raw usage object from Anthropic API response

    Returns:
        Normalized usage metrics, or None if no usage data provided
    """
    if usage is None:
        return None

    raw = _to_dict(usage)
    if not raw:
        return None

    input_tokens = raw.get("input_tokens", 0) or 0
    output_tokens = raw.get("output_tokens", 0) or 0

    return NormalizedUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cached_tokens=raw.get("cache_read_input_tokens", 0) or 0,
        reasoning_tokens=0,  # Anthropic doesn't have reasoning tokens yet
        accepted_prediction_tokens=0,
        rejected_prediction_tokens=0,
    )


def normalize_gemini_usage(usage_metadata: Any) -> NormalizedUsage | None:
    """
    Normalizes usage data from Google Gemini API responses.

    Gemini GenerateContent usage_metadata fields:
    - promptTokenCount: Input tokens
    - candidatesTokenCount: Output tokens
    - totalTokenCount: Total tokens
    - cachedContentTokenCount: Cached tokens (optional)

    Args:
        usage_metadata: Raw usage_metadata from Gemini API response

    Returns:
        Normalized usage metrics, or None if no usage data provided
    """
    if usage_metadata is None:
        return None

    raw = _to_dict(usage_metadata)

    # Try dict access first (for JSON/dict responses), then getattr (for protobuf objects)
    def get_field(camel_name: str, snake_name: str, default: int = 0) -> int:
        # Try dict access with both naming conventions
        if raw:
            val = raw.get(camel_name) or raw.get(snake_name)
            if val:
                return val
        # Fall back to getattr for protobuf objects
        val = getattr(usage_metadata, snake_name, None)
        if val:
            return val
        val = getattr(usage_metadata, camel_name, None)
        if val:
            return val
        return default

    input_tokens = get_field("promptTokenCount", "prompt_token_count", 0)
    output_tokens = get_field("candidatesTokenCount", "candidates_token_count", 0)
    total_tokens = get_field("totalTokenCount", "total_token_count", 0) or (input_tokens + output_tokens)
    cached_tokens = get_field("cachedContentTokenCount", "cached_content_token_count", 0)

    return NormalizedUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
        reasoning_tokens=0,  # Gemini doesn't expose reasoning tokens
        accepted_prediction_tokens=0,
        rejected_prediction_tokens=0,
    )


def normalize_gemini_grpc_usage(usage_metadata: Any) -> NormalizedUsage | None:
    """
    Normalizes usage data from Google Gemini gRPC API responses.

    The gRPC client (GenerativeServiceClient) returns usage_metadata with
    snake_case attribute names, accessed via getattr().

    Args:
        usage_metadata: Raw usage_metadata from Gemini gRPC response

    Returns:
        Normalized usage metrics, or None if no usage data provided
    """
    if usage_metadata is None:
        return None

    input_tokens = getattr(usage_metadata, 'prompt_token_count', 0) or 0
    output_tokens = getattr(usage_metadata, 'candidates_token_count', 0) or 0

    return NormalizedUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=getattr(usage_metadata, 'total_token_count', 0) or (input_tokens + output_tokens),
        cached_tokens=getattr(usage_metadata, 'cached_content_token_count', 0) or 0,
        reasoning_tokens=0,
        accepted_prediction_tokens=0,
        rejected_prediction_tokens=0,
    )


def normalize_usage(
    usage: Any,
    provider: Literal["openai", "anthropic", "gemini"] = "openai",
) -> NormalizedUsage | None:
    """
    Normalizes usage data from any supported LLM provider.

    Args:
        usage: Raw usage object from API response
        provider: The provider the usage came from (default: "openai")

    Returns:
        Normalized usage metrics, or None if no usage data provided

    Example:
        >>> # OpenAI
        >>> response = openai_client.chat.completions.create(...)
        >>> normalized = normalize_usage(response.usage, "openai")

        >>> # Anthropic
        >>> response = anthropic_client.messages.create(...)
        >>> normalized = normalize_usage(response.usage, "anthropic")

        >>> # Gemini
        >>> response = model.generate_content(...)
        >>> normalized = normalize_usage(response.usage_metadata, "gemini")
    """
    if provider == "anthropic":
        return normalize_anthropic_usage(usage)
    elif provider == "gemini":
        return normalize_gemini_usage(usage)
    else:
        return normalize_openai_usage(usage)


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
