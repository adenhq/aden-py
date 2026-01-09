"""
Content storage utilities for Layer 0 content capture.

Handles content hashing, truncation, and creating ContentReference objects
for large content that needs to be stored on the control server.
"""

import hashlib
import json
import re
from typing import Any
from uuid import uuid4

from .types import ContentCaptureOptions, ContentReference


def compute_content_hash(content: str | bytes) -> str:
    """Compute SHA-256 hash of content.

    Args:
        content: String or bytes to hash

    Returns:
        Hexadecimal SHA-256 hash string
    """
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def get_content_bytes(content: str | dict[str, Any] | list[Any]) -> bytes:
    """Convert content to bytes for size calculation and hashing.

    Args:
        content: String, dict, or list content

    Returns:
        UTF-8 encoded bytes
    """
    if isinstance(content, str):
        return content.encode("utf-8")
    # For dicts/lists, serialize to JSON
    return json.dumps(content, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def should_store_separately(
    content: str | dict[str, Any] | list[Any],
    options: ContentCaptureOptions,
) -> bool:
    """Determine if content should be stored separately on the server.

    Args:
        content: The content to check
        options: Content capture configuration

    Returns:
        True if content exceeds max_content_bytes threshold
    """
    content_bytes = get_content_bytes(content)
    return len(content_bytes) > options.max_content_bytes


def create_truncated_preview(
    content_bytes: bytes,
    max_preview_bytes: int,
) -> str:
    """Create a truncated preview of content.

    Args:
        content_bytes: Full content as bytes
        max_preview_bytes: Maximum bytes for preview

    Returns:
        Truncated string with '...[truncated]' marker if needed
    """
    if len(content_bytes) <= max_preview_bytes:
        return content_bytes.decode("utf-8", errors="replace")

    # Truncate at byte boundary, being careful not to cut mid-character
    truncated = content_bytes[:max_preview_bytes]
    # Decode with replace to handle partial characters
    preview = truncated.decode("utf-8", errors="replace")
    return preview + "...[truncated]"


def create_content_reference(
    content: str | dict[str, Any] | list[Any],
    options: ContentCaptureOptions,
) -> ContentReference:
    """Create a ContentReference for large content.

    Args:
        content: The large content to reference
        options: Content capture configuration

    Returns:
        ContentReference with ID, hash, size, and preview
    """
    content_bytes = get_content_bytes(content)

    # Create truncated preview (25% of max_content_bytes)
    preview_bytes = options.max_content_bytes // 4
    preview = create_truncated_preview(content_bytes, preview_bytes)

    return ContentReference(
        content_id=str(uuid4()),
        content_hash=compute_content_hash(content_bytes),
        byte_size=len(content_bytes),
        truncated_preview=preview,
    )


def apply_redaction(content: str, patterns: list[str]) -> str:
    """Apply redaction patterns to content.

    Args:
        content: String content to redact
        patterns: List of regex patterns to match

    Returns:
        Content with matched patterns replaced by [REDACTED]
    """
    result = content
    for pattern in patterns:
        try:
            result = re.sub(pattern, "[REDACTED]", result)
        except re.error:
            # Skip invalid regex patterns
            pass
    return result


def serialize_content(content: str | dict[str, Any] | list[Any]) -> str:
    """Serialize content to a string for storage.

    Args:
        content: String, dict, or list content

    Returns:
        String representation
    """
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, indent=None, separators=(",", ":"))


def process_content(
    content: str | dict[str, Any] | list[Any] | None,
    options: ContentCaptureOptions,
) -> tuple[str | dict[str, Any] | list[Any] | ContentReference | None, dict[str, Any] | None]:
    """Process content for capture, handling truncation and large content.

    This is the main function for content processing. It:
    1. Returns None if content is None
    2. Applies redaction patterns (for strings)
    3. Checks if content exceeds max_content_bytes
    4. If small: returns content directly
    5. If large: returns ContentReference + payload for server storage

    Args:
        content: The content to process (string, dict, list, or None)
        options: Content capture configuration

    Returns:
        Tuple of:
        - processed_content: The content to store in MetricEvent
          (original if small, ContentReference if large)
        - large_content_payload: If not None, dict to send to control server
          with keys: content_id, content_hash, content, byte_size
    """
    if content is None:
        return None, None

    # Apply redaction for strings
    if isinstance(content, str) and options.redact_patterns:
        content = apply_redaction(content, options.redact_patterns)

    # Check if content should be stored separately
    if should_store_separately(content, options):
        ref = create_content_reference(content, options)
        # Return reference and the full content for server storage
        payload = {
            "content_id": ref.content_id,
            "content_hash": ref.content_hash,
            "content": serialize_content(content),
            "byte_size": ref.byte_size,
        }
        return ref, payload

    # Content is small enough to store directly
    return content, None


def process_string_content(
    content: str | None,
    options: ContentCaptureOptions,
) -> tuple[str | ContentReference | None, dict[str, Any] | None]:
    """Process string content for capture.

    Convenience wrapper around process_content for string-only content.

    Args:
        content: String content or None
        options: Content capture configuration

    Returns:
        Tuple of (processed_content, large_content_payload)
    """
    result, payload = process_content(content, options)
    # Type narrowing: if input was str, output is str | ContentReference
    return result, payload  # type: ignore
