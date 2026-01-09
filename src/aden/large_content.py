"""
Large content storage utilities.

Handles sending large content payloads to the control server.
"""

import logging
from typing import Any, Callable

logger = logging.getLogger("aden")

# Global reference to the store function (set by instrument.py)
_store_func: Callable[[list[dict[str, Any]]], None] | None = None


def set_store_func(func: Callable[[list[dict[str, Any]]], None] | None) -> None:
    """Set the function to use for storing large content."""
    global _store_func
    _store_func = func


def store_large_content(payloads: list[dict[str, Any]]) -> None:
    """
    Store large content payloads on the control server.

    Called by instrumentation when content exceeds max_content_bytes.
    The content is stored separately and referenced via ContentReference.

    Args:
        payloads: List of content items to store, each containing:
            - content_id: Unique ID for the content
            - content_hash: SHA-256 hash of the content
            - content: The actual content string
    """
    if not payloads:
        return

    if _store_func is None:
        logger.debug("[aden] No control agent - skipping large content storage")
        return

    try:
        _store_func(payloads)
        logger.debug(f"[aden] Stored {len(payloads)} large content items")
    except Exception as e:
        logger.warning(f"[aden] Failed to store large content: {e}")
