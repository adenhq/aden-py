"""
OpenAI Meter - SDK metering for AI Cost ERP

Usage tracking, budget enforcement, and cost control for OpenAI API calls.
Designed for integration with LiveKit voice agents and other OpenAI SDK consumers.
"""

from .types import (
    NormalizedUsage,
    RequestMetadata,
    MetricEvent,
    RateLimitInfo,
    ToolCallMetric,
    MetricEmitter,
    MeterOptions,
    BeforeRequestContext,
    BeforeRequestResult,
    BudgetConfig,
    BudgetExceededInfo,
    RequestCancelledError,
    BudgetExceededError,
)
from .meter import make_metered_openai, is_metered
from .normalize import normalize_usage, empty_usage, merge_usage
from .emitters import (
    create_console_emitter,
    create_batch_emitter,
    create_multi_emitter,
    create_filtered_emitter,
    create_transform_emitter,
    create_noop_emitter,
    create_memory_emitter,
)
from .file_logger import MetricFileLogger, create_file_emitter, DEFAULT_LOG_DIR

__version__ = "0.1.0"

__all__ = [
    # Types
    "NormalizedUsage",
    "RequestMetadata",
    "MetricEvent",
    "RateLimitInfo",
    "ToolCallMetric",
    "MetricEmitter",
    "MeterOptions",
    "BeforeRequestContext",
    "BeforeRequestResult",
    "BudgetConfig",
    "BudgetExceededInfo",
    # Errors
    "RequestCancelledError",
    "BudgetExceededError",
    # Core metering
    "make_metered_openai",
    "is_metered",
    # Usage normalization
    "normalize_usage",
    "empty_usage",
    "merge_usage",
    # Emitters
    "create_console_emitter",
    "create_batch_emitter",
    "create_multi_emitter",
    "create_filtered_emitter",
    "create_transform_emitter",
    "create_noop_emitter",
    "create_memory_emitter",
    # File logging
    "MetricFileLogger",
    "create_file_emitter",
    "DEFAULT_LOG_DIR",
]
