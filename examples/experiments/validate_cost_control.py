#!/usr/bin/env python3
"""
EXP-1.x: Cost Control Validation

Validates:
- EXP-1.1: Block Request
- EXP-1.2: Throttle Request
- EXP-1.3: Model Degradation

Usage:
    python examples/experiments/validate_cost_control.py
"""

import sys
import time

sys.path.insert(0, "src")

from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

from aden import (
    BeforeRequestAction,
    BeforeRequestContext,
    BeforeRequestResult,
    MeterOptions,
    MetricEvent,
    RequestCancelledError,
)
from aden.instrument_openai import instrument_openai, uninstrument_openai

# Collect metrics for verification
collected_metrics: list[MetricEvent] = []


def metric_collector(event: MetricEvent):
    collected_metrics.append(event)
    print(f"  [metric] model={event.model}, tokens={event.total_tokens}")


def reset():
    global collected_metrics
    collected_metrics = []
    uninstrument_openai()


# =============================================================================
# EXP-1.1: Block Request
# =============================================================================
def test_block_request():
    """Validate that requests can be blocked before reaching the LLM."""
    print("\n" + "=" * 60)
    print("EXP-1.1: Block Request")
    print("=" * 60)

    reset()

    block_reason = "Budget exceeded - $0.00 remaining"

    def blocking_hook(params: dict, context: BeforeRequestContext) -> BeforeRequestResult:
        print(f"  [hook] Blocking request for model={params.get('model')}")
        return BeforeRequestResult.cancel(block_reason)

    instrument_openai(
        MeterOptions(
            emit_metric=metric_collector,
            before_request=blocking_hook,
        )
    )

    client = OpenAI()

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say hello"}],
        )
        print("  [FAIL] Request was NOT blocked!")
        return False
    except RequestCancelledError as e:
        print(f"  [PASS] RequestCancelledError raised")
        print(f"  [PASS] Reason: {e.reason}")
        print(f"  [PASS] Context present: {e.context is not None}")

        # Verify no metrics (no LLM call made)
        if len(collected_metrics) == 0:
            print("  [PASS] No metrics emitted (request blocked before LLM)")
        else:
            print(f"  [FAIL] Unexpected metrics: {len(collected_metrics)}")
            return False

        return True
    except Exception as e:
        print(f"  [FAIL] Wrong exception type: {type(e).__name__}: {e}")
        return False


# =============================================================================
# EXP-1.2: Throttle Request
# =============================================================================
def test_throttle_request():
    """Validate that throttling adds predictable delay."""
    print("\n" + "=" * 60)
    print("EXP-1.2: Throttle Request")
    print("=" * 60)

    reset()

    throttle_delay_ms = 500

    def throttling_hook(params: dict, context: BeforeRequestContext) -> BeforeRequestResult:
        print(f"  [hook] Throttling request by {throttle_delay_ms}ms")
        return BeforeRequestResult.throttle(throttle_delay_ms)

    instrument_openai(
        MeterOptions(
            emit_metric=metric_collector,
            before_request=throttling_hook,
        )
    )

    client = OpenAI()

    t0 = time.time()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say 'ok'"}],
        max_tokens=5,
    )
    elapsed_ms = (time.time() - t0) * 1000

    print(f"  [info] Total elapsed: {elapsed_ms:.0f}ms")
    print(f"  [info] Response: {response.choices[0].message.content}")

    # Verify throttle was applied (elapsed should be >= throttle_delay)
    if elapsed_ms >= throttle_delay_ms:
        print(f"  [PASS] Elapsed >= throttle delay ({throttle_delay_ms}ms)")
    else:
        print(f"  [FAIL] Elapsed < throttle delay")
        return False

    # Verify metric was emitted
    if len(collected_metrics) == 1:
        print("  [PASS] Metric emitted after throttle")
    else:
        print(f"  [FAIL] Expected 1 metric, got {len(collected_metrics)}")
        return False

    return True


# =============================================================================
# EXP-1.3: Model Degradation
# =============================================================================
def test_model_degradation():
    """Validate that model degradation switches to cheaper model."""
    print("\n" + "=" * 60)
    print("EXP-1.3: Model Degradation")
    print("=" * 60)

    reset()

    original_model = "gpt-4o"
    degraded_model = "gpt-4o-mini"

    def degradation_hook(params: dict, context: BeforeRequestContext) -> BeforeRequestResult:
        if params.get("model") == original_model:
            print(f"  [hook] Degrading {original_model} -> {degraded_model}")
            return BeforeRequestResult.degrade(degraded_model)
        return BeforeRequestResult.proceed()

    instrument_openai(
        MeterOptions(
            emit_metric=metric_collector,
            before_request=degradation_hook,
        )
    )

    client = OpenAI()

    response = client.chat.completions.create(
        model=original_model,  # Request gpt-4o
        messages=[{"role": "user", "content": "Say 'degraded'"}],
        max_tokens=5,
    )

    print(f"  [info] Response: {response.choices[0].message.content}")

    # Verify the metric shows the degraded model
    if len(collected_metrics) == 1:
        metric = collected_metrics[0]
        if metric.model == degraded_model:
            print(f"  [PASS] MetricEvent.model = {degraded_model} (degraded)")
        else:
            print(f"  [FAIL] MetricEvent.model = {metric.model}, expected {degraded_model}")
            return False
    else:
        print(f"  [FAIL] Expected 1 metric, got {len(collected_metrics)}")
        return False

    return True


# =============================================================================
# Main
# =============================================================================
def main():
    print("\nCost Control Validation (EXP-1.x)")
    print("Using real OpenAI API")

    results = {
        "EXP-1.1 Block": test_block_request(),
        "EXP-1.2 Throttle": test_throttle_request(),
        "EXP-1.3 Degrade": test_model_degradation(),
    }

    # Cleanup
    uninstrument_openai()

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
