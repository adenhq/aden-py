"""
Control Actions Example

Demonstrates all control actions in the Aden SDK:
1. allow   - Request proceeds normally
2. block   - Request is rejected (budget exceeded)
3. throttle - Request is delayed before proceeding
4. degrade  - Request uses a cheaper model
5. alert    - Request proceeds but triggers notification

Prerequisites:
1. Set ADEN_API_KEY in environment
2. Set ADEN_API_URL to your control server (or use default)
3. Set OPENAI_API_KEY for making actual LLM calls

Run: python examples/control_actions.py
"""

import asyncio
import os
import sys
import time
from datetime import datetime
from typing import Any

import httpx

# Add parent directory to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from openai import OpenAI

from aden import (
    instrument_async,
    uninstrument_async,
    create_console_emitter,
    MeterOptions,
    AlertEvent,
    RequestCancelledError,
)

USER_ID = "demo_user_control_actions"
API_KEY = os.environ.get("ADEN_API_KEY", "")
SERVER_URL = os.environ.get("ADEN_API_URL", "http://localhost:8888")
BUDGET_LIMIT = 0.002  # Must match the budget rule

# Track alerts received
alerts_received: list[dict[str, Any]] = []


async def setup_policy() -> None:
    """Set up the control policy on the server."""
    print("=" * 60)
    print("Setting up control policy...")
    print("=" * 60 + "\n")

    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {API_KEY}"}
        json_headers = {**headers, "Content-Type": "application/json"}

        # Clear existing policy
        await client.delete(f"{SERVER_URL}/v1/control/policy", headers=headers)

        # Reset budget
        await client.post(
            f"{SERVER_URL}/v1/control/budget/{USER_ID}/reset",
            headers=headers,
        )

        # 1. Budget with $0.002 limit - small enough to trigger all thresholds
        await client.post(
            f"{SERVER_URL}/v1/control/policy/budgets",
            headers=json_headers,
            json={
                "context_id": USER_ID,
                "limit_usd": BUDGET_LIMIT,
                "action_on_exceed": "block",
            },
        )
        print("  Budget: $0.002 limit, block on exceed")

        # 2. Throttle rule: 3 requests per minute
        await client.post(
            f"{SERVER_URL}/v1/control/policy/throttles",
            headers=json_headers,
            json={
                "context_id": USER_ID,
                "requests_per_minute": 3,
                "delay_ms": 2000,
            },
        )
        print("  Throttle: 3 req/min, 2s delay when exceeded")

        # 3. Degradation rule: gpt-4o -> gpt-4o-mini at 50% budget
        await client.post(
            f"{SERVER_URL}/v1/control/policy/degradations",
            headers=json_headers,
            json={
                "from_model": "gpt-4o",
                "to_model": "gpt-4o-mini",
                "trigger": "budget_threshold",
                "threshold_percent": 50,
                "context_id": USER_ID,
            },
        )
        print("  Degradation: gpt-4o -> gpt-4o-mini at 50% budget")

        # 4. Alert rule: Warn when any gpt-4* model is used
        await client.post(
            f"{SERVER_URL}/v1/control/policy/alerts",
            headers=json_headers,
            json={
                "model_pattern": "gpt-4*",
                "trigger": "model_usage",
                "level": "warning",
                "message": "Expensive model (gpt-4*) is being used",
            },
        )
        print("  Alert: Warning when gpt-4* model is used")

        # 5. Alert rule: Critical when budget > 80%
        await client.post(
            f"{SERVER_URL}/v1/control/policy/alerts",
            headers=json_headers,
            json={
                "context_id": USER_ID,
                "trigger": "budget_threshold",
                "threshold_percent": 80,
                "level": "critical",
                "message": "Budget nearly exhausted (>80%)",
            },
        )
        print("  Alert: Critical when budget > 80%\n")

        # Get and display the full policy
        policy_res = await client.get(
            f"{SERVER_URL}/v1/control/policy",
            headers=headers,
        )
        policy = policy_res.json()
        print("Full policy:")
        import json
        print(json.dumps(policy, indent=2))


async def get_budget_status() -> dict[str, float]:
    """Get current budget status from server."""
    async with httpx.AsyncClient() as client:
        res = await client.get(
            f"{SERVER_URL}/v1/control/budget/{USER_ID}",
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
        data = res.json()
        spend = data.get("current_spend_usd", 0)
        return {
            "spend": spend,
            "limit": BUDGET_LIMIT,
            "percent": (spend / BUDGET_LIMIT) * 100,
        }


def on_alert(alert: AlertEvent) -> None:
    """Alert callback - invoked when an alert is triggered."""
    alerts_received.append({
        "level": alert.level,
        "message": alert.message,
        "timestamp": alert.timestamp,
    })
    print(f"\n   [ALERT CALLBACK] [{alert.level.upper()}] {alert.message}")
    print(f"   Provider: {alert.provider}, Model: {alert.model}\n")


async def main() -> None:
    """Run the control actions demo."""
    print("\n" + "=" * 60)
    print("Aden SDK - Control Actions Demo")
    print("=" * 60 + "\n")

    if not API_KEY:
        print("ADEN_API_KEY required")
        sys.exit(1)

    await setup_policy()

    # Instrument with alert handler
    print("\n" + "=" * 60)
    print("Initializing Aden instrumentation...")
    print("=" * 60 + "\n")

    await instrument_async(
        MeterOptions(
            api_key=API_KEY,
            server_url=SERVER_URL,
            emit_metric=create_console_emitter(pretty=True),
            get_context_id=lambda: USER_ID,
            on_alert=on_alert,
        )
    )

    # Create client AFTER instrumentation
    openai = OpenAI()

    print("\n" + "=" * 60)
    print("Making LLM requests to demonstrate control actions...")
    print("=" * 60)

    prompts = [
        "What is 2+2?",           # Request 1: ALLOW + ALERT (gpt-4o)
        "Say hello",              # Request 2: ALLOW + ALERT (gpt-4o)
        "What color is the sky?", # Request 3: ALLOW + ALERT + likely DEGRADE (>50% budget)
        "Count to 3",             # Request 4: THROTTLE (>3/min) + DEGRADE + possibly BLOCK
        "Name a fruit",           # Request 5: THROTTLE + likely BLOCKED (>100% budget)
        "Say bye",                # Request 6: THROTTLE + BLOCKED
        "Last request",           # Request 7: THROTTLE + BLOCKED
    ]

    for i, prompt in enumerate(prompts):
        status = await get_budget_status()

        print(f"\n[Request {i + 1}/{len(prompts)}] \"{prompt}\"")
        print(f"   Budget: ${status['spend']:.6f} / ${status['limit']} ({status['percent']:.1f}%)")

        start_time = time.time()

        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
            )

            duration_ms = int((time.time() - start_time) * 1000)
            content = response.choices[0].message.content
            actual_model = response.model

            # Check if model was degraded
            was_degraded = "mini" in actual_model

            print(f"   Response ({duration_ms}ms): \"{content}\"")
            print(f"   Model: {actual_model}{' (DEGRADED from gpt-4o)' if was_degraded else ''}, Tokens: {response.usage.total_tokens if response.usage else 'N/A'}")

            # Check for throttle (if request took > 1.5s, it was likely throttled)
            if duration_ms > 1500:
                print(f"   (Request was THROTTLED - {duration_ms}ms latency)")

        except RequestCancelledError as e:
            duration_ms = int((time.time() - start_time) * 1000)
            print(f"   BLOCKED ({duration_ms}ms): {e}")
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            error_msg = str(e)
            if "cancelled" in error_msg.lower() or "budget" in error_msg.lower():
                print(f"   BLOCKED ({duration_ms}ms): {error_msg}")
            elif "rate limit" in error_msg.lower():
                print(f"   THROTTLED: {error_msg}")
            else:
                print(f"   ERROR: {error_msg}")

        # Brief delay between requests
        await asyncio.sleep(0.3)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    final_status = await get_budget_status()
    print(f"\nFinal Budget Status:")
    print(f"  User: {USER_ID}")
    print(f"  Spent: ${final_status['spend']:.6f}")
    print(f"  Limit: ${final_status['limit']}")
    print(f"  Usage: {final_status['percent']:.1f}%")

    print(f"\nAlerts Received: {len(alerts_received)}")
    for alert in alerts_received:
        print(f"  [{alert['level'].upper()}] {alert['message']}")

    print("\nControl Actions Demonstrated:")
    print("  - allow: Requests 1-3 proceeded normally")
    print("  - alert: Triggered for gpt-4* model usage")
    print("  - throttle: Applied after 3 requests/min exceeded")
    print("  - degrade: gpt-4o -> gpt-4o-mini after 50% budget")
    print("  - block: Requests blocked after budget exceeded")

    await uninstrument_async()
    print("\nDemo complete!\n")


if __name__ == "__main__":
    asyncio.run(main())
