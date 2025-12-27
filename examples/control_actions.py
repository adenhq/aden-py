"""
Control Actions Example

Demonstrates all control actions in the Aden SDK:
1. allow   - Request proceeds normally
2. block   - Request is rejected (budget exceeded)
3. throttle - Request is delayed before proceeding
4. degrade  - Request uses a cheaper model
5. alert    - Request proceeds but triggers notification

Prerequisites:
1. Set ADEN_API_KEY in environment (or .env file)
2. Set ADEN_API_URL to your control server (or use default)
3. Set OPENAI_API_KEY for making actual LLM calls

Run: python examples/control_actions.py
"""

import asyncio
import os
import sys
import time
from typing import Any

# Add parent directory to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use shell environment

import httpx

from openai import AsyncOpenAI

from aden import (
    instrument_async,
    uninstrument_async,
    create_console_emitter,
    MeterOptions,
    AlertEvent,
    RequestCancelledError,
)

USER_ID = "demo_user_control_actions"
AGENT_NAME = "enterprise"  # Agent name for multi-budget demo
API_KEY = os.environ.get("ADEN_API_KEY", "")
SERVER_URL = os.environ.get("ADEN_API_URL", "http://localhost:8888")
BUDGET_LIMIT = 0.0003  # Very tight limit to trigger blocking quickly

# Track alerts received
alerts_received: list[dict[str, Any]] = []


async def setup_policy() -> None:
    """Set up the control policy on the server.

    Updates the global budget limit to a very small amount to demonstrate
    control actions being triggered.
    """
    print("=" * 60)
    print("Setting up control policy...")
    print("=" * 60 + "\n")

    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {API_KEY}"}
        json_headers = {**headers, "Content-Type": "application/json"}

        # Get current policy to find global budget
        try:
            policy_res = await client.get(
                f"{SERVER_URL}/v1/control/policy",
                headers=headers,
            )
            if policy_res.status_code != 200:
                print(f"  Failed to get policy: HTTP {policy_res.status_code}")
                print(f"  Response: {policy_res.text[:200] if policy_res.text else '(empty)'}")
                print(f"  Make sure the control server is running at {SERVER_URL}")
                return
            policy = policy_res.json()
        except Exception as e:
            print(f"  Failed to connect to control server: {e}")
            print(f"  Make sure the control server is running at {SERVER_URL}")
            return

        # Find global budget and update its limit
        global_budget = None
        for budget in policy.get("budgets", []):
            if budget.get("type") == "global":
                global_budget = budget
                break

        if global_budget:
            budget_id = global_budget.get("id")
            current_spend = global_budget.get("spent", 0)
            print(f"  Found global budget: {budget_id}")
            print(f"  Current spend: ${current_spend:.6f}")

            # Set limit to current_spend + $0.002 to trigger thresholds quickly
            # This means we're already at ~90%+ usage
            new_limit = current_spend + BUDGET_LIMIT
            print(f"  Setting limit to: ${new_limit:.6f} (spend + ${BUDGET_LIMIT})")

            # Update the budget in the policy via PUT
            updated_budget = {**global_budget, "limit": new_limit, "limitAction": "kill"}
            updated_budgets = [updated_budget] + [
                b for b in policy.get("budgets", []) if b.get("type") != "global"
            ]

            update_res = await client.put(
                f"{SERVER_URL}/v1/control/policies/default",
                headers=json_headers,
                json={"budgets": updated_budgets},
            )
            if update_res.status_code == 200:
                print(f"  Updated global budget limit to ${new_limit:.6f}")
            else:
                print(f"  Failed to update budget: {update_res.status_code}")

            # Calculate starting usage percentage
            usage_pct = (current_spend / new_limit * 100) if new_limit > 0 else 0
            print(f"  Starting at {usage_pct:.1f}% budget usage\n")
        else:
            print("  No global budget found in policy\n")

        # Get and display the updated policy
        try:
            policy_res = await client.get(
                f"{SERVER_URL}/v1/control/policy",
                headers=headers,
            )
            if policy_res.status_code != 200:
                print(f"  Failed to fetch updated policy: HTTP {policy_res.status_code}")
                return
            policy = policy_res.json()
        except Exception as e:
            print(f"  Failed to fetch updated policy: {e}")
            return
        print("Updated policy budgets:")
        import json
        for budget in policy.get("budgets", []):
            if budget.get("type") == "global":
                print(json.dumps(budget, indent=2))


async def get_budget_status(debug: bool = False) -> dict[str, float]:
    """Get current budget status from policy.

    Gets the global budget from the policy, which is what the SDK uses for enforcement.
    """
    async with httpx.AsyncClient() as client:
        try:
            res = await client.get(
                f"{SERVER_URL}/v1/control/policy",
                headers={"Authorization": f"Bearer {API_KEY}"},
            )
            if res.status_code != 200:
                if debug:
                    print(f"   [DEBUG] Failed to get policy: HTTP {res.status_code}")
                return {"spend": 0, "limit": BUDGET_LIMIT, "percent": 0}
            data = res.json()
        except Exception as e:
            if debug:
                print(f"   [DEBUG] Failed to get policy: {e}")
            return {"spend": 0, "limit": BUDGET_LIMIT, "percent": 0}

        # Find the global budget from the policy
        budgets = data.get("budgets", [])
        global_budget = None
        for budget in budgets:
            if budget.get("type") == "global":
                global_budget = budget
                break

        if global_budget:
            spend = global_budget.get("spent", 0)
            limit = global_budget.get("limit", 0.07)
            if debug:
                print(f"   [DEBUG] Global budget: ${spend:.6f} / ${limit} ({(spend/limit*100) if limit > 0 else 0:.1f}%)")
            return {
                "spend": spend,
                "limit": limit,
                "percent": (spend / limit) * 100 if limit > 0 else 0,
            }

        if debug:
            print(f"   [DEBUG] No global budget found in policy")
        return {"spend": 0, "limit": BUDGET_LIMIT, "percent": 0}


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
    openai = AsyncOpenAI()

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
        status = await get_budget_status(debug=(i == 0))  # Debug first request

        print(f"\n[Request {i + 1}/{len(prompts)}] \"{prompt}\"")
        print(f"   Budget: ${status['spend']:.6f} / ${status['limit']} ({status['percent']:.1f}%)")

        start_time = time.time()

        try:
            # Pass agent metadata to trigger multi-budget matching
            # This will match both: global budget + enterprise agent budget
            response = await openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                extra_body={"metadata": {"agent": AGENT_NAME}},
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
    print("  - allow: Requests allowed when under budget")
    print("  - block: Requests blocked when budget exceeded")
    print("  - multi-budget: Both global + agent budgets validated")
    print("")
    print("Multi-Budget Validation:")
    print(f"  - Requests used agent='{AGENT_NAME}' metadata")
    print("  - This matches BOTH global budget AND enterprise agent budget")
    print("  - Most restrictive decision is returned (if any budget blocks, request is blocked)")

    await uninstrument_async()
    print("\nDemo complete!\n")


if __name__ == "__main__":
    asyncio.run(main())
