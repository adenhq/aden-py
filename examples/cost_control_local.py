"""
Cost Control Example (Local Mode)

Demonstrates the Aden SDK cost control logic without requiring a server.
Shows how budget limits, throttling, and model degradation work.

Run: python examples/cost_control_local.py
"""

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Literal

# Add parent directory to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use shell environment


# =============================================================================
# Simulated Policy Types
# =============================================================================

@dataclass
class BudgetRule:
    context_id: str
    limit_usd: float
    current_spend_usd: float = 0.0
    action_on_exceed: Literal["block", "degrade", "throttle", "allow"] = "block"
    degrade_to_model: str | None = None


@dataclass
class ThrottleRule:
    context_id: str | None = None
    provider: str | None = None
    requests_per_minute: int | None = None
    delay_ms: int | None = None


@dataclass
class BlockRule:
    reason: str
    context_id: str | None = None
    provider: str | None = None
    model_pattern: str | None = None


@dataclass
class DegradeRule:
    from_model: str
    to_model: str
    trigger: Literal["budget_threshold", "rate_limit", "always"]
    threshold_percent: float | None = None
    context_id: str | None = None


@dataclass
class ControlPolicy:
    version: str
    updated_at: str
    budgets: list[BudgetRule] = field(default_factory=list)
    throttles: list[ThrottleRule] = field(default_factory=list)
    blocks: list[BlockRule] = field(default_factory=list)
    degradations: list[DegradeRule] = field(default_factory=list)


@dataclass
class ControlRequest:
    provider: str
    model: str
    context_id: str | None = None
    estimated_cost: float | None = None


@dataclass
class ControlDecision:
    action: Literal["allow", "block", "throttle", "degrade"]
    reason: str | None = None
    degrade_to_model: str | None = None
    throttle_delay_ms: int | None = None


# =============================================================================
# Simulated Policy Engine (what the server does)
# =============================================================================

class LocalPolicyEngine:
    """Local policy engine that simulates server-side policy evaluation."""

    def __init__(self, policy: ControlPolicy) -> None:
        self.policy = policy
        self.budget_spend: dict[str, float] = {}
        self.request_counts: dict[str, dict[str, Any]] = {}

    def get_decision(self, request: ControlRequest) -> ControlDecision:
        """Evaluate policy and return a decision."""
        import re

        # 1. Check block rules
        for block in self.policy.blocks:
            if self._matches_block_rule(request, block):
                return ControlDecision(action="block", reason=block.reason)

        # 2. Check budget limits
        if request.context_id:
            budget = next(
                (b for b in self.policy.budgets if b.context_id == request.context_id),
                None,
            )
            if budget:
                current_spend = self.budget_spend.get(request.context_id, 0)
                projected_spend = current_spend + (request.estimated_cost or 0)

                if projected_spend > budget.limit_usd:
                    if budget.action_on_exceed == "degrade" and budget.degrade_to_model:
                        return ControlDecision(
                            action="degrade",
                            reason=f"Budget exceeded: ${projected_spend:.4f} > ${budget.limit_usd}",
                            degrade_to_model=budget.degrade_to_model,
                        )
                    return ControlDecision(
                        action=budget.action_on_exceed,
                        reason=f"Budget exceeded: ${projected_spend:.4f} > ${budget.limit_usd}",
                    )

                # Check degradation threshold
                for degrade in self.policy.degradations:
                    if (
                        degrade.from_model == request.model
                        and degrade.trigger == "budget_threshold"
                        and degrade.threshold_percent
                    ):
                        usage_percent = (current_spend / budget.limit_usd) * 100
                        if usage_percent >= degrade.threshold_percent:
                            return ControlDecision(
                                action="degrade",
                                reason=f"Budget at {usage_percent:.1f}% (threshold: {degrade.threshold_percent}%)",
                                degrade_to_model=degrade.to_model,
                            )

        # 3. Check throttle rules
        for throttle in self.policy.throttles:
            if not throttle.context_id or throttle.context_id == request.context_id:
                if throttle.requests_per_minute:
                    key = f"{throttle.context_id or 'global'}:{throttle.provider or 'all'}"
                    rate_info = self._check_rate_limit(key, throttle.requests_per_minute)
                    if rate_info["exceeded"]:
                        return ControlDecision(
                            action="throttle",
                            reason=f"Rate limit: {rate_info['count']}/{throttle.requests_per_minute}/min",
                            throttle_delay_ms=throttle.delay_ms or 1000,
                        )

        return ControlDecision(action="allow")

    def record_spend(self, context_id: str, amount: float) -> None:
        """Record spend for a context."""
        current = self.budget_spend.get(context_id, 0)
        self.budget_spend[context_id] = current + amount

    def get_spend(self, context_id: str) -> float:
        """Get current spend for a context."""
        return self.budget_spend.get(context_id, 0)

    def _matches_block_rule(self, request: ControlRequest, block: BlockRule) -> bool:
        """Check if request matches a block rule."""
        import re

        if block.context_id and block.context_id != request.context_id:
            return False
        if block.provider and block.provider != request.provider:
            return False
        if block.model_pattern:
            pattern = "^" + block.model_pattern.replace("*", ".*") + "$"
            if not re.match(pattern, request.model):
                return False
        return True

    def _check_rate_limit(self, key: str, limit: int) -> dict[str, Any]:
        """Check rate limit for a key."""
        now = time.time() * 1000  # ms
        window_ms = 60000

        info = self.request_counts.get(key)
        if not info or now - info["window_start"] > window_ms:
            info = {"count": 0, "window_start": now}

        info["count"] += 1
        self.request_counts[key] = info

        return {"exceeded": info["count"] > limit, "count": info["count"]}


# =============================================================================
# Demo
# =============================================================================

def run_local_demo() -> None:
    """Run local policy demo without actual API calls."""
    from datetime import datetime

    print("=" * 60)
    print("Aden SDK - Cost Control Demo (Local Mode)")
    print("=" * 60)

    # Define a cost control policy
    policy = ControlPolicy(
        version="demo-1",
        updated_at=datetime.now().isoformat(),
        budgets=[
            BudgetRule(
                context_id="user_free_tier",
                limit_usd=0.01,  # $0.01 for free tier
                action_on_exceed="block",
            ),
            BudgetRule(
                context_id="user_pro_tier",
                limit_usd=1.0,  # $1.00 for pro tier
                action_on_exceed="degrade",
                degrade_to_model="gpt-4o-mini",
            ),
        ],
        degradations=[
            DegradeRule(
                from_model="gpt-4o",
                to_model="gpt-4o-mini",
                trigger="budget_threshold",
                threshold_percent=80,
                context_id="user_pro_tier",
            ),
        ],
        throttles=[
            ThrottleRule(
                context_id="user_free_tier",
                requests_per_minute=5,
            ),
        ],
        blocks=[
            BlockRule(
                context_id="user_banned",
                reason="Account suspended for policy violation",
            ),
            BlockRule(
                model_pattern="gpt-4o",
                context_id="user_free_tier",
                reason="GPT-4o not available on free tier",
            ),
        ],
    )

    print("\nPolicy Configuration:")
    print(f"  Budgets: {len(policy.budgets)}")
    print(f"  Degradations: {len(policy.degradations)}")
    print(f"  Throttles: {len(policy.throttles)}")
    print(f"  Blocks: {len(policy.blocks)}")

    # Create policy engine
    engine = LocalPolicyEngine(policy)

    # Test scenarios
    print("\n" + "=" * 60)
    print("Testing Policy Decisions")
    print("=" * 60)

    test_cases = [
        {
            "name": "Free tier user - GPT-4o-mini request",
            "request": ControlRequest(
                context_id="user_free_tier",
                provider="openai",
                model="gpt-4o-mini",
                estimated_cost=0.001,
            ),
        },
        {
            "name": "Free tier user - GPT-4o (blocked model)",
            "request": ControlRequest(
                context_id="user_free_tier",
                provider="openai",
                model="gpt-4o",
                estimated_cost=0.01,
            ),
        },
        {
            "name": "Banned user - any request",
            "request": ControlRequest(
                context_id="user_banned",
                provider="openai",
                model="gpt-4o-mini",
                estimated_cost=0.001,
            ),
        },
        {
            "name": "Pro tier user - GPT-4o at 50% budget",
            "request": ControlRequest(
                context_id="user_pro_tier",
                provider="openai",
                model="gpt-4o",
                estimated_cost=0.05,
            ),
            "simulate_spend": 0.5,  # Already spent $0.50 of $1.00
        },
        {
            "name": "Pro tier user - GPT-4o at 85% budget (should degrade)",
            "request": ControlRequest(
                context_id="user_pro_tier",
                provider="openai",
                model="gpt-4o",
                estimated_cost=0.05,
            ),
            "simulate_spend": 0.85,  # Already spent $0.85 of $1.00
        },
        {
            "name": "Free tier user - budget exceeded",
            "request": ControlRequest(
                context_id="user_free_tier",
                provider="openai",
                model="gpt-4o-mini",
                estimated_cost=0.005,
            ),
            "simulate_spend": 0.008,  # Already spent $0.008 of $0.01
        },
    ]

    action_emoji = {
        "allow": "OK",
        "block": "XX",
        "throttle": "..",
        "degrade": ">>",
    }

    for test_case in test_cases:
        request: ControlRequest = test_case["request"]
        print(f"\n  {test_case['name']}")
        print(f"   Request: {request.model} (est. ${request.estimated_cost})")

        # Simulate prior spend if specified
        if "simulate_spend" in test_case and request.context_id:
            spend = test_case["simulate_spend"]
            current = engine.get_spend(request.context_id)
            engine.record_spend(request.context_id, spend - current)
            print(f"   Prior spend: ${spend:.4f}")

        decision = engine.get_decision(request)

        emoji = action_emoji.get(decision.action, "?")
        print(f"   Decision: [{emoji}] {decision.action.upper()}")
        if decision.reason:
            print(f"   Reason: {decision.reason}")
        if decision.degrade_to_model:
            print(f"   Use instead: {decision.degrade_to_model}")
        if decision.throttle_delay_ms:
            print(f"   Wait: {decision.throttle_delay_ms}ms")

    # Show rate limiting
    print("\n" + "=" * 60)
    print("Testing Rate Limiting (5 req/min for free tier)")
    print("=" * 60)

    # Create fresh engine for rate limit test
    rate_limit_engine = LocalPolicyEngine(policy)

    for i in range(1, 8):
        decision = rate_limit_engine.get_decision(
            ControlRequest(
                context_id="user_free_tier",
                provider="openai",
                model="gpt-4o-mini",
                estimated_cost=0.0001,
            )
        )

        status = "allowed" if decision.action == "allow" else f"throttled ({decision.reason})"
        emoji = "OK" if decision.action == "allow" else ".."
        print(f"   Request {i}: [{emoji}] {status}")

    print("\nDemo complete!\n")


# =============================================================================
# Live Demo with OpenAI (requires OPENAI_API_KEY)
# =============================================================================

def run_live_demo() -> None:
    """Run live demo with actual OpenAI API calls."""
    print("\n" + "=" * 60)
    print("Live Demo with OpenAI")
    print("=" * 60)

    if not os.environ.get("OPENAI_API_KEY"):
        print("\nOPENAI_API_KEY not set - skipping live demo\n")
        return

    from openai import OpenAI
    from aden import (
        instrument,
        uninstrument,
        create_console_emitter,
        MeterOptions,
    )

    # Instrument with console output
    instrument(
        MeterOptions(
            emit_metric=create_console_emitter(pretty=True),
        )
    )

    client = OpenAI()

    print("\nMaking a real request...\n")

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is 2+2? Reply in one word."}],
            max_tokens=10,
        )

        print(f"\nResponse: {response.choices[0].message.content}")
        print(f"Tokens used: {response.usage.total_tokens if response.usage else 'N/A'}")
    except Exception as e:
        print(f"Request failed: {e}")

    uninstrument()


# Run demos
def main() -> None:
    """Run all demos."""
    run_local_demo()
    run_live_demo()


if __name__ == "__main__":
    main()
