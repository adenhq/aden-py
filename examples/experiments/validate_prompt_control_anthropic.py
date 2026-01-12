#!/usr/bin/env python3
"""
EXP-2.x: Prompt Control Validation - Anthropic

Validates modify_params hook functionality for Anthropic SDK:
- EXP-2.1: Prompt Injection (sync + async)
- EXP-2.4: Context Compaction
- EXP-2.6: Prompt Diff (via ModifyParamsResult)
- EXP-3.1: Tools Injection

Usage:
    python examples/experiments/validate_prompt_control_anthropic.py
"""

import asyncio
import json
import sys
from dataclasses import dataclass, field

sys.path.insert(0, "src")

from dotenv import load_dotenv

load_dotenv()

from anthropic import Anthropic, AsyncAnthropic

from aden import (
    BeforeRequestContext,
    MeterOptions,
    MetricEvent,
    ModifyParamsResult,
)
from aden.instrument_anthropic import instrument_anthropic, uninstrument_anthropic


@dataclass
class TestResult:
    name: str
    passed: bool
    checks: list[tuple[str, bool, str]] = field(default_factory=list)
    error: str | None = None

    def add_check(self, name: str, passed: bool, detail: str = ""):
        self.checks.append((name, passed, detail))
        if not passed:
            self.passed = False

    def print_report(self):
        status = "PASS" if self.passed else "FAIL"
        print(f"\n{'='*70}")
        print(f"{self.name}: {status}")
        print("="*70)

        for check_name, check_passed, detail in self.checks:
            icon = "+" if check_passed else "x"
            print(f"  {icon} {check_name}")
            if detail:
                for line in detail.split("\n"):
                    print(f"      {line}")

        if self.error:
            print(f"\n  ERROR: {self.error}")


class HookCapture:
    def __init__(self):
        self.calls: list[dict] = []
        self.metrics: list[MetricEvent] = []

    def reset(self):
        self.calls = []
        self.metrics = []

    def capture_metric(self, event: MetricEvent):
        self.metrics.append(event)


capture = HookCapture()


def reset_instrumentation():
    capture.reset()
    uninstrument_anthropic()


# =============================================================================
# EXP-2.1: Prompt Injection (Sync)
# =============================================================================
def test_prompt_injection_sync() -> TestResult:
    """Test sync prompt injection with prepend for Anthropic."""
    result = TestResult("EXP-2.1a: Prompt Injection (sync, Anthropic)", True)
    reset_instrumentation()

    injected_system = "INJECTED_SYSTEM_PROMPT_ANTHROPIC"

    def inject_system(params: dict, context: BeforeRequestContext) -> dict:
        original_system = params.get("system", "")
        original_messages = params.get("messages", [])

        capture.calls.append({
            "type": "inject_system",
            "original_system": original_system,
            "original_message_count": len(original_messages),
        })

        # Anthropic uses 'system' param separately from messages
        new_system = f"{injected_system}\n\n{original_system}" if original_system else injected_system

        capture.calls[-1]["modified_system"] = new_system
        return {**params, "system": new_system}

    try:
        instrument_anthropic(MeterOptions(
            emit_metric=capture.capture_metric,
            modify_params=inject_system,
        ))

        client = Anthropic()
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            messages=[{"role": "user", "content": "Say 'test'"}],
            max_tokens=10,
        )

        # Check 1: Hook was called
        result.add_check(
            "Hook called",
            len(capture.calls) == 1,
            f"Expected 1 call, got {len(capture.calls)}"
        )

        if capture.calls:
            call = capture.calls[0]

            # Check 2: System was injected
            result.add_check(
                "System prompt injected",
                injected_system in call.get("modified_system", ""),
                f"Modified system: {call.get('modified_system', '')[:50]}"
            )

        # Check 3: Response received
        response_text = response.content[0].text if response.content else None
        result.add_check(
            "Response received",
            response_text is not None,
            f"Response: {response_text}"
        )

        # Check 4: Metric emitted
        result.add_check(
            "Metric emitted",
            len(capture.metrics) == 1,
            f"Metrics: {len(capture.metrics)}, provider: {capture.metrics[0].provider if capture.metrics else 'N/A'}"
        )

    except Exception as e:
        result.error = str(e)
        result.passed = False

    return result


# =============================================================================
# EXP-2.1: Prompt Injection (Async)
# =============================================================================
async def test_prompt_injection_async() -> TestResult:
    """Test async prompt injection for Anthropic."""
    result = TestResult("EXP-2.1b: Prompt Injection (async, Anthropic)", True)
    reset_instrumentation()

    injected_instruction = "ASYNC_INSTRUCTION_ANTHROPIC"

    async def inject_async(params: dict, context: BeforeRequestContext) -> dict:
        messages = list(params.get("messages", []))

        capture.calls.append({
            "type": "inject_async",
            "original_count": len(messages),
        })

        # Add instruction as assistant prefill
        messages.append({"role": "assistant", "content": f"[{injected_instruction}]"})

        capture.calls[-1]["modified_count"] = len(messages)
        return {**params, "messages": messages}

    try:
        instrument_anthropic(MeterOptions(
            emit_metric=capture.capture_metric,
            modify_params=inject_async,
        ))

        client = AsyncAnthropic()
        response = await client.messages.create(
            model="claude-3-haiku-20240307",
            messages=[{"role": "user", "content": "Continue"}],
            max_tokens=20,
        )

        # Check 1: Hook called
        result.add_check(
            "Hook called (async)",
            len(capture.calls) == 1,
            f"Calls: {len(capture.calls)}"
        )

        if capture.calls:
            call = capture.calls[0]

            # Check 2: Message added
            result.add_check(
                "Message added",
                call["modified_count"] == call["original_count"] + 1,
                f"{call['original_count']} -> {call['modified_count']}"
            )

        # Check 3: Response received
        response_text = response.content[0].text if response.content else None
        result.add_check(
            "Async response received",
            response_text is not None,
            f"Response: {response_text[:50] if response_text else 'N/A'}"
        )

    except Exception as e:
        result.error = str(e)
        result.passed = False

    return result


# =============================================================================
# EXP-2.4: Context Compaction
# =============================================================================
def test_context_compaction() -> TestResult:
    """Test context compaction for Anthropic."""
    result = TestResult("EXP-2.4: Context Compaction (Anthropic)", True)
    reset_instrumentation()

    max_messages = 4

    def compact_context(params: dict, context: BeforeRequestContext) -> dict:
        messages = params.get("messages", [])
        original_count = len(messages)

        capture.calls.append({
            "type": "compaction",
            "original_count": original_count,
            "threshold": max_messages,
        })

        if len(messages) > max_messages:
            # Keep last N messages
            compacted = messages[-max_messages:]
            capture.calls[-1]["compacted_count"] = len(compacted)
            capture.calls[-1]["removed_count"] = original_count - len(compacted)
            return {**params, "messages": compacted}

        capture.calls[-1]["compacted_count"] = original_count
        capture.calls[-1]["removed_count"] = 0
        return params

    try:
        instrument_anthropic(MeterOptions(
            emit_metric=capture.capture_metric,
            modify_params=compact_context,
        ))

        client = Anthropic()

        # Build conversation (alternating user/assistant for Anthropic)
        long_conversation = []
        for i in range(5):
            long_conversation.append({"role": "user", "content": f"Message {i}"})
            long_conversation.append({"role": "assistant", "content": f"Response {i}"})
        long_conversation.append({"role": "user", "content": "What was message 0 about?"})

        response = client.messages.create(
            model="claude-3-haiku-20240307",
            messages=long_conversation,
            max_tokens=30,
        )

        if capture.calls:
            call = capture.calls[0]

            # Check 1: Original count captured
            result.add_check(
                "Original count captured",
                call["original_count"] == len(long_conversation),
                f"Expected {len(long_conversation)}, got {call['original_count']}"
            )

            # Check 2: Compacted
            result.add_check(
                "Compacted to threshold",
                call["compacted_count"] == max_messages,
                f"Expected {max_messages}, got {call['compacted_count']}"
            )

            # Check 3: Messages removed
            expected_removed = len(long_conversation) - max_messages
            result.add_check(
                "Messages removed",
                call["removed_count"] == expected_removed,
                f"Removed {call['removed_count']} (expected {expected_removed})"
            )

        # Check 4: Response works
        response_text = response.content[0].text if response.content else None
        result.add_check(
            "Response after compaction",
            response_text is not None,
            f"Response: {response_text[:50] if response_text else 'N/A'}"
        )

    except Exception as e:
        result.error = str(e)
        result.passed = False

    return result


# =============================================================================
# EXP-2.6: Prompt Diff (ModifyParamsResult)
# =============================================================================
def test_prompt_diff() -> TestResult:
    """Test ModifyParamsResult for diff tracking with Anthropic."""
    result = TestResult("EXP-2.6: Prompt Diff (Anthropic)", True)
    reset_instrumentation()

    def tracked_modification(params: dict, context: BeforeRequestContext) -> ModifyParamsResult:
        import copy
        original = copy.deepcopy(params)

        # Add system prompt
        modified = {**params, "system": "Be concise.", "temperature": 0.7}

        diff = {
            "params_added": [k for k in modified if k not in original],
            "params_modified": [k for k in modified if k in original and modified[k] != original[k]],
        }

        capture.calls.append({
            "type": "modify_params_result",
            "has_original_params": True,
            "diff": diff,
        })

        return ModifyParamsResult(params=modified, original_params=original)

    try:
        instrument_anthropic(MeterOptions(
            emit_metric=capture.capture_metric,
            modify_params=tracked_modification,
        ))

        client = Anthropic()
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=10,
        )

        if capture.calls:
            call = capture.calls[0]

            # Check 1: ModifyParamsResult works
            result.add_check(
                "ModifyParamsResult used",
                call["has_original_params"],
                "original_params captured"
            )

            # Check 2: Diff shows additions
            diff = call.get("diff", {})
            result.add_check(
                "Params added detected",
                "system" in diff.get("params_added", []) or "temperature" in diff.get("params_added", []),
                f"Added: {diff.get('params_added', [])}"
            )

            # Check 3: Show full diff
            diff_str = json.dumps(diff, indent=2)
            result.add_check(
                "Full diff available",
                True,
                f"Diff:\n{diff_str}"
            )

        # Check 4: Response works
        response_text = response.content[0].text if response.content else None
        result.add_check(
            "Response received",
            response_text is not None,
            f"Response: {response_text}"
        )

    except Exception as e:
        result.error = str(e)
        result.passed = False

    return result


# =============================================================================
# EXP-3.1: Tools Injection
# =============================================================================
def test_tools_injection() -> TestResult:
    """Test tools injection for Anthropic."""
    result = TestResult("EXP-3.1: Tools Injection (Anthropic)", True)
    reset_instrumentation()

    audit_tool = {
        "name": "audit_log",
        "description": "Log an audit event",
        "input_schema": {
            "type": "object",
            "properties": {
                "event": {"type": "string", "description": "Event to log"},
            },
            "required": ["event"],
        },
    }

    def inject_tools(params: dict, context: BeforeRequestContext) -> dict:
        original_tools = params.get("tools") or []

        capture.calls.append({
            "type": "tools_injection",
            "original_count": len(original_tools),
            "original_names": [t.get("name") for t in original_tools],
        })

        tools = list(original_tools)
        tools.append(audit_tool)

        capture.calls[-1]["final_count"] = len(tools)
        capture.calls[-1]["final_names"] = [t.get("name") for t in tools]

        return {**params, "tools": tools}

    try:
        instrument_anthropic(MeterOptions(
            emit_metric=capture.capture_metric,
            modify_params=inject_tools,
        ))

        client = Anthropic()

        original_tools = [
            {
                "name": "get_weather",
                "description": "Get weather for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        ]

        response = client.messages.create(
            model="claude-3-haiku-20240307",
            messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
            tools=original_tools,
            max_tokens=100,
        )

        if capture.calls:
            call = capture.calls[0]

            # Check 1: Original tools captured
            result.add_check(
                "Original tools captured",
                call["original_count"] == 1,
                f"Original: {call['original_names']}"
            )

            # Check 2: Tool injected
            result.add_check(
                "Tool injected",
                call["final_count"] == 2,
                f"Final: {call['final_names']}"
            )

            # Check 3: Correct tool
            result.add_check(
                "audit_log injected",
                "audit_log" in call["final_names"],
                f"Injected audit_log"
            )

        # Check 4: Response works
        has_content = len(response.content) > 0
        result.add_check(
            "Response received",
            has_content,
            f"Content blocks: {len(response.content)}"
        )

    except Exception as e:
        result.error = str(e)
        result.passed = False

    return result


# =============================================================================
# EXP-2.5: Tool Use Instruction
# =============================================================================
def test_tool_use_instruction() -> TestResult:
    """Test tool instruction injection for Anthropic."""
    result = TestResult("EXP-2.5: Tool Use Instruction (Anthropic)", True)
    reset_instrumentation()

    def inject_tool_instruction(params: dict, context: BeforeRequestContext) -> dict:
        tools = params.get("tools")
        original_system = params.get("system", "")

        capture.calls.append({
            "type": "tool_instruction",
            "has_tools": tools is not None,
            "tool_count": len(tools) if tools else 0,
        })

        if tools:
            tool_names = [t.get("name", "unknown") for t in tools]
            capture.calls[-1]["tool_names"] = tool_names

            instruction = f"IMPORTANT: You MUST call one of these tools: {tool_names}. Do not respond with plain text."
            new_system = f"{instruction}\n\n{original_system}" if original_system else instruction

            capture.calls[-1]["instruction_added"] = True
            return {**params, "system": new_system}

        capture.calls[-1]["instruction_added"] = False
        return params

    try:
        instrument_anthropic(MeterOptions(
            emit_metric=capture.capture_metric,
            modify_params=inject_tool_instruction,
        ))

        client = Anthropic()

        tools = [
            {
                "name": "get_weather",
                "description": "Get weather for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        ]

        response = client.messages.create(
            model="claude-3-haiku-20240307",
            messages=[{"role": "user", "content": "What's the weather in Paris?"}],
            tools=tools,
            max_tokens=100,
        )

        if capture.calls:
            call = capture.calls[0]

            # Check 1: Tools detected
            result.add_check(
                "Tools detected",
                call["has_tools"] and call["tool_count"] == 1,
                f"Tools: {call.get('tool_names', [])}"
            )

            # Check 2: Instruction added
            result.add_check(
                "Instruction added",
                call.get("instruction_added", False),
                "System prompt modified with tool instruction"
            )

        # Check 3: Tool was called
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        tool_called = len(tool_use_blocks) > 0
        tool_name = tool_use_blocks[0].name if tool_called else "none"
        result.add_check(
            "Tool was called",
            tool_called,
            f"Tool called: {tool_name}"
        )

    except Exception as e:
        result.error = str(e)
        result.passed = False

    return result


# =============================================================================
# EXP-2.1c: Conditional Injection
# =============================================================================
def test_conditional_injection() -> TestResult:
    """Test conditional injection based on model for Anthropic."""
    result = TestResult("EXP-2.1c: Conditional Injection (Anthropic)", True)
    reset_instrumentation()

    def conditional_inject(params: dict, context: BeforeRequestContext) -> dict:
        model = params.get("model", "")

        capture.calls.append({
            "type": "conditional_injection",
            "model": model,
            "should_inject": "haiku" in model,
        })

        # Only inject for haiku models
        if "haiku" in model:
            original_system = params.get("system", "")
            new_system = f"CONDITIONAL_INJECTION_ACTIVE\n\n{original_system}" if original_system else "CONDITIONAL_INJECTION_ACTIVE"
            capture.calls[-1]["injected"] = True
            return {**params, "system": new_system}

        capture.calls[-1]["injected"] = False
        return params

    try:
        instrument_anthropic(MeterOptions(
            emit_metric=capture.capture_metric,
            modify_params=conditional_inject,
        ))

        client = Anthropic()

        response = client.messages.create(
            model="claude-3-haiku-20240307",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=5,
        )

        if len(capture.calls) >= 1:
            call = capture.calls[0]
            result.add_check(
                "claude-3-haiku: injection applied",
                call.get("injected", False),
                f"Model: {call['model']}, Injected: {call.get('injected')}"
            )

    except Exception as e:
        result.error = str(e)
        result.passed = False

    return result


# =============================================================================
# Main
# =============================================================================
async def main():
    print("\n" + "=" * 70)
    print("PROMPT CONTROL VALIDATION - ANTHROPIC")
    print("Using real Anthropic API")
    print("=" * 70)

    results: list[TestResult] = []

    results.append(test_prompt_injection_sync())
    results.append(await test_prompt_injection_async())
    results.append(test_context_compaction())
    results.append(test_tool_use_instruction())
    results.append(test_prompt_diff())
    results.append(test_tools_injection())
    results.append(test_conditional_injection())

    uninstrument_anthropic()

    for r in results:
        r.print_report()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed_count = sum(1 for r in results if r.passed)
    total_count = len(results)

    for r in results:
        status = "+ PASS" if r.passed else "x FAIL"
        checks_passed = sum(1 for _, p, _ in r.checks if p)
        checks_total = len(r.checks)
        print(f"  {status} {r.name} ({checks_passed}/{checks_total} checks)")

    print(f"\nOverall: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n+ ALL TESTS PASSED")
        return 0
    else:
        print("\nx SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
