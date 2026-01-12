#!/usr/bin/env python3
"""
EXP-2.x: Prompt Control Validation (Comprehensive)

Validates:
- EXP-2.1: Prompt Injection (sync + async, prepend + append)
- EXP-2.4: Context Compaction (truncation + token counting)
- EXP-2.5: Tool Use Instruction
- EXP-2.6: Prompt Diff (via ModifyParamsResult)
- EXP-3.1: Tools Injection

Usage:
    python examples/experiments/validate_prompt_control.py
"""

import asyncio
import json
import sys
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, "src")

from dotenv import load_dotenv

load_dotenv()

from openai import AsyncOpenAI, OpenAI

from aden import (
    BeforeRequestContext,
    MeterOptions,
    MetricEvent,
    ModifyParamsResult,
)
from aden.instrument_openai import instrument_openai, uninstrument_openai


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
            icon = "✓" if check_passed else "✗"
            print(f"  {icon} {check_name}")
            if detail:
                # Indent multi-line details
                for line in detail.split("\n"):
                    print(f"      {line}")

        if self.error:
            print(f"\n  ERROR: {self.error}")


# Global state for capturing hook data
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
    uninstrument_openai()


def format_messages(messages: list[dict], max_content_len: int = 50) -> str:
    """Format messages for display."""
    lines = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if content and len(content) > max_content_len:
            content = content[:max_content_len] + "..."
        lines.append(f"[{i}] {role}: {content!r}")
    return "\n".join(lines)


def format_tools(tools: list[dict] | None) -> str:
    """Format tools for display."""
    if not tools:
        return "(none)"
    names = [t.get("function", {}).get("name", "?") for t in tools]
    return ", ".join(names)


# =============================================================================
# EXP-2.1: Prompt Injection (Comprehensive)
# =============================================================================
def test_prompt_injection_sync_prepend() -> TestResult:
    """Test sync prompt injection with prepend."""
    result = TestResult("EXP-2.1a: Prompt Injection (sync, prepend)", True)
    reset_instrumentation()

    injected_system = "INJECTED_SYSTEM_PROMPT_12345"

    def inject_prepend(params: dict, context: BeforeRequestContext) -> dict:
        original_messages = params.get("messages", [])
        capture.calls.append({
            "type": "inject_prepend",
            "original_messages": list(original_messages),
            "original_count": len(original_messages),
        })

        messages = list(original_messages)
        messages.insert(0, {"role": "system", "content": injected_system})

        modified = {**params, "messages": messages}
        capture.calls[-1]["modified_messages"] = list(messages)
        capture.calls[-1]["modified_count"] = len(messages)
        return modified

    try:
        instrument_openai(MeterOptions(
            emit_metric=capture.capture_metric,
            modify_params=inject_prepend,
        ))

        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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

            # Check 2: Original message count
            result.add_check(
                "Original message count",
                call["original_count"] == 1,
                f"Expected 1, got {call['original_count']}"
            )

            # Check 3: Modified message count
            result.add_check(
                "Modified message count",
                call["modified_count"] == 2,
                f"Expected 2, got {call['modified_count']}"
            )

            # Check 4: Injection at position 0
            if call["modified_messages"]:
                first_msg = call["modified_messages"][0]
                is_system = first_msg.get("role") == "system"
                has_content = injected_system in first_msg.get("content", "")
                result.add_check(
                    "Injection at position 0",
                    is_system and has_content,
                    f"First message: role={first_msg.get('role')}, content={first_msg.get('content', '')[:50]}"
                )

            # Check 5: Original user message preserved
            if len(call["modified_messages"]) > 1:
                user_msg = call["modified_messages"][1]
                result.add_check(
                    "Original user message preserved",
                    user_msg.get("role") == "user",
                    f"Second message: {user_msg}"
                )

        # Check 6: Response received
        response_content = response.choices[0].message.content
        result.add_check(
            "Response received",
            response_content is not None,
            f"Response: {response_content}"
        )

        # Check 7: Metric emitted
        result.add_check(
            "Metric emitted",
            len(capture.metrics) == 1,
            f"Metrics: {len(capture.metrics)}"
        )

    except Exception as e:
        result.error = str(e)
        result.passed = False

    return result


async def test_prompt_injection_async_append() -> TestResult:
    """Test async prompt injection with append."""
    result = TestResult("EXP-2.1b: Prompt Injection (async, append)", True)
    reset_instrumentation()

    injected_instruction = "INJECTED_INSTRUCTION_67890"

    def inject_append(params: dict, context: BeforeRequestContext) -> dict:
        original_messages = params.get("messages", [])
        capture.calls.append({
            "type": "inject_append",
            "original_count": len(original_messages),
        })

        messages = list(original_messages)
        messages.append({"role": "system", "content": injected_instruction})

        capture.calls[-1]["modified_count"] = len(messages)
        capture.calls[-1]["last_message"] = messages[-1]
        return {**params, "messages": messages}

    try:
        instrument_openai(MeterOptions(
            emit_metric=capture.capture_metric,
            modify_params=inject_append,
        ))

        client = AsyncOpenAI()
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'async'"}],
            max_tokens=10,
        )

        # Check 1: Hook called
        result.add_check(
            "Hook called (async)",
            len(capture.calls) == 1,
            f"Calls: {len(capture.calls)}"
        )

        if capture.calls:
            call = capture.calls[0]

            # Check 2: Message appended
            result.add_check(
                "Message appended",
                call["modified_count"] == call["original_count"] + 1,
                f"{call['original_count']} -> {call['modified_count']}"
            )

            # Check 3: Last message is injection
            last_msg = call.get("last_message", {})
            result.add_check(
                "Last message is injection",
                last_msg.get("role") == "system" and injected_instruction in last_msg.get("content", ""),
                f"Last: {last_msg}"
            )

        # Check 4: Async response
        result.add_check(
            "Async response received",
            response.choices[0].message.content is not None,
            f"Response: {response.choices[0].message.content}"
        )

    except Exception as e:
        result.error = str(e)
        result.passed = False

    return result


# =============================================================================
# EXP-2.4: Context Compaction
# =============================================================================
def test_context_compaction() -> TestResult:
    """Test context compaction with detailed tracking."""
    result = TestResult("EXP-2.4: Context Compaction", True)
    reset_instrumentation()

    max_messages = 5
    compaction_triggered = False

    def compact_context(params: dict, context: BeforeRequestContext) -> dict:
        nonlocal compaction_triggered
        messages = params.get("messages", [])
        original_count = len(messages)

        capture.calls.append({
            "type": "compaction_check",
            "original_count": original_count,
            "threshold": max_messages,
        })

        if len(messages) > max_messages:
            compaction_triggered = True
            # Keep system messages + last N non-system messages
            system_msgs = [m for m in messages if m.get("role") == "system"]
            non_system = [m for m in messages if m.get("role") != "system"]

            keep_count = max_messages - len(system_msgs)
            compacted = system_msgs + non_system[-keep_count:]

            capture.calls[-1]["compacted_count"] = len(compacted)
            capture.calls[-1]["removed_count"] = original_count - len(compacted)
            capture.calls[-1]["kept_messages"] = [
                {"role": m["role"], "content": m.get("content", "")[:30]}
                for m in compacted
            ]

            return {**params, "messages": compacted}

        capture.calls[-1]["compacted_count"] = original_count
        capture.calls[-1]["removed_count"] = 0
        return params

    try:
        instrument_openai(MeterOptions(
            emit_metric=capture.capture_metric,
            modify_params=compact_context,
        ))

        client = OpenAI()

        # Build long conversation (12 messages)
        long_conversation = []
        for i in range(11):
            long_conversation.append({"role": "user", "content": f"Message number {i}"})
            if i < 10:
                long_conversation.append({"role": "assistant", "content": f"Response to {i}"})

        # Add final question
        long_conversation.append({"role": "user", "content": "What was my first message about?"})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=long_conversation,
            max_tokens=30,
        )

        # Check 1: Compaction was triggered
        result.add_check(
            "Compaction triggered",
            compaction_triggered,
            f"Original: {len(long_conversation)} messages, threshold: {max_messages}"
        )

        if capture.calls:
            call = capture.calls[0]

            # Check 2: Original count captured
            result.add_check(
                "Original count captured",
                call["original_count"] == len(long_conversation),
                f"Expected {len(long_conversation)}, got {call['original_count']}"
            )

            # Check 3: Compacted to threshold
            result.add_check(
                "Compacted to threshold",
                call["compacted_count"] == max_messages,
                f"Expected {max_messages}, got {call['compacted_count']}"
            )

            # Check 4: Messages removed
            removed = call.get("removed_count", 0)
            expected_removed = len(long_conversation) - max_messages
            result.add_check(
                "Correct messages removed",
                removed == expected_removed,
                f"Removed {removed} messages (expected {expected_removed})"
            )

            # Check 5: Show kept messages
            kept = call.get("kept_messages", [])
            kept_summary = "\n".join([f"  - {m['role']}: {m['content']}" for m in kept])
            result.add_check(
                "Kept messages logged",
                len(kept) > 0,
                f"Kept messages:\n{kept_summary}"
            )

        # Check 6: Response still works
        result.add_check(
            "Response received after compaction",
            response.choices[0].message.content is not None,
            f"Response: {response.choices[0].message.content}"
        )

    except Exception as e:
        result.error = str(e)
        result.passed = False

    return result


# =============================================================================
# EXP-2.5: Tool Use Instruction
# =============================================================================
def test_tool_use_instruction() -> TestResult:
    """Test tool instruction injection with detailed tracking."""
    result = TestResult("EXP-2.5: Tool Use Instruction", True)
    reset_instrumentation()

    def inject_tool_instruction(params: dict, context: BeforeRequestContext) -> dict:
        tools = params.get("tools")
        original_messages = params.get("messages", [])

        capture.calls.append({
            "type": "tool_instruction",
            "has_tools": tools is not None,
            "tool_count": len(tools) if tools else 0,
            "original_msg_count": len(original_messages),
        })

        if tools:
            tool_names = [t.get("function", {}).get("name", "unknown") for t in tools]
            capture.calls[-1]["tool_names"] = tool_names

            messages = list(original_messages)
            instruction = f"IMPORTANT: You MUST call one of these tools: {tool_names}. Do not respond with plain text."
            messages.append({"role": "system", "content": instruction})

            capture.calls[-1]["instruction_added"] = True
            capture.calls[-1]["modified_msg_count"] = len(messages)
            return {**params, "messages": messages}

        capture.calls[-1]["instruction_added"] = False
        return params

    try:
        instrument_openai(MeterOptions(
            emit_metric=capture.capture_metric,
            modify_params=inject_tool_instruction,
        ))

        client = OpenAI()

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get current time for a timezone",
                    "parameters": {
                        "type": "object",
                        "properties": {"timezone": {"type": "string"}},
                    },
                },
            },
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
            tools=tools,
            max_tokens=100,
        )

        if capture.calls:
            call = capture.calls[0]

            # Check 1: Tools detected
            result.add_check(
                "Tools detected",
                call["has_tools"] and call["tool_count"] == 2,
                f"Tools: {call.get('tool_names', [])}"
            )

            # Check 2: Instruction added
            result.add_check(
                "Instruction added",
                call.get("instruction_added", False),
                f"Messages: {call['original_msg_count']} -> {call.get('modified_msg_count', 'N/A')}"
            )

        # Check 3: Tool was called
        tool_calls = response.choices[0].message.tool_calls
        tool_called = tool_calls is not None and len(tool_calls) > 0
        tool_name = tool_calls[0].function.name if tool_called else "none"
        result.add_check(
            "Tool was called",
            tool_called,
            f"Tool called: {tool_name}"
        )

        # Check 4: If tool called, show arguments
        if tool_called:
            args = tool_calls[0].function.arguments
            result.add_check(
                "Tool arguments captured",
                args is not None,
                f"Arguments: {args}"
            )

    except Exception as e:
        result.error = str(e)
        result.passed = False

    return result


# =============================================================================
# EXP-2.6: Prompt Diff (ModifyParamsResult)
# =============================================================================
def test_prompt_diff() -> TestResult:
    """Test ModifyParamsResult for diff tracking."""
    result = TestResult("EXP-2.6: Prompt Diff (ModifyParamsResult)", True)
    reset_instrumentation()

    def tracked_modification(params: dict, context: BeforeRequestContext) -> ModifyParamsResult:
        import copy
        original = copy.deepcopy(params)

        messages = list(params.get("messages", []))
        # Add system message
        messages.insert(0, {"role": "system", "content": "Be concise."})
        # Modify would-be temperature (simulating param modification)
        modified = {**params, "messages": messages, "temperature": 0.5}

        result_obj = ModifyParamsResult(params=modified, original_params=original)

        # Compute diff
        diff = {
            "messages_added": len(messages) - len(original.get("messages", [])),
            "params_added": [k for k in modified if k not in original],
            "params_modified": [k for k in modified if k in original and modified[k] != original[k]],
        }

        capture.calls.append({
            "type": "modify_params_result",
            "original_keys": list(original.keys()),
            "modified_keys": list(modified.keys()),
            "has_original_params": result_obj.original_params is not None,
            "diff": diff,
        })

        return result_obj

    try:
        instrument_openai(MeterOptions(
            emit_metric=capture.capture_metric,
            modify_params=tracked_modification,
        ))

        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=10,
        )

        if capture.calls:
            call = capture.calls[0]

            # Check 1: ModifyParamsResult has original
            result.add_check(
                "original_params captured",
                call["has_original_params"],
                "ModifyParamsResult.original_params is set"
            )

            # Check 2: Diff computed correctly
            diff = call.get("diff", {})
            result.add_check(
                "Messages diff detected",
                diff.get("messages_added", 0) == 1,
                f"Messages added: {diff.get('messages_added')}"
            )

            # Check 3: Param additions detected
            result.add_check(
                "Param additions detected",
                "temperature" in diff.get("params_added", []),
                f"Params added: {diff.get('params_added')}"
            )

            # Check 4: Show full diff
            diff_str = json.dumps(diff, indent=2)
            result.add_check(
                "Full diff available",
                True,
                f"Diff:\n{diff_str}"
            )

        # Check 5: Response works
        result.add_check(
            "Response received",
            response.choices[0].message.content is not None,
            f"Response: {response.choices[0].message.content}"
        )

    except Exception as e:
        result.error = str(e)
        result.passed = False

    return result


# =============================================================================
# EXP-3.1: Tools Injection
# =============================================================================
def test_tools_injection() -> TestResult:
    """Test tools injection with detailed tracking."""
    result = TestResult("EXP-3.1: Tools Injection", True)
    reset_instrumentation()

    audit_tool = {
        "type": "function",
        "function": {
            "name": "audit_log",
            "description": "Log an audit event for compliance tracking",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_type": {"type": "string", "description": "Type of event"},
                    "details": {"type": "string", "description": "Event details"},
                },
                "required": ["event_type"],
            },
        },
    }

    def inject_tools(params: dict, context: BeforeRequestContext) -> dict:
        original_tools = params.get("tools") or []

        capture.calls.append({
            "type": "tools_injection",
            "original_tool_count": len(original_tools),
            "original_tool_names": [t.get("function", {}).get("name") for t in original_tools],
        })

        tools = list(original_tools)
        tools.append(audit_tool)

        capture.calls[-1]["final_tool_count"] = len(tools)
        capture.calls[-1]["final_tool_names"] = [t.get("function", {}).get("name") for t in tools]
        capture.calls[-1]["injected_tool"] = "audit_log"

        return {**params, "tools": tools}

    try:
        instrument_openai(MeterOptions(
            emit_metric=capture.capture_metric,
            modify_params=inject_tools,
        ))

        client = OpenAI()

        original_tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
            },
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Search for Python tutorials"}],
            tools=original_tools,
            max_tokens=100,
        )

        if capture.calls:
            call = capture.calls[0]

            # Check 1: Original tools captured
            result.add_check(
                "Original tools captured",
                call["original_tool_count"] == 1,
                f"Original: {call['original_tool_names']}"
            )

            # Check 2: Tool injected
            result.add_check(
                "Tool injected",
                call["final_tool_count"] == 2,
                f"Final: {call['final_tool_names']}"
            )

            # Check 3: Correct tool injected
            result.add_check(
                "Correct tool injected",
                "audit_log" in call["final_tool_names"],
                f"Injected: {call['injected_tool']}"
            )

            # Check 4: Original tool preserved
            result.add_check(
                "Original tool preserved",
                "search" in call["final_tool_names"],
                f"Original 'search' still present"
            )

        # Check 5: Response works
        tool_calls = response.choices[0].message.tool_calls
        if tool_calls:
            called_tools = [tc.function.name for tc in tool_calls]
            result.add_check(
                "Response with tool call",
                True,
                f"Tools called: {called_tools}"
            )
        else:
            result.add_check(
                "Response received",
                response.choices[0].message.content is not None,
                f"Text response: {response.choices[0].message.content}"
            )

    except Exception as e:
        result.error = str(e)
        result.passed = False

    return result


# =============================================================================
# EXP-2.1c: Conditional Injection
# =============================================================================
def test_conditional_injection() -> TestResult:
    """Test conditional injection based on model."""
    result = TestResult("EXP-2.1c: Conditional Injection (by model)", True)
    reset_instrumentation()

    def conditional_inject(params: dict, context: BeforeRequestContext) -> dict:
        model = params.get("model", "")

        capture.calls.append({
            "type": "conditional_injection",
            "model": model,
            "should_inject": "gpt-4o" in model,
        })

        # Only inject for gpt-4o models
        if "gpt-4o" in model:
            messages = list(params.get("messages", []))
            messages.insert(0, {"role": "system", "content": "CONDITIONAL_INJECTION_ACTIVE"})
            capture.calls[-1]["injected"] = True
            return {**params, "messages": messages}

        capture.calls[-1]["injected"] = False
        return params

    try:
        instrument_openai(MeterOptions(
            emit_metric=capture.capture_metric,
            modify_params=conditional_inject,
        ))

        client = OpenAI()

        # Test 1: gpt-4o-mini (should inject)
        response1 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Test 1"}],
            max_tokens=5,
        )

        # Check for first call
        if len(capture.calls) >= 1:
            call1 = capture.calls[0]
            result.add_check(
                "gpt-4o-mini: injection applied",
                call1.get("injected", False),
                f"Model: {call1['model']}, Injected: {call1.get('injected')}"
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
    print("PROMPT CONTROL VALIDATION (Comprehensive)")
    print("Using real OpenAI API")
    print("=" * 70)

    results: list[TestResult] = []

    # Run sync tests
    results.append(test_prompt_injection_sync_prepend())
    results.append(await test_prompt_injection_async_append())
    results.append(test_context_compaction())
    results.append(test_tool_use_instruction())
    results.append(test_prompt_diff())
    results.append(test_tools_injection())
    results.append(test_conditional_injection())

    # Cleanup
    uninstrument_openai()

    # Print all results
    for r in results:
        r.print_report()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed_count = sum(1 for r in results if r.passed)
    total_count = len(results)

    for r in results:
        status = "✓ PASS" if r.passed else "✗ FAIL"
        checks_passed = sum(1 for _, p, _ in r.checks if p)
        checks_total = len(r.checks)
        print(f"  {status} {r.name} ({checks_passed}/{checks_total} checks)")

    print(f"\nOverall: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n✓ ALL TESTS PASSED")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
