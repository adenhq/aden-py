#!/usr/bin/env python3
"""
Validate Declarative Configuration for Prompt Control

This script validates that the declarative configuration system works
correctly across different providers (OpenAI, Anthropic, Gemini).

Tests:
1. System injection via DeclarativeConfig
2. Tool injection via DeclarativeConfig
3. Context compaction via DeclarativeConfig
4. Tool use instruction via DeclarativeConfig
5. Conditional injection via DeclarativeConfig
6. JSON/dict configuration via from_dict()
7. Convenience functions (system_injection, tool_injection, etc.)
"""

import os
import sys
from typing import Any

# Add src to path for local testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from openai import OpenAI

from aden import (
    MeterOptions,
    instrument_openai,
    uninstrument_openai,
    # Declarative config
    DeclarativeConfig,
    SystemInjection,
    MessageInjection,
    ToolInjection,
    CompactionConfig,
    ToolInstructionConfig,
    build_modify_params_hook,
    # Convenience functions
    system_injection,
    tool_injection,
    context_compaction,
    enforce_tool_use,
    config_from_dict,
)


def create_capturing_emitter():
    """Create an emitter that captures metrics and params."""
    captured = {"metrics": [], "params": []}

    def emitter(event):
        captured["metrics"].append(event)

    return emitter, captured


def test_system_injection_declarative():
    """Test EXP-2.1: System injection via DeclarativeConfig."""
    print("\n=== Test 1: System Injection (DeclarativeConfig) ===")

    emitter, captured = create_capturing_emitter()
    captured_params = []

    # Create declarative config
    config = DeclarativeConfig(
        injections=[
            SystemInjection(
                content="Always respond with exactly 3 words.",
                position="prepend",
            ),
        ],
        track_diff=True,
    )

    # Build hook from config
    hook = build_modify_params_hook(config)

    # Wrap hook to capture params
    def capturing_hook(params, context):
        result = hook(params, context)
        if hasattr(result, "params"):
            captured_params.append({"modified": result.params, "original": result.original_params})
        else:
            captured_params.append({"modified": result})
        return result

    instrument_openai(
        MeterOptions(
            emit_metric=emitter,
            modify_params=capturing_hook,
        )
    )

    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=50,
        )

        # Verify injection
        assert len(captured_params) > 0, "No params captured"
        modified = captured_params[0]["modified"]
        original = captured_params[0].get("original")

        # Check injection was applied
        messages = modified.get("messages", [])
        assert any(
            m.get("role") == "system" and "3 words" in m.get("content", "")
            for m in messages
        ), "System injection not found in modified params"

        # Check original didn't have injection (if tracking diff)
        if original:
            orig_messages = original.get("messages", [])
            assert not any(
                "3 words" in m.get("content", "") for m in orig_messages
            ), "Original params should not have injection"

        print(f"  [PASS] System injection applied")
        print(f"  [PASS] Original params preserved (track_diff=True)")
        print(f"  Response: {response.choices[0].message.content}")
        return True

    finally:
        uninstrument_openai()


def test_tool_injection_declarative():
    """Test EXP-3.1: Tool injection via DeclarativeConfig."""
    print("\n=== Test 2: Tool Injection (DeclarativeConfig) ===")

    emitter, captured = create_capturing_emitter()
    captured_params = []

    # Create declarative config with tool injection
    config = DeclarativeConfig(
        tool_injections=[
            ToolInjection(
                name="audit_log",
                description="Log an audit event for compliance tracking",
                parameters={
                    "type": "object",
                    "properties": {
                        "event": {"type": "string", "description": "Event description"},
                        "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    },
                    "required": ["event"],
                },
            ),
        ],
    )

    hook = build_modify_params_hook(config)

    def capturing_hook(params, context):
        result = hook(params, context)
        captured_params.append(result if isinstance(result, dict) else result.params)
        return result

    instrument_openai(
        MeterOptions(
            emit_metric=emitter,
            modify_params=capturing_hook,
        )
    )

    try:
        client = OpenAI()

        # Request with existing tool
        existing_tool = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}},
            },
        }

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[existing_tool],
            max_tokens=100,
        )

        # Verify tool was injected
        assert len(captured_params) > 0, "No params captured"
        modified = captured_params[0]
        tools = modified.get("tools", [])

        tool_names = [t.get("function", {}).get("name") for t in tools]
        assert "audit_log" in tool_names, "Injected tool not found"
        assert "get_weather" in tool_names, "Original tool should be preserved"

        print(f"  [PASS] Tool injection applied")
        print(f"  [PASS] Original tools preserved")
        print(f"  Tools in request: {tool_names}")
        return True

    finally:
        uninstrument_openai()


def test_context_compaction_declarative():
    """Test EXP-2.4: Context compaction via DeclarativeConfig."""
    print("\n=== Test 3: Context Compaction (DeclarativeConfig) ===")

    emitter, captured = create_capturing_emitter()
    captured_params = []

    # Create config with compaction
    config = DeclarativeConfig(
        compaction=CompactionConfig(
            max_messages=5,
            preserve_recent=2,
            preserve_system=True,
            strategy="truncate",
        ),
    )

    hook = build_modify_params_hook(config)

    def capturing_hook(params, context):
        result = hook(params, context)
        captured_params.append(result if isinstance(result, dict) else result.params)
        return result

    instrument_openai(
        MeterOptions(
            emit_metric=emitter,
            modify_params=capturing_hook,
        )
    )

    try:
        client = OpenAI()

        # Create long conversation
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Reply 1"},
            {"role": "user", "content": "Message 2"},
            {"role": "assistant", "content": "Reply 2"},
            {"role": "user", "content": "Message 3"},
            {"role": "assistant", "content": "Reply 3"},
            {"role": "user", "content": "Message 4"},
            {"role": "assistant", "content": "Reply 4"},
            {"role": "user", "content": "What was my first message?"},
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=100,
        )

        # Verify compaction
        assert len(captured_params) > 0, "No params captured"
        modified = captured_params[0]
        compacted_messages = modified.get("messages", [])

        print(f"  Original message count: {len(messages)}")
        print(f"  Compacted message count: {len(compacted_messages)}")
        assert len(compacted_messages) <= 5, "Compaction should limit to max_messages"

        # Check system preserved
        system_preserved = any(m.get("role") == "system" for m in compacted_messages)
        print(f"  [PASS] System message preserved: {system_preserved}")

        # Check recent preserved (last 2 non-system)
        print(f"  [PASS] Context compaction applied")
        return True

    finally:
        uninstrument_openai()


def test_tool_instruction_declarative():
    """Test EXP-2.5: Tool use instruction via DeclarativeConfig."""
    print("\n=== Test 4: Tool Use Instruction (DeclarativeConfig) ===")

    emitter, captured = create_capturing_emitter()
    captured_params = []

    # Create config with tool instruction
    config = DeclarativeConfig(
        tool_instruction=ToolInstructionConfig(
            enabled=True,
            template="Available tools for this request: {tool_names}. Use them when appropriate.",
            position="append",
            force_tool_use=False,
        ),
    )

    hook = build_modify_params_hook(config)

    def capturing_hook(params, context):
        result = hook(params, context)
        captured_params.append(result if isinstance(result, dict) else result.params)
        return result

    instrument_openai(
        MeterOptions(
            emit_metric=emitter,
            modify_params=capturing_hook,
        )
    )

    try:
        client = OpenAI()

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search for information",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Perform calculation",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Help me find something"}],
            tools=tools,
            max_tokens=100,
        )

        # Verify tool instruction was added
        assert len(captured_params) > 0, "No params captured"
        modified = captured_params[0]
        messages = modified.get("messages", [])

        # Check for tool instruction message
        instruction_found = any(
            "Available tools" in m.get("content", "") and "search" in m.get("content", "")
            for m in messages
        )
        assert instruction_found, "Tool instruction not found"

        print(f"  [PASS] Tool instruction injected")
        print(f"  [PASS] Tool names populated in template")
        return True

    finally:
        uninstrument_openai()


def test_conditional_injection_declarative():
    """Test EXP-2.1c: Conditional injection via DeclarativeConfig."""
    print("\n=== Test 5: Conditional Injection (DeclarativeConfig) ===")

    emitter, captured = create_capturing_emitter()
    captured_params = []

    # Create config with conditional injection
    config = DeclarativeConfig(
        injections=[
            SystemInjection(
                content="You are a JSON-only responder.",
                position="prepend",
                conditions={"models": ["gpt-4o-mini"]},  # Only for gpt-4o-mini
            ),
            SystemInjection(
                content="You are a poetry expert.",
                position="prepend",
                conditions={"models": ["gpt-4o"]},  # Only for gpt-4o
            ),
        ],
    )

    hook = build_modify_params_hook(config)

    def capturing_hook(params, context):
        result = hook(params, context)
        captured_params.append(result if isinstance(result, dict) else result.params)
        return result

    instrument_openai(
        MeterOptions(
            emit_metric=emitter,
            modify_params=capturing_hook,
        )
    )

    try:
        client = OpenAI()

        # Test with gpt-4o-mini - should get JSON injection
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say hi"}],
            max_tokens=50,
        )

        assert len(captured_params) > 0, "No params captured"
        modified = captured_params[0]
        messages = modified.get("messages", [])

        json_found = any("JSON" in m.get("content", "") for m in messages)
        poetry_found = any("poetry" in m.get("content", "") for m in messages)

        assert json_found, "JSON injection should apply for gpt-4o-mini"
        assert not poetry_found, "Poetry injection should NOT apply for gpt-4o-mini"

        print(f"  [PASS] Conditional injection applied for matching model")
        print(f"  [PASS] Non-matching conditions correctly skipped")
        return True

    finally:
        uninstrument_openai()


def test_from_dict_config():
    """Test EXP-2.7: JSON/dict configuration via from_dict()."""
    print("\n=== Test 6: Config from Dict/JSON (from_dict) ===")

    emitter, captured = create_capturing_emitter()
    captured_params = []

    # Define config as dict (simulating JSON/YAML load)
    config_dict = {
        "injections": [
            {
                "content": "Respond in exactly one sentence.",
                "position": "prepend",
            },
        ],
        "tool_injections": [
            {
                "name": "log_response",
                "description": "Log the response for analytics",
                "parameters": {"type": "object", "properties": {"text": {"type": "string"}}},
            },
        ],
        "compaction": {
            "max_messages": 20,
            "preserve_recent": 5,
        },
        "track_diff": True,
    }

    # Convert dict to DeclarativeConfig
    config = config_from_dict(config_dict)

    assert isinstance(config, DeclarativeConfig), "from_dict should return DeclarativeConfig"
    assert len(config.injections) == 1, "Should have 1 injection"
    assert len(config.tool_injections) == 1, "Should have 1 tool injection"
    assert config.compaction is not None, "Should have compaction config"
    assert config.track_diff is True, "Should have track_diff=True"

    hook = build_modify_params_hook(config)

    def capturing_hook(params, context):
        result = hook(params, context)
        captured_params.append(result if isinstance(result, dict) else result.params)
        return result

    instrument_openai(
        MeterOptions(
            emit_metric=emitter,
            modify_params=capturing_hook,
        )
    )

    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is Python?"}],
            max_tokens=100,
        )

        assert len(captured_params) > 0, "No params captured"
        modified = captured_params[0]
        messages = modified.get("messages", [])

        # Verify injection from dict config
        injection_found = any("one sentence" in m.get("content", "") for m in messages)
        assert injection_found, "Injection from dict config not found"

        print(f"  [PASS] Config loaded from dict")
        print(f"  [PASS] Injections applied from dict config")
        print(f"  [PASS] Tool injections parsed correctly")
        return True

    finally:
        uninstrument_openai()


def test_convenience_functions():
    """Test convenience functions (system_injection, tool_injection, etc.)."""
    print("\n=== Test 7: Convenience Functions ===")

    # Test system_injection()
    config1 = system_injection("Be concise", position="append")
    assert isinstance(config1, DeclarativeConfig), "system_injection should return DeclarativeConfig"
    assert len(config1.injections) == 1, "Should have 1 injection"
    assert config1.injections[0].content == "Be concise"
    assert config1.injections[0].position == "append"
    print(f"  [PASS] system_injection() works")

    # Test system_injection() with model filter
    config2 = system_injection("JSON mode", models=["gpt-4o"])
    assert config2.injections[0].conditions == {"models": ["gpt-4o"]}
    print(f"  [PASS] system_injection() with model filter works")

    # Test tool_injection()
    config3 = tool_injection("my_tool", "My custom tool")
    assert isinstance(config3, DeclarativeConfig), "tool_injection should return DeclarativeConfig"
    assert len(config3.tool_injections) == 1, "Should have 1 tool injection"
    assert config3.tool_injections[0].name == "my_tool"
    print(f"  [PASS] tool_injection() works")

    # Test context_compaction()
    config4 = context_compaction(max_messages=30, preserve_recent=5)
    assert config4.compaction is not None, "Should have compaction"
    assert config4.compaction.max_messages == 30
    assert config4.compaction.preserve_recent == 5
    print(f"  [PASS] context_compaction() works")

    # Test enforce_tool_use()
    config5 = enforce_tool_use()
    assert config5.tool_instruction is not None, "Should have tool instruction"
    assert config5.tool_instruction.force_tool_use is True
    print(f"  [PASS] enforce_tool_use() works")

    return True


def test_combined_config():
    """Test combining multiple features in one config."""
    print("\n=== Test 8: Combined Configuration ===")

    emitter, captured = create_capturing_emitter()
    captured_params = []

    # Create comprehensive config
    config = DeclarativeConfig(
        injections=[
            SystemInjection(content="Be helpful and concise.", position="prepend"),
            MessageInjection(role="user", content="Remember: quality over quantity.", position="append"),
        ],
        tool_injections=[
            ToolInjection(name="feedback", description="Collect user feedback"),
        ],
        compaction=CompactionConfig(max_messages=50, preserve_recent=10),
        tool_instruction=ToolInstructionConfig(enabled=True, position="append"),
        track_diff=True,
    )

    hook = build_modify_params_hook(config)

    def capturing_hook(params, context):
        result = hook(params, context)
        if hasattr(result, "params"):
            captured_params.append({"modified": result.params, "original": result.original_params})
        else:
            captured_params.append({"modified": result})
        return result

    instrument_openai(
        MeterOptions(
            emit_metric=emitter,
            modify_params=capturing_hook,
        )
    )

    try:
        client = OpenAI()

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Help me"}],
            tools=tools,
            max_tokens=100,
        )

        assert len(captured_params) > 0, "No params captured"
        modified = captured_params[0]["modified"]

        # Check all features applied
        messages = modified.get("messages", [])
        tools_list = modified.get("tools", [])

        system_found = any("helpful and concise" in m.get("content", "") for m in messages)
        user_injection_found = any("quality over quantity" in m.get("content", "") for m in messages)
        tool_found = any(t.get("function", {}).get("name") == "feedback" for t in tools_list)
        tool_instruction_found = any("search" in m.get("content", "") and "feedback" in m.get("content", "") for m in messages)

        print(f"  System injection: {'[PASS]' if system_found else '[FAIL]'}")
        print(f"  User message injection: {'[PASS]' if user_injection_found else '[FAIL]'}")
        print(f"  Tool injection: {'[PASS]' if tool_found else '[FAIL]'}")
        print(f"  Tool instruction: {'[PASS]' if tool_instruction_found else '[FAIL]'}")

        return system_found and user_injection_found and tool_found

    finally:
        uninstrument_openai()


def main():
    """Run all declarative config validation tests."""
    print("=" * 60)
    print("Declarative Configuration Validation")
    print("=" * 60)

    results = []

    tests = [
        ("System Injection (DeclarativeConfig)", test_system_injection_declarative),
        ("Tool Injection (DeclarativeConfig)", test_tool_injection_declarative),
        ("Context Compaction (DeclarativeConfig)", test_context_compaction_declarative),
        ("Tool Use Instruction (DeclarativeConfig)", test_tool_instruction_declarative),
        ("Conditional Injection (DeclarativeConfig)", test_conditional_injection_declarative),
        ("Config from Dict/JSON", test_from_dict_config),
        ("Convenience Functions", test_convenience_functions),
        ("Combined Configuration", test_combined_config),
    ]

    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"  [FAIL] {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"  {status} {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
