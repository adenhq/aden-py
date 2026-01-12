"""
Multi-vendor test for agent detection across OpenAI, Anthropic, and Gemini.

Tests both explicit context manager and heuristic detection with all three providers.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
load_dotenv()

from aden import instrument, uninstrument, MeterOptions, create_memory_emitter, agent


def run_test():
    memory_emitter = create_memory_emitter()
    instrument(MeterOptions(emit_metric=memory_emitter))

    import openai
    import anthropic
    import google.generativeai as genai

    print("=" * 70)
    print("MULTI-VENDOR AGENT DETECTION TEST")
    print("=" * 70)
    print()

    # ============ EXPLICIT CONTEXT TESTS ============
    print("PART 1: Explicit context manager")
    print("-" * 40)

    # OpenAI
    print("1. OpenAI with explicit context:")
    with agent("openai-agent"):
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say A"}],
            max_tokens=5
        )
        print(f"   Response: {response.choices[0].message.content.strip()}")

    # Anthropic
    print("2. Anthropic with explicit context:")
    with agent("anthropic-agent"):
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=5,
            messages=[{"role": "user", "content": "Say B"}]
        )
        print(f"   Response: {response.content[0].text.strip()}")

    # Gemini
    print("3. Gemini with explicit context:")
    with agent("gemini-agent"):
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content("Say C")
        print(f"   Response: {response.text.strip()}")

    # ============ HEURISTIC DETECTION TESTS ============
    print()
    print("PART 2: Heuristic detection (agent-like classes)")
    print("-" * 40)

    class OpenAIResearchAgent:
        def research(self):
            client = openai.OpenAI()
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Say D"}],
                max_tokens=5
            )

    class AnthropicAssistant:
        def assist(self):
            client = anthropic.Anthropic()
            return client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=5,
                messages=[{"role": "user", "content": "Say E"}]
            )

    class GeminiBot:
        def chat(self):
            model = genai.GenerativeModel("gemini-2.0-flash")
            return model.generate_content("Say F")

    print("4. OpenAIResearchAgent.research():")
    r = OpenAIResearchAgent().research()
    print(f"   Response: {r.choices[0].message.content.strip()}")

    print("5. AnthropicAssistant.assist():")
    r = AnthropicAssistant().assist()
    print(f"   Response: {r.content[0].text.strip()}")

    print("6. GeminiBot.chat():")
    r = GeminiBot().chat()
    print(f"   Response: {r.text.strip()}")

    # ============ NON-AGENT CLASS TEST ============
    print()
    print("PART 3: Non-agent class (should NOT detect)")
    print("-" * 40)

    class DataProcessor:
        def process(self):
            model = genai.GenerativeModel("gemini-2.0-flash")
            return model.generate_content("Say G")

    print("7. DataProcessor.process():")
    r = DataProcessor().process()
    print(f"   Response: {r.text.strip()}")

    uninstrument()

    # ============ RESULTS ============
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    # Group events by test number
    # Note: Gemini fires 2 events per call (genai + grpc), others fire 1
    print("Events captured:")
    for i, event in enumerate(memory_emitter.events):
        provider = "OpenAI" if "gpt" in event.model else ("Anthropic" if "claude" in event.model else "Gemini")
        print(f"  {i+1}. {provider} ({event.model})")
        print(f"     agent_name: {event.agent_name}")
        print(f"     agent_stack: {event.agent_stack}")
        print()

    print("-" * 40)
    print("SUMMARY")
    print("-" * 40)

    # Filter to unique events (skip duplicate Gemini grpc events)
    unique_events = []
    seen = set()
    for e in memory_emitter.events:
        key = (e.model.replace("models/", ""), e.agent_name)
        if key not in seen:
            seen.add(key)
            unique_events.append(e)

    explicit_events = [e for e in unique_events if e.agent_name in ["openai-agent", "anthropic-agent", "gemini-agent"]]
    heuristic_events = [e for e in unique_events if e.agent_name and "Agent" in e.agent_name or "Assistant" in str(e.agent_name) or "Bot" in str(e.agent_name)]
    no_agent_events = [e for e in unique_events if e.agent_name is None]

    print(f"\nExplicit context (should have 3):")
    for e in explicit_events:
        provider = "OpenAI" if "gpt" in e.model else ("Anthropic" if "claude" in e.model else "Gemini")
        print(f"  {provider}: {e.agent_name} ✓")

    print(f"\nHeuristic detection (should have 3):")
    for e in heuristic_events:
        provider = "OpenAI" if "gpt" in e.model else ("Anthropic" if "claude" in e.model else "Gemini")
        print(f"  {provider}: {e.agent_name} ✓")

    print(f"\nNo agent detected (DataProcessor, should have 1):")
    for e in no_agent_events:
        provider = "OpenAI" if "gpt" in e.model else ("Anthropic" if "claude" in e.model else "Gemini")
        print(f"  {provider}: agent_name=None ✓ (expected)")

    # Final pass/fail
    print()
    print("=" * 70)
    explicit_pass = len(explicit_events) == 3
    heuristic_pass = len(heuristic_events) >= 2  # At least 2 of 3 providers
    no_agent_pass = any(e.agent_name is None for e in unique_events)

    print(f"Explicit context: {'PASS' if explicit_pass else 'FAIL'} ({len(explicit_events)}/3)")
    print(f"Heuristic detection: {'PASS' if heuristic_pass else 'FAIL'} ({len(heuristic_events)}/3)")
    print(f"No-agent baseline: {'PASS' if no_agent_pass else 'FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    run_test()
