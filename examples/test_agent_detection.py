"""
Comprehensive test for agent detection heuristics.

Tests various class naming patterns, nesting scenarios, and edge cases.
"""

import sys
import os
import importlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Force reload to ensure fresh code
import aden.call_stack
importlib.reload(aden.call_stack)

from dotenv import load_dotenv
load_dotenv()

from aden import instrument, uninstrument, MeterOptions, create_memory_emitter, agent
from aden.call_stack import capture_call_stack, _is_agent_frame
import inspect


def test_frame_detection():
    """Test _is_agent_frame detection directly."""
    print("=" * 70)
    print("TEST: Direct frame detection")
    print("=" * 70)

    results = []

    class ResearchAgent:
        def run(self):
            frame = inspect.currentframe()
            return _is_agent_frame(frame)

    class DataProcessor:
        def process(self):
            frame = inspect.currentframe()
            return _is_agent_frame(frame)

    class MyAssistant:
        def help(self):
            frame = inspect.currentframe()
            return _is_agent_frame(frame)

    class TaskOrchestrator:
        def orchestrate(self):
            frame = inspect.currentframe()
            return _is_agent_frame(frame)

    class WorkerBot:
        def work(self):
            frame = inspect.currentframe()
            return _is_agent_frame(frame)

    class AICoordinator:
        def coordinate(self):
            frame = inspect.currentframe()
            return _is_agent_frame(frame)

    class PlainClass:
        def do_something(self):
            frame = inspect.currentframe()
            return _is_agent_frame(frame)

    # Function with agent pattern in name
    class SomeClass:
        def execute_agent_task(self):
            frame = inspect.currentframe()
            return _is_agent_frame(frame)

        def run_handler(self):
            frame = inspect.currentframe()
            return _is_agent_frame(frame)

        def invoke_action(self):
            frame = inspect.currentframe()
            return _is_agent_frame(frame)

    # Test all classes
    tests = [
        ("ResearchAgent.run", ResearchAgent().run()),
        ("DataProcessor.process", DataProcessor().process()),
        ("MyAssistant.help", MyAssistant().help()),
        ("TaskOrchestrator.orchestrate", TaskOrchestrator().orchestrate()),
        ("WorkerBot.work", WorkerBot().work()),
        ("AICoordinator.coordinate", AICoordinator().coordinate()),
        ("PlainClass.do_something", PlainClass().do_something()),
        ("SomeClass.execute_agent_task", SomeClass().execute_agent_task()),
        ("SomeClass.run_handler", SomeClass().run_handler()),
        ("SomeClass.invoke_action", SomeClass().invoke_action()),
    ]

    print("\nResults:")
    for name, result in tests:
        status = "DETECTED" if result else "MISSED"
        print(f"  {name}: {result} [{status}]")

    print("\nExpected detections:")
    print("  - ResearchAgent.run (class name contains 'agent')")
    print("  - MyAssistant.help (class name contains 'assistant')")
    print("  - TaskOrchestrator.orchestrate (class name contains 'orchestrator')")
    print("  - WorkerBot.work (class name contains 'bot')")
    print("  - AICoordinator.coordinate (class name contains 'coordinator')")
    print("  - SomeClass.execute_agent_task (function contains 'agent')")
    print("  - SomeClass.run_handler (function contains 'handler')")
    print("  - SomeClass.invoke_action (function contains 'invoke')")
    print("\nExpected misses:")
    print("  - DataProcessor.process (no matching pattern)")
    print("  - PlainClass.do_something (no matching pattern)")


def test_stack_capture():
    """Test capture_call_stack with nested agent classes."""
    print("\n" + "=" * 70)
    print("TEST: Stack capture with nested calls")
    print("=" * 70)

    class OuterAgent:
        def run(self):
            inner = InnerWorker()
            return inner.process()

    class InnerWorker:
        def process(self):
            stack_info = capture_call_stack(skip_frames=2)
            return stack_info

    result = OuterAgent().run()
    print(f"\nNested call (OuterAgent -> InnerWorker):")
    print(f"  agent_stack: {result.agent_stack}")
    print(f"  call_site_function: {result.call_site_function}")
    if result.call_stack:
        print(f"  call_stack (first 5):")
        for entry in result.call_stack[:5]:
            print(f"    {entry}")


def test_context_manager_integration():
    """Test context manager with actual LLM calls."""
    print("\n" + "=" * 70)
    print("TEST: Context manager with real LLM calls")
    print("=" * 70)

    memory_emitter = create_memory_emitter()
    result = instrument(MeterOptions(emit_metric=memory_emitter))
    print(f"Instrumented: {result}")

    import google.generativeai as genai

    # Test 1: Explicit context
    print("\n1. With explicit context manager:")
    with agent("explicit-agent", role="tester"):
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content("Say 1")
        print(f"   Response: {response.text.strip()}")

    # Test 2: Agent-like class without context
    print("\n2. Agent-like class (ResearchAgent) without context:")
    class ResearchAgent:
        def research(self):
            model = genai.GenerativeModel("gemini-2.0-flash")
            return model.generate_content("Say 2")

    response = ResearchAgent().research()
    print(f"   Response: {response.text.strip()}")

    # Test 3: Plain class without context
    print("\n3. Plain class (DataProcessor) without context:")
    class DataProcessor:
        def process(self):
            model = genai.GenerativeModel("gemini-2.0-flash")
            return model.generate_content("Say 3")

    response = DataProcessor().process()
    print(f"   Response: {response.text.strip()}")

    # Test 4: Nested contexts
    print("\n4. Nested context managers:")
    with agent("outer"):
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content("Say 4a")
        print(f"   Outer response: {response.text.strip()}")

        with agent("inner"):
            response = model.generate_content("Say 4b")
            print(f"   Inner response: {response.text.strip()}")

    uninstrument()

    # Analyze results
    print("\n" + "-" * 40)
    print("CAPTURED EVENTS ANALYSIS:")
    print("-" * 40)

    for i, event in enumerate(memory_emitter.events):
        print(f"\nEvent {i + 1}:")
        print(f"  model: {event.model}")
        print(f"  agent_name: {event.agent_name}")
        print(f"  agent_stack: {event.agent_stack}")

    # Summary
    print("\n" + "-" * 40)
    print("AGENT NAME SUMMARY:")
    print("-" * 40)
    agent_names = [e.agent_name for e in memory_emitter.events]

    # Group by expected test
    print(f"\nTest 1 (explicit context): {agent_names[0:2]}")
    print(f"Test 2 (ResearchAgent): {agent_names[2:4]}")
    print(f"Test 3 (DataProcessor): {agent_names[4:6]}")
    print(f"Test 4 (nested): {agent_names[6:]}")

    print("\nExpected:")
    print("  Test 1: ['explicit-agent', 'explicit-agent']")
    print("  Test 2: ['ResearchAgent.research', ...] or similar")
    print("  Test 3: [None, None] (no pattern match)")
    print("  Test 4: ['outer', 'outer', 'inner', 'inner']")


def main():
    print("AGENT DETECTION HEURISTICS - COMPREHENSIVE TEST")
    print("=" * 70)

    test_frame_detection()
    test_stack_capture()
    test_context_manager_integration()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
