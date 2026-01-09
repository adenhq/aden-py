"""
Large Content Capture Example

Demonstrates content capture with large payloads that exceed max_content_bytes.
When content is too large, it's stored separately and a ContentReference is kept.

Run: python examples/large_content_capture.py
"""

import json
import os
import sys

# Add parent directory to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from aden import (
    instrument,
    uninstrument,
    MeterOptions,
    MetricEvent,
    ContentCaptureOptions,
)
from aden.types import ContentReference


def create_capture_emitter():
    """Create an emitter that displays captured content with reference detection."""

    def emit(event: MetricEvent) -> None:
        print("\n" + "=" * 70)
        print(f"METRIC EVENT: {event.provider} / {event.model}")
        print(f"  Trace: {event.trace_id[:8]}... | Latency: {event.latency_ms:.0f}ms")
        print(f"  Tokens: {event.input_tokens} in / {event.output_tokens} out")

        if event.content_capture:
            cc = event.content_capture
            print("\n  [Layer 0: Content Capture]")

            # System prompt
            if cc.system_prompt:
                if isinstance(cc.system_prompt, ContentReference):
                    print(f"    System Prompt: <LARGE CONTENT REFERENCE>")
                    print(f"      - content_id: {cc.system_prompt.content_id}")
                    print(f"      - byte_size: {cc.system_prompt.byte_size:,} bytes")
                    print(f"      - preview: {cc.system_prompt.truncated_preview[:60]}...")
                else:
                    print(f"    System Prompt: {cc.system_prompt[:80]}{'...' if len(cc.system_prompt) > 80 else ''}")

            # Messages
            if cc.messages:
                if isinstance(cc.messages, ContentReference):
                    print(f"    Messages: <LARGE CONTENT REFERENCE>")
                    print(f"      - content_id: {cc.messages.content_id}")
                    print(f"      - byte_size: {cc.messages.byte_size:,} bytes")
                else:
                    print(f"    Messages: {len(cc.messages)} messages")
                    for msg in cc.messages[:2]:
                        if isinstance(msg.content, ContentReference):
                            print(f"      [{msg.role}]: <LARGE CONTENT> ({msg.content.byte_size:,} bytes)")
                        elif msg.content:
                            preview = msg.content[:50]
                            print(f"      [{msg.role}]: {preview}{'...' if len(msg.content) > 50 else ''}")

            # Response
            if cc.response_content:
                if isinstance(cc.response_content, ContentReference):
                    print(f"    Response: <LARGE CONTENT REFERENCE>")
                    print(f"      - content_id: {cc.response_content.content_id}")
                    print(f"      - byte_size: {cc.response_content.byte_size:,} bytes")
                    print(f"      - preview: {cc.response_content.truncated_preview[:60]}...")
                else:
                    print(f"    Response: {cc.response_content[:80]}{'...' if len(cc.response_content) > 80 else ''}")

            if cc.finish_reason:
                print(f"    Finish Reason: {cc.finish_reason}")

        print("=" * 70)

    return emit


def test_large_system_prompt() -> None:
    """Test with a large system prompt that exceeds max_content_bytes."""
    try:
        from anthropic import Anthropic
    except ImportError:
        print("Anthropic SDK not installed, skipping...")
        return

    print("\n### Test 1: Large System Prompt ###")
    print("(System prompt > 1KB will be stored as ContentReference)")

    client = Anthropic()

    # Create a large system prompt (> 1KB)
    large_system_prompt = """You are an expert assistant with deep knowledge in multiple domains.

DOMAIN EXPERTISE:
1. Software Engineering: You have extensive experience in software architecture, design patterns,
   clean code principles, test-driven development, continuous integration, and DevOps practices.
   You understand multiple programming paradigms including object-oriented, functional, and
   reactive programming. You're familiar with microservices, event-driven architectures, and
   distributed systems design.

2. Data Science & Machine Learning: You understand statistical methods, machine learning algorithms,
   deep learning architectures, natural language processing, computer vision, and reinforcement
   learning. You can explain concepts like gradient descent, backpropagation, attention mechanisms,
   transformer architectures, and various optimization techniques.

3. Cloud Computing: You're proficient with AWS, GCP, and Azure services. You understand concepts
   like auto-scaling, load balancing, container orchestration with Kubernetes, serverless
   computing, and infrastructure as code with Terraform or CloudFormation.

4. Security: You understand OWASP top 10, encryption, authentication protocols, OAuth/OIDC,
   zero-trust architecture, and security best practices for web applications and APIs.

COMMUNICATION STYLE:
- Be concise but thorough
- Use examples when helpful
- Acknowledge uncertainty when appropriate
- Provide actionable recommendations
"""

    print(f"System prompt size: {len(large_system_prompt.encode('utf-8')):,} bytes")

    response = client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=100,
        system=large_system_prompt,
        messages=[{"role": "user", "content": "What is Docker in one sentence?"}],
    )
    print(f"Response: {response.content[0].text}")


def test_large_conversation() -> None:
    """Test with a large conversation history."""
    try:
        from anthropic import Anthropic
    except ImportError:
        print("Anthropic SDK not installed, skipping...")
        return

    print("\n### Test 2: Large Conversation History ###")
    print("(Messages array > 1KB will be stored as ContentReference)")

    client = Anthropic()

    # Create a conversation with multiple long messages
    messages = [
        {
            "role": "user",
            "content": """I'm building a complex e-commerce platform and need advice on the architecture.

Here are my requirements:
1. Handle 10,000+ concurrent users
2. Support multiple payment gateways (Stripe, PayPal, Apple Pay)
3. Real-time inventory management across multiple warehouses
4. Personalized product recommendations
5. Multi-language and multi-currency support
6. Integration with shipping providers (FedEx, UPS, DHL)
7. Advanced search with faceted filtering
8. User reviews and ratings system
9. Wishlist and save for later functionality
10. Admin dashboard with analytics"""
        },
        {
            "role": "assistant",
            "content": """For your e-commerce platform, I recommend a microservices architecture with the following components:

1. **API Gateway**: Kong or AWS API Gateway for routing, rate limiting, and authentication
2. **User Service**: Handle authentication, profiles, wishlists
3. **Product Service**: Catalog management, search (Elasticsearch)
4. **Order Service**: Order processing, state management
5. **Payment Service**: Multi-gateway integration with idempotency
6. **Inventory Service**: Real-time stock management with event sourcing
7. **Recommendation Service**: ML-based personalization
8. **Notification Service**: Email, SMS, push notifications

For the tech stack:
- Backend: Node.js or Go for high concurrency
- Database: PostgreSQL for transactions, Redis for caching
- Message Queue: Apache Kafka for event streaming
- Search: Elasticsearch with custom analyzers
- Containerization: Kubernetes for orchestration"""
        },
        {
            "role": "user",
            "content": "How should I handle the payment processing to ensure PCI compliance?"
        }
    ]

    total_size = sum(len(json.dumps(m).encode('utf-8')) for m in messages)
    print(f"Messages total size: {total_size:,} bytes")

    response = client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=150,
        messages=messages,
    )
    print(f"Response: {response.content[0].text[:200]}...")


def test_large_response() -> None:
    """Test that generates a large response."""
    try:
        from anthropic import Anthropic
    except ImportError:
        print("Anthropic SDK not installed, skipping...")
        return

    print("\n### Test 3: Large Response (Anthropic) ###")
    print("(Response > 1KB will be stored as ContentReference)")

    client = Anthropic()

    response = client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=1000,  # Allow for longer response
        messages=[{
            "role": "user",
            "content": "Write a detailed explanation of how OAuth 2.0 works, including all grant types and security considerations."
        }],
    )

    response_text = response.content[0].text
    print(f"Response size: {len(response_text.encode('utf-8')):,} bytes")
    print(f"Response preview: {response_text[:150]}...")


def test_gemini_large_content() -> None:
    """Test large content with Gemini (google-generativeai)."""
    try:
        import google.generativeai as genai
    except ImportError:
        print("Gemini SDK (google-generativeai) not installed, skipping...")
        return

    print("\n### Test 4: Large Content (Gemini) ###")
    print("(System instruction > 1KB will be stored as ContentReference)")

    # Create model with large system instruction
    large_system = """You are an expert software architect with comprehensive knowledge across multiple domains.

YOUR EXPERTISE INCLUDES:

1. BACKEND DEVELOPMENT:
   - Microservices architecture and distributed systems
   - API design (REST, GraphQL, gRPC)
   - Database optimization (SQL, NoSQL, NewSQL)
   - Message queues and event-driven architecture
   - Caching strategies (Redis, Memcached)

2. FRONTEND DEVELOPMENT:
   - React, Vue, Angular frameworks
   - State management patterns
   - Performance optimization
   - Accessibility standards (WCAG)
   - Progressive Web Apps

3. DEVOPS & INFRASTRUCTURE:
   - Container orchestration (Kubernetes, Docker Swarm)
   - CI/CD pipelines
   - Infrastructure as Code (Terraform, Pulumi)
   - Cloud platforms (AWS, GCP, Azure)
   - Monitoring and observability

4. SECURITY:
   - Authentication and authorization
   - Encryption and key management
   - Secure coding practices
   - Compliance (SOC2, GDPR, HIPAA)
"""

    print(f"System instruction size: {len(large_system.encode('utf-8')):,} bytes")

    model = genai.GenerativeModel(
        "gemini-2.0-flash",
        system_instruction=large_system,
    )

    response = model.generate_content("What's the best approach for API versioning?")
    print(f"Response: {response.text[:150]}...")


def main() -> None:
    """Run large content capture examples."""
    print("=" * 70)
    print("Large Content Capture Example")
    print("=" * 70)
    print("\nContent larger than max_content_bytes (1KB) will be stored")
    print("as ContentReference with just the ID, hash, size, and preview.")

    # Use a small max_content_bytes to trigger large content handling
    result = instrument(
        MeterOptions(
            emit_metric=create_capture_emitter(),
            capture_content=True,
            content_capture_options=ContentCaptureOptions(
                max_content_bytes=1024,  # 1KB threshold - content larger than this becomes a reference
                capture_system_prompt=True,
                capture_messages=True,
                capture_tools_schema=True,
                capture_response=True,
            ),
        )
    )
    print(f"\nInstrumented: anthropic={result.anthropic}, gemini={result.gemini}")

    try:
        test_large_system_prompt()
        test_large_conversation()
        test_large_response()
        test_gemini_large_content()
    finally:
        uninstrument()

    print("\n" + "=" * 70)
    print("Large content capture tests complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
