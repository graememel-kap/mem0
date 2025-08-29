#!/usr/bin/env python3
"""
Demonstration of HTTP 429 rate limiting mitigation in mem0.

This script shows how the rate limiting works with a simulated LLM that
experiences rate limiting issues.
"""

import time
import logging
import sys
import os
from unittest.mock import Mock

# Add path to use mem0 modules
sys.path.insert(0, "/Users/graememelville/localrepos/mem0")

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimulatedRateLimitError(Exception):
    """Simulates a rate limit error from an LLM provider."""

    def __init__(self, message="Rate limit exceeded", retry_after=None):
        super().__init__(message)
        self.response = Mock()
        self.response.status_code = 429
        self.response.headers = {}
        if retry_after:
            self.response.headers["retry-after"] = str(retry_after)


class DemoLLM:
    """
    Demo LLM class that simulates rate limiting behavior.
    """

    def __init__(self, fail_count=2):
        self.call_count = 0
        self.fail_count = fail_count
        self.rate_limit_manager = None

        # Simulate rate limiting setup
        from mem0.llms.rate_limit import (
            RateLimitManager,
            get_provider_rate_limit_config,
        )

        config = get_provider_rate_limit_config("openai")
        config["base_delay"] = 0.5  # Shorter delays for demo
        config["max_delay"] = 2.0

        self.rate_limit_manager = RateLimitManager(config)

        # Apply rate limiting to our API call method
        self._original_api_call = self._make_api_call
        self._make_api_call = self.rate_limit_manager.apply_rate_limiting(
            self._make_api_call
        )

    def _make_api_call(self, prompt):
        """Simulate an API call that might be rate limited."""
        self.call_count += 1

        logger.info(
            f"Making API call #{self.call_count} for prompt: '{prompt[:50]}...'"
        )

        # Simulate rate limiting on first few calls
        if self.call_count <= self.fail_count:
            if self.call_count == 1:
                # First call fails with retry-after header
                raise SimulatedRateLimitError(
                    "Rate limit exceeded - too many requests", retry_after=1.0
                )
            else:
                # Subsequent calls fail without retry-after
                raise SimulatedRateLimitError("Rate limit exceeded - quota exhausted")

        # Success!
        time.sleep(0.1)  # Simulate API latency
        return f"Response to '{prompt}' (generated after {self.call_count} attempts)"

    def generate_response(self, prompt):
        """Public method that uses rate-limited API calls."""
        return self._make_api_call(prompt)


def demo_basic_rate_limiting():
    """Demonstrate basic rate limiting functionality."""
    logger.info("=" * 60)
    logger.info("DEMO: Basic Rate Limiting with HTTP 429 Mitigation")
    logger.info("=" * 60)

    # Create LLM that will fail twice before succeeding
    llm = DemoLLM(fail_count=2)

    start_time = time.time()
    try:
        result = llm.generate_response("What is the capital of France?")
        end_time = time.time()

        logger.info(f"✅ SUCCESS: {result}")
        logger.info(f"⏱️  Total time: {end_time - start_time:.2f} seconds")
        logger.info(f"🔄 Total API calls made: {llm.call_count}")

    except Exception as e:
        logger.error(f"❌ FAILED: {e}")


def demo_exhausted_retries():
    """Demonstrate what happens when retries are exhausted."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO: Exhausted Retries (will fail after max attempts)")
    logger.info("=" * 60)

    # Create LLM that will always fail (more failures than max retries)
    llm = DemoLLM(fail_count=10)  # Will always fail

    start_time = time.time()
    try:
        result = llm.generate_response("This will fail after retries are exhausted")
        logger.info(f"Unexpected success: {result}")

    except Exception as e:
        end_time = time.time()
        logger.error(f"❌ EXPECTED FAILURE: {e}")
        logger.info(f"⏱️  Total time: {end_time - start_time:.2f} seconds")
        logger.info(f"🔄 Total API calls made: {llm.call_count}")


def demo_different_providers():
    """Demonstrate different provider configurations."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO: Different Provider Configurations")
    logger.info("=" * 60)

    from mem0.llms.rate_limit import get_provider_rate_limit_config

    providers = ["openai", "anthropic", "litellm", "unknown_provider"]

    for provider in providers:
        config = get_provider_rate_limit_config(provider)
        logger.info(
            f"{provider.upper():15} -> max_retries: {config['max_retries']}, "
            f"fallback_delay: {config['fallback_delay']}s, "
            f"max_delay: {config['max_delay']}s"
        )


def main():
    """Run the demonstration."""
    logger.info("🚀 Starting HTTP 429 Rate Limiting Demonstration")
    logger.info(
        "This demo shows how mem0's rate limiting protects against API rate limits\n"
    )

    try:
        demo_basic_rate_limiting()
        demo_exhausted_retries()
        demo_different_providers()

        logger.info("\n" + "=" * 60)
        logger.info("🎉 DEMONSTRATION COMPLETE")
        logger.info("=" * 60)
        logger.info(
            "✅ Rate limiting is now enabled for all major LLM providers in mem0!"
        )
        logger.info("✅ Automatic retry with exponential backoff")
        logger.info("✅ Provider-specific configurations")
        logger.info("✅ Graceful handling of exhausted retries")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
