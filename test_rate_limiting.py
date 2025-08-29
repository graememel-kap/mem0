#!/usr/bin/env python3
"""
Test script for HTTP 429 rate limiting mitigation in mem0 LLMs.

This script tests the rate limiting functionality by simulating rate limit errors
and verifying that the retry mechanism works correctly.
"""

import os
import sys
import time
import logging
from unittest.mock import Mock, patch
from mem0.llms.rate_limit import (
    with_rate_limit_handling,
    is_rate_limit_error,
    RateLimitError,
    get_provider_rate_limit_config,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MockRateLimitError(Exception):
    """Mock rate limit error for testing."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.response = Mock()
        self.response.status_code = 429
        self.response.headers = {}
        if retry_after:
            self.response.headers["retry-after"] = str(retry_after)


def test_rate_limit_detection():
    """Test rate limit error detection."""
    logger.info("Testing rate limit error detection...")

    # Test OpenAI-style 429 error
    error_429 = MockRateLimitError("Rate limit exceeded", retry_after=10)
    is_limit, retry_after = is_rate_limit_error(error_429)
    assert is_limit, "Should detect 429 error as rate limit"
    assert retry_after == 10.0, f"Should extract retry-after value, got {retry_after}"

    # Test error message detection
    error_msg = Exception("Rate limit exceeded. Please try again later.")
    is_limit, _ = is_rate_limit_error(error_msg)
    assert is_limit, "Should detect rate limit from error message"

    # Test non-rate-limit error
    normal_error = Exception("Connection timeout")
    is_limit, _ = is_rate_limit_error(normal_error)
    assert not is_limit, "Should not detect normal error as rate limit"

    logger.info("✓ Rate limit error detection tests passed")


def test_rate_limiting_decorator():
    """Test the rate limiting decorator."""
    logger.info("Testing rate limiting decorator...")

    # Mock function that fails with rate limit on first 2 calls, succeeds on 3rd
    call_count = 0

    @with_rate_limit_handling(max_retries=3, base_delay=0.1, max_delay=1.0)
    def mock_api_call():
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise MockRateLimitError(f"Rate limit hit (attempt {call_count})")
        return f"Success on attempt {call_count}"

    start_time = time.time()
    result = mock_api_call()
    end_time = time.time()

    assert call_count == 3, f"Should have made 3 attempts, made {call_count}"
    assert "Success on attempt 3" in result, f"Unexpected result: {result}"
    assert end_time - start_time >= 0.2, "Should have waited for retries"

    logger.info("✓ Rate limiting decorator tests passed")


def test_exhausted_retries():
    """Test behavior when retries are exhausted."""
    logger.info("Testing exhausted retries...")

    @with_rate_limit_handling(max_retries=2, base_delay=0.1)
    def always_fails():
        raise MockRateLimitError("Always rate limited")

    try:
        always_fails()
        assert False, "Should have raised RateLimitError"
    except RateLimitError as e:
        assert "exhausted" in str(e), f"Unexpected error message: {e}"
        logger.info("✓ Exhausted retries test passed")
    except Exception as e:
        assert False, f"Should have raised RateLimitError, got {type(e)}: {e}"


def test_provider_configs():
    """Test provider-specific configurations."""
    logger.info("Testing provider configurations...")

    # Test known providers
    openai_config = get_provider_rate_limit_config("openai")
    assert (
        openai_config["fallback_delay"] == 20.0
    ), "OpenAI should have 20s fallback delay"

    anthropic_config = get_provider_rate_limit_config("anthropic")
    assert (
        anthropic_config["max_delay"] == 120.0
    ), "Anthropic should have 120s max delay"

    # Test unknown provider (should get default)
    unknown_config = get_provider_rate_limit_config("unknown_provider")
    assert (
        unknown_config["max_retries"] == 3
    ), "Unknown provider should get default config"

    logger.info("✓ Provider configuration tests passed")


def test_integration_with_mock_llm():
    """Test integration with a mock LLM class."""
    logger.info("Testing integration with mock LLM...")

    from mem0.llms.base import LLMBase
    from mem0.configs.llms.base import BaseLlmConfig

    class MockLLM(LLMBase):
        def __init__(self):
            config = BaseLlmConfig(model="test-model")
            super().__init__(config)
            self.call_count = 0

        def generate_response(self, messages, **kwargs):
            """Mock implementation that rate limits on first call."""
            self.call_count += 1
            if self.call_count == 1:
                raise MockRateLimitError("Rate limited on first call")
            return f"Response after {self.call_count} attempts"

    # Test the mock LLM
    llm = MockLLM()

    # Apply rate limiting to generate_response
    original_method = llm.generate_response
    llm.generate_response = llm._apply_rate_limiting(original_method)

    # Test that it retries and succeeds
    result = llm.generate_response([{"role": "user", "content": "test"}])
    assert "Response after 2 attempts" in result, f"Unexpected result: {result}"
    assert llm.call_count == 2, f"Should have made 2 calls, made {llm.call_count}"

    logger.info("✓ Integration test passed")


def main():
    """Run all tests."""
    logger.info("Starting rate limiting tests...")

    try:
        test_rate_limit_detection()
        test_rate_limiting_decorator()
        test_exhausted_retries()
        test_provider_configs()
        test_integration_with_mock_llm()

        logger.info("🎉 All tests passed!")
        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
