#!/usr/bin/env python3
"""
Simple test for rate limiting functionality.
"""

import sys
import os
import time
import logging
from unittest.mock import Mock

# Add the mem0 directory to the path
sys.path.insert(0, "/Users/graememelville/localrepos/mem0")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_rate_limit_imports():
    """Test that we can import our rate limiting modules."""
    try:
        from mem0.llms.rate_limit import (
            with_rate_limit_handling,
            is_rate_limit_error,
            RateLimitError,
            get_provider_rate_limit_config,
        )

        logger.info("✓ Successfully imported rate limiting modules")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to import rate limiting modules: {e}")
        return False


def test_rate_limit_detection():
    """Test rate limit error detection without full imports."""
    try:
        from mem0.llms.rate_limit import is_rate_limit_error

        # Create a mock 429 error
        class MockError(Exception):
            def __init__(self, message):
                super().__init__(message)
                self.response = Mock()
                self.response.status_code = 429
                self.response.headers = {"retry-after": "30"}

        error = MockError("Rate limit exceeded")
        is_limit, retry_after = is_rate_limit_error(error)

        assert is_limit == True, "Should detect rate limit error"
        assert retry_after == 30.0, f"Should extract retry-after, got {retry_after}"

        logger.info("✓ Rate limit detection test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Rate limit detection test failed: {e}")
        return False


def test_provider_configs():
    """Test provider configurations."""
    try:
        from mem0.llms.rate_limit import get_provider_rate_limit_config

        # Test OpenAI config
        openai_config = get_provider_rate_limit_config("openai")
        assert (
            openai_config["fallback_delay"] == 20.0
        ), "OpenAI should have 20s fallback"

        # Test default config for unknown provider
        unknown_config = get_provider_rate_limit_config("unknown")
        assert unknown_config["max_retries"] == 3, "Should get default config"

        logger.info("✓ Provider configuration test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Provider configuration test failed: {e}")
        return False


def test_decorator_basic():
    """Test the basic decorator functionality."""
    try:
        from mem0.llms.rate_limit import with_rate_limit_handling

        call_count = 0

        @with_rate_limit_handling(max_retries=2, base_delay=0.01)
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                # Simulate rate limit error
                class RateError(Exception):
                    def __init__(self):
                        self.response = Mock()
                        self.response.status_code = 429
                        self.response.headers = {}

                raise RateError()
            return f"Success on attempt {call_count}"

        result = test_function()
        assert call_count == 2, f"Should have made 2 attempts, made {call_count}"
        assert "Success on attempt 2" in result, f"Unexpected result: {result}"

        logger.info("✓ Decorator test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Decorator test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("Starting rate limiting tests...")

    tests = [
        test_rate_limit_imports,
        test_rate_limit_detection,
        test_provider_configs,
        test_decorator_basic,
    ]

    passed = 0
    for test in tests:
        if test():
            passed += 1

    logger.info(f"Tests completed: {passed}/{len(tests)} passed")

    if passed == len(tests):
        logger.info("🎉 All tests passed!")
        return True
    else:
        logger.error("❌ Some tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
