#!/usr/bin/env python3
"""
Direct test of rate limiting functionality without package imports.
"""

import time
import logging
import random
from functools import wraps
from typing import Callable, Any, Optional
from unittest.mock import Mock

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Copy the core functionality directly to test it
class RateLimitError(Exception):
    """Custom exception for rate limit errors."""

    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = None,
        provider: Optional[str] = None,
    ):
        super().__init__(message)
        self.retry_after = retry_after
        self.provider = provider


def is_rate_limit_error(exception: Exception) -> tuple[bool, Optional[float]]:
    """Detect if an exception is a rate limit error and extract retry-after time."""
    retry_after = None

    # Check for HTTP 429 errors
    if hasattr(exception, "response") and hasattr(exception.response, "status_code"):
        if exception.response.status_code == 429:
            # Try to extract retry-after header
            if hasattr(exception.response, "headers"):
                retry_after_header = exception.response.headers.get("retry-after")
                if retry_after_header:
                    try:
                        retry_after = float(retry_after_header)
                    except ValueError:
                        pass
            return True, retry_after

    # Check for specific error types and messages - be more specific to avoid false positives
    error_str = str(exception).lower()

    if any(
        phrase in error_str
        for phrase in [
            "rate limit exceeded",
            "too many requests",
            "quota exceeded",
            "rate_limit_exceeded",
            "requests per minute exceeded",
        ]
    ):
        return True, retry_after

    # Check for standalone '429' without other context
    if " 429 " in error_str or error_str.startswith("429") or error_str.endswith("429"):
        return True, retry_after

    # Check for specific exception types
    exception_name = exception.__class__.__name__
    if exception_name in [
        "RateLimitError",
        "TooManyRequestsError",
        "QuotaExceededError",
    ]:
        return True, retry_after

    return False, None


def with_rate_limit_handling(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    fallback_delay: float = 5.0,
) -> Callable:
    """Decorator to add rate limit handling with exponential backoff."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    is_rate_limit, retry_after = is_rate_limit_error(e)

                    if not is_rate_limit:
                        # Not a rate limit error, re-raise immediately
                        raise e

                    last_exception = e

                    if attempt == max_retries:
                        # Last attempt, re-raise the exception
                        logger.error(
                            f"Rate limit retry exhausted after {max_retries} attempts for {func.__name__}"
                        )
                        raise RateLimitError(
                            f"Rate limit retry exhausted after {max_retries} attempts: {str(e)}",
                            retry_after=retry_after,
                            provider="test",
                        ) from e

                    # Calculate delay for next attempt
                    if retry_after is not None:
                        delay = min(retry_after, max_delay)
                    else:
                        # Exponential backoff
                        delay = min(base_delay * (exponential_base**attempt), max_delay)
                        if delay == max_delay:
                            delay = fallback_delay

                    # Add jitter to prevent thundering herd
                    if jitter:
                        delay = delay * (0.5 + random.random() * 0.5)

                    logger.warning(
                        f"Rate limit hit for {func.__name__} (attempt {attempt + 1}/{max_retries + 1}). "
                        f"Retrying in {delay:.2f} seconds. Error: {str(e)}"
                    )

                    time.sleep(delay)

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception

        return wrapper

    return decorator


# Mock rate limit error for testing
class MockRateLimitError(Exception):
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

    # Test 429 error with retry-after
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


def test_non_rate_limit_error():
    """Test that non-rate-limit errors are passed through immediately."""
    logger.info("Testing non-rate-limit error passthrough...")

    @with_rate_limit_handling(max_retries=3, base_delay=0.1)
    def non_rate_limit_error():
        raise ValueError("This is not a rate limit error")

    try:
        non_rate_limit_error()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "not a rate limit error" in str(e), f"Unexpected error: {e}"
        logger.info("✓ Non-rate-limit error passthrough test passed")
    except Exception as e:
        assert False, f"Should have raised ValueError, got {type(e)}: {e}"


def main():
    """Run all tests."""
    logger.info("Starting direct rate limiting tests...")

    try:
        test_rate_limit_detection()
        test_rate_limiting_decorator()
        test_exhausted_retries()
        test_non_rate_limit_error()

        logger.info("🎉 All direct rate limiting tests passed!")
        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    import sys

    sys.exit(0 if success else 1)
