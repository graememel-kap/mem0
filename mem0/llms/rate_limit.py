"""
Rate limit handling utilities for LLM providers.

This module provides decorators and utilities to handle HTTP 429 (Too Many Requests)
and other rate limiting scenarios across different LLM providers.
"""

import time
import logging
import random
from functools import wraps
from typing import Callable, Any, Optional, List
import inspect

logger = logging.getLogger(__name__)


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
    """
    Detect if an exception is a rate limit error and extract retry-after time.

    Args:
        exception: The exception to check

    Returns:
        Tuple of (is_rate_limit, retry_after_seconds)
    """
    retry_after = None

    # Check for OpenAI rate limit errors
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

    # Check for specific error types and messages
    error_str = str(exception).lower()

    # OpenAI specific errors - be more specific to avoid false positives
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

    # Anthropic specific errors
    if "rate_limit_error" in error_str or "overloaded_error" in error_str:
        return True, retry_after

    # Check for specific exception types (without importing them to avoid dependencies)
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
    """
    Decorator to add rate limit handling with exponential backoff to LLM methods.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for the first retry
        max_delay: Maximum delay in seconds between retries
        exponential_base: Base for exponential backoff calculation
        jitter: Whether to add random jitter to delays
        fallback_delay: Default delay when retry-after header is not available

    Returns:
        Decorated function with rate limit handling
    """

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
                            provider=(
                                getattr(args[0], "__class__", {}).get(
                                    "__name__", "Unknown"
                                )
                                if args
                                else "Unknown"
                            ),
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

        # Handle async functions
        if inspect.iscoroutinefunction(func):
            import asyncio

            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> Any:
                last_exception = None

                for attempt in range(max_retries + 1):
                    try:
                        return await func(*args, **kwargs)
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
                                provider=(
                                    getattr(args[0], "__class__", {}).get(
                                        "__name__", "Unknown"
                                    )
                                    if args
                                    else "Unknown"
                                ),
                            ) from e

                        # Calculate delay for next attempt
                        if retry_after is not None:
                            delay = min(retry_after, max_delay)
                        else:
                            # Exponential backoff
                            delay = min(
                                base_delay * (exponential_base**attempt), max_delay
                            )
                            if delay == max_delay:
                                delay = fallback_delay

                        # Add jitter to prevent thundering herd
                        if jitter:
                            delay = delay * (0.5 + random.random() * 0.5)

                        logger.warning(
                            f"Rate limit hit for {func.__name__} (attempt {attempt + 1}/{max_retries + 1}). "
                            f"Retrying in {delay:.2f} seconds. Error: {str(e)}"
                        )

                        await asyncio.sleep(delay)

                # This should never be reached, but just in case
                if last_exception:
                    raise last_exception

            return async_wrapper

        return wrapper

    return decorator


def get_default_rate_limit_config() -> dict:
    """
    Get default rate limit configuration.

    Returns:
        Dictionary with default rate limit settings
    """
    return {
        "max_retries": 3,
        "base_delay": 1.0,
        "max_delay": 60.0,
        "exponential_base": 2.0,
        "jitter": True,
        "fallback_delay": 5.0,
    }


class RateLimitManager:
    """
    Manager class for handling rate limits across different providers.
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize rate limit manager.

        Args:
            config: Configuration dictionary for rate limiting behavior
        """
        self.config = config or get_default_rate_limit_config()

    def apply_rate_limiting(self, func: Callable) -> Callable:
        """
        Apply rate limiting to a function.

        Args:
            func: Function to wrap with rate limiting

        Returns:
            Rate limited function
        """
        return with_rate_limit_handling(**self.config)(func)

    def update_config(self, **kwargs):
        """
        Update rate limit configuration.

        Args:
            **kwargs: Configuration parameters to update
        """
        self.config.update(kwargs)


# Provider-specific rate limit configurations
PROVIDER_CONFIGS = {
    "openai": {
        "max_retries": 3,
        "base_delay": 1.0,
        "max_delay": 60.0,
        "exponential_base": 2.0,
        "jitter": True,
        "fallback_delay": 20.0,  # OpenAI often requires longer waits
    },
    "anthropic": {
        "max_retries": 3,
        "base_delay": 2.0,
        "max_delay": 120.0,  # Anthropic can have longer rate limit windows
        "exponential_base": 2.0,
        "jitter": True,
        "fallback_delay": 30.0,
    },
    "google": {
        "max_retries": 3,
        "base_delay": 1.0,
        "max_delay": 60.0,
        "exponential_base": 2.0,
        "jitter": True,
        "fallback_delay": 10.0,
    },
    "litellm": {
        "max_retries": 4,  # LiteLLM aggregates multiple providers
        "base_delay": 1.0,
        "max_delay": 90.0,
        "exponential_base": 2.0,
        "jitter": True,
        "fallback_delay": 15.0,
    },
    "default": get_default_rate_limit_config(),
}


def get_provider_rate_limit_config(provider: str) -> dict:
    """
    Get rate limit configuration for a specific provider.

    Args:
        provider: Name of the LLM provider

    Returns:
        Rate limit configuration dictionary
    """
    return PROVIDER_CONFIGS.get(provider.lower(), PROVIDER_CONFIGS["default"])
