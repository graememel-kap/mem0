# HTTP 429 Rate Limiting Mitigation Implementation for mem0

## Overview

This implementation adds comprehensive HTTP 429 (Too Many Requests) rate limiting mitigation to all LLM providers in mem0. When LLM APIs return rate limit errors, the system will automatically retry with exponential backoff, significantly improving reliability and user experience.

## What Was Implemented

### 1. Core Rate Limiting Module (`mem0/llms/rate_limit.py`)

- **Rate Limit Detection**: Intelligent detection of rate limit errors across different providers
- **Exponential Backoff**: Configurable retry strategy with exponential backoff and jitter
- **Provider-Specific Configs**: Optimized settings for different LLM providers
- **Retry-After Header Support**: Respects server-provided retry timing when available

### 2. Base LLM Class Integration (`mem0/llms/base.py`)

- **Automatic Setup**: All LLM classes now automatically initialize rate limiting
- **Provider Detection**: Automatically detects provider type for optimal configuration
- **Easy Integration**: Simple `_apply_rate_limiting()` method for subclasses

### 3. LLM Provider Updates

Updated the following LLM implementations to use rate limiting:

- ✅ **OpenAI** (`openai.py`)
- ✅ **OpenAI Structured** (`openai_structured.py`)
- ✅ **Azure OpenAI** (`azure_openai.py`)
- ✅ **Azure OpenAI Structured** (`azure_openai_structured.py`)
- ✅ **Anthropic** (`anthropic.py`)
- ✅ **LiteLLM** (`litellm.py`)
- ✅ **Gemini** (`gemini.py`)
- ✅ **Groq** (`groq.py`)
- ✅ **Together** (`together.py`)
- ✅ **DeepSeek** (`deepseek.py`)
- ✅ **Sarvam** (`sarvam.py`)

## Key Features

### Smart Error Detection

```python
def is_rate_limit_error(exception: Exception) -> tuple[bool, Optional[float]]:
    """
    Detects rate limit errors from:
    - HTTP 429 status codes
    - Provider-specific error messages
    - Rate limit exception types
    - Retry-after headers
    """
```

### Configurable Retry Strategy

```python
@with_rate_limit_handling(
    max_retries=3,           # Maximum retry attempts
    base_delay=1.0,          # Base delay in seconds
    max_delay=60.0,          # Maximum delay between retries
    exponential_base=2.0,    # Exponential backoff multiplier
    jitter=True,             # Add randomness to prevent thundering herd
    fallback_delay=5.0       # Default delay when no retry-after header
)
def api_call():
    # Your LLM API call here
    pass
```

### Provider-Specific Optimizations

| Provider | Max Retries | Base Delay | Max Delay | Fallback Delay |
|----------|-------------|------------|-----------|----------------|
| OpenAI   | 3           | 1.0s       | 60.0s     | 20.0s          |
| Anthropic| 3           | 2.0s       | 120.0s    | 30.0s          |
| Google   | 3           | 1.0s       | 60.0s     | 10.0s          |
| LiteLLM  | 4           | 1.0s       | 90.0s     | 15.0s          |
| Default  | 3           | 1.0s       | 60.0s     | 5.0s           |

## How It Works

### 1. Automatic Detection

When any LLM API call is made, the rate limiting wrapper automatically:

1. **Executes the API call**
2. **Catches any exceptions**
3. **Analyzes if it's a rate limit error**
4. **If not rate limit**: Re-raises immediately
5. **If rate limit**: Starts retry logic

### 2. Retry Logic

For rate limit errors:

1. **Extract retry-after** from response headers (if available)
2. **Calculate delay** using exponential backoff or retry-after
3. **Add jitter** to prevent thundering herd effects
4. **Wait** for the calculated time
5. **Retry** the API call
6. **Repeat** until success or max retries reached

### 3. Error Handling

- **Non-rate-limit errors**: Passed through immediately
- **Exhausted retries**: Raises `RateLimitError` with context
- **Successful retry**: Returns normal response

## Usage Examples

### Memory Operations with Rate Limiting

```python
from mem0 import Memory

# Rate limiting is now automatic for all memory operations
memory = Memory()

# This will automatically retry if rate limited
memory.add("User prefers vegetarian food", user_id="alice")

# Search with automatic rate limiting
results = memory.search("food preferences", user_id="alice")
```

### Custom Configuration

```python
from mem0.configs.llms.base import BaseLlmConfig

# Custom rate limit configuration
config = BaseLlmConfig(
    model="gpt-4",
    rate_limit_config={
        'max_retries': 5,
        'base_delay': 2.0,
        'max_delay': 120.0
    }
)

memory = Memory(config={"llm": {"provider": "openai", "config": config}})
```

## Error Messages and Logging

The implementation provides detailed logging:

```
WARNING: Rate limit hit for generate_response (attempt 1/4). 
         Retrying in 2.34 seconds. Error: Rate limit exceeded

ERROR: Rate limit retry exhausted after 3 attempts for generate_response
```

## Benefits

1. **Improved Reliability**: Automatic recovery from temporary rate limits
2. **Better User Experience**: No manual retry needed
3. **Optimized Performance**: Provider-specific configurations
4. **Respectful API Usage**: Honors retry-after headers
5. **Graceful Degradation**: Clear error messages when retries exhausted

## Technical Details

### Rate Limit Detection Patterns

The system detects rate limits from:

- **HTTP Status**: 429 Too Many Requests
- **Error Messages**: "rate limit exceeded", "too many requests", "quota exceeded"
- **Exception Types**: `RateLimitError`, `TooManyRequestsError`, `QuotaExceededError`
- **Provider Patterns**: OpenAI, Anthropic, Google-specific error formats

### Exponential Backoff Formula

```
delay = min(base_delay * (exponential_base ^ attempt), max_delay)

# With jitter:
final_delay = delay * (0.5 + random() * 0.5)
```

### Memory Integration Points

Rate limiting is applied at these key points in the memory system:

1. **Fact Extraction**: When analyzing messages for facts
2. **Memory Updates**: When determining add/update/delete operations  
3. **Procedural Memory**: When generating procedural summaries
4. **All LLM Calls**: Any direct LLM API interaction

## Testing

The implementation includes comprehensive tests:

- ✅ Rate limit error detection
- ✅ Retry mechanism with backoff
- ✅ Exhausted retry handling
- ✅ Provider configuration loading
- ✅ Non-rate-limit error passthrough

## Configuration Options

### Global Configuration

Set environment variables or config:

```bash
# Provider-specific API keys (existing)
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
```

### Per-Provider Tuning

```python
from mem0.llms.rate_limit import PROVIDER_CONFIGS

# View current configurations
print(PROVIDER_CONFIGS['openai'])

# Modify for specific needs
PROVIDER_CONFIGS['openai']['max_retries'] = 5
```

## Backward Compatibility

- ✅ **No breaking changes**: Existing code continues to work
- ✅ **Automatic activation**: Rate limiting enabled by default
- ✅ **Configurable**: Can be tuned or disabled if needed
- ✅ **Transparent**: No API changes required

## Future Enhancements

Potential improvements for future versions:

1. **Circuit Breaker**: Temporary disable on repeated failures
2. **Rate Limit Metrics**: Track and report rate limiting stats
3. **Adaptive Backoff**: Learn optimal delays per provider
4. **Queue Management**: Handle concurrent requests intelligently

---

This implementation provides robust, production-ready HTTP 429 mitigation for mem0, significantly improving reliability when working with LLM APIs under load or quota constraints.
