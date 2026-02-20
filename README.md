# Prompt Caching Optimizer SDK

A lightweight Python SDK that optimizes chat message ordering for better prefix caching and estimates token/cost savings.

## What It Does

- Classifies messages into `static`, `semi_static`, and `dynamic`
- Reorders messages to maximize cacheable prefix
- Merges adjacent compatible messages to reduce overhead
- Computes token stats (including 1024-min and 128-aligned cacheable prefix)
- Supports calibration from real OpenAI API usage (`cached_tokens`)
- Estimates per-call and monthly savings using model pricing

## Pure SDK Structure

- `app/__init__.py`: package-level public SDK exports
- `app/sdk.py`: SDK entry (`PromptCacheOptimizer`, `optimize`, calibration helpers)
- `app/core/`: optimization pipeline (analyzer, optimizer, tokenizer, calibration)
- `app/providers/`: provider strategies (`OpenAIStrategy` implemented)
- `app/models/schemas.py`: request/response models
- `tests/test.py`: local SDK usage example
- `tests/test_openai.py`: real OpenAI API + calibration integration test

## Installation

Use Python 3.11+.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install pydantic tiktoken requests
```

## Quick Start

```python
from app import PromptCacheOptimizer

payload = {
    "messages": [
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": "How do I read a file in Python?"},
        {"role": "assistant", "content": "Use open() with a context manager."},
        {"role": "user", "content": "How do I sort a list?"}
    ],
    "provider": "openai",
    "model": "gpt-4o",
    "calls_per_day": 1000
}

optimizer = PromptCacheOptimizer()
result = optimizer.optimize(payload)

print(result.token_stats)
print(result.cost_estimate)
print(result.diff.notes)
```

Or one-shot:

```python
from app import optimize

result = optimize(payload)
```

## Calibration API

```python
from app import PromptCacheOptimizer

optimizer = PromptCacheOptimizer()

# estimated_cached_tokens: SDK raw estimate
# actual_cached_tokens: OpenAI usage.prompt_tokens_details.cached_tokens
optimizer.calibrate_from_usage(
    model="gpt-4o",
    estimated_cached_tokens=6656,
    actual_cached_tokens=6144,
)

print(optimizer.get_calibration("gpt-4o"))
```

Calibration data is stored in `config/calibration.json` by default.
You can override path with env var: `CACHE_OPTIMIZER_CALIBRATION_FILE`.

## Run Scripts

```powershell
$env:PYTHONPATH='.'
.\venv\Scripts\python.exe tests\test.py
```

Real OpenAI integration test (writes calibration observations unless `--no-calibrate`):

```powershell
$env:OPENAI_API_KEY='your_key'
.\venv\Scripts\python.exe tests\test_openai.py --repeats 10
```

## Data Models

`OptimizeRequest` fields:

- `messages`: list of chat messages (`role`, `content`, optional metadata)
- `provider`: `openai` (implemented), `anthropic`, `gemini` (not yet implemented)
- `model`: model name, e.g. `gpt-4o`
- `static_prefix_count`: optional override for first N messages as static
- `calls_per_day`: optional value for monthly projection

`TokenStats` includes:

- `cache_aligned_prefix`: calibrated estimate used for cost calculation
- `raw_cache_aligned_prefix`: pre-calibration estimate
- `calibration_factor`: model-specific correction ratio (if available)

## Notes

- OpenAI pricing values are defined in `app/providers/openai_strategy.py`.
- If an unknown model is provided, pricing falls back to `gpt-4o` defaults.
- Prefix caching requires at least 1024 prefix tokens and is 128-token aligned.
