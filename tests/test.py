from app import PromptCacheOptimizer

long_policy = " ".join(
        [
            "You are a precise coding assistant.",
            "Be concise and correct.",
            "Use clear reasoning.",
            "Prefer maintainable solutions.",
        ]
        * 320
    )

payload = {
    "messages": [
        {"role": "system", "content": long_policy},
        {
            "role": "user",
            "content": "Write a Python function that returns factorial(n).",
        },
    ],
    "provider": "openai",
    "model": "model",
    "calls_per_day": 1000,
}

optimizer = PromptCacheOptimizer()
result = optimizer.optimize(payload)

print("provider:", result.provider)
print("model:", result.model)
print("total_tokens:", result.token_stats.total_tokens)
print("cache_aligned_prefix:", result.token_stats.cache_aligned_prefix)
print("saving_per_call:", result.cost_estimate.saving_per_call)
print("saving_percent:", result.cost_estimate.saving_percent)
print("notes:")
for note in result.diff.notes:
    print("-", note)