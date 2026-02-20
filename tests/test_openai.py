import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import PromptCacheOptimizer

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_REPEATS = 5
DEFAULT_MAX_TOKENS = 128


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenAI prompt cache integration test")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model name")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help="Number of repeated calls")
    parser.add_argument(
        "--cache-key",
        default=None,
        help="Optional fixed prompt_cache_key. If omitted, derived from prompt prefix.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Max completion tokens (lower to avoid TPM 429).",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Sleep between calls to reduce burstiness (e.g., 1.0 or 2.0).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=6,
        help="Retries on HTTP 429 rate limit errors.",
    )
    parser.add_argument(
        "--vary-user",
        action="store_true",
        help="Slightly vary the user message each call to test prefix caching under changing suffix.",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Disable writing calibration observations to config/calibration.json.",
    )
    return parser.parse_args()


def _require_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable.")
    return api_key


def _build_payload(model: str) -> dict:
    long_policy = " ".join(
        [
            "You are a precise coding assistant.",
            "Be concise and correct.",
            "Use clear reasoning.",
            "Prefer maintainable solutions.",
        ]
        * 320
    )

    return {
        "messages": [
            {"role": "system", "content": long_policy},
            {
                "role": "user",
                "content": "Write a Python function that returns factorial(n).",
            },
        ],
        "provider": "openai",
        "model": model,
        "calls_per_day": 1000,
    }


def _to_openai_messages(messages: list) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for m in messages:
        result.append({"role": m.role.value, "content": m.content})
    return result


def _derive_cache_key(model: str, messages: list[dict[str, str]]) -> str:
    payload = {"model": model, "messages": messages}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"caching-optimizer:{model}:{digest}"


def _maybe_wait_seconds_from_429(text: str) -> float | None:
    m = re.search(r"Please try again in ([0-9.]+)s", text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _call_openai(
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    prompt_cache_key: str,
    max_tokens: int,
    retries: int,
) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "prompt_cache_key": prompt_cache_key,
        "max_tokens": max_tokens,
    }

    last_err = None
    for attempt in range(retries + 1):
        response = requests.post(OPENAI_API_URL, headers=headers, json=body, timeout=60)

        if response.status_code < 400:
            return response.json()

        if response.status_code == 429:
            wait = _maybe_wait_seconds_from_429(response.text)
            if wait is None:
                wait = min(2**attempt, 30)
            time.sleep(wait)
            last_err = response
            continue

        raise RuntimeError(f"OpenAI API error {response.status_code}: {response.text}")

    assert last_err is not None
    raise RuntimeError(f"OpenAI API error {last_err.status_code}: {last_err.text}")


def _extract_usage(resp: dict) -> tuple[int | None, int | None, int | None, int | None]:
    usage = resp.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    prompt_details = usage.get("prompt_tokens_details") or {}
    cached_tokens = prompt_details.get("cached_tokens")
    return prompt_tokens, completion_tokens, total_tokens, cached_tokens


def _vary_user_message(messages: list[dict[str, str]], i: int) -> list[dict[str, str]]:
    copied = [dict(m) for m in messages]
    for idx in range(len(copied) - 1, -1, -1):
        if copied[idx].get("role") == "user":
            copied[idx] = dict(copied[idx])
            copied[idx]["content"] = copied[idx]["content"].rstrip() + f"\n\nTest run id: {i}"
            break
    return copied


def main() -> None:
    args = _parse_args()
    api_key = _require_api_key()

    sdk_payload = _build_payload(args.model)
    optimizer = PromptCacheOptimizer()
    optimized = optimizer.optimize(sdk_payload)
    openai_messages = _to_openai_messages(optimized.optimized_messages)

    prompt_cache_key = args.cache_key or _derive_cache_key(optimized.model, openai_messages)
    calibration_on = not args.no_calibrate

    print("model:", optimized.model)
    print("repeats:", args.repeats)
    print("prompt_cache_key:", prompt_cache_key)
    print("max_tokens:", args.max_tokens)
    print("sleep_seconds:", args.sleep_seconds)
    print("retries:", args.retries)
    print("vary_user:", args.vary_user)
    print("calibration:", calibration_on)
    print("SDK token_stats:", optimized.token_stats)
    print("SDK notes:")
    for note in optimized.diff.notes:
        print("-", note)
    print()

    raw_estimated = optimized.token_stats.raw_cache_aligned_prefix or optimized.token_stats.cache_aligned_prefix

    calls_with_cache = 0
    total_cached_tokens = 0

    for i in range(1, args.repeats + 1):
        msgs = _vary_user_message(openai_messages, i) if args.vary_user else openai_messages

        resp = _call_openai(
            api_key=api_key,
            model=optimized.model,
            messages=msgs,
            prompt_cache_key=prompt_cache_key,
            max_tokens=args.max_tokens,
            retries=args.retries,
        )
        prompt_tokens, completion_tokens, total_tokens, cached_tokens = _extract_usage(resp)

        cached = cached_tokens or 0
        total_cached_tokens += cached
        if cached > 0:
            calls_with_cache += 1

        print(f"Call #{i}")
        print("prompt_tokens:", prompt_tokens)
        print("completion_tokens:", completion_tokens)
        print("total_tokens:", total_tokens)
        print("cached_tokens:", cached_tokens)

        if calibration_on and cached_tokens is not None and raw_estimated > 0:
            calibration = optimizer.calibrate_from_usage(
                model=optimized.model,
                estimated_cached_tokens=raw_estimated,
                actual_cached_tokens=cached,
            )
            print("calibration_ratio_ema:", round(calibration["ratio_ema"], 6))
            print("calibration_samples:", calibration["samples"])

        if args.sleep_seconds and i != args.repeats:
            time.sleep(args.sleep_seconds)

    cache_hit_rate = (calls_with_cache / args.repeats * 100) if args.repeats > 0 else 0.0

    print("\nSummary")
    print("calls_with_cache:", calls_with_cache)
    print("total_calls:", args.repeats)
    print("cache_hit_rate_percent:", round(cache_hit_rate, 2))
    print("total_cached_tokens:", total_cached_tokens)

    latest_ratio = optimizer.get_calibration(optimized.model)
    print("calibration_ratio_current:", latest_ratio)


if __name__ == "__main__":
    main()