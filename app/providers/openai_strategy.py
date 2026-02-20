from app.models.schemas import (
    Message, Volatility, TokenStats, CostEstimate, OptimizeDiff,
)
from app.providers.base import ProviderStrategy
from app.core.calibration import get_calibration_ratio
from app.core.tokenizer import count_message_tokens, count_messages_tokens

CACHE_MIN_TOKENS = 1024
CACHE_ALIGNMENT = 128

# Pricing ($ per 1M tokens)
# source: https://platform.openai.com/docs/pricing
OPENAI_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "cached": 1.25},
    "gpt-4o-mini": {"input": 0.15, "cached": 0.075},
    "gpt-4.1": {"input": 2.00, "cached": 0.50},
    "gpt-4.1-mini": {"input": 0.40, "cached": 0.10},
    "gpt-4.1-nano": {"input": 0.10, "cached": 0.025},
    "o3": {"input": 10.00, "cached": 2.50},
    "o3-mini": {"input": 1.10, "cached": 0.55},
    "o4-mini": {"input": 1.10, "cached": 0.275},
}


def _align_down(n: int, alignment: int) -> int:
    return (n // alignment) * alignment


class OpenAIStrategy(ProviderStrategy):

    @staticmethod
    def _reorder(messages: list[Message]) -> tuple[list[Message], bool]:
        """Sort by volatility: STATIC -> SEMI_STATIC -> DYNAMIC."""
        order = {Volatility.STATIC: 0, Volatility.SEMI_STATIC: 1, Volatility.DYNAMIC: 2}
        reordered = sorted(messages, key=lambda m: order.get(m.volatility, 2))

        changed = [m.content for m in reordered] != [m.content for m in messages]
        return reordered, changed

    @staticmethod
    def _merge_adjacent(messages: list[Message]) -> tuple[list[Message], int]:
        """Merge adjacent messages with same role + same volatility (static/semi only)."""
        if not messages:
            return messages, 0

        merged: list[Message] = [messages[0].model_copy()]
        merge_count = 0

        for msg in messages[1:]:
            prev = merged[-1]
            can_merge = (
                prev.role == msg.role
                and prev.volatility == msg.volatility
                and msg.volatility in (Volatility.STATIC, Volatility.SEMI_STATIC)
            )
            if can_merge:
                merged[-1] = prev.model_copy(
                    update={"content": prev.content + "\n\n" + msg.content}
                )
                merge_count += 1
            else:
                merged.append(msg.model_copy())

        return merged, merge_count

    @staticmethod
    def _compute_prefix_length(messages: list[Message], model: str) -> int:
        """Count tokens in the cacheable prefix before the first dynamic message."""
        prefix_tokens = 0
        for msg in messages:
            if msg.volatility in (Volatility.STATIC, Volatility.SEMI_STATIC):
                prefix_tokens += count_message_tokens(msg, model)
            else:
                break
        return prefix_tokens

    def optimize(self, messages: list[Message], model: str) -> tuple[list[Message], OptimizeDiff]:
        notes: list[str] = []

        reordered, did_reorder = self._reorder(messages)
        if did_reorder:
            notes.append("Reordered messages: static -> semi-static -> dynamic")

        merged, merge_count = self._merge_adjacent(reordered)
        if merge_count > 0:
            notes.append(f"Merged {merge_count} adjacent same-role messages")

        prefix_tokens = self._compute_prefix_length(merged, model)

        if prefix_tokens == 0:
            notes.append("No cacheable prefix found (no static/semi-static messages)")
        elif prefix_tokens < CACHE_MIN_TOKENS:
            notes.append(
                f"Prefix is {prefix_tokens} tokens, below {CACHE_MIN_TOKENS} minimum; "
                "cache will not activate. Add more static content to reach the threshold."
            )
        else:
            aligned = _align_down(prefix_tokens, CACHE_ALIGNMENT)
            wasted = prefix_tokens - aligned
            notes.append(
                f"Prefix: {prefix_tokens} tokens -> estimated cached: {aligned} tokens "
                f"(128-aligned, {wasted} tokens past boundary not cached)"
            )

        diff = OptimizeDiff(
            reordered=did_reorder,
            messages_merged=merge_count,
            padding_added=False,
            notes=notes,
        )
        return merged, diff

    def compute_tokens(self, messages: list[Message], model: str) -> TokenStats:
        total = count_messages_tokens(messages, model)
        prefix = self._compute_prefix_length(messages, model)
        raw_aligned = _align_down(prefix, CACHE_ALIGNMENT) if prefix >= CACHE_MIN_TOKENS else 0

        calibration_factor = get_calibration_ratio(model)
        calibrated_aligned = raw_aligned
        if calibration_factor is not None and raw_aligned > 0:
            calibrated_aligned = _align_down(int(raw_aligned * calibration_factor), CACHE_ALIGNMENT)

        return TokenStats(
            total_tokens=total,
            prefix_tokens=prefix,
            dynamic_tokens=total - prefix,
            cache_aligned_prefix=calibrated_aligned,
            raw_cache_aligned_prefix=raw_aligned,
            calibration_factor=calibration_factor,
        )

    def estimate_cost(self, token_stats: TokenStats, model: str) -> CostEstimate:
        pricing = OPENAI_PRICING.get(model, OPENAI_PRICING["gpt-4o"])

        input_price = pricing["input"]
        cached_price = pricing["cached"]
        cached_tokens = token_stats.cache_aligned_prefix
        uncached_tokens = token_stats.total_tokens - cached_tokens

        cost_without = token_stats.total_tokens * input_price / 1_000_000
        cost_with = (
            uncached_tokens * input_price / 1_000_000
            + cached_tokens * cached_price / 1_000_000
        )
        saving = cost_without - cost_with
        saving_pct = (saving / cost_without * 100) if cost_without > 0 else 0.0

        return CostEstimate(
            price_per_1m_input=input_price,
            price_per_1m_cached=cached_price,
            cost_without_cache=round(cost_without, 8),
            cost_with_cache=round(cost_with, 8),
            saving_per_call=round(saving, 8),
            saving_percent=round(saving_pct, 2),
        )