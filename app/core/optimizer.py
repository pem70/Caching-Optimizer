from app.models.schemas import (
    Provider, OptimizeRequest, OptimizeResponse,
)
from app.core.analyzer import classify_messages
from app.providers.openai_strategy import OpenAIStrategy
from app.providers.base import ProviderStrategy


def _get_strategy(provider: Provider) -> ProviderStrategy:
    strategies = {
        Provider.OPENAI: OpenAIStrategy,
    }
    cls = strategies.get(provider)
    if cls is None:
        raise ValueError(f"Provider '{provider}' not supported yet")
    return cls()


def optimize(request: OptimizeRequest) -> OptimizeResponse:
    # 1. classify
    classified = classify_messages(request.messages, request.static_prefix_count)

    # 2. optimize
    strategy = _get_strategy(request.provider)
    optimized_msgs, diff = strategy.optimize(classified, request.model)

    # 3. count tokens
    token_stats = strategy.compute_tokens(optimized_msgs, request.model)

    # 4. estimate cost
    cost = strategy.estimate_cost(token_stats, request.model)
    if request.calls_per_day:
        cost.monthly_projection = round(cost.saving_per_call * request.calls_per_day * 30, 6)

    # 5. assemble response
    return OptimizeResponse(
        original_messages=request.messages,
        optimized_messages=optimized_msgs,
        token_stats=token_stats,
        cost_estimate=cost,
        diff=diff,
        provider=request.provider,
        model=request.model,
    )