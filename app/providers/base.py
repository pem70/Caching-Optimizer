from abc import ABC, abstractmethod

from app.models.schemas import Message, TokenStats, CostEstimate, OptimizeDiff


class ProviderStrategy(ABC):

    @abstractmethod
    def optimize(self, messages: list[Message], model: str) -> tuple[list[Message], OptimizeDiff]:
        ...

    @abstractmethod
    def compute_tokens(self, messages: list[Message], model: str) -> TokenStats:
        ...

    @abstractmethod
    def estimate_cost(self, token_stats: TokenStats, model: str) -> CostEstimate:
        ...