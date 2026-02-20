from enum import Enum
from pydantic import BaseModel, Field


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class Volatility(str, Enum):
    STATIC = "static"
    SEMI_STATIC = "semi_static"
    DYNAMIC = "dynamic"


class Message(BaseModel):
    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    volatility: Volatility | None = None


class OptimizeRequest(BaseModel):
    messages: list[Message]
    provider: Provider = Provider.OPENAI
    model: str = "gpt-4o"
    static_prefix_count: int | None = Field(
        None,
        description="Override: treat the first N messages as static",
    )
    calls_per_day: int | None = Field(
        None,
        description="Used for monthly cost projection",
    )


class TokenStats(BaseModel):
    total_tokens: int
    prefix_tokens: int
    dynamic_tokens: int
    cache_aligned_prefix: int
    raw_cache_aligned_prefix: int | None = None
    calibration_factor: float | None = None


class CostEstimate(BaseModel):
    price_per_1m_input: float
    price_per_1m_cached: float
    cost_without_cache: float
    cost_with_cache: float
    saving_per_call: float
    saving_percent: float
    monthly_projection: float | None = None


class OptimizeDiff(BaseModel):
    reordered: bool
    messages_merged: int
    padding_added: bool
    notes: list[str] = []


class OptimizeResponse(BaseModel):
    original_messages: list[Message]
    optimized_messages: list[Message]
    token_stats: TokenStats
    cost_estimate: CostEstimate
    diff: OptimizeDiff
    provider: Provider
    model: str