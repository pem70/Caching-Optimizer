from app.core.calibration import get_calibration_ratio, record_calibration_observation
from app.core.optimizer import optimize as _optimize
from app.models.schemas import OptimizeRequest, OptimizeResponse


class PromptCacheOptimizer:
    """SDK-style interface for prompt cache optimization."""

    def optimize(self, request: OptimizeRequest | dict) -> OptimizeResponse:
        if isinstance(request, dict):
            request = OptimizeRequest(**request)
        return _optimize(request)

    def calibrate_from_usage(
        self,
        model: str,
        estimated_cached_tokens: int,
        actual_cached_tokens: int,
    ) -> dict:
        """Record real API usage and update model calibration factor."""
        return record_calibration_observation(
            model=model,
            estimated_cached_tokens=estimated_cached_tokens,
            actual_cached_tokens=actual_cached_tokens,
        )

    @staticmethod
    def get_calibration(model: str) -> float | None:
        return get_calibration_ratio(model)

    @staticmethod
    def health() -> dict[str, str]:
        return {"status": "ok"}


def optimize(request: OptimizeRequest | dict) -> OptimizeResponse:
    """Convenience function for one-shot optimization."""
    return PromptCacheOptimizer().optimize(request)


def calibrate_from_usage(model: str, estimated_cached_tokens: int, actual_cached_tokens: int) -> dict:
    """Convenience function to update calibration data."""
    return record_calibration_observation(
        model=model,
        estimated_cached_tokens=estimated_cached_tokens,
        actual_cached_tokens=actual_cached_tokens,
    )