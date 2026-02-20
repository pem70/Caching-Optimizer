import json
import os
from pathlib import Path
from typing import Any

_CALIBRATION_ENV = "CACHE_OPTIMIZER_CALIBRATION_FILE"
_DEFAULT_ALPHA = 0.25


def _calibration_file() -> Path:
    configured = os.getenv(_CALIBRATION_ENV)
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[2] / "config" / "calibration.json"


def _load_data(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "models": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"version": 1, "models": {}}
    if not isinstance(data, dict):
        return {"version": 1, "models": {}}
    if "models" not in data or not isinstance(data["models"], dict):
        data["models"] = {}
    return data


def _save_data(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def get_calibration_ratio(model: str) -> float | None:
    path = _calibration_file()
    data = _load_data(path)
    model_data = data["models"].get(model)
    if not isinstance(model_data, dict):
        return None
    ratio = model_data.get("ratio_ema")
    if not isinstance(ratio, (int, float)):
        return None
    if ratio <= 0:
        return None
    return float(ratio)


def record_calibration_observation(
    model: str,
    estimated_cached_tokens: int,
    actual_cached_tokens: int,
    alpha: float = _DEFAULT_ALPHA,
) -> dict[str, Any]:
    if estimated_cached_tokens <= 0:
        raise ValueError("estimated_cached_tokens must be > 0")
    if actual_cached_tokens < 0:
        raise ValueError("actual_cached_tokens must be >= 0")
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1]")

    path = _calibration_file()
    data = _load_data(path)
    models = data["models"]

    observed_ratio = actual_cached_tokens / estimated_cached_tokens
    observed_ratio = max(0.0, min(observed_ratio, 1.25))

    existing = models.get(model)
    if isinstance(existing, dict) and isinstance(existing.get("ratio_ema"), (int, float)):
        prev = float(existing["ratio_ema"])
        ratio_ema = alpha * observed_ratio + (1 - alpha) * prev
        samples = int(existing.get("samples", 0)) + 1
    else:
        ratio_ema = observed_ratio
        samples = 1

    models[model] = {
        "ratio_ema": ratio_ema,
        "samples": samples,
        "last_observed_ratio": observed_ratio,
        "last_estimated_cached_tokens": int(estimated_cached_tokens),
        "last_actual_cached_tokens": int(actual_cached_tokens),
    }
    _save_data(path, data)

    return {
        "model": model,
        "ratio_ema": ratio_ema,
        "samples": samples,
        "observed_ratio": observed_ratio,
        "path": str(path),
    }