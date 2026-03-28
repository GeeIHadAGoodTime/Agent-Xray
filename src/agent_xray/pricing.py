"""Model pricing database with bundled data, local cache, and live updates.

Provides cost calculation for LLM API calls using a layered pricing lookup:
custom path > AGENT_XRAY_PRICING env var > local cache > bundled data.

Zero external dependencies -- uses only stdlib.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

_BUNDLED_PATH = Path(__file__).parent / "pricing.json"
_CACHE_DIR = Path.home() / ".agent-xray"
_CACHE_PATH = _CACHE_DIR / "pricing.json"
_REMOTE_URL = (
    "https://raw.githubusercontent.com/GeeIHadAGoodTime/Agent-Xray"
    "/main/src/agent_xray/pricing.json"
)
_CACHE_MAX_AGE_SECONDS = 7 * 24 * 3600  # 7 days

# Module-level cache so we don't re-read JSON on every call
_loaded_pricing: dict[str, Any] | None = None


def _load_json(path: Path) -> dict[str, Any]:
    """Read and parse a JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _resolve_model(name: str, data: dict[str, Any]) -> dict[str, float] | None:
    """Resolve a model name to its pricing entry.

    Tries in order: exact match, alias lookup, prefix/suffix matching.
    """
    models = data.get("models", {})
    aliases = data.get("aliases", {})

    # 1. Exact match
    if name in models:
        return models[name]

    # 2. Alias match
    resolved = aliases.get(name)
    if resolved and resolved in models:
        return models[resolved]

    # 3. Prefix match: the name in the trace may have a date suffix
    #    (e.g., "gpt-4.1-nano-2025-04-14" should match "gpt-4.1-nano")
    #    Only allow trace name to be longer than or equal to the pricing key
    #    (not shorter — "gpt-5" must NOT match "gpt-5-mini").
    #    Prefer the longest matching key to avoid "gpt-4.1" matching before "gpt-4.1-nano".
    best_key: str | None = None
    best_len = 0
    for model_key in models:
        if name.startswith(model_key) and len(model_key) > best_len:
            best_len = len(model_key)
            best_key = model_key
    if best_key is not None:
        return models[best_key]

    return None


def load_pricing(custom_path: str | None = None) -> dict[str, Any]:
    """Load pricing data with priority: custom > env var > cache > bundled.

    Args:
        custom_path: Explicit path to a pricing JSON file.

    Returns:
        Parsed pricing dictionary with ``models`` and ``aliases`` keys.
    """
    global _loaded_pricing  # noqa: PLW0603

    # Custom path always reloads (user explicitly asked for it)
    if custom_path:
        data = _load_json(Path(custom_path))
        _loaded_pricing = data
        return data

    # Env var override
    env_path = os.environ.get("AGENT_XRAY_PRICING")
    if env_path:
        data = _load_json(Path(env_path))
        _loaded_pricing = data
        return data

    # Return cached if already loaded this session
    if _loaded_pricing is not None:
        return _loaded_pricing

    # Local disk cache (if fresh)
    if _CACHE_PATH.exists():
        try:
            age = time.time() - _CACHE_PATH.stat().st_mtime
            if age < _CACHE_MAX_AGE_SECONDS:
                data = _load_json(_CACHE_PATH)
                _loaded_pricing = data
                return data
        except (OSError, json.JSONDecodeError, ValueError):
            pass  # Fall through to bundled

    # Bundled (always available)
    data = _load_json(_BUNDLED_PATH)
    _loaded_pricing = data
    return data


def get_model_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    pricing_data: dict[str, Any] | None = None,
) -> float:
    """Calculate cost for a model call in USD.

    Args:
        model_name: Model identifier from the trace.
        input_tokens: Total input tokens (including cached).
        output_tokens: Output tokens generated.
        cached_tokens: Subset of input_tokens served from cache.
        pricing_data: Pre-loaded pricing dict. Loaded automatically if None.

    Returns:
        Estimated cost in USD. Returns 0.0 if model is not found.
    """
    if pricing_data is None:
        pricing_data = load_pricing()

    entry = _resolve_model(model_name, pricing_data)
    if not entry:
        return 0.0

    input_price = entry.get("input", 0.0)
    output_price = entry.get("output", 0.0)
    cached_price = entry.get("cached_input", input_price)

    regular_input = max(0, input_tokens - cached_tokens)
    cost = (
        regular_input * input_price
        + cached_tokens * cached_price
        + output_tokens * output_price
    ) / 1_000_000

    return cost


def update_pricing_cache() -> tuple[bool, str]:
    """Fetch latest pricing from remote URL and save to local cache.

    Returns:
        Tuple of (success, message).
    """
    try:
        req = Request(_REMOTE_URL, headers={"User-Agent": "agent-xray"})
        with urlopen(req, timeout=10) as resp:  # noqa: S310
            raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict) or "models" not in data:
            return False, "Invalid pricing data: missing 'models' key."
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        model_count = len(data.get("models", {}))
        version = data.get("_meta", {}).get("version", "unknown")
        # Invalidate the in-memory cache so next load picks up new data
        global _loaded_pricing  # noqa: PLW0603
        _loaded_pricing = None
        return True, f"Updated pricing cache: {model_count} models (version {version})"
    except (URLError, OSError, json.JSONDecodeError, ValueError) as exc:
        return False, f"Failed to update: {exc}. Using existing pricing data."


def list_models(pricing_data: dict[str, Any] | None = None) -> list[str]:
    """Return sorted list of all known model names."""
    if pricing_data is None:
        pricing_data = load_pricing()
    return sorted(pricing_data.get("models", {}).keys())


def pricing_source(custom_path: str | None = None) -> str:
    """Describe which pricing source is active.

    Returns:
        Human-readable string describing the active pricing source.
    """
    if custom_path:
        return f"custom: {custom_path}"
    env_path = os.environ.get("AGENT_XRAY_PRICING")
    if env_path:
        return f"env AGENT_XRAY_PRICING: {env_path}"
    if _CACHE_PATH.exists():
        try:
            age = time.time() - _CACHE_PATH.stat().st_mtime
            if age < _CACHE_MAX_AGE_SECONDS:
                return f"cache: {_CACHE_PATH} (age: {age / 3600:.0f}h)"
        except OSError:
            pass
    return f"bundled: {_BUNDLED_PATH}"


def format_model_pricing(model_name: str, pricing_data: dict[str, Any] | None = None) -> str:
    """Format pricing info for a single model as a human-readable string."""
    if pricing_data is None:
        pricing_data = load_pricing()
    entry = _resolve_model(model_name, pricing_data)
    if not entry:
        return f"{model_name}: not found in pricing data"
    parts = [f"{model_name}:"]
    parts.append(f"  Input:  ${entry.get('input', 0.0):.4f} / 1M tokens")
    parts.append(f"  Output: ${entry.get('output', 0.0):.4f} / 1M tokens")
    if "cached_input" in entry:
        parts.append(f"  Cached: ${entry['cached_input']:.4f} / 1M tokens")
    return "\n".join(parts)


def _reset_cache() -> None:
    """Reset the in-memory pricing cache. For testing only."""
    global _loaded_pricing  # noqa: PLW0603
    _loaded_pricing = None
