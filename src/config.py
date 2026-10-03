from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass(frozen=True)
class LabConfig:
    """Shared paths, memory settings, and model settings for the lab."""

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}.") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}.") from exc


def _provider_api_key(provider: str, prefix: str = "") -> str | None:
    names = {
        "openai": "OPENAI_API_KEY",
        "custom": "CUSTOM_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "ollama": "OLLAMA_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    return os.getenv(f"{prefix}{names[provider]}") or os.getenv(names[provider])


def _provider_base_url(provider: str, prefix: str = "") -> str | None:
    names = {
        "openai": "OPENAI_BASE_URL",
        "custom": "CUSTOM_BASE_URL",
        "gemini": "GEMINI_BASE_URL",
        "anthropic": "ANTHROPIC_BASE_URL",
        "ollama": "OLLAMA_BASE_URL",
        "openrouter": "OPENROUTER_BASE_URL",
    }
    value = os.getenv(f"{prefix}{names[provider]}") or os.getenv(names[provider])
    if provider == "openrouter" and not value:
        return "https://openrouter.ai/api/v1"
    if provider == "ollama" and not value:
        return "http://localhost:11434"
    return value


def _build_provider_config(prefix: str = "") -> ProviderConfig:
    provider_value = (
        os.getenv(f"{prefix}LLM_PROVIDER")
        or os.getenv("LLM_PROVIDER")
        or "openai"
    )
    provider = normalize_provider(provider_value)
    default_models = {
        "openai": "gpt-4o-mini",
        "custom": "gpt-4o-mini",
        "gemini": "gemini-2.0-flash",
        "anthropic": "claude-3-5-haiku-latest",
        "ollama": "llama3.2",
        "openrouter": "openai/gpt-4o-mini",
    }
    model_name = (
        os.getenv(f"{prefix}LLM_MODEL")
        or os.getenv("LLM_MODEL")
        or default_models[provider]
    )
    temperature = _env_float(
        f"{prefix}LLM_TEMPERATURE",
        _env_float("LLM_TEMPERATURE", 0.0),
    )
    return ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=_provider_api_key(provider, prefix),
        base_url=_provider_base_url(provider, prefix),
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load `.env`, create state storage, and return a complete configuration."""

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env", override=False)
    except ImportError:
        pass

    state_dir = Path(os.getenv("MEMORY_STATE_DIR", root / "state")).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=_env_int("COMPACT_THRESHOLD_TOKENS", 1_600, 32),
        compact_keep_messages=_env_int("COMPACT_KEEP_MESSAGES", 4, 1),
        model=_build_provider_config(),
        judge_model=_build_provider_config("JUDGE_"),
    )
