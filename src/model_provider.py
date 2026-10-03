from __future__ import annotations

from dataclasses import dataclass


SUPPORTED_PROVIDERS = {
    "openai",
    "custom",
    "gemini",
    "anthropic",
    "ollama",
    "openrouter",
}


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration shared by all supported chat-model providers."""

    provider: str
    model_name: str
    temperature: float = 0.0
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Normalize common provider aliases and reject unsupported values."""

    normalized = value.strip().lower().replace("_", "-")
    aliases = {
        "open-ai": "openai",
        "openai-compatible": "custom",
        "openai-compatible-api": "custom",
        "google": "gemini",
        "google-gemini": "gemini",
        "anthorpic": "anthropic",
        "claude": "anthropic",
        "open-router": "openrouter",
    }
    provider = aliases.get(normalized, normalized)
    if provider not in SUPPORTED_PROVIDERS:
        choices = ", ".join(sorted(SUPPORTED_PROVIDERS))
        raise ValueError(f"Unsupported provider '{value}'. Choose one of: {choices}.")
    return provider


def build_chat_model(config: ProviderConfig):
    """Instantiate a LangChain chat model using lazy optional imports."""

    provider = normalize_provider(config.provider)
    common = {"model": config.model_name, "temperature": config.temperature}

    try:
        if provider in {"openai", "custom"}:
            from langchain_openai import ChatOpenAI

            kwargs = dict(common)
            if config.api_key:
                kwargs["api_key"] = config.api_key
            if config.base_url:
                kwargs["base_url"] = config.base_url
            return ChatOpenAI(**kwargs)

        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            kwargs = dict(common)
            if config.api_key:
                kwargs["google_api_key"] = config.api_key
            return ChatGoogleGenerativeAI(**kwargs)

        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            kwargs = dict(common)
            if config.api_key:
                kwargs["anthropic_api_key"] = config.api_key
            return ChatAnthropic(**kwargs)

        if provider == "ollama":
            from langchain_ollama import ChatOllama

            kwargs = dict(common)
            if config.base_url:
                kwargs["base_url"] = config.base_url
            return ChatOllama(**kwargs)

        if provider == "openrouter":
            from langchain_openrouter import ChatOpenRouter

            kwargs = dict(common)
            if config.api_key:
                kwargs["api_key"] = config.api_key
            return ChatOpenRouter(**kwargs)
    except ImportError as exc:
        package_by_provider = {
            "openai": "langchain-openai",
            "custom": "langchain-openai",
            "gemini": "langchain-google-genai",
            "anthropic": "langchain-anthropic",
            "ollama": "langchain-ollama",
            "openrouter": "langchain-openrouter",
        }
        package = package_by_provider[provider]
        raise RuntimeError(
            f"Provider '{provider}' requires the optional package '{package}'."
        ) from exc

    raise AssertionError(f"Unhandled provider: {provider}")
