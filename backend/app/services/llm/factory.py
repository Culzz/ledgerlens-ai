"""Build the configured provider.

One place where a provider name turns into a client, so nothing else in the
codebase needs to know which vendors exist.
"""

from __future__ import annotations

from app.core.config import Settings

from .base import LLMProvider
from .errors import ProviderNotConfigured
from .fake import FakeLLM
from .openai_compatible import OpenAICompatibleProvider

# Providers that speak the OpenAI chat-completions shape. The base url is the
# only thing that actually differs between them.
COMPATIBLE_ENDPOINTS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}

# Where to look when the free model list has rotated again.
MODEL_LIST_HINTS: dict[str, str] = {
    "openai": "https://platform.openai.com/docs/models",
    "groq": "https://console.groq.com/docs/models",
    "cerebras": "https://inference-docs.cerebras.ai/models/overview",
    "openrouter": "https://openrouter.ai/models?max_price=0",
}


def build_provider(settings: Settings) -> LLMProvider:
    """Construct the provider named by ``LLM_PROVIDER``."""
    name = (settings.llm_provider or "").strip().lower()

    if name == "fake":
        return FakeLLM(model=settings.llm_model or "fake-model", free_tier=True)

    if name not in COMPATIBLE_ENDPOINTS:
        known = ", ".join(["fake", *sorted(COMPATIBLE_ENDPOINTS)])
        raise ProviderNotConfigured(f"unknown provider {name!r}; known providers: {known}")

    base_url = settings.llm_base_url.strip() or COMPATIBLE_ENDPOINTS[name]

    api_key = settings.llm_api_key.strip() or str(
        getattr(settings, f"{name}_api_key", "") or ""
    ).strip()
    if not api_key:
        raise ProviderNotConfigured(
            f"{name} needs an API key. Set LLM_API_KEY, or {name.upper()}_API_KEY, in .env."
        )

    model = settings.llm_model.strip()
    if not model:
        # Not defaulted on purpose. Free model ids get rotated out from under
        # you, and a stale default fails in a way that looks like a code bug.
        raise ProviderNotConfigured(
            f"{name} needs LLM_MODEL set. Model ids change often, so there is no "
            f"default here — check {MODEL_LIST_HINTS.get(name, 'the provider docs')} "
            f"and put the id in .env."
        )

    return OpenAICompatibleProvider(
        name=name,
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=settings.llm_request_timeout_seconds,
        max_retries=settings.llm_max_retries,
        min_seconds_between_calls=settings.llm_min_seconds_between_calls,
        max_output_tokens=settings.llm_max_output_tokens,
        free_tier=settings.llm_free_tier,
    )
