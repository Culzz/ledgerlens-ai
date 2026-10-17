"""Build the configured provider.

One place where a provider name turns into a client, so nothing else in the
codebase needs to know which vendors exist.

Anything that speaks the OpenAI chat-completions shape is a base url and a key,
which is why the list below is short and boring. The entries earn their place by
being reachable without a credit card, because a free tier's terms are what
actually decide whether a split can be evaluated at all - and they change
without notice, so the useful check is per model rather than per provider.
Gemini's free tier allows twenty requests a day, Groq's a thousand, and
Mistral's free mode currently answers the Medium and Small chat models with a
429 while still answering the Ministral ones.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "mistral": "https://api.mistral.ai/v1",
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}

# Where to look when a model id has rotated out, which free ones regularly do.
MODEL_LIST_HINTS: dict[str, str] = {
    "openai": "https://platform.openai.com/docs/models",
    "groq": "https://console.groq.com/docs/models",
    "gemini": "https://aistudio.google.com/",
    "mistral": "https://docs.mistral.ai/getting-started/models/models_overview/",
    "nvidia": "https://build.nvidia.com/",
    "cerebras": "https://inference-docs.cerebras.ai/models/overview",
    "openrouter": "https://openrouter.ai/models?max_price=0",
}


@dataclass(frozen=True)
class Connection:
    """The endpoint, credentials and model the settings describe."""

    name: str
    base_url: str
    api_key: str
    model: str

    @property
    def redacted_key(self) -> str:
        return mask_key(self.api_key)


def mask_key(api_key: str) -> str:
    """Show enough of a key to recognise it, and not enough to use it."""
    if not api_key:
        return "(unset)"
    return "****" + api_key[-4:]


def resolve_connection(settings: Settings) -> Connection:
    """Turn the settings into an endpoint, a key and a model.

    Split out from ``build_provider`` so a diagnostic can check the parts
    without constructing a client, and so the missing-key and missing-model
    errors are raised in one place.
    """
    name = (settings.llm_provider or "").strip().lower()

    if name == "fake":
        raise ProviderNotConfigured(
            "the fake provider answers offline and has no endpoint or key to resolve"
        )

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
            f"default here - check {MODEL_LIST_HINTS.get(name, 'the provider docs')} "
            f"and put the id in .env."
        )

    return Connection(name=name, base_url=base_url, api_key=api_key, model=model)


def build_provider(settings: Settings) -> LLMProvider:
    """Construct the provider named by ``LLM_PROVIDER``."""
    if (settings.llm_provider or "").strip().lower() == "fake":
        return FakeLLM(model=settings.llm_model or "fake-model", free_tier=True)

    connection = resolve_connection(settings)

    return OpenAICompatibleProvider(
        name=connection.name,
        model=connection.model,
        base_url=connection.base_url,
        api_key=connection.api_key,
        timeout_seconds=settings.llm_request_timeout_seconds,
        max_retries=settings.llm_max_retries,
        min_seconds_between_calls=settings.llm_min_seconds_between_calls,
        max_output_tokens=settings.llm_max_output_tokens,
        free_tier=settings.llm_free_tier,
    )
