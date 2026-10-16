"""Check the configured provider before committing to a long run.

    python -m evaluation.check_provider

Asks three questions in the order they are useful: is the configuration
complete, does the key authenticate, and does the model actually answer. An
evaluation that dies forty minutes in because the key was mistyped wastes an
evening, and this costs exactly one request.

The key is never printed. It is described by length and last four characters,
which is enough to tell a wrong key from a truncated paste.

Exit code is 0 when the provider answered, 1 otherwise, so it can gate a script.
"""

from __future__ import annotations

import sys

import httpx
from app.core.config import Settings, get_settings
from app.services.llm import (
    MODEL_LIST_HINTS,
    LLMError,
    ProviderNotConfigured,
    build_provider,
    mask_key,
    resolve_connection,
    user,
)

# Recognisable starts, so an obviously wrong key is caught without a request.
KEY_PREFIXES = {
    "openai": "sk-",
    "groq": "gsk_",
    "gemini": "AIza",
    "openrouter": "sk-or-",
    "cerebras": "csk-",
}

# Speech and safety models clutter the list and will never answer a question.
NOISE = ("whisper", "tts", "guard", "orpheus", "embedding", "moderation")


def list_models(base_url: str, api_key: str, *, limit: int = 40) -> None:
    """Print what the key can reach. Diagnostic only, so failures are not fatal."""
    try:
        response = httpx.get(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
        )
    except httpx.HTTPError as exc:
        print(f"  (could not list models: {type(exc).__name__}: {exc})")
        return

    if response.status_code != 200:
        print(f"  (model list returned {response.status_code}: {response.text[:160]})")
        return

    try:
        entries = response.json().get("data") or []
    except ValueError:
        print("  (model list was not JSON)")
        return

    names = [
        str(entry.get("id"))
        for entry in entries
        if entry.get("id") and not any(bad in str(entry["id"]).lower() for bad in NOISE)
    ]
    if not names:
        print("  (no chat models listed)")
        return

    print(f"  {len(names)} models reachable with this key:")
    for name in sorted(names)[:limit]:
        print(f"    {name}")


def check(settings: Settings) -> int:
    provider_name = (settings.llm_provider or "").strip().lower()
    print(f"provider:  {provider_name}")
    print(f"model:     {settings.llm_model or '(not set)'}")
    print(f"pacing:    {settings.llm_min_seconds_between_calls}s between calls")

    if provider_name == "fake":
        print("\nthe fake provider answers offline. nothing to check, and nothing to spend.")
        return 0

    try:
        connection = resolve_connection(settings)
    except ProviderNotConfigured as exc:
        print(f"\nconfiguration incomplete: {exc}")
        return 1

    print(f"endpoint:  {connection.base_url}")
    print(f"key:       {mask_key(connection.api_key)} ({len(connection.api_key)} characters)")

    expected = KEY_PREFIXES.get(provider_name)
    if expected and not connection.api_key.startswith(expected):
        print(
            f"  warning: {provider_name} keys usually start with {expected!r}. "
            f"this one does not, which may mean it came from somewhere else."
        )

    print("\nlisting models...")
    list_models(connection.base_url, connection.api_key)

    print("\nmaking one real call...")
    provider = build_provider(settings)
    try:
        completion = provider.complete(
            [user("Reply with the single word: ready")], max_tokens=64
        )
    except LLMError as exc:
        print(f"  FAILED  {type(exc).__name__}: {exc}")
        print(f"\n  if the model id is rejected, pick one from {MODEL_LIST_HINTS.get(provider_name, 'the provider docs')}")
        return 1
    finally:
        close = getattr(provider, "close", None)
        if callable(close):
            close()

    print(
        f"  ok      {completion.latency_seconds}s  "
        f"{completion.usage.total_tokens} tokens  model={completion.model}"
    )
    print(f"  reply:  {completion.text.strip()[:120]!r}")
    print(
        "\nthe provider works. run the evaluation with:\n"
        "  python -m evaluation.run_eval --split data/finqa/dev.json --system direct --limit 150"
    )
    return 0


def main() -> int:
    return check(get_settings())


if __name__ == "__main__":
    sys.exit(main())
