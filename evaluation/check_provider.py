"""Check the configured provider before committing to a long run.

    python -m evaluation.check_provider
    python -m evaluation.check_provider --provider mistral --model mistral-large-latest
    python -m evaluation.check_provider --provider mistral --list-only

Asks three questions in the order they are useful: is the configuration
complete, does the key authenticate, and does the model actually answer. An
evaluation that dies forty minutes in because the key was mistyped wastes an
evening, and this costs exactly one request.

``--provider`` checks one you have a key for without changing what the rest of
the project is configured to use, which is how a candidate gets tried before it
gets committed to. ``--list-only`` stops after the model list, because choosing
a model id costs nothing at all.

The key is never printed. It is described by length and last four characters,
which is enough to tell a wrong key from a truncated paste.

Exit code is 0 when the provider answered, 1 otherwise, so it can gate a script.
"""

from __future__ import annotations

import argparse
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
# Gemini lists two on purpose: AI Studio moved from the older "AIza" traffic keys
# to the newer "AQ." authentication keys, and both kinds are in circulation.
# A provider that is not listed here simply gets no prefix check, because a lot
# of keys have no distinctive shape and inventing one would only cry wolf.
KEY_PREFIXES: dict[str, tuple[str, ...]] = {
    "openai": ("sk-",),
    "groq": ("gsk_",),
    "gemini": ("AIza", "AQ."),
    "openrouter": ("sk-or-",),
    "nvidia": ("nvapi-",),
    "cerebras": ("csk-",),
}

# Providers whose keys have no distinctive shape, so there is nothing honest to
# match against. Listed rather than merely absent, so that adding a provider
# forces a decision about whether it belongs here or in the table above.
NO_PREFIX_CHECK: frozenset[str] = frozenset({"mistral"})

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


def check(settings: Settings, *, list_only: bool = False) -> int:
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
    if expected and not any(connection.api_key.startswith(p) for p in expected):
        print(
            f"  warning: {provider_name} keys usually start with "
            f"{' or '.join(expected)}. this one does not, which may mean it came "
            f"from somewhere else."
        )

    print("\nlisting models...")
    list_models(connection.base_url, connection.api_key)

    if list_only:
        # Listing is a GET on the model index, so it spends no quota. Stopping
        # here is how a model id gets chosen for free.
        return 0

    print("\nmaking one real call...")
    provider = build_provider(settings)
    try:
        # Deliberately no output cap. The current Gemini models spend their token
        # budget on internal reasoning before emitting anything, so a small cap
        # gets consumed there and the call comes back truncated with no content.
        completion = provider.complete([user("Reply with the single word: ready")])
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


def settings_for(settings: Settings, provider: str | None, model: str | None) -> Settings:
    """The given settings with a provider or model swapped in, if either was asked for.

    Kept separate from the argument parsing so that the override can be tested
    without reaching the network.
    """
    overrides: dict[str, str] = {}
    if provider:
        overrides["llm_provider"] = provider.strip().lower()
    if model:
        overrides["llm_model"] = model.strip()
    return settings.model_copy(update=overrides) if overrides else settings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", help="check this provider instead of the configured one")
    parser.add_argument("--model", help="check this model id instead of the configured one")
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="stop after the model list; spends no quota",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = settings_for(get_settings(), args.provider, args.model)
    return check(settings, list_only=args.list_only)


if __name__ == "__main__":
    sys.exit(main())
