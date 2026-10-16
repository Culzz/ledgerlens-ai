"""What models cost, so a run can report spend instead of guessing.

Prices are USD per million tokens. They change, so the table below is
deliberately short: every entry is one that was actually checked, and anything
missing returns ``None`` rather than a confident zero.

Since the default configuration runs on free tiers, the usual answer is that
cost is zero by policy, not by price. See ``Settings.llm_free_tier``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input_per_million: float
    output_per_million: float


# Checked against provider pricing pages in October 2026.
PRICES: dict[str, Price] = {
    "gpt-4o": Price(2.50, 10.00),
    "gpt-4o-mini": Price(0.15, 0.60),
    "gemini-2.5-flash": Price(0.30, 2.50),
    # Nothing leaves the machine.
    "fake-model": Price(0.0, 0.0),
}


def price_for(model: str) -> Price | None:
    """Look up a model's price, tolerating a vendor prefix.

    Providers disagree about whether a model id carries the vendor, so
    ``openai/gpt-4o`` and ``gpt-4o`` should resolve the same way.
    """
    if not model:
        return None
    if model in PRICES:
        return PRICES[model]
    if "/" in model:
        return PRICES.get(model.split("/", 1)[1])
    return None


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """Estimated USD for one call, or ``None`` if the model is not priced."""
    price = price_for(model)
    if price is None:
        return None
    return round(
        prompt_tokens / 1_000_000 * price.input_per_million
        + completion_tokens / 1_000_000 * price.output_per_million,
        6,
    )
