"""Message and result types shared by every provider.

Kept deliberately small and vendor-free. If a type here started looking like one
company's API, that would be the signal that the abstraction had leaked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


def system(content: str) -> Message:
    return Message("system", content)


def user(content: str) -> Message:
    return Message("user", content)


def assistant(content: str) -> Message:
    return Message("assistant", content)


@dataclass(frozen=True)
class Usage:
    """Token counts as reported by the provider.

    ``reported_total`` is here because reasoning models bill for tokens that never
    appear in the message. Gemini 3.x counts its internal thinking in
    ``total_tokens`` while leaving ``completion_tokens`` at the visible count, so
    adding those two fields together quietly understates both what a run cost and
    how much of a daily quota it consumed.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reported_total: int | None = None

    @property
    def total_tokens(self) -> int:
        if self.reported_total is not None:
            return self.reported_total
        return self.prompt_tokens + self.completion_tokens

    @property
    def reasoning_tokens(self) -> int:
        """Billed but never shown: whatever the total does not account for."""
        return max(0, self.total_tokens - self.prompt_tokens - self.completion_tokens)

    def __add__(self, other: Usage) -> Usage:
        # A sum can only claim a reported total when both sides had one.
        totals = [u.reported_total for u in (self, other)]
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            reported_total=sum(totals) if all(t is not None for t in totals) else None,
        )


@dataclass(frozen=True)
class Completion:
    """One model response, with what it cost and how long it took.

    ``cost_usd`` is ``None`` when the price is unknown, never a silent zero.
    Reporting a run as free when the truth is "we could not tell" would be worse
    than reporting nothing, and a budget ceiling built on silent zeros is not a
    ceiling.
    """

    text: str
    model: str
    provider: str
    usage: Usage = field(default_factory=Usage)
    latency_seconds: float = 0.0
    attempts: int = 1
    cost_usd: float | None = None
    # Why the model stopped. "length" means the answer was cut off, which is the
    # one failure that silently depresses accuracy if nothing records it.
    finish_reason: str | None = None

    @property
    def prompt_tokens(self) -> int:
        return self.usage.prompt_tokens

    @property
    def completion_tokens(self) -> int:
        return self.usage.completion_tokens

    @property
    def cost_known(self) -> bool:
        return self.cost_usd is not None
