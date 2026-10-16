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
    """Token counts as reported by the provider."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
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

    @property
    def prompt_tokens(self) -> int:
        return self.usage.prompt_tokens

    @property
    def completion_tokens(self) -> int:
        return self.usage.completion_tokens

    @property
    def cost_known(self) -> bool:
        return self.cost_usd is not None
