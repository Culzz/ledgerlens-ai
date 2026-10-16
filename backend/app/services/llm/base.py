"""The interface every provider implements.

Deliberately narrow. Anything wider would start leaking one vendor's ideas into
the rest of the codebase, which is the exact thing this exists to prevent.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from .types import Completion, Message


@runtime_checkable
class LLMProvider(Protocol):
    """A chat completion provider."""

    name: str
    model: str

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Completion:
        ...
