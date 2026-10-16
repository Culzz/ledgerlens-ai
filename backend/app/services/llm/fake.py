"""An offline provider for development and tests.

This is not a stub. It exists for two reasons that matter:

* every part of the evaluation harness can be exercised with no network, no key
  and no spend, which is the only way to test a resumable, rate-limited runner
  without burning quota on the tests themselves
* it makes ``LLM_PROVIDER=fake`` the default, so a fresh checkout runs end to
  end before anyone has signed up for anything

Token counts are estimated from character length. They are not tokenizer counts
and are not pretending to be.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from .errors import ScriptExhausted
from .pricing import estimate_cost
from .types import Completion, Message, Usage

Responder = Callable[[Sequence[Message]], str]


def estimate_tokens(text: str) -> int:
    """Rough token estimate: about four characters per token, never zero."""
    return max(1, round(len(text) / 4))


class FakeLLM:
    """A deterministic provider.

    Three modes, in order of precedence:

    * ``responder`` -- a callable, for "same answer every time"
    * ``responses`` -- a fixed script, consumed in order; running out raises,
      because silently repeating is how a scripted test passes for the wrong
      reason
    * neither -- echoes a short, clearly-fake rendering of the last user message
    """

    name = "fake"
    max_retries = 0

    def __init__(
        self,
        *,
        model: str = "fake-model",
        responses: Iterable[str] | None = None,
        responder: Responder | None = None,
        free_tier: bool = True,
    ) -> None:
        self.model = model
        self.free_tier = free_tier
        self._script: list[str] | None = list(responses) if responses is not None else None
        self._responder = responder
        self.calls: list[dict[str, object]] = []

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Completion:
        self.calls.append(
            {
                "messages": list(messages),
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        text = self._next(messages)

        prompt_text = "\n".join(message.content for message in messages)
        usage = Usage(
            prompt_tokens=estimate_tokens(prompt_text),
            completion_tokens=estimate_tokens(text),
        )
        cost = (
            0.0
            if self.free_tier
            else estimate_cost(self.model, usage.prompt_tokens, usage.completion_tokens)
        )

        return Completion(
            text=text,
            model=self.model,
            provider=self.name,
            usage=usage,
            latency_seconds=0.0,
            attempts=1,
            cost_usd=cost,
        )

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def _next(self, messages: Sequence[Message]) -> str:
        if self._responder is not None:
            return self._responder(messages)
        if self._script is not None:
            if not self._script:
                raise ScriptExhausted(
                    "the fake provider was called more times than it was given responses"
                )
            return self._script.pop(0)

        last = messages[-1].content if messages else ""
        return f"[fake] {last[:60]}"
