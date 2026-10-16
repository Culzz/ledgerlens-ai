"""Provider access over the OpenAI chat-completions shape.

OpenAI, Groq, Cerebras and OpenRouter all expose the same request and response
shape, so one client covers them and switching between them is a base url and a
key. That is what makes the provider a config value instead of a rewrite.

Built on httpx rather than a vendor SDK on purpose. The SDKs each carry their
own opinions about retries, defaults and telemetry, and the whole point here is
that behaviour is the same everywhere. httpx is also already a core dependency.

Retries distinguish two failures: a 429 or a 5xx is worth waiting out, a 400 or
a 401 is not, and retrying it just wastes the quota you were trying to protect.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Sequence
from typing import Self

import httpx

from .errors import (
    InvalidResponse,
    ProviderRateLimited,
    ProviderRequestFailed,
    ProviderUnavailable,
)
from .pricing import estimate_cost
from .rate_limit import RateLimiter
from .types import Completion, Message, Usage

HTTP_OK = 200

# Worth trying again: the request was fine, the moment was not.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

MAX_BACKOFF_SECONDS = 30.0


def retry_after_seconds(response: httpx.Response) -> float | None:
    """Read a ``Retry-After`` header, if the provider sent a usable one.

    Only the delay-seconds form is honoured; the HTTP-date form is ignored
    rather than parsed wrongly.
    """
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def _truncate(text: str, limit: int = 300) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "..."


class OpenAICompatibleProvider:
    """Chat completions against any provider that speaks the OpenAI shape."""

    def __init__(
        self,
        *,
        name: str,
        model: str,
        base_url: str,
        api_key: str,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        min_seconds_between_calls: float = 0.0,
        max_output_tokens: int | None = None,
        free_tier: bool = False,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self.name = name
        self.model = model
        self._api_key = api_key
        self.max_retries = max(0, int(max_retries))
        self.max_output_tokens = max_output_tokens
        self.free_tier = free_tier
        self._sleep = sleep
        self._clock = clock
        self._jitter = jitter
        self.base_url = base_url.rstrip("/")
        self._endpoint = f"{self.base_url}/chat/completions"
        self._timeout_seconds = timeout_seconds
        self._limiter = RateLimiter(min_seconds_between_calls, clock=clock, sleep=sleep)
        self._client = httpx.Client(timeout=timeout_seconds, transport=transport)

    # -- lifecycle ---------------------------------------------------------

    @property
    def redacted_key(self) -> str:
        """The key with everything but the last four characters hidden.

        Safe to put in a log line or an error message, which is the point.
        """
        if not self._api_key:
            return "(unset)"
        return "****" + self._api_key[-4:]

    @property
    def closed(self) -> bool:
        return self._client.is_closed

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- the interface -----------------------------------------------------

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Completion:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [message.as_dict() for message in messages],
            "temperature": temperature,
        }
        limit = max_tokens if max_tokens is not None else self.max_output_tokens
        if limit is not None:
            payload["max_tokens"] = limit

        started = self._clock()
        attempts = self.max_retries + 1

        for attempt in range(1, attempts + 1):
            self._limiter.wait()
            retry_after: float | None = None

            try:
                response = self._client.post(
                    self._endpoint,
                    json=payload,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
            except httpx.TimeoutException:
                failure: Exception = ProviderUnavailable(
                    f"{self.name} timed out after {self._timeout_seconds:g}s on attempt {attempt}"
                )
            except httpx.HTTPError as exc:
                failure = ProviderUnavailable(f"{self.name} transport error: {exc}")
            else:
                if response.status_code == HTTP_OK:
                    return self._parse(response, started=started, attempt=attempt)
                if response.status_code not in RETRYABLE_STATUS:
                    # Someone else's 400 is our bug, and repeating it costs quota.
                    raise ProviderRequestFailed(
                        f"{self.name} returned {response.status_code}: "
                        f"{_truncate(response.text)}"
                    )
                retry_after = retry_after_seconds(response)
                detail = (
                    f"{self.name} rate limited (429) on attempt {attempt}"
                    if response.status_code == 429
                    else f"{self.name} returned {response.status_code} on attempt {attempt}: "
                    f"{_truncate(response.text, 160)}"
                )
                failure = (
                    ProviderRateLimited(detail)
                    if response.status_code == 429
                    else ProviderUnavailable(detail)
                )

            if attempt == attempts:
                raise failure

            delay = retry_after if retry_after is not None else self._backoff(attempt)
            self._sleep(delay)

        # Unreachable: the loop either returns or raises. Kept for type checkers.
        raise ProviderUnavailable(f"{self.name} exhausted retries")

    # -- internals ---------------------------------------------------------

    def _backoff(self, attempt: int) -> float:
        """Exponential backoff with jitter, so concurrent runs do not sync up."""
        ceiling = min(MAX_BACKOFF_SECONDS, 2.0 ** (attempt - 1))
        return ceiling * (0.5 + 0.5 * self._jitter())

    def _parse(self, response: httpx.Response, *, started: float, attempt: int) -> Completion:
        try:
            body = response.json()
        except ValueError as exc:
            raise InvalidResponse(
                f"{self.name} returned a non-JSON body: {_truncate(response.text)}"
            ) from exc

        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise InvalidResponse(
                f"{self.name} response had no choices[0].message.content: {_truncate(str(body))}"
            ) from exc

        if not isinstance(text, str):
            raise InvalidResponse(f"{self.name} returned a non-string message content")

        usage = body.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)

        if self.free_tier:
            cost: float | None = 0.0
        elif usage:
            cost = estimate_cost(self.model, prompt_tokens, completion_tokens)
        else:
            # No usage block and no free-tier flag: we cannot claim this was free.
            cost = None

        return Completion(
            text=text,
            model=body.get("model") or self.model,
            provider=self.name,
            usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
            latency_seconds=round(self._clock() - started, 4),
            attempts=attempt,
            cost_usd=cost,
        )
