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

# Where the wait starts. A 503 is usually a capacity spike that clears in
# seconds, and a first retry of one second is spent inside the spike rather than
# past it - three attempts then burn in under seven seconds and the run records a
# failure that another moment would have answered.
BASE_BACKOFF_SECONDS = 2.0


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


# A provider asking for longer than this is not asking us to retry, it is telling
# us the quota is spent until later. Sitting through it would hold a terminal for
# hours, so the run stops and says when to come back.
MAX_RETRY_AFTER_SECONDS = 300.0


def _body_retry_delay(response: httpx.Response) -> float | None:
    """The provider's own estimate from the body, for when there is no header.

    Gemini reports a spent daily quota here rather than in ``Retry-After``, as
    ``details[].retryDelay`` holding something like ``"34484.99s"``. That is the
    difference between resuming in a minute and resuming tomorrow, so it is worth
    reading even though it is not the standard place for it.
    """
    try:
        body = response.json()
    except ValueError:
        return None

    # Gemini's OpenAI-compatible layer wraps its error in a one-element list,
    # which is not the shape the OpenAI error object has.
    if isinstance(body, list):
        body = body[0] if body else None

    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict):
        return None

    for detail in error.get("details") or []:
        if not isinstance(detail, dict):
            continue
        raw = detail.get("retryDelay")
        if isinstance(raw, str) and raw.endswith("s"):
            try:
                return max(0.0, float(raw[:-1]))
            except ValueError:
                continue
    return None


def retry_delay_seconds(response: httpx.Response) -> float | None:
    """How long the provider says to wait: the header first, then the body."""
    header = retry_after_seconds(response)
    return header if header is not None else _body_retry_delay(response)


def humanise_delay(seconds: float) -> str:
    """Render a wait readably. "34484s" says nothing at a glance."""
    total = round(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


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
                asked = retry_delay_seconds(response)
                detail = (
                    f"{self.name} rate limited (429) on attempt {attempt}"
                    if response.status_code == 429
                    else f"{self.name} returned {response.status_code} on attempt {attempt}: "
                    f"{_truncate(response.text, 160)}"
                )
                if (
                    response.status_code == 429
                    and asked is not None
                    and asked > MAX_RETRY_AFTER_SECONDS
                ):
                    # A daily quota, not a burst. Retrying cannot help within this
                    # run, so stop and report when the quota comes back.
                    raise ProviderRateLimited(
                        f"{detail}; the quota returns in about {humanise_delay(asked)}"
                    )
                failure = (
                    ProviderRateLimited(detail)
                    if response.status_code == 429
                    else ProviderUnavailable(detail)
                )
                retry_after = asked

            if attempt == attempts:
                raise failure

            delay = retry_after if retry_after is not None else self._backoff(attempt)
            self._sleep(delay)

        # Unreachable: the loop either returns or raises. Kept for type checkers.
        raise ProviderUnavailable(f"{self.name} exhausted retries")

    # -- internals ---------------------------------------------------------

    def _backoff(self, attempt: int) -> float:
        """Exponential backoff with jitter, so concurrent runs do not sync up."""
        ceiling = min(MAX_BACKOFF_SECONDS, BASE_BACKOFF_SECONDS * 2.0 ** (attempt - 1))
        return ceiling * (0.5 + 0.5 * self._jitter())

    def _parse(self, response: httpx.Response, *, started: float, attempt: int) -> Completion:
        try:
            body = response.json()
        except ValueError as exc:
            raise InvalidResponse(
                f"{self.name} returned a non-JSON body: {_truncate(response.text)}"
            ) from exc

        try:
            choice = body["choices"][0]
            text = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise InvalidResponse(
                f"{self.name} response had no choices[0].message.content: {_truncate(str(body))}"
            ) from exc

        if not isinstance(text, str):
            raise InvalidResponse(f"{self.name} returned a non-string message content")

        usage = body.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        reported_total = usage.get("total_tokens")
        reported_total = int(reported_total) if reported_total is not None else None
        total_tokens = (
            reported_total if reported_total is not None else prompt_tokens + completion_tokens
        )
        # Reasoning tokens are charged as output. They are absent from the message,
        # so billing off completion_tokens alone would understate a paid run.
        billable_output = max(completion_tokens, total_tokens - prompt_tokens)

        if self.free_tier:
            cost: float | None = 0.0
        elif usage:
            cost = estimate_cost(self.model, prompt_tokens, billable_output)
        else:
            # No usage block and no free-tier flag: we cannot claim this was free.
            cost = None

        return Completion(
            text=text,
            model=body.get("model") or self.model,
            provider=self.name,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                reported_total=reported_total,
            ),
            latency_seconds=round(self._clock() - started, 4),
            attempts=attempt,
            cost_usd=cost,
            finish_reason=choice.get("finish_reason"),
        )
