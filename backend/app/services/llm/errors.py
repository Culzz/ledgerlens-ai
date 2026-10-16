"""Failure modes for the LLM adapter.

Separated into types because the evaluation runner treats them very differently.
One bad question should be recorded and stepped over; a rate limit should pause
the run; a missing key should stop it before it starts.
"""

from __future__ import annotations


class LLMError(Exception):
    """Base class for everything this package raises."""


class ProviderNotConfigured(LLMError):
    """The selected provider has no key, or no model, or is not a known name."""


class ProviderRequestFailed(LLMError):
    """The provider answered, and the answer was unusable. Not worth retrying."""


class ProviderRateLimited(LLMError):
    """Rate limited, and out of retries. Pause the run rather than fail it."""


class ProviderUnavailable(LLMError):
    """Server error or timeout, and out of retries."""


class InvalidResponse(LLMError):
    """The response body was not the shape we expect."""


class BudgetExceeded(LLMError):
    """The spend ceiling has been reached. Stop before spending more."""


class ScriptExhausted(LLMError):
    """A scripted fake provider ran out of canned responses."""


# Errors that mean "stop the run", rather than "this one question failed".
#
# The distinction matters. A malformed reply or a stray 400 is one bad question
# and the run carries on. A rate limit, an exhausted budget or a missing key will
# fail the same way on every remaining question, so carrying on just burns the
# quota the pause was protecting.
RUN_HALTING_ERRORS = (ProviderNotConfigured, ProviderRateLimited, BudgetExceeded)
