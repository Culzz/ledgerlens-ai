"""Provider-neutral access to a chat model.

The rest of the codebase talks to ``LLMProvider`` and never to a vendor. Which
provider that is, and whether it costs anything, is decided by configuration.
"""

from .base import LLMProvider
from .budget import BudgetLedger
from .errors import (
    RUN_HALTING_ERRORS,
    BudgetExceeded,
    InvalidResponse,
    LLMError,
    ProviderNotConfigured,
    ProviderRateLimited,
    ProviderRequestFailed,
    ProviderUnavailable,
    ScriptExhausted,
)
from .factory import (
    COMPATIBLE_ENDPOINTS,
    MODEL_LIST_HINTS,
    Connection,
    build_provider,
    mask_key,
    resolve_connection,
)
from .fake import FakeLLM
from .openai_compatible import OpenAICompatibleProvider
from .pricing import Price, estimate_cost, price_for
from .rate_limit import RateLimiter
from .types import Completion, Message, Usage, assistant, system, user

__all__ = [
    "COMPATIBLE_ENDPOINTS",
    "MODEL_LIST_HINTS",
    "RUN_HALTING_ERRORS",
    "BudgetExceeded",
    "BudgetLedger",
    "Completion",
    "Connection",
    "FakeLLM",
    "InvalidResponse",
    "LLMError",
    "LLMProvider",
    "Message",
    "OpenAICompatibleProvider",
    "Price",
    "ProviderNotConfigured",
    "ProviderRateLimited",
    "ProviderRequestFailed",
    "ProviderUnavailable",
    "RateLimiter",
    "ScriptExhausted",
    "Usage",
    "assistant",
    "build_provider",
    "estimate_cost",
    "mask_key",
    "price_for",
    "resolve_connection",
    "system",
    "user",
]
