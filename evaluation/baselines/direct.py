"""Baseline A: direct prompting with the whole document in context.

This is the comparison the rest of the project has to beat, so it is set up to
succeed rather than to fail. The full table and the full narrative go in the
prompt, and the model does its own arithmetic.

On FinQA that is a genuinely strong baseline, because FinQA documents are small
- a handful of table rows and a couple of dozen narrative lines - so nothing is
being hidden from the model by not retrieving. The measured retrieval ceiling
says the same thing from the other side: retrieval is not the bottleneck here,
reasoning is.
"""

from __future__ import annotations

from app.services.dataset import FinQAExample
from app.services.llm import RUN_HALTING_ERRORS, LLMError, LLMProvider
from app.services.reasoning import parse_answer

from .base import Prediction
from .prompts import DIRECT_PROMPT_VERSION, build_direct_messages


class DirectLLM:
    """The whole document in the prompt. The model does the arithmetic."""

    prompt_version = DIRECT_PROMPT_VERSION

    def __init__(
        self,
        provider: LLMProvider,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.name = f"direct@{provider.name}"

    def answer(self, example: FinQAExample) -> Prediction:
        messages = build_direct_messages(example)

        try:
            completion = self.provider.complete(
                messages, temperature=self.temperature, max_tokens=self.max_tokens
            )
        except RUN_HALTING_ERRORS:
            # Not this question's fault, and it will not get better by trying the
            # next one. Let the runner decide whether to stop.
            raise
        except LLMError as exc:
            # A single bad question is recorded and stepped over. The runner
            # decides whether the failure means the whole run should stop.
            return Prediction(
                system=self.name,
                question_id=example.id,
                parsed=parse_answer(""),
                provider=self.provider.name,
                model=self.provider.model,
                call_error=f"{type(exc).__name__}: {exc}",
            )

        return Prediction(
            system=self.name,
            question_id=example.id,
            parsed=parse_answer(completion.text),
            provider=completion.provider,
            model=completion.model,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            latency_seconds=completion.latency_seconds,
            cost_usd=completion.cost_usd,
            attempts=completion.attempts,
        )
