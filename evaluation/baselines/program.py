"""Baseline B: retrieve the evidence, then write a program the engine runs.

Same model, same document, different division of labour. Baseline A asks the model
for the answer; this asks it for the arithmetic and takes the number from the
engine. Whatever opens up between the two is the part of the task the model was
doing badly, which is the thing this project is about.

It also fails differently, and that difference is worth seeing. When a program
names a row the table does not have, there is no number to compare, so the run
records an unexecuted program rather than a wrong answer. On this baseline that is
the characteristic failure and it stays visible in the report instead of being
absorbed into the wrong-answer count.
"""

from __future__ import annotations

from app.services.dataset import FinQAExample
from app.services.llm import RUN_HALTING_ERRORS, LLMError, LLMProvider
from app.services.reasoning import (
    FORMAT_JSON,
    FORMAT_UNEXECUTED,
    FORMAT_UNPARSED,
    Document,
    Orchestrator,
    ParsedAnswer,
)

from .base import Prediction


class ProgramLLM:
    """Retrieval plus a program. The engine supplies the number."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        top_k: int = 10,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        self.provider = provider
        self.orchestrator = Orchestrator(
            provider, top_k=top_k, temperature=temperature, max_tokens=max_tokens
        )
        self.name = f"program@{provider.name}"
        self.prompt_version = self.orchestrator.prompt_version

    def answer(self, example: FinQAExample) -> Prediction:
        document = Document.from_example(example)

        try:
            grounded = self.orchestrator.answer(document, example.question)
        except RUN_HALTING_ERRORS:
            # Not this question's fault and not fixed by moving on.
            raise
        except LLMError as exc:
            return Prediction(
                system=self.name,
                question_id=example.id,
                parsed=ParsedAnswer(answer=None, format=FORMAT_UNPARSED, raw=""),
                provider=self.provider.name,
                model=self.provider.model,
                call_error=f"{type(exc).__name__}: {exc}",
            )

        completion = grounded.completion
        return Prediction(
            system=self.name,
            question_id=example.id,
            parsed=ParsedAnswer(
                # The answer is the engine's result, never a number from the reply.
                answer=grounded.answer,
                format=FORMAT_JSON if grounded.ok else FORMAT_UNEXECUTED,
                raw=grounded.reply,
                reasoning=grounded.program or "",
                evidence=tuple(citation.label for citation in grounded.citations),
            ),
            provider=completion.provider if completion else self.provider.name,
            model=completion.model if completion else self.provider.model,
            prompt_tokens=completion.prompt_tokens if completion else 0,
            completion_tokens=completion.completion_tokens if completion else 0,
            total_tokens=completion.usage.total_tokens if completion else 0,
            reasoning_tokens=completion.usage.reasoning_tokens if completion else 0,
            latency_seconds=completion.latency_seconds if completion else 0.0,
            cost_usd=completion.cost_usd if completion else None,
            attempts=completion.attempts if completion else 1,
            finish_reason=completion.finish_reason if completion else None,
            detail={
                "program": grounded.program,
                "steps": list(grounded.steps),
                "failure": grounded.failure,
                "citations": [citation.label for citation in grounded.citations],
                "retrieval_fallback": grounded.retrieval_fallback,
            },
        )
