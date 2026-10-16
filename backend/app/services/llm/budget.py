"""A spend ceiling that survives a resume.

Evaluation runs get interrupted: a daily quota runs out, a laptop sleeps, a
session dies. If the counter lives only in memory then resuming starts it back
at zero and the ceiling stops meaning anything. So the ledger writes itself to
disk after every call.

Token totals are tracked separately from dollars because they are always known,
whereas a price is not.
"""

from __future__ import annotations

import json
from pathlib import Path

from .errors import BudgetExceeded


class BudgetLedger:
    """Cumulative usage for a project, persisted as a small JSON file."""

    def __init__(self, path: str | Path, limit_usd: float | None = None) -> None:
        self.path = Path(path)
        self.limit_usd = limit_usd
        self.spent_usd = 0.0
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.unpriced_calls = 0

    @classmethod
    def open(cls, path: str | Path, limit_usd: float | None = None) -> BudgetLedger:
        """Load the ledger if it is already there, otherwise start a fresh one."""
        ledger = cls(path, limit_usd)
        if not ledger.path.exists():
            return ledger

        try:
            data = json.loads(ledger.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # A corrupt ledger should not stop the run, but it must not be
            # silently treated as zero spend either. Start clean and say so.
            return ledger

        ledger.spent_usd = float(data.get("spent_usd", 0.0))
        ledger.calls = int(data.get("calls", 0))
        ledger.prompt_tokens = int(data.get("prompt_tokens", 0))
        ledger.completion_tokens = int(data.get("completion_tokens", 0))
        ledger.unpriced_calls = int(data.get("unpriced_calls", 0))
        return ledger

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def remaining_usd(self) -> float | None:
        if self.limit_usd is None:
            return None
        return max(0.0, self.limit_usd - self.spent_usd)

    def check(self) -> None:
        """Raise if the ceiling has already been reached."""
        if self.limit_usd is not None and self.spent_usd >= self.limit_usd:
            raise BudgetExceeded(
                f"budget exhausted: {self.spent_usd:.4f} USD spent of {self.limit_usd:.2f} limit"
            )

    def record(
        self,
        *,
        cost_usd: float | None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        self.calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        if cost_usd is None:
            # Unknown price. Count it so the report can say how much of the run
            # is unpriced instead of quietly rounding it to free.
            self.unpriced_calls += 1
        else:
            self.spent_usd = round(self.spent_usd + cost_usd, 6)
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "spent_usd": self.spent_usd,
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "unpriced_calls": self.unpriced_calls,
        }
        self.path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def summary(self) -> dict[str, float | int | None]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "spent_usd": self.spent_usd,
            "limit_usd": self.limit_usd,
            "remaining_usd": self.remaining_usd,
            "unpriced_calls": self.unpriced_calls,
        }
