"""A minimum interval between calls.

Free tiers answer with 429 if you send too fast, and a 429 that burns a retry is
slower than just waiting. Deliberately simple: one call at a time, at most one
every N seconds. Good enough for an evaluation run, which is a sequential
workload by nature.

The clock and sleep are injectable so tests do not actually wait.
"""

from __future__ import annotations

import time
from collections.abc import Callable

SleepFn = Callable[[float], None]
ClockFn = Callable[[], float]


class RateLimiter:
    def __init__(
        self,
        min_interval_seconds: float = 0.0,
        *,
        clock: ClockFn = time.monotonic,
        sleep: SleepFn = time.sleep,
    ) -> None:
        self.min_interval = max(0.0, float(min_interval_seconds))
        self._clock = clock
        self._sleep = sleep
        self._last_call: float | None = None

    def wait(self) -> float:
        """Block until the next call is allowed. Returns the seconds spent waiting."""
        now = self._clock()
        if self._last_call is None or self.min_interval <= 0.0:
            self._last_call = now
            return 0.0

        remaining = self.min_interval - (now - self._last_call)
        if remaining > 0.0:
            self._sleep(remaining)
            self._last_call = self._clock()
            return remaining

        self._last_call = now
        return 0.0
