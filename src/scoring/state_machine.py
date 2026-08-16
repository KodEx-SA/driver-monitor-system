"""
Responsibility: turn a fatigue score into a Normal -> Warning -> Critical
state, with hysteresis so a single noisy score reading can't flip the
state back and forth.
"""

import time
from enum import Enum
from src import config

class FatigueState(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"

class FatigueStateMachine:
    """Classifies a fatigue score into a state, with hysteresis."""

    def __init__(
        self,
        warning_threshold: float = config.WARNING_SCORE_THRESHOLD,
        critical_threshold: float = config.CRITICAL_SCORE_THRESHOLD,
        sustain_seconds: float = config.STATE_SUSTAIN_SECONDS,
    ) -> None:
        self._warning_threshold = warning_threshold
        self._critical_threshold = critical_threshold
        self._sustain_seconds = sustain_seconds

        self._state = FatigueState.NORMAL
        self._pending_state: FatigueState | None = None
        self._pending_since: float | None = None

    @property
    def state(self) -> FatigueState:
        """The current, confirmed state (not a pending/unconfirmed one)."""
        return self._state

    def update(
        self,
        score: float,
        continuous_closed_seconds: float = 0.0,
        timestamp: float | None = None,
    ) -> FatigueState:
        """
        Feed in the latest fatigue score. Returns the current state
        (which may or may not have just changed).
        """
        ts = timestamp if timestamp is not None else time.monotonic()

        if continuous_closed_seconds >= config.MICROSLEEP_SECONDS:
            self._state = FatigueState.CRITICAL
            self._pending_state = None
            self._pending_since = None
            return self._state

        target_state = self._classify(score)

        if target_state == self._state:
            # Already there - clear any pending transition away from it.
            self._pending_state = None
            self._pending_since = None
            return self._state

        if target_state != self._pending_state:
            # A new candidate transition - start the clock on it.
            self._pending_state = target_state
            self._pending_since = ts
            return self._state

        # Same candidate as last time - check if it's been sustained long enough.
        if ts - self._pending_since >= self._sustain_seconds:
            self._state = target_state
            self._pending_state = None
            self._pending_since = None
        return self._state

    def _classify(self, score: float) -> FatigueState:
        if score >= self._critical_threshold:
            return FatigueState.CRITICAL
        if score >= self._warning_threshold:
            return FatigueState.WARNING
        return FatigueState.NORMAL
