"""
Responsibility: turn a fatigue score into a Normal -> Warning -> Critical
state, with hysteresis so a single noisy score reading can't flip the
state back and forth.

Without hysteresis, a score bouncing around a threshold (say, 39 -> 41
-> 38 -> 42) would flip the state every frame, which would be useless
for driving alerts off of. Instead, a new state must be "pending" for
STATE_SUSTAIN_SECONDS before it actually takes effect. This applies to
de-escalation too - dropping back to Normal also requires sustained
evidence, not just one good frame after a bad stretch.

This module only tracks state. It doesn't decide what an alert should
do about a given state - that's alert_manager.py.
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

    def update(self, score: float, timestamp: float | None = None) -> FatigueState:
        """
        Feed in the latest fatigue score. Returns the current state
        (which may or may not have just changed).
        """
        ts = timestamp if timestamp is not None else time.monotonic()
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
