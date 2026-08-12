"""
fatigue_score.py

Responsibility: combine PERCLOS and recent yawn activity into a single
0-100 fatigue score.

Design choice worth calling out: PERCLOS is the primary driver of the
score (it's the more research-backed signal), and yawning adds a capped
bonus on top. The cap matters — without it, someone yawning repeatedly
while otherwise perfectly alert (e.g. bored, not tired) could max out
the score on yawns alone, which isn't the story yawning actually tells
on its own. PERCLOS being high is a much stronger fatigue signal than
yawning being frequent.

This module only computes a number. It doesn't decide what to do about
a high score — that's state_machine.py (whether the number represents
Normal/Warning/Critical) and, later, alert_manager.py (what to actually
do about it).
"""

import time
from collections import deque
from dataclasses import dataclass

from src import config
from src.scoring.perclos import ClosureDurationTracker, PerclosTracker


@dataclass(frozen=True)
class FatigueScore:
    """A snapshot of the fatigue signal at a single point in time."""

    perclos: float             # 0.0-1.0, fraction of recent time eyes were closed
    yawn_count_in_window: int  # yawn events in the last YAWN_WINDOW_SECONDS
    continuous_closed_seconds: float  # how long eyes have been shut, right now
    score: float                # 0-100 composite fatigue score


class FatigueScorer:
    """Combines EAR and MAR readings, over time, into a fatigue score.

    Usage: call update() once per frame with the current EAR/MAR and the
    calibrated baseline EAR. It maintains all rolling state internally.
    """

    def __init__(self) -> None:
        self._perclos_tracker = PerclosTracker()
        self._closure_duration_tracker = ClosureDurationTracker()
        self._yawn_timestamps: deque[float] = deque()
        self._mouth_was_open = False  # tracks yawn *events*, not "currently open"

    def update(
        self,
        avg_ear: float,
        baseline_ear: float,
        mar: float,
        timestamp: float | None = None,
    ) -> FatigueScore:
        ts = timestamp if timestamp is not None else time.monotonic()

        closure_seconds = self._update_perclos(avg_ear, baseline_ear, ts)
        self._update_yawns(mar, ts)

        perclos = self._perclos_tracker.perclos
        yawn_count = len(self._yawn_timestamps)
        score = self._compute_score(perclos, yawn_count)

        return FatigueScore(
            perclos=perclos,
            yawn_count_in_window=yawn_count,
            continuous_closed_seconds=closure_seconds,
            score=score,
        )

    def _update_perclos(self, avg_ear: float, baseline_ear: float, ts: float) -> float:
        openness_ratio = (avg_ear / baseline_ear) if baseline_ear > 0 else 1.0
        is_closed = openness_ratio < config.EYE_CLOSED_OPENNESS_RATIO
        self._perclos_tracker.add_sample(is_closed, ts)
        return self._closure_duration_tracker.add_sample(is_closed, ts)

    def _update_yawns(self, mar: float, ts: float) -> None:
        mouth_is_open = mar > config.YAWN_MAR_THRESHOLD

        # Only count the rising edge (mouth just opened past threshold),
        # not every frame the mouth happens to stay open — otherwise one
        # long yawn would be counted dozens of times.
        if mouth_is_open and not self._mouth_was_open:
            self._yawn_timestamps.append(ts)
        self._mouth_was_open = mouth_is_open

        cutoff = ts - config.YAWN_WINDOW_SECONDS
        while self._yawn_timestamps and self._yawn_timestamps[0] < cutoff:
            self._yawn_timestamps.popleft()

    def _compute_score(self, perclos: float, yawn_count: int) -> float:
        perclos_component = perclos * 100.0
        yawn_component = min(yawn_count * config.YAWN_SCORE_WEIGHT, config.YAWN_SCORE_CAP)
        return min(perclos_component + yawn_component, 100.0)
