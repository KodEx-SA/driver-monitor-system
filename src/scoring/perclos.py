"""
perclos.py

Responsibility: track PERCLOS (PERcentage of eye CLOSure) — the fraction
of the last N seconds during which the eyes were classified as closed.

Why PERCLOS instead of "eyes closed for K consecutive frames": a
frame-counter approach is fragile — it depends entirely on framerate,
and a single misdetected frame breaks the streak. PERCLOS instead asks
a more robust question: over a meaningful window of time, how much of
it was spent with eyes closed? That smooths out blinks and single-frame
noise while still reacting to sustained drowsiness within about a
minute.

This module has one job: given a stream of "was this frame closed"
classifications with timestamps, report the rolling closed percentage.
It doesn't decide what counts as "closed" (that's the caller's
threshold logic) and it doesn't turn PERCLOS into a fatigue score
(that's fatigue_score.py).
"""

import time
from collections import deque

from src import config


class PerclosTracker:
    """Maintains a rolling window of closed/open classifications and
    reports the fraction of the window that was "closed".
    """

    def __init__(self, window_seconds: float = config.PERCLOS_WINDOW_SECONDS) -> None:
        self._window_seconds = window_seconds
        # Each entry: (timestamp, is_closed). A deque so pruning old
        # entries from the left is O(1) instead of O(n) list slicing.
        self._samples: deque[tuple[float, bool]] = deque()

    def add_sample(self, is_closed: bool, timestamp: float | None = None) -> None:
        """Record one frame's closed/open classification."""
        ts = timestamp if timestamp is not None else time.monotonic()
        self._samples.append((ts, is_closed))
        self._prune(ts)

    def _prune(self, now: float) -> None:
        cutoff = now - self._window_seconds
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    @property
    def perclos(self) -> float:
        """Fraction of samples currently in the window that were closed,
        from 0.0 (never closed) to 1.0 (closed the entire window).

        Returns 0.0 if no samples have been added yet, rather than
        raising — an empty window reasonably means "no evidence of
        closure observed", not an error state.
        """
        if not self._samples:
            return 0.0
        closed_count = sum(1 for _, is_closed in self._samples if is_closed)
        return closed_count / len(self._samples)


class ClosureDurationTracker:
    """Tracks how long the eyes have been continuously closed, right now.

    This answers a different question than PerclosTracker: not "what
    fraction of the last 60s was closed" but "how long has the current,
    uninterrupted closure lasted". A single open frame resets it to
    zero — that's intentional, since a brief reopening means the driver
    isn't in a continuous microsleep anymore, whatever PERCLOS says
    about the last minute overall.
    """

    def __init__(self) -> None:
        self._closed_since: float | None = None
        self._current_duration: float = 0.0

    def add_sample(self, is_closed: bool, timestamp: float | None = None) -> float:
        """Record one frame's closed/open classification.

        Returns the current continuous-closure duration in seconds
        (0.0 if the eyes are currently open).
        """
        ts = timestamp if timestamp is not None else time.monotonic()

        if is_closed:
            if self._closed_since is None:
                self._closed_since = ts
            self._current_duration = ts - self._closed_since
        else:
            self._closed_since = None
            self._current_duration = 0.0

        return self._current_duration

    @property
    def current_duration(self) -> float:
        """Seconds the eyes have been continuously closed right now."""
        return self._current_duration
