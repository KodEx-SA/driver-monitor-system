"""
Responsibility: track PERCLOS (PERcentage of eye CLOSure) - the fraction
of the last N seconds during which the eyes were classified as closed.
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
        """
        Fraction of samples currently in the window that were closed,
        from 0.0 (never closed) to 1.0 (closed the entire window).

        Returns 0.0 if no samples have been added yet, rather than
        raising - an empty window reasonably means "no evidence of
        closure observed", not an error state.
        """
        if not self._samples:
            return 0.0
        closed_count = sum(1 for _, is_closed in self._samples if is_closed)
        return closed_count / len(self._samples)

class ClosureDurationTracker:
    """
    Tracks how long the eyes have been continuously closed, right now.
    """

    def __init__(self) -> None:
        self._closed_since: float | None = None
        self._current_duration: float = 0.0

    def add_sample(self, is_closed: bool, timestamp: float | None = None) -> float:
        """
        Record one frame's closed/open classification.
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
