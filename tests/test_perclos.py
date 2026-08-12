"""
test_perclos.py

Tests for src/scoring/perclos.py.
"""

import pytest

from src.scoring.perclos import ClosureDurationTracker, PerclosTracker


def test_perclos_zero_with_no_samples():
    tracker = PerclosTracker(window_seconds=60.0)
    assert tracker.perclos == 0.0


def test_perclos_zero_when_all_samples_open():
    tracker = PerclosTracker(window_seconds=60.0)
    for i in range(10):
        tracker.add_sample(is_closed=False, timestamp=float(i))
    assert tracker.perclos == 0.0


def test_perclos_one_when_all_samples_closed():
    tracker = PerclosTracker(window_seconds=60.0)
    for i in range(10):
        tracker.add_sample(is_closed=True, timestamp=float(i))
    assert tracker.perclos == 1.0


def test_perclos_reflects_mixed_samples():
    tracker = PerclosTracker(window_seconds=60.0)
    # 3 closed out of 10 samples -> 0.3
    closed_flags = [True, True, True, False, False, False, False, False, False, False]
    for i, closed in enumerate(closed_flags):
        tracker.add_sample(is_closed=closed, timestamp=float(i))
    assert tracker.perclos == pytest.approx(0.3)


def test_perclos_prunes_samples_outside_window():
    tracker = PerclosTracker(window_seconds=10.0)
    # An old "closed" sample, well outside the window...
    tracker.add_sample(is_closed=True, timestamp=0.0)
    # ...followed by recent "open" samples inside a 10s window ending at t=100.
    for t in range(91, 101):
        tracker.add_sample(is_closed=False, timestamp=float(t))

    # The old closed sample should have been pruned, leaving only opens.
    assert tracker.perclos == 0.0


class TestClosureDurationTracker:
    """Tests for the continuous-closure fast path used for microsleep detection."""

    def test_duration_zero_when_open(self):
        tracker = ClosureDurationTracker()
        duration = tracker.add_sample(is_closed=False, timestamp=0.0)
        assert duration == 0.0

    def test_duration_grows_while_continuously_closed(self):
        tracker = ClosureDurationTracker()
        tracker.add_sample(is_closed=True, timestamp=0.0)
        tracker.add_sample(is_closed=True, timestamp=1.0)
        duration = tracker.add_sample(is_closed=True, timestamp=2.0)
        assert duration == pytest.approx(2.0)

    def test_a_single_open_frame_resets_duration_to_zero(self):
        tracker = ClosureDurationTracker()
        tracker.add_sample(is_closed=True, timestamp=0.0)
        tracker.add_sample(is_closed=True, timestamp=1.0)
        # One open frame in the middle of an otherwise-closed run...
        duration = tracker.add_sample(is_closed=False, timestamp=1.5)
        assert duration == 0.0

        # ...and closing again starts a fresh streak from zero, not
        # continuing the earlier one.
        duration = tracker.add_sample(is_closed=True, timestamp=1.6)
        assert duration == pytest.approx(0.0)
