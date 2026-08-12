"""
test_calibration.py

Tests for src/detection/calibration.py.

BaselineCalibrator depends on wall-clock time, so we monkeypatch
time.monotonic() to a controllable fake clock rather than sleeping in
tests (slow, flaky). Everything else — sample collection, percentile
math — is tested directly with plain floats.
"""

import pytest

from src.detection.calibration import BaselineCalibrator


class FakeClock:
    """A controllable stand-in for time.monotonic()."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def fake_clock(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr("src.detection.calibration.time.monotonic", clock)
    return clock


def test_is_complete_false_before_start(fake_clock):
    calibrator = BaselineCalibrator(duration_seconds=10.0)
    assert calibrator.is_complete is False


def test_is_complete_false_before_duration_elapses(fake_clock):
    calibrator = BaselineCalibrator(duration_seconds=10.0)
    calibrator.start()
    fake_clock.advance(5.0)
    assert calibrator.is_complete is False


def test_is_complete_true_after_duration_elapses(fake_clock):
    calibrator = BaselineCalibrator(duration_seconds=10.0)
    calibrator.start()
    fake_clock.advance(10.0)
    assert calibrator.is_complete is True


def test_progress_reports_fraction_complete(fake_clock):
    calibrator = BaselineCalibrator(duration_seconds=10.0)
    calibrator.start()
    fake_clock.advance(2.5)
    assert calibrator.progress == pytest.approx(0.25)


def test_progress_caps_at_one(fake_clock):
    calibrator = BaselineCalibrator(duration_seconds=10.0)
    calibrator.start()
    fake_clock.advance(999.0)
    assert calibrator.progress == 1.0


def test_compute_baseline_uses_high_percentile_not_mean(fake_clock):
    # Mostly "open eye" readings (~0.30) with a couple of low blink dips.
    # The mean would be dragged down by the dips; the 90th percentile
    # should stay close to the resting-open value instead.
    calibrator = BaselineCalibrator(baseline_percentile=90.0)
    calibrator.start()
    samples = [0.30, 0.31, 0.29, 0.30, 0.05, 0.30, 0.32, 0.06, 0.30, 0.31]
    for s in samples:
        calibrator.add_sample(s)

    baseline = calibrator.compute_baseline()
    mean = sum(samples) / len(samples)

    assert baseline > mean  # percentile should sit above the blink-dragged mean
    assert baseline > 0.28  # close to the resting-open cluster, not the dips


def test_compute_baseline_raises_with_no_samples(fake_clock):
    calibrator = BaselineCalibrator()
    with pytest.raises(RuntimeError):
        calibrator.compute_baseline()
