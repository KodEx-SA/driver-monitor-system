"""
Responsibility: learn the driver's personal "eyes open" EAR baseline at
the start of a session, instead of assuming a fixed threshold works for
everyone.

This module only collects samples and computes a number, it doesn't
touch the camera, the detector, or drawing. That keeps it testable with
plain float samples.
"""

import time
import numpy as np
from src import config

class BaselineCalibrator:
    """
    Collects EAR samples over a fixed duration, then computes a baseline.
    Usage:
        calibrator = BaselineCalibrator()
        calibrator.start()
        while not calibrator.is_complete:
            calibrator.add_sample(current_ear)
        baseline = calibrator.compute_baseline()
    """

    def __init__(
        self,
        duration_seconds: float = config.EAR_CALIBRATION_SECONDS,
        baseline_percentile: float = config.EAR_BASELINE_PERCENTILE,
    ) -> None:
        self._duration_seconds = duration_seconds
        self._baseline_percentile = baseline_percentile
        self._samples: list[float] = []
        self._start_time: float | None = None

    def start(self) -> None:
        """Begin (or restart) a calibration window."""
        self._start_time = time.monotonic()
        self._samples = []

    def add_sample(self, ear: float) -> None:
        """Record one EAR reading. Call once per frame during calibration."""
        self._samples.append(ear)

    @property
    def is_complete(self) -> bool:
        """Whether the calibration window has elapsed."""
        if self._start_time is None:
            return False
        return self._elapsed_seconds() >= self._duration_seconds

    @property
    def progress(self) -> float:
        """Fraction of the calibration window completed, from 0.0 to 1.0."""
        if self._start_time is None:
            return 0.0
        return min(self._elapsed_seconds() / self._duration_seconds, 1.0)

    def compute_baseline(self) -> float:
        """
        Return the baseline EAR from collected samples.
        """
        if not self._samples:
            raise RuntimeError(
                "compute_baseline() called with no samples. "
                "Did calibration actually run before this was called?"
            )
        return float(np.percentile(self._samples, self._baseline_percentile))

    def _elapsed_seconds(self) -> float:
        return time.monotonic() - self._start_time
