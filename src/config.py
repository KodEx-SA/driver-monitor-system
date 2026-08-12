"""
Single source of truth for every tunable value in the project.

Rule of thumb: if a number could reasonably change (a threshold, a window
size, a resolution), it belongs here, never hardcoded inside logic files.
This keeps detection/scoring code readable and makes tuning the system a
one-file job instead of a search-and-replace across the codebase.
"""

# Camera settings

CAMERA_INDEX = 0 # 0 = default webcam
FRAME_WIDTH = 640 # modest - lighter to process on modest hardware.
FRAME_HEIGHT = 480
TARGET_FPS = 30 # Requested FPS; actual FPS depends on hardware.

WINDOW_NAME = "Driver Drowsiness Monitor"

# Calibration settings

# How long to sample EAR at the start of a session to learn the driver's
# personal "eyes open" baseline. Blinking normally during this window is
# fine and expected - see EAR_BASELINE_PERCENTILE below for why.
EAR_CALIBRATION_SECONDS = 15.0

# We take this percentile of the collected samples as the baseline, not the
# mean. A straight average would be dragged down by normal blinks during
# calibration; the 90th percentile instead reflects where EAR sits when the
# eyes are genuinely open, ignoring the brief low dips from blinking.
EAR_BASELINE_PERCENTILE = 90.0