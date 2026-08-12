"""
config.py

Single source of truth for every tunable value in the project.

Rule of thumb: if a number could reasonably change (a threshold, a window
size, a resolution), it belongs here — never hardcoded inside logic files.
This keeps detection/scoring code readable and makes tuning the system a
one-file job instead of a search-and-replace across the codebase.
"""

# --- Camera settings -------------------------------------------------------

CAMERA_INDEX = 0          # 0 = default webcam. Change if you have multiple cameras.
FRAME_WIDTH = 640         # Kept modest on purpose — lighter to process on modest hardware.
FRAME_HEIGHT = 480
TARGET_FPS = 30           # Requested FPS; actual FPS depends on hardware.

WINDOW_NAME = "Driver Drowsiness Monitor"

# --- Calibration settings ---------------------------------------------------

# How long to sample EAR at the start of a session to learn the driver's
# personal "eyes open" baseline. Blinking normally during this window is
# fine and expected — see EAR_BASELINE_PERCENTILE below for why.
EAR_CALIBRATION_SECONDS = 15.0

# We take this percentile of the collected samples as the baseline, not the
# mean. A straight average would be dragged down by normal blinks during
# calibration; the 90th percentile instead reflects where EAR sits when the
# eyes are genuinely open, ignoring the brief low dips from blinking.
EAR_BASELINE_PERCENTILE = 90.0

# --- PERCLOS settings --------------------------------------------------------

# Rolling window over which we compute "what percentage of recent time
# were the eyes closed". 60s matches the window used in real drowsiness
# research (PERCLOS) — long enough to smooth out blinks and single-frame
# jitter, short enough to react to a genuinely tired driver within a minute.
PERCLOS_WINDOW_SECONDS = 60.0

# A frame counts as "eyes closed" for PERCLOS purposes when EAR drops
# below this fraction of the calibrated baseline. 0.3 means the eyes need
# to be about 70% closed relative to the driver's own open-eye baseline —
# deliberately stricter than "below baseline", so ordinary partial
# closure (looking down, a slow blink) doesn't get counted as "closed".
EYE_CLOSED_OPENNESS_RATIO = 0.3

# --- Yawn settings -----------------------------------------------------------

YAWN_MAR_THRESHOLD = 0.6      # MAR above this counts as a yawn in progress
YAWN_WINDOW_SECONDS = 60.0    # how far back yawns still count toward the score
YAWN_SCORE_WEIGHT = 15.0      # points added to the fatigue score per yawn in the window
YAWN_SCORE_CAP = 30.0         # yawns alone can never push the score higher than this

# --- Fatigue state machine ---------------------------------------------------

WARNING_SCORE_THRESHOLD = 40.0
CRITICAL_SCORE_THRESHOLD = 70.0

# A score must stay past a threshold for this long before the state
# actually changes. This is the hysteresis that stops one noisy score
# spike from flipping Normal -> Critical -> Normal within a second.
STATE_SUSTAIN_SECONDS = 3.0

# --- Microsleep fast-path -----------------------------------------------

# PERCLOS is a 60s rolling trend — deliberately slow to react, since
# that's what makes it noise-resistant. But it's the wrong tool for
# "eyes have been shut for the last 2 seconds, right now" — that needs
# an instant response, not a rolling average catching up. This is a
# separate, faster signal: continuous, uninterrupted closure duration.
# At or above this many seconds, we treat it as conclusive on its own
# and skip the state machine's hysteresis delay entirely.
MICROSLEEP_SECONDS = 1.5
