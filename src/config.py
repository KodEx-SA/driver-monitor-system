"""
Single source of truth for every tunable value in the project.

Rule of thumb: if a number could reasonably change (a threshold, a window
size, a resolution), it belongs here - never hardcoded inside logic files.
This keeps detection/scoring code readable and makes tuning the system a
one-file job instead of a search-and-replace across the codebase.
"""

# Camera settings 

CAMERA_INDEX = 0          # 0 = default webcam. Change if you have multiple cameras.
FRAME_WIDTH = 640         # Kept modest on purpose - lighter to process on modest hardware.
FRAME_HEIGHT = 480
TARGET_FPS = 30           # Requested FPS; actual FPS depends on hardware.

WINDOW_NAME = "Driver Monitor"
