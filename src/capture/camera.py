"""
Responsibility: open a camera, hand out frames, clean up when done.
That's it. This module knows nothing about faces, eyes, or fatigue
keeping it this narrow means it can be swapped (e.g. for a video file,
or a different camera backend) without touching any detection code.
"""

import cv2

from src import config


class Camera:
    """A thin, well-behaved wrapper around cv2.VideoCapture.

    Using a class (instead of bare cv2 calls scattered through the app)
    means the open/read/release lifecycle is guaranteed to be handled
    consistently everywhere the camera is used.
    """

    def __init__(
        self,
        index: int = config.CAMERA_INDEX,
        width: int = config.FRAME_WIDTH,
        height: int = config.FRAME_HEIGHT,
    ) -> None:
        self._index = index
        self._width = width
        self._height = height
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        """Open the camera. Raises if the camera can't be reached."""
        self._cap = cv2.VideoCapture(self._index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera at index {self._index}. "
                "Check that it's connected and not in use by another app."
            )

    def read_frame(self):
        """Return the next frame as a BGR numpy array, or None if unavailable."""
        if self._cap is None:
            raise RuntimeError("Camera not opened. Call open() first.")

        success, frame = self._cap.read()
        if not success:
            return None
        return frame

    def release(self) -> None:
        """Release the camera. Always safe to call, even if never opened."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
