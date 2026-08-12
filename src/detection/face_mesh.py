"""
face_mesh.py

Responsibility: wrap MediaPipe Face Mesh so the rest of the app never
imports `mediapipe` directly. If we ever swap face-landmark models,
only this file changes — nothing downstream needs to know or care.

This module only detects and hands back landmarks. It does not decide
what those landmarks mean (that's features.py, Step 3) and it does not
decide whether to draw them (that's a caller's choice, kept separate so
this class works the same whether there's a display or not).
"""

import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2


class FaceMeshDetector:
    """Detects facial landmarks in a single BGR frame using MediaPipe."""

    def __init__(
        self,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=True,  # also gives iris landmarks, useful later for gaze
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles

    def process(self, frame):
        """Run detection on a BGR frame.

        Returns a list of landmarks for the first detected face (each
        landmark has .x, .y, .z in normalized 0-1 coordinates), or None
        if no face was found in this frame.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None

        # max_num_faces=1, so there's at most one face to return.
        return results.multi_face_landmarks[0].landmark

    def draw_landmarks(self, frame, face_landmarks) -> None:
        """Draw the face mesh tessellation onto `frame` in place.

        Kept separate from `process()` on purpose — detection should work
        identically whether or not anything gets drawn (e.g. in tests, or
        once we run headless without a display).
        """
        # drawing_utils expects the original NormalizedLandmarkList-style
        # object, so we wrap the raw landmarks back into that shape.
        landmark_list = landmark_pb2.NormalizedLandmarkList(landmark=face_landmarks)
        self._mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=landmark_list,
            connections=self._mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self._mp_drawing_styles
            .get_default_face_mesh_tesselation_style(),
        )

    def close(self) -> None:
        """Release MediaPipe resources. Call when done, e.g. on shutdown."""
        self._face_mesh.close()

    def __enter__(self) -> "FaceMeshDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
