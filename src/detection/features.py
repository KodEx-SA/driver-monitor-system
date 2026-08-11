import math
from dataclasses import dataclass


# MediaPipe Face Mesh landmark indices used for each ratio.
LEFT_EYE_INDICES = (362, 385, 387, 263, 373, 380)
RIGHT_EYE_INDICES = (33, 160, 158, 133, 153, 144)
MOUTH_INDICES = (13, 14, 78, 308)  # (top, bottom, left corner, right corner)


@dataclass(frozen=True)
class FaceFeatures:
    """A snapshot of the ratio values for a single frame."""

    left_ear: float
    right_ear: float
    avg_ear: float
    mar: float


def extract_points(landmarks, indices: tuple[int, ...]) -> list[tuple[float, float]]:
    """Pull (x, y) tuples out of mediapipe landmarks at the given indices.

    This is the one place that touches mediapipe's landmark objects.
    Everything after this function deals only in plain tuples.
    """
    return [(landmarks[i].x, landmarks[i].y) for i in indices]


def _distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Euclidean distance between two (x, y) points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def eye_aspect_ratio(eye_points: list[tuple[float, float]]) -> float:
    """Compute EAR from exactly 6 (x, y) points, in this order:

        eye_points[0] -> left corner
        eye_points[1] -> top, inner
        eye_points[2] -> top, outer
        eye_points[3] -> right corner
        eye_points[4] -> bottom, outer
        eye_points[5] -> bottom, inner

    Returns a small positive float. Lower = more closed. There is no
    universal "closed" threshold - Step 4 calibrates a personal
    baseline per user rather than assuming a fixed number here.
    """
    if len(eye_points) != 6:
        raise ValueError(f"eye_aspect_ratio expects 6 points, got {len(eye_points)}")

    p1, p2, p3, p4, p5, p6 = eye_points

    vertical_1 = _distance(p2, p6)
    vertical_2 = _distance(p3, p5)
    horizontal = _distance(p1, p4)

    if horizontal == 0:
        return 0.0  # degenerate input (e.g. identical points) - treat as closed

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def mouth_aspect_ratio(mouth_points: list[tuple[float, float]]) -> float:
    """Compute MAR from exactly 4 (x, y) points, in this order:

        mouth_points[0] -> top lip, inner
        mouth_points[1] -> bottom lip, inner
        mouth_points[2] -> left corner
        mouth_points[3] -> right corner

    Returns a small positive float. Higher = more open (a yawn spikes
    this well above resting/talking values).
    """
    if len(mouth_points) != 4:
        raise ValueError(f"mouth_aspect_ratio expects 4 points, got {len(mouth_points)}")

    top, bottom, left, right = mouth_points

    vertical = _distance(top, bottom)
    horizontal = _distance(left, right)

    if horizontal == 0:
        return 0.0  # degenerate input - treat as closed/resting

    return vertical / horizontal


def get_face_features(landmarks) -> FaceFeatures:
    """Convenience entry point: raw mediapipe landmarks in, all ratios out.

    This is what callers (main.py, and later the scorer) should use.
    They shouldn't need to know about landmark indices at all.
    """
    left_eye_points = extract_points(landmarks, LEFT_EYE_INDICES)
    right_eye_points = extract_points(landmarks, RIGHT_EYE_INDICES)
    mouth_points = extract_points(landmarks, MOUTH_INDICES)

    left_ear = eye_aspect_ratio(left_eye_points)
    right_ear = eye_aspect_ratio(right_eye_points)
    avg_ear = (left_ear + right_ear) / 2.0
    mar = mouth_aspect_ratio(mouth_points)

    return FaceFeatures(left_ear=left_ear, right_ear=right_ear, avg_ear=avg_ear, mar=mar)