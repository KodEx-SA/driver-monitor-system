import pytest
from src.detection.features import eye_aspect_ratio, mouth_aspect_ratio

def test_eye_aspect_ratio_open_eye():
    # A wide, clearly-open eye: horizontal span of 0.3, vertical span of 0.1
    # on both the inner and outer point pairs.
    eye_points = [
        (0.0, 0.05), # left corner
        (0.1, 0.0), # top inner
        (0.2, 0.0), # top outer
        (0.3, 0.05), # right corner
        (0.2, 0.1), # bottom outer
        (0.1, 0.1), # bottom inner
    ]
    ear = eye_aspect_ratio(eye_points)
    # vertical gaps (0.1) over horizontal gap (0.3) -> should be a
    # solidly "open" ratio, comfortably above zero.
    assert ear == pytest.approx(0.333, abs=0.01)


def test_eye_aspect_ratio_closed_eye():
    # A closed eye: top and bottom lids meet, so vertical distance is ~0.
    eye_points = [
        (0.0, 0.05),
        (0.1, 0.05),
        (0.2, 0.05),
        (0.3, 0.05),
        (0.2, 0.05),
        (0.1, 0.05),
    ]
    ear = eye_aspect_ratio(eye_points)
    assert ear == pytest.approx(0.0, abs=0.001)


def test_eye_aspect_ratio_rejects_wrong_point_count():
    with pytest.raises(ValueError):
        eye_aspect_ratio([(0.0, 0.0)] * 5)  # only 5 points, needs 6


def test_mouth_aspect_ratio_closed_mouth():
    # Resting mouth: small vertical gap relative to width.
    mouth_points = [
        (0.15, 0.0), # top lip
        (0.15, 0.02), # bottom lip - small gap
        (0.0, 0.01), # left corner
        (0.3, 0.01), # right corner
    ]
    mar = mouth_aspect_ratio(mouth_points)
    assert mar < 0.15 # small relative to a wide-open mouth


def test_mouth_aspect_ratio_yawn():
    # Wide open mouth: large vertical gap relative to width.
    mouth_points = [
        (0.15, 0.0), # top lip
        (0.15, 0.25), # bottom lip - big gap, i.e. a yawn
        (0.0, 0.12), # left corner
        (0.3, 0.12), # right corner
    ]
    mar = mouth_aspect_ratio(mouth_points)
    assert mar > 0.5 # clearly higher than the closed-mouth case


def test_mouth_aspect_ratio_rejects_wrong_point_count():
    with pytest.raises(ValueError):
        mouth_aspect_ratio([(0.0, 0.0)] * 3) # only 3 points, needs 4