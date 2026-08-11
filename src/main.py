import cv2
from src import config
from src.capture.camera import Camera
from src.detection.face_mesh import FaceMeshDetector
from src.detection.features import get_face_features

def _draw_features_overlay(frame, avg_ear: float, mar: float) -> None:
    """Draw the current EAR/MAR values in the corner of the frame.

    Kept as a small standalone function (not a method on any class)
    since it's pure display logic - it doesn't belong to the camera,
    the detector, or the feature math.
    """
    cv2.putText(
        frame,
        f"EAR: {avg_ear:.3f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"MAR: {mar:.3f}",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )


def main() -> None:
    with Camera() as camera, FaceMeshDetector() as detector:
        print("Camera + face mesh ready. Press 'q' to quit.")

        while True:
            frame = camera.read_frame()
            if frame is None:
                print("Failed to read frame - stopping.")
                break

            landmarks = detector.process(frame)
            if landmarks is not None:
                detector.draw_landmarks(frame, landmarks)
                features = get_face_features(landmarks)
                _draw_features_overlay(frame, features.avg_ear, features.mar)

            cv2.imshow(config.WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
