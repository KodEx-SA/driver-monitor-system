"""
Step 2 checkpoint: prove face landmark detection works on the live feed.
You should see a mesh drawn over your face that tracks as you move.
Press 'q' to quit.
"""

import cv2

from src import config
from src.capture.camera import Camera
from src.detection.face_mesh import FaceMeshDetector


def main() -> None:
    with Camera() as camera, FaceMeshDetector() as detector:
        print("Camera + face mesh ready. Press 'q' to quit.")

        while True:
            frame = camera.read_frame()
            if frame is None:
                print("Failed to read frame -stopping.")
                break

            landmarks = detector.process(frame)
            if landmarks is not None:
                detector.draw_landmarks(frame, landmarks)

            cv2.imshow(config.WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    