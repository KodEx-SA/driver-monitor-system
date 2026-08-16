"""
Press 'q' to quit at any time.
"""

import cv2
from src import config
from src.capture.camera import Camera
from src.detection.calibration import BaselineCalibrator
from src.detection.face_mesh import FaceMeshDetector
from src.detection.features import get_face_features
from src.scoring.fatigue_score import FatigueScorer
from src.scoring.state_machine import FatigueState, FatigueStateMachine

_STATE_COLORS = {
    FatigueState.NORMAL: (0, 255, 0), # green
    FatigueState.WARNING: (0, 200, 255), # amber
    FatigueState.CRITICAL: (0, 0, 255), # red
}

def _draw_calibration_overlay(frame, progress: float) -> None:
    percent = int(progress * 100)
    cv2.putText(
        frame, f"Calibrating... blink normally ({percent}%)", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2
    )

def _draw_monitoring_overlay(frame, fatigue_result, state: FatigueState) -> None:
    color = _STATE_COLORS[state]

    cv2.putText(
        frame, f"PERCLOS: {fatigue_result.perclos * 100:.0f}%", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
    )
    cv2.putText(
        frame, f"Yawns (last 60s): {fatigue_result.yawn_count_in_window}", (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
    )
    cv2.putText(
        frame, f"Eyes closed: {fatigue_result.continuous_closed_seconds:.1f}s", (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
    )
    cv2.putText(
        frame, f"Fatigue score: {fatigue_result.score:.0f}/100", (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
    )
    cv2.putText(
        frame, f"State: {state.value.upper()}", (10, 155),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
    )

def _run_calibration(camera: Camera, detector: FaceMeshDetector) -> float | None:
    calibrator = BaselineCalibrator()
    calibrator.start()

    while not calibrator.is_complete:
        frame = camera.read_frame()
        if frame is None:
            continue

        landmarks = detector.process(frame)
        if landmarks is not None:
            features = get_face_features(landmarks)
            calibrator.add_sample(features.avg_ear)
            detector.draw_landmarks(frame, landmarks)
        _draw_calibration_overlay(frame, calibrator.progress)
        cv2.imshow(config.WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return None
    return calibrator.compute_baseline()

def main() -> None:
    with Camera() as camera, FaceMeshDetector() as detector:
        print("Camera + face mesh ready. Calibrating... blink normally.")

        baseline_ear = _run_calibration(camera, detector)
        if baseline_ear is None:
            print("Quit during calibration.")
            cv2.destroyAllWindows()
            return
        print(f"Calibration complete. Baseline EAR: {baseline_ear:.3f}")

        scorer = FatigueScorer()
        state_machine = FatigueStateMachine()

        while True:
            frame = camera.read_frame()
            if frame is None:
                print("Failed to read frame. Stopping...")
                break

            landmarks = detector.process(frame)
            if landmarks is not None:
                detector.draw_landmarks(frame, landmarks)
                features = get_face_features(landmarks)

                fatigue_result = scorer.update(
                    avg_ear=features.avg_ear,
                    baseline_ear=baseline_ear,
                    mar=features.mar,
                )
                state = state_machine.update(
                    fatigue_result.score,
                    continuous_closed_seconds=fatigue_result.continuous_closed_seconds,
                )

                _draw_monitoring_overlay(frame, fatigue_result, state)

            cv2.imshow(config.WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
