"""
YOLOv8-based person detection implementation.

Uses the ultralytics library for fast, accurate person detection.
"""

import logging
from typing import Optional

import numpy as np

from interfaces.detector import IDetector, Detection

logger = logging.getLogger(__name__)


class YoloDetector(IDetector):
    """
    Person detector using YOLOv8.

    This implementation uses the ultralytics YOLO library which provides
    pre-trained models optimized for various speed/accuracy tradeoffs:
    - yolov8n.pt: Nano (fastest, ~30ms on GPU)
    - yolov8s.pt: Small (balanced)
    - yolov8m.pt: Medium (more accurate)
    """

    # COCO class ID for "person"
    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        device: Optional[str] = None,
        enable_tracking: bool = True,
    ):
        """
        Initialize YOLOv8 detector.

        Args:
            model: Model name or path (yolov8n.pt, yolov8s.pt, etc.)
            confidence_threshold: Minimum confidence for detections
            device: Device to run on ("cuda", "cuda:0", "cpu", or None for auto)
            enable_tracking: Whether to enable object tracking (assigns IDs)
        """
        self.model_name = model
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.enable_tracking = enable_tracking
        self._model = None

        logger.info(
            f"YoloDetector initialized: model={model}, "
            f"confidence={confidence_threshold}, tracking={enable_tracking}"
        )

    def _ensure_loaded(self) -> None:
        """Lazy-load the YOLO model on first use."""
        if self._model is not None:
            return

        try:
            from ultralytics import YOLO

            logger.info(f"Loading YOLO model: {self.model_name}")
            self._model = YOLO(self.model_name)

            # Move to specified device if provided
            if self.device:
                self._model.to(self.device)

            logger.info("YOLO model loaded successfully")

        except ImportError:
            raise ImportError(
                "ultralytics package not installed. "
                "Install with: pip install ultralytics"
            )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Detect all objects in the frame.

        Args:
            frame: BGR image as numpy array

        Returns:
            List of Detection objects
        """
        self._ensure_loaded()

        if self.enable_tracking:
            # Use tracking for consistent IDs across frames
            results = self._model.track(
                frame,
                conf=self.confidence_threshold,
                persist=True,
                verbose=False,
            )
        else:
            results = self._model(
                frame,
                conf=self.confidence_threshold,
                verbose=False,
            )

        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i, box in enumerate(boxes):
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Get confidence
                conf = float(box.conf[0])

                # Get class
                cls_id = int(box.cls[0])
                cls_name = self._model.names[cls_id]

                # Get tracking ID if available
                track_id = None
                if self.enable_tracking and box.id is not None:
                    track_id = int(box.id[0])

                detections.append(
                    Detection(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        confidence=conf,
                        label=cls_name,
                        track_id=track_id,
                    )
                )

        return detections

    def detect_people(self, frame: np.ndarray) -> list[Detection]:
        """
        Detect only people in the frame.

        Args:
            frame: BGR image as numpy array

        Returns:
            List of Detection objects for people only
        """
        self._ensure_loaded()

        if self.enable_tracking:
            results = self._model.track(
                frame,
                classes=[self.PERSON_CLASS_ID],  # Only detect people
                conf=self.confidence_threshold,
                persist=True,
                verbose=False,
            )
        else:
            results = self._model(
                frame,
                classes=[self.PERSON_CLASS_ID],
                conf=self.confidence_threshold,
                verbose=False,
            )

        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])

                track_id = None
                if self.enable_tracking and box.id is not None:
                    track_id = int(box.id[0])

                detections.append(
                    Detection(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        confidence=conf,
                        label="person",
                        track_id=track_id,
                    )
                )

        return detections

    def warmup(self) -> None:
        """
        Run warmup inference to load model into GPU memory.
        """
        self._ensure_loaded()

        logger.info("Running YOLO warmup inference...")

        # Create a dummy frame for warmup
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)

        # Run inference to warm up
        _ = self._model(dummy_frame, verbose=False)

        logger.info("YOLO warmup complete")

    def cleanup(self) -> None:
        """
        Clean up model resources.
        """
        if self._model is not None:
            # Clear CUDA cache if using GPU
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            self._model = None
            logger.info("YOLO model unloaded")


# Convenience function for testing
def test_webcam_detection():
    """Test YOLOv8 detection with webcam."""
    import cv2

    detector = YoloDetector()
    detector.warmup()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    print("Press 'q' to quit, 's' to save frame with detections")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect people
            detections = detector.detect_people(frame)

            # Draw bounding boxes
            for det in detections:
                color = (0, 255, 0)  # Green
                cv2.rectangle(
                    frame,
                    (det.x1, det.y1),
                    (det.x2, det.y2),
                    color,
                    2,
                )

                # Label with confidence and track ID
                label = f"Person {det.confidence:.2f}"
                if det.track_id is not None:
                    label += f" ID:{det.track_id}"

                cv2.putText(
                    frame,
                    label,
                    (det.x1, det.y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            # Show detection count
            cv2.putText(
                frame,
                f"People detected: {len(detections)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.imshow("YOLOv8 Person Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                cv2.imwrite("detection_frame.jpg", frame)
                print("Saved detection_frame.jpg")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.cleanup()


if __name__ == "__main__":
    test_webcam_detection()
