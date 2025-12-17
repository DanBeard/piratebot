"""
Interface for person/object detection.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Detection:
    """Represents a detected person or object."""

    # Bounding box coordinates (x1, y1, x2, y2)
    x1: int
    y1: int
    x2: int
    y2: int

    # Detection confidence (0.0 - 1.0)
    confidence: float

    # Class label (e.g., "person")
    label: str

    # Unique tracking ID (if tracking is enabled)
    track_id: Optional[int] = None

    @property
    def center(self) -> tuple[int, int]:
        """Returns the center point of the bounding box."""
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def area(self) -> int:
        """Returns the area of the bounding box in pixels."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


class IDetector(ABC):
    """
    Abstract interface for person/object detection.

    Implementations:
    - YoloDetector: YOLOv8-based detection
    - (Future) MediaPipeDetector: MediaPipe-based detection
    """

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Detect people/objects in the given frame.

        Args:
            frame: BGR image as numpy array (OpenCV format)

        Returns:
            List of Detection objects for each detected person
        """
        pass

    @abstractmethod
    def detect_people(self, frame: np.ndarray) -> list[Detection]:
        """
        Detect only people (person class) in the frame.

        Args:
            frame: BGR image as numpy array (OpenCV format)

        Returns:
            List of Detection objects for detected people only
        """
        pass

    def warmup(self) -> None:
        """
        Optional: Run a warmup inference to load model into GPU memory.
        Call this at startup to avoid cold-start latency.
        """
        pass

    def cleanup(self) -> None:
        """
        Optional: Clean up resources (e.g., release GPU memory).
        """
        pass
