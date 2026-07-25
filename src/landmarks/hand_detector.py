"""
hand_detector.py
================
Wraps MediaPipe Hands to provide a clean, configurable detection interface.

This module is the entry point for the landmark extraction pipeline.
It initialises the MediaPipe ``Hands`` solution, exposes all key
configuration parameters, and returns raw MediaPipe result objects that
can be consumed by :mod:`src.landmarks.landmark_extractor`.

Usage
-----
::

    import cv2
    from src.landmarks.hand_detector import HandDetector

    # As a context manager (recommended — ensures proper cleanup)
    with HandDetector(max_num_hands=2, min_detection_confidence=0.7) as detector:
        frame = cv2.imread("frame.jpg")
        results = detector.detect(frame)
        if results.multi_hand_landmarks:
            print(f"Detected {len(results.multi_hand_landmarks)} hand(s)")

    # Manual lifecycle management
    detector = HandDetector()
    results = detector.detect(frame)
    detector.close()
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Lazy MediaPipe import ─────────────────────────────────────────────────────

def _require_mediapipe() -> Any:
    """Import and return the ``mediapipe`` module, raising a helpful error if absent."""
    try:
        import mediapipe as mp  # noqa: PLC0415
        return mp
    except ImportError as exc:
        raise ImportError(
            "MediaPipe is required for landmark extraction.\n"
            "  → Install it with:  pip install mediapipe\n"
            "  → On Kaggle:        !pip install mediapipe"
        ) from exc


# ── HandDetector ──────────────────────────────────────────────────────────────

class HandDetector:
    """Configurable MediaPipe Hands wrapper for hand detection.

    Parameters
    ----------
    static_image_mode:
        If ``True``, each frame is treated as a static image (no temporal
        tracking between frames). Use ``True`` for offline batch processing
        of individual frames; ``False`` for video streams.
    max_num_hands:
        Maximum number of hands to detect per frame (1 or 2).
    model_complexity:
        MediaPipe model complexity: ``0`` (lite, faster) or ``1`` (full, more
        accurate). Default is ``1``.
    min_detection_confidence:
        Minimum confidence score for a detection to be considered successful.
        Range: ``[0.0, 1.0]``. Default is ``0.5``.
    min_tracking_confidence:
        Minimum confidence for hand tracking to be considered successful
        (only relevant when ``static_image_mode=False``).
        Range: ``[0.0, 1.0]``. Default is ``0.5``.

    Attributes
    ----------
    is_closed : bool
        ``True`` after :meth:`close` has been called.

    Example
    -------
    ::

        with HandDetector(max_num_hands=1, min_detection_confidence=0.7) as det:
            results = det.detect(frame_bgr)
            n_hands = len(results.multi_hand_landmarks or [])
    """

    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_hands: int = 2,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        if not (1 <= max_num_hands <= 2):
            raise ValueError(f"max_num_hands must be 1 or 2, got {max_num_hands}.")
        if not (0.0 <= min_detection_confidence <= 1.0):
            raise ValueError("min_detection_confidence must be in [0, 1].")
        if not (0.0 <= min_tracking_confidence <= 1.0):
            raise ValueError("min_tracking_confidence must be in [0, 1].")

        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        mp = _require_mediapipe()
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.is_closed: bool = False

        logger.debug(
            "HandDetector initialised | max_hands=%d | det_conf=%.2f | track_conf=%.2f",
            max_num_hands, min_detection_confidence, min_tracking_confidence,
        )

    # ── Detection ─────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> Any:
        """Run hand detection on a single frame.

        Parameters
        ----------
        frame:
            An image frame as a NumPy array. Accepted colour orderings:
            BGR (OpenCV default) or RGB. The array is converted to RGB
            internally before passing to MediaPipe.
            Expected shape: ``(H, W, 3)`` with dtype ``uint8``.

        Returns
        -------
        mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
            Raw MediaPipe result object. Key attributes:

            - ``results.multi_hand_landmarks``: list of hand landmark sets,
              or ``None`` if no hands detected.
            - ``results.multi_handedness``: list of handedness classifications
              (``"Left"`` / ``"Right"``).

        Raises
        ------
        RuntimeError
            If called after :meth:`close` has been invoked.
        ValueError
            If *frame* is not a valid 3-channel uint8 image.
        """
        if self.is_closed:
            raise RuntimeError(
                "HandDetector has been closed. Create a new instance to detect again."
            )

        if frame is None or frame.size == 0:
            raise ValueError("frame must be a non-empty numpy array.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"frame must have shape (H, W, 3), got {frame.shape}."
            )

        # Convert BGR → RGB (MediaPipe expects RGB)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False  # perf optimisation recommended by MediaPipe
        results = self._hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        n_detected = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
        logger.debug("Detected %d hand(s) in frame of shape %s", n_detected, frame.shape)
        return results

    def detect_rgb(self, frame_rgb: np.ndarray) -> Any:
        """Run hand detection on an RGB frame (no colour conversion).

        Parameters
        ----------
        frame_rgb:
            Frame in RGB channel order, shape ``(H, W, 3)``, dtype ``uint8``.

        Returns
        -------
        MediaPipe result object (same as :meth:`detect`).
        """
        if self.is_closed:
            raise RuntimeError("HandDetector has been closed.")
        if frame_rgb is None or frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError(f"frame_rgb must have shape (H, W, 3), got {getattr(frame_rgb, 'shape', None)}.")
        if frame_rgb.dtype != np.uint8:
            frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)
        frame_rgb.flags.writeable = False
        results = self._hands.process(frame_rgb)
        frame_rgb.flags.writeable = True
        return results

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Release MediaPipe resources.

        Call this when the detector is no longer needed, or use the detector
        as a context manager to ensure automatic cleanup.
        """
        if not self.is_closed:
            self._hands.close()
            self.is_closed = True
            logger.debug("HandDetector closed.")

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"HandDetector("
            f"max_num_hands={self.max_num_hands}, "
            f"det_conf={self.min_detection_confidence}, "
            f"track_conf={self.min_tracking_confidence}, "
            f"static={self.static_image_mode})"
        )
