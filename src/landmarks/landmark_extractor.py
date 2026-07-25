"""
landmark_extractor.py
=====================
Extracts structured hand landmark arrays from individual video frames.

This module sits between :mod:`src.landmarks.hand_detector` (raw MediaPipe
results) and :mod:`src.landmarks.landmark_sequence` (per-video sequences).

Output format
-------------
Each call to :meth:`LandmarkExtractor.extract_from_frame` returns a
``dict[str, np.ndarray | None]`` with keys ``"left"`` and ``"right"``.
Each value is either:

- A ``(21, 3)`` float32 array of ``(x, y, z)`` normalised coordinates, or
- ``None`` / a zero array if that hand was not detected.

The :class:`HandLandmarks` dataclass provides a richer typed container for
single-hand results when the raw structured data is needed.

Usage
-----
::

    from src.landmarks.hand_detector import HandDetector
    from src.landmarks.landmark_extractor import LandmarkExtractor
    import cv2

    frame = cv2.imread("frame.jpg")

    with HandDetector(max_num_hands=2) as detector:
        extractor = LandmarkExtractor(detector)
        hand_dict = extractor.extract_from_frame(frame)

        left = hand_dict["left"]   # np.ndarray (21,3) or zero array
        right = hand_dict["right"] # np.ndarray (21,3) or zero array
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from src.landmarks.hand_detector import HandDetector
from src.landmarks.landmark_utils import (
    NUM_COORDS,
    NUM_LANDMARKS,
    SINGLE_HAND_SHAPE,
    zero_landmarks,
    zero_two_hands,
)

logger = logging.getLogger(__name__)


# ── Data container ────────────────────────────────────────────────────────────

@dataclass
class HandLandmarks:
    """Structured container for a single detected hand's landmarks.

    Attributes
    ----------
    landmarks:
        NumPy array of shape ``(21, 3)`` with ``(x, y, z)`` coordinates.
        All values are in the normalised range ``[0, 1]`` for x and y;
        z is a relative depth value (scale varies with model).
    handedness:
        ``"Left"`` or ``"Right"`` as reported by MediaPipe.
        Note: MediaPipe labels from the detector's perspective (i.e.
        mirrored vs. the subject's actual hand).
    confidence:
        MediaPipe handedness classification confidence in ``[0.0, 1.0]``.
    hand_index:
        Zero-based index of this hand in the MediaPipe result list.
    """

    landmarks: np.ndarray                        # (21, 3) float32
    handedness: str                              # "Left" | "Right"
    confidence: float = 1.0
    hand_index: int = 0

    def __post_init__(self) -> None:
        if self.landmarks.shape != SINGLE_HAND_SHAPE:
            raise ValueError(
                f"HandLandmarks.landmarks must have shape {SINGLE_HAND_SHAPE}, "
                f"got {self.landmarks.shape}."
            )
        if self.handedness not in {"Left", "Right"}:
            raise ValueError(
                f"handedness must be 'Left' or 'Right', got {self.handedness!r}."
            )

    @property
    def is_left(self) -> bool:
        """``True`` if this is a left-hand detection."""
        return self.handedness == "Left"

    @property
    def is_right(self) -> bool:
        """``True`` if this is a right-hand detection."""
        return self.handedness == "Right"

    def to_array(self) -> np.ndarray:
        """Return the landmark array (alias for ``self.landmarks``)."""
        return self.landmarks

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"HandLandmarks(hand={self.handedness}, "
            f"conf={self.confidence:.3f}, "
            f"shape={self.landmarks.shape})"
        )


# ── LandmarkExtractor ─────────────────────────────────────────────────────────

class LandmarkExtractor:
    """Extract structured landmark arrays from a single video frame.

    Parameters
    ----------
    detector:
        An initialised :class:`~src.landmarks.hand_detector.HandDetector`.
    num_hands:
        Expected number of hands (1 or 2). Affects the shape of the output
        when using :meth:`extract_two_hands`.
    fill_missing_with_zeros:
        When ``True`` (default), missing hands are represented as a zero
        array of shape ``(21, 3)`` rather than ``None``. This ensures
        consistent tensor shapes downstream.

    Notes
    -----
    The extractor does **not** own the detector — it does not close it.
    Use the detector as a context manager or call ``detector.close()``
    explicitly.
    """

    def __init__(
        self,
        detector: HandDetector,
        num_hands: int = 2,
        fill_missing_with_zeros: bool = True,
    ) -> None:
        if not isinstance(detector, HandDetector):
            raise TypeError("detector must be a HandDetector instance.")
        self.detector = detector
        self.num_hands = num_hands
        self.fill_missing_with_zeros = fill_missing_with_zeros

    # ── Internal parsers ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_hand_result(
        hand_landmarks: Any,
        handedness_info: Any,
        hand_index: int,
    ) -> HandLandmarks:
        """Parse one MediaPipe hand result into a :class:`HandLandmarks` object.

        Parameters
        ----------
        hand_landmarks:
            One element from ``results.multi_hand_landmarks``.
        handedness_info:
            One element from ``results.multi_handedness``.
        hand_index:
            Index of this hand in the results list.

        Returns
        -------
        HandLandmarks
            Structured container with the parsed landmark array.
        """
        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
            dtype=np.float32,
        )                                             # (21, 3)

        handedness_label = handedness_info.classification[0].label  # "Left" | "Right"
        confidence = float(handedness_info.classification[0].score)

        return HandLandmarks(
            landmarks=coords,
            handedness=handedness_label,
            confidence=confidence,
            hand_index=hand_index,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def extract_raw(self, frame: np.ndarray) -> list[HandLandmarks]:
        """Detect hands and return all results as a list of :class:`HandLandmarks`.

        Parameters
        ----------
        frame:
            BGR or RGB frame array of shape ``(H, W, 3)``.

        Returns
        -------
        list[HandLandmarks]
            One :class:`HandLandmarks` per detected hand (0, 1, or 2 items).
            Returns an empty list if no hands are detected.
        """
        results = self.detector.detect(frame)

        if not results.multi_hand_landmarks:
            return []

        hand_list: list[HandLandmarks] = []
        for idx, (hl, hedness) in enumerate(
            zip(results.multi_hand_landmarks, results.multi_handedness)
        ):
            try:
                hand_list.append(self._parse_hand_result(hl, hedness, idx))
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to parse hand %d: %s", idx, exc)
        return hand_list

    def extract_from_frame(
        self, frame: np.ndarray
    ) -> dict[str, Optional[np.ndarray]]:
        """Extract landmarks split by handedness for one frame.

        Parameters
        ----------
        frame:
            BGR frame array of shape ``(H, W, 3)``.

        Returns
        -------
        dict[str, np.ndarray | None]
            Keys are ``"left"`` and ``"right"``.
            Values are ``(21, 3)`` float32 arrays, or:

            - Zero array if ``fill_missing_with_zeros=True`` (default).
            - ``None`` if ``fill_missing_with_zeros=False``.
        """
        filler = zero_landmarks() if self.fill_missing_with_zeros else None
        result: dict[str, Optional[np.ndarray]] = {"left": filler, "right": filler}

        detected = self.extract_raw(frame)
        for hand in detected:
            key = hand.handedness.lower()  # "left" or "right"
            result[key] = hand.landmarks

        return result

    def extract_two_hands(self, frame: np.ndarray) -> np.ndarray:
        """Extract both hands as a single stacked array.

        Parameters
        ----------
        frame:
            BGR frame array of shape ``(H, W, 3)``.

        Returns
        -------
        np.ndarray
            Array of shape ``(2, 21, 3)`` where:

            - Index 0 → left hand (zeros if not detected).
            - Index 1 → right hand (zeros if not detected).
        """
        hand_dict = self.extract_from_frame(frame)
        left = hand_dict["left"] if hand_dict["left"] is not None else zero_landmarks()
        right = hand_dict["right"] if hand_dict["right"] is not None else zero_landmarks()
        return np.stack([left, right], axis=0)  # (2, 21, 3)

    def detection_summary(self, frame: np.ndarray) -> dict[str, Any]:
        """Return a human-readable summary of detection results for one frame.

        Useful for debugging. Includes detection counts, handedness, and
        per-landmark wrist position.

        Parameters
        ----------
        frame:
            BGR frame array.

        Returns
        -------
        dict[str, Any]
            Summary with keys ``"num_detected"``, ``"hands"``.
        """
        detected = self.extract_raw(frame)
        return {
            "num_detected": len(detected),
            "hands": [
                {
                    "handedness": h.handedness,
                    "confidence": round(h.confidence, 4),
                    "wrist_xy": (round(float(h.landmarks[0, 0]), 4),
                                 round(float(h.landmarks[0, 1]), 4)),
                }
                for h in detected
            ],
        }
