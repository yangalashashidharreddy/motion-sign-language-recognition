"""
landmark_utils.py
=================
Stateless utility functions for processing hand landmark arrays.

All functions operate on NumPy arrays and have no side effects.
They are designed to be used independently of any detector or extractor class.

Conventions
-----------
- A **single-hand landmark array** has shape ``(21, 3)`` where the 21 rows
  correspond to the 21 MediaPipe hand landmarks and the 3 columns are
  ``(x, y, z)`` in normalised image coordinates (values in ``[0, 1]`` for
  x and y; z is a relative depth estimate).

- A **flattened landmark vector** has shape ``(63,)`` — the (21, 3) array
  read row-by-row.

- A **full-frame landmark array** has shape ``(2, 21, 3)`` — one entry for
  the left hand and one for the right hand.

- A **sequence array** has shape ``(T, 2, 21, 3)`` — T frames of the above.

MediaPipe landmark index reference
-----------------------------------
::

    0  = WRIST
    1  = THUMB_CMC       2  = THUMB_MCP      3  = THUMB_IP       4  = THUMB_TIP
    5  = INDEX_FINGER_MCP 6  = INDEX_MCP      7  = INDEX_PIP      8  = INDEX_DIP
    9  = INDEX_FINGER_TIP (same as 8 in some versions; 8 = DIP, no 9)
    ...
    (See MediaPipe documentation for the complete 0-20 index list)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Number of landmarks per hand in MediaPipe Hands
NUM_LANDMARKS: int = 21
# Number of coordinates per landmark (x, y, z)
NUM_COORDS: int = 3
# Index of the wrist landmark (used as origin for relative normalisation)
WRIST_IDX: int = 0
# Expected shape of a single-hand landmark array
SINGLE_HAND_SHAPE: tuple[int, int] = (NUM_LANDMARKS, NUM_COORDS)
# Expected shape of a two-hand frame array
TWO_HAND_SHAPE: tuple[int, int, int] = (2, NUM_LANDMARKS, NUM_COORDS)


# ── Validation ────────────────────────────────────────────────────────────────

def validate_landmark_shape(
    landmarks: np.ndarray,
    expected_shape: tuple[int, ...],
    name: str = "landmarks",
) -> bool:
    """Check that *landmarks* has the expected shape.

    Parameters
    ----------
    landmarks:
        The array to validate.
    expected_shape:
        The required shape tuple (e.g. ``(21, 3)``).
    name:
        Human-readable variable name, used in the raised error message.

    Returns
    -------
    bool
        ``True`` if the shape matches.

    Raises
    ------
    ValueError
        If the shape does not match or if *landmarks* is not a NumPy array.
    """
    if not isinstance(landmarks, np.ndarray):
        raise ValueError(f"'{name}' must be a NumPy array, got {type(landmarks).__name__}.")
    if landmarks.shape != expected_shape:
        raise ValueError(
            f"'{name}' has shape {landmarks.shape}, expected {expected_shape}."
        )
    return True


def is_empty_landmarks(landmarks: np.ndarray) -> bool:
    """Return ``True`` if the landmark array is all-zeros (i.e. no detection).

    Parameters
    ----------
    landmarks:
        Array of any shape. Typically ``(21, 3)`` or ``(2, 21, 3)``.
    """
    return np.all(landmarks == 0.0)


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalize_to_wrist(landmarks: np.ndarray) -> np.ndarray:
    """Translate landmarks so the wrist (index 0) is at the origin.

    This removes absolute hand position from the feature vector, making the
    representation translation-invariant.

    Parameters
    ----------
    landmarks:
        Array of shape ``(21, 3)``.

    Returns
    -------
    np.ndarray
        Translated array of the same shape. Dtype is preserved (float32 recommended).

    Raises
    ------
    ValueError
        If *landmarks* does not have shape ``(21, 3)``.
    """
    validate_landmark_shape(landmarks, SINGLE_HAND_SHAPE)
    wrist = landmarks[WRIST_IDX].copy()          # (3,)
    return landmarks - wrist                     # broadcast over 21 rows


def normalize_to_unit_scale(landmarks: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Scale landmarks so the maximum absolute value is 1.

    Applied after :func:`normalize_to_wrist` this produces a scale-invariant
    feature representation.

    Parameters
    ----------
    landmarks:
        Array of shape ``(21, 3)``.
    eps:
        Small constant to avoid division by zero.

    Returns
    -------
    np.ndarray
        Scaled array of the same shape.
    """
    validate_landmark_shape(landmarks, SINGLE_HAND_SHAPE)
    scale = np.max(np.abs(landmarks)) + eps
    return landmarks / scale


def normalize_landmarks(
    landmarks: np.ndarray,
    wrist_relative: bool = True,
    unit_scale: bool = True,
) -> np.ndarray:
    """Apply full normalisation pipeline to a single-hand landmark array.

    Combines :func:`normalize_to_wrist` and :func:`normalize_to_unit_scale`
    in a single call.

    Parameters
    ----------
    landmarks:
        Array of shape ``(21, 3)`` with dtype float32.
    wrist_relative:
        Translate so the wrist is at the origin.
    unit_scale:
        Scale so the maximum absolute coordinate is 1.

    Returns
    -------
    np.ndarray
        Normalised array of shape ``(21, 3)``.
    """
    validate_landmark_shape(landmarks, SINGLE_HAND_SHAPE)
    result = landmarks.copy().astype(np.float32)
    if wrist_relative:
        result = normalize_to_wrist(result)
    if unit_scale:
        result = normalize_to_unit_scale(result)
    return result


# ── Flattening ────────────────────────────────────────────────────────────────

def flatten_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Flatten a ``(21, 3)`` landmark array to a 1-D vector of length 63.

    Parameters
    ----------
    landmarks:
        Array of shape ``(21, 3)``.

    Returns
    -------
    np.ndarray
        Array of shape ``(63,)`` — row-major flatten.
    """
    validate_landmark_shape(landmarks, SINGLE_HAND_SHAPE)
    return landmarks.flatten()


def flatten_two_hands(landmarks: np.ndarray) -> np.ndarray:
    """Flatten a ``(2, 21, 3)`` two-hand array to a 1-D vector of length 126.

    Parameters
    ----------
    landmarks:
        Array of shape ``(2, 21, 3)``.

    Returns
    -------
    np.ndarray
        Array of shape ``(126,)``.
    """
    validate_landmark_shape(landmarks, TWO_HAND_SHAPE)
    return landmarks.flatten()


# ── Bounding box ──────────────────────────────────────────────────────────────

def compute_bounding_box(
    landmarks: np.ndarray,
    frame_width: Optional[int] = None,
    frame_height: Optional[int] = None,
) -> tuple[float, float, float, float]:
    """Compute the axis-aligned bounding box of a hand's landmarks.

    Parameters
    ----------
    landmarks:
        Array of shape ``(21, 3)``. Only x (col 0) and y (col 1) are used.
    frame_width:
        If provided, x coordinates are scaled to pixel values.
    frame_height:
        If provided, y coordinates are scaled to pixel values.

    Returns
    -------
    tuple[float, float, float, float]
        ``(x_min, y_min, x_max, y_max)`` in the same units as the inputs
        (normalised [0, 1] or pixel space if frame dimensions are given).
    """
    validate_landmark_shape(landmarks, SINGLE_HAND_SHAPE)
    x_coords = landmarks[:, 0]
    y_coords = landmarks[:, 1]

    x_min, x_max = float(x_coords.min()), float(x_coords.max())
    y_min, y_max = float(y_coords.min()), float(y_coords.max())

    if frame_width is not None:
        x_min *= frame_width
        x_max *= frame_width
    if frame_height is not None:
        y_min *= frame_height
        y_max *= frame_height

    return x_min, y_min, x_max, y_max


def compute_bounding_box_pixels(
    landmarks: np.ndarray,
    frame_width: int,
    frame_height: int,
    padding: int = 10,
) -> tuple[int, int, int, int]:
    """Compute a pixel-space bounding box with optional padding.

    Parameters
    ----------
    landmarks:
        Normalised landmark array of shape ``(21, 3)``.
    frame_width, frame_height:
        Dimensions of the source frame in pixels.
    padding:
        Extra pixels to add on each side (clamped to frame boundaries).

    Returns
    -------
    tuple[int, int, int, int]
        ``(x_min, y_min, x_max, y_max)`` in pixel coordinates,
        clamped to ``[0, frame_width/frame_height]``.
    """
    x_min, y_min, x_max, y_max = compute_bounding_box(
        landmarks, frame_width, frame_height
    )
    x_min = max(0, int(x_min) - padding)
    y_min = max(0, int(y_min) - padding)
    x_max = min(frame_width, int(x_max) + padding)
    y_max = min(frame_height, int(y_max) + padding)
    return x_min, y_min, x_max, y_max


# ── Conversion helpers ────────────────────────────────────────────────────────

def landmarks_to_pixel_coords(
    landmarks: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    """Convert normalised landmark coordinates to pixel coordinates.

    Only x (col 0) and y (col 1) are scaled; z (col 2) is left unchanged.

    Parameters
    ----------
    landmarks:
        Normalised array of shape ``(21, 3)``.
    frame_width, frame_height:
        Frame dimensions in pixels.

    Returns
    -------
    np.ndarray
        Array of shape ``(21, 3)`` with x, y in pixel space and z unchanged.
    """
    validate_landmark_shape(landmarks, SINGLE_HAND_SHAPE)
    pixel_landmarks = landmarks.copy().astype(np.float32)
    pixel_landmarks[:, 0] *= frame_width
    pixel_landmarks[:, 1] *= frame_height
    return pixel_landmarks


def zero_landmarks() -> np.ndarray:
    """Return a zero-filled array representing a missing/undetected hand.

    Returns
    -------
    np.ndarray
        Zero array of shape ``(21, 3)`` and dtype ``float32``.
    """
    return np.zeros(SINGLE_HAND_SHAPE, dtype=np.float32)


def zero_two_hands() -> np.ndarray:
    """Return a zero-filled array for a frame with no detected hands.

    Returns
    -------
    np.ndarray
        Zero array of shape ``(2, 21, 3)`` and dtype ``float32``.
    """
    return np.zeros(TWO_HAND_SHAPE, dtype=np.float32)
