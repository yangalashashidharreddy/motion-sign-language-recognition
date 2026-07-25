"""
visualize_landmarks.py
======================
Utilities for drawing MediaPipe hand landmarks on video frames and
displaying or saving the annotated output.

All drawing is done with OpenCV and the MediaPipe drawing utilities.
These utilities are intended for **debugging and inspection only** —
they are not part of the training pipeline.

Usage
-----
::

    import cv2
    from src.landmarks.hand_detector import HandDetector
    from src.landmarks.landmark_extractor import LandmarkExtractor
    from src.landmarks.visualize_landmarks import (
        draw_landmarks_on_frame,
        annotate_frame,
        display_frame,
        save_frame,
        visualize_video,
    )

    cap = cv2.VideoCapture("data/raw/videos/00001.mp4")
    ret, frame = cap.read()
    cap.release()

    with HandDetector(max_num_hands=2) as detector:
        results = detector.detect(frame)

    annotated = draw_landmarks_on_frame(frame, results)
    display_frame(annotated, title="Landmarks")
    save_frame(annotated, "debug_frame.jpg")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── MediaPipe drawing solutions ───────────────────────────────────────────────

def _get_mp_drawing():
    """Return ``mediapipe.solutions.drawing_utils`` lazily."""
    try:
        import mediapipe as mp  # noqa: PLC0415
        return mp.solutions.drawing_utils, mp.solutions.drawing_styles, mp.solutions.hands
    except ImportError as exc:
        raise ImportError(
            "MediaPipe is required for landmark visualisation.\n"
            "  → pip install mediapipe"
        ) from exc


def _get_cv2():
    """Return cv2 lazily with a helpful error."""
    try:
        import cv2  # noqa: PLC0415
        return cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for frame visualisation.\n"
            "  → pip install opencv-python"
        ) from exc


# ── Drawing functions ─────────────────────────────────────────────────────────

def draw_landmarks_on_frame(
    frame: np.ndarray,
    results: Any,
    draw_connections: bool = True,
    landmark_color: tuple[int, int, int] = (0, 255, 0),
    connection_color: tuple[int, int, int] = (255, 255, 255),
    landmark_radius: int = 4,
    connection_thickness: int = 2,
    use_mediapipe_style: bool = True,
) -> np.ndarray:
    """Draw detected hand landmarks on a copy of *frame*.

    Parameters
    ----------
    frame:
        BGR frame array of shape ``(H, W, 3)``.
    results:
        Raw MediaPipe result object returned by
        :meth:`~src.landmarks.hand_detector.HandDetector.detect`.
    draw_connections:
        If ``True``, draw the 21-landmark skeleton connections in addition
        to the landmark dots.
    landmark_color:
        BGR color tuple for landmark dots (default: green).
    connection_color:
        BGR color tuple for connection lines (default: white).
    landmark_radius:
        Radius of landmark dots in pixels.
    connection_thickness:
        Thickness of connection lines in pixels.
    use_mediapipe_style:
        If ``True``, use MediaPipe's built-in styled drawing (overrides
        the manual color/thickness parameters above for a richer look).

    Returns
    -------
    np.ndarray
        Annotated frame (copy of input, shape ``(H, W, 3)``).
    """
    cv2 = _get_cv2()
    annotated = frame.copy()

    if results is None or not results.multi_hand_landmarks:
        return annotated

    mp_drawing, mp_styles, mp_hands = _get_mp_drawing()

    for hand_landmarks in results.multi_hand_landmarks:
        if use_mediapipe_style:
            mp_drawing.draw_landmarks(
                annotated,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )
        else:
            h, w = annotated.shape[:2]
            # Draw connection lines first (so dots appear on top)
            if draw_connections:
                for connection in mp_hands.HAND_CONNECTIONS:
                    start_idx, end_idx = connection
                    start_lm = hand_landmarks.landmark[start_idx]
                    end_lm = hand_landmarks.landmark[end_idx]
                    start_px = (int(start_lm.x * w), int(start_lm.y * h))
                    end_px = (int(end_lm.x * w), int(end_lm.y * h))
                    cv2.line(annotated, start_px, end_px, connection_color, connection_thickness)
            # Draw landmark dots
            for lm in hand_landmarks.landmark:
                px = (int(lm.x * w), int(lm.y * h))
                cv2.circle(annotated, px, landmark_radius, landmark_color, -1)

    return annotated


def draw_handedness_labels(
    frame: np.ndarray,
    results: Any,
    font_scale: float = 0.8,
    thickness: int = 2,
) -> np.ndarray:
    """Overlay handedness labels (Left/Right) near each detected wrist.

    Parameters
    ----------
    frame:
        BGR frame of shape ``(H, W, 3)``.
    results:
        Raw MediaPipe result object.
    font_scale:
        OpenCV font scale factor.
    thickness:
        Text stroke thickness.

    Returns
    -------
    np.ndarray
        Annotated frame.
    """
    cv2 = _get_cv2()
    annotated = frame.copy()

    if results is None or not results.multi_hand_landmarks:
        return annotated

    h, w = annotated.shape[:2]
    for hand_lm, handedness in zip(
        results.multi_hand_landmarks,
        results.multi_handedness or [],
    ):
        wrist = hand_lm.landmark[0]
        px = (int(wrist.x * w), max(0, int(wrist.y * h) - 10))
        label = handedness.classification[0].label
        color = (255, 100, 0) if label == "Left" else (0, 100, 255)
        cv2.putText(
            annotated, label, px,
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness,
        )
    return annotated


def annotate_frame(
    frame: np.ndarray,
    results: Any,
    text: Optional[str] = None,
    draw_bboxes: bool = True,
) -> np.ndarray:
    """Full annotation pipeline: landmarks + handedness labels + optional text.

    Parameters
    ----------
    frame:
        BGR frame.
    results:
        MediaPipe result.
    text:
        Optional text to overlay in the top-left corner (e.g. frame number
        or gloss label).
    draw_bboxes:
        If ``True``, draw bounding boxes around each detected hand.

    Returns
    -------
    np.ndarray
        Fully annotated frame.
    """
    cv2 = _get_cv2()
    annotated = draw_landmarks_on_frame(frame, results)
    annotated = draw_handedness_labels(annotated, results)

    if draw_bboxes and results is not None and results.multi_hand_landmarks:
        h, w = annotated.shape[:2]
        for hand_lm in results.multi_hand_landmarks:
            xs = [lm.x * w for lm in hand_lm.landmark]
            ys = [lm.y * h for lm in hand_lm.landmark]
            x_min, x_max = int(min(xs)) - 10, int(max(xs)) + 10
            y_min, y_max = int(min(ys)) - 10, int(max(ys)) + 10
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)
            cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

    if text:
        cv2.putText(
            annotated, text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
        )

    return annotated


# ── Display / save ────────────────────────────────────────────────────────────

def display_frame(
    frame: np.ndarray,
    title: str = "Landmark Visualisation",
    wait_ms: int = 0,
) -> None:
    """Display *frame* in an OpenCV window.

    Parameters
    ----------
    frame:
        BGR frame to show.
    title:
        Window title.
    wait_ms:
        Milliseconds to wait for a key press. ``0`` waits indefinitely.
        Use a positive value (e.g. ``30``) for video playback.

    Notes
    -----
    This function is a no-op in headless environments (e.g. Kaggle) where
    ``cv2.imshow`` is unavailable. A warning is logged instead.
    """
    cv2 = _get_cv2()
    try:
        cv2.imshow(title, frame)
        cv2.waitKey(wait_ms)
    except cv2.error:
        logger.warning(
            "cv2.imshow() failed — likely a headless environment (e.g. Kaggle). "
            "Use save_frame() to write the image to disk instead."
        )


def save_frame(
    frame: np.ndarray,
    path: Path,
    quality: int = 95,
) -> Path:
    """Save *frame* to disk as a JPEG or PNG image.

    Parameters
    ----------
    frame:
        BGR frame array.
    path:
        Output file path. Extension determines format (``.jpg`` or ``.png``).
    quality:
        JPEG quality (1–100). Only used for JPEG output.

    Returns
    -------
    Path
        The resolved path to the saved file.

    Raises
    ------
    IOError
        If the frame could not be written to disk.
    """
    cv2 = _get_cv2()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, max(0, min(9, (100 - quality) // 10))]
    else:
        params = []

    success = cv2.imwrite(str(path), frame, params)
    if not success:
        raise IOError(f"cv2.imwrite failed for path: {path}")

    logger.debug("Frame saved → %s", path)
    return path


# ── Video-level visualiser ────────────────────────────────────────────────────

def visualize_video(
    video_path: Path,
    output_dir: Optional[Path] = None,
    max_frames: Optional[int] = None,
    show: bool = False,
    save_frames: bool = True,
    min_detection_confidence: float = 0.5,
) -> int:
    """Visualise landmark extraction over an entire video.

    For each frame, runs detection and saves / displays the annotated output.

    Parameters
    ----------
    video_path:
        Path to the input video file.
    output_dir:
        Directory to save annotated frames. Required when ``save_frames=True``.
    max_frames:
        Stop after this many frames (``None`` = process all).
    show:
        Display frames interactively (requires a display; not supported on Kaggle).
    save_frames:
        Save each annotated frame as a JPEG image.
    min_detection_confidence:
        Detection confidence threshold.

    Returns
    -------
    int
        Total number of frames processed.
    """
    from src.landmarks.hand_detector import HandDetector  # noqa: PLC0415

    cv2 = _get_cv2()
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    if save_frames and output_dir is None:
        raise ValueError("output_dir must be provided when save_frames=True.")

    frame_count = 0
    with HandDetector(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
    ) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames is not None and frame_count >= max_frames:
                break

            results = detector.detect(frame)
            annotated = annotate_frame(frame, results, text=f"Frame {frame_count:04d}")

            if show:
                display_frame(annotated, title=video_path.name, wait_ms=30)

            if save_frames and output_dir is not None:
                frame_path = Path(output_dir) / f"frame_{frame_count:04d}.jpg"
                save_frame(annotated, frame_path)

            frame_count += 1

    cap.release()
    if show:
        cv2.destroyAllWindows()

    logger.info("Visualised %d frame(s) from %s", frame_count, video_path.name)
    return frame_count
