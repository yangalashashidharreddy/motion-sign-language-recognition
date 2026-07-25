"""
landmark_sequence.py
====================
Convert full video files into per-frame landmark sequences.

This module provides :class:`LandmarkSequenceBuilder` which opens a video,
iterates over every frame (or a uniform sample of frames), extracts hand
landmarks via :class:`~src.landmarks.landmark_extractor.LandmarkExtractor`,
and assembles the results into a single NumPy array of shape
``(T, 2, 21, 3)``.

Output array layout
-------------------
::

    (T, 2, 21, 3)
     │  │  │   └── x, y, z coordinates
     │  │  └────── 21 MediaPipe hand landmarks (0 = wrist … 20 = pinky tip)
     │  └───────── 2 hands (index 0 = left, index 1 = right)
     └──────────── T frames (padded / truncated to target_length if set)

Usage
-----
::

    from src.landmarks.hand_detector import HandDetector
    from src.landmarks.landmark_extractor import LandmarkExtractor
    from src.landmarks.landmark_sequence import LandmarkSequenceBuilder
    from pathlib import Path

    with HandDetector(max_num_hands=2) as detector:
        extractor = LandmarkExtractor(detector)
        builder = LandmarkSequenceBuilder(
            extractor=extractor,
            target_length=30,       # pad/truncate to 30 frames
            pad_mode="repeat",      # repeat last frame for padding
        )
        seq = builder.build_from_video(
            video_path=Path("data/raw/videos/00001.mp4"),
            video_id="00001",
            label="book",
        )
        print(seq.sequence.shape)   # (30, 2, 21, 3)
        print(seq.num_detected)     # frames where ≥1 hand was detected
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from src.landmarks.landmark_extractor import LandmarkExtractor
from src.landmarks.landmark_utils import NUM_COORDS, NUM_LANDMARKS, zero_two_hands

logger = logging.getLogger(__name__)

# Supported padding modes
PadMode = Literal["zero", "repeat", "reflect"]


# ── Data container ────────────────────────────────────────────────────────────

@dataclass
class LandmarkSequence:
    """Container for a single video's landmark sequence.

    Attributes
    ----------
    video_id:
        Unique identifier (typically the WLASL video ID, e.g. ``"00001"``).
    label:
        Gloss label (e.g. ``"book"``).
    sequence:
        Landmark array of shape ``(T, 2, 21, 3)`` where T is the number of
        frames after optional padding/truncation.
    source_frame_count:
        Total frames in the original video before sampling/padding.
    num_detected:
        Number of frames where at least one hand was detected.
    was_padded:
        ``True`` if the sequence was shorter than ``target_length`` and was padded.
    was_truncated:
        ``True`` if the sequence was longer than ``target_length`` and was truncated.
    """

    video_id: str
    label: str
    sequence: np.ndarray              # (T, 2, 21, 3) float32
    source_frame_count: int = 0
    num_detected: int = 0
    was_padded: bool = False
    was_truncated: bool = False

    @property
    def num_frames(self) -> int:
        """Number of frames in the (possibly padded) sequence."""
        return self.sequence.shape[0]

    @property
    def detection_rate(self) -> float:
        """Fraction of frames where at least one hand was detected."""
        if self.num_frames == 0:
            return 0.0
        return self.num_detected / self.num_frames

    def flattened(self) -> np.ndarray:
        """Return sequence flattened to shape ``(T, 126)`` — 2 hands × 21 × 3."""
        T = self.sequence.shape[0]
        return self.sequence.reshape(T, -1)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"LandmarkSequence("
            f"id={self.video_id!r}, label={self.label!r}, "
            f"shape={self.sequence.shape}, "
            f"detected={self.num_detected}/{self.num_frames})"
        )


# ── Builder ───────────────────────────────────────────────────────────────────

class LandmarkSequenceBuilder:
    """Process a video file into a padded/truncated landmark sequence.

    Parameters
    ----------
    extractor:
        An instantiated :class:`~src.landmarks.landmark_extractor.LandmarkExtractor`.
    target_length:
        If set, all sequences are padded or truncated to exactly this many frames.
        If ``None``, the raw frame count is preserved (sequences will be variable-length).
    sample_frames:
        If ``True`` and ``target_length`` is set, frames are uniformly sampled
        from the video instead of processing every frame then truncating.
        This is faster for long videos.
    pad_mode:
        Padding strategy when the video has fewer frames than ``target_length``:

        - ``"zero"``    — fill with all-zero frames.
        - ``"repeat"``  — repeat the last frame.
        - ``"reflect"`` — mirror the sequence (numpy reflect mode).
    skip_empty:
        If ``True``, frames where no hands are detected are still included in
        the sequence (as zero arrays). Set to ``False`` to skip them — this
        changes the effective sequence length and may interfere with
        ``target_length``.

    Raises
    ------
    ValueError
        If ``target_length`` is ≤ 0 or ``pad_mode`` is invalid.
    """

    _VALID_PAD_MODES: frozenset[str] = frozenset({"zero", "repeat", "reflect"})

    def __init__(
        self,
        extractor: LandmarkExtractor,
        target_length: Optional[int] = None,
        sample_frames: bool = True,
        pad_mode: PadMode = "repeat",
        skip_empty: bool = False,
    ) -> None:
        if not isinstance(extractor, LandmarkExtractor):
            raise TypeError("extractor must be a LandmarkExtractor instance.")
        if target_length is not None and target_length <= 0:
            raise ValueError(f"target_length must be a positive integer, got {target_length}.")
        if pad_mode not in self._VALID_PAD_MODES:
            raise ValueError(
                f"pad_mode must be one of {sorted(self._VALID_PAD_MODES)}, got {pad_mode!r}."
            )

        self.extractor = extractor
        self.target_length = target_length
        self.sample_frames = sample_frames
        self.pad_mode: PadMode = pad_mode
        self.skip_empty = skip_empty

    # ── Frame sampling ────────────────────────────────────────────────────────

    @staticmethod
    def _uniform_indices(total: int, count: int) -> list[int]:
        """Return *count* evenly-spaced integer indices in ``[0, total)``.

        When ``total < count``, indices are repeated cyclically.
        """
        if total >= count:
            return np.linspace(0, total - 1, count, dtype=int).tolist()
        base = np.arange(total)
        return np.resize(base, count).tolist()

    # ── Padding / truncation ──────────────────────────────────────────────────

    def _pad_sequence(
        self,
        frames: list[np.ndarray],
        target: int,
    ) -> tuple[list[np.ndarray], bool]:
        """Pad *frames* to *target* length. Returns (padded_list, was_padded)."""
        if len(frames) >= target:
            return frames, False

        deficit = target - len(frames)

        if self.pad_mode == "zero":
            padding = [zero_two_hands() for _ in range(deficit)]
        elif self.pad_mode == "repeat":
            last = frames[-1] if frames else zero_two_hands()
            padding = [last.copy() for _ in range(deficit)]
        elif self.pad_mode == "reflect":
            # Use numpy's reflect to mirror the existing frames
            arr = np.stack(frames, axis=0)              # (T, 2, 21, 3)
            pad_width = [(0, deficit)] + [(0, 0)] * (arr.ndim - 1)
            padded_arr = np.pad(arr, pad_width, mode="reflect")
            return list(padded_arr), True
        else:  # pragma: no cover
            padding = [zero_two_hands() for _ in range(deficit)]

        return frames + padding, True

    def _truncate_sequence(
        self,
        frames: list[np.ndarray],
        target: int,
    ) -> tuple[list[np.ndarray], bool]:
        """Truncate *frames* to *target* length. Returns (truncated_list, was_truncated)."""
        if len(frames) <= target:
            return frames, False
        return frames[:target], True

    # ── Core builder ──────────────────────────────────────────────────────────

    def build_from_video(
        self,
        video_path: Path,
        video_id: str = "",
        label: str = "",
    ) -> LandmarkSequence:
        """Extract landmarks from every (or sampled) frame of a video file.

        Parameters
        ----------
        video_path:
            Absolute or relative path to the video file.
        video_id:
            Unique identifier string (preserved in the output container).
        label:
            Gloss label string (preserved in the output container).

        Returns
        -------
        LandmarkSequence
            Container with the ``(T, 2, 21, 3)`` landmark array and metadata.

        Raises
        ------
        FileNotFoundError
            If *video_path* does not exist.
        RuntimeError
            If the video cannot be opened by OpenCV.
        """
        try:
            import cv2  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "OpenCV is required for video processing.\n"
                "  → pip install opencv-python"
            ) from exc

        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            logger.warning("Video has 0 frames: %s — returning empty sequence.", video_path)
            cap.release()
            return self._empty_sequence(video_id, label)

        logger.debug("Processing video: %s | total_frames=%d", video_path.name, total_frames)

        # Determine which frames to read
        if self.sample_frames and self.target_length is not None:
            frame_indices = self._uniform_indices(total_frames, self.target_length)
            use_sampling = True
        else:
            frame_indices = list(range(total_frames))
            use_sampling = False

        frames: list[np.ndarray] = []
        num_detected = 0

        if use_sampling:
            # Seek-based reading
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    frames.append(zero_two_hands())
                    continue
                two_hands = self.extractor.extract_two_hands(frame)
                detected = not np.all(two_hands == 0)
                if detected:
                    num_detected += 1
                if not self.skip_empty or detected:
                    frames.append(two_hands)
        else:
            # Sequential reading (more accurate for tracked video)
            frame_set = set(frame_indices)
            current_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if current_idx in frame_set:
                    two_hands = self.extractor.extract_two_hands(frame)
                    detected = not np.all(two_hands == 0)
                    if detected:
                        num_detected += 1
                    if not self.skip_empty or detected:
                        frames.append(two_hands)
                current_idx += 1

        cap.release()

        if not frames:
            logger.warning("No frames extracted from: %s", video_path)
            return self._empty_sequence(video_id, label)

        # Apply padding / truncation
        was_padded = was_truncated = False
        if self.target_length is not None:
            frames, was_truncated = self._truncate_sequence(frames, self.target_length)
            frames, was_padded = self._pad_sequence(frames, self.target_length)

        sequence = np.stack(frames, axis=0).astype(np.float32)  # (T, 2, 21, 3)

        logger.debug(
            "Built sequence for %s | shape=%s | detected=%d/%d | "
            "padded=%s | truncated=%s",
            video_id, sequence.shape, num_detected, len(frames),
            was_padded, was_truncated,
        )

        return LandmarkSequence(
            video_id=video_id,
            label=label,
            sequence=sequence,
            source_frame_count=total_frames,
            num_detected=num_detected,
            was_padded=was_padded,
            was_truncated=was_truncated,
        )

    def _empty_sequence(self, video_id: str, label: str) -> LandmarkSequence:
        """Return an empty (all-zeros) :class:`LandmarkSequence`."""
        T = self.target_length or 1
        return LandmarkSequence(
            video_id=video_id,
            label=label,
            sequence=np.zeros((T, 2, NUM_LANDMARKS, NUM_COORDS), dtype=np.float32),
            source_frame_count=0,
            num_detected=0,
        )
