"""
dataset.py
==========
PyTorch Dataset class for the WLASL (Word-Level American Sign Language) dataset.

The :class:`WLASLDataset` class:
    - Reads the WLASL JSON annotation file.
    - Filters instances by split (``"train"`` / ``"val"`` / ``"test"``).
    - Loads video frames using OpenCV (no MediaPipe, no landmark extraction).
    - Uniformly samples a fixed number of frames per clip.
    - Returns ``(frames_tensor, label_index)`` pairs for use with a DataLoader.

Frame tensor shape: ``(T, 3, H, W)`` — T frames, 3 RGB channels, H×W pixels.

Missing videos are skipped with a warning during dataset construction.
Frame loading errors are handled gracefully at ``__getitem__`` time.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


# ── Default transforms ────────────────────────────────────────────────────────

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    """Return augmentation transforms for training.

    Parameters
    ----------
    img_size:
        Target spatial resolution (height and width) in pixels.

    Returns
    -------
    transforms.Compose
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def get_val_transforms(img_size: int = 224) -> transforms.Compose:
    """Return deterministic transforms for validation / test.

    Parameters
    ----------
    img_size:
        Target spatial resolution in pixels.

    Returns
    -------
    transforms.Compose
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


# ── Dataset ───────────────────────────────────────────────────────────────────

class WLASLDataset(Dataset):
    """PyTorch Dataset for the WLASL video dataset.

    Parameters
    ----------
    annotation_file:
        Path to the WLASL JSON annotation file (e.g. ``WLASL_v0.3.json``).
    videos_dir:
        Directory that contains ``<video_id>.mp4`` files.
    split:
        Data partition to load: ``"train"``, ``"val"``, or ``"test"``.
    num_frames:
        Number of frames to uniformly sample from each clip (default: 16).
    transform:
        Optional torchvision transform applied to each individual frame.
        If ``None``, raw uint8 tensors are returned (not recommended).
    label_to_idx:
        Optional pre-built mapping of ``{gloss: class_index}``.
        When ``None``, a mapping is built alphabetically from the annotation
        file (consistent order is guaranteed across splits when the same
        annotation file is used).
    max_classes:
        If provided, only keep the first *max_classes* glosses (sorted
        alphabetically). Useful for training on WLASL100 / WLASL300 subsets.

    Attributes
    ----------
    samples : list[tuple[Path, int]]
        ``(video_path, class_index)`` pairs for every valid instance.
    label_to_idx : dict[str, int]
        Mapping from gloss string to integer class index.
    idx_to_label : dict[int, str]
        Reverse mapping from integer class index to gloss string.
    num_classes : int
        Total number of distinct classes in this split's label set.
    """

    def __init__(
        self,
        annotation_file: Path,
        videos_dir: Path,
        split: str,
        num_frames: int = 16,
        transform: Optional[Callable] = None,
        label_to_idx: Optional[dict[str, int]] = None,
        max_classes: Optional[int] = None,
    ) -> None:
        self.annotation_file = Path(annotation_file)
        self.videos_dir = Path(videos_dir)
        self.split = split.lower()
        self.num_frames = num_frames
        self.transform = transform

        if self.split not in {"train", "val", "test"}:
            raise ValueError(
                f"split must be 'train', 'val', or 'test', got: {self.split!r}"
            )

        raw_entries = self._load_json()
        self.label_to_idx, self.idx_to_label = self._build_label_map(
            raw_entries, label_to_idx, max_classes
        )
        self.num_classes = len(self.label_to_idx)
        self.samples = self._build_sample_list(raw_entries)

        logger.info(
            "WLASLDataset[%s] | classes=%d | samples=%d | num_frames=%d",
            self.split, self.num_classes, len(self.samples), self.num_frames,
        )

    # ── Internal builders ─────────────────────────────────────────────────────

    def _load_json(self) -> list[dict]:
        """Load and parse the annotation JSON file."""
        if not self.annotation_file.is_file():
            raise FileNotFoundError(
                f"Annotation file not found: {self.annotation_file}\n"
                "  → Ensure WLASL_v0.3.json is placed inside data/raw/ (local) "
                "or /kaggle/input/wlasl-complete/ (Kaggle)."
            )
        with self.annotation_file.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError("WLASL annotation JSON must be a top-level array.")
        return data  # type: ignore[return-value]

    def _build_label_map(
        self,
        entries: list[dict],
        label_to_idx: Optional[dict[str, int]],
        max_classes: Optional[int],
    ) -> tuple[dict[str, int], dict[int, str]]:
        """Build or validate the gloss → index mapping."""
        if label_to_idx is not None:
            idx_to_label = {v: k for k, v in label_to_idx.items()}
            return label_to_idx, idx_to_label

        glosses = sorted({e["gloss"] for e in entries})
        if max_classes is not None:
            glosses = glosses[:max_classes]

        label_to_idx = {gloss: idx for idx, gloss in enumerate(glosses)}
        idx_to_label = {idx: gloss for gloss, idx in label_to_idx.items()}
        return label_to_idx, idx_to_label

    def _build_sample_list(self, entries: list[dict]) -> list[tuple[Path, int]]:
        """Return (video_path, class_index) for every valid video in *split*."""
        samples: list[tuple[Path, int]] = []
        missing_count = 0
        skipped_class = 0

        for entry in entries:
            gloss = entry.get("gloss", "")
            if gloss not in self.label_to_idx:
                skipped_class += 1
                continue
            label_idx = self.label_to_idx[gloss]

            for instance in entry.get("instances", []):
                if instance.get("split", "").lower() != self.split:
                    continue

                video_id = str(instance.get("video_id", "")).zfill(5)
                # Search for video file with any supported extension
                video_path = self._find_video_file(video_id)
                if video_path is None:
                    missing_count += 1
                    continue
                samples.append((video_path, label_idx))

        if missing_count > 0:
            logger.warning(
                "[%s] %d video file(s) not found in %s and were skipped.",
                self.split, missing_count, self.videos_dir,
            )
        if skipped_class > 0:
            logger.debug(
                "[%s] %d entries skipped (gloss not in label map).",
                self.split, skipped_class,
            )
        return samples

    def _find_video_file(self, video_id: str) -> Optional[Path]:
        """Return the path to a video file for *video_id*, or ``None``."""
        for ext in (".mp4", ".avi", ".mov", ".webm", ".mkv"):
            candidate = self.videos_dir / f"{video_id}{ext}"
            if candidate.is_file():
                return candidate
        return None

    # ── Frame sampling ────────────────────────────────────────────────────────

    def _load_frames(self, video_path: Path) -> torch.Tensor:
        """Load and uniformly sample frames from a video file.

        Parameters
        ----------
        video_path:
            Absolute path to the video file.

        Returns
        -------
        torch.Tensor
            Frame tensor of shape ``(T, 3, H, W)`` after transforms,
            or a zero tensor of the same shape if loading fails.
        """
        try:
            import cv2  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "OpenCV is required for video loading.\n"
                "  → Install it with: pip install opencv-python"
            ) from exc

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning("Cannot open video: %s — returning zeros.", video_path)
            return self._zero_frames()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            logger.warning("Video has 0 frames: %s — returning zeros.", video_path)
            cap.release()
            return self._zero_frames()

        # Uniformly sample indices across the video duration
        indices = self._sample_indices(total_frames, self.num_frames)

        frames: list[torch.Tensor] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                # Duplicate last good frame if seek fails
                if frames:
                    frames.append(frames[-1].clone())
                else:
                    frames.append(self._blank_frame())
                continue
            # OpenCV reads BGR; convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # (H, W, 3) uint8
            if self.transform is not None:
                tensor = self.transform(frame_rgb)               # (3, H, W) float32
            else:
                tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
            frames.append(tensor)

        cap.release()
        return torch.stack(frames, dim=0)  # (T, 3, H, W)

    @staticmethod
    def _sample_indices(total_frames: int, num_frames: int) -> list[int]:
        """Uniformly sample *num_frames* indices from [0, total_frames).

        If ``total_frames < num_frames``, frames are repeated cyclically.
        """
        if total_frames >= num_frames:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        else:
            # Repeat frames when clip is shorter than requested
            indices_base = np.arange(total_frames)
            indices = np.resize(indices_base, num_frames)
        return indices.tolist()

    def _zero_frames(self) -> torch.Tensor:
        """Return a zero tensor of shape ``(T, 3, H, W)``."""
        # Infer spatial size from the transform if possible, else default 224
        h = w = 224
        return torch.zeros(self.num_frames, 3, h, w)

    def _blank_frame(self) -> torch.Tensor:
        """Return a single blank frame tensor of shape ``(3, H, W)``."""
        h = w = 224
        return torch.zeros(3, h, w)

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Return the number of valid video clips in this split."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Return the *index*-th (frames_tensor, label_index) pair.

        Parameters
        ----------
        index:
            Dataset index in ``[0, len(self))``.

        Returns
        -------
        tuple[torch.Tensor, int]
            - ``frames``: shape ``(T, 3, H, W)``
            - ``label``:  integer class index
        """
        video_path, label = self.samples[index]
        frames = self._load_frames(video_path)
        return frames, label

    # ── Utilities ─────────────────────────────────────────────────────────────

    def get_class_name(self, idx: int) -> str:
        """Return the gloss string for a given class index."""
        return self.idx_to_label.get(idx, f"<unknown:{idx}>")

    def class_counts(self) -> dict[str, int]:
        """Return ``{gloss: count}`` for each class in this split."""
        counts: dict[str, int] = {}
        for _, label_idx in self.samples:
            gloss = self.idx_to_label[label_idx]
            counts[gloss] = counts.get(gloss, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"WLASLDataset(split={self.split!r}, "
            f"samples={len(self.samples)}, "
            f"classes={self.num_classes}, "
            f"num_frames={self.num_frames})"
        )
