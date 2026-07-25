"""
save_landmarks.py
=================
Save extracted landmark sequences to disk as NumPy (.npy) or CSV files.

Output structure
----------------
::

    output_dir/
    ├── manifest.csv             ← index of all saved files
    ├── train/
    │   ├── 00001_book.npy       ← (T, 2, 21, 3) float32 array
    │   ├── 00002_drink.npy
    │   └── ...
    ├── val/
    │   └── ...
    └── test/
        └── ...

    For CSV output, each video produces one .csv file where each row is one
    frame and columns encode flattened landmark coordinates.

Usage
-----
::

    from src.landmarks.save_landmarks import LandmarkSaver, process_dataset
    from src.dataset.dataset_config import DatasetConfig

    cfg = DatasetConfig()
    process_dataset(
        annotation_file=cfg.annotation_file,
        videos_dir=cfg.videos_dir,
        output_dir=cfg.processed_dir,
        fmt="npy",
        target_length=30,
        splits=["train", "val"],
    )
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from src.landmarks.landmark_sequence import LandmarkSequence
from src.landmarks.landmark_utils import NUM_COORDS, NUM_LANDMARKS

logger = logging.getLogger(__name__)

# Supported save formats
SaveFormat = Literal["npy", "csv"]


# ── File-naming helper ────────────────────────────────────────────────────────

def _safe_filename(video_id: str, label: str) -> str:
    """Build a safe filename stem from *video_id* and *label*.

    Replaces spaces and special characters with underscores.
    """
    safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
    return f"{video_id}_{safe_label}"


# ── LandmarkSaver ─────────────────────────────────────────────────────────────

class LandmarkSaver:
    """Save :class:`~src.landmarks.landmark_sequence.LandmarkSequence` objects to disk.

    Parameters
    ----------
    output_dir:
        Root directory where landmark files will be written.
        Sub-directories are created automatically.
    fmt:
        Output format: ``"npy"`` (binary NumPy) or ``"csv"`` (text).
    overwrite:
        If ``False`` (default), existing files are skipped.
        If ``True``, existing files are overwritten.
    """

    def __init__(
        self,
        output_dir: Path,
        fmt: SaveFormat = "npy",
        overwrite: bool = False,
    ) -> None:
        if fmt not in {"npy", "csv"}:
            raise ValueError(f"fmt must be 'npy' or 'csv', got {fmt!r}.")
        self.output_dir = Path(output_dir)
        self.fmt = fmt
        self.overwrite = overwrite

    # ── NumPy save ────────────────────────────────────────────────────────────

    def save_npy(
        self,
        seq: LandmarkSequence,
        split: str = "train",
    ) -> Optional[Path]:
        """Save a landmark sequence as a ``.npy`` binary file.

        The array saved is ``seq.sequence`` of shape ``(T, 2, 21, 3)``
        with dtype ``float32``.

        Parameters
        ----------
        seq:
            The :class:`~src.landmarks.landmark_sequence.LandmarkSequence` to save.
        split:
            Dataset split name (``"train"``, ``"val"``, or ``"test"``).
            Used to organise files into sub-directories.

        Returns
        -------
        Path | None
            The path to the saved file, or ``None`` if the file was skipped
            (because it already exists and ``overwrite=False``).
        """
        split_dir = self.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        filename = _safe_filename(seq.video_id, seq.label) + ".npy"
        filepath = split_dir / filename

        if filepath.exists() and not self.overwrite:
            logger.debug("Skipping (already exists): %s", filepath)
            return None

        np.save(filepath, seq.sequence)
        logger.debug("Saved .npy: %s | shape=%s", filepath, seq.sequence.shape)
        return filepath

    # ── CSV save ──────────────────────────────────────────────────────────────

    def save_csv(
        self,
        seq: LandmarkSequence,
        split: str = "train",
    ) -> Optional[Path]:
        """Save a landmark sequence as a ``.csv`` text file.

        CSV format
        ----------
        One row per frame. Columns:

        - ``frame_idx`` — 0-based frame index.
        - ``left_x0`` … ``left_z20`` — 63 columns for left-hand landmarks (x,y,z × 21).
        - ``right_x0`` … ``right_z20`` — 63 columns for right-hand landmarks.

        Total: 1 + 63 + 63 = 127 columns per row.

        Parameters
        ----------
        seq:
            The sequence to save.
        split:
            Dataset split sub-directory name.

        Returns
        -------
        Path | None
            Path to the saved CSV file, or ``None`` if skipped.
        """
        split_dir = self.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        filename = _safe_filename(seq.video_id, seq.label) + ".csv"
        filepath = split_dir / filename

        if filepath.exists() and not self.overwrite:
            logger.debug("Skipping (already exists): %s", filepath)
            return None

        # Build CSV header
        left_cols = [
            f"left_{'xyz'[c]}{lm}"
            for lm in range(NUM_LANDMARKS)
            for c in range(NUM_COORDS)
        ]
        right_cols = [
            f"right_{'xyz'[c]}{lm}"
            for lm in range(NUM_LANDMARKS)
            for c in range(NUM_COORDS)
        ]
        header = ["frame_idx"] + left_cols + right_cols

        with filepath.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for frame_idx, frame in enumerate(seq.sequence):
                # frame shape: (2, 21, 3) — index 0=left, 1=right
                left_flat = frame[0].flatten().tolist()
                right_flat = frame[1].flatten().tolist()
                writer.writerow([frame_idx] + left_flat + right_flat)

        logger.debug("Saved .csv: %s | frames=%d", filepath, len(seq.sequence))
        return filepath

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def save(
        self,
        seq: LandmarkSequence,
        split: str = "train",
    ) -> Optional[Path]:
        """Save a sequence in the configured format.

        Dispatches to :meth:`save_npy` or :meth:`save_csv` based on ``self.fmt``.

        Parameters
        ----------
        seq:
            Sequence to save.
        split:
            Dataset split name.

        Returns
        -------
        Path | None
            Path to the saved file, or ``None`` if skipped.
        """
        if self.fmt == "npy":
            return self.save_npy(seq, split)
        return self.save_csv(seq, split)


# ── Manifest writer ───────────────────────────────────────────────────────────

def write_manifest(
    records: list[dict],
    output_dir: Path,
) -> Path:
    """Write a manifest CSV file that indexes all saved landmark files.

    Parameters
    ----------
    records:
        List of dicts, each with at minimum:
        ``{"video_id", "label", "split", "filepath", "num_frames",
        "num_detected", "detection_rate", "was_padded", "was_truncated"}``.
    output_dir:
        Root output directory.

    Returns
    -------
    Path
        Path to the written ``manifest.csv``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"

    if not records:
        logger.warning("No records to write to manifest.")
        return manifest_path

    fieldnames = list(records[0].keys())
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    logger.info("Manifest written → %s  (%d records)", manifest_path, len(records))
    return manifest_path


# ── Dataset-level processing function ────────────────────────────────────────

def process_dataset(
    annotation_file: Path,
    videos_dir: Path,
    output_dir: Path,
    fmt: SaveFormat = "npy",
    target_length: int = 30,
    splits: Optional[list[str]] = None,
    max_classes: Optional[int] = None,
    max_videos: Optional[int] = None,
    overwrite: bool = False,
    min_detection_confidence: float = 0.5,
    pad_mode: str = "repeat",
) -> dict[str, int]:
    """Extract and save landmarks for the full WLASL dataset.

    This is the top-level entry point for batch landmark extraction.
    It reads the WLASL annotation JSON, processes each video, extracts
    hand landmarks, and saves the results to *output_dir*.

    Parameters
    ----------
    annotation_file:
        Path to ``WLASL_v0.3.json``.
    videos_dir:
        Directory containing ``<video_id>.mp4`` files.
    output_dir:
        Root directory for landmark output files.
    fmt:
        ``"npy"`` or ``"csv"``.
    target_length:
        Number of frames to sample per video (pad/truncate to this value).
    splits:
        Which splits to process (default: ``["train", "val", "test"]``).
    max_classes:
        Restrict to the first N glosses (alphabetically). ``None`` = all.
    max_videos:
        Stop after processing this many videos total (for debugging).
    overwrite:
        Overwrite existing landmark files.
    min_detection_confidence:
        MediaPipe detection confidence threshold.
    pad_mode:
        Padding mode: ``"zero"``, ``"repeat"``, or ``"reflect"``.

    Returns
    -------
    dict[str, int]
        Summary counts: ``{"processed", "skipped", "missing", "errors"}``.
    """
    # Deferred imports
    import json  # noqa: PLC0415
    from src.landmarks.hand_detector import HandDetector  # noqa: PLC0415
    from src.landmarks.landmark_extractor import LandmarkExtractor  # noqa: PLC0415
    from src.landmarks.landmark_sequence import LandmarkSequenceBuilder  # noqa: PLC0415

    if splits is None:
        splits = ["train", "val", "test"]
    splits_set = set(splits)

    if not annotation_file.is_file():
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
    if not videos_dir.is_dir():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")

    with annotation_file.open("r", encoding="utf-8") as fh:
        entries: list[dict] = json.load(fh)

    # Build sorted gloss list
    glosses = sorted({e["gloss"] for e in entries})
    if max_classes is not None:
        glosses = glosses[:max_classes]
    gloss_set = set(glosses)

    saver = LandmarkSaver(output_dir=output_dir, fmt=fmt, overwrite=overwrite)
    counters = {"processed": 0, "skipped": 0, "missing": 0, "errors": 0}
    manifest_records: list[dict] = []

    with HandDetector(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
    ) as detector:
        extractor = LandmarkExtractor(detector, num_hands=2)
        builder = LandmarkSequenceBuilder(
            extractor=extractor,
            target_length=target_length,
            sample_frames=True,
            pad_mode=pad_mode,  # type: ignore[arg-type]
        )

        for entry in entries:
            gloss = entry.get("gloss", "")
            if gloss not in gloss_set:
                continue

            for instance in entry.get("instances", []):
                split = instance.get("split", "").lower()
                if split not in splits_set:
                    continue

                if max_videos is not None and sum(counters.values()) >= max_videos:
                    break

                video_id = str(instance.get("video_id", "")).zfill(5)

                # Find the video file
                video_path: Optional[Path] = None
                for ext in (".mp4", ".avi", ".mov", ".webm", ".mkv"):
                    candidate = videos_dir / f"{video_id}{ext}"
                    if candidate.is_file():
                        video_path = candidate
                        break

                if video_path is None:
                    counters["missing"] += 1
                    continue

                try:
                    seq = builder.build_from_video(
                        video_path=video_path,
                        video_id=video_id,
                        label=gloss,
                    )
                    saved_path = saver.save(seq, split=split)

                    if saved_path is None:
                        counters["skipped"] += 1
                    else:
                        counters["processed"] += 1
                        manifest_records.append({
                            "video_id": video_id,
                            "label": gloss,
                            "split": split,
                            "filepath": str(saved_path),
                            "num_frames": seq.num_frames,
                            "source_frames": seq.source_frame_count,
                            "num_detected": seq.num_detected,
                            "detection_rate": round(seq.detection_rate, 4),
                            "was_padded": seq.was_padded,
                            "was_truncated": seq.was_truncated,
                        })

                    if counters["processed"] % 50 == 0 and counters["processed"] > 0:
                        logger.info(
                            "Progress: %s", {k: v for k, v in counters.items()}
                        )

                except Exception as exc:
                    logger.warning("Error processing %s (%s): %s", video_id, gloss, exc)
                    counters["errors"] += 1

    if manifest_records:
        write_manifest(manifest_records, output_dir)

    logger.info("Landmark extraction complete: %s", counters)
    return counters


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import logging as _logging

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Extract and save WLASL hand landmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fmt", choices=["npy", "csv"], default="npy")
    parser.add_argument("--target_length", type=int, default=30)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--max_classes", type=int, default=None)
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--pad_mode", choices=["zero", "repeat", "reflect"], default="repeat")
    args = parser.parse_args()

    from src.dataset.dataset_config import get_default_config  # noqa: PLC0415
    cfg = get_default_config()

    summary = process_dataset(
        annotation_file=cfg.annotation_file,
        videos_dir=cfg.videos_dir,
        output_dir=cfg.processed_dir,
        fmt=args.fmt,
        target_length=args.target_length,
        splits=args.splits,
        max_classes=args.max_classes,
        max_videos=args.max_videos,
        overwrite=args.overwrite,
        pad_mode=args.pad_mode,
    )
    print("\nExtraction Summary:", summary)
