"""
dataset_config.py
=================
Centralised, environment-aware path configuration for the WLASL dataset.

This module provides a single :class:`DatasetConfig` object that resolves all
dataset paths correctly whether the code is running:

* **Locally** — paths rooted at the repository's ``data/`` directory.
* **On Kaggle** — paths rooted at ``/kaggle/input/<slug>/``.
* **Custom** — paths supplied via environment variables or constructor arguments.

Priority order (highest → lowest)
----------------------------------
1. Constructor keyword arguments (``raw_dir``, ``annotation_file``, ``videos_dir``).
2. Environment variables (``WLASL_RAW_DIR``, ``WLASL_ANNOTATION_FILE``,
   ``WLASL_VIDEOS_DIR``, ``WLASL_OUTPUT_DIR``).
3. Auto-detection: Kaggle environment → Kaggle paths; otherwise → local paths.

Usage
-----
::

    from src.dataset.dataset_config import DatasetConfig

    # Auto-detected (works on both Kaggle and local)
    cfg = DatasetConfig()

    # Override dataset slug (if your Kaggle dataset has a different name)
    cfg = DatasetConfig(kaggle_dataset_slug="my-wlasl")

    # Fully manual override
    cfg = DatasetConfig(
        raw_dir="/mnt/nas/wlasl/raw",
        annotation_file="/mnt/nas/wlasl/raw/WLASL_v0.3.json",
        videos_dir="/mnt/nas/wlasl/raw/videos",
    )

    # Validate all paths and print a status report
    report = cfg.validate()

Environment Variables
---------------------
``WLASL_RAW_DIR``
    Override the raw data root directory.

``WLASL_ANNOTATION_FILE``
    Override the full path to the WLASL annotation JSON.

``WLASL_VIDEOS_DIR``
    Override the videos directory.

``WLASL_OUTPUT_DIR``
    Override the output directory (checkpoints, logs, results).

``KAGGLE_DATASET_SLUG``
    Override the Kaggle dataset slug (default: ``wlasl-complete``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Constants ─────────────────────────────────────────────────────────────────

#: Absolute path to the repository root (two levels up from this file).
_REPO_ROOT: Path = Path(__file__).resolve().parents[2]

#: Standard Kaggle input mount point.
_KAGGLE_INPUT_ROOT: Path = Path("/kaggle/input")

#: Standard Kaggle working directory (writable).
_KAGGLE_WORKING_ROOT: Path = Path("/kaggle/working")

#: Default Kaggle dataset slug for WLASL.
_DEFAULT_KAGGLE_SLUG: str = "wlasl-complete"

#: Known WLASL annotation file names (searched in order).
_ANNOTATION_CANDIDATES: tuple[str, ...] = (
    "WLASL_v0.3.json",
    "WLASL_v0.2.json",
    "WLASL_v0.1.json",
    "WLASL.json",
)


# ── Environment detection ─────────────────────────────────────────────────────

def _is_kaggle_environment() -> bool:
    """Return ``True`` if running inside a Kaggle Notebook or Kaggle worker.

    Detection heuristics (any one is sufficient):

    * The ``KAGGLE_KERNEL_RUN_TYPE`` environment variable is set (always set
      by Kaggle's container).
    * The ``/kaggle/input`` directory exists on the filesystem.
    """
    return (
        os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
        or _KAGGLE_INPUT_ROOT.exists()
    )


# ── Annotation file resolver ──────────────────────────────────────────────────

def _find_annotation_file(raw_dir: Path) -> Optional[Path]:
    """Search *raw_dir* for a WLASL annotation JSON file.

    Parameters
    ----------
    raw_dir:
        Directory to search.

    Returns
    -------
    Path | None
        The first matching file, or ``None`` if not found.
    """
    for name in _ANNOTATION_CANDIDATES:
        candidate = raw_dir / name
        if candidate.is_file():
            return candidate
    # Fallback: any top-level JSON
    jsons = sorted(raw_dir.glob("*.json"))
    return jsons[0] if jsons else None


# ── Main configuration class ──────────────────────────────────────────────────

@dataclass
class DatasetConfig:
    """Environment-aware path configuration for the WLASL dataset.

    All path parameters accept either a ``str`` or ``pathlib.Path``.
    ``None`` means "use the auto-detected default".

    Parameters
    ----------
    raw_dir:
        Root directory containing the annotation JSON and ``videos/`` folder.
    annotation_file:
        Full path to the WLASL annotation JSON.
    videos_dir:
        Directory containing the ``.mp4`` video files.
    output_dir:
        Directory where checkpoints, logs, and results are saved.
    kaggle_dataset_slug:
        The Kaggle dataset slug used to build the Kaggle mount path.
        Overridden by the ``KAGGLE_DATASET_SLUG`` environment variable.
    """

    raw_dir: Optional[Path] = field(default=None)
    annotation_file: Optional[Path] = field(default=None)
    videos_dir: Optional[Path] = field(default=None)
    output_dir: Optional[Path] = field(default=None)
    kaggle_dataset_slug: str = field(default=_DEFAULT_KAGGLE_SLUG)

    def __post_init__(self) -> None:
        # Allow callers to pass str values; normalise to Path
        if self.raw_dir is not None:
            self.raw_dir = Path(self.raw_dir)
        if self.annotation_file is not None:
            self.annotation_file = Path(self.annotation_file)
        if self.videos_dir is not None:
            self.videos_dir = Path(self.videos_dir)
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)

        # Allow the Kaggle slug to be overridden via environment variable
        env_slug = os.environ.get("KAGGLE_DATASET_SLUG")
        if env_slug:
            self.kaggle_dataset_slug = env_slug

        # Detect environment once
        self._is_kaggle: bool = _is_kaggle_environment()

        # Resolve all paths
        self._resolve_paths()

    # ── Resolution logic ──────────────────────────────────────────────────────

    def _resolve_paths(self) -> None:
        """Resolve and finalise all path attributes."""
        # raw_dir: constructor → env-var → auto-detect
        if self.raw_dir is None:
            env_raw = os.environ.get("WLASL_RAW_DIR")
            if env_raw:
                self.raw_dir = Path(env_raw)
            elif self._is_kaggle:
                self.raw_dir = _KAGGLE_INPUT_ROOT / self.kaggle_dataset_slug
            else:
                self.raw_dir = _REPO_ROOT / "data" / "raw"

        # annotation_file: constructor → env-var → search raw_dir
        if self.annotation_file is None:
            env_ann = os.environ.get("WLASL_ANNOTATION_FILE")
            if env_ann:
                self.annotation_file = Path(env_ann)
            else:
                # Try to locate the JSON; leave as None if raw_dir doesn't exist yet
                if self.raw_dir.is_dir():
                    self.annotation_file = _find_annotation_file(self.raw_dir)
                else:
                    # Best-guess default name — will be checked in validate()
                    self.annotation_file = self.raw_dir / "WLASL_v0.3.json"

        # videos_dir: constructor → env-var → raw_dir/videos
        if self.videos_dir is None:
            env_vid = os.environ.get("WLASL_VIDEOS_DIR")
            if env_vid:
                self.videos_dir = Path(env_vid)
            else:
                self.videos_dir = self.raw_dir / "videos"

        # output_dir: constructor → env-var → auto-detect
        if self.output_dir is None:
            env_out = os.environ.get("WLASL_OUTPUT_DIR")
            if env_out:
                self.output_dir = Path(env_out)
            elif self._is_kaggle:
                self.output_dir = _KAGGLE_WORKING_ROOT / "outputs"
            else:
                self.output_dir = _REPO_ROOT / "outputs"

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def is_kaggle(self) -> bool:
        """``True`` if the code is running inside a Kaggle environment."""
        return self._is_kaggle

    @property
    def environment_name(self) -> str:
        """Human-readable environment label (``'kaggle'`` or ``'local'``)."""
        return "kaggle" if self._is_kaggle else "local"

    @property
    def processed_dir(self) -> Path:
        """Directory for preprocessed tensors / feature files."""
        if self._is_kaggle:
            return _KAGGLE_WORKING_ROOT / "data" / "processed"
        return _REPO_ROOT / "data" / "processed"

    @property
    def checkpoints_dir(self) -> Path:
        """Subdirectory inside *output_dir* for model checkpoints."""
        return self.output_dir / "checkpoints"  # type: ignore[operator]

    @property
    def logs_dir(self) -> Path:
        """Subdirectory inside *output_dir* for training logs."""
        return self.output_dir / "logs"  # type: ignore[operator]

    # ── Validation ────────────────────────────────────────────────────────────

    def validate(self) -> dict[str, tuple[bool, str]]:
        """Check that all configured paths exist and return a status report.

        Returns
        -------
        dict[str, tuple[bool, str]]
            Maps a descriptive key to ``(ok: bool, message: str)``.
            ``ok`` is ``True`` if the path exists (or the flag is correct).

        Example
        -------
        ::

            report = cfg.validate()
            for key, (ok, msg) in report.items():
                print(f"{'✔' if ok else '✘'}  {key}: {msg}")
        """
        report: dict[str, tuple[bool, str]] = {}

        # Environment detection
        report["is_kaggle"] = (
            True,
            f"{self._is_kaggle}  (environment: {self.environment_name})",
        )

        # raw_dir
        raw_ok = self.raw_dir is not None and self.raw_dir.is_dir()
        report["raw_dir"] = (
            raw_ok,
            f"{self.raw_dir}  ({'exists' if raw_ok else 'NOT FOUND'})",
        )

        # annotation_file
        ann_ok = self.annotation_file is not None and self.annotation_file.is_file()
        report["annotation_file"] = (
            ann_ok,
            f"{self.annotation_file}  ({'exists' if ann_ok else 'NOT FOUND'})",
        )

        # videos_dir
        vid_ok = self.videos_dir is not None and self.videos_dir.is_dir()
        report["videos_dir"] = (
            vid_ok,
            f"{self.videos_dir}  ({'exists' if vid_ok else 'NOT FOUND'})",
        )

        # output_dir (allowed not to exist yet)
        out_exists = self.output_dir is not None and self.output_dir.exists()
        report["output_dir"] = (
            True,  # non-fatal: output dir is created at training time
            f"{self.output_dir}  ({'exists' if out_exists else 'will be created'})",
        )

        return report

    def ensure_output_dirs(self) -> None:
        """Create output directories (and sub-dirs) if they do not yet exist.

        This is a no-op on Kaggle's read-only input mount; it only creates
        directories under *output_dir* (``/kaggle/working/`` on Kaggle,
        ``outputs/`` locally).
        """
        for directory in (self.output_dir, self.checkpoints_dir, self.logs_dir):
            if directory is not None:
                directory.mkdir(parents=True, exist_ok=True)

    # ── String representation ─────────────────────────────────────────────────

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"DatasetConfig(\n"
            f"  environment    = {self.environment_name!r}\n"
            f"  raw_dir        = {self.raw_dir}\n"
            f"  annotation_file= {self.annotation_file}\n"
            f"  videos_dir     = {self.videos_dir}\n"
            f"  output_dir     = {self.output_dir}\n"
            f"  processed_dir  = {self.processed_dir}\n"
            f"  checkpoints_dir= {self.checkpoints_dir}\n"
            f"  logs_dir       = {self.logs_dir}\n"
            f")"
        )

    def as_dict(self) -> dict[str, str]:
        """Return all resolved paths as a plain ``str`` dictionary.

        Useful for logging, serialising to YAML/JSON, or passing to
        experiment-tracking tools (MLflow, W&B).
        """
        return {
            "environment": self.environment_name,
            "raw_dir": str(self.raw_dir),
            "annotation_file": str(self.annotation_file),
            "videos_dir": str(self.videos_dir),
            "processed_dir": str(self.processed_dir),
            "output_dir": str(self.output_dir),
            "checkpoints_dir": str(self.checkpoints_dir),
            "logs_dir": str(self.logs_dir),
        }


# ── Convenience singleton ─────────────────────────────────────────────────────

def get_default_config(**kwargs) -> DatasetConfig:  # type: ignore[no-untyped-def]
    """Return a :class:`DatasetConfig` built from auto-detection + env-vars.

    Any keyword argument accepted by :class:`DatasetConfig` can be passed to
    override specific fields.

    Example
    -------
    ::

        from src.dataset.dataset_config import get_default_config

        cfg = get_default_config()
        cfg = get_default_config(kaggle_dataset_slug="my-wlasl")
    """
    return DatasetConfig(**kwargs)


# ── CLI quick-check ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = get_default_config()

    print("\n" + "─" * 60)
    print("  DatasetConfig — Environment Report")
    print("─" * 60)
    print(repr(cfg))
    print()

    print("─" * 60)
    print("  Path Validation")
    print("─" * 60)
    report = cfg.validate()
    all_ok = True
    for key, (ok, msg) in report.items():
        icon = "✔" if ok else "✘"
        print(f"  {icon}  {key:<20} {msg}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("  ✔ All paths are valid. Ready to proceed.")
    else:
        print("  ✘ Some paths are missing. See notes above.")
    print("─" * 60 + "\n")
