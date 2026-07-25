"""
video_info.py
=============
Inspect metadata of a single video file without extracting frames or running
any AI model.

Reads the following properties via OpenCV:
    - Codec FourCC
    - Frame rate (FPS)
    - Resolution (width × height)
    - Total frame count
    - Duration (seconds)

Also cross-references WLASL annotation data (if available) to display the
gloss label and split for the requested video ID.

Usage
-----
    python src/dataset/video_info.py --video data/raw/videos/00001.mp4
    python src/dataset/video_info.py --id 00001
    python src/dataset/video_info.py --all          # summarise every video

Arguments
---------
--video PATH    Direct path to a video file.
--id    ID      WLASL video_id to look up (searches data/raw/videos/).
--all           Print a summary table for every video in data/raw/videos/.
--json  PATH    Path to annotation JSON (auto-detected if omitted).
--limit N       Max videos to process with --all (default: 20).

Dependencies
------------
    pip install opencv-python
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Optional rich ─────────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    _console = Console()

    def _print(msg: str = "") -> None:
        _console.print(msg)

    def _header(title: str) -> None:
        _console.print(Panel(f"[bold cyan]{title}[/bold cyan]", expand=False))

    def _ok(msg: str) -> None:
        _console.print(f"[bold green]✔[/bold green] {msg}")

    def _warn(msg: str) -> None:
        _console.print(f"[bold yellow]⚠[/bold yellow]  {msg}")

    def _err(msg: str) -> None:
        _console.print(f"[bold red]✘[/bold red]  {msg}")

    HAS_RICH = True

except ImportError:
    HAS_RICH = False

    def _print(msg: str = "") -> None:  # type: ignore[misc]
        print(msg)

    def _header(title: str) -> None:  # type: ignore[misc]
        sep = "─" * (len(title) + 4)
        print(f"\n{sep}\n  {title}\n{sep}")

    def _ok(msg: str) -> None:  # type: ignore[misc]
        print(f"[OK]   {msg}")

    def _warn(msg: str) -> None:  # type: ignore[misc]
        print(f"[WARN] {msg}")

    def _err(msg: str) -> None:  # type: ignore[misc]
        print(f"[ERR]  {msg}")


# ── Constants ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
VIDEOS_DIR = RAW_DIR / "videos"
DEFAULT_ANNOTATION_NAMES = [
    "WLASL_v0.3.json",
    "WLASL_v0.2.json",
    "WLASL_v0.1.json",
    "WLASL.json",
]
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ── OpenCV import (graceful failure) ──────────────────────────────────────────

def _import_cv2():  # type: ignore[return]
    """Import OpenCV and exit with a clear message if not installed."""
    try:
        import cv2  # noqa: PLC0415
        return cv2
    except ImportError:
        _err(
            "OpenCV is not installed.\n"
            "  → Install it with:  pip install opencv-python\n"
            "  → Then retry."
        )
        sys.exit(1)


# ── Annotation helpers ────────────────────────────────────────────────────────

def load_annotation_index(raw_dir: Path) -> dict[str, dict]:
    """Return a mapping of video_id → {gloss, split, fps, frame_start, frame_end, bbox}.

    Returns an empty dict if no annotation file is found (non-fatal).
    """
    import json  # noqa: PLC0415

    annotation_path: Path | None = None
    for name in DEFAULT_ANNOTATION_NAMES:
        candidate = raw_dir / name
        if candidate.is_file():
            annotation_path = candidate
            break
    if annotation_path is None:
        jsons = sorted(raw_dir.glob("*.json"))
        annotation_path = jsons[0] if jsons else None

    if annotation_path is None:
        return {}

    try:
        with annotation_path.open("r", encoding="utf-8") as fh:
            entries = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}

    index: dict[str, dict] = {}
    for entry in entries:
        gloss = entry.get("gloss", "?")
        for inst in entry.get("instances", []):
            vid_id = str(inst.get("video_id", "")).zfill(5)
            index[vid_id] = {
                "gloss": gloss,
                "split": inst.get("split", "unknown"),
                "fps": inst.get("fps", None),
                "frame_start": inst.get("frame_start", None),
                "frame_end": inst.get("frame_end", None),
                "bbox": inst.get("bbox", None),
                "signer_id": inst.get("signer_id", None),
                "source": inst.get("source", None),
            }
    return index


# ── Core video inspection ─────────────────────────────────────────────────────

def get_video_metadata(video_path: Path, cv2) -> dict:  # type: ignore[no-untyped-def]
    """Extract metadata from a video file using OpenCV.

    Parameters
    ----------
    video_path : Path
        Absolute path to the video file.
    cv2 :
        The imported cv2 module.

    Returns
    -------
    dict with keys: path, fps, width, height, total_frames, duration_s,
                    fourcc, readable (bool), error (str|None)
    """
    result: dict = {
        "path": str(video_path),
        "fps": None,
        "width": None,
        "height": None,
        "total_frames": None,
        "duration_s": None,
        "fourcc": None,
        "readable": False,
        "error": None,
    }

    if not video_path.is_file():
        result["error"] = f"File not found: {video_path}"
        return result

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        result["error"] = "OpenCV could not open the video file."
        cap.release()
        return result

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join([chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)]).strip()

        duration_s = total_frames / fps if fps and fps > 0 else None

        result.update({
            "fps": round(fps, 3),
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "duration_s": round(duration_s, 3) if duration_s else None,
            "fourcc": fourcc,
            "readable": True,
        })
    finally:
        cap.release()

    return result


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_single_video_report(
    meta: dict,
    ann: dict | None = None,
) -> None:
    """Pretty-print the metadata for a single video."""
    path = Path(meta["path"])
    video_id = path.stem.zfill(5)
    _header(f"Video Inspector — {path.name}")

    if meta["error"]:
        _err(meta["error"])
        return

    file_size_mb = path.stat().st_size / (1024 ** 2) if path.is_file() else 0

    rows: list[tuple[str, str]] = [
        ("File", str(path)),
        ("File size", f"{file_size_mb:.2f} MB"),
        ("Codec (FourCC)", meta["fourcc"] or "unknown"),
        ("FPS", str(meta["fps"])),
        ("Resolution", f"{meta['width']} × {meta['height']} px"),
        ("Total frames", f"{meta['total_frames']:,}"),
        ("Duration", f"{meta['duration_s']:.3f} s" if meta["duration_s"] else "unknown"),
    ]

    if ann:
        rows.extend([
            ("─" * 20, "─" * 30),
            ("Gloss (label)", ann.get("gloss", "?")),
            ("Split", ann.get("split", "?")),
            ("Signer ID", str(ann.get("signer_id", "?"))),
            ("Source", str(ann.get("source", "?"))),
            ("Annotation FPS", str(ann.get("fps", "?"))),
            ("Frame start", str(ann.get("frame_start", "?"))),
            ("Frame end", str(ann.get("frame_end", "?"))),
            ("Bounding box", str(ann.get("bbox", "?"))),
        ])

    if HAS_RICH:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Property", style="cyan", min_width=20)
        table.add_column("Value", style="white")
        for k, v in rows:
            table.add_row(k, v)
        _console.print(table)
    else:
        print(f"{'Property':<22} Value")
        print("-" * 60)
        for k, v in rows:
            print(f"  {k:<20} {v}")

    _print()


def print_batch_report(video_paths: list[Path], ann_index: dict, cv2) -> None:  # type: ignore[no-untyped-def]
    """Print a summary table for a list of video files."""
    _header(f"Batch Video Summary — {len(video_paths)} files")

    headers = [
        ("Video ID", "cyan", "left"),
        ("Gloss", "white", "left"),
        ("FPS", "green", "right"),
        ("Resolution", "white", "left"),
        ("Frames", "green", "right"),
        ("Duration (s)", "green", "right"),
        ("Split", "yellow", "left"),
        ("Status", "white", "left"),
    ]

    rows = []
    for path in video_paths:
        meta = get_video_metadata(path, cv2)
        vid_id = path.stem.zfill(5)
        ann = ann_index.get(vid_id, {})

        if meta["readable"]:
            rows.append((
                vid_id,
                ann.get("gloss", "—"),
                str(meta["fps"]),
                f"{meta['width']}×{meta['height']}",
                str(meta["total_frames"]),
                str(meta["duration_s"]),
                ann.get("split", "—"),
                "✔ OK",
            ))
        else:
            rows.append((vid_id, ann.get("gloss", "—"), "—", "—", "—", "—",
                         ann.get("split", "—"), f"✘ {meta['error']}"))

    if HAS_RICH:
        table = Table(show_header=True, header_style="bold magenta")
        for col, style, justify in headers:
            table.add_column(col, style=style, justify=justify)  # type: ignore[arg-type]
        for row in rows:
            table.add_row(*row)
        _console.print(table)
    else:
        col_widths = [max(len(h[0]), max((len(r[i]) for r in rows), default=0))
                      for i, h in enumerate(headers)]
        header_line = "  ".join(h[0].ljust(col_widths[i]) for i, h in enumerate(headers))
        print(header_line)
        print("-" * len(header_line))
        for row in rows:
            print("  ".join(str(v).ljust(col_widths[i]) for i, v in enumerate(row)))

    _print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect video metadata for the WLASL dataset."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--video", type=Path, help="Direct path to a video file.")
    mode.add_argument("--id", type=str, dest="video_id",
                      help="WLASL video_id (zero-padded, e.g. 00001).")
    mode.add_argument("--all", action="store_true",
                      help="Summarise all videos in data/raw/videos/.")
    parser.add_argument("--json", type=Path, default=None,
                        help="Path to WLASL annotation JSON (auto-detected if omitted).")
    parser.add_argument("--limit", type=int, default=20,
                        help="Max videos to process with --all (default: 20).")
    return parser.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cv2 = _import_cv2()

    # Load annotation index (optional — used to enrich output)
    ann_index = load_annotation_index(RAW_DIR)
    if not ann_index:
        _warn(
            "No WLASL annotation JSON found in data/raw/. "
            "Video metadata will be shown without gloss labels."
        )

    # ── Single video by direct path ───────────────────────────────────────────
    if args.video:
        video_path = args.video.resolve()
        meta = get_video_metadata(video_path, cv2)
        vid_id = video_path.stem.zfill(5)
        ann = ann_index.get(vid_id)
        print_single_video_report(meta, ann)
        return

    # ── Single video by WLASL ID ──────────────────────────────────────────────
    if args.video_id:
        vid_id = str(args.video_id).zfill(5)
        # Search for file with that stem
        if not VIDEOS_DIR.is_dir():
            _err(
                f"Videos directory not found: {VIDEOS_DIR}\n"
                "  → Place WLASL videos inside data/raw/videos/ and retry."
            )
            sys.exit(1)
        matches = [
            p for p in VIDEOS_DIR.iterdir()
            if p.stem.zfill(5) == vid_id and p.suffix.lower() in VIDEO_EXTENSIONS
        ]
        if not matches:
            _err(f"No video file found for ID '{vid_id}' in {VIDEOS_DIR}.")
            sys.exit(1)
        meta = get_video_metadata(matches[0], cv2)
        ann = ann_index.get(vid_id)
        print_single_video_report(meta, ann)
        return

    # ── Batch mode: all videos ────────────────────────────────────────────────
    if args.all:
        if not VIDEOS_DIR.is_dir():
            _err(
                f"Videos directory not found: {VIDEOS_DIR}\n"
                "  → Place WLASL videos inside data/raw/videos/ and retry."
            )
            sys.exit(1)
        video_paths = sorted(
            p for p in VIDEOS_DIR.iterdir()
            if p.suffix.lower() in VIDEO_EXTENSIONS
        )[: args.limit]
        if not video_paths:
            _err(f"No video files found in {VIDEOS_DIR}.")
            sys.exit(1)
        if len(video_paths) == args.limit:
            _warn(f"Showing first {args.limit} videos. Use --limit to increase.")
        print_batch_report(video_paths, ann_index, cv2)
        _ok(f"Processed {len(video_paths)} video(s).")
        return


if __name__ == "__main__":
    main()
