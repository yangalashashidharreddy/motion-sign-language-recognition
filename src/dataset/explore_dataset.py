"""
explore_dataset.py
==================
High-level statistics explorer for the WLASL (Word-Level American Sign Language)
dataset annotation file.

Usage
-----
    python -m src.dataset.explore_dataset
    # or directly:
    python src/dataset/explore_dataset.py

Expected dataset layout
-----------------------
    data/
    └── raw/
        ├── WLASL_v0.3.json          # annotation file
        └── videos/                  # video files (optional for this script)

Notes
-----
- This script performs READ-ONLY analysis; it does not modify any files.
- No frames are extracted and no models are loaded.
- Requires only: Python standard library + (optionally) rich for pretty output.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ── Optional pretty-printing via `rich` ───────────────────────────────────────
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
DEFAULT_ANNOTATION_NAMES = [
    "WLASL_v0.3.json",
    "WLASL_v0.2.json",
    "WLASL_v0.1.json",
    "WLASL.json",
]
NUM_SAMPLE_GLOSSES = 10


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_annotation_file(raw_dir: Path) -> Path | None:
    """Return the first WLASL annotation JSON found in *raw_dir*, or None."""
    for name in DEFAULT_ANNOTATION_NAMES:
        candidate = raw_dir / name
        if candidate.is_file():
            return candidate
    # Fallback: any .json at the top level of raw/
    jsons = sorted(raw_dir.glob("*.json"))
    return jsons[0] if jsons else None


def load_annotations(annotation_path: Path) -> list[dict]:
    """Load and parse the WLASL JSON annotation file.

    Parameters
    ----------
    annotation_path:
        Absolute path to the JSON annotation file.

    Returns
    -------
    list[dict]
        Parsed list of gloss entries, each containing ``gloss`` and
        ``instances`` keys.

    Raises
    ------
    SystemExit
        If the file cannot be read or parsed.
    """
    try:
        with annotation_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        _err(f"Failed to parse JSON: {exc}")
        sys.exit(1)
    except OSError as exc:
        _err(f"Cannot read annotation file: {exc}")
        sys.exit(1)

    if not isinstance(data, list):
        _err(
            "Unexpected annotation format — expected a top-level JSON array. "
            "Please verify the file is a valid WLASL annotation."
        )
        sys.exit(1)

    return data  # type: ignore[return-value]


def compute_split_distribution(entries: list[dict]) -> dict[str, int]:
    """Return a mapping of split name → video count across all entries."""
    counter: Counter[str] = Counter()
    for entry in entries:
        for instance in entry.get("instances", []):
            split = instance.get("split", "unknown").lower()
            counter[split] += 1
    return dict(counter)


def compute_per_class_video_counts(entries: list[dict]) -> dict[str, int]:
    """Return a mapping of gloss → number of video instances."""
    return {
        entry["gloss"]: len(entry.get("instances", []))
        for entry in entries
    }


def sample_glosses(entries: list[dict], n: int = NUM_SAMPLE_GLOSSES) -> list[str]:
    """Return the first *n* gloss labels from the annotation data."""
    return [entry["gloss"] for entry in entries[:n]]


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_overview(entries: list[dict], annotation_path: Path) -> None:
    """Print a high-level overview of the dataset."""
    _header("WLASL Dataset — Overview")

    num_classes = len(entries)
    total_videos = sum(len(e.get("instances", [])) for e in entries)
    split_dist = compute_split_distribution(entries)
    per_class = compute_per_class_video_counts(entries)
    avg_vids = total_videos / num_classes if num_classes else 0
    max_class = max(per_class, key=per_class.get)  # type: ignore[arg-type]
    min_class = min(per_class, key=per_class.get)  # type: ignore[arg-type]

    _ok(f"Annotation file : {annotation_path}")
    _ok(f"Number of classes (glosses) : {num_classes:,}")
    _ok(f"Total video instances        : {total_videos:,}")
    _ok(f"Avg videos per class         : {avg_vids:.1f}")
    _ok(f"Class with most videos       : '{max_class}' ({per_class[max_class]})")
    _ok(f"Class with fewest videos     : '{min_class}' ({per_class[min_class]})")
    _print()


def print_split_distribution(entries: list[dict]) -> None:
    """Print the train / validation / test split distribution."""
    _header("Train / Val / Test Split")

    split_dist = compute_split_distribution(entries)
    total = sum(split_dist.values()) or 1

    if HAS_RICH:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Split", style="cyan", width=12)
        table.add_column("Videos", justify="right")
        table.add_column("Percentage", justify="right")
        for split_name in ["train", "val", "test", "unknown"]:
            count = split_dist.get(split_name, 0)
            if count == 0:
                continue
            table.add_row(
                split_name.capitalize(),
                str(count),
                f"{count / total * 100:.1f}%",
            )
        _console.print(table)
    else:
        print(f"{'Split':<12} {'Videos':>8} {'%':>8}")
        print("-" * 32)
        for split_name in ["train", "val", "test", "unknown"]:
            count = split_dist.get(split_name, 0)
            if count == 0:
                continue
            print(f"{split_name.capitalize():<12} {count:>8,} {count / total * 100:>7.1f}%")
    _print()


def print_sample_glosses(entries: list[dict]) -> None:
    """Print a sample of gloss labels."""
    _header(f"Sample Gloss Labels (first {NUM_SAMPLE_GLOSSES})")
    glosses = sample_glosses(entries)
    for i, gloss in enumerate(glosses, start=1):
        _print(f"  {i:>3}. {gloss}")
    _print()


def print_annotation_schema(entries: list[dict]) -> None:
    """Inspect and display the keys found inside a sample instance."""
    _header("Annotation Schema (first instance)")
    if not entries or not entries[0].get("instances"):
        _warn("No instances found in the first entry — cannot display schema.")
        return

    sample_instance = entries[0]["instances"][0]
    _ok(f"Gloss : '{entries[0]['gloss']}'")
    _print()

    if HAS_RICH:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Field", style="cyan")
        table.add_column("Value")
        for key, value in sample_instance.items():
            table.add_row(str(key), str(value))
        _console.print(table)
    else:
        print(f"{'Field':<20} Value")
        print("-" * 60)
        for key, value in sample_instance.items():
            print(f"  {key:<18} {value}")
    _print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Run all dataset exploration reports."""
    _print()
    _header("Motion Sign Language Recognition — Dataset Explorer")

    # ── Locate annotation file ────────────────────────────────────────────────
    if not RAW_DIR.is_dir():
        _err(
            f"Raw data directory not found: {RAW_DIR}\n"
            "  → Create it and place the WLASL dataset inside:\n"
            "      data/raw/WLASL_v0.3.json\n"
            "      data/raw/videos/"
        )
        sys.exit(1)

    annotation_path = find_annotation_file(RAW_DIR)
    if annotation_path is None:
        _err(
            "No WLASL annotation JSON found in data/raw/.\n"
            "  → Download the dataset and place the annotation file at:\n"
            "      data/raw/WLASL_v0.3.json\n"
            "  → Dataset homepage: https://dxli94.github.io/WLASL/"
        )
        sys.exit(1)

    # ── Load & explore ────────────────────────────────────────────────────────
    entries = load_annotations(annotation_path)

    print_overview(entries, annotation_path)
    print_split_distribution(entries)
    print_sample_glosses(entries)
    print_annotation_schema(entries)

    _ok("Exploration complete. Run `src/dataset/video_info.py` to inspect a video file.")
    _print()


if __name__ == "__main__":
    main()
