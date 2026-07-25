"""
explore_annotations.py
=======================
Deep-dive annotation analysis for the WLASL dataset.

Provides per-class statistics, signer distribution, source breakdown,
bounding-box coverage, and frame-range analysis — all without touching
any video files.

Usage
-----
    python -m src.dataset.explore_annotations
    # or directly:
    python src/dataset/explore_annotations.py [--json PATH] [--top N]

Arguments
---------
--json PATH   Path to the WLASL annotation JSON.
              Defaults to the first .json found in data/raw/.
--top  N      Number of top/bottom classes to display (default: 10).

Expected dataset layout
-----------------------
    data/
    └── raw/
        └── WLASL_v0.3.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, stdev

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
DEFAULT_ANNOTATION_NAMES = [
    "WLASL_v0.3.json",
    "WLASL_v0.2.json",
    "WLASL_v0.1.json",
    "WLASL.json",
]


# ── I/O helpers ───────────────────────────────────────────────────────────────

def find_annotation_file(raw_dir: Path) -> Path | None:
    """Return the first WLASL annotation JSON found in *raw_dir*."""
    for name in DEFAULT_ANNOTATION_NAMES:
        candidate = raw_dir / name
        if candidate.is_file():
            return candidate
    jsons = sorted(raw_dir.glob("*.json"))
    return jsons[0] if jsons else None


def load_json(path: Path) -> list[dict]:
    """Load and validate the WLASL JSON annotation file."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        _err(f"Malformed JSON: {exc}")
        sys.exit(1)
    except OSError as exc:
        _err(f"Cannot read file: {exc}")
        sys.exit(1)

    if not isinstance(data, list):
        _err("Expected a JSON array at the top level. Is this a valid WLASL file?")
        sys.exit(1)

    return data  # type: ignore[return-value]


# ── Analysis functions ────────────────────────────────────────────────────────

def per_class_counts(entries: list[dict]) -> dict[str, int]:
    """Return {gloss: instance_count} for every class."""
    return {e["gloss"]: len(e.get("instances", [])) for e in entries}


def collect_all_instances(entries: list[dict]) -> list[dict]:
    """Flatten all instances from all classes into one list."""
    instances = []
    for entry in entries:
        for inst in entry.get("instances", []):
            inst_copy = dict(inst)
            inst_copy["_gloss"] = entry["gloss"]
            instances.append(inst_copy)
    return instances


def split_distribution(instances: list[dict]) -> Counter:
    return Counter(inst.get("split", "unknown").lower() for inst in instances)


def signer_distribution(instances: list[dict]) -> Counter:
    return Counter(inst.get("signer_id", "unknown") for inst in instances)


def source_distribution(instances: list[dict]) -> Counter:
    return Counter(inst.get("source", "unknown") for inst in instances)


def frame_range_stats(instances: list[dict]) -> dict[str, float]:
    """Compute statistics about clip lengths (frame_end − frame_start)."""
    lengths = []
    for inst in instances:
        start = inst.get("frame_start", None)
        end = inst.get("frame_end", None)
        if start is not None and end is not None and end >= start:
            lengths.append(end - start)

    if not lengths:
        return {}

    return {
        "count": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "mean": mean(lengths),
        "median": median(lengths),
        "stdev": stdev(lengths) if len(lengths) > 1 else 0.0,
    }


def bbox_coverage(instances: list[dict]) -> float:
    """Return fraction of instances that have a non-empty bounding box."""
    total = len(instances)
    if total == 0:
        return 0.0
    with_bbox = sum(
        1 for inst in instances
        if inst.get("bbox") and isinstance(inst["bbox"], list) and len(inst["bbox"]) == 4
    )
    return with_bbox / total


# ── Printing helpers ──────────────────────────────────────────────────────────

def _table(columns: list[tuple[str, str, str]], rows: list[tuple]) -> None:
    """Print a rich or plain table.

    Parameters
    ----------
    columns : list of (header, style, justify)
    rows    : list of row tuples matching columns
    """
    if HAS_RICH:
        table = Table(show_header=True, header_style="bold magenta")
        for header, style, justify in columns:
            table.add_column(header, style=style, justify=justify)  # type: ignore[arg-type]
        for row in rows:
            table.add_row(*[str(v) for v in row])
        _console.print(table)
    else:
        widths = [max(len(c[0]), max((len(str(r[i])) for r in rows), default=0))
                  for i, c in enumerate(columns)]
        header_row = "  ".join(c[0].ljust(widths[i]) for i, c in enumerate(columns))
        print(header_row)
        print("-" * len(header_row))
        for row in rows:
            print("  ".join(str(v).ljust(widths[i]) for i, v in enumerate(row)))


def print_class_distribution(pcc: dict[str, int], top_n: int) -> None:
    _header(f"Class Distribution — Top {top_n} & Bottom {top_n}")
    sorted_classes = sorted(pcc.items(), key=lambda x: x[1], reverse=True)
    top = sorted_classes[:top_n]
    bottom = sorted_classes[-top_n:]

    _print(f"[bold]Top {top_n} most-represented classes:[/bold]" if HAS_RICH
           else f"Top {top_n} most-represented classes:")
    _table(
        [("Rank", "cyan", "right"), ("Gloss", "white", "left"), ("Videos", "green", "right")],
        [(i + 1, gloss, count) for i, (gloss, count) in enumerate(top)],
    )
    _print()
    _print(f"[bold]Bottom {top_n} least-represented classes:[/bold]" if HAS_RICH
           else f"Bottom {top_n} least-represented classes:")
    _table(
        [("Rank", "cyan", "right"), ("Gloss", "white", "left"), ("Videos", "yellow", "right")],
        [(len(sorted_classes) - top_n + i + 1, gloss, count)
         for i, (gloss, count) in enumerate(bottom)],
    )
    _print()


def print_split_table(split_dist: Counter) -> None:
    _header("Split Distribution")
    total = sum(split_dist.values()) or 1
    rows = [
        (name.capitalize(), count, f"{count / total * 100:.1f}%")
        for name, count in sorted(split_dist.items())
        if count > 0
    ]
    _table([("Split", "cyan", "left"), ("Videos", "green", "right"), ("%", "white", "right")], rows)
    _print()


def print_signer_table(signer_dist: Counter, top_n: int) -> None:
    _header(f"Top {top_n} Signers by Video Count")
    rows = [
        (signer_id, count)
        for signer_id, count in signer_dist.most_common(top_n)
    ]
    _table([("Signer ID", "cyan", "left"), ("Videos", "green", "right")], rows)
    _ok(f"Total unique signers: {len(signer_dist)}")
    _print()


def print_source_table(source_dist: Counter) -> None:
    _header("Video Source Breakdown")
    total = sum(source_dist.values()) or 1
    rows = [
        (src, count, f"{count / total * 100:.1f}%")
        for src, count in source_dist.most_common()
    ]
    _table([("Source", "cyan", "left"), ("Videos", "green", "right"), ("%", "white", "right")], rows)
    _print()


def print_frame_stats(stats: dict[str, float]) -> None:
    _header("Clip Length Statistics (frames)")
    if not stats:
        _warn("No frame_start / frame_end data found in annotations.")
        return
    for key, value in stats.items():
        if isinstance(value, float):
            _ok(f"{key:<10}: {value:.1f}")
        else:
            _ok(f"{key:<10}: {int(value)}")
    _print()


def print_bbox_coverage(coverage: float) -> None:
    _header("Bounding Box Coverage")
    _ok(f"{coverage * 100:.1f}% of instances have bounding box annotations.")
    _print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deep-dive annotation analysis for the WLASL dataset."
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Path to WLASL annotation JSON (auto-detected if omitted).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top/bottom classes to display (default: 10).",
    )
    return parser.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    _print()
    _header("Motion Sign Language Recognition — Annotation Analyser")

    # Locate annotation file
    if args.json:
        annotation_path = Path(args.json).resolve()
        if not annotation_path.is_file():
            _err(f"Annotation file not found: {annotation_path}")
            sys.exit(1)
    else:
        if not RAW_DIR.is_dir():
            _err(
                f"Raw data directory not found: {RAW_DIR}\n"
                "  → Place the WLASL JSON file inside data/raw/ and retry."
            )
            sys.exit(1)
        annotation_path = find_annotation_file(RAW_DIR)
        if annotation_path is None:
            _err(
                "No WLASL annotation JSON found in data/raw/.\n"
                "  → Expected file: data/raw/WLASL_v0.3.json\n"
                "  → Dataset: https://dxli94.github.io/WLASL/"
            )
            sys.exit(1)

    _ok(f"Annotation file: {annotation_path}")
    entries = load_json(annotation_path)

    # Compute all statistics
    pcc = per_class_counts(entries)
    instances = collect_all_instances(entries)
    split_dist = split_distribution(instances)
    signer_dist = signer_distribution(instances)
    source_dist = source_distribution(instances)
    frame_stats = frame_range_stats(instances)
    bbox_cov = bbox_coverage(instances)

    # Print reports
    print_class_distribution(pcc, args.top)
    print_split_table(split_dist)
    print_signer_table(signer_dist, args.top)
    print_source_table(source_dist)
    print_frame_stats(frame_stats)
    print_bbox_coverage(bbox_cov)

    _ok("Annotation analysis complete.")
    _print()


if __name__ == "__main__":
    main()
