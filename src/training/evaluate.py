"""
evaluate.py
===========
Standalone evaluation pass for the WLASL baseline model.

Runs a full forward pass over a DataLoader (no gradient computation) and
returns per-batch and epoch-level metrics.

Usage
-----
As a module (called from train.py)::

    from src.training.evaluate import evaluate

    metrics = evaluate(model, val_loader, criterion, device)
    print(f"Val Loss: {metrics['loss']:.4f}  Top-1: {metrics['top1']:.2f}%")

As a CLI script (runs evaluation from a saved checkpoint)::

    python src/training/evaluate.py \\
        --checkpoint outputs/checkpoints/best.pt \\
        --split val \\
        --batch_size 8
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.metrics import AverageMeter, MetricTracker, topk_accuracy

logger = logging.getLogger(__name__)


# ── Core evaluation function ──────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    topk: tuple[int, ...] = (1,),
) -> dict[str, float]:
    """Run a full evaluation pass over *loader*.

    The model is set to ``eval()`` mode before the pass and restored to its
    previous state afterwards.

    Parameters
    ----------
    model:
        The neural network to evaluate.
    loader:
        DataLoader producing ``(frames, labels)`` batches where
        ``frames`` has shape ``(B, T, 3, H, W)`` and ``labels`` is ``(B,)``.
    criterion:
        Loss function (e.g. ``nn.CrossEntropyLoss()``).
    device:
        Target computation device.
    topk:
        Tuple of k values for Top-k accuracy computation.
        Default: ``(1,)`` — only Top-1.

    Returns
    -------
    dict[str, float]
        Dictionary with keys:

        - ``"loss"``   — mean cross-entropy loss over the full split.
        - ``"top1"``   — Top-1 accuracy in percent (always included).
        - ``"top5"``   — Top-5 accuracy in percent (only when ``5 in topk``).
        - ``"n"``      — total number of evaluated samples.
    """
    was_training = model.training
    model.eval()

    metric_names = ["loss"] + [f"top{k}" for k in topk]
    tracker = MetricTracker(metric_names)

    total_samples = 0

    for batch_idx, (frames, labels) in enumerate(loader):
        frames: torch.Tensor = frames.to(device, non_blocking=True)
        labels: torch.Tensor = labels.to(device, non_blocking=True)
        batch_size = frames.size(0)

        logits = model(frames)                          # (B, num_classes)
        loss = criterion(logits, labels)

        accs = topk_accuracy(logits, labels, topk=topk)

        tracker.update("loss", loss.item(), n=batch_size)
        for k, acc in zip(topk, accs):
            tracker.update(f"top{k}", acc, n=batch_size)

        total_samples += batch_size

        if (batch_idx + 1) % max(1, len(loader) // 5) == 0:
            logger.debug(
                "  [eval %d/%d] loss=%.4f  top1=%.2f%%",
                batch_idx + 1, len(loader),
                tracker.avg("loss"), tracker.avg("top1"),
            )

    if was_training:
        model.train()

    summary = tracker.summary()
    summary["n"] = float(total_samples)

    logger.info(
        "Evaluation complete | %s | samples=%d",
        "  ".join(f"{k}={v:.4f}" for k, v in summary.items() if k != "n"),
        total_samples,
    )
    return summary


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved WLASL baseline checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to a saved .pt checkpoint file.",
    )
    parser.add_argument(
        "--split", type=str, default="val", choices=["val", "test"],
        help="Dataset split to evaluate on.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument(
        "--max_classes", type=int, default=None,
        help="Restrict to first N classes (e.g. 100 for WLASL100).",
    )
    return parser.parse_args()


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    """CLI evaluation: load checkpoint → build dataloader → run evaluate()."""
    _setup_logging()
    args = _parse_args()

    # Deferred imports to keep module-level import cost low
    from src.dataset.dataset_config import get_default_config  # noqa: PLC0415
    from src.models.baseline_model import build_model          # noqa: PLC0415
    from src.training.dataloader import build_dataloaders_from_config  # noqa: PLC0415
    from src.training.dataset import get_val_transforms        # noqa: PLC0415

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    if not args.checkpoint.is_file():
        logger.error("Checkpoint not found: %s", args.checkpoint)
        sys.exit(1)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    logger.info(
        "Loaded checkpoint: epoch=%d  best_top1=%.2f%%",
        checkpoint.get("epoch", -1),
        checkpoint.get("best_top1", float("nan")),
    )

    # ── Config & DataLoader ───────────────────────────────────────────────────
    cfg = get_default_config()
    train_loader, val_loader, label_to_idx = build_dataloaders_from_config(
        cfg=cfg,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        num_workers=args.num_workers,
        img_size=args.img_size,
        max_classes=args.max_classes,
    )
    num_classes = len(label_to_idx)
    eval_loader = val_loader  # could extend to test_loader in future

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(num_classes=num_classes, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(model, eval_loader, criterion, device, topk=(1,))

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print(f"  Evaluation Results  ({args.split} split)")
    print("─" * 50)
    for key, val in metrics.items():
        if key == "n":
            print(f"  Samples evaluated : {int(val)}")
        else:
            print(f"  {key:<18}: {val:.4f}")
    print("─" * 50 + "\n")


if __name__ == "__main__":
    main()
