"""
train.py
========
End-to-end training script for the WLASL baseline model.

Features
--------
- Full train + validation loop with per-epoch logging.
- Cross-entropy loss with optional label smoothing.
- Adam optimiser with CosineAnnealingLR scheduler.
- Model checkpoint saving (best validation accuracy + last epoch).
- Resume training from an existing checkpoint (``--resume``).
- Backbone freeze/unfreeze strategy (freeze CNN for first N epochs).
- Kaggle-compatible: auto-detects environment via :class:`DatasetConfig`.

Usage
-----
Local::

    python src/training/train.py \\
        --epochs 30 \\
        --batch_size 8 \\
        --num_frames 16 \\
        --num_classes 100 \\
        --lr 1e-4

Resume from checkpoint::

    python src/training/train.py --resume outputs/checkpoints/last.pt

Kaggle (inside a notebook cell)::

    import subprocess
    subprocess.run([
        "python", "src/training/train.py",
        "--epochs", "30",
        "--batch_size", "16",
        "--num_classes", "100",
        "--num_workers", "2",
    ], check=True)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.dataset.dataset_config import get_default_config
from src.models.baseline_model import SignLanguageBaseline, build_model
from src.training.dataloader import build_dataloaders_from_config
from src.training.evaluate import evaluate
from src.training.metrics import AverageMeter, MetricTracker, topk_accuracy

logger = logging.getLogger(__name__)


# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_logging(log_dir: Optional[Path] = None, level: int = logging.INFO) -> None:
    """Configure root logger with console (and optionally file) handlers.

    Parameters
    ----------
    log_dir:
        If provided, a ``train.log`` file is written inside *log_dir*.
    level:
        Logging verbosity level (default: ``logging.INFO``).
    """
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "train.log")
        fh.setFormatter(logging.Formatter(fmt, datefmt))
        handlers.append(fh)

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(
    state: dict,
    checkpoint_dir: Path,
    filename: str = "last.pt",
) -> Path:
    """Save a training checkpoint to *checkpoint_dir*/*filename*.

    Parameters
    ----------
    state:
        Dictionary to serialise. Typically contains:
        ``epoch``, ``model_state_dict``, ``optimizer_state_dict``,
        ``scheduler_state_dict``, ``best_top1``, ``label_to_idx``.
    checkpoint_dir:
        Directory to write the checkpoint file.
    filename:
        Output filename (default: ``"last.pt"``).

    Returns
    -------
    Path
        Absolute path to the saved file.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / filename
    torch.save(state, path)
    logger.info("Checkpoint saved → %s", path)
    return path


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """Load a training checkpoint and restore model (and optionally optimiser) state.

    Parameters
    ----------
    path:
        Path to the ``.pt`` checkpoint file.
    model:
        Model whose ``state_dict`` will be restored.
    optimizer:
        If provided, the optimiser state is also restored.
    scheduler:
        If provided, the LR scheduler state is also restored.
    device:
        Map location for tensor loading (default: CPU).

    Returns
    -------
    dict
        The full checkpoint dictionary (caller can access ``epoch``,
        ``best_top1``, ``label_to_idx``, etc.).

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if device is None:
        device = torch.device("cpu")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    logger.info(
        "Resumed from checkpoint: %s  (epoch=%d, best_top1=%.2f%%)",
        path,
        checkpoint.get("epoch", -1),
        checkpoint.get("best_top1", float("nan")),
    )
    return checkpoint


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    log_interval: int = 10,
) -> dict[str, float]:
    """Run one full training epoch.

    Parameters
    ----------
    model:
        The model to train. Must be in ``train()`` mode on entry.
    loader:
        DataLoader for the training split.
    optimizer:
        Gradient-based optimiser.
    criterion:
        Loss function.
    device:
        Computation device.
    epoch:
        Current epoch number (1-indexed; used for logging only).
    log_interval:
        Log a progress line every *log_interval* batches.

    Returns
    -------
    dict[str, float]
        ``{"loss": float, "top1": float}`` — epoch-level averages.
    """
    model.train()
    tracker = MetricTracker(["loss", "top1"])
    epoch_start = time.time()

    for batch_idx, (frames, labels) in enumerate(loader):
        frames = frames.to(device, non_blocking=True)   # (B, T, 3, H, W)
        labels = labels.to(device, non_blocking=True)   # (B,)
        batch_size = frames.size(0)

        optimizer.zero_grad(set_to_none=True)
        logits = model(frames)                           # (B, num_classes)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping prevents exploding gradients with RNNs
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        top1 = topk_accuracy(logits.detach(), labels, topk=(1,))[0]
        tracker.update("loss", loss.item(), n=batch_size)
        tracker.update("top1", top1, n=batch_size)

        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(loader):
            elapsed = time.time() - epoch_start
            logger.info(
                "Epoch [%d] [%d/%d] | loss=%.4f | top1=%.2f%% | %.1fs elapsed",
                epoch, batch_idx + 1, len(loader),
                tracker.avg("loss"), tracker.avg("top1"), elapsed,
            )

    return tracker.summary()


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train WLASL baseline (CNN + GRU) model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--num_classes", type=int, default=100,
                        help="Number of sign classes (e.g. 100 for WLASL100).")
    parser.add_argument("--num_frames", type=int, default=16,
                        help="Frames sampled per video clip.")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Spatial resize dimension (height = width).")

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Initial learning rate for Adam.")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing factor for CrossEntropyLoss.")
    parser.add_argument("--freeze_epochs", type=int, default=5,
                        help="Freeze CNN backbone for the first N epochs.")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader subprocess workers. Use 0 on Windows.")

    # Model
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_rnn_layers", type=int, default=2)
    parser.add_argument("--no_pretrained", action="store_true",
                        help="Train CNN backbone from scratch (not recommended).")

    # Checkpointing
    parser.add_argument("--resume", type=Path, default=None,
                        help="Path to a checkpoint to resume from.")
    parser.add_argument("--checkpoint_dir", type=Path, default=None,
                        help="Override checkpoint directory (default: cfg.checkpoints_dir).")

    # Logging
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Log a progress line every N batches.")

    return parser.parse_args()


# ── Main training orchestration ───────────────────────────────────────────────

def main() -> None:
    """Full training run: data → model → train loop → checkpoint."""
    args = parse_args()

    # ── Config and paths ──────────────────────────────────────────────────────
    cfg = get_default_config()
    cfg.ensure_output_dirs()

    checkpoint_dir = args.checkpoint_dir or cfg.checkpoints_dir
    setup_logging(log_dir=cfg.logs_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 60)
    logger.info("WLASL Baseline Training")
    logger.info("Environment : %s", cfg.environment_name)
    logger.info("Device      : %s", device)
    logger.info("Config      : %s", cfg.as_dict())
    logger.info("=" * 60)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader, val_loader, label_to_idx = build_dataloaders_from_config(
        cfg=cfg,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        num_workers=args.num_workers,
        img_size=args.img_size,
        max_classes=args.num_classes,
    )
    num_classes = len(label_to_idx)
    logger.info("Classes: %d  Train batches: %d  Val batches: %d",
                num_classes, len(train_loader), len(val_loader))

    # Save label map alongside checkpoints
    label_map_path = checkpoint_dir / "label_to_idx.json"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with label_map_path.open("w") as fh:
        json.dump(label_to_idx, fh, indent=2)
    logger.info("Label map saved → %s", label_map_path)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(
        num_classes=num_classes,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        num_rnn_layers=args.num_rnn_layers,
        pretrained_cnn=not args.no_pretrained,
        device=device,
    )

    # Freeze backbone for initial epochs if requested
    if args.freeze_epochs > 0:
        model.freeze_backbone()

    # ── Optimiser and scheduler ───────────────────────────────────────────────
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # ── Resume training ───────────────────────────────────────────────────────
    start_epoch = 1
    best_top1: float = 0.0

    if args.resume is not None:
        ckpt = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_top1 = ckpt.get("best_top1", 0.0)
        logger.info("Resuming from epoch %d (best top-1: %.2f%%)", start_epoch, best_top1)

    # ── Training loop ─────────────────────────────────────────────────────────
    logger.info("Starting training for %d epoch(s)...", args.epochs)

    for epoch in range(start_epoch, args.epochs + 1):

        # Unfreeze backbone after freeze_epochs
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs + 1:
            model.unfreeze_backbone()
            # Reinitialise optimizer to include newly unfrozen parameters
            optimizer = Adam(
                model.parameters(),
                lr=args.lr * 0.1,   # lower LR for fine-tuning the backbone
                weight_decay=args.weight_decay,
            )
            logger.info("Backbone unfrozen at epoch %d (lr=%.2e)", epoch, args.lr * 0.1)

        # ── Train ─────────────────────────────────────────────────────────────
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            log_interval=args.log_interval,
        )

        # ── Validate ──────────────────────────────────────────────────────────
        val_metrics = evaluate(model, val_loader, criterion, device, topk=(1,))
        scheduler.step()

        val_top1 = val_metrics["top1"]
        is_best = val_top1 > best_top1
        if is_best:
            best_top1 = val_top1

        # ── Logging ───────────────────────────────────────────────────────────
        logger.info(
            "Epoch [%d/%d] | "
            "train_loss=%.4f train_top1=%.2f%% | "
            "val_loss=%.4f val_top1=%.2f%% | "
            "best_top1=%.2f%% | lr=%.2e%s",
            epoch, args.epochs,
            train_metrics["loss"], train_metrics["top1"],
            val_metrics["loss"], val_top1,
            best_top1,
            scheduler.get_last_lr()[0],
            " ← BEST" if is_best else "",
        )

        # ── Checkpointing ──────────────────────────────────────────────────────
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_top1": best_top1,
            "val_top1": val_top1,
            "train_loss": train_metrics["loss"],
            "label_to_idx": label_to_idx,
            "args": vars(args),
        }
        save_checkpoint(state, checkpoint_dir, filename="last.pt")
        if is_best:
            save_checkpoint(state, checkpoint_dir, filename="best.pt")

    logger.info("Training complete. Best Val Top-1: %.2f%%", best_top1)


if __name__ == "__main__":
    main()
