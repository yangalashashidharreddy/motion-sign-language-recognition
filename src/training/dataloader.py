"""
dataloader.py
=============
Factory functions for building PyTorch DataLoaders for WLASL training.

Usage
-----
::

    from src.dataset.dataset_config import DatasetConfig
    from src.training.dataloader import build_dataloaders

    cfg = DatasetConfig()
    train_loader, val_loader, label_to_idx = build_dataloaders(
        cfg=cfg,
        batch_size=8,
        num_frames=16,
        num_workers=2,
        img_size=224,
        max_classes=100,      # WLASL100 subset
    )

    for frames, labels in train_loader:
        # frames: (B, T, 3, H, W)
        # labels: (B,)
        ...
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from src.training.dataset import WLASLDataset, get_train_transforms, get_val_transforms

logger = logging.getLogger(__name__)


# ── Single DataLoader factory ─────────────────────────────────────────────────

def build_single_dataloader(
    dataset: WLASLDataset,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 2,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """Wrap a :class:`~src.training.dataset.WLASLDataset` in a DataLoader.

    Parameters
    ----------
    dataset:
        An instantiated :class:`WLASLDataset`.
    batch_size:
        Number of clips per mini-batch.
    shuffle:
        Randomly shuffle samples at the start of every epoch.
        Should be ``True`` for training and ``False`` for validation.
    num_workers:
        Number of subprocess workers for parallel data loading.
        Use ``0`` on Windows or when debugging.
    pin_memory:
        Pin host memory for faster GPU transfers.
        Automatically disabled when no CUDA device is available.
    drop_last:
        Drop the last incomplete batch. Useful when batch normalisation
        layers require a minimum batch size.

    Returns
    -------
    DataLoader
        Configured DataLoader ready for iteration.
    """
    # Disable pin_memory when no GPU is available to avoid warnings
    effective_pin = pin_memory and torch.cuda.is_available()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=effective_pin,
        drop_last=drop_last,
        persistent_workers=(num_workers > 0),
    )
    logger.info(
        "DataLoader[%s] | batches=%d | batch_size=%d | workers=%d | shuffle=%s",
        dataset.split,
        len(loader),
        batch_size,
        num_workers,
        shuffle,
    )
    return loader


# ── Combined train + val factory ──────────────────────────────────────────────

def build_dataloaders(
    annotation_file: Path,
    videos_dir: Path,
    batch_size: int = 8,
    num_frames: int = 16,
    num_workers: int = 2,
    img_size: int = 224,
    max_classes: Optional[int] = None,
    pin_memory: bool = True,
    drop_last_train: bool = True,
) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    """Build train and validation DataLoaders for WLASL.

    A single :attr:`label_to_idx` mapping is built from the training split
    and reused for the validation split to guarantee consistent class indices.

    Parameters
    ----------
    annotation_file:
        Path to the WLASL annotation JSON (e.g. ``WLASL_v0.3.json``).
    videos_dir:
        Directory containing ``<video_id>.mp4`` files.
    batch_size:
        Number of clips per mini-batch (applies to both loaders).
    num_frames:
        Number of frames uniformly sampled per video clip.
    num_workers:
        DataLoader subprocess workers.
    img_size:
        Spatial resolution for resizing frames (height = width = ``img_size``).
    max_classes:
        Restrict to the first *max_classes* glosses (alphabetically).
        Use ``100`` for WLASL100, ``300`` for WLASL300, ``None`` for all.
    pin_memory:
        Pin host memory (auto-disabled without CUDA).
    drop_last_train:
        Drop the last incomplete training batch.

    Returns
    -------
    tuple[DataLoader, DataLoader, dict[str, int]]
        - ``train_loader``  — DataLoader for the training split.
        - ``val_loader``    — DataLoader for the validation split.
        - ``label_to_idx``  — Consistent ``{gloss: class_index}`` mapping.
    """
    train_transform = get_train_transforms(img_size)
    val_transform = get_val_transforms(img_size)

    # Build training dataset first to establish the label map
    train_dataset = WLASLDataset(
        annotation_file=annotation_file,
        videos_dir=videos_dir,
        split="train",
        num_frames=num_frames,
        transform=train_transform,
        max_classes=max_classes,
    )
    label_to_idx = train_dataset.label_to_idx

    # Build validation dataset with the same label map
    val_dataset = WLASLDataset(
        annotation_file=annotation_file,
        videos_dir=videos_dir,
        split="val",
        num_frames=num_frames,
        transform=val_transform,
        label_to_idx=label_to_idx,  # ensures consistent class indices
        max_classes=max_classes,
    )

    train_loader = build_single_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last_train,
    )
    val_loader = build_single_dataloader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, label_to_idx


# ── Convenience wrapper using DatasetConfig ───────────────────────────────────

def build_dataloaders_from_config(
    cfg,  # DatasetConfig — avoid circular import by not type-hinting
    batch_size: int = 8,
    num_frames: int = 16,
    num_workers: int = 2,
    img_size: int = 224,
    max_classes: Optional[int] = None,
) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    """Build DataLoaders directly from a :class:`~src.dataset.dataset_config.DatasetConfig`.

    This is the recommended entry point when ``DatasetConfig`` is used
    (e.g. in Kaggle notebooks or when using environment auto-detection).

    Parameters
    ----------
    cfg:
        An instantiated :class:`~src.dataset.dataset_config.DatasetConfig`.
    batch_size, num_frames, num_workers, img_size, max_classes:
        Forwarded to :func:`build_dataloaders`.

    Returns
    -------
    tuple[DataLoader, DataLoader, dict[str, int]]
        ``(train_loader, val_loader, label_to_idx)``

    Raises
    ------
    FileNotFoundError
        If ``cfg.annotation_file`` or ``cfg.videos_dir`` does not exist.
    """
    if cfg.annotation_file is None or not cfg.annotation_file.is_file():
        raise FileNotFoundError(
            f"Annotation file not found: {cfg.annotation_file}\n"
            "  → Run `python src/dataset/dataset_config.py` to check your config."
        )
    if cfg.videos_dir is None or not cfg.videos_dir.is_dir():
        raise FileNotFoundError(
            f"Videos directory not found: {cfg.videos_dir}\n"
            "  → Ensure WLASL videos are present in the expected location."
        )

    return build_dataloaders(
        annotation_file=cfg.annotation_file,
        videos_dir=cfg.videos_dir,
        batch_size=batch_size,
        num_frames=num_frames,
        num_workers=num_workers,
        img_size=img_size,
        max_classes=max_classes,
    )
