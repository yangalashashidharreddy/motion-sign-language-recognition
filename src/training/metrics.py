"""
metrics.py
==========
Reusable metric utilities for training and evaluation.

Provides:
    - :class:`AverageMeter`  — tracks the running mean of a scalar value.
    - :func:`topk_accuracy`  — computes Top-k accuracy from logits.
    - :func:`accuracy`       — computes simple Top-1 accuracy (convenience wrapper).
    - :class:`MetricTracker` — aggregates multiple named metrics in one object.

All functions are stateless except for :class:`AverageMeter` and
:class:`MetricTracker`, making them safe to use across processes.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F  # noqa: N812


# ── AverageMeter ──────────────────────────────────────────────────────────────

class AverageMeter:
    """Track and compute the running mean and current value of a scalar.

    Typical usage inside a training loop::

        loss_meter = AverageMeter("loss", fmt=".4f")

        for batch in loader:
            loss = criterion(output, target)
            loss_meter.update(loss.item(), n=batch_size)

        print(loss_meter)            # "loss: avg=1.2345 | last=1.1234 | n=512"
        print(loss_meter.avg)        # 1.2345

    Parameters
    ----------
    name:
        Human-readable label for this metric (used in ``__str__``).
    fmt:
        Python format specification for float values (default ``".4f"``).
    """

    def __init__(self, name: str = "", fmt: str = ".4f") -> None:
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        """Reset all accumulators to zero."""
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        """Accumulate a new value.

        Parameters
        ----------
        val:
            The new measurement (e.g. loss or accuracy for one batch).
        n:
            Weight of this measurement (typically the batch size).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __str__(self) -> str:
        fmt = f"{{:{self.fmt}}}"
        return (
            f"{self.name}: avg={fmt.format(self.avg)} "
            f"| last={fmt.format(self.val)} "
            f"| n={self.count}"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"AverageMeter(name={self.name!r}, avg={self.avg:.6f}, n={self.count})"


# ── Top-k Accuracy ────────────────────────────────────────────────────────────

def topk_accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    topk: Sequence[int] = (1,),
) -> list[float]:
    """Compute Top-k accuracy for a batch of predictions.

    Parameters
    ----------
    output:
        Raw logits from the model, shape ``(batch_size, num_classes)``.
    target:
        Ground-truth class indices, shape ``(batch_size,)``.
    topk:
        Tuple of k values to compute accuracy for.
        Defaults to ``(1,)`` for Top-1 only.

    Returns
    -------
    list[float]
        Accuracy percentage (0–100) for each k in *topk*, in the same order.

    Example
    -------
    ::

        top1, top5 = topk_accuracy(logits, labels, topk=(1, 5))
    """
    with torch.no_grad():
        max_k = max(topk)
        batch_size = target.size(0)

        # Get the top-max_k predicted class indices per sample
        # pred: (batch_size, max_k)
        _, pred = output.topk(max_k, dim=1, largest=True, sorted=True)
        pred = pred.t()                                    # (max_k, batch_size)
        correct = pred.eq(target.unsqueeze(0).expand_as(pred))  # (max_k, batch_size)

        results: list[float] = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            results.append((correct_k * 100.0 / batch_size).item())
        return results


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Top-1 accuracy as a percentage (0–100).

    Convenience wrapper around :func:`topk_accuracy`.

    Parameters
    ----------
    output:
        Raw logits, shape ``(batch_size, num_classes)``.
    target:
        Ground-truth class indices, shape ``(batch_size,)``.

    Returns
    -------
    float
        Top-1 accuracy in percent.
    """
    return topk_accuracy(output, target, topk=(1,))[0]


# ── MetricTracker ─────────────────────────────────────────────────────────────

class MetricTracker:
    """Aggregate multiple named metrics in a single object.

    Parameters
    ----------
    names:
        Metric names to track (e.g. ``["loss", "acc_top1", "acc_top5"]``).
    fmt:
        Default format spec for all meters (can be changed per-meter via
        ``self.meters[name].fmt``).

    Example
    -------
    ::

        tracker = MetricTracker(["loss", "top1", "top5"])
        tracker.update("loss", loss.item(), n=batch_size)
        tracker.update("top1", t1, n=batch_size)
        print(tracker.summary())
    """

    def __init__(self, names: list[str], fmt: str = ".4f") -> None:
        self.meters: dict[str, AverageMeter] = {
            name: AverageMeter(name, fmt) for name in names
        }

    def reset(self) -> None:
        """Reset all tracked meters."""
        for meter in self.meters.values():
            meter.reset()

    def update(self, name: str, val: float, n: int = 1) -> None:
        """Update a named metric.

        Parameters
        ----------
        name:
            Metric name (must exist in the tracker).
        val:
            New measurement value.
        n:
            Batch weight (typically batch size).
        """
        if name not in self.meters:
            raise KeyError(f"Metric '{name}' is not tracked. Known: {list(self.meters)}")
        self.meters[name].update(val, n)

    def avg(self, name: str) -> float:
        """Return the current running average for *name*."""
        return self.meters[name].avg

    def summary(self) -> dict[str, float]:
        """Return a ``{name: average}`` dict for all tracked metrics."""
        return {name: meter.avg for name, meter in self.meters.items()}

    def __str__(self) -> str:
        return " | ".join(str(m) for m in self.meters.values())
