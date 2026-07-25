"""
baseline_model.py
=================
Baseline CNN + GRU model for isolated Word-Level Sign Language Recognition.

Architecture
------------
Input: a clip of T RGB frames, each resized to (H, W).

    [B, T, C, H, W]
         │
    ┌────▼────────┐
    │ FrameEncoder│  ResNet-18 backbone (pretrained on ImageNet)
    │  per frame  │  Output: [B, T, feature_dim]
    └────┬────────┘
         │
    ┌────▼────────┐
    │  GRU / LSTM │  Sequence model across T frames
    │             │  Output: [B, hidden_dim]  (last hidden state)
    └────┬────────┘
         │
    ┌────▼────────┐
    │  Classifier │  Linear → num_classes
    └─────────────┘

Design decisions
----------------
- ResNet-18 is lightweight enough for Kaggle's free GPU tier.
- GRU is used instead of LSTM for slightly fewer parameters and comparable
  performance on short sequences.
- The CNN backbone is optionally frozen for the first few epochs to allow
  the GRU and classifier to warm up (see ``freeze_backbone()``).
- The model accepts clips of any length T at inference time.

Usage
-----
::

    from src.models.baseline_model import SignLanguageBaseline, build_model

    model = build_model(num_classes=100)
    x = torch.randn(4, 16, 3, 224, 224)   # (batch=4, T=16, C=3, H=224, W=224)
    logits = model(x)                       # (4, 100)
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models

logger = logging.getLogger(__name__)


# ── Frame-level CNN encoder ───────────────────────────────────────────────────

class FrameEncoder(nn.Module):
    """Extract a fixed-size feature vector from a single RGB frame.

    Uses a ResNet-18 backbone pre-trained on ImageNet.  The original
    fully-connected head is replaced by a configurable projection layer.

    Parameters
    ----------
    feature_dim:
        Output feature dimensionality (default: 512).
    pretrained:
        Load ImageNet weights (default: ``True``).
    dropout:
        Dropout probability applied before the projection (default: 0.2).
    """

    def __init__(
        self,
        feature_dim: int = 512,
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet18(weights=weights)

        # Keep all layers except the final FC
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        backbone_out_dim = 512  # ResNet-18 penultimate dim

        self.projection = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(backbone_out_dim, feature_dim),
            nn.ReLU(inplace=True),
        )
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for a single frame (or a batch of frames).

        Parameters
        ----------
        x:
            Frame tensor of shape ``(batch_size, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Feature vector of shape ``(batch_size, feature_dim)``.
        """
        # backbone output: (B, 512, 1, 1) for ResNet-18
        features = self.backbone(x)
        features = features.flatten(1)          # (B, 512)
        return self.projection(features)        # (B, feature_dim)


# ── Sequence model ────────────────────────────────────────────────────────────

class SignLanguageBaseline(nn.Module):
    """End-to-end baseline for isolated sign language recognition.

    Parameters
    ----------
    num_classes:
        Number of output sign classes (e.g. 100 for WLASL100).
    feature_dim:
        Dimensionality of the frame-level CNN feature (default: 512).
    hidden_dim:
        GRU hidden state size (default: 256).
    num_rnn_layers:
        Number of stacked GRU layers (default: 2).
    rnn_dropout:
        Dropout between GRU layers (only applied when ``num_rnn_layers > 1``).
    classifier_dropout:
        Dropout before the final linear classifier (default: 0.5).
    pretrained_cnn:
        Whether to initialise the CNN backbone with ImageNet weights.
    bidirectional:
        Use a bidirectional GRU (doubles effective hidden dim).

    Example
    -------
    ::

        model = SignLanguageBaseline(num_classes=100)
        x = torch.randn(2, 16, 3, 224, 224)
        logits = model(x)   # (2, 100)
    """

    def __init__(
        self,
        num_classes: int,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        num_rnn_layers: int = 2,
        rnn_dropout: float = 0.3,
        classifier_dropout: float = 0.5,
        pretrained_cnn: bool = True,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        # CNN: per-frame feature extractor
        self.frame_encoder = FrameEncoder(
            feature_dim=feature_dim,
            pretrained=pretrained_cnn,
        )

        # GRU: temporal sequence model
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_rnn_layers,
            batch_first=True,
            dropout=rnn_dropout if num_rnn_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Classifier: maps GRU output → class logits
        gru_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=classifier_dropout),
            nn.Linear(gru_out_dim, num_classes),
        )

        self._log_architecture()

    def _log_architecture(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "SignLanguageBaseline | classes=%d | total_params=%s | trainable=%s",
            self.num_classes,
            f"{total:,}",
            f"{trainable:,}",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass over a video clip.

        Parameters
        ----------
        x:
            Clip tensor of shape ``(batch_size, T, C, H, W)`` where T is the
            number of sampled frames.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch_size, num_classes)``.
        """
        B, T, C, H, W = x.shape

        # Encode each frame independently: merge batch + time dims
        x_flat = x.view(B * T, C, H, W)            # (B*T, C, H, W)
        frame_feats = self.frame_encoder(x_flat)    # (B*T, feature_dim)
        frame_feats = frame_feats.view(B, T, -1)   # (B, T, feature_dim)

        # GRU over the frame sequence
        gru_out, _ = self.gru(frame_feats)          # (B, T, hidden_dim)

        # Use only the last time-step's hidden state for classification
        last_hidden = gru_out[:, -1, :]             # (B, hidden_dim)

        return self.classifier(last_hidden)          # (B, num_classes)

    def freeze_backbone(self) -> None:
        """Freeze the CNN backbone weights.

        Useful at the start of training to let the GRU and classifier
        warm up before fine-tuning the backbone.
        """
        for param in self.frame_encoder.backbone.parameters():
            param.requires_grad = False
        logger.info("CNN backbone frozen.")

    def unfreeze_backbone(self) -> None:
        """Unfreeze the CNN backbone weights for end-to-end fine-tuning."""
        for param in self.frame_encoder.backbone.parameters():
            param.requires_grad = True
        logger.info("CNN backbone unfrozen for end-to-end fine-tuning.")


# ── Factory function ──────────────────────────────────────────────────────────

def build_model(
    num_classes: int,
    feature_dim: int = 512,
    hidden_dim: int = 256,
    num_rnn_layers: int = 2,
    rnn_dropout: float = 0.3,
    classifier_dropout: float = 0.5,
    pretrained_cnn: bool = True,
    bidirectional: bool = False,
    device: Optional[torch.device] = None,
) -> SignLanguageBaseline:
    """Build and return a :class:`SignLanguageBaseline` model.

    Parameters
    ----------
    num_classes:
        Number of sign classes.
    feature_dim:
        CNN feature output dimension.
    hidden_dim:
        GRU hidden state dimension.
    num_rnn_layers:
        Number of stacked GRU layers.
    rnn_dropout:
        GRU inter-layer dropout.
    classifier_dropout:
        Dropout before the classifier head.
    pretrained_cnn:
        Load ImageNet weights for the CNN backbone.
    bidirectional:
        Use bidirectional GRU.
    device:
        Target device. Auto-detected if ``None``.

    Returns
    -------
    SignLanguageBaseline
        Model moved to *device*.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SignLanguageBaseline(
        num_classes=num_classes,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_rnn_layers=num_rnn_layers,
        rnn_dropout=rnn_dropout,
        classifier_dropout=classifier_dropout,
        pretrained_cnn=pretrained_cnn,
        bidirectional=bidirectional,
    )
    model = model.to(device)
    logger.info("Model moved to device: %s", device)
    return model
