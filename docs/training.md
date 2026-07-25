# Training Guide — WLASL Baseline Model

> **Phase 5 document** — End-to-end training pipeline for the CNN + GRU baseline.

---

## 1. Overview

This guide documents the Phase 5 training pipeline. The goal is a **working
baseline** — not maximum accuracy — using a straightforward CNN + GRU architecture
trained end-to-end on raw video frames from the WLASL dataset.

### Architecture summary

```
Input clip: (B, T, 3, 224, 224)
      │
   ┌──▼──────────────────────────────┐
   │  FrameEncoder (per frame)        │
   │  ResNet-18 backbone (ImageNet)   │
   │  → Linear projection → feature  │
   └──┬──────────────────────────────┘
      │  (B, T, 512)
   ┌──▼──────────────────────────────┐
   │  GRU (2 layers, hidden=256)      │
   │  Processes temporal sequence     │
   └──┬──────────────────────────────┘
      │  (B, 256) — last hidden state
   ┌──▼──────────────────────────────┐
   │  Classifier (Dropout + Linear)   │
   └──────────────────────────────────┘
      │  (B, num_classes)
```

---

## 2. Folder Responsibilities

| File | Responsibility |
|---|---|
| `src/models/baseline_model.py` | CNN + GRU model architecture and `build_model()` factory |
| `src/training/dataset.py` | `WLASLDataset` — reads JSON, samples frames, returns tensors |
| `src/training/dataloader.py` | `build_dataloaders()` factory with configurable batch/workers |
| `src/training/train.py` | Full training loop, checkpoint saving, resume support |
| `src/training/evaluate.py` | Evaluation pass — loss and Top-1 accuracy |
| `src/training/metrics.py` | `AverageMeter`, `topk_accuracy`, `MetricTracker` utilities |
| `src/dataset/dataset_config.py` | Environment-aware path resolution (local ↔ Kaggle) |

---

## 3. Expected Inputs

### Dataset (required before training)

```
data/raw/                          (local)
├── WLASL_v0.3.json                ← annotation file
└── videos/
    ├── 00001.mp4
    ├── 00002.mp4
    └── ...

/kaggle/input/wlasl-complete/      (Kaggle)
├── WLASL_v0.3.json
└── videos/
```

### Training script inputs (CLI arguments)

| Argument | Default | Description |
|---|---|---|
| `--epochs` | `30` | Total training epochs |
| `--batch_size` | `8` | Clips per mini-batch |
| `--num_classes` | `100` | Number of sign classes (use 100/300/1000/2000) |
| `--num_frames` | `16` | Frames uniformly sampled per clip |
| `--img_size` | `224` | Frame spatial resolution (H = W) |
| `--lr` | `1e-4` | Initial Adam learning rate |
| `--weight_decay` | `1e-4` | L2 regularisation |
| `--label_smoothing` | `0.1` | Cross-entropy label smoothing |
| `--freeze_epochs` | `5` | Epochs to keep CNN backbone frozen |
| `--num_workers` | `2` | DataLoader subprocess workers |
| `--resume` | `None` | Path to checkpoint to resume from |
| `--no_pretrained` | `False` | Train CNN from scratch (not recommended) |

---

## 4. Expected Outputs

```
outputs/
├── checkpoints/
│   ├── best.pt          ← best validation accuracy checkpoint
│   ├── last.pt          ← most recent epoch checkpoint
│   └── label_to_idx.json ← {gloss: class_index} mapping
└── logs/
    └── train.log        ← full training log
```

### Checkpoint contents (`best.pt` / `last.pt`)

| Key | Type | Description |
|---|---|---|
| `epoch` | int | Epoch at which checkpoint was saved |
| `model_state_dict` | dict | PyTorch model weights |
| `optimizer_state_dict` | dict | Adam optimiser state |
| `scheduler_state_dict` | dict | LR scheduler state |
| `best_top1` | float | Best validation Top-1 (%) seen so far |
| `val_top1` | float | Validation Top-1 at this epoch |
| `train_loss` | float | Training loss at this epoch |
| `label_to_idx` | dict | Class label map (embedded for portability) |
| `args` | dict | Full argparse namespace (for reproducibility) |

---

## 5. Training Workflow

### Step 1 — Verify environment

```python
from src.dataset.dataset_config import DatasetConfig
cfg = DatasetConfig()
for k, (ok, msg) in cfg.validate().items():
    print(f"{'✔' if ok else '✘'}  {k}: {msg}")
```

### Step 2 — Run training (local)

```bash
# WLASL100 subset — good starting point
python src/training/train.py \
    --epochs 30 \
    --batch_size 8 \
    --num_classes 100 \
    --num_frames 16 \
    --lr 1e-4 \
    --num_workers 2
```

### Step 3 — Resume from checkpoint

```bash
python src/training/train.py \
    --resume outputs/checkpoints/last.pt \
    --epochs 50
```

### Step 4 — Evaluate the best checkpoint

```bash
python src/training/evaluate.py \
    --checkpoint outputs/checkpoints/best.pt \
    --split val \
    --batch_size 16
```

---

## 6. Backbone Freeze Strategy

To prevent the pre-trained ResNet-18 from being destroyed early in training,
the CNN backbone is **frozen** for the first `--freeze_epochs` epochs (default: 5).
During this period, only the GRU and classifier weights are updated.

After epoch `freeze_epochs`, the backbone is **unfrozen** and the entire network
is trained end-to-end at a reduced learning rate (`lr × 0.1`).

```
Epoch 1–5:   Backbone FROZEN  → GRU + Classifier train at lr=1e-4
Epoch 6–30:  Backbone UNFROZEN → full network trains at lr=1e-5
```

---

## 7. Frame Sampling

For each video clip:
1. Open the video with OpenCV.
2. Determine `total_frames` via `CAP_PROP_FRAME_COUNT`.
3. Compute `T` linearly-spaced indices across `[0, total_frames-1]`.
4. Seek to each index using `CAP_PROP_POS_FRAMES`.
5. Convert BGR → RGB.
6. Apply the split-specific transform (augmentation for train, resize-only for val).

If `total_frames < num_frames`, frames are cyclically repeated to fill the clip.

---

## 8. Data Augmentation

| Transform | Train | Val |
|---|---|---|
| Resize to `img_size × img_size` | ✅ | ✅ |
| Random horizontal flip (p=0.5) | ✅ | ❌ |
| Color jitter (brightness/contrast) | ✅ | ❌ |
| ImageNet normalisation | ✅ | ✅ |

---

## 9. Baseline Performance Expectations

> These are rough estimates for a fresh ResNet-18 + GRU model on WLASL.
> Actual results depend on GPU, batch size, and available videos.

| Subset | Expected Top-1 (30 epochs) |
|---|---|
| WLASL100 | ~30–45% |
| WLASL300 | ~20–35% |
| WLASL2000 | ~5–15% |

These are **baselines**, not state-of-the-art results. Future phases will
improve accuracy using MediaPipe keypoints, attention mechanisms, and GNN models.

---

## 10. Known Limitations

| Issue | Detail |
|---|---|
| Missing videos | WLASL has ~15–20% missing videos (URL rot). These are skipped automatically. |
| Variable FPS | Annotation FPS ≠ actual video FPS. Frame sampling uses actual video metadata. |
| Short clips | Clips shorter than `num_frames` are padded by repetition. |
| No multi-GPU | `DataParallel` / `DistributedDataParallel` not yet implemented. |

---

*Last updated: Phase 5 — Baseline Model Training*
