# MediaPipe Landmark Extraction Pipeline

> **Phase 5 (MediaPipe) document** — Complete guide to extracting hand landmarks from WLASL videos.

---

## 1. Overview

This pipeline converts raw WLASL video files into structured NumPy arrays of hand keypoints using **MediaPipe Hands**. The output is used in Phase 6 to train a keypoint-based sign language model.

```
Raw Video (.mp4)
    │
    ▼
HandDetector             ← MediaPipe Hands inference (configurable confidence)
    │
    ▼
LandmarkExtractor        ← Parses raw results → structured (21, 3) arrays per hand
    │
    ▼
LandmarkSequenceBuilder  ← Processes all frames → (T, 2, 21, 3) sequence with pad/truncate
    │
    ▼
LandmarkSaver            ← Writes .npy or .csv to data/processed/
    │
    ▼
manifest.csv             ← Index of all saved files
```

---

## 2. Folder Responsibilities

| File | Responsibility |
|---|---|
| `hand_detector.py` | Wraps MediaPipe Hands. Configurable `max_num_hands`, confidence, `model_complexity`. Context manager support. |
| `landmark_extractor.py` | Parses raw MediaPipe results. Returns `{"left": (21,3), "right": (21,3)}` per frame. Defines `HandLandmarks` dataclass. |
| `landmark_sequence.py` | Opens a video, samples/reads frames, extracts landmarks, pads/truncates to `target_length`. Returns `(T, 2, 21, 3)` array. |
| `landmark_utils.py` | Stateless helpers: normalize, flatten, bounding box, pixel conversion, validation. |
| `save_landmarks.py` | `LandmarkSaver` for `.npy` and `.csv`. `process_dataset()` for batch extraction of all WLASL videos. Writes `manifest.csv`. |
| `visualize_landmarks.py` | OpenCV drawing utilities for debugging. `draw_landmarks_on_frame`, `annotate_frame`, `save_frame`, `visualize_video`. |

---

## 3. Landmark Format

### Per-hand array: `(21, 3)` float32

Each of the 21 rows corresponds to one MediaPipe hand landmark:

| Index | Landmark name |
|---|---|
| 0 | WRIST |
| 1–4 | THUMB (CMC, MCP, IP, TIP) |
| 5–8 | INDEX FINGER (MCP, PIP, DIP, TIP) |
| 9–12 | MIDDLE FINGER (MCP, PIP, DIP, TIP) |
| 13–16 | RING FINGER (MCP, PIP, DIP, TIP) |
| 17–20 | PINKY (MCP, PIP, DIP, TIP) |

Each row contains `(x, y, z)`:
- `x`, `y` — Normalised image coordinates in `[0, 1]`.
- `z` — Relative depth (roughly scale-relative to the hand size).

### Per-frame array: `(2, 21, 3)` float32

- Index 0 → **left hand** (zeros if not detected).
- Index 1 → **right hand** (zeros if not detected).

### Per-video sequence: `(T, 2, 21, 3)` float32

T = target number of frames (after uniform sampling + padding/truncation).

### Flattened vector: `(T, 126)` float32

`seq.flattened()` reshapes `(T, 2, 21, 3)` → `(T, 126)` for easy input into dense layers.

---

## 4. Output Files

```
data/processed/
├── manifest.csv                  ← full index of all saved files
├── train/
│   ├── 00001_book.npy            ← (T, 2, 21, 3) float32
│   ├── 00002_drink.npy
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

### `.npy` format (default, recommended)

Binary NumPy format. Load with:

```python
import numpy as np
seq = np.load("data/processed/train/00001_book.npy")
print(seq.shape)  # (30, 2, 21, 3)
```

### `.csv` format (optional, human-readable)

One file per video. Columns: `frame_idx, left_x0, left_y0, left_z0, ..., right_x20, right_y20, right_z20` (127 columns total).

```python
import pandas as pd
df = pd.read_csv("data/processed/train/00001_book.csv")
print(df.shape)  # (30, 127)
```

### `manifest.csv` columns

| Column | Description |
|---|---|
| `video_id` | WLASL 5-digit video ID |
| `label` | Gloss string |
| `split` | `"train"` / `"val"` / `"test"` |
| `filepath` | Absolute path to the saved `.npy` or `.csv` |
| `num_frames` | Frames in the saved sequence (= `target_length`) |
| `source_frames` | Total frames in the original video |
| `num_detected` | Frames where ≥1 hand was detected |
| `detection_rate` | `num_detected / num_frames` |
| `was_padded` | `True` if video was shorter than `target_length` |
| `was_truncated` | `True` if video was longer than `target_length` |

---

## 5. How to Run

### Step 1 — Install dependencies

```bash
pip install mediapipe opencv-python numpy
# On Kaggle:
# !pip install mediapipe
```

### Step 2 — Verify environment

```python
from src.dataset.dataset_config import DatasetConfig
cfg = DatasetConfig()
for k, (ok, msg) in cfg.validate().items():
    print(f"{'✔' if ok else '✘'}  {k}: {msg}")
```

### Step 3 — Extract landmarks (full dataset)

```bash
# WLASL100 subset, 30 frames per video, NumPy output
python src/landmarks/save_landmarks.py \
    --fmt npy \
    --target_length 30 \
    --splits train val \
    --max_classes 100

# CSV output for manual inspection
python src/landmarks/save_landmarks.py \
    --fmt csv \
    --target_length 30 \
    --max_videos 10        # quick smoke test
```

### Step 4 — Inspect a single video

```python
from pathlib import Path
from src.landmarks.hand_detector import HandDetector
from src.landmarks.landmark_extractor import LandmarkExtractor
from src.landmarks.landmark_sequence import LandmarkSequenceBuilder

with HandDetector(max_num_hands=2, min_detection_confidence=0.6) as det:
    ext = LandmarkExtractor(det)
    builder = LandmarkSequenceBuilder(ext, target_length=30)
    seq = builder.build_from_video(
        Path("data/raw/videos/00001.mp4"),
        video_id="00001",
        label="book",
    )

print(seq.sequence.shape)      # (30, 2, 21, 3)
print(seq.detection_rate)      # 0.0–1.0
```

### Step 5 — Visualise landmarks (debugging)

```python
import cv2
from src.landmarks.hand_detector import HandDetector
from src.landmarks.visualize_landmarks import annotate_frame, save_frame

cap = cv2.VideoCapture("data/raw/videos/00001.mp4")
_, frame = cap.read()
cap.release()

with HandDetector(max_num_hands=2) as det:
    results = det.detect(frame)

annotated = annotate_frame(frame, results, text="00001 – book")
save_frame(annotated, "outputs/debug/00001_frame0.jpg")
```

### Kaggle (inside a notebook cell)

```python
import subprocess, sys
REPO = "/kaggle/working/motion-sign-language-recognition"
sys.path.insert(0, REPO)

subprocess.run(["pip", "install", "mediapipe"], check=True)

from src.landmarks.save_landmarks import process_dataset
from src.dataset.dataset_config import get_default_config

cfg = get_default_config()
summary = process_dataset(
    annotation_file=cfg.annotation_file,
    videos_dir=cfg.videos_dir,
    output_dir=cfg.processed_dir,
    fmt="npy",
    target_length=30,
    splits=["train", "val"],
    max_classes=100,
)
print(summary)
```

---

## 6. Normalisation (landmark_utils)

Available normalisation functions in `landmark_utils.py`:

| Function | Effect |
|---|---|
| `normalize_to_wrist(landmarks)` | Translate so wrist (landmark 0) is at the origin. Removes absolute position. |
| `normalize_to_unit_scale(landmarks)` | Scale so max absolute coord = 1. Removes hand size variation. |
| `normalize_landmarks(landmarks)` | Both of the above in sequence. Recommended for training. |
| `flatten_landmarks(landmarks)` | `(21, 3)` → `(63,)` |
| `flatten_two_hands(landmarks)` | `(2, 21, 3)` → `(126,)` |
| `compute_bounding_box(landmarks)` | Returns `(x_min, y_min, x_max, y_max)` in normalised coords. |

---

## 7. Future Integration with Training Pipeline

In Phase 6, the saved landmark sequences will replace raw video frames as the input to the model:

```
manifest.csv
    ↓
LandmarkDataset (PyTorch Dataset)
    - Reads .npy files via np.load()
    - Applies normalize_landmarks() per hand
    - Returns (T, 126) flattened tensor + label
    ↓
LandmarkDataLoader
    ↓
LandmarkMLP / LandmarkLSTM / LandmarkGNN model
```

Benefits over raw frame training (Phase 5):
- Much smaller data size (`.npy` vs `.mp4`).
- Faster DataLoader (no OpenCV video decode per batch).
- Explicit structural prior (skeleton topology for GNN).
- Rotation/scale invariant with wrist-relative normalisation.

---

## 8. Known Limitations

| Issue | Detail |
|---|---|
| ~15–20% missing videos | WLASL has broken URLs. Missing videos are skipped automatically and logged. |
| MediaPipe mirror convention | MediaPipe reports handedness from the camera's perspective. Left/right labels may be flipped compared to the signer's actual hand. |
| `static_image_mode=True` for batch | Temporal tracking is disabled in batch mode (each frame is processed independently). This is intentional for offline extraction. |
| No pose / face landmarks | Only hand landmarks (21 per hand) are extracted. Pose and face landmarks (available in MediaPipe Holistic) are not yet used. |
| Depth (`z`) coordinate | The z value is MediaPipe's approximate relative depth, not metric depth. Reliability is lower than x, y. |

---

*Last updated: Phase 5 (MediaPipe) — Landmark Extraction Pipeline*
