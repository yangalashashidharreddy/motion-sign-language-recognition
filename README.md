# Motion Sign Language Recognition

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Phase%205%20%E2%80%93%20MediaPipe%20Landmarks-orange)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red?logo=pytorch&logoColor=white)

**A production-quality, deep-learning-powered system for recognizing sign language gestures from video and motion data.**

</div>

---

## 📌 Project Overview

This repository implements an end-to-end **Motion Sign Language Recognition** (MSLR) pipeline, capable of:

- Ingesting raw video / skeletal motion-capture sequences
- Extracting keypoint and optical-flow features
- Training sequence models (CNN + LSTM, Transformer, Graph Neural Networks)
- Running real-time inference from a webcam or pre-recorded video

The system targets both **isolated sign recognition** (one gloss per clip) and **continuous sign recognition** (segmented streams), and is designed to scale from research prototypes to production deployment.

---

## 🗂️ Repository Structure

```
motion-sign-language-recognition/
│
├── data/
│   ├── raw/            # Original, untouched input data (video files, motion CSVs)
│   ├── processed/      # Normalised tensors / features ready for training
│   └── annotations/    # Label files, vocabulary lists, split manifests
│
├── src/
│   ├── dataset/        # Dataset classes, data loaders, train/val/test split logic
│   ├── preprocessing/  # Video decoding, pose extraction, augmentation pipelines
│   ├── models/         # Neural-network architectures (CNN, LSTM, GNN, Transformer)
│   ├── training/       # Training loops, loss functions, schedulers, callbacks
│   ├── inference/      # Inference engine, real-time capture, post-processing
│   └── utils/          # Config management, logging, metrics, visualisation helpers
│
├── notebooks/          # Jupyter notebooks for EDA, prototyping, and visualisation
├── docs/               # Architecture diagrams, API docs, design decisions
├── outputs/            # Saved model checkpoints, logs, evaluation reports
├── tests/              # Unit and integration tests (pytest)
│
├── requirements.txt    # Python dependencies
├── pyproject.toml      # Build system config and tool settings
├── .gitignore          # Git exclusion rules
├── LICENSE             # MIT License
└── README.md           # This file
```

---

## 🚀 Planned Phases

| Phase | Title | Status |
|-------|-------|--------|
| **1** | Project Initialisation | ✅ Complete |
| **2** | Dataset Exploration | ✅ Complete |
| **3** | Kaggle Environment Preparation | ✅ Complete |
| **4** | Baseline Model Training (CNN + GRU) | ✅ Complete |
| **5** | MediaPipe Landmark Extraction | ✅ Complete |
| **6** | Advanced Models (GNN, Transformer) | 🔜 Upcoming |
| **7** | Real-time Inference & Optimisation | 🔜 Upcoming |
| **8** | Evaluation, Benchmarking & Docs | 🔜 Upcoming |

---

## 🛠️ Tech Stack

| Category | Libraries |
|----------|-----------|
| Deep Learning | PyTorch, torchvision |
| Computer Vision | OpenCV, MediaPipe, Albumentations |
| Pose Estimation | MediaPipe Holistic, OpenPose (optional) |
| Sequence Modelling | PyTorch (LSTM, GRU, Transformer) |
| Graph Neural Nets | PyTorch Geometric |
| Data | NumPy, Pandas, h5py |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Experiment Tracking | MLflow / Weights & Biases (wandb) |
| Testing | pytest, pytest-cov |
| Code Quality | black, isort, flake8, mypy |

---

## ⚙️ Getting Started

### Prerequisites

- Python **3.11+**
- A CUDA-capable GPU (recommended for training)
- `git`, `pip` / `conda`

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yangalashashidharreddy/motion-sign-language-recognition.git
cd motion-sign-language-recognition

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📦 Dataset Setup (WLASL)

> The dataset is **not** included in the repository. You must obtain it separately.

### Download

1. Visit the official WLASL homepage: **https://dxli94.github.io/WLASL/**
2. Complete the data request form and download:
   - `WLASL_v0.3.json` — annotation file
   - `videos.zip` — video archive
3. Extract and place files in the structure below.

### Directory layout

```
data/
└── raw/
    ├── WLASL_v0.3.json      ← annotation file
    └── videos/
        ├── 00001.mp4
        ├── 00002.mp4
        └── ...
```

> `data/raw/*` is excluded from version control via `.gitignore`.
> Only `.gitkeep` placeholder files are tracked.

---

## 🔍 Dataset Exploration Scripts

Once the dataset is in place, run the following scripts to explore it:

### 1. High-level overview

```bash
python src/dataset/explore_dataset.py
```

Prints: number of classes, total videos, average videos per class, train/val/test split, and sample gloss labels.

### 2. Deep annotation analysis

```bash
# Default: top 10 classes
python src/dataset/explore_annotations.py

# Custom: top 20 classes, custom annotation path
python src/dataset/explore_annotations.py --top 20 --json data/raw/WLASL_v0.3.json
```

Prints: per-class video counts, signer distribution, source breakdown, frame-length statistics, and bounding-box coverage.

### 3. Video metadata inspector

```bash
# Inspect a single video file
python src/dataset/video_info.py --video data/raw/videos/00001.mp4

# Inspect by WLASL video ID
python src/dataset/video_info.py --id 00001

# Batch summary of first 50 videos
python src/dataset/video_info.py --all --limit 50
```

Prints: FPS, resolution, duration, codec (FourCC), total frames, and matched annotation metadata (gloss, split, signer).

> 📖 See [docs/dataset_analysis.md](docs/dataset_analysis.md) for a full breakdown of the WLASL annotation format and known caveats.

---

## 💻 Development Environments

This project supports two environments with **zero code changes**.
All paths are resolved automatically by `src/dataset/dataset_config.py`.

### Local Development

```bash
# After placing the dataset in data/raw/ (see Dataset Setup above)
python -c "
from src.dataset.dataset_config import DatasetConfig
cfg = DatasetConfig()
for k, (ok, msg) in cfg.validate().items():
    print(f'  {\"✔\" if ok else \"✘\"}  {k}: {msg}')
"
```

| Path | Value |
|---|---|
| Raw data | `data/raw/` |
| Annotation JSON | `data/raw/WLASL_v0.3.json` |
| Videos | `data/raw/videos/` |
| Outputs | `outputs/` |

### Kaggle Training Workflow

1. **Attach the WLASL dataset** to your Kaggle Notebook:
   - Search for **"WLASL Complete"** in the Kaggle dataset panel and click **Add**.
   - It mounts at `/kaggle/input/wlasl-complete/`.

2. **Clone this repository** inside your notebook:
   ```python
   import subprocess, sys, os
   REPO = "/kaggle/working/motion-sign-language-recognition"
   subprocess.run(["git", "clone",
       "https://github.com/yangalashashidharreddy/motion-sign-language-recognition.git",
       REPO], check=True)
   sys.path.insert(0, REPO)
   ```

3. **Auto-detect config** — no changes needed:
   ```python
   from src.dataset.dataset_config import DatasetConfig
   cfg = DatasetConfig()     # detects Kaggle automatically
   print(cfg.is_kaggle)      # True
   print(cfg.annotation_file) # /kaggle/input/wlasl-complete/WLASL_v0.3.json
   ```

4. **Override paths** with environment variables if needed:
   ```bash
   export KAGGLE_DATASET_SLUG="my-custom-wlasl"
   export WLASL_OUTPUT_DIR="/kaggle/working/my-outputs"
   ```

| Path | Kaggle Value |
|---|---|
| Raw data | `/kaggle/input/wlasl-complete/` |
| Annotation JSON | `/kaggle/input/wlasl-complete/WLASL_v0.3.json` |
| Videos | `/kaggle/input/wlasl-complete/videos/` |
| Outputs | `/kaggle/working/outputs/` |

> 📖 See [notebooks/kaggle/wlasl_setup.md](notebooks/kaggle/wlasl_setup.md) for the full Kaggle setup guide.

---

## 🏋️‍♂️ Training the Baseline Model

A CNN + GRU baseline is implemented in `src/models/baseline_model.py` and
trained by `src/training/train.py`.

### Local Training

```bash
# Train on WLASL100 subset (100 classes)
python src/training/train.py \
    --epochs 30 \
    --batch_size 8 \
    --num_classes 100 \
    --num_frames 16 \
    --lr 1e-4 \
    --num_workers 2

# Resume from a checkpoint
python src/training/train.py \
    --resume outputs/checkpoints/last.pt \
    --epochs 50

# Evaluate the best checkpoint
python src/training/evaluate.py \
    --checkpoint outputs/checkpoints/best.pt
```

### Kaggle Training

Inside a Kaggle Notebook cell:

```python
import subprocess, sys, os

# Clone the repo
REPO = "/kaggle/working/motion-sign-language-recognition"
if not os.path.exists(REPO):
    subprocess.run(["git", "clone",
        "https://github.com/yangalashashidharreddy/motion-sign-language-recognition.git",
        REPO], check=True)
sys.path.insert(0, REPO)

# Install extra deps
subprocess.run(["pip", "install", "rich", "loguru"], check=True)

# Run training (dataset auto-detected at /kaggle/input/wlasl-complete/)
subprocess.run([
    "python", "src/training/train.py",
    "--epochs", "30",
    "--batch_size", "16",
    "--num_classes", "100",
    "--num_frames", "16",
    "--num_workers", "2",
], cwd=REPO, check=True)
```

### Outputs

| File | Description |
|---|---|
| `outputs/checkpoints/best.pt` | Checkpoint with the best validation Top-1 |
| `outputs/checkpoints/last.pt` | Most recent epoch checkpoint |
| `outputs/checkpoints/label_to_idx.json` | Gloss → class index mapping |
| `outputs/logs/train.log` | Full training log |

> 📖 See [docs/training.md](docs/training.md) for a complete guide including architecture
> details, all CLI arguments, frame sampling strategy, and expected accuracy ranges.

---

## 🖐️ MediaPipe Landmark Extraction

### Installation

```bash
# Local
pip install mediapipe

# Kaggle notebook
!pip install mediapipe
```

### Extract landmarks (full WLASL100 dataset)

```bash
python src/landmarks/save_landmarks.py \
    --fmt npy \
    --target_length 30 \
    --splits train val \
    --max_classes 100
```

### Extract from a single video (Python)

```python
from pathlib import Path
from src.landmarks.hand_detector import HandDetector
from src.landmarks.landmark_extractor import LandmarkExtractor
from src.landmarks.landmark_sequence import LandmarkSequenceBuilder

with HandDetector(max_num_hands=2) as det:
    ext = LandmarkExtractor(det)
    builder = LandmarkSequenceBuilder(ext, target_length=30)
    seq = builder.build_from_video(
        Path("data/raw/videos/00001.mp4"),
        video_id="00001", label="book",
    )
print(seq.sequence.shape)  # (30, 2, 21, 3)
```

### Expected output structure

```
data/processed/
├── manifest.csv
├── train/
│   ├── 00001_book.npy        ← (T, 2, 21, 3) float32
│   └── ...
└── val/
    └── ...
```

> 📖 See [docs/mediapipe_pipeline.md](docs/mediapipe_pipeline.md) for the full pipeline guide,
> landmark format reference, normalisation utilities, and Kaggle instructions.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or pull request following the project's coding standards (black, isort, flake8, type annotations).

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
