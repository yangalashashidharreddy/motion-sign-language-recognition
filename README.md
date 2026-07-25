# Motion Sign Language Recognition

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Phase%201%20%E2%80%93%20Initialized-orange)
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
| **2** | Data Pipeline & Preprocessing | 🔜 Upcoming |
| **3** | Baseline Model (CNN + LSTM) | 🔜 Upcoming |
| **4** | Advanced Models (GNN, Transformer) | 🔜 Upcoming |
| **5** | Real-time Inference & Optimisation | 🔜 Upcoming |
| **6** | Evaluation, Benchmarking & Docs | 🔜 Upcoming |

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

## 🤝 Contributing

Contributions are welcome! Please open an issue or pull request following the project's coding standards (black, isort, flake8, type annotations).

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
