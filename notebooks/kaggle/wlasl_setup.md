# Kaggle Environment Setup — WLASL Dataset

> **Phase 3 guide** — How to run Motion Sign Language Recognition training on Kaggle.

---

## 1. Overview

[Kaggle Notebooks](https://www.kaggle.com/docs/notebooks) provide free GPU/TPU
compute and native dataset mounting. This guide explains how to attach the WLASL
dataset to a Kaggle Notebook so the project code runs without any path changes.

---

## 2. Expected Kaggle Directory Structure

When a Kaggle dataset is attached to a notebook, it is mounted read-only under
`/kaggle/input/<dataset-slug>/`. The WLASL dataset should be mounted as:

```
/kaggle/
├── input/
│   └── wlasl-complete/           ← Kaggle dataset slug (see §4)
│       ├── WLASL_v0.3.json       ← annotation file
│       └── videos/
│           ├── 00001.mp4
│           ├── 00002.mp4
│           └── ...
└── working/                      ← your notebook's writable output dir
    └── motion-sign-language-recognition/   ← cloned repo (optional)
```

> **Note**: The exact slug depends on which Kaggle dataset you attach.
> Common slugs for WLASL are `wlasl-complete` or `risangbaskoro/wlasl-complete`.
> Adjust `KAGGLE_DATASET_SLUG` in `src/dataset/dataset_config.py` to match.

---

## 3. How to Attach the WLASL Dataset to a Kaggle Notebook

### Step-by-step

1. **Open your Kaggle Notebook** (or create a new one at kaggle.com/notebooks).

2. **Click "Add Data"** in the right-hand panel.

3. **Search for WLASL**:
   - In the search bar type: `WLASL`
   - Select the dataset: **WLASL Complete** by `risangbaskoro`
     (URL: `https://www.kaggle.com/datasets/risangbaskoro/wlasl-complete`)

4. **Click "Add"** — the dataset will be mounted at:
   ```
   /kaggle/input/wlasl-complete/
   ```

5. **Verify the mount** by running this cell in your notebook:
   ```python
   import os
   base = "/kaggle/input/wlasl-complete"
   print(os.listdir(base))
   # Expected: ['WLASL_v0.3.json', 'videos']
   ```

### Alternative: Upload your own dataset

If you downloaded WLASL directly from the official source:

1. Go to **kaggle.com/datasets** → **New Dataset**
2. Upload `WLASL_v0.3.json` and the `videos/` folder
3. Set the dataset slug (e.g., `my-wlasl`)
4. In your notebook, attach it — it will be mounted at:
   ```
   /kaggle/input/my-wlasl/
   ```
5. Update `KAGGLE_DATASET_SLUG = "my-wlasl"` in `dataset_config.py`

---

## 4. Cloning the Repository Inside Kaggle

Add this to the **first cell** of your Kaggle Notebook:

```python
# ── Clone repository ──────────────────────────────────────────────────────────
import subprocess, sys, os

REPO_URL  = "https://github.com/yangalashashidharreddy/motion-sign-language-recognition.git"
REPO_DIR  = "/kaggle/working/motion-sign-language-recognition"

if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)

# Add src/ to Python path so imports work
sys.path.insert(0, REPO_DIR)

print("Repository ready at:", REPO_DIR)
```

---

## 5. Environment Configuration

The project uses `src/dataset/dataset_config.py` to automatically detect
whether it is running on Kaggle or locally, and to set all paths accordingly.

```python
from src.dataset.dataset_config import DatasetConfig

cfg = DatasetConfig()          # auto-detects environment
print(cfg.raw_dir)             # /kaggle/input/wlasl-complete  OR  data/raw
print(cfg.annotation_file)    # /kaggle/input/wlasl-complete/WLASL_v0.3.json
print(cfg.videos_dir)          # /kaggle/input/wlasl-complete/videos
print(cfg.is_kaggle)           # True  (on Kaggle)  / False (local)
```

To override paths (e.g., if your dataset slug is different):

```python
cfg = DatasetConfig(kaggle_dataset_slug="my-wlasl")
```

Or use environment variables (no code change needed):

```bash
export WLASL_RAW_DIR=/path/to/custom/raw
export WLASL_ANNOTATION_FILE=/path/to/WLASL_v0.3.json
export WLASL_VIDEOS_DIR=/path/to/videos
```

---

## 6. Installing Dependencies in Kaggle

Most scientific libraries are pre-installed. Install project-specific ones:

```python
# In a Kaggle notebook cell
import subprocess
subprocess.run(["pip", "install", "rich", "omegaconf", "einops", "loguru"], check=True)
```

---

## 7. Verifying the Full Setup

Run this verification cell after cloning the repo and attaching the dataset:

```python
from src.dataset.dataset_config import DatasetConfig

cfg = DatasetConfig()
report = cfg.validate()

for key, (ok, msg) in report.items():
    icon = "✔" if ok else "✘"
    print(f"  {icon}  {key}: {msg}")
```

Expected output on Kaggle:

```
  ✔  is_kaggle:        True
  ✔  raw_dir:          /kaggle/input/wlasl-complete  (exists)
  ✔  annotation_file:  /kaggle/input/wlasl-complete/WLASL_v0.3.json  (exists)
  ✔  videos_dir:       /kaggle/input/wlasl-complete/videos  (exists)
  ✔  output_dir:       /kaggle/working/outputs  (exists)
```

---

## 8. Quick Reference

| Item | Local | Kaggle |
|---|---|---|
| Raw data | `data/raw/` | `/kaggle/input/wlasl-complete/` |
| Annotation JSON | `data/raw/WLASL_v0.3.json` | `/kaggle/input/wlasl-complete/WLASL_v0.3.json` |
| Videos directory | `data/raw/videos/` | `/kaggle/input/wlasl-complete/videos/` |
| Output / checkpoints | `outputs/` | `/kaggle/working/outputs/` |
| Config auto-detected | ✅ | ✅ |

---

*Last updated: Phase 3 — Kaggle Environment Preparation*
