# Dataset Analysis — WLASL (Word-Level American Sign Language)

> **Phase 2 document** — updated as exploration progresses.

---

## 1. Dataset Overview

The **WLASL** (Word-Level American Sign Language) dataset is one of the largest
publicly available video datasets for American Sign Language recognition.

| Property | Details |
|---|---|
| Full name | Word-Level American Sign Language |
| Task | Isolated sign language recognition |
| Language | American Sign Language (ASL) |
| Classes (glosses) | Up to **2,000** sign words (WLASL2000 subset) |
| Total video instances | ~**21,083** across all splits |
| Video format | MP4 (H.264), collected from the web |
| Annotation format | JSON (one file covers all classes) |
| Homepage | https://dxli94.github.io/WLASL/ |
| Paper | *Word-level Deep Sign Language Recognition from Video*, WACV 2020 |

### Subsets

| Subset | Classes | Videos (approx.) |
|--------|---------|------------------|
| WLASL100 | 100 | ~2,000 |
| WLASL300 | 300 | ~5,100 |
| WLASL1000 | 1,000 | ~13,200 |
| WLASL2000 | 2,000 | ~21,083 |

---

## 2. Annotation Format

The dataset ships as a **single JSON file** (e.g., `WLASL_v0.3.json`).

### Top-level structure

```json
[
  {
    "gloss": "book",
    "instances": [ ... ]
  },
  {
    "gloss": "drink",
    "instances": [ ... ]
  }
]
```

The file is a JSON **array** where each element represents one sign class (gloss).

### Instance object

Each element inside `instances` describes a single video clip:

```json
{
  "video_id":    "00001",
  "split":       "train",
  "fps":         25,
  "frame_start":  1,
  "frame_end":   75,
  "bbox":        [0, 0, 640, 480],
  "signer_id":   3,
  "source":      "aslpro",
  "url":         "https://...",
  "variation_id": 0
}
```

### Field descriptions

| Field | Type | Description |
|---|---|---|
| `video_id` | string | Zero-padded 5-digit unique identifier (e.g., `"00001"`) |
| `split` | string | Dataset partition: `"train"`, `"val"`, or `"test"` |
| `fps` | int | Frames per second declared in the annotation |
| `frame_start` | int | First frame of the sign clip within the source video |
| `frame_end` | int | Last frame of the sign clip within the source video |
| `bbox` | list[int] | Bounding box `[x, y, width, height]` around the signer |
| `signer_id` | int | Anonymous signer identifier |
| `source` | string | Data source (e.g., `"aslpro"`, `"youtube"`, `"signingsavvy"`) |
| `url` | string | Original video URL (may be dead) |
| `variation_id` | int | Variation index when multiple signers produced the same gloss |

> **Note on `frame_start` / `frame_end`**: These are 1-indexed frame offsets
> relative to the raw downloaded video. The actual clip is the sub-sequence
> `[frame_start, frame_end]`. A clip length of `frame_end − frame_start + 1`
> frames applies.

---

## 3. Video Metadata Explanation

Each video in `data/raw/videos/` is identified by its zero-padded `video_id`
(e.g., `00001.mp4`). The following metadata properties matter during preprocessing:

| Property | Description | Typical value |
|---|---|---|
| **FPS** | Capture frame rate. Can vary between 25–30 fps across sources | 25 / 29.97 / 30 |
| **Resolution** | Width × Height in pixels. Highly variable across sources | 320×240 to 1920×1080 |
| **Total frames** | Raw frame count as reported by the video container | varies |
| **Duration** | Derived as `total_frames / fps` (seconds) | 1–5 s typical |
| **Codec (FourCC)** | Video compression format; H.264 (`avc1`) is most common | `avc1`, `mp4v` |

> **Important**: The `fps` field in the JSON annotation may **not** match the
> actual video FPS (it reflects the originally scraped source). Always read FPS
> directly from the file using OpenCV for reliable values.

---

## 4. Expected Directory Structure

Place the dataset files in the following layout before running any scripts:

```
motion-sign-language-recognition/
└── data/
    └── raw/
        ├── WLASL_v0.3.json          ← annotation file (required)
        └── videos/
            ├── 00001.mp4
            ├── 00002.mp4
            ├── 00003.mp4
            └── ...
```

> **Rules enforced by `.gitignore`**:
> - `data/raw/*` is excluded from version control (except `.gitkeep`)
> - Large dataset files must **never** be committed to the repository
> - Only the exploration scripts and documentation live in git

### Where to get the dataset

1. Visit the official WLASL homepage: https://dxli94.github.io/WLASL/
2. Fill in the data request form
3. Download `WLASL_v0.3.json` and the video archive
4. Extract videos into `data/raw/videos/`

---

## 5. Exploration Scripts

| Script | Purpose |
|---|---|
| [`src/dataset/explore_dataset.py`](../src/dataset/explore_dataset.py) | High-level stats: class count, video count, sample glosses, split distribution |
| [`src/dataset/explore_annotations.py`](../src/dataset/explore_annotations.py) | Deep-dive: per-class counts, signer distribution, frame-length stats, bbox coverage |
| [`src/dataset/video_info.py`](../src/dataset/video_info.py) | Video metadata: FPS, resolution, duration, codec for one or all videos |

### Quick start

```bash
# 1. High-level overview
python src/dataset/explore_dataset.py

# 2. Deep annotation analysis (top 15 classes)
python src/dataset/explore_annotations.py --top 15

# 3. Inspect a single video by file path
python src/dataset/video_info.py --video data/raw/videos/00001.mp4

# 4. Inspect by WLASL video ID
python src/dataset/video_info.py --id 00001

# 5. Batch summary of first 50 videos
python src/dataset/video_info.py --all --limit 50
```

---

## 6. Expected Output (when dataset is present)

### `explore_dataset.py` — example output

```
✔ Annotation file          : data/raw/WLASL_v0.3.json
✔ Number of classes        : 2,000
✔ Total video instances    : 21,083
✔ Avg videos per class     : 10.5
✔ Class with most videos   : 'drink' (92)
✔ Class with fewest videos : 'world' (1)
```

### `video_info.py` — example output

```
Property              Value
──────────────────────────────────────────────
File                  data/raw/videos/00001.mp4
File size             2.31 MB
Codec (FourCC)        avc1
FPS                   29.97
Resolution            640 × 480 px
Total frames          75
Duration              2.502 s
─────────────────────────────────────────────
Gloss (label)         book
Split                 train
Signer ID             3
```

---

## 7. Known Issues & Caveats

| Issue | Detail |
|---|---|
| Missing videos | Some URLs are dead; ~15–20% of videos may be unavailable |
| FPS mismatch | Annotation `fps` ≠ actual video FPS in many cases |
| Variable resolution | No consistent resolution — normalisation required in Phase 3 |
| Short clips | Some clips are < 10 frames (noise / mis-annotations) |
| Duplicate signers | Some signers appear in both train and test sets |

---

*Last updated: Phase 2 — Dataset Exploration*
