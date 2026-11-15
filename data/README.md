# Data

Nothing in this folder is committed except this file.

## UCF-Crime I3D features (`data/ucf_crime/`)

- **Source:** Hugging Face mirror [`jinmang2/ucf_crime_tencrop_i3d_seg32`](https://huggingface.co/datasets/jinmang2/ucf_crime_tencrop_i3d_seg32/tree/tushar-n), branch `tushar-n`. It's an unofficial community upload of the widely used I3D (RGB, ten-crop) features for UCF-Crime, the same benchmark as Sultani et al., CVPR 2018.
- **Files:**
  | File | Size | Contents |
  |---|---|---|
  | `train.zip` | 4.2 GB | training videos (video-level labels only: normal vs anomalous) |
  | `test.zip` | 5.7 GB | 290 test videos (140 anomalous, 150 normal) |
  | `ground_truth.json` | 5.6 MB | `{"<Video>_x264_i3d.npy": [0, 0, 1, 1, ...]}`: one 0/1 label **per frame** for every test video |
- **Layouts differ by split** (checked on load in `surveillance/data.py`):
  - train `.npy`: `(10 crops, 32 segments, 2048)`. Each video is already pooled into 32 temporal segments.
  - test `.npy`: `(T clips, 10 crops, 2048)`. There's one feature per 16-frame clip, and `len(ground truth) == 16 * T` for all 290 videos.
- **Cached form:** `python -m surveillance.data` averages the 10 crops and writes `cache_train.npz` (~400 MB) and `cache_test.npz` (~550 MB). The zips are read directly and never extracted.
- **Verified counts:** train 1610 (800 normal / 810 anomalous). Test 290 (150 / 140), 69,634 clips and 1,114,144 frames, of which **7.6% are anomalous**.
- **Labels:** training is *weakly supervised*. We only know whether a whole video contains an anomaly, not when. Frame-level labels exist only for the test set and are used only for evaluation.

Download (about 10 GB), then build the caches:

```bash
python -m surveillance.download
python -m surveillance.data
```

## Sample video (`data/samples/vtest.avi`)

Used to demo and benchmark the detection + tracking module.

- **Source:** [`opencv/opencv` `samples/data/vtest.avi`](https://github.com/opencv/opencv/blob/4.x/samples/data/vtest.avi) (Apache-2.0).
- **Contents:** 8.1 MB, 768×576, 10 fps, 795 frames (about 80 s). A fixed outdoor camera with pedestrians.

```bash
mkdir -p data/samples && curl -L -o data/samples/vtest.avi https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi
```

## Original capstone data

The 2025 capstone used a privately shared zip of the same UCF-Crime features (two-stream: RGB + optical flow). That zip no longer exists, which is why this rebuild uses the public RGB-only mirror above.
