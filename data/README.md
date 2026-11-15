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
- **Labels:** training is *weakly supervised*. We only know whether a whole video contains an anomaly, not when. Frame-level labels exist only for the test set and are used only for evaluation.

Download (about 10 GB):

```bash
python -m surveillance.download
```

## Original capstone data

The 2025 capstone used a privately shared zip of the same UCF-Crime features (two-stream: RGB + optical flow). That zip no longer exists, which is why this rebuild uses the public RGB-only mirror above.
