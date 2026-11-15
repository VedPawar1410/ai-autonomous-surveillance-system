"""Load UCF-Crime I3D features as compact arrays.

Raw layout: one .npy per video inside train.zip / test.zip. The two splits are stored differently:
  train: (10 crops, 32 segments, 2048)  -> video pre-pooled into 32 segments (the MIL "bag")
  test:  (T clips, 10 crops, 2048)      -> one feature per 16-frame clip, so we can score every frame
We average the 10 crops and cache each split as one .npz (train: 4.2 GB zip -> ~420 MB cache).

    python -m surveillance.data     # builds the caches and prints a summary
"""
import io
import json
import zipfile

import numpy as np
from sklearn.model_selection import train_test_split

from surveillance import config


def video_label(name: str) -> int:
    """Weak (video-level) label from the filename: 0 = normal, 1 = contains an anomaly somewhere."""
    return 0 if name.startswith("Normal") else 1


def _read_zip(zip_path, split):
    """Yield (name, crop-averaged features) for every .npy in a zip, in sorted order.
    train -> (32, 2048); test -> (T, 2048). Shapes are asserted, because averaging the wrong
    axis would silently produce garbage features."""
    with zipfile.ZipFile(zip_path) as z:
        for name in sorted(n for n in z.namelist() if n.endswith(".npy")):
            feats = np.load(io.BytesIO(z.read(name)))
            if split == "train":
                assert feats.shape == (config.N_CROPS, config.N_SEGMENTS, config.FEATURE_DIM), (name, feats.shape)
                yield name, feats.mean(axis=0)
            else:
                assert feats.shape[1:] == (config.N_CROPS, config.FEATURE_DIM), (name, feats.shape)
                yield name, feats.mean(axis=1)


def _check_counts(y, expected, split):
    got = {"normal": int((y == 0).sum()), "anomalous": int((y == 1).sum())}
    assert got == expected, f"{split}: expected {expected}, got {got} - files missing or mislabelled?"


def build_cache(split: str):
    zip_path = config.TRAIN_ZIP if split == "train" else config.TEST_ZIP
    names, feats = zip(*_read_zip(zip_path, split))
    names = np.array(names)
    y = np.array([video_label(n) for n in names])
    _check_counts(y, config.N_TRAIN if split == "train" else config.N_TEST, split)

    if split == "train":
        arrays = {"X": np.stack(feats).astype(np.float32), "y": y, "names": names}
    else:
        gt_all = json.loads(config.GT_JSON.read_text())
        gts = [np.asarray(gt_all[n], dtype=np.int8) for n in names]  # KeyError here = a test video has no labels
        for n, label, f, gt in zip(names, y, feats, gts):
            assert len(gt) == config.FRAMES_PER_CLIP * len(f), f"{n}: {len(gt)} frames vs {len(f)} clips"
            # Sanity: anomalous videos must contain anomalous frames, normal videos none.
            assert (gt.sum() > 0) == bool(label), f"ground truth disagrees with filename for {n}"
        # Videos have different lengths, so store everything concatenated + offsets (no pickled objects).
        arrays = {
            "X": np.concatenate(feats).astype(np.float32),
            "clip_offsets": np.cumsum([0] + [len(f) for f in feats]),
            "y": y,
            "names": names,
            "gt": np.concatenate(gts),
            "gt_offsets": np.cumsum([0] + [len(g) for g in gts]),
        }

    np.savez(config.UCF_DIR / f"cache_{split}.npz", **arrays)


def _load(split: str):
    path = config.UCF_DIR / f"cache_{split}.npz"
    if not path.exists():
        build_cache(split)
    return np.load(path)


def load_train():
    """Returns X (N, 32, 2048), y (N,), names (N,)."""
    d = _load("train")
    return d["X"], d["y"], d["names"]


def _split(arr, offsets):
    return [arr[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]


def load_test():
    """Returns clips: list of (T_i, 2048) arrays, y (290,), names (290,), gts: list of per-frame 0/1 arrays
    (len(gts[i]) == 16 * T_i)."""
    d = _load("test")
    return _split(d["X"], d["clip_offsets"]), d["y"], d["names"], _split(d["gt"], d["gt_offsets"])


def train_val_split(y, seed: int = config.SEED):
    """Stratified video-level split of TRAIN into train/val indices.
    Val only has video-level labels, which is exactly the weak supervision we would have in real life."""
    idx = np.arange(len(y))
    return train_test_split(idx, test_size=config.VAL_FRACTION, stratify=y, random_state=seed)


if __name__ == "__main__":
    X, y, names = load_train()
    print(f"train: {X.shape}, normal={int((y == 0).sum())}, anomalous={int((y == 1).sum())}")
    clips, yt, nt, gts = load_test()
    n_frames = sum(len(g) for g in gts)
    n_anom = sum(int(g.sum()) for g in gts)
    print(f"test:  {len(clips)} videos, normal={int((yt == 0).sum())}, anomalous={int((yt == 1).sum())}, "
          f"clips={sum(len(c) for c in clips)}, frames={n_frames}, "
          f"anomalous frames={n_anom} ({n_anom / n_frames:.1%})")
