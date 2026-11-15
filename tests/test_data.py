import numpy as np
import pytest

from surveillance import config, data


def test_video_label_from_filename():
    assert data.video_label("Normal_Videos308_x264_i3d.npy") == 0
    assert data.video_label("Robbery012_x264_i3d.npy") == 1


def test_train_val_split_is_stratified_disjoint_and_deterministic():
    y = np.array([0] * 800 + [1] * 810)
    tr, va = data.train_val_split(y)
    assert len(set(tr) & set(va)) == 0 and len(tr) + len(va) == len(y)
    assert abs(y[va].mean() - y.mean()) < 0.01           # same class balance as the full set
    tr2, va2 = data.train_val_split(y)
    assert (va == va2).all()                              # same seed -> same split


def test_split_by_offsets():
    arr = np.arange(10)
    parts = data._split(arr, np.array([0, 3, 3, 10]))
    assert [p.tolist() for p in parts] == [[0, 1, 2], [], list(range(3, 10))]


@pytest.mark.skipif(not (config.UCF_DIR / "cache_test.npz").exists(), reason="dataset not downloaded")
def test_real_test_split_alignment():
    clips, y, names, gts = data.load_test()
    assert len(clips) == len(gts) == 290
    for c, g in zip(clips, gts):
        assert len(g) == config.FRAMES_PER_CLIP * len(c)
