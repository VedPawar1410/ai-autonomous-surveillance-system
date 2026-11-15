import numpy as np

from surveillance import evaluate


def test_frame_scores_repeat_each_clip_16_times():
    (f,) = evaluate.frame_scores([np.array([0.1, 0.9])])
    assert f.tolist() == [0.1] * 16 + [0.9] * 16


def test_pooled_auc_perfect_and_inverted():
    gts = [np.array([0, 0, 1, 1]), np.array([0, 0, 0, 0])]
    good = [np.array([0.1, 0.2, 0.8, 0.9]), np.array([0.1, 0.1, 0.2, 0.3])]
    assert evaluate.pooled_auc(good, gts) == 1.0
    assert evaluate.pooled_auc([1 - s for s in good], gts) == 0.0


def test_false_alarm_rate_counts_only_normal_videos():
    scores = [np.array([0.9, 0.9]), np.array([0.6, 0.1, 0.1, 0.1])]
    gts = [np.array([1, 1]), np.zeros(4, dtype=int)]
    assert evaluate.false_alarm_rate(scores, gts, y=np.array([1, 0])) == 0.25


def test_pairwise_2025_metric_hides_cross_video_errors():
    # Inside each anomalous+normal pair the anomaly is ranked first -> every per-pair AUC is 1.0.
    # But video 2's NORMAL frame (0.7) outscores video 1's ANOMALOUS frame (0.2): a real false alarm
    # that one global threshold would hit. Pooled AUC sees it (11/12); per-pair averaging does not.
    gts = [np.array([0, 1]), np.array([0, 1]), np.zeros(2, dtype=int), np.zeros(2, dtype=int)]
    scores = [np.array([0.1, 0.2]), np.array([0.7, 0.8]), np.zeros(2), np.zeros(2)]
    y = np.array([1, 1, 0, 0])
    assert evaluate.pairwise_auc_2025(scores, gts, y, seed=0) == 1.0
    assert abs(evaluate.pooled_auc(scores, gts) - 11 / 12) < 1e-9
