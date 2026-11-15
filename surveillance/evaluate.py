"""Frame-level evaluation on the UCF-Crime test set.

Standard protocol (Sultani 2018, RTFM, ...): score every frame of all 290 test videos, pool them,
and compute ONE ROC AUC. Each 16-frame clip's score is repeated over its 16 frames.

We also reproduce the metric the 2025 version of this project reported (average of per-pair AUCs
over randomly paired anomalous+normal videos) to show why it is not comparable.
"""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from surveillance import config, data


def clip_scores(model, clips):
    """clips: list of (T_i, 2048) arrays -> list of (T_i,) anomaly scores."""
    model.eval()
    with torch.no_grad():
        return [model(torch.from_numpy(c)).numpy() for c in clips]


def frame_scores(scores):
    """Repeat each clip score over its 16 frames (ground truth is aligned to clip boundaries)."""
    return [np.repeat(s, config.FRAMES_PER_CLIP) for s in scores]


def pooled_auc(f_scores, gts) -> float:
    """THE benchmark number: one ROC AUC over every frame of every test video."""
    return float(roc_auc_score(np.concatenate(gts), np.concatenate(f_scores)))


def false_alarm_rate(f_scores, gts, y, threshold: float = 0.5) -> float:
    """Fraction of frames in NORMAL videos scored above the threshold (what an operator would see as alarms)."""
    normal = np.concatenate([s for s, label in zip(f_scores, y) if label == 0])
    return float((normal > threshold).mean())


def pairwise_auc_2025(f_scores, gts, y, seed: int = 0) -> float:
    """The 2025 project's metric: shuffle normal videos, drop 10, pair each with an anomalous video,
    compute AUC per pair, average. Non-standard (and seed-dependent) - kept only for comparison."""
    rng = np.random.default_rng(seed)
    anom = [i for i, label in enumerate(y) if label == 1]
    norm = rng.permutation([i for i, label in enumerate(y) if label == 0])[:len(anom)]
    aucs = [roc_auc_score(np.concatenate([gts[a], gts[n]]), np.concatenate([f_scores[a], f_scores[n]]))
            for a, n in zip(anom, norm)]
    return float(np.mean(aucs))


def evaluate(model) -> dict:
    clips, y, _, gts = data.load_test()
    fs = frame_scores(clip_scores(model, clips))
    anom = [i for i, label in enumerate(y) if label == 1]
    pairwise = [pairwise_auc_2025(fs, gts, y, seed=s) for s in range(5)]
    return {
        "frame_auc": pooled_auc(fs, gts),
        "frame_auc_anomalous_videos_only": pooled_auc([fs[i] for i in anom], [gts[i] for i in anom]),
        "false_alarm_rate@0.5": false_alarm_rate(fs, gts, y),
        "pairwise_auc_2025_metric_mean": float(np.mean(pairwise)),
        "pairwise_auc_2025_metric_range": [float(min(pairwise)), float(max(pairwise))],
    }
