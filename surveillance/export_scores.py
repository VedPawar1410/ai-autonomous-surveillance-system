"""Export the final model's per-clip anomaly scores for all 290 UCF-Crime test videos to a small JSON file,
so the demo can show score timelines without the 550 MB feature cache.

    python -m surveillance.export_scores        # -> reports/test_scores.json
"""
import json

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from surveillance import config, data
from surveillance.evaluate import clip_scores, frame_scores
from surveillance.model import AnomalyScorer


def main():
    model = AnomalyScorer()
    model.load_state_dict(torch.load(config.MODELS_DIR / "final.pt"))
    clips, y, names, gts = data.load_test()
    scores = clip_scores(model, clips)
    fs = frame_scores(scores)

    videos = []
    for name, label, s, f, gt in zip(names, y, scores, fs, gts):
        # Ground truth per clip: a clip counts as anomalous if any of its 16 frames is.
        clip_gt = gt.reshape(-1, config.FRAMES_PER_CLIP).max(axis=1)
        videos.append({
            "name": str(name).replace("_x264_i3d.npy", ""),
            "label": int(label),
            "clip_scores": [round(float(v), 3) for v in s],
            "clip_gt": clip_gt.astype(int).tolist(),
            "frame_auc": round(float(roc_auc_score(gt, f)), 4) if label == 1 else None,
            "max_score": round(float(np.max(s)), 3),
        })
    out = config.REPORTS_DIR / "test_scores.json"
    out.write_text(json.dumps({"model": "models/final.pt", "frames_per_clip": config.FRAMES_PER_CLIP,
                               "videos": videos}, separators=(",", ":")))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, {len(videos)} videos)")


if __name__ == "__main__":
    main()
