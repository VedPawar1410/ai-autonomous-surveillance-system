"""Aggregate experiment runs -> reports/metrics.json + charts, and pick the final model.

    python -m surveillance.report

The final model is chosen on VALIDATION only (best mean val AUC across k, then best-val seed).
Test numbers are reported, never used for selection.
"""
import json
import shutil
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from surveillance import config, data
from surveillance.evaluate import clip_scores, frame_scores
from surveillance.model import AnomalyScorer

# Reference palette (dataviz skill), light mode
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GT_FILL = "#2a78d6", "#eb6834", "#d9d8d3"
METRICS = ["frame_auc", "frame_auc_anomalous_videos_only", "false_alarm_rate@0.5", "pairwise_auc_2025_metric_mean"]


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.tick_params(colors=INK_2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _fig(w=7, h=4):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def load_runs():
    runs = defaultdict(list)
    for path in sorted((config.REPORTS_DIR / "runs").glob("k*_seed*.json")):
        r = json.loads(path.read_text())
        runs[r["k"]].append(r)
    return dict(sorted(runs.items()))


def aggregate(runs):
    out = {}
    for k, rs in runs.items():
        entry = {"n_seeds": len(rs), "seeds": [r["seed"] for r in rs]}
        vals = [r["best_val_auc"] for r in rs]
        entry["val_video_auc"] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "per_seed": vals}
        for m in METRICS:
            v = [r["test"][m] for r in rs]
            entry[m] = {"mean": float(np.mean(v)), "std": float(np.std(v)), "per_seed": v}
        out[f"k{k}"] = entry
    return out


def pick_final(runs, agg):
    best_k = max(runs, key=lambda k: agg[f"k{k}"]["val_video_auc"]["mean"])
    best_run = max(runs[best_k], key=lambda r: r["best_val_auc"])
    src = config.MODELS_DIR / f"mil_k{best_k}_seed{best_run['seed']}.pt"
    shutil.copy(src, config.MODELS_DIR / "final.pt")
    return {"k": best_k, "seed": best_run["seed"], "selected_by": "validation video-AUC",
            "val_video_auc": best_run["best_val_auc"], "test": best_run["test"]}


def chart_ablation(runs):
    fig, ax = _fig(6, 3.6)
    ks = list(runs)
    for i, k in enumerate(ks):
        v = [r["test"]["frame_auc"] for r in runs[k]]
        ax.scatter([i] * len(v), v, s=40, color=BLUE, alpha=0.6, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        ax.errorbar(i + 0.15, np.mean(v), yerr=np.std(v), fmt="o", color=INK, markersize=6, capsize=4, linewidth=2)
        ax.annotate(f"{np.mean(v):.3f} ± {np.std(v):.3f}", (i + 0.22, np.mean(v)), color=INK, fontsize=9,
                    va="center")
    ax.set_xticks(range(len(ks)), [f"top-{k}" + (" (paper)" if k == 1 else "") for k in ks])
    ax.set_xlim(-0.5, len(ks) - 0.1)
    _style(ax, "Test frame-AUC per seed (black: mean ± std)", "", "frame-level ROC AUC")
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "ablation.png")
    plt.close(fig)


def chart_training(runs):
    fig, ax = _fig()
    for k, color in zip(runs, [BLUE, ORANGE]):
        curves = np.array([r["val_auc"] for r in runs[k]])
        mean = curves.mean(axis=0)
        ax.plot(mean, color=color, linewidth=2, label=f"top-{k}")
        ax.fill_between(range(len(mean)), curves.min(0), curves.max(0), color=color, alpha=0.12, linewidth=0)
    ax.legend(frameon=False, labelcolor=INK, loc="lower right")
    _style(ax, "Validation video-AUC per epoch (band: min–max over 5 seeds)", "epoch", "val video AUC")
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "training_curves.png")
    plt.close(fig)


def chart_roc_and_timelines(final):
    model = AnomalyScorer()
    model.load_state_dict(torch.load(config.MODELS_DIR / "final.pt"))
    clips, y, names, gts = data.load_test()
    fs = frame_scores(clip_scores(model, clips))

    fpr, tpr, _ = roc_curve(np.concatenate(gts), np.concatenate(fs))
    fig, ax = _fig(4.6, 4.2)
    ax.plot(fpr, tpr, color=BLUE, linewidth=2)
    ax.plot([0, 1], [0, 1], color=INK_2, linewidth=1, linestyle="--")
    ax.annotate(f"AUC {final['test']['frame_auc']:.3f}", (0.55, 0.3), color=INK, fontsize=11)
    _style(ax, "Final model ROC (all test frames)", "false positive rate", "true positive rate")
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "roc.png")
    plt.close(fig)

    # Timelines picked by a fixed rule (not cherry-picked): anomalous videos at the 75th and 25th
    # percentile of per-video AUC, plus the normal video with the median peak score.
    anom = [i for i in range(len(y)) if y[i] == 1]
    per_video = sorted(anom, key=lambda i: roc_auc_score(gts[i], fs[i]))
    normal = sorted([i for i in range(len(y)) if y[i] == 0], key=lambda i: fs[i].max())
    picks = [per_video[int(0.75 * len(per_video))], per_video[int(0.25 * len(per_video))],
             normal[len(normal) // 2]]
    labels = ["anomalous video (75th pct.)", "anomalous video (25th pct.)", "normal video (median)"]

    fig, axes = plt.subplots(3, 1, figsize=(8, 6.6), dpi=150, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, i, lab in zip(axes, picks, labels):
        t = np.arange(len(fs[i]))
        ax.fill_between(t, 0, gts[i], color=GT_FILL, step="mid", linewidth=0, label="ground-truth anomaly")
        ax.plot(t, fs[i], color=BLUE, linewidth=1.6, label="anomaly score")
        ax.set_ylim(-0.04, 1.05)   # headroom so a flat-zero score line stays visible
        _style(ax, f"{names[i].replace('_x264_i3d.npy', '')} — {lab}", "", "score")
    axes[-1].set_xlabel("frame", color=INK_2)
    axes[0].legend(frameon=False, labelcolor=INK, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "timelines.png")
    plt.close(fig)


def main():
    runs = load_runs()
    agg = aggregate(runs)
    final = pick_final(runs, agg)
    (config.REPORTS_DIR / "metrics.json").write_text(json.dumps({"configs": agg, "final": final}, indent=1))
    chart_ablation(runs)
    chart_training(runs)
    chart_roc_and_timelines(final)
    for name, e in agg.items():
        print(f"{name}: test frame-AUC {e['frame_auc']['mean']:.4f} ± {e['frame_auc']['std']:.4f} | "
              f"anomalous-only {e['frame_auc_anomalous_videos_only']['mean']:.4f} | "
              f"FAR@0.5 {e['false_alarm_rate@0.5']['mean']:.4f} | "
              f"2025-style {e['pairwise_auc_2025_metric_mean']['mean']:.4f} | val {e['val_video_auc']['mean']:.4f}")
    print("final:", {k: v for k, v in final.items() if k != "test"}, "test frame-AUC", round(final["test"]["frame_auc"], 4))


if __name__ == "__main__":
    main()
