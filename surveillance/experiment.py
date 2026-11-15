"""Run the experiment grid: for each seed, train (selecting the epoch on validation) then evaluate ONCE on test.

    python -m surveillance.experiment --k 1 --seeds 0 1 2 3 4
    python -m surveillance.experiment --k 3 --seeds 0 1 2 3 4

One JSON per run goes to reports/runs/; surveillance.report aggregates them.
"""
import argparse
import json

from surveillance import config
from surveillance.evaluate import evaluate
from surveillance.train import train_one


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=100)
    args = p.parse_args()

    out_dir = config.REPORTS_DIR / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        model, history = train_one(k=args.k, seed=seed, epochs=args.epochs)
        history["test"] = evaluate(model)
        (out_dir / f"k{args.k}_seed{seed}.json").write_text(json.dumps(history, indent=1))
        t = history["test"]
        print(f"k={args.k} seed={seed}: val video-AUC {history['best_val_auc']:.4f} | "
              f"test frame-AUC {t['frame_auc']:.4f} | 2025-style metric {t['pairwise_auc_2025_metric_mean']:.4f}",
              flush=True)


if __name__ == "__main__":
    main()
