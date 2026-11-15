"""Train the MIL anomaly scorer on UCF-Crime features.

    python -m surveillance.train --k 1 --seed 0

Model selection uses a held-out 10% of TRAIN videos with video-level labels only
(score of a video = its max segment score). Test frame labels are never used here.
"""
import argparse
import json

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from surveillance import config, data
from surveillance.model import AnomalyScorer, mil_ranking_loss

BATCH = 30  # bags per class per step (30 anomalous + 30 normal)


def video_scores(model, X) -> np.ndarray:
    """Max segment score per video, X: (N, 32, 2048)."""
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X)).max(dim=1).values.numpy()


def train_one(k: int = 1, seed: int = config.SEED, epochs: int = 100, lr: float = 0.01,
              weight_decay: float = 1e-3, save: bool = True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    X, y, _ = data.load_train()
    tr_idx, va_idx = data.train_val_split(y)          # fixed split (config.SEED), same for every run
    anom_idx, norm_idx = tr_idx[y[tr_idx] == 1], tr_idx[y[tr_idx] == 0]
    X_val, y_val = X[va_idx], y[va_idx]

    model = AnomalyScorer()
    # Adagrad + weight decay, as in the paper.
    opt = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"loss": [], "val_auc": []}
    best_auc, best_state, best_epoch = -1.0, None, -1
    for epoch in range(epochs):
        model.train()
        a, n = rng.permutation(anom_idx), rng.permutation(norm_idx)
        n_steps = min(len(a), len(n)) // BATCH
        losses = []
        for s in range(n_steps):
            xa = torch.from_numpy(X[a[s * BATCH:(s + 1) * BATCH]])
            xn = torch.from_numpy(X[n[s * BATCH:(s + 1) * BATCH]])
            loss = mil_ranking_loss(model(xa), model(xn), k=k)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        val_auc = roc_auc_score(y_val, video_scores(model, X_val))
        history["loss"].append(float(np.mean(losses)))
        history["val_auc"].append(float(val_auc))
        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            best_state = {kk: v.clone() for kk, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    history.update({"best_epoch": best_epoch, "best_val_auc": float(best_auc), "k": k, "seed": seed,
                    "epochs": epochs, "lr": lr, "weight_decay": weight_decay})
    if save:
        config.MODELS_DIR.mkdir(exist_ok=True)
        torch.save(best_state, config.MODELS_DIR / f"mil_k{k}_seed{seed}.pt")
        (config.MODELS_DIR / f"mil_k{k}_seed{seed}.json").write_text(json.dumps(history, indent=1))
    return model, history


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=1, help="top-k segments in the ranking loss (1 = original paper)")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--epochs", type=int, default=100)
    args = p.parse_args()
    _, h = train_one(args.k, args.seed, args.epochs)
    print(f"k={args.k} seed={args.seed}: best val video-AUC {h['best_val_auc']:.4f} at epoch {h['best_epoch']}")
