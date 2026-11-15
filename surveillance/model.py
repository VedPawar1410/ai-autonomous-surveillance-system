"""Segment anomaly scorer + MIL ranking loss (Sultani, Chen & Shah, CVPR 2018), written from the paper.

Weak supervision: we only know whether a whole video (a "bag" of 32 segments) contains an anomaly.
The ranking loss says: the most anomalous-looking segment of an anomalous video should score higher
than the most anomalous-looking segment of a normal video, by a margin of 1.
"""
import torch
import torch.nn as nn

from surveillance import config


class AnomalyScorer(nn.Module):
    """MLP: one I3D feature (2048) -> anomaly score in [0, 1]. Applied independently to every segment/clip."""

    def __init__(self, in_dim: int = config.FEATURE_DIM, dropout: float = 0.6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1), nn.Sigmoid(),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def mil_ranking_loss(anom: torch.Tensor, norm: torch.Tensor, k: int = 1,
                     lambda_smooth: float = 8e-5, lambda_sparse: float = 8e-5) -> torch.Tensor:
    """anom, norm: (B, 32) segment scores for B anomalous and B normal videos.

    k = 1 is the original paper (max segment). k > 1 averages the top-k segments, which is less
    sensitive to a single noisy segment (the idea behind RTFM-style top-k MIL).
    """
    top_anom = anom.topk(k, dim=1).values.mean(dim=1)
    top_norm = norm.topk(k, dim=1).values.mean(dim=1)
    ranking = torch.relu(1.0 - top_anom + top_norm)
    # Regularisers only on the anomalous bag (the paper's formulation):
    smooth = ((anom[:, 1:] - anom[:, :-1]) ** 2).sum(dim=1)   # anomalies are temporally contiguous
    sparse = anom.sum(dim=1)                                   # anomalies are rare inside a video
    return (ranking + lambda_smooth * smooth + lambda_sparse * sparse).mean()
