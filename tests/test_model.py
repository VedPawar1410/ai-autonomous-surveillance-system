import torch

from surveillance.model import AnomalyScorer, mil_ranking_loss


def test_scorer_shape_and_range():
    scores = AnomalyScorer().eval()(torch.randn(4, 32, 2048))
    assert scores.shape == (4, 32)
    assert ((scores >= 0) & (scores <= 1)).all()


def test_ranking_loss_zero_when_anomaly_clearly_outranks_normal():
    anom = torch.zeros(2, 32)
    anom[:, 5] = 1.0                      # one clearly anomalous segment
    norm = torch.zeros(2, 32)
    loss = mil_ranking_loss(anom, norm, lambda_smooth=0, lambda_sparse=0)
    assert loss.item() == 0.0


def test_ranking_loss_penalises_wrong_order():
    anom, norm = torch.zeros(2, 32), torch.ones(2, 32)
    assert mil_ranking_loss(anom, norm, lambda_smooth=0, lambda_sparse=0).item() == 2.0


def test_topk_averages_highest_segments():
    anom = torch.zeros(1, 32)
    anom[0, :2] = torch.tensor([1.0, 0.5])
    norm = torch.zeros(1, 32)
    l1 = mil_ranking_loss(anom, norm, k=1, lambda_smooth=0, lambda_sparse=0)   # max = 1.0   -> 0
    l2 = mil_ranking_loss(anom, norm, k=2, lambda_smooth=0, lambda_sparse=0)   # mean = 0.75 -> 0.25
    assert l1.item() == 0.0 and abs(l2.item() - 0.25) < 1e-6


def test_smoothness_penalises_jagged_scores():
    # Both bags have max 1.0 so the ranking term is 0 for both; only smoothness differs.
    flat = torch.ones(1, 32)
    jagged = torch.tensor([[0.0, 1.0] * 16])
    norm = torch.zeros(1, 32)
    assert mil_ranking_loss(flat, norm, lambda_sparse=0).item() == 0.0
    assert mil_ranking_loss(jagged, norm, lambda_sparse=0).item() > 0.0
