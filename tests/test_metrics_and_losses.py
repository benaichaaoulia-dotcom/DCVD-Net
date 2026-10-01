from __future__ import annotations

import numpy as np
import pytest
import torch

from dcvd_net.losses import bce_dice_loss
from dcvd_net.metrics import image_metrics, r4_axes


def test_binary_objective_has_finite_loss_and_gradient() -> None:
    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    target = (torch.rand(2, 1, 16, 16) > 0.93).float()

    loss = bce_dice_loss(torch.sigmoid(logits), target)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_per_image_metrics_use_the_eroded_thin_stratum_and_fixed_threshold() -> None:
    target = np.zeros((24, 24), dtype=np.float32)
    target[2:22, 2:22] = 1.0
    metrics = image_metrics(target, target, threshold=0.6)

    assert metrics["DSC"] == pytest.approx(1.0)
    assert metrics["sensitivity"] == pytest.approx(1.0)
    assert metrics["thin_sensitivity"] == pytest.approx(1.0)
    assert metrics["thin_pixels"] >= 64


def test_r4_primary_decision_does_not_require_secondary_axes_to_pass() -> None:
    summary = {
        "DRIVE": {"macro_sensitivity": 0.78, "thin_mean": 0.69, "thin_worst": 0.56,
                  "macro_dsc": 0.80, "sub_0_75_count": 0},
        "STARE": {"macro_sensitivity": 0.79, "thin_mean": 0.71, "thin_worst": 0.54,
              "macro_dsc": 0.82, "sub_0_75_count": 2},
        "CHASE_DB1": {"macro_sensitivity": 0.82, "thin_mean": 0.76, "thin_worst": 0.67,
                      "macro_dsc": 0.78, "sub_0_75_count": 0},
    }
    per_image = [{"DSC": 0.74}, {"DSC": 0.73}]

    axes = r4_axes(summary, per_image)

    assert axes["F1_pass"] is True
    assert axes["F3_pass"] is False
    assert axes["F4_pass"] is False
    assert axes["candidate_pass"] is True
