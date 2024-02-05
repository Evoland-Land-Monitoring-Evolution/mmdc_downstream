#!/usr/bin/env python3
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import torch

from mmdc_downstream.models.components.losses import compute_losses
from mmdc_downstream.models.components.metrics import compute_val_metrics
from mmdc_downstream.models.lightning.lai_regression import MMDCBatch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bs, h, w = 2, 64, 64

batch = MMDCBatch(
    s2_x=torch.rand((bs, 10, h, w)).to(device),
    s2_m=torch.randint(size=(bs, 1, h, w), high=1).to(device),
    s2_a=torch.rand((bs, 6, h, w)).to(device),
    s1_x=torch.rand((bs, 6, h, w)).to(device),
    s1_vm=torch.randint(size=(bs, 2, h, w), high=1).to(device),
    s1_a=torch.rand((bs, 2, h, w)).to(device),
    meteo_x=torch.rand((bs, 48, h, w)).to(device),
    dem_x=torch.rand((bs, 4, h, w)).to(device),
)

lai_gt = torch.zeros((bs, 1, h, w))
lai_pred = torch.zeros((bs, 1, h, w))
mask = torch.zeros((bs, 1, h, w))


mask[0, 0, :2, :2] = 1
lai_gt[0, 0, :2, :2] = torch.nan


def test_losses():
    """Test losses"""

    losses = compute_losses(
        preds=lai_pred,
        target=lai_gt,
        mask=mask,
        margin=0,
        losses_list=["MSE", "RMSE", "L1"],
    )

    for name, value in losses.items():
        assert value == 0


def test_metrcis():
    """Test metrics"""
    metrics = compute_val_metrics(
        preds=lai_pred,
        target=lai_gt,
        mask=mask,
        margin=0,
        metrics_list=["RSE", "R2", "MAE", "MAPE"],
    )

    for name, value in metrics.items():
        if name in ("RSE", "MAE", "MAPE"):
            assert value == 0
        else:
            assert value == 1
