#!/usr/bin/env python3
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import torch

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


def test_losses():
    """Test losses"""


def test_metrcis():
    """Test metrics"""
