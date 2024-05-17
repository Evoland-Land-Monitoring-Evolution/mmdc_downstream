#!/usr/bin/env python3
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import os

import pytest
import torch

from mmdc_downstream_lai.mmdc_model.model import PretrainedMMDC
from mmdc_downstream_lai.models.datatypes import RegressionConfig, RegressionModel
from mmdc_downstream_lai.models.lightning.lai_regression import (
    MMDCBatch,
    MMDCDownstreamRegressionLitModule,
)
from mmdc_downstream_lai.models.torch.lai_regression import (
    MMDCDownstreamRegressionModule,
)
from mmdc_downstream_lai.snap.lai_snap import BVNET

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results_path = "/work/scratch/data/kalinie/MMDC/results"

model_mmdc_baseline = PretrainedMMDC(
    pretrained_path=os.path.join(
        results_path, "latent/checkpoints/mmdc_full/2024-02-05_09-45-27"
    ),
    model_name="last",
    model_type="baseline",
)

model_mmdc_experts = PretrainedMMDC(
    pretrained_path=os.path.join(
        results_path, "latent_old/checkpoints/mmdc_poe_mask/2024-01-10_16-19-03"
    ),
    model_name="epoch_079",
    model_type="experts",
)

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


@pytest.mark.parametrize(
    "model_mmdc, stats_path, input_data, reg_in_size",
    [
        (None, None, "S2", 11),
        (
            model_mmdc_baseline,
            os.path.join(model_mmdc_baseline.pretrained_path, "stats.pt"),
            "lat_S1",
            6,
        ),
        (
            model_mmdc_baseline,
            os.path.join(model_mmdc_baseline.pretrained_path, "stats.pt"),
            "lat_S2",
            6,
        ),
        (
            model_mmdc_experts,
            os.path.join(model_mmdc_experts.pretrained_path, "stats.pt"),
            "experts",
            6,
        ),
    ],
)
def test_init_model(
    model_mmdc: PretrainedMMDC | None,
    stats_path: str | None,
    input_data: str,
    reg_in_size: int,
):
    """Test regression model"""

    config = RegressionConfig(
        RegressionModel(hidden_sizes=[5], input_size=reg_in_size, output_size=1)
    )

    model = MMDCDownstreamRegressionModule(config=config)
    model_snap = BVNET(
        device="cuda" if torch.cuda.is_available() else "cpu", ver="2", variable="lai"
    )
    pl_module = MMDCDownstreamRegressionLitModule(
        model=model,
        model_snap=model_snap,
        model_mmdc=model_mmdc,
        stats_path=stats_path,
        input_data=input_data,
    )

    assert pl_module
    if model_mmdc is not None:
        assert pl_module.model_mmdc.lightning_module
        assert pl_module.model_mmdc.model_mmdc.scales is not None

    lai_gt = pl_module.compute_gt(batch)
    reg_input = pl_module.get_regression_input(batch)
    lai_pred = pl_module.forward(reg_input)

    assert lai_gt.size() == torch.Size([bs, 1, h, w])
    assert reg_input.shape[1] == reg_in_size
    assert lai_gt.shape == lai_pred.shape
