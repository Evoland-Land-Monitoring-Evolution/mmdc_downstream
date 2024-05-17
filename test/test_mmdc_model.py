#!/usr/bin/env python3
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
"""Pretrained MMDC model module"""
import os

from mmdc_downstream_pastis.mmdc_model.model import PretrainedMMDCPastis


def test_mmdc() -> None:
    results_path = "/work/scratch/data/kalinie/MMDC/results"

    model_mmdc_baseline = PretrainedMMDCPastis(
        pretrained_path=os.path.join(
            results_path, "latent/checkpoints/mmdc_full/2024-02-05_09-45-27"
        ),
        model_name="last",
        model_type="baseline",
    )
    assert model_mmdc_baseline.model_mmdc is not None
    assert model_mmdc_baseline.model_mmdc.scales is not None
