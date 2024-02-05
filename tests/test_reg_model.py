#!/usr/bin/env python3
# Copyright: (c) 2022 CESBIO / Centre National d'Etudes Spatiales

import torch
import pytest
from mmdc_downstream.models.torch.lai_regression import MMDCDownstreamRegressionModule

from mmdc_downstream.models.datatypes import RegressionConfig, RegressionModel


input_size = 6

config1 = RegressionConfig(RegressionModel(hidden_sizes=[5],
                                           input_size=input_size,
                                           output_size=1))

config2 = RegressionConfig(RegressionModel(hidden_sizes=[8, 16, 32],
                                           input_size=input_size,
                                           output_size=1))

input_data = torch.rand((5, 5, input_size))

device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("config", [config1, config2])
def test_build_model(config: RegressionConfig):
    """ Test regression model"""

    module = MMDCDownstreamRegressionModule(config=config)
    reg_model = module.build_regression_model()

    assert reg_model
    assert reg_model.forward(input_data.to(device)).size() == torch.Size([5, 5, 1])
    assert len([module for module in reg_model.modules()
                if isinstance(module, torch.nn.Linear)]) == \
           len(config.reg_model.hidden_sizes)+1
