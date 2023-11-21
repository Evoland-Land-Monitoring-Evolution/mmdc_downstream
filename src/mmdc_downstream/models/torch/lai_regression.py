#!/usr/bin/env python3
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
""" Base classes for the MMDC Lightning module and Pytorch model"""
import logging
from collections import OrderedDict

import torch
from torch import nn

from .base import MMDCDownstreamBaseModule
from ..datatypes import RegressionConfig

# Configure logging
NUMERIC_LEVEL = getattr(logging, "INFO", None)
logging.basicConfig(level=NUMERIC_LEVEL,
                    format="%(asctime)-15s %(levelname)s: %(message)s")

logger = logging.getLogger(__name__)


class MMDCDownstreamRegressionModule(MMDCDownstreamBaseModule):

    def __init__(self, config: RegressionConfig):
        """Constructor"""
        super().__init__(config)
        self.regression_model = self.build_regression_model()
        self.config: RegressionConfig = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.regression_model(x)

    def build_regression_model(self) -> nn.Sequential:

        input_size = self.config.reg_model.input_size
        output_size = self.config.reg_model.output_size
        hidden_sizes = self.config.reg_model.hidden_sizes
        sizes = [input_size] + hidden_sizes
        reg_layers = []
        for i in range(len(sizes) - 1):
            reg_layers.append((f'layer_{i}',
                               nn.Conv2d(in_channels=sizes[i],
                                         out_channels=sizes[i + 1],
                                         kernel_size=1)))
            reg_layers.append(('tanh', nn.Tanh()))
        reg_layers.append((f'layer_{len(sizes)}',
                           nn.Conv2d(in_channels=sizes[-1],
                                     out_channels=output_size,
                                     kernel_size=1)))

        return nn.Sequential(OrderedDict(reg_layers)).to(self.device)
