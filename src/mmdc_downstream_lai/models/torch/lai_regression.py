#!/usr/bin/env python3
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
""" Base classes for the MMDC Lightning module and Pytorch model"""
import logging
from collections import OrderedDict

import torch
from torch import nn

from ..datatypes import RegressionConfig
from .base import MMDCDownstreamBaseModule

# Configure logging
NUMERIC_LEVEL = getattr(logging, "INFO", None)
logging.basicConfig(
    level=NUMERIC_LEVEL, format="%(asctime)-15s %(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


class MMDCDownstreamRegressionModule(MMDCDownstreamBaseModule):
    """Downstream task regression module"""

    def __init__(self, config: RegressionConfig):
        """Constructor"""
        super().__init__(config)
        self.config: RegressionConfig = config
        self.regression_model = self.build_regression_model()  # pylint: disable=E1102

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.regression_model(data)

    # def build_regression_model(self) -> nn.Sequential:
    #     """Build regression model"""
    #     input_size = self.config.reg_model.input_size
    #     output_size = self.config.reg_model.output_size
    #     hidden_sizes = self.config.reg_model.hidden_sizes
    #     sizes = [input_size] + hidden_sizes
    #     reg_layers = []
    #     for i in range(len(sizes) - 1):
    #         reg_layers.append((f'layer_{i}',
    #                            nn.Conv2d(in_channels=sizes[i],
    #                                      out_channels=sizes[i + 1],
    #                                      kernel_size=1)))
    #         reg_layers.append(('tanh', nn.Tanh()))
    #     reg_layers.append((f'layer_{len(sizes)}',
    #                        nn.Conv2d(in_channels=sizes[-1],
    #                                  out_channels=output_size,
    #                                  kernel_size=1)))
    #
    #     return nn.Sequential(OrderedDict(reg_layers)).to(self.device)

    def build_regression_model(self) -> nn.Sequential:
        """Build regression model"""
        input_size = self.config.reg_model.input_size
        output_size = self.config.reg_model.output_size
        hidden_sizes = self.config.reg_model.hidden_sizes
        sizes = [input_size] + hidden_sizes
        reg_layers = []
        for i in range(len(sizes) - 1):
            reg_layers.extend(
                (
                    (
                        f"layer_{i}",
                        nn.Linear(
                            in_features=sizes[i], out_features=sizes[i + 1]
                        ).float(),
                    ),
                    ("tanh", nn.Tanh()),
                ),
            )
        reg_layers.append(
            (
                f"layer_{len(sizes) - 1}",
                nn.Linear(in_features=sizes[-1], out_features=output_size).float(),
            )
        )

        return nn.Sequential(OrderedDict(reg_layers)).to(self.device)
