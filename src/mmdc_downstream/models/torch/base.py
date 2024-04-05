#!/usr/bin/env python3
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
""" Base classes for the MMDC Downstream Lightning module and Pytorch model"""
import logging
from abc import abstractmethod
from typing import Any

import torch
from torch import nn

# Configure logging
NUMERIC_LEVEL = getattr(logging, "INFO", None)
logging.basicConfig(
    level=NUMERIC_LEVEL, format="%(asctime)-15s %(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


class MMDCDownstreamBaseModule(nn.Module):
    """
    Base MMDC Single Date Module
    """

    def __init__(self, config: Any) -> None:
        """
        Constructor
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config

    @abstractmethod
    def forward(self, data: Any) -> Any:
        """Forward pass of the model"""

    @abstractmethod
    def predict(self, data: Any) -> Any:
        """Prediction function"""
