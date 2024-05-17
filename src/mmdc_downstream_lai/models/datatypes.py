""" Type definitions for MMDC models """
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class DownStreamConfig:
    """Base class for configs"""

    mmdc_model: MMDCModel


@dataclass
class RegressionConfig:
    """Configuration for a Regression network"""

    reg_model: RegressionModel


@dataclass
class RegressionModel:
    """Configuration for regression MLP"""

    hidden_sizes: list[int]
    input_size: int
    output_size: int


@dataclass
class MMDCModel:
    """Class for loading pretrained MMDC model"""

    pretrained_path: str | Path | None
    model_name: str | Path | None


@dataclass
class OutputLAI:
    """LAI output"""

    lai_pred: torch.Tensor
    latent: torch.Tensor
    lai_gt: torch.Tensor | None = None


@dataclass
class LatentPred:
    """Predicted latent variables"""

    latent_s1_mu: torch.Tensor
    latent_s1_logvar: torch.Tensor
    latent_s2_mu: torch.Tensor
    latent_s2_logvar: torch.Tensor
    latent_experts_mu: torch.Tensor | None = None
    latent_experts_logvar: torch.Tensor | None = None
