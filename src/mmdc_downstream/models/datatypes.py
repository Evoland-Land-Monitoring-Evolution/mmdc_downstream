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
    pretrained_path: str | Path | None
    model_name: str | Path | None


# @dataclass
# class MMDCRegBatch:
#     # pylint: disable=too-many-instance-attributes
#     """Lightning module batch"""
#
#     s2_x: torch.Tensor
#     s2_m: torch.Tensor
#     s2_a: torch.Tensor
#     s1_x: torch.Tensor
#     s1_vm: torch.Tensor
#     s1_a: torch.Tensor
#     meteo_x: torch.Tensor
#     dem_x: torch.Tensor
#     lai_y: torch.Tensor


@dataclass
class OutputLAI:
    lai_pred: torch.Tensor
    latent: torch.Tensor
    lai_gt: torch.Tensor | None = None


@dataclass
class LatentPred:
    latent_S1_mu: torch.Tensor
    latent_S1_logvar: torch.Tensor
    latent_S2_mu: torch.Tensor
    latent_S2_logvar: torch.Tensor
    latent_experts_mu: torch.Tensor | None = None
    latent_experts_logvar: torch.Tensor | None = None
