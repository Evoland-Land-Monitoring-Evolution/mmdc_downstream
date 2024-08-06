#!/usr/bin/env python3
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
""" Lightning module for lai regression prediction """

import logging
from typing import Any

import torch

from mmdc_downstream_lai.mmdc_model.model import PretrainedMMDC

from ...snap.lai_snap import BVNET
from ..torch.lai_regression_conv import MMDCDownstreamRegressionModuleConv
from .lai_regression import MMDCDownstreamRegressionLitModule

logging.getLogger().setLevel(logging.INFO)


class MMDCDLAILitModuleConv(MMDCDownstreamRegressionLitModule):
    """
    LAI regression lightning module.
    Attributes:
        model: regression model
        model_snap: Fixed weights SNAP model for LAI GT computation
        model_mmdc: Pretrained MMDC model for latent representations
    """

    def __init__(
        self,
        model: MMDCDownstreamRegressionModuleConv,
        model_snap: BVNET,
        metrics_list: list[str] = ("RSE", "R2", "MAE", "MAPE"),
        losses_list: list[str] = ("MSE"),
        model_mmdc: PretrainedMMDC | None = None,
        input_data: str = "experts",
        lr: float = 0.001,
        resume_from_checkpoint: str | None = None,
        histogram_path: str | None = None,
        stats_path: str = None,
    ):
        super().__init__(
            model,
            model_snap,
            metrics_list,
            losses_list,
            model_mmdc,
            input_data,
            lr,
            resume_from_checkpoint,
            histogram_path,
            stats_path,
        )
        self.margin = (
            self.model_mmdc.model_mmdc.nb_cropped_hw
            if self.model_mmdc is not None
            else 0
        ) + 6

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward step"""
        return self.model.forward(data)

    def configure_optimizers(self) -> dict[str, Any]:
        """A single optimizer with a LR scheduler"""
        optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=self.learning_rate, weight_decay=0.01
        )

        training_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.9, patience=2, threshold=0.01
        )

        scheduler = {
            "scheduler": training_scheduler,
            "interval": "epoch",
            "monitor": f"val/{self.losses_list[0]}",
            "frequency": 1,
            "strict": False,
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
