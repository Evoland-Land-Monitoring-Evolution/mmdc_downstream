#!/usr/bin/env python3
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
""" Lightning module for pastis classification with alise """

import logging
from typing import Any

import torch

from ...datamodule.datatypes import BatchInputUTAE
from ..torch.shallow import Shallow
from .pastis_utae_semantic import PastisUTAE

# Configure logging
NUMERIC_LEVEL = getattr(logging, "INFO", None)
logging.basicConfig(
    level=NUMERIC_LEVEL, format="%(asctime)-15s %(levelname)s: %(message)s"
)
logging.getLogger(__name__).setLevel(logging.INFO)


class PastisShallow(PastisUTAE):
    """
    Pastis Semantic segmentation module.
    """

    def __init__(
        self,
        model: Shallow,
        metrics_list: list[str] = ["cross"],
        losses_list: list[str] = ["cross"],
        lr: float = 0.001,
        lr_type: str = "plateau",
        resume_from_checkpoint: str | None = None,
        margin: int = 32,
    ):
        super().__init__(
            model,
            metrics_list,
            losses_list,
            lr,
            lr_type,
            resume_from_checkpoint,
            margin,
        )

        self.losses_list = losses_list
        self.metrics_list = metrics_list

        self.model = model

        for param in self.model.parameters():
            param.to(torch.float32)

    def forward(self, batch: BatchInputUTAE) -> torch.Tensor:
        """Forward step"""

        return self.model.forward(batch.sits)

    def configure_optimizers(self) -> dict[str, Any]:
        """A single optimizer with a LR scheduler"""
        optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=self.learning_rate  # , weight_decay=0.01
        )
        training_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            # optimizer, mode="min", factor=0.8, patience=2, threshold=0.05
            optimizer,
            mode="min",
            factor=0.75,
            patience=3,
            threshold=0.01,
            min_lr=0.000001,
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
