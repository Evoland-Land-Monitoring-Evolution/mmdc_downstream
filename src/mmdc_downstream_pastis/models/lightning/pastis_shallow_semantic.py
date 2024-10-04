#!/usr/bin/env python3
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
""" Lightning module for pastis classification with alise """

import logging

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
        lr_type: str = "constant",
        resume_from_checkpoint: str | None = None,
    ):
        super().__init__(
            model, metrics_list, losses_list, lr, lr_type, resume_from_checkpoint
        )

        self.losses_list = losses_list
        self.metrics_list = metrics_list

        self.model = model

    def forward(self, batch: BatchInputUTAE) -> torch.Tensor:
        """Forward step"""

        return self.model.forward(batch.sits)
