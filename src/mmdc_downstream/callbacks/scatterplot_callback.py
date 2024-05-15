#!/usr/bin/env python3
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
""" Lightning image callbacks """
import logging
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from torchmetrics import MeanSquaredError

from mmdc_downstream.snap.lai_snap import denormalize

from ..models.lightning.lai_regression_csv import MMDCLAILitModule

# Configure logging
NUMERIC_LEVEL = getattr(logging, "INFO", None)
logging.basicConfig(
    level=NUMERIC_LEVEL, format="%(asctime)-15s %(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


@dataclass
class SampleInfo:
    """Information about the sample to be displayed"""

    batch_idx: int
    batch_size: int
    patch_margin: int
    current_epoch: int


class MMDCLAIScatterplotCallback(Callback):
    """
    Callback to inspect the LAI image predicted from Latent Experts
    """

    def __init__(
        self,
        save_dir: str,
    ):
        self.save_dir = save_dir
        self.loss_fn = MeanSquaredError(squared=False)  # root mean square error

    def lai_gt_pred_scatterplots(
        self,
        preds: torch.Tensor,
        gt: torch.Tensor,
        epoch: int,
        lai_min: torch.Tensor,
        lai_max: torch.Tensor,
    ) -> None:
        """Save the PNG image of the scatterplots of prediction vs gt"""
        idx = np.random.choice(len(preds), int(len(preds) * 0.05), replace=False)
        preds = denormalize(preds, lai_min, lai_max)
        y_val_denorm = denormalize(gt, lai_min, lai_max)
        print(y_val_denorm)

        logging.info("pred max " + str(preds.max()))
        logging.info("pred min " + str(preds.min()))

        plt.close()
        plt.scatter(preds[idx].numpy(), y_val_denorm[idx].numpy(), s=0.1)
        plt.plot(y_val_denorm[idx].numpy(), y_val_denorm[idx].numpy())
        plt.title("loss=" + str(np.round(self.loss_fn(preds, y_val_denorm).item(), 4)))
        plt.xlabel("pred")
        plt.ylabel("gt")

        plt.savefig(self.save_dir + f"/mlp_{epoch}.png")

    def on_validation_epoch_end(
        self,
        trainer: pl.trainer.Trainer,
        pl_module: MMDCLAILitModule,
    ) -> None:
        """Plot scatterplot"""

        self.lai_gt_pred_scatterplots(
            torch.Tensor(pl_module.pred_val),
            torch.Tensor(pl_module.gt_val),
            trainer.current_epoch,
            pl_module.model_snap.variable_min.cpu(),
            pl_module.model_snap.variable_max.cpu(),
        )
