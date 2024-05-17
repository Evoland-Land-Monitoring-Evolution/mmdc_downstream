#!/usr/bin/env python3
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
""" Lightning image callbacks """
import logging
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from torchmetrics import MeanSquaredError

from mmdc_downstream_lai.snap.lai_snap import denormalize

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


class MMDCLAIScatterplotCallbackEpoch(Callback):
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
        batch_idx: int | None,
    ) -> None:
        """Save the PNG image of the scatterplots of prediction vs gt"""
        preds, gt, rmse = self.prepare_data(
            preds,
            gt,
            lai_min,
            lai_max,
        )

        plt.close()
        plt.scatter(preds, gt, s=0.1)
        plt.plot(gt, gt)
        plt.title("loss=" + str(rmse))
        plt.xlabel("pred")
        plt.ylabel("gt")

        if batch_idx is not None:
            plt.savefig(self.save_dir + f"/mlp_{epoch}_{batch_idx}.png")
        else:
            plt.savefig(self.save_dir + f"/mlp_{epoch}.png")

    def prepare_data(
        self,
        preds: torch.Tensor,
        gt: torch.Tensor,
        lai_min: torch.Tensor,
        lai_max: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        idx = np.random.choice(len(preds), int(len(preds) * 0.05), replace=False)
        preds = denormalize(preds, lai_min, lai_max)
        gt = denormalize(gt, lai_min, lai_max)
        rmse = np.round(self.loss_fn(preds, gt).item(), 2)

        logging.info("pred max " + str(preds.max()))
        logging.info("pred min " + str(preds.min()))

        return preds[idx].numpy(), gt[idx].numpy(), rmse

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


class MMDCLAIScatterplotCallbackStep(MMDCLAIScatterplotCallbackEpoch):
    def __init__(
        self,
        save_dir: str,
    ):
        super().__init__(save_dir)

    def on_validation_batch_end(
        self,
        trainer: pl.trainer.Trainer,
        pl_module: MMDCLAILitModule,
        outputs: Any,
        batch: torch.Tensor,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        reg_input, lai_gt = batch
        lai_pred = pl_module.model.forward(reg_input)
        self.lai_gt_pred_scatterplots(
            lai_pred.reshape(-1),
            lai_gt.reshape(-1),
            trainer.current_epoch,
            pl_module.model_snap.variable_min.cpu(),
            pl_module.model_snap.variable_max.cpu(),
            batch_idx,
        )
