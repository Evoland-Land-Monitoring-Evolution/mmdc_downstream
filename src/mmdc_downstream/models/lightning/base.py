# Copyright: (c) 2022 CESBIO / Centre National d'Etudes Spatiales
""" Base classes for the MMDC Lightning module and Pytorch model"""
import logging
from abc import abstractmethod
from typing import Any

import torch
from mmdc_singledate.datamodules.datatypes import (
    MMDCDataStats,
    MMDCShiftScales,
    ShiftScale,
)
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric
from torchutils import metrics

from mmdc_downstream.mmdc_model.model import PretrainedMMDC

from ..torch.base import MMDCDownstreamBaseModule

# Configure logging
NUMERIC_LEVEL = getattr(logging, "INFO", None)
logging.basicConfig(
    level=NUMERIC_LEVEL, format="%(asctime)-15s %(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


class MMDCDownstreamBaseLitModule(
    LightningModule
):  # pylint: disable=too-many-ancestors
    """
    Base Lightning Module for the MMDC Single Date networks
    """

    def __init__(
        self,
        model: MMDCDownstreamBaseModule,
        model_mmdc: PretrainedMMDC | None = None,
        lr: float = 0.001,
        resume_from_checkpoint: str | None = None,
        stats_path: str = None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.model = model
        self.resume_from_checkpoint = resume_from_checkpoint

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.test_rmse = metrics.rmse

        # for logging best so far validation rmse
        self.val_rmse_best = MinMetric()
        self.learning_rate = lr

        self.model_mmdc: PretrainedMMDC = model_mmdc

        if stats_path is not None:
            self.stats = self.get_stats(torch.load(stats_path))

    def get_stats(self, stats: MMDCDataStats) -> MMDCShiftScales:
        """Set shift and scale for model"""
        scale_regul = torch.nn.Threshold(1e-10, 1.0)
        shift_scale_s2 = ShiftScale(
            stats.sen2.median,
            scale_regul((stats.sen2.qmax - stats.sen2.qmin) / 2.0),
        )
        shift_scale_s1 = ShiftScale(
            stats.sen1.median,
            scale_regul((stats.sen1.qmax - stats.sen1.qmin) / 2.0),
        )
        shift_scale_meteo = ShiftScale(
            stats.meteo.concat_stats().median,
            scale_regul(
                (stats.meteo.concat_stats().qmax - stats.meteo.concat_stats().qmin)
                / 2.0
            ),
        )
        shift_scale_dem = ShiftScale(
            stats.dem.median,
            scale_regul((stats.dem.qmax - stats.dem.qmin) / 2.0),
        )

        return MMDCShiftScales(
            shift_scale_s2,
            shift_scale_s1,
            shift_scale_meteo,
            shift_scale_dem,
        )

    @abstractmethod
    def step(self, batch: Any) -> Any:
        """Perform one optimization step"""

    @abstractmethod
    def forward(self, data: Any) -> Any:  # pylint: disable=arguments-differ
        """Generic forward pass of the model. Just delegate to the Pytorch model."""

    def training_step(  # pylint: disable=arguments-differ
        self,
        batch: Any,
        batch_idx: int,  # pylint: disable=unused-argument
    ) -> dict[str, Any]:
        """Training step. Step and return loss."""
        torch.autograd.set_detect_anomaly(True)
        loss = self.step(batch)

        # log training metrics
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        return {"loss": loss}

    def validation_step(  # pylint: disable=arguments-differ
        self,
        batch: Any,
        batch_idx: int,  # pylint: disable=unused-argument
    ) -> dict[str, Any]:
        """Validation step. Step and return loss."""
        loss = self.step(batch)

        # log val metrics
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        return {"loss": loss}

    def test_step(  # pylint: disable=arguments-differ
        self,
        batch: Any,
        batch_idx: int,  # pylint: disable=unused-argument
    ) -> dict[str, Any]:
        """Test step. Step and return loss."""
        loss = self.step(batch)

        self.log(
            "test/loss",
            loss,
        )

        return {"loss": loss}

    def on_train_epoch_end(self) -> None:
        logger.info("Ended traning epoch %s", self.trainer.current_epoch)

    def on_validation_epoch_end(self) -> None:
        logger.info("Ended validation epoch %s", self.trainer.current_epoch)

    def on_test_epoch_end(self) -> None:
        """Callback after a test epoch"""

    def on_fit_start(self) -> None:
        """
        On fit start, load pretrained MMDC model if its path is set.
        The scales are already set in the loaded model
        """
        logger.info("On fit start")

    def configure_optimizers(self) -> dict[str, Any]:
        """A single optimizer with a LR scheduler"""
        optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=self.learning_rate
        )

        # training_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer, T_0=4, T_mult=2, eta_min=0, last_epoch=-1
        # )
        # scheduler = {
        #     "scheduler": training_scheduler,
        #     "interval": "epoch",
        #     "monitor": "val/loss",
        #     "frequency": 1,
        # }
        return {
            "optimizer": optimizer,
            # "lr_scheduler": scheduler,
        }
