# Copyright: (c) 2022 CESBIO / Centre National d'Etudes Spatiales
""" Base classes for the MMDC Lightning module and Pytorch model"""
import logging
from abc import abstractmethod
from typing import Any

import torch
from pytorch_lightning import LightningModule
from torchutils import metrics

from ...utils.miou import IoU
from ...utils.weight_init import weight_init
from ..torch.utae import UTAE

# Configure logging
NUMERIC_LEVEL = getattr(logging, "INFO", None)
logging.basicConfig(
    level=NUMERIC_LEVEL, format="%(asctime)-15s %(levelname)s: %(message)s"
)
logging.getLogger(__name__).setLevel(logging.INFO)


class MMDCPastisBaseLitModule(LightningModule):  # pylint: disable=too-many-ancestors
    """
    Base Lightning Module for the MMDC Single Date networks
    """

    def __init__(
        self, model: UTAE, lr: float = 0.001, resume_from_checkpoint: str | None = None
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

        self.learning_rate = lr

        self.iou_meter = {
            stage: IoU(
                num_classes=model.num_classes,
                ignore_index=0,
                cm_device="cuda",
            )
            for stage in ("train", "val", "test")
        }
        logging.info(self.iou_meter)

    @abstractmethod
    def step(self, batch: Any) -> Any:
        """Perform one optimization step"""

    @abstractmethod
    def forward(self, data: Any, **kwargs) -> Any:  # pylint: disable=arguments-differ
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

    def predict(self, batch: Any, batch_positions: torch.Tensor) -> torch.Tensor:
        """Generic prediction of the model. Just delegate to the Pytorch model."""

    def on_train_epoch_start(self) -> None:
        """On train epoch start"""
        # Otherwise reset does not work
        meter = self.iou_meter["train"]
        meter.reset()
        self.iou_meter.update({"train": meter})

    def on_train_epoch_end(self) -> None:
        """On train epoch end"""
        logging.info("Ended traning epoch %s", self.trainer.current_epoch)
        miou, acc = self.iou_meter["train"].get_miou_acc()
        self.log(
            "train/mIoU",
            miou,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "train/accuracy",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

    def on_validation_epoch_start(self) -> None:
        """On validation epoch start"""
        # Otherwise reset does not work
        meter = self.iou_meter["val"]
        meter.reset()
        self.iou_meter.update({"val": meter})

    def on_validation_epoch_end(self) -> None:
        """On validation epoch end"""
        logging.info("Ended validation epoch %s", self.trainer.current_epoch)
        miou, acc = self.iou_meter["val"].get_miou_acc()
        self.log(
            "val/mIoU",
            miou,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "val/accuracy",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

    def on_test_epoch_end(self) -> None:
        """Callback after a test epoch"""
        miou, acc = self.iou_meter["test"].get_miou_acc()
        self.log(
            "test/mIoU",
            miou,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "test/accuracy",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

    def on_fit_start(self) -> None:
        """
        On fit start, load pretrained MMDC model if its path is set.
        The scales are already set in the loaded model
        """
        logging.info("On fit start")
        if self.resume_from_checkpoint is None:
            self.model.apply(weight_init)

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
