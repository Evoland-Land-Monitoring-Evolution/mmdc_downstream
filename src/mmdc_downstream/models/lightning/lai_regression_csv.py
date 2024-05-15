# Copyright: (c) 2022 CESBIO / Centre National d'Etudes Spatiales
""" Base classes for the MMDC Lightning module and Pytorch model"""
import logging
from typing import Any

import torch
from pytorch_lightning import LightningModule
from torchutils import metrics

from mmdc_downstream.models.components.losses import compute_losses_flat
from mmdc_downstream.models.torch.lai_regression import MMDCDownstreamRegressionModule
from mmdc_downstream.snap.lai_snap import BVNET, denormalize

from ..components.metrics import compute_val_metrics_flat

# Configure logging
NUMERIC_LEVEL = getattr(logging, "INFO", None)
logging.basicConfig(
    level=NUMERIC_LEVEL, format="%(asctime)-15s %(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


class MMDCLAILitModule(LightningModule):  # pylint: disable=too-many-ancestors
    """
    Base Lightning Module for the MMDC Single Date networks
    """

    def __init__(
        self,
        model: MMDCDownstreamRegressionModule,
        losses_list: list[str],
        metrics_list: list[str],
        lr: float = 0.0001,
        resume_from_checkpoint: str | None = None,
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
        self.learning_rate = lr

        self.losses_list = losses_list
        self.metrics_list = metrics_list

        self.model_snap = BVNET(device="cuda", ver="2", variable="lai")

    def step(self, batch: Any, stage: str = "train") -> Any:
        """Perform one optimization step"""
        """
        One step.
        We produce GT LAI with SNAP.
        We generate regression input depending on the task.
        """
        reg_input, lai_gt = batch
        lai_pred = self.forward(reg_input)
        losses = compute_losses_flat(
            preds=lai_pred,
            target=lai_gt,
            losses_list=self.losses_list,
        )
        # logging.info(torch.min(lai_gt[~torch.isnan(lai_gt)]))
        # logging.info(torch.max(lai_gt[~torch.isnan(lai_gt)]))
        # logging.info(torch.mean(lai_gt[~torch.isnan(lai_gt)]))

        if stage == "val":
            self.pred_val.extend(lai_pred.cpu().detach().to(torch.float16).reshape(-1))
            self.gt_val.extend(lai_gt.cpu().to(torch.float16).reshape(-1))
        # if stage == "test":
        #     self.pred_test.extend(lai_pred.cpu().detach().numpy())
        #     self.gt_test.extend(lai_gt.cpu().numpy())

        if stage != "train":
            metrics = compute_val_metrics_flat(
                preds=denormalize(
                    lai_pred, self.model_snap.variable_min, self.model_snap.variable_max
                ),
                target=denormalize(
                    lai_gt, self.model_snap.variable_min, self.model_snap.variable_max
                ),
                metrics_list=self.metrics_list,
            )
            return losses, metrics

        return losses

    def forward(self, data: Any) -> Any:  # pylint: disable=arguments-differ
        """Forward"""
        return self.model.forward(data)

    def training_step(  # pylint: disable=arguments-differ
        self,
        batch: Any,
        batch_idx: int,  # pylint: disable=unused-argument
    ) -> dict[str, Any]:
        """Training step. Step and return loss."""
        torch.autograd.set_detect_anomaly(True)
        losses = self.step(batch)

        # log training metrics
        for loss_name, loss_value in losses.items():
            self.log(
                f"train/{loss_name}",
                loss_value,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

        return {"loss": sum(losses.values())}

    def validation_step(  # pylint: disable=arguments-differ
        self,
        batch: Any,
        batch_idx: int,  # pylint: disable=unused-argument
        prefix: str = "val",
    ) -> dict[str, Any]:
        """Validation step. Step and return loss."""
        losses, metrics = self.step(batch, stage=prefix)

        for loss_name, loss_value in losses.items():
            self.log(
                f"{prefix}/{loss_name}",
                loss_value,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

        for metric_name, metric_value in metrics.items():
            self.log(
                f"{prefix}/{metric_name}",
                metric_value,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
        return {"loss": sum(losses.values())}

    def test_step(  # pylint: disable=arguments-differ
        self,
        batch: Any,
        batch_idx: int,  # pylint: disable=unused-argument
    ) -> dict[str, Any]:
        """Test step. Step and return loss. Delegate to validation step"""
        return self.validation_step(batch, batch_idx, prefix="test")

    def on_train_epoch_end(self) -> None:
        logger.info("Ended traning epoch %s", self.trainer.current_epoch)

    def on_validation_epoch_start(self) -> None:
        self.pred_val = []
        self.gt_val = []

    def on_test_epoch_start(self) -> None:
        self.pred_test = []
        self.gt_test = []

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
        training_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.5
        )
        scheduler = {
            "scheduler": training_scheduler,
            "interval": "epoch",
            "monitor": "val/loss",
            "frequency": 1,
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
