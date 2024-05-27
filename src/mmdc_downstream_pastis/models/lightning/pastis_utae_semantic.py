#!/usr/bin/env python3
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
""" Lightning module for lai regression prediction """

import logging
from typing import Any

import torch

from ...datamodule.datatypes import BatchInputUTAE
from ..components.losses import compute_losses
from ..torch.utae import UTAE
from ..torch.utae_fusion import UTAEFusion
from .base import MMDCPastisBaseLitModule

# Configure logging
NUMERIC_LEVEL = getattr(logging, "INFO", None)
logging.basicConfig(
    level=NUMERIC_LEVEL, format="%(asctime)-15s %(levelname)s: %(message)s"
)
logging.getLogger(__name__).setLevel(logging.INFO)


def to_class_label(logits: torch.Tensor) -> torch.Tensor:
    """Get class labels from logits"""
    soft = torch.nn.Softmax()
    probs = soft(logits)
    pred = probs.max(1).indices
    return pred


class PastisUTAE(MMDCPastisBaseLitModule):
    """
    Pastis Semantic segmentation module.
    """

    def __init__(
        self,
        model: UTAE | UTAEFusion,
        metrics_list: list[str] = ["cross"],
        losses_list: list[str] = ["cross"],
        lr: float = 0.001,
        lr_type: str = "constant",
        resume_from_checkpoint: str | None = None,
    ):
        super().__init__(model, lr, lr_type, resume_from_checkpoint)

        self.losses_list = losses_list
        self.metrics_list = metrics_list

        self.model = model

    def step(self, batch: BatchInputUTAE, stage: str = "train") -> Any:
        """
        One step.
        We compute logits and data classes for PASTIS
        """
        gt = batch.gt.long()

        logits = self.forward(batch)
        logits_clip = logits[:, :, 32:-32, 32:-32].contiguous()
        gt_clip = gt[:, 32:-32, 32:-32].contiguous()
        gt_mask_clip = (
            ((gt_clip == 0) | (gt_clip == 19))
            if batch.gt_mask is None
            else batch.gt_mask[:, 32:-32, 32:-32].contiguous()
        )

        losses = compute_losses(
            preds=logits_clip,
            target=gt_clip,
            mask=gt_mask_clip,
            losses_list=self.losses_list,
        )

        if stage == "train":
            self.iou_meter_train.add(to_class_label(logits_clip), gt_clip)
        if stage == "val":
            self.iou_meter_val.add(to_class_label(logits_clip), gt_clip)
        if stage == "test":
            self.iou_meter_test.add(to_class_label(logits_clip), gt_clip)
        return losses

    def training_step(  # pylint: disable=arguments-differ
        self,
        batch: Any,
        batch_idx: int,  # pylint: disable=unused-argument
    ) -> dict[str, Any]:
        """Training step. Step and return loss."""
        torch.autograd.set_detect_anomaly(True)
        losses = self.step(batch)
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
        losses = self.step(batch, stage=prefix)

        for loss_name, loss_value in losses.items():
            self.log(
                f"{prefix}/{loss_name}",
                loss_value,
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

    def forward(self, batch: BatchInputUTAE) -> torch.Tensor:
        """Forward step"""
        return self.model.forward(batch.sits, batch_positions=batch.doy)

    def predict(self, batch: BatchInputUTAE) -> torch.Tensor:
        """Predict classification map"""
        self.model.eval()
        with torch.no_grad():
            return to_class_label(self.forward(batch))
