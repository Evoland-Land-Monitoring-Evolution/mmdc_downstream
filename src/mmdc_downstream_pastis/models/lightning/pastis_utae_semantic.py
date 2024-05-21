#!/usr/bin/env python3
# Copyright: (c) 2022 CESBIO / Centre National d'Etudes Spatiales
""" Lightning module for lai regression prediction """

import logging
from pathlib import Path
from typing import Any

import torch
from mmdc_singledate.datamodules.datatypes import ShiftScale
from mmdc_singledate.utils.train_utils import standardize_data
from torch import nn

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


def compute_stats(data_type: str, sat_stats: dict) -> ShiftScale:
    scale_regul = nn.Threshold(1e-10, 1.0)
    return ShiftScale(
        sat_stats[data_type][1],
        scale_regul((sat_stats[data_type][2] - sat_stats[data_type][0]) / 2.0),
    )


def set_aux_stats(
    stats_path: dict[str | Path] | str | Path | None,
    device: torch.device,
    sat: str,
) -> dict[str, ShiftScale] | ShiftScale:
    if type(stats_path) != dict:
        sat_stats = torch.load(stats_path)
        stats_sat = {d: compute_stats(d, sat_stats) for d in sat_stats}
        angles_len = 6 if sat == "S2" else 1
        return ShiftScale(
            torch.stack(
                [
                    stats_sat["img"].shift,
                    torch.zeros(angles_len),
                    stats_sat["meteo"].shift,
                    stats_sat["dem"].shift,
                ]
            ).to(device),
            torch.stack(
                [
                    stats_sat["img"].scale,
                    torch.ones(angles_len),
                    stats_sat["meteo"].scale,
                    stats_sat["dem"].scale,
                ]
            ).to(device),
        )
    return {sat: set_aux_stats(stats_path, device, sat) for sat in stats_path}


class PastisUTAE(MMDCPastisBaseLitModule):
    """
    LAI regression lightning module.
    Attributes:
        model: regression model
        model_snap: Fixed weights SNAP model for LAI GT computation
        model_mmdc: Pretrained MMDC model for latent representations
    """

    def __init__(
        self,
        model: UTAE | UTAEFusion,
        metrics_list: list[str] = ["cross"],
        losses_list: list[str] = ["cross"],
        lr: float = 0.001,
        resume_from_checkpoint: str | None = None,
        stats_aux_path: dict[str | Path] | None = None,
    ):
        super().__init__(model, lr, resume_from_checkpoint)

        self.losses_list = losses_list
        self.metrics_list = metrics_list

        self.model = model
        self.set_aux_stats = stats_aux_path
        self.stats = None

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
            else batch.gt_mask[32:-32, 32:-32].contiguous()
        )

        losses = compute_losses(
            preds=logits_clip,
            target=gt_clip,
            mask=gt_mask_clip,
            losses_list=self.losses_list,
        )
        self.iou_meter[stage].add(to_class_label(logits_clip), gt_clip)

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
        sits = batch.sits
        if self.stats is not None:
            if type(sits) is dict:
                sits = {
                    sat: standardize_data(
                        sits[sat],
                        shift=self.stats[sat].shift,
                        scale=self.stats[sat].scale,
                    )
                    for sat in sits
                }
            else:
                sits = standardize_data(
                    sits, shift=self.stats.shift, scale=self.stats.scale
                )
        return self.model.forward(sits, batch_positions=batch.doy)

    def predict(self, batch: BatchInputUTAE) -> torch.Tensor:
        """Predict classification map"""
        self.model.eval()
        return to_class_label(self.model.forward(batch))

    def on_fit_start(self) -> None:
        """On fit start, get the stats, and set them into the model"""
        assert hasattr(self.trainer, "datamodule")
        if self.stats_aux_path is not None:
            self.stats = set_aux_stats(
                self.stats_aux_path,
                device=self.model.device,
                sat=self.trainer.datamodule.sats[0]
                if type(self.stats_aux_path) is not dict
                else None,
            )

    def configure_optimizers(self) -> dict[str, Any]:
        """A single optimizer with a LR scheduler"""
        optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=self.learning_rate  # , weight_decay=0.01
        )

        # training_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer, T_0=4, T_mult=2, eta_min=0, last_epoch=-1)
        # training_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode="min", factor=0.9, patience=200, threshold=0.01
        # )
        #
        # scheduler = {
        #     # "scheduler": training_scheduler,
        #     "interval": "step",
        #     "monitor": f"val/{self.losses_list[0]}",
        #     "frequency": 1,
        #     "strict": False,
        # }
        return {
            "optimizer": optimizer,
            # "lr_scheduler": scheduler,
        }

    # def configure_optimizers(self) -> dict[str, Any]:
    #     """A single optimizer with a LR scheduler"""
    #     optimizer = torch.optim.Adam(
    #         params=self.model.parameters(),
    #         lr=self.learning_rate  # , weight_decay=0.01
    #     )
    #
    #     training_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #         optimizer, T_0=4, T_mult=2, eta_min=0, last_epoch=-1)
    #     # training_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     #     optimizer, mode="min", factor=0.9, patience=200, threshold=0.01
    #     # )
    #
    #     scheduler = {
    #         "scheduler": training_scheduler,
    #         "interval": "step",
    #         "monitor": f"val/{self.losses_list[0]}",
    #         "frequency": 1,
    #         "strict": False,
    #     }
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": scheduler,
    #     }
