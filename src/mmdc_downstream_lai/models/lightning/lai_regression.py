#!/usr/bin/env python3
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
""" Lightning module for lai regression prediction """

import logging
from typing import Any

import torch
from mmdc_singledate.datamodules.datatypes import MMDCBatch
from mmdc_singledate.datamodules.mmdc_datamodule import destructure_batch

from mmdc_downstream_lai.mmdc_model.model import PretrainedMMDC

from ...snap.components.compute_bio_var import (
    predict_variable_from_tensors,
    prepare_s2_image,
    process_output,
)
from ...snap.lai_snap import BVNET, denormalize, normalize
from ..components.losses import compute_losses
from ..components.metrics import compute_val_metrics
from ..datatypes import OutputLAI
from ..torch.lai_regression import MMDCDownstreamRegressionModule
from .base import MMDCDownstreamBaseLitModule

logging.getLogger().setLevel(logging.INFO)


class MMDCDownstreamRegressionLitModule(MMDCDownstreamBaseLitModule):
    """
    LAI regression lightning module.
    Attributes:
        model: regression model
        model_snap: Fixed weights SNAP model for LAI GT computation
        model_mmdc: Pretrained MMDC model for latent representations
    """

    def __init__(
        self,
        model: MMDCDownstreamRegressionModule,
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
        super().__init__(model, model_mmdc, lr, resume_from_checkpoint, stats_path)

        self.model_snap = model_snap
        self.model_snap.set_snap_weights()
        self.model_snap.eval()
        self.input_data = input_data

        self.losses_list = losses_list
        self.metrics_list = metrics_list

        self.margin = (
            self.model_mmdc.model_mmdc.nb_cropped_hw
            if self.model_mmdc is not None
            else 0
        )

        self.bin_weights = self.compute_bin_weights(histogram_path)

    @staticmethod
    def compute_bin_weights(histogram_path):
        """
        Compute weights for each value bin.
        Will be used in loss computation as weights.
        """
        if histogram_path is None:
            return None
        hist_bins = torch.load(histogram_path)
        hist_bins_norm = hist_bins / hist_bins.sum()
        a = -10
        weight = torch.exp((hist_bins_norm**0.5) * a)
        weights_norm = weight / weight.sum()
        return weights_norm.to(torch.float16)

    def get_regression_input(self, batch: MMDCBatch) -> torch.Tensor:
        """
        Prepare input for downstream model.
        It can be normalized S1/S2 data or one of latent representations
        produced by MMDC model.
        """
        # [experts, lat_S1, lat_S2, S2, S1, S1_asc, S1_desc]
        if self.input_data == "experts":
            return self.model_mmdc.get_latent_mmdc(batch).latent_experts_mu
        if self.input_data == "lat_S1":
            return self.model_mmdc.get_latent_mmdc(batch).latent_s1_mu
        if self.input_data == "lat_S2":
            return self.model_mmdc.get_latent_mmdc(batch).latent_s2_mu
        if self.input_data == "lat_S1_mulogvar":
            latent = self.model_mmdc.get_latent_mmdc(batch)
            return torch.cat((latent.latent_s1_mu, latent.latent_s1_logvar), 1)
        if self.input_data == "lat_S2_mulogvar":
            latent = self.model_mmdc.get_latent_mmdc(batch)
            return torch.cat((latent.latent_s2_mu, latent.latent_s2_logvar), 1)
        if self.input_data == "S2":
            data = prepare_s2_image(
                batch.s2_x / 10000, batch.s2_a, reshape=False
            ).nan_to_num()
            return normalize(
                data,
                self.model_snap.input_min.reshape(1, data.shape[1], 1, 1),
                self.model_snap.input_max.reshape(1, data.shape[1], 1, 1),
            )

        if "S1" in self.input_data:
            # TODO add standardization
            s1_x = batch.s1_x
            # s1_x = standardize_data(
            #     batch.s1_x,
            #     shift=self.stats.sen1.shift.type_as(
            #         batch.s1_x),
            #     scale=self.stats.sen1.shift.type_as(
            #         batch.s1_x),
            # )
            if self.input_data == "S1":
                return torch.cat((s1_x, batch.s1_a), 1)
            if self.input_data == "S1_asc":  # TODO: add angles
                return s1_x[:, :3]
            return s1_x[:, 3:]

    def step(self, batch: Any, stage: str = "train") -> Any:
        """
        One step.
        We produce GT LAI with SNAP.
        We generate regression input depending on the task.
        """
        batch: MMDCBatch = destructure_batch(batch)
        lai_gt = self.compute_gt(batch)
        reg_input = self.get_regression_input(batch)
        mask = batch.s2_m
        if "lat" in self.input_data or "experts" in self.input_data:
            H, W = reg_input.shape[-2:]
            reg_input = reg_input[
                :, :, self.margin : H - self.margin, self.margin : W - self.margin
            ]
            lai_gt = lai_gt[
                :, :, self.margin : H - self.margin, self.margin : W - self.margin
            ]
            mask = mask[
                :, :, self.margin : H - self.margin, self.margin : W - self.margin
            ]
        lai_pred = self.forward(reg_input)
        logging.info(f"MMDC {self.model_mmdc.model_mmdc.training}")
        logging.info(f"LAI NN {self.model.training}")
        losses = compute_losses(
            preds=lai_pred,
            target=lai_gt,
            mask=mask,
            losses_list=self.losses_list,
            bin_weights=self.bin_weights,
            denorm_min_max=(self.model_snap.variable_min, self.model_snap.variable_max),
        )
        logging.info(torch.min(lai_gt[~torch.isnan(lai_gt)]))
        logging.info(torch.max(lai_gt[~torch.isnan(lai_gt)]))
        logging.info(torch.mean(lai_gt[~torch.isnan(lai_gt)]))

        if stage != "train":
            metrics = compute_val_metrics(
                preds=denormalize(
                    lai_pred, self.model_snap.variable_min, self.model_snap.variable_max
                ),
                target=denormalize(
                    lai_gt, self.model_snap.variable_min, self.model_snap.variable_max
                ),
                mask=mask,
                metrics_list=self.metrics_list,
            )
            return losses, metrics

        return losses

    def training_step(  # pylint: disable=arguments-differ
        self,
        batch: Any,
        batch_idx: int,  # pylint: disable=unused-argument
    ) -> dict[str, Any]:
        """Training step. Step and return loss."""
        # torch.autograd.set_detect_anomaly(True)
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

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward step"""
        data_prep = data.permute(0, 2, 3, 1).reshape(-1, data.size(1))
        return process_output(
            self.model.forward(data_prep),
            torch.Size([data.size(0), 1, data.size(2), data.size(3)]),
        )

    def predict(self, batch: MMDCBatch) -> OutputLAI:
        """Predict LAI"""
        self.model.eval()
        lai_gt = self.compute_gt(batch, stand=False)
        reg_input = self.get_regression_input(batch)
        lai_pred = denormalize(
            self.forward(reg_input),
            self.model_snap.variable_min,
            self.model_snap.variable_max,
        )
        return OutputLAI(lai_pred, reg_input, lai_gt)

    def compute_gt(self, batch: MMDCBatch, stand: bool = True) -> torch.Tensor:
        """Compute LAI GT data wiht standardisation or not"""
        gt = predict_variable_from_tensors(
            batch.s2_x, batch.s2_a, batch.s2_m, self.model_snap
        )
        if stand:
            return normalize(
                gt, self.model_snap.variable_min, self.model_snap.variable_max
            )
        return gt

    def configure_optimizers(self) -> dict[str, Any]:
        """A single optimizer with a LR scheduler"""
        optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=self.learning_rate, weight_decay=0.01
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
