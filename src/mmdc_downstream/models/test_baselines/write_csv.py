#!/usr/bin/env python3
# Copyright: (c) 2022 CESBIO / Centre National d'Etudes Spatiales
""" Lightning module for lai regression prediction """

import csv
from typing import Any

import numpy as np
import torch
from einops import rearrange
from mmdc_singledate.datamodules.datatypes import MMDCBatch
from mmdc_singledate.datamodules.mmdc_datamodule import destructure_batch
from mmdc_singledate.utils.train_utils import standardize_data

from mmdc_downstream.mmdc_model.model import PretrainedMMDC

from ...snap.components.compute_bio_var import (
    predict_variable_from_tensors,
    prepare_s2_image,
    process_output,
)
from ...snap.lai_snap import BVNET, denormalize, normalize
from ..components.losses import compute_losses
from ..datatypes import OutputLAI
from ..lightning.base import MMDCDownstreamBaseLitModule
from ..torch.lai_regression import MMDCDownstreamRegressionModule


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
        metrics_list: list[str] = ["RSE", "R2", "MAE", "MAPE"],
        losses_list: list[str] = ["MSE"],
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
        self.input_data = input_data

        self.losses_list = losses_list
        self.metrics_list = metrics_list

        self.margin = (
            self.model_mmdc.model_mmdc.nb_cropped_hw
            if self.model_mmdc is not None
            else 0
        )

        # self.hist = torch.zeros(150).to("cuda")
        self.bin_weights = self.compute_bin_weights(histogram_path)

        self.header = None

        if stats_path is not None:
            self.stats = self.get_stats(torch.load(stats_path))

    @staticmethod
    def compute_bin_weights(histogram_path):
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
            return self.model_mmdc.get_latent_mmdc(batch).latent_S1_mu
        if self.input_data == "lat_S2":
            return self.model_mmdc.get_latent_mmdc(batch).latent_S2_mu
        if self.input_data == "lat_S1_mulogvar":
            latent = self.model_mmdc.get_latent_mmdc(batch)
            return torch.cat((latent.latent_S1_mu, latent.latent_S1_logvar), 1)
        if self.input_data == "lat_S2_mulogvar":
            latent = self.model_mmdc.get_latent_mmdc(batch)
            return torch.cat((latent.latent_S2_mu, latent.latent_S2_logvar), 1)
        if self.input_data == "S2":
            data = prepare_s2_image(
                batch.s2_x / 10000, batch.s2_a, reshape=False
            ).nan_to_num()
            return normalize(
                data,
                self.model_snap.input_min.reshape(1, data.shape[1], 1, 1),
                self.model_snap.input_max.reshape(1, data.shape[1], 1, 1),
            )
            # s2_x = standardize_data(
            #     batch.s2_x,
            #     shift=self.stats.sen2.shift.type_as(
            #         batch.s2_x),
            #     scale=self.stats.sen2.shift.type_as(
            #         batch.s2_x),
            # )
        if "S1" in self.input_data:
            s1_x = standardize_data(
                batch.s1_x,
                shift=self.stats.sen1.shift.type_as(batch.s1_x),
                scale=self.stats.sen1.shift.type_as(batch.s1_x),
            )
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
        lai_pred = self.forward(reg_input)
        mask = batch.s2_m.reshape(-1).bool()
        nbr_pixels = (mask == 0).sum().cpu().numpy()
        idx = np.random.choice(nbr_pixels, int(nbr_pixels * 0.1), replace=False)
        s2_ref = (
            rearrange(batch.s2_x, "b c h w -> (b h w) c")[~mask][idx].cpu().numpy()
            / 10000
        )
        s2_angles = (
            rearrange(batch.s2_a, "b c h w -> (b h w) c")[~mask][idx].cpu().numpy()
        )
        s1 = standardize_data(
            rearrange(batch.s1_x, "b c h w -> (b h w) c")[~mask][idx].cpu().numpy(),
            shift=self.stats.sen1.shift.type_as(batch.s1_x),
            scale=self.stats.sen1.shift.type_as(batch.s1_x),
        )
        s1_angles = (
            rearrange(batch.s1_a, "b c h w -> (b h w) c")[~mask][idx].cpu().numpy()
        )
        s1_mask = (
            rearrange(batch.s1_vm, "b c h w -> (b h w) c")[~mask][idx].cpu().numpy()
        )
        s2_input = (
            rearrange(reg_input, "b c h w -> (b h w) c")[~mask][idx].cpu().numpy()
        )
        latent = self.model_mmdc.get_latent_mmdc(batch)
        s1_lat = (
            rearrange(latent.latent_S1_mu, "b c h w -> (b h w) c")[~mask][idx]
            .to(torch.float32)
            .cpu()
            .numpy()
        )
        s2_lat = (
            rearrange(latent.latent_S2_mu, "b c h w -> (b h w) c")[~mask][idx]
            .to(torch.float32)
            .cpu()
            .numpy()
        )
        exp_lat = (
            rearrange(latent.latent_experts_mu, "b c h w -> (b h w) c")[~mask][idx]
            .to(torch.float32)
            .cpu()
            .numpy()
        )
        gt = (
            lai_gt.reshape(-1)[~mask][idx]
            .to(torch.float32)
            .unsqueeze(-1)
            .cpu()
            .numpy()
            .round(3)
        )

        if self.header is None:
            self.header = []
            self.header = np.concatenate(
                (
                    [f"s2_{i}" for i in range(s2_ref.shape[1])],
                    [f"s2_ang_{i}" for i in range(s2_angles.shape[1])],
                    [f"s1_{i}" for i in range(s1.shape[1])],
                    [f"s1_ang_{i}" for i in range(s1_angles.shape[1])],
                    [f"s1_mask_{i}" for i in range(s1_mask.shape[1])],
                    [f"s2_input_{i}" for i in range(s2_input.shape[1])],
                    [f"s1_lat_{i}" for i in range(s1_lat.shape[1])],
                    [f"s2_lat_{i}" for i in range(s2_lat.shape[1])],
                    [f"exp_lat_{i}" for i in range(exp_lat.shape[1])],
                    ["gt"],
                ),
                axis=0,
            )
            print(self.header)
            with open("data_values_exp.csv", "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(self.header)
        data_to_write = np.hstack(
            (
                s2_ref,
                s2_angles,
                s1,
                s1_angles,
                s1_mask,
                s2_input,
                s1_lat,
                s2_lat,
                exp_lat,
                gt,
            )
        )
        with open("data_values_exp_norm.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(data_to_write)
        losses = compute_losses(
            preds=lai_pred,
            target=lai_gt,
            mask=batch.s2_m,
            margin=self.margin,
            losses_list=self.losses_list,
            bin_weights=self.bin_weights,
            denorm_min_max=(self.model_snap.variable_min, self.model_snap.variable_max),
        )

        return losses

    def training_step(  # pylint: disable=arguments-differ
        self,
        batch: Any,
        batch_idx: int,  # pylint: disable=unused-argument
    ) -> dict[str, Any]:
        """Training step. Step and return loss."""
        torch.autograd.set_detect_anomaly(True)
        losses = self.step(batch)

        return {"loss": sum(losses.values())}

    def validation_step(  # pylint: disable=arguments-differ
        self,
        batch: Any,
        batch_idx: int,  # pylint: disable=unused-argument
        prefix: str = "val",
    ) -> dict[str, Any]:
        """Validation step. Step and return loss."""
        losses = self.step(batch, stage=prefix)

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
        # lai_pred = unstand_lai(self.forward(reg_input))
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
            # return stand_lai(gt)
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
        training_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.9, patience=200, threshold=0.01
        )

        scheduler = {
            "scheduler": training_scheduler,
            "interval": "step",
            "monitor": f"val/{self.losses_list[0]}",
            "frequency": 1,
            "strict": False,
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    # def on_train_epoch_start(self) -> None:
    #     self.hist = torch.zeros(150).to("cuda")
    #
    # def on_train_epoch_end(self) -> None:
    #     torch.save(self.hist, "./hist_bins.pt")

    def on_train_epoch_end(self) -> None:
        exit()
