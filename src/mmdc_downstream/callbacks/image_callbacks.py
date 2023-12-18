#!/usr/bin/env python3
# Copyright: (c) 2022 CESBIO / Centre National d'Etudes Spatiales
""" Lightning image callbacks """
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from sensorsio.utils import rgb_render

from ..models.lightning.lai_regression import MMDCDownstreamRegressionLitModule, OutputLAI
from mmdc_singledate.datamodules.mmdc_datamodule import destructure_batch

from dataclasses import dataclass

# Configure logging
NUMERIC_LEVEL = getattr(logging, "INFO", None)
logging.basicConfig(level=NUMERIC_LEVEL,
                    format="%(asctime)-15s %(levelname)s: %(message)s")

logger = logging.getLogger(__name__)


@dataclass
class SampleInfo:
    """Information about the sample to be displayed"""

    batch_idx: int
    batch_size: int
    patch_margin: int
    current_epoch: int


class MMDCLAIExpertsCallback(Callback):
    """
    Callback to inspect the LAI image predicted from Latent Experts
    """

    def __init__(
            self,
            save_dir: str,
            n_samples: int = 5,
    ):
        self.save_dir = save_dir
        self.n_samples = n_samples
        self.labels = ["S1 Asc", "S2 Desc", "S2", "Lat", "LAI GT", "LAI Pred"]

    def save_image_grid(
        self,
        sen_data: tuple[torch.Tensor, torch.Tensor],
        pred: OutputLAI,
        sample: SampleInfo,
    ) -> None:
        """
        Generate the matplotlib figure and save it to a file.
        We save one sample per line.
        """

        prepared_data = self.prepare_data(sen_data, pred, sample.patch_margin)

        plt.close()

        fig, axes = plt.subplots(
            nrows=self.n_samples,
            ncols=len(self.labels),
            sharex=True,
            sharey=True,
            figsize=((len(self.labels) + 1) * 2.5, (self.n_samples + 1) * 2.5),
        )
        fig.suptitle("Prediction Inspection", fontsize=20)

        for samp_idx in range(self.n_samples):  # We iterate through samples to plot
            data_to_plot = self.prepare_sample(prepared_data, samp_idx)
            for col in range(len(self.labels)):     # We plot each image of the sample
                axes[samp_idx, col].imshow(data_to_plot[col],
                                           interpolation="bicubic")
                axes[samp_idx, col].set_title(self.labels[col])

        del prepared_data

        # export name
        image_basename = (f"LAI_val_ep_{sample.current_epoch:03}_samples_"
                          f"0-{self.n_samples:01}_{sample.batch_idx}")
        image_name = Path(f"{self.save_dir}/{image_basename}.png")
        fig.savefig(image_name)

    def prepare_data(
        self,
        sen_data: tuple[torch.Tensor, torch.Tensor],
        pred: OutputLAI,
        margin: int,
    ) -> tuple[torch.Tensor, ...]:
        """
        We clip and select bands from images
        """
        s1, s2 = sen_data

        s1_asc = s1[:self.n_samples, :3, margin:-margin, margin:-margin]
        s1_desc = s1[:self.n_samples, 3:, margin:-margin, margin:-margin]
        s2_rgb = s2[:self.n_samples, [2, 1, 0], margin:-margin, margin:-margin]
        latent_mu = pred.latent.clone()[:self.n_samples, [0, 1, 2],
                                        margin:-margin, margin:-margin]
        lai_gt = pred.lai_gt.clone()[:self.n_samples, [0, 0, 0],
                                     margin:-margin, margin:-margin]
        lai_pred = pred.lai_pred.clone()[:self.n_samples, [0, 0, 0],
                                         margin:-margin, margin:-margin]
        return s1_asc, s1_desc, s2_rgb, latent_mu, lai_gt, lai_pred

    def prepare_sample(
        self,
        prepared_data: tuple[torch.Tensor, ...],
        samp_idx: int,
    ) -> tuple[np.array, ...]:
        """
        Sample rendering for matplotlib
        """
        s1_asc, s1_desc, s2_rgb, latent_mu, lai_gt, lai_pred = prepared_data
        render_s1_asc, _, _ = rgb_render(
            s1_asc[samp_idx].cpu().detach().numpy())
        render_s1_desc, _, _ = rgb_render(
            s1_desc[samp_idx].cpu().detach().numpy())
        render_s2, _, _ = rgb_render(s2_rgb[samp_idx].cpu().detach().numpy())
        render_lat, _, _ = rgb_render(
            latent_mu[samp_idx].cpu().detach().numpy())
        _, min_lai, max_lai = rgb_render(
            lai_gt[samp_idx].cpu().detach().numpy())
        lai_gt = lai_gt.nan_to_num()
        render_lai_gt, _, _ = rgb_render(
            lai_gt[samp_idx].cpu().detach().numpy(),
            dmin=min_lai,
            dmax=max_lai)
        render_lai_pred, _, _ = rgb_render(
            lai_pred[samp_idx].cpu().detach().numpy(),
            dmin=min_lai,
            dmax=max_lai)

        return render_s1_asc, render_s1_desc, render_s2, render_lat, render_lai_gt, render_lai_pred

    def lai_gt_pred_scatterplots(
        self,
        lai: tuple[torch.Tensor, torch.Tensor],
        sample: SampleInfo,
    ) -> None:
        """Save the PNG image of the scatterplots of the latent space of
        a sample of the batch"""
        (pred, gt) = lai
        image_basename = (
            f"MMDC_val_ep_{sample.current_epoch:03}_scat_pred_gt_"
            f"0-{self.n_samples:03}_{sample.batch_idx}")
        image_name = Path(f"{self.save_dir}/{image_basename}.png")
        gt = gt.nan_to_num()
        plt.close()
        fig, axes = plt.subplots(nrows=1,
                                 ncols=self.n_samples,
                                 sharex=False,
                                 sharey=False,
                                 figsize=((self.n_samples + 1) * 3, 8))
        fig.suptitle(f"LAI {sample.current_epoch = }", fontsize=20)
        for i, axis in enumerate(axes.flatten()):
            axis.set_title(f"LAI")
            axis.set_xlabel("Pred")
            axis.set_ylabel("GT")
            axis.scatter(pred[i].detach().cpu().numpy().flatten(),
                         gt[i].detach().cpu().numpy().flatten())
            axis.plot(gt[i].detach().cpu().numpy().flatten(),
                      gt[i].detach().cpu().numpy().flatten())
        fig.savefig(image_name)

    def on_validation_batch_end(  # pylint: disable=too-many-arguments
        self,
        trainer: pl.trainer.Trainer,
        pl_module: MMDCDownstreamRegressionLitModule,
        outputs: Any,
        batch: torch.Tensor,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Method called from the validation loop"""
        if batch_idx < 2:
            debatch = destructure_batch(batch)[:self.n_samples]

            patch_margin = pl_module.margin
            assert isinstance(patch_margin, int)

            pred = pl_module.predict(debatch)

            s2_x, s1_x = debatch.s2_x.nan_to_num(), debatch.s1_x.nan_to_num()
            batch_size = s1_x.shape[0]

            self.save_image_grid(
                (s1_x, s2_x),
                pred,
                SampleInfo(batch_idx, batch_size, patch_margin,
                           trainer.current_epoch),
            )
            self.lai_gt_pred_scatterplots(
                (pred.lai_pred, pred.lai_gt),
                SampleInfo(batch_idx, batch_size, patch_margin, trainer.current_epoch),
            )


class MMDCLAIS2Callback(MMDCLAIExpertsCallback):
    def __init__(
            self,
            save_dir: str,
            n_samples: int = 5,
    ):
        """
        Image callback for LAI prediction from S2 images
        """
        super().__init__(save_dir, n_samples)
        self.labels = ["S2", "LAI GT", "LAI Pred"]

    def prepare_data(
        self,
        sen_data: tuple[torch.Tensor, torch.Tensor],
        pred: OutputLAI,
        margin: int,
    ) -> tuple[torch.Tensor, ...]:
        """
        We select bands from images.
        """
        s1, s2 = sen_data

        s2_rgb = s2[:self.n_samples, [2, 1, 0], :, :]
        lai_gt = pred.lai_gt.clone()[:self.n_samples, [0, 0, 0], :, :]
        lai_pred = pred.lai_pred.clone()[:self.n_samples, [0, 0, 0], :, :]
        return s2_rgb, lai_gt, lai_pred

    def prepare_sample(
        self,
        prepared_data: tuple[torch.Tensor, ...],
        samp_idx: int,
    ) -> tuple[np.array, ...]:
        """
        Sample rendering for matplotlib
        """
        s2_rgb, lai_gt, lai_pred = prepared_data
        render_s2, _, _ = rgb_render(s2_rgb[samp_idx].cpu().detach().numpy())
        _, min_lai, max_lai = rgb_render(
            lai_gt[samp_idx].cpu().detach().numpy())
        lai_gt = lai_gt.nan_to_num()
        render_lai_gt, _, _ = rgb_render(
            lai_gt[samp_idx].cpu().detach().numpy(),
            dmin=min_lai,
            dmax=max_lai)
        render_lai_pred, _, _ = rgb_render(
            lai_pred[samp_idx].cpu().detach().numpy(),
            dmin=min_lai,
            dmax=max_lai)

        return render_s2, render_lai_gt, render_lai_pred

            # self.lai_gt_pred_scatterplots(
            #     (pred.lai_pred, pred.lai_gt),
            #     SampleInfo(batch_idx, batch_size, trainer.current_epoch),
            # )
