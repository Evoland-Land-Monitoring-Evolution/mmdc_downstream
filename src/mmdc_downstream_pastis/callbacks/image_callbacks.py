#!/usr/bin/env python3
# Copyright: (c) 2022 CESBIO / Centre National d'Etudes Spatiales
""" Lightning image callbacks """
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

from mmdc_downstream_pastis.encode_series.encode import back_to_date
from mmdc_downstream_pastis.models.lightning.pastis_encoded_utae_semantic import (
    MMDCPastisEncodedUTAE,
)
from mmdc_downstream_pastis.models.lightning.pastis_utae_semantic import PastisUTAE

# from sensorsio.utils import rgb_render


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
    current_epoch: int


class MMDCAECallback(Callback):
    """
    Callback to inspect the reconstruction image
    """

    def __init__(
        self,
        save_dir: str,
        n_samples: int = 3,
        normalize: bool = False,
        value_range: tuple[float, float] = (-1.0, 1.0),
    ):
        self.save_dir = save_dir
        self.n_samples = n_samples
        self.normalize = normalize
        self.value_range = value_range

    def prepare_s1_s2(
        self,
        s1_data: torch.Tensor,
        s2_data: torch.Tensor,
        samp_idx: int,
        patch_margin: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare the patches for the S1 and S2 data"""
        s1_asc = s1_data[
            samp_idx,
            :3,
            patch_margin:-patch_margin,
            patch_margin:-patch_margin,
        ]
        s1_desc = s1_data[
            samp_idx,
            3:,
            patch_margin:-patch_margin,
            patch_margin:-patch_margin,
        ]
        s2_rgb = s2_data[
            samp_idx,
            [2, 1, 0],
            patch_margin:-patch_margin,
            patch_margin:-patch_margin,
        ]
        s2_nes = s2_data[
            samp_idx,
            [3, 5, 8],
            patch_margin:-patch_margin,
            patch_margin:-patch_margin,
        ]
        return s1_asc, s1_desc, s2_rgb, s2_nes

    def prepare_lat(
        self, pred: torch.Tensor, samp_idx: int, patch_margin: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare latents for display"""
        lat_s1mu_0 = pred[
            samp_idx,
            [0, 0, 0],
            patch_margin:-patch_margin,
            patch_margin:-patch_margin,
        ]
        lat_s1mu_1 = pred[
            samp_idx,
            [1, 1, 1],
            patch_margin:-patch_margin,
            patch_margin:-patch_margin,
        ]
        lat_s1mu_2 = pred[
            samp_idx,
            [2, 2, 2],
            patch_margin:-patch_margin,
            patch_margin:-patch_margin,
        ]
        lat_s1mu_3 = pred[
            samp_idx,
            [3, 3, 3],
            patch_margin:-patch_margin,
            patch_margin:-patch_margin,
        ]
        return lat_s1mu_0, lat_s1mu_1, lat_s1mu_2, lat_s1mu_3

    # def prepare_patches_grid(
    #     self,
    #     samp_idx: int,
    #     sent_data: torch.Tensor | dict[str, torch.Tensor],
    #     pred: torch.Tensor,
    #     gt: torch.Tensor,
    #     dates: np.ndarray | dict[str, np.ndarray],
    # ) -> tuple[
    #     tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    #     tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    #     tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    #     tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    #     tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    #     tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    #     tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    # ]:
    #     """Generate the patches for the visualization grid
    #     Select the appropriate channels and crop with the patch margin.
    #     """
    #
    #     (s1_data, s2_data) = sent_data
    #     inputs = self.prepare_s1_s2(s1_data, s2_data, samp_idx, patch_margin)
    #
    #     preds = self.prepare_s1_s2(
    #         pred.encs1_mu_decs1, pred.encs2_mu_decs2, samp_idx, patch_margin
    #     )
    #
    #     preds_cross = self.prepare_s1_s2(
    #         pred.encs2_mu_decs1, pred.encs1_mu_decs2, samp_idx, patch_margin
    #     )
    #
    #     lat_s1_mu = self.prepare_lat(pred.latent.sen1.mean, samp_idx, patch_margin)
    #
    #     lat_s2_mu = self.prepare_lat(pred.latent.sen2.mean, samp_idx, patch_margin)
    #
    #     lat_s1_logvar = self.prepare_lat(
    #         pred.latent.sen1.logvar, samp_idx, patch_margin
    #     )
    #
    #     lat_s2_logvar = self.prepare_lat(
    #         pred.latent.sen2.logvar, samp_idx, patch_margin
    #     )
    #
    #     return (
    #         inputs,
    #         preds,
    #         preds_cross,
    #         lat_s1_mu,
    #         lat_s2_mu,
    #         lat_s1_logvar,
    #         lat_s2_logvar,
    #     )

    def render_row(
        self,
        imcol: tuple[torch.Tensor, ...],
        j: int,
        d_min_max: tuple[tuple[float, float], ...],
        last_rgb_row: int = 3,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Render a row of the image grid"""
        # calculate render the reconstruction use the same
        # dmin and dmax as the input
        if 0 < j < last_rgb_row:
            render_j0, _, _ = rgb_render(
                imcol[0].cpu().detach().numpy(),
                dmin=d_min_max[0][0],
                dmax=d_min_max[0][1],
            )
            render_j1, _, _ = rgb_render(
                imcol[1].cpu().detach().numpy(),
                dmin=d_min_max[1][0],
                dmax=d_min_max[1][1],
            )
            render_j2, _, _ = rgb_render(
                imcol[2].cpu().detach().numpy(),
                dmin=d_min_max[2][0],
                dmax=d_min_max[2][1],
            )
            render_j3, _, _ = rgb_render(
                imcol[3].cpu().detach().numpy(),
                dmin=d_min_max[3][0],
                dmax=d_min_max[3][1],
            )
        else:
            render_j0, _, _ = rgb_render(imcol[0].cpu().detach().numpy())
            render_j1, _, _ = rgb_render(imcol[1].cpu().detach().numpy())
            render_j2, _, _ = rgb_render(imcol[2].cpu().detach().numpy())
            render_j3, _, _ = rgb_render(imcol[3].cpu().detach().numpy())

        return render_j0, render_j1, render_j2, render_j3

    def generate_figure_for_sample(
        self,
        img_columns: list[tuple[torch.Tensor, ...]],
        d_min_max: tuple[tuple[np.array, np.array], ...],
    ) -> plt.Figure:
        """Generate figure for sample"""
        plt.close()
        fig, axes = plt.subplots(
            nrows=len(img_columns),
            ncols=4,
            sharex=True,
            sharey=True,
            figsize=((len(img_columns) + 1) * 2.5, 16),
        )
        fig.suptitle("Reconstructions Inspection", fontsize=20)

        plt.subplots_adjust(
            left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4
        )

        for j, imcol in enumerate(img_columns):
            (render_j0, render_j1, render_j2, render_j3) = self.render_row(
                imcol, j, d_min_max
            )

            # apply render
            axes[j, 0].imshow(render_j0, interpolation="bicubic")
            axes[j, 0].set_title(self.labels_list[j][0])
            axes[j, 1].imshow(render_j1, interpolation="bicubic")
            axes[j, 1].set_title(self.labels_list[j][1])
            axes[j, 2].imshow(render_j2, interpolation="bicubic")
            axes[j, 2].set_title(self.labels_list[j][2])
            axes[j, 3].imshow(render_j3, interpolation="bicubic")
            axes[j, 3].set_title(self.labels_list[j][3])
        return fig

    # def save_image_for_sample(
    #     self,
    #     img_columns: list[
    #         tuple[
    #             torch.Tensor,
    #             torch.Tensor,
    #             torch.Tensor,
    #             torch.Tensor,
    #         ]
    #     ],
    #     image_name: Path,
    # ) -> None:
    #     """Save the image grid for a sample"""
    #     # apply rgb render to the input to compute the min and max
    #     min_max = []
    #     for col in range(len(img_columns[0])):
    #         if col in (2, 3) or (img_columns[0][col].sum() != 0 and col in (0, 1)):
    #             _, d_min, d_max =
    #             rgb_render(img_columns[0][col].cpu().detach().numpy())
    #         else:  # in case an orbit is missing we use preds for normalization
    #             _, d_min, d_max =
    #             rgb_render(img_columns[1][col].cpu().detach().numpy())
    #         min_max.append((d_min, d_max))
    #
    #     fig = self.generate_figure_for_sample(
    #         img_columns,
    #         tuple(min_max),
    #     )
    #
    #     fig.savefig(image_name)

    def find_dates(
        self, dates: torch.Tensor | dict[str, torch.Tensor]
    ) -> tuple[np.array, np.array] | dict[str, tuple[np.array, np.array]]:
        """
        Transform dates from tensor integer to YYYY-MM-DD format.
        Select one date per chosen month to visualize
        """
        if type(dates) is dict:
            return {s: self.find_dates(d) for s, d in dates.items()}
        true_dates = back_to_date(dates)
        selected_dates = ["2018-10", "2019-03", "2018-07", "2018-11"]
        return [
            dates.index[pd.to_datetime(true_dates).dt.strftime("%Y-%m") == d].to_list()[
                0
            ]
            for d in selected_dates
        ], true_dates

    def save_image_grid(
        self,
        sen_data: torch.Tensor | dict[str, torch.Tensor],
        dates: torch.Tensor | dict[str, torch.Tensor],
        pred: torch.Tensor,
        gt: torch.Tensor,
        sample: SampleInfo,
    ) -> None:
        """Generate the matplotlib figure and save it to a file"""
        for samp_idx in range(
            0,
            sample.batch_size,
            sample.batch_size // min(sample.batch_size, self.n_samples),
        ):
            sel_index, sel_dates = self.find_dates(dates)
            if type(sen_data) is dict:
                sen_data_sel = {s: v[sel_index[s]] for s, v in sel_index.items()}
            else:
                sen_data_sel = sen_data[sel_index]
            # img_columns = list(
            #     self.prepare_patches_grid(samp_idx, sen_data_sel, pred, gt, sel_dates)
            # )
            # # export name
            # image_basename = (
            #     f"Pastis_UTAE_val_ep_{sample.current_epoch:03}_samp_"
            #     f"{samp_idx:03}_{sample.batch_idx}"
            # )
            # image_name = Path(f"{self.save_dir}/{image_basename}.png")
            # if not image_name.is_file() or sample.current_epoch == 0:
            #     self.save_image_for_sample(img_columns, image_name)

    @staticmethod
    def extract_crop(
        data: torch.Tensor, idx: int, bands: list[int] | int, margin: int
    ) -> torch.Tensor:
        """Return a cropped patch with a band subset from the whole batch"""
        return data[
            idx,
            bands,
            margin:-margin,
            margin:-margin,
        ]

    def to_flat_vector(
        self, data: torch.Tensor, idx: int, bands: list[int] | int, margin: int
    ) -> np.ndarray:
        """Select a patch from the batch, crop the center, select some
        bands and return a 1D vector with the pixels"""
        result: np.ndarray = (
            self.extract_crop(data, idx, bands, margin).flatten().cpu().detach().numpy()
        )
        return result

    def save_latent_scatterplots(
        self,
        pred: S1S2VAEOutput | S1S2VAEOutputPreExperts,
        samp_idx: int,
        sample: SampleInfo,
    ) -> None:
        """Save the PNG image of the scatterplots of the latent space of
        a sample of the batch"""
        image_basename = (
            f"MMDC_val_ep_{sample.current_epoch:03}_scat_latent_"
            f"{samp_idx:03}_{sample.batch_idx}"
        )
        image_name = Path(f"{self.save_dir}/{image_basename}.png")
        if not image_name.is_file() or sample.current_epoch == 0:
            lat_s1 = [
                self.to_flat_vector(
                    pred.latent.sen1.mean, samp_idx, i, sample.patch_margin
                )
                for i in range(4)
            ]
            lat_s2 = [
                self.to_flat_vector(
                    pred.latent.sen2.mean, samp_idx, i, sample.patch_margin
                )
                for i in range(4)
            ]

            plt.close()
            fig, axes = plt.subplots(
                nrows=2, ncols=2, sharex=False, sharey=False, figsize=(20, 16)
            )
            fig.suptitle(f"Latent {sample.current_epoch = }", fontsize=20)
            for i, axis in enumerate(axes.flatten()):
                axis.set_title(f"Latent {i}")
                axis.set_xlabel("S1")
                axis.set_ylabel("S2")
                axis.text(
                    0.01,
                    0.99,
                    f"Corr={np.round(np.corrcoef(lat_s1[i], lat_s2[i]), 2)[0, 1]}",
                    ha="left",
                    va="top",
                    transform=axis.transAxes,
                )
                axis.scatter(lat_s1[i], lat_s2[i])
                axis.plot(lat_s1[i], lat_s1[i])
            fig.savefig(image_name)

    # def plot_scatter(
    #     self,
    #     axis: plt.axis,
    #     label: str,
    #     x_data: np.ndarray,
    #     y_data: np.ndarray,
    # ) -> None:
    #     """Scatter plot with regression line and label"""
    #     axis.set_title(label)
    #     axis.scatter(x_data, y_data)
    #     axis.plot(x_data, x_data)
    #
    # def plot_correlation(
    #     self,
    #     image_name: Path,
    #     current_epoch: int,
    #     datasets: CorrelationData,
    # ):
    #     """Scatterplots between (cross-)reconstructions and target data"""
    #     bands = list(datasets.data.keys())
    #     satellite = "S1" if bands[0] == self.keys_s1[0] else "S2"
    #     nrows = 4 if datasets.cross is not None else 2
    #     ncols = int(len(bands) / 2)
    #     plt.close()
    #     fig, axes = plt.subplots(
    #         nrows=nrows,
    #         ncols=ncols,
    #         sharex=False,
    #         sharey=False,
    #         figsize=(20, nrows * 3.5 + 2),
    #     )
    #     fig.suptitle(f"{satellite} {current_epoch = }", fontsize=20)
    #     for i in range(nrows):
    #         for j in range(ncols):
    #             if i < 2:
    #                 band = bands[i * ncols + j]
    #                 self.plot_scatter(
    #                     axes[i, j],
    #                     f"{satellite} {band.upper()} Pred",
    #                     datasets.data[band],
    #                     datasets.pred[band],
    #                 )
    #             else:
    #                 band = bands[(i - 2) * ncols + j]
    #                 self.plot_scatter(
    #                     axes[i, j],
    #                     f"{satellite} {band.upper()} Cross",
    #                     datasets.data[band],
    #                     datasets.cross[band],
    #                 )
    #     fig.savefig(image_name)

    def on_validation_batch_end(  # pylint: disable=too-many-arguments
        self,
        trainer: pl.trainer.Trainer,
        pl_module: PastisUTAE | MMDCPastisEncodedUTAE,
        outputs: Any,
        batch: torch.Tensor,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Method called from the validation loop"""
        model = pl_module.model
        # ref_date = (
        #     trainer.datamodule.reference_date
        # )  # TODO check it later and integrate

        if batch_idx < 2:  # TODO: Add a constraint with nb of images per epoch
            (x, dates), gt = batch

            pred: torch.Tensor = model.predict(x, dates)

            # s2_x, s1_x = debatch.s2_x.nan_to_num(), debatch.s1_x.nan_to_num()
            batch_size = (
                x.shape[0] if type(x) is torch.Tensor else x[list(x.keys())[0]].shape[0]
            )

            self.save_image_grid(
                x,
                dates,
                pred,
                gt,
                SampleInfo(batch_idx, batch_size, trainer.current_epoch),
            )
