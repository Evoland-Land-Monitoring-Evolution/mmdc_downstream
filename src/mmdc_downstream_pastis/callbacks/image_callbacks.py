#!/usr/bin/env python3
# Copyright: (c) 2022 CESBIO / Centre National d'Etudes Spatiales
""" Lightning image callbacks """
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from einops import rearrange
from matplotlib import colors
from pastis_eo.dataclass import PASTISItem
from pytorch_lightning.callbacks import Callback
from sensorsio.utils import rgb_render
from torch import Tensor

from mmdc_downstream_pastis.datamodule.datatypes import BatchInputUTAE
from mmdc_downstream_pastis.encode_series.encode import back_to_date
from mmdc_downstream_pastis.models.lightning.pastis_utae_semantic import PastisUTAE

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


LABEL_PASTIS = {
    0: "Background",
    1: "Meadow",
    2: "Soft winter wheat",
    3: "Corn",
    4: "Winter barley",
    5: "Winter rapeseed",
    6: "Spring barley",
    7: "Sunflower",
    8: "Grapevine",
    9: "Beet",
    10: "Winter triticale",
    11: "Winter durum wheat",
    12: "Fruits,  vegetables, flowers",
    13: "Potatoes",
    14: "Leguminous fodder",
    15: "Soybeans",
    16: "Orchard",
    17: "Mixed cereal",
    18: "Sorghum",
}

COLORS_PASTIS = {
    0: "black",
    1: "limegreen",
    2: "lightcyan",
    3: "gold",
    4: "deepskyblue",
    5: "turquoise",
    6: "floralwhite",
    7: "yellow",
    8: "purple",
    9: "darkred",
    10: "slategray",
    11: "dodgerblue",
    12: "lightcoral",
    13: "seashell",
    14: "olive",
    15: "bisque",
    16: "greenyellow",
    17: "grey",
    18: "lightpink",
}


class PastisCallback(Callback):
    """
    Callback to inspect the reconstruction image
    """

    def __init__(
        self,
        save_dir: str,
        n_samples: int = 6,
    ):
        self.save_dir = save_dir
        self.n_samples = n_samples
        self.pastis_path = f"{os.environ['SCRATCH']}/scratch_data/Pastis_OE_corr"
        self.selected_months = ["2018-10", "2019-02", "2019-05", "2019-08", "2019-11"]

    def prepare_encoded(
        self, data: torch.Tensor, samp_idx: int, sel_index: int
    ) -> tuple[np.array, np.array]:
        """Prepare the patches for the S1 and S2 data"""
        nb_feat = int(data.shape[2] / 2)
        lat_mu = self.render_row(data[samp_idx, sel_index, :3, 32:-32, 32:-32])
        if nb_feat <= 3:
            lat_logvar = self.render_row(data[samp_idx, sel_index, 3:, 32:-32, 32:-32])
        else:
            lat_logvar = self.render_row(
                data[samp_idx, sel_index, nb_feat : nb_feat + 3, 32:-32, 32:-32]
            )
        return lat_mu, lat_logvar

    def prepare_patches_grid(
        self,
        samp_idx: int,
        sent_data: torch.Tensor | dict[str, torch.Tensor],
        pred: torch.Tensor,
        gt: torch.Tensor,
        sel_index: np.ndarray | dict[str, np.ndarray],
    ) -> tuple[dict | tuple[Tensor, Tensor], Any, Any]:
        """Generate the patches for the visualization grid
        Select the appropriate channels and crop with the patch margin.
        """

        if type(sent_data) is dict:
            latent = {
                sat: self.prepare_encoded(sent_data[sat], samp_idx, sel_index[sat])
                for sat in sent_data
            }
        else:
            latent = self.prepare_encoded(sent_data, samp_idx, sel_index)

        pred = pred[samp_idx, 32:-32, 32:-32].cpu().detach().numpy()
        gt = gt[samp_idx, 32:-32, 32:-32].cpu().detach().numpy()

        return (
            latent,
            pred,
            gt,
        )

    def render_row(self, images: torch.tensor, mask: torch.tensor = None) -> np.array:
        """Render a SITS row of the image grid.
        One row represents selected images from one SITS;
        We find min/max values for those dates band_wise ignoring masked values
        """
        # Normalize SITS using the same dmin, dmax per series
        d_min, d_max = np.array([100000, 100000, 100000]), np.array(
            [-100000, -100000, -100000]
        )
        rearr = rearrange(images, "t c h w -> t (h w) c")
        if mask is not None:
            mask = rearrange(mask, "t h w -> t (h w)")
        for im in range(len(images)):
            if mask is not None:
                if mask[im].sum() < int(images[im].shape[-1] ** 2 * 0.5):
                    d_min_ = np.percentile(
                        rearr[im]
                        .cpu()
                        .detach()
                        .numpy()[~mask[im].cpu().detach().numpy()],
                        q=2,
                        axis=0,
                    )

                    d_max_ = np.percentile(
                        rearr[im]
                        .cpu()
                        .detach()
                        .numpy()[~mask[im].cpu().detach().numpy()],
                        q=98,
                        axis=0,
                    )
                else:
                    d_min_, d_max_ = np.array([100000, 100000, 100000]), np.array(
                        [-100000, -100000, -100000]
                    )
            else:
                d_min_ = np.percentile(rearr[im].cpu().detach().numpy(), q=2, axis=0)
                d_max_ = np.percentile(rearr[im].cpu().detach().numpy(), q=98, axis=0)

            d_min = np.min(np.stack([d_min, d_min_]), axis=0)
            d_max = np.max(np.stack([d_max, d_max_]), axis=0)
        return np.array(
            [
                rgb_render(
                    images[im].cpu().detach().numpy(),
                    dmin=d_min,
                    dmax=d_max,
                )[0]
                for im in range(len(images))
            ]
        )

    def save_image_for_sample(
        self,
        img_columns: tuple[dict | tuple[Tensor, Tensor], Any, Any],
        original_patches: np.ndarray | dict[str, Any],
        image_name: Path,
        dates: dict[str, np.ndarray] | np.ndarray,
        sat: str | None = None,
    ) -> None:
        """Save the image grid for a sample"""
        latent, pred, gt = img_columns
        plt.close()
        if type(latent) is not dict:
            nb_rows = 3
            if sat == "S1":
                nb_rows += 2
            if sat == "S2":
                nb_rows += 1
        else:
            nb_rows = 1 + len(latent[list(latent.keys())[0]]) * 2
            if "S1" in list(latent.keys()):
                nb_rows += 2
            if "S2" in list(latent.keys()):
                nb_rows += 1
        fig, axes = plt.subplots(
            nrows=nb_rows,
            ncols=len(self.selected_months),
            sharex=True,
            sharey=True,
            figsize=(len(self.selected_months) * 5, nb_rows * 5),
        )
        fig.suptitle("Crop Classification Inspection", fontsize=20)

        plt.subplots_adjust(
            left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4
        )
        row_counter = 0
        if type(latent) is dict:
            for idx_sat, sat in enumerate(latent):
                date = dates[sat]
                latent_mu, latent_logvar = latent[sat]
                if sat == "S2":
                    for im in range(len(latent_mu)):
                        axes[row_counter][im].imshow(original_patches[sat][im])
                        axes[row_counter][im].set_title(
                            f"{sat.upper()}, date={date[im]}"
                        )
                    row_counter += 1
                    for im in range(len(latent_mu)):
                        axes[row_counter][im].imshow(latent_mu[im])
                    row_counter += 1
                    for im in range(len(latent_mu)):
                        axes[row_counter][im].imshow(latent_logvar[im])
                    row_counter += 1

                else:
                    for sat_ in ("S1_ASC", "S1_DESC"):
                        for im in range(len(latent_mu)):
                            axes[row_counter][im].imshow(
                                original_patches[sat][sat_][im]
                            )
                            axes[row_counter][im].set_title(
                                f"{sat_.upper()}, date={date[im]}"
                            )
                        row_counter += 1
                    for im in range(len(latent_mu)):
                        axes[row_counter][im].imshow(latent_mu[im])
                    row_counter += 1
                    for im in range(len(latent_mu)):
                        axes[row_counter][im].imshow(latent_logvar[im])
                    row_counter += 1
        else:
            date = dates
            latent_mu, latent_logvar = latent

            if sat == "S1":
                for sat_ in ("S1_ASC", "S1_DESC"):
                    for im in range(len(latent_mu)):
                        axes[row_counter][im].imshow(original_patches[sat_][im])
                        axes[row_counter][im].set_title(
                            f"{sat_.upper()}, date={date[im]}"
                        )
                    row_counter += 1
            else:
                for im in range(len(latent_mu)):
                    axes[row_counter][im].imshow(original_patches[im])
                    axes[row_counter][im].set_title(f"{sat.upper()}, date={date[im]}")
                row_counter += 1
            for im in range(len(latent_mu)):
                axes[row_counter][im].imshow(latent_mu[im])
                axes[row_counter][im].set_title(f"{sat.upper()}, date={date[im]}")
                axes[row_counter + 1][im].imshow(latent_logvar[im])

        colormap = colors.ListedColormap([color for color in COLORS_PASTIS.values()])

        axes[-1][1].imshow(pred, vmin=0, vmax=len(COLORS_PASTIS), cmap=colormap)
        axes[-1][1].set_title("Pred")
        axes[-1][2].imshow(gt, vmin=0, vmax=len(COLORS_PASTIS), cmap=colormap)
        axes[-1][2].set_title("GT")
        fig.savefig(image_name)

    def find_dates(
        self, dates: torch.Tensor | dict[str, torch.Tensor], samp_idx: int
    ) -> tuple[np.array, np.array] | tuple[dict[np.array]]:
        """
        Transform dates from tensor integer to YYYY-MM-DD format.
        Select one date per chosen month to visualize
        """
        if type(dates) is dict:
            res = {s: self.find_dates(d, samp_idx) for s, d in dates.items()}
            return {s: v[0] for s, v in res.items()}, {s: v[1] for s, v in res.items()}

        true_dates = back_to_date(dates)[samp_idx]

        sel_index = []
        for d in self.selected_months:
            try:
                sel_index.append(
                    random.choice(
                        pd.to_datetime(true_dates)
                        .to_frame(index=False)
                        .loc[pd.to_datetime(true_dates).strftime("%Y-%m") == d]
                        .index.to_list()
                    )
                )
            except IndexError:
                try:
                    sel_index.append(
                        random.choice(
                            pd.to_datetime(true_dates)
                            .to_frame(index=False)
                            .loc[
                                (
                                    pd.to_datetime(true_dates) + pd.DateOffset(months=1)
                                ).strftime("%Y-%m")
                                == d
                            ]
                            .index.to_list()
                        )
                    )
                except IndexError:
                    sel_index.append(
                        random.choice(
                            pd.to_datetime(true_dates)
                            .to_frame(index=False)
                            .loc[
                                (
                                    pd.to_datetime(true_dates) - pd.DateOffset(months=1)
                                ).strftime("%Y-%m")
                                == d
                            ]
                            .index.to_list()
                        )
                    )
        return (
            sel_index,
            pd.to_datetime(true_dates).strftime("%Y-%m-%d").values[sel_index],
        )

    def open_and_prepare_original(
        self,
        sel_index: np.ndarray | dict[str, np.ndarray],
        id_patch: int,
        sel_dates: np.ndarray | dict[str, np.ndarray],
        satellite: str | list,
    ) -> np.ndarray | dict[str, Any]:
        """Open original Sentinel images and render them"""
        if type(satellite) is list:
            return {
                sat: self.open_and_prepare_original(
                    sel_index[sat], id_patch, sel_dates[sat], sat
                )
                for sat in satellite
            }
        if satellite == "S1":
            s1_asc: PASTISItem = torch.load(
                os.path.join(
                    self.pastis_path,
                    "S1_ASC",
                    f"S1_ASC_{id_patch}.pt",
                ),
                map_location="cpu",
            )
            s1_desc = torch.load(
                os.path.join(
                    self.pastis_path,
                    "S1_DESC",
                    f"S1_DESC_{id_patch}.pt",
                ),
                map_location="cpu",
            )

            img_asc = s1_asc.sits[:, :, 32:-32, 32:-32]
            dates_asc = [str(t)[:10] for t in s1_asc.dates_dt.values[:, 0]]
            img_desc = s1_desc.sits[:, :, 32:-32, 32:-32]
            dates_desc = [str(t)[:10] for t in s1_desc.dates_dt.values[:, 0]]

            s1_sel = torch.zeros(2, len(sel_dates), 3, 64, 64)
            s1_mask = torch.ones(2, len(sel_dates), 3, 64, 64)

            for d, sel_date in enumerate(sel_dates):
                if sel_date in dates_asc:
                    s1_sel[0, d] = img_asc[[d == sel_date for d in dates_asc]]
                    s1_mask[0, d][~torch.isnan(s1_sel[0, d])] = 0
                if sel_date in dates_desc:
                    s1_sel[1, d] = img_desc[[d == sel_date for d in dates_desc]]
                    s1_mask[1, d][~torch.isnan(s1_sel[1, d])] = 0
            s1_mask = s1_mask.sum(2) > 0
            return {
                "S1_ASC": self.render_row(s1_sel[0].nan_to_num(), s1_mask[0]),
                "S1_DESC": self.render_row(s1_sel[1].nan_to_num(), s1_mask[1]),
            }
        else:
            sits = torch.load(
                os.path.join(
                    self.pastis_path,
                    satellite,
                    f"{satellite}_{id_patch}.pt",
                ),
                map_location="cpu",
            )
            img = sits.sits[sel_index]
            mask = sits.mask[sel_index]

            bands = [2, 1, 0] if satellite == "S2" else [0, 1, 2]

            return self.render_row(
                img[:, bands, 32:-32, 32:-32].nan_to_num(),
                mask[:, 0, 32:-32, 32:-32].bool(),
            )

    def save_image_grid(
        self,
        sen_data: torch.Tensor | dict[str, torch.Tensor],
        dates: torch.Tensor | dict[str, torch.Tensor],
        pred: torch.Tensor,
        gt: torch.Tensor,
        patches_id: torch.Tensor,
        sample: SampleInfo,
        sat: str | None = None,
    ) -> None:
        """Generate the matplotlib figure and save it to a file"""
        for samp_idx in range(
            0,
            sample.batch_size,
            sample.batch_size // min(sample.batch_size, self.n_samples),
        ):
            sel_index, sel_dates = self.find_dates(dates, samp_idx)

            prepared_patches = self.prepare_patches_grid(
                samp_idx, sen_data, pred, gt, sel_index
            )
            original_patches = self.open_and_prepare_original(
                sel_index,
                patches_id[samp_idx],
                sel_dates,
                satellite=sat if type(sen_data) is not dict else list(sen_data.keys()),
            )
            # export name
            image_basename = (
                f"Pastis_UTAE_val_ep_{sample.current_epoch:03}_samp_"
                f"{samp_idx:03}_{sample.batch_idx}"
            )
            image_name = Path(f"{self.save_dir}/{image_basename}.png")
            if not image_name.is_file() or sample.current_epoch == 0:
                self.save_image_for_sample(
                    prepared_patches, original_patches, image_name, sel_dates, sat
                )

    def on_validation_batch_end(  # pylint: disable=too-many-arguments
        self,
        trainer: pl.trainer.Trainer,
        pl_module: PastisUTAE,
        outputs: Any,
        batch: BatchInputUTAE,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Method called from the validation loop"""
        # ref_date = (
        #     trainer.datamodule.reference_date
        # )  # TODO check it later and integrate

        if trainer.current_epoch % 5 == 0:
            if batch_idx < 3:  # TODO: Add a constraint with nb of images per epoch
                pred: torch.Tensor = pl_module.predict(batch)

                x = batch.sits

                batch_size = (
                    x.shape[0]
                    if type(x) is torch.Tensor
                    else x[list(x.keys())[0]].shape[0]
                )
                pred[batch.gt_mask] = 0

                self.save_image_grid(
                    x,
                    batch.doy,
                    pred,
                    batch.gt,
                    batch.id_patch,
                    SampleInfo(batch_idx, batch_size, trainer.current_epoch),
                    sat=trainer.datamodule.sats[0] if x is not dict else None,
                )
