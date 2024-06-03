#!/usr/bin/env python3
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import logging
import os
from pathlib import Path

import torch
from einops import rearrange

from mmdc_downstream_pastis.datamodule.pastis_oe import PastisOEDataModule

log = logging.getLogger(__name__)

MAX_SAMPLES_QUANTILE = 16_000_000


def build_dm(
    dataset_path_oe: str | Path,
    dataset_path_pastis: str | Path,
    sats: list[str],
    crop_size: int = None,
    pad_value: int = -9999,
) -> PastisOEDataModule:
    """Builds datamodule"""
    return PastisOEDataModule(
        dataset_path_oe=dataset_path_oe,
        dataset_path_pastis=dataset_path_pastis,
        folds=None,
        sats=sats,
        task="semantic",
        batch_size=1,
        crop_size=crop_size,
        pad_value=pad_value,
    )


def get_quantile(
    data: torch.Tensor, quant: torch.Tensor = torch.Tensor([0.05, 0.5, 0.95])
) -> torch.Tensor:
    nbpixels = len(data)
    if nbpixels > MAX_SAMPLES_QUANTILE:
        return data[torch.randperm(16_000_000)].quantile(
            torch.tensor(quant, dtype=torch.float), dim=0
        )
    return data.quantile(torch.tensor(quant, dtype=torch.float), dim=0)


def compute_stats(
    dataset_path_oe: str | Path,
    dataset_path_pastis: str | Path,
    sats: list[str],
    pad_value: int = -9999,
):
    """Encode PASTIS SITS into S1 and S2 latent embeddings"""

    dem_min_med_max = None

    log.info("Loader is ready")
    for sat in sats:
        sat_stats = {}

        dm = build_dm(
            dataset_path_oe,
            dataset_path_pastis,
            [sat],
            crop_size=64,
            pad_value=pad_value,
        )

        dataset = dm.instanciate_dataset(fold=None)
        loader = dm.instanciate_data_loader(dataset, shuffle=False, drop_last=False)

        meteo_stats = torch.empty(0, 8)
        dem_stats = torch.empty(0, 4)

        if sat == "S2":
            img_stats = torch.empty(0, 10)
        else:
            img_stats = torch.empty(0, 3)

        for batch in loader:
            log.info(batch.id_patch)

            sits = batch.sits[sat].sits

            meteo = rearrange(sits.meteo, "b t (d c) h w -> (b t d h w) c", c=8, d=6)
            mask_meteo = (torch.isnan(meteo).sum(1) + (meteo == pad_value).sum(1)) > 0
            meteo = meteo[~mask_meteo]
            indices = torch.randperm(meteo.shape[0])
            indices = indices[: int(0.1 * len(indices))]
            meteo_stats = torch.concat(
                [meteo_stats, meteo[indices[: int(0.2 * len(indices))]]], 0
            )

            if dem_min_med_max is None:
                dem = rearrange(sits.dem, "b c h w -> (b h w) c")
                mask_dem = (torch.isnan(dem).sum(1) + (dem == pad_value).sum(1)) > 0
                dem = dem[~mask_dem]
                dem_stats = torch.concat([dem_stats, dem], 0)

            img = rearrange(sits.data.img, "b t c h w -> (b t h w) c")
            mask = rearrange(sits.data.mask, "b t c h w -> (b t c h w)").bool()
            img = img[~mask]
            mask_img = (torch.isnan(img).sum(1) + (img == pad_value).sum(1)) > 0
            img = img[~mask_img]
            indices = torch.randperm(img.shape[0])
            img_stats = torch.concat(
                [img_stats, img[indices[: int(0.2 * len(indices))]]], 0
            )

        sat_stats["img"] = get_quantile(img_stats)

        sat_stats["meteo"] = get_quantile(meteo_stats)

        if dem_min_med_max is None:
            dem_min_med_max = get_quantile(dem_stats)

        sat_stats["dem"] = dem_min_med_max

        torch.save(sat_stats, os.path.join(dataset_path_oe, f"stats_{sat}.pt"))


# dataset_path_oe = "/home/kalinichevae/scratch_jeanzay/scratch_data/Pastis_OE"
# dataset_path_pastis = "/home/kalinichevae/scratch_jeanzay/scratch_data/Pastis"

dataset_path_oe = f"{os.environ['SCRATCH']}/scratch_data/Pastis_OE_corr"
dataset_path_pastis = f"{os.environ['SCRATCH']}/scratch_data/Pastis"


sats = ["S1_DESC", "S1_ASC", "S2"]

compute_stats(dataset_path_oe, dataset_path_pastis, sats)
