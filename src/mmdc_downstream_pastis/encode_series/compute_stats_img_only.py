#!/usr/bin/env python3
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import logging
import os
from pathlib import Path

import torch
from einops import rearrange

log = logging.getLogger(__name__)

MAX_SAMPLES_QUANTILE = 16_000_000


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
    sats: list[str],
    pad_value: int = -9999,
):
    """Encode PASTIS SITS into S1 and S2 latent embeddings"""

    for sat in sats:
        files = [
            f
            for f in os.listdir(os.path.join(dataset_path_oe, sat))
            if f.endswith(".pt")
        ]

        if sat == "S2":
            img_stats = torch.empty(0, 10)
        else:
            img_stats = torch.empty(0, 3)

        for file in files:
            img_all = torch.load(os.path.join(dataset_path_oe, sat, file))
            log.info(file)
            img = rearrange(img_all.sits, "t c h w -> (t h w) c")
            mask = rearrange(img_all.mask, "t c h w -> (t c h w)").bool()
            img = img[~mask]
            mask_img = (torch.isnan(img).sum(1) + (img == pad_value).sum(1)) > 0
            img = img[~mask_img]
            indices = torch.randperm(img.shape[0])
            img_stats = torch.concat(
                [img_stats, img[indices[: int(0.2 * len(indices))]]], 0
            )

        sat_stats = get_quantile(img_stats)

        torch.save(sat_stats, os.path.join(dataset_path_oe, f"stats_{sat}_img.pt"))


# dataset_path_oe = "/home/kalinichevae/scratch_jeanzay/scratch_data/Pastis_OE_corr"
# dataset_path_pastis = "/home/kalinichevae/scratch_jeanzay/scratch_data/Pastis"

dataset_path_oe = f"{os.environ['SCRATCH']}/scratch_data/Pastis_OE_corr"

# dataset_path_oe = "/home/kalinichevae/scratch_data/Pastis_OE"
# dataset_path_pastis = "/home/kalinichevae/scratch_data/Pastis"

sats = ["S2"]

compute_stats(dataset_path_oe, sats)
