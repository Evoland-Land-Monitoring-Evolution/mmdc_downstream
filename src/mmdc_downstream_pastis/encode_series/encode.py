#!/usr/bin/env python3
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales

from pathlib import Path

import numpy as np
import torch
from mmdc_singledate.models.datatypes import VAELatentSpace

from mmdc_downstream_pastis.datamodule.pastis_oe import PastisDataModule
from mmdc_downstream_pastis.utils.utils import MMDCPartialBatch


def build_dm(
    dataset_path_oe: str | Path, dataset_path_pastis: str | Path, sats: list[str]
) -> PastisDataModule:
    """Builds datamodule"""
    return PastisDataModule(
        dataset_path_oe=dataset_path_oe,
        dataset_path_pastis=dataset_path_pastis,
        folds=None,
        sats=sats,
        task="semantic",
        batch_size=1,
        max_len=0,
    )


def create_s1(batch_asc, batch_desc, days_asc, days_desc):
    bs, t, ch, h, w = batch_asc.img.shape
    if bs == 1:
        common_days = np.union1d(days_asc, days_desc)
        # torch.cat((a, b)).unique()

        _, asc_ind, _ = np.intersect1d(
            common_days, days_asc, assume_unique=True, return_indices=True
        )
        _, desc_ind, _ = np.intersect1d(
            common_days, days_desc, assume_unique=True, return_indices=True
        )

        batch_s1 = MMDCPartialBatch.create_empty_s1_with_shape(
            batch_size=bs,
            times=len(common_days),
            height=h,
            width=w,
            nb_channels=6,
            nb_mask=2,
            nb_angles=2,
        )
        if len(asc_ind) > 0:
            batch_s1.img[:, asc_ind, :3, :, :] = batch_asc.img
            batch_s1.angles[:, asc_ind, :1, :, :] = batch_asc.angles
            batch_s1.mask[:, asc_ind, :1, :, :] = batch_asc.mask
            batch_s1.meteo[:, asc_ind, :, :, :] = batch_asc.meteo
            batch_s1.dem = batch_asc.dem
        if len(desc_ind) > 0:
            batch_s1.img[:, desc_ind, 3:, :, :] = batch_desc.img
            batch_s1.angles[:, desc_ind, 1:, :, :] = batch_desc.angles
            batch_s1.mask[:, desc_ind, 1:, :, :] = batch_desc.mask
            batch_s1.meteo[:, desc_ind, :, :, :] = batch_desc.meteo
            if len(asc_ind) == 0:
                batch_s1.dem = batch_desc.dem

        return batch_s1, asc_ind, desc_ind


def encode_one_batch(
    mmdc_model, batch_dict
) -> tuple[VAELatentSpace, VAELatentSpace, VAELatentSpace, VAELatentSpace]:
    batch_asc = MMDCPartialBatch.fill_from(batch_dict["S1_ASC"], "S1")
    batch_desc = MMDCPartialBatch.fill_from(batch_dict["S1_DESC"], "S1")
    days_asc = batch_dict["S1_ASC"].true_doy
    days_desc = batch_dict["S1_DESC"].true_doy
    batch_s1, asc_ind, desc_ind = create_s1(batch_asc, batch_desc, days_asc, days_desc)

    print("batch_s1.img.shape", batch_s1.img.shape)
    print("batch_s1.angles.shape", batch_s1.angles.shape)
    print("batch_s1.meteo.shape", batch_s1.meteo.shape)
    print("batch_s1.dem.shape", batch_s1.dem.shape)

    if batch_s1.img.shape[1] >= 40 and batch_s1.img.shape[-1] >= 500:
        batch_s1_0 = batch_s1[:, : int(batch_s1.img.shape[1] / 2)]
        batch_s1_1 = batch_s1[:, int(batch_s1.img.shape[1] / 2) :]
        print("batch_s1.img.shape", batch_s1_0.img.shape)
        latents1_0 = mmdc_model.get_latent_s1_mmdc(
            batch_s1_0.to_device(mmdc_model.device)
        )
        latents1_1 = mmdc_model.get_latent_s1_mmdc(
            batch_s1_1.to_device(mmdc_model.device)
        )
        latents1 = VAELatentSpace(
            torch.cat([latents1_0.mean, latents1_1.mean], dim=1),
            torch.cat([latents1_0.logvar, latents1_1.logvar], dim=1),
        )
    else:
        latents1 = mmdc_model.get_latent_s1_mmdc(batch_s1.to_device(mmdc_model.device))

    latents1_asc = VAELatentSpace(
        latents1.mean[:, asc_ind], latents1.logvar[:, asc_ind]
    )
    latents1_desc = VAELatentSpace(
        latents1.mean[:, desc_ind], latents1.logvar[:, desc_ind]
    )

    # days_delta = 1643

    batch_s2 = MMDCPartialBatch.fill_from(batch_dict["S2"], "S2")

    if batch_s2.img.shape[1] >= 50 and batch_s2.img.shape[-1] >= 500:
        batch_s2_0 = batch_s2[:, : int(batch_s2.img.shape[1] / 2)]
        batch_s2_1 = batch_s2[:, int(batch_s2.img.shape[1] / 2) :]
        latents2_0 = mmdc_model.get_latent_s2_mmdc(
            batch_s2_0.to_device(mmdc_model.device)
        )
        latents2_1 = mmdc_model.get_latent_s2_mmdc(
            batch_s2_1.to_device(mmdc_model.device)
        )
        latents2 = VAELatentSpace(
            torch.cat([latents2_0.mean, latents2_1.mean], dim=1),
            torch.cat([latents2_0.logvar, latents2_1.logvar], dim=1),
        )
    else:
        latents2 = mmdc_model.get_latent_s2_mmdc(batch_s2.to_device(mmdc_model.device))
    return latents1, latents1_asc, latents1_desc, latents2


def encode_series(mmdc_model, dataset_path_oe, dataset_path_pastis, sats):
    dm = build_dm(dataset_path_oe, dataset_path_pastis, sats)

    dataset = dm.instanciate_dataset(fold=None)
    loader = dm.instanciate_data_loader(dataset, shuffle=False, drop_last=False)
    for batch_dict, target, mask, id_patch in loader:
        print(id_patch)
        latents1, latents1_asc, latents1_desc, latents2 = encode_one_batch(
            mmdc_model, batch_dict
        )
