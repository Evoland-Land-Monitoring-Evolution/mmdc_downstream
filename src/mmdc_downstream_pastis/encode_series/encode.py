#!/usr/bin/env python3
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from mmdc_singledate.models.datatypes import VAELatentSpace

from mmdc_downstream_pastis.datamodule.pastis_oe import PastisDataModule
from mmdc_downstream_pastis.utils.utils import MMDCPartialBatch

log = logging.getLogger(__name__)


def back_to_date(days_int, ref_date):
    return [
        pd.to_datetime(
            [
                pd.Timedelta(dd, "d") + pd.to_datetime(ref_date)
                for dd in days_int.numpy()[d]
            ]
        )
        for d in range(len(days_int))
    ]


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


def fill_batch_s1(
    batch_s1: MMDCPartialBatch, asc_ind, desc_ind, batch_asc, batch_desc
) -> MMDCPartialBatch:
    only_in_desc = desc_ind[~np.in1d(desc_ind, asc_ind)]

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
        batch_s1.meteo[:, only_in_desc, :, :, :] = batch_desc.meteo[
            :, ~np.in1d(desc_ind, asc_ind)
        ]
        if len(asc_ind) == 0:
            batch_s1.dem = batch_desc.dem

    return batch_s1


def create_s1(
    batch_asc: MMDCPartialBatch,
    batch_desc: MMDCPartialBatch,
    days_asc: np.ndarray,
    days_desc: np.ndarray,
) -> tuple[MMDCPartialBatch, np.ndarray, np.ndarray, np.ndarray]:
    bs, t, ch, h, w = batch_asc.img.shape
    if bs == 1:
        days_asc, days_desc = days_asc[0], days_desc[0]
        s1_asc_desc = pd.merge_asof(
            pd.DatetimeIndex(days_asc).to_frame(name="asc"),
            pd.DatetimeIndex(days_desc).to_frame(name="desc"),
            left_index=True,
            right_index=True,
            tolerance=pd.Timedelta(1, "D"),
            direction="nearest",
        )
        s1_desc_asc = pd.merge_asof(
            pd.DatetimeIndex(days_desc).to_frame(name="desc"),
            pd.DatetimeIndex(days_asc).to_frame(name="asc"),
            left_index=True,
            right_index=True,
            tolerance=pd.Timedelta(1, "D"),
            direction="nearest",
        )

        df_merged_days = (
            pd.concat([s1_asc_desc, s1_desc_asc]).sort_index().drop_duplicates()
        )
        days_s1 = df_merged_days.index.values
        log.info(days_s1)
        df_merged_days = df_merged_days.reset_index()
        asc_ind = df_merged_days.index[df_merged_days["asc"].notnull()].values
        desc_ind = df_merged_days.index[df_merged_days["desc"].notnull()].values

        batch_s1 = MMDCPartialBatch.create_empty_s1_with_shape(
            batch_size=bs,
            times=len(days_s1.reshape(-1)),
            height=h,
            width=w,
            nb_channels=6,
            nb_mask=2,
            nb_angles=2,
        )
        batch_s1 = fill_batch_s1(batch_s1, asc_ind, desc_ind, batch_asc, batch_desc)

        return batch_s1, asc_ind, desc_ind, days_s1


def encode_one_batch(
    mmdc_model, batch_dict, ref_date=None
) -> tuple[
    VAELatentSpace,
    VAELatentSpace,
    VAELatentSpace,
    VAELatentSpace,
    torch.Tensor,
    np.ndarray,
]:
    batch_asc = MMDCPartialBatch.fill_from(batch_dict["S1_ASC"], "S1")
    batch_desc = MMDCPartialBatch.fill_from(batch_dict["S1_DESC"], "S1")
    days_asc = batch_dict["S1_ASC"].true_doy
    days_desc = batch_dict["S1_DESC"].true_doy
    if ref_date is not None:
        days_asc = back_to_date(days_asc, ref_date)
        days_desc = back_to_date(days_desc, ref_date)
    batch_s1, asc_ind, desc_ind, days_s1 = create_s1(
        batch_asc, batch_desc, days_asc, days_desc
    )

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

    batch_s1_mask = (batch_s1.mask.sum(2) > 1).unsqueeze(2)

    return latents1, latents1_asc, latents1_desc, latents2, batch_s1_mask, days_s1


def encode_series(mmdc_model, dataset_path_oe, dataset_path_pastis, sats):
    dm = build_dm(dataset_path_oe, dataset_path_pastis, sats)

    dataset = dm.instanciate_dataset(fold=None)
    loader = dm.instanciate_data_loader(dataset, shuffle=False, drop_last=False)
    for batch_dict, target, mask, id_patch in loader:
        print(id_patch)
        (
            latents1,
            latents1_asc,
            latents1_desc,
            latents2,
            batch_s1_mask,
            days_s1,
        ) = encode_one_batch(mmdc_model, batch_dict, ref_date=dm.reference_date)
