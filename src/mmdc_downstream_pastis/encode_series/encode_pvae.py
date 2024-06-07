#!/usr/bin/env python3
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from mmdc_singledate.models.datatypes import VAELatentSpace

from mmdc_downstream_pastis.datamodule.pastis_oe import PastisOEDataModule
from mmdc_downstream_pastis.mmdc_model.model import PretrainedMMDCPastis
from mmdc_downstream_pastis.utils.utils import MMDCPartialBatch

log = logging.getLogger(__name__)


def back_to_date(
    days_int: torch.Tensor, ref_date: str = "2018-09-01"
) -> list[np.array]:
    """
    Go back from integral doy (number of days from ref date)
    to calendar doy
    """
    return [
        np.asarray(
            pd.to_datetime(
                [
                    pd.Timedelta(dd, "d") + pd.to_datetime(ref_date)
                    for dd in days_int.cpu().numpy()[d]
                ]
            )
        )
        for d in range(len(days_int))
    ]


def build_dm(
    dataset_path_oe: str | Path, dataset_path_pastis: str | Path, sats: list[str]
) -> PastisOEDataModule:
    """Builds datamodule"""
    return PastisOEDataModule(
        dataset_path_oe=dataset_path_oe,
        dataset_path_pastis=dataset_path_pastis,
        folds=None,
        sats=sats,
        task="semantic",
        batch_size=1,
        crop_size=None,
    )


def fill_batch_s1(
    batch_s1: MMDCPartialBatch, asc_ind, desc_ind, batch_asc, batch_desc
) -> MMDCPartialBatch:
    """
    Fill S1 batch with data from both orbits, according to each orbit available days
    """
    only_in_desc = desc_ind[~np.in1d(desc_ind, asc_ind)] if len(desc_ind) > 0 else []
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


def match_asc_desc_both_available(
    days_asc: np.ndarray, days_desc: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match Asc et Desc dates when both orbits are available.
    Tolerance is +-1 day
    """
    df_asc = pd.DatetimeIndex(days_asc).to_frame(name="asc")
    df_desc = pd.DatetimeIndex(days_desc).to_frame(name="desc")
    s1_asc_desc = pd.merge_asof(
        df_asc,
        df_desc,
        left_index=True,
        right_index=True,
        tolerance=pd.Timedelta(1, "D"),
        direction="nearest",
    )
    s1_desc_asc = pd.merge_asof(
        df_desc,
        df_asc,
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

    # Get indices of days of each orbit within all the available S1 dates
    asc_ind = df_merged_days.index[df_merged_days["asc"].notnull()].values
    desc_ind = df_merged_days.index[df_merged_days["desc"].notnull()].values
    return asc_ind, desc_ind, days_s1


def match_asc_desc(
    days_asc: np.ndarray, days_desc: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match asc and desc orbits to find co-ocurences
    Returns Asc and Desc dates indices within all available S1 dates
    and days_s1 - all available dates
    """
    # if both orbits are available for patch
    if len(days_asc) > 0 and len(days_desc) > 0:
        return match_asc_desc_both_available(days_asc, days_desc)
    # if only one orbit is available for patch
    else:
        if len(days_asc) > 0:
            asc_ind = np.arange(len(days_asc))
            days_s1 = days_asc
            desc_ind = []
        else:
            desc_ind = np.arange(len(days_desc))
            days_s1 = days_desc
            asc_ind = []

    return asc_ind, desc_ind, days_s1


def create_full_s1(
    batch_asc: MMDCPartialBatch,
    batch_desc: MMDCPartialBatch,
    days_asc: np.ndarray,
    days_desc: np.ndarray,
) -> tuple[MMDCPartialBatch, np.ndarray, np.ndarray, np.ndarray]:
    """
    Match S1 asc and desc dates and fill MMDC S1 batch according to available data
    """
    bs, t, ch, h, w = batch_asc.img.shape
    if h == w == 0:
        bs, t, ch, h, w = batch_desc.img.shape
    if bs == 1:
        days_asc, days_desc = days_asc[0], days_desc[0]
        asc_ind, desc_ind, days_s1 = match_asc_desc(days_asc, days_desc)

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
    else:
        raise NotImplementedError


def partial_fill_s1(
    sat: str, batch_dict: dict[str, Any], ref_date: str = None
) -> tuple[MMDCPartialBatch, list[np.array]]:
    """
    Get batch and doy for one S1 orbit
    """
    batch = MMDCPartialBatch.fill_from(batch_dict[sat], "S1")
    days = batch_dict[sat].true_doy
    if ref_date is not None:
        days = back_to_date(days, ref_date)
    return batch, days


def create_empty_s1_orbit_batch(
    batch_size: int = 1,
) -> tuple[MMDCPartialBatch, np.ndarray]:
    """
    Create an empty batch for a missing S1 orbit
    """
    days = np.empty((batch_size, 0))

    batch = MMDCPartialBatch.create_empty_s1_with_shape(
        batch_size=batch_size,
        times=0,
        height=0,
        width=0,
        nb_channels=3,
        nb_mask=1,
        nb_angles=1,
    )
    return batch, days


def encode_one_batch_s1(
    mmdc_model: PretrainedMMDCPastis, batch_dict: dict[str, Any], ref_date: str = None
) -> tuple[VAELatentSpace, VAELatentSpace, VAELatentSpace, torch.Tensor, np.ndarray,]:
    """
    Encode one S1 batch.
    Can be done when both orbits are available,
    or when only one orbit of the patch is available.
    Works only for batch size = 1
    """
    if "S1_ASC" in batch_dict:
        batch_asc, days_asc = partial_fill_s1("S1_ASC", batch_dict, ref_date)
    else:
        bs = batch_dict["S1_DESC"].true_doy.shape[0]
        batch_asc, days_asc = create_empty_s1_orbit_batch(bs)

    if "S1_DESC" in batch_dict:
        batch_desc, days_desc = partial_fill_s1("S1_DESC", batch_dict, ref_date)
    else:
        bs = batch_dict["S1_ASC"].true_doy.shape[0]
        batch_desc, days_desc = create_empty_s1_orbit_batch(bs)

    batch_s1, asc_ind, desc_ind, days_s1 = create_full_s1(
        batch_asc, batch_desc, days_asc, days_desc
    )

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
    batch_s1_mask = (batch_s1.mask.sum(2) > 1).unsqueeze(2)

    return latents1, latents1_asc, latents1_desc, batch_s1_mask, days_s1


def encode_one_batch_s2(
    mmdc_model: PretrainedMMDCPastis, batch_dict: dict[str, Any]
) -> VAELatentSpace:
    """
    Encode S2 modality
    """
    batch_s2 = MMDCPartialBatch.fill_from(batch_dict["S2"], "S2")

    # If SITS is too long, divide into two
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

    return latents2


def encode_one_batch(
    mmdc_model: PretrainedMMDCPastis, batch_dict: dict[str, Any], ref_date: str = None
) -> tuple[
    VAELatentSpace,
    VAELatentSpace,
    VAELatentSpace,
    VAELatentSpace,
    torch.Tensor,
    np.ndarray,
]:
    """
    Encode both S1 and S2 modalities
    """
    latents2 = encode_one_batch_s2(mmdc_model, batch_dict)
    latents1, latents1_asc, latents1_desc, batch_s1_mask, days_s1 = encode_one_batch_s1(
        mmdc_model, batch_dict, ref_date
    )
    return latents1, latents1_asc, latents1_desc, latents2, batch_s1_mask, days_s1


def encode_series_pvae(
    mmdc_model: PretrainedMMDCPastis,
    dataset_path_oe: str | Path,
    dataset_path_pastis: str | Path,
    sats: list[str],
    output_path: str | Path,
    join_s1: bool = False,  # TODO integrate in code
):
    """Encode PASTIS SITS into S1 and S2 latent embeddings"""
    dm = build_dm(dataset_path_oe, dataset_path_pastis, sats)

    dataset = dm.instanciate_dataset(fold=None)
    loader = dm.instanciate_data_loader(dataset, shuffle=False, drop_last=False)
    log.info("Loader is ready")
    for batch in loader:
        log.info(batch.id_patch)

        latents2 = encode_one_batch_s2(mmdc_model, batch.sits)

        encoded_pastis_s2 = {
            "latents": VAELatentSpace(
                latents2.mean.squeeze(0), latents2.logvar.squeeze(0)
            ),
            "mask": batch.sits["S2"].sits.data.mask.squeeze(0),
            "dates": back_to_date(batch.sits["S2"].true_doy, dm.reference_date)[0],
        }
        torch.save(
            encoded_pastis_s2,
            os.path.join(output_path, "S2", f"S2_{batch.id_patch[0]}.pt"),
        )
