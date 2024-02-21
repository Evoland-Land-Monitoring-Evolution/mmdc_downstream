#!/usr/bin/env python3
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
import os

import numpy as np

from mmdc_downstream_pastis.datamodule.pastis_oe import PastisDataModule
from mmdc_downstream_pastis.mmdc_model.model import PretrainedMMDCPastis
from mmdc_downstream_pastis.utils.utils import MMDCPartialBatch

dataset_path_oe = "/work/CESBIO/projects/DeepChange/Ekaterina/Pastis_OE"
dataset_path_pastis = "/work/CESBIO/projects/DeepChange/Iris/PASTIS"

results_path = "/work/scratch/data/kalinie/MMDC/results"

pretrained_path = os.path.join(
    results_path, "latent/checkpoints/mmdc_full/2024-02-05_09-45-27"
)
model_name = "last"
model_type = "baseline"


def build_dm(sats) -> PastisDataModule:
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


mmdc_model = PretrainedMMDCPastis(pretrained_path, model_name, model_type)

sats = ["S2", "S1_ASC", "S1_DESC"]
dm = build_dm(sats)

dataset = dm.instanciate_dataset(fold=None)
loader = dm.instanciate_data_loader(dataset, shuffle=False, drop_last=False)
for batch_dict, target, mask, id_patch in loader:
    print(id_patch)
    # for satellite in batch_dict.keys():
    # if len(batch_dict.keys()) == 1:
    #     satellite = batch_dict.keys()[0]
    #     if satellite == "S2":
    #         batch_sat = batch_dict[satellite]
    #         input_doy = batch_sat.input_doy
    #         padd_val = batch_sat.padd_val
    #         padd_index = ~batch_sat.padd_index
    #         true_doy = batch_sat.true_doy
    #         data = batch_sat.sits.data
    #
    #         # batch = MMDCPartialBatch(x=data.img,
    #         #                          angles=data.angles,
    #         #                          mask=data.mask,
    #         #                          meteo=batch_sat.sits.meteo,
    #         #                          dem=batch_sat.sits.dem,
    #         #                          type="S2" if satellite == "S2" else "S1")
    #
    #         batch = MMDCPartialBatch.fill_from(batch_sat, satellite)
    #
    #         latents2 = mmdc_model.get_latent_s2_mmdc(batch.to(mmdc_model.device))

    batch_asc = MMDCPartialBatch.fill_from(batch_dict["S1_ASC"], "S1")
    batch_desc = MMDCPartialBatch.fill_from(batch_dict["S1_DESC"], "S1")
    days_asc = batch_dict["S1_ASC"].true_doy
    days_desc = batch_dict["S1_DESC"].true_doy
    batch_s1, asc_ind, desc_ind = create_s1(batch_asc, batch_desc, days_asc, days_desc)

    print(batch_asc.img.shape, batch_desc.img.shape)
    latents1 = mmdc_model.get_latent_s1_mmdc(batch_s1.to_device(mmdc_model.device))
    print(latents1.mean.shape)

    days_delta = 1643

    batch_s2 = MMDCPartialBatch.fill_from(batch_dict["S2"], "S2")
    latents2 = mmdc_model.get_latent_s2_mmdc(batch_s2.to_device(mmdc_model.device))
