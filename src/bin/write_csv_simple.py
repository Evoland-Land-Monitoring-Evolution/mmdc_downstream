#!/usr/bin/env python3
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
""" Write data for lai regression prediction """

import csv
import os
from pathlib import Path
from typing import Literal, get_args

import numpy as np
import torch
from einops import rearrange
from mmdc_singledate.datamodules.datatypes import (
    MMDCBatch,
    MMDCDataLoaderConfig,
    MMDCDataPaths,
    MMDCDataStats,
    MMDCDataStruct,
    MMDCShiftScales,
    ShiftScale,
)
from mmdc_singledate.datamodules.mmdc_datamodule import MMDCDataModule
from mmdc_singledate.utils.train_utils import standardize_data

from mmdc_downstream.mmdc_model.model import PretrainedMMDC
from mmdc_downstream.snap.components.compute_bio_var import (
    predict_variable_from_tensors,
    prepare_s2_image,
)
from mmdc_downstream.snap.lai_snap import BVNET, normalize
from mmdc_downstream.utils import get_logger

TILES_CONFIG_DIR = (
    "/work/CESBIO/projects/DeepChange/Ekaterina/MMDC_OE/tiles_conf_training/LAI_micro/"
)
DATASET_DIR = "/work/CESBIO/projects/DeepChange/Ekaterina/MMDC_OE/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log = get_logger(__name__)

MMDCDataComponents = Literal[
    "s2_set",
    "s2_masks",
    "s2_angles",
    "s1_set",
    "s1_valmasks",
    "s1_angles",
    "meteo_dew_temp",
    "meteo_prec",
    "meteo_sol_rad",
    "meteo_temp_max",
    "meteo_temp_mean",
    "meteo_temp_min",
    "meteo_vap_press",
    "meteo_wind_speed",
    "dem",
]


def build_datamodule() -> tuple[MMDCDataModule, MMDCDataLoaderConfig]:
    dlc = MMDCDataLoaderConfig(
        max_open_files=6,
        batch_size_train=10,
        batch_size_val=10,
        num_workers=6,
        pin_memory=False,
    )
    dpth = MMDCDataPaths(
        tensors_dir=Path(DATASET_DIR),
        train_rois=Path(f"{TILES_CONFIG_DIR}/train.txt"),
        val_rois=Path(f"{TILES_CONFIG_DIR}/val.txt"),
        test_rois=Path(f"{TILES_CONFIG_DIR}/test.txt"),
    )
    dm = MMDCDataModule(
        data_paths=dpth,
        dl_config=dlc,
    )
    return dm, dlc


def build_data_loader() -> tuple[MMDCDataLoaderConfig, MMDCDataModule]:
    dm, dlc = build_datamodule()
    dm.setup("fit")
    dm.setup("test")

    return dlc, dm


def compute_gt(model_snap: BVNET, batch: MMDCBatch, stand: bool = True) -> torch.Tensor:
    """Compute LAI GT data wiht standardisation or not"""
    gt = predict_variable_from_tensors(batch.s2_x, batch.s2_a, batch.s2_m, model_snap)
    if stand:
        return normalize(gt, model_snap.variable_min, model_snap.variable_max)
    return gt


def get_stats(stats: MMDCDataStats) -> MMDCShiftScales:
    """Set shift and scale for model"""
    scale_regul = torch.nn.Threshold(1e-10, 1.0)
    shift_scale_s2 = ShiftScale(
        stats.sen2.median,
        scale_regul((stats.sen2.qmax - stats.sen2.qmin) / 2.0),
    )
    shift_scale_s1 = ShiftScale(
        stats.sen1.median,
        scale_regul((stats.sen1.qmax - stats.sen1.qmin) / 2.0),
    )
    shift_scale_meteo = ShiftScale(
        stats.meteo.concat_stats().median,
        scale_regul(
            (stats.meteo.concat_stats().qmax - stats.meteo.concat_stats().qmin) / 2.0
        ),
    )
    shift_scale_dem = ShiftScale(
        stats.dem.median,
        scale_regul((stats.dem.qmax - stats.dem.qmin) / 2.0),
    )

    return MMDCShiftScales(
        shift_scale_s2,
        shift_scale_s1,
        shift_scale_meteo,
        shift_scale_dem,
    )


dlc, dm = build_data_loader()
results_path = "/work/scratch/data/kalinie/MMDC/results"
model_folder = "latent/checkpoints/mmdc_full/2024-02-27_14-47-18"
pretrained_path = os.path.join(results_path, model_folder)
model_name = "last"
model_type = "baseline"

model_mmdc = PretrainedMMDC(
    pretrained_path=os.path.join(results_path, model_folder),
    model_name=model_name,
    model_type=model_type,
)
stats_path = os.path.join(results_path, model_folder, "stats.pt")
model_snap = BVNET(device=str(DEVICE), ver="2", variable="lai", third_layer=False)
model_snap.set_snap_weights()
# model_snap.eval()

stats = get_stats(torch.load(stats_path))


def load_files(file_list: list[list[str]]) -> MMDCDataStruct:
    """Read all files at once, shuffle the sample order and yield the items"""

    # get back the original type of data
    file_zip = (tuple(i) for i in file_list)
    # unzip the variables
    file_unzip = list(zip(*file_zip, strict=True))

    keys = get_args(MMDCDataComponents)

    data = {
        k: torch.concat([torch.load(f) for f in list(v)])
        for k, v in zip(keys, tuple(file_unzip), strict=True)
    }

    data["s1_angles"] = data["s1_angles"].nan_to_num()
    data["s2_angles"] = data["s2_angles"].nan_to_num()

    res: MMDCDataStruct = MMDCDataStruct.init_empty().fill_empty_from_dict(
        dictionary=data
    )
    return res


def batch_to_device(batch):
    for b in range(len(batch)):
        for a in range(len(batch[b])):
            batch[b][a] = batch[b][a].to(DEVICE)
    return batch


dpth = MMDCDataPaths(
    tensors_dir=Path(DATASET_DIR),
    train_rois=Path(f"{TILES_CONFIG_DIR}/train.txt"),
    val_rois=Path(f"{TILES_CONFIG_DIR}/val.txt"),
    test_rois=Path(f"{TILES_CONFIG_DIR}/test.txt"),
)


margin = model_mmdc.model_mmdc.nb_cropped_hw
log.info(f"Margin {margin}")

files_dict = {
    "train": dm.files.train_data_files,
    "val": dm.files.val_data_files,
    "test": dm.files.test_data_files,
}

for type, files in files_dict.items():
    header = None

    csv_name = f"data_values_{model_type}_{pretrained_path.split('/')[-1]}_{type}.csv"

    for enum, file in enumerate(files[:-1]):
        log.info(f"{file}")
        tensors = load_files([file])

        batch_all = MMDCBatch(
            tensors.s2_data.s2_set.to(DEVICE),
            tensors.s2_data.s2_masks.to(DEVICE),
            tensors.s2_data.s2_angles.to(DEVICE),
            tensors.s1_data.s1_set.to(DEVICE),
            tensors.s1_data.s1_valmasks.to(DEVICE),
            tensors.s1_data.s1_angles.to(DEVICE),
            torch.flatten(tensors.meteo.concat_data(), start_dim=1, end_dim=2).to(
                DEVICE
            ),
            tensors.dem.to(DEVICE),
        )

        batches = [
            batch_all[: int(len(tensors.s2_data.s2_set) / 2)],
            batch_all[int(len(tensors.s2_data.s2_set) / 2) :],
        ]

        for batch in batches:
            log.info(f"{enum}/{len(files)}")

            lai_gt = compute_gt(model_snap, batch)
            # print(lai_gt[0])
            # lai_gt_denorm = denormalize(
            #     lai_gt.clone(), model_snap.variable_min, model_snap.variable_max
            # )
            #
            # mask = batch.s2_m
            # H, W = mask.shape[-2:]
            #
            # mask[:, :, : margin, : margin] = 1
            # mask[:, :, -margin:, -margin:] = 1
            # mask = mask.reshape(-1).bool()
            #
            # gt = (
            #     lai_gt_denorm.reshape(-1)[~mask]
            #     .to(torch.float32)
            #     .unsqueeze(-1)
            #     .cpu()
            #     .numpy()
            #     .round(3)
            # )
            # log.info(np.min(gt))
            # log.info(np.max(gt))

            mask = batch.s2_m
            H, W = mask.shape[-2:]

            mask[:, :, :margin, :margin] = 1
            mask[:, :, -margin:, -margin:] = 1
            mask = mask.reshape(-1).bool()
            nbr_pixels = (mask == 0).sum().cpu().numpy()
            # idx = np.random.choice(nbr_pixels, int(nbr_pixels * 0.1), replace=False)
            idx = np.arange(nbr_pixels)
            s2_ref = (
                rearrange(batch.s2_x, "b c h w -> (b h w) c")[~mask][idx].cpu().numpy()
                / 10000
            )
            s2_angles = (
                rearrange(batch.s2_a, "b c h w -> (b h w) c")[~mask][idx].cpu().numpy()
            )
            s1 = (
                standardize_data(
                    rearrange(batch.s1_x, "b c h w -> (b h w) c")[~mask][idx],
                    shift=stats.sen1.shift.type_as(batch.s1_x),
                    scale=stats.sen1.shift.type_as(batch.s1_x),
                )
                .cpu()
                .numpy()
            )
            s1_angles = (
                rearrange(batch.s1_a, "b c h w -> (b h w) c")[~mask][idx].cpu().numpy()
            )
            s1_mask = (
                rearrange(batch.s1_vm, "b c h w -> (b h w) c")[~mask][idx].cpu().numpy()
            )

            data = (
                prepare_s2_image(batch.s2_x / 10000, batch.s2_a, reshape=False)
                .nan_to_num()
                .to("cuda")
            )
            s2_snap_input = normalize(
                data,
                model_snap.input_min.reshape(1, data.shape[1], 1, 1),
                model_snap.input_max.reshape(1, data.shape[1], 1, 1),
            )

            s2_input = (
                rearrange(s2_snap_input, "b c h w -> (b h w) c")[~mask][idx]
                .cpu()
                .numpy()
            )
            latent = model_mmdc.get_latent_mmdc(batch)
            s1_lat_mu = (
                rearrange(latent.latent_S1_mu, "b c h w -> (b h w) c")[~mask][idx]
                .to(torch.float32)
                .cpu()
                .numpy()
            )
            s1_lat_logvar = (
                rearrange(latent.latent_S1_logvar, "b c h w -> (b h w) c")[~mask][idx]
                .to(torch.float32)
                .cpu()
                .numpy()
            )
            s2_lat_mu = (
                rearrange(latent.latent_S2_mu, "b c h w -> (b h w) c")[~mask][idx]
                .to(torch.float32)
                .cpu()
                .numpy()
            )
            s2_lat_logvar = (
                rearrange(latent.latent_S2_logvar, "b c h w -> (b h w) c")[~mask][idx]
                .to(torch.float32)
                .cpu()
                .numpy()
            )

            if model_type == "experts":
                exp_lat_mu = (
                    rearrange(latent.latent_experts_mu, "b c h w -> (b h w) c")[~mask][
                        idx
                    ]
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                )
                exp_lat_logvar = (
                    rearrange(latent.latent_experts_logvar, "b c h w -> (b h w) c")[
                        ~mask
                    ][idx]
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

            if header is None:
                header = np.concatenate(
                    (
                        [f"s2_{i}" for i in range(s2_ref.shape[1])],
                        [f"s2_ang_{i}" for i in range(s2_angles.shape[1])],
                        [f"s1_{i}" for i in range(s1.shape[1])],
                        [f"s1_ang_{i}" for i in range(s1_angles.shape[1])],
                        [f"s1_mask_{i}" for i in range(s1_mask.shape[1])],
                        [f"s2_input_{i}" for i in range(s2_input.shape[1])],
                        [f"s1_lat_mu_{i}" for i in range(s1_lat_mu.shape[1])],
                        [f"s1_lat_logvar_{i}" for i in range(s1_lat_logvar.shape[1])],
                        [f"s2_lat_mu_{i}" for i in range(s2_lat_mu.shape[1])],
                        [f"s2_lat_logvar_{i}" for i in range(s2_lat_logvar.shape[1])],
                    ),
                    axis=0,
                )
                if model_type == "experts":
                    header = np.concatenate(
                        (
                            header,
                            [f"exp_lat_{i}" for i in range(exp_lat_mu.shape[1])],
                            [f"exp_lat_{i}" for i in range(exp_lat_logvar.shape[1])],
                        ),
                        axis=0,
                    )
                header = np.concatenate((header, ["gt"]), axis=0)

                print(header)
                with open(csv_name, "w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(header)
            if model_type == "experts":
                data_to_write = np.hstack(
                    (
                        s2_ref,
                        s2_angles,
                        s1,
                        s1_angles,
                        s1_mask,
                        s2_input,
                        s1_lat_mu,
                        s1_lat_logvar,
                        s2_lat_mu,
                        s2_lat_logvar,
                        exp_lat_mu,
                        exp_lat_logvar,
                        gt,
                    )
                )
            else:
                data_to_write = np.hstack(
                    (
                        s2_ref,
                        s2_angles,
                        s1,
                        s1_angles,
                        s1_mask,
                        s2_input,
                        s1_lat_mu,
                        s1_lat_logvar,
                        s2_lat_mu,
                        s2_lat_logvar,
                        gt,
                    )
                )
            log.info("writing")
            with open(csv_name, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(data_to_write)
