#!/usr/bin/env python3
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
""" Write data for lai regression prediction """

import csv
import os
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from mmdc_singledate.datamodules.datatypes import (
    MMDCBatch,
    MMDCDataLoaderConfig,
    MMDCDataPaths,
    MMDCDataStats,
    MMDCShiftScales,
    ShiftScale,
)
from mmdc_singledate.datamodules.mmdc_datamodule import (
    MMDCDataModule,
    destructure_batch,
)
from mmdc_singledate.utils.train_utils import standardize_data
from torch.utils.data import DataLoader

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


def build_datamodule() -> tuple[MMDCDataModule, MMDCDataLoaderConfig]:
    dlc = MMDCDataLoaderConfig(
        max_open_files=1,
        batch_size_train=200,
        batch_size_val=200,
        num_workers=1,
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


def build_data_loader() -> tuple[DataLoader, MMDCDataLoaderConfig, MMDCDataModule]:
    dm, dlc = build_datamodule()
    dm.setup("fit")
    dm.setup("test")
    dl = dm.train_dataloader()

    return dl, dlc, dm


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


dl, dlc, dm = build_data_loader()
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

csv_name = f"data_values_{model_type}_{pretrained_path.split('/')[-1]}.csv"

stats = get_stats(torch.load(stats_path))
header = None


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


for dl in (dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()):
    # ds = IterableMMDCDataset(
    #     files=dm.build_file_list(dpth.train_rois),
    #     batch_size=dlc.batch_size_train,
    #     max_open_files=dlc.max_open_files,
    # )
    for enum, batch in enumerate(dl):
        if enum >= len(dl):
            break

        batch = batch_to_device(batch)
        log.info(f"{enum}/{len(dl)}")
        batch: MMDCBatch = destructure_batch(batch)

        lai_gt = compute_gt(model_snap, batch)
        mask = batch.s2_m.reshape(-1).bool()
        nbr_pixels = (mask == 0).sum().cpu().numpy()
        idx = np.random.choice(nbr_pixels, int(nbr_pixels * 0.1), replace=False)
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
            rearrange(s2_snap_input, "b c h w -> (b h w) c")[~mask][idx].cpu().numpy()
        )
        latent = model_mmdc.get_latent_mmdc(batch)
        s1_lat = (
            rearrange(latent.latent_S1_mu, "b c h w -> (b h w) c")[~mask][idx]
            .to(torch.float32)
            .cpu()
            .numpy()
        )
        s2_lat = (
            rearrange(latent.latent_S2_mu, "b c h w -> (b h w) c")[~mask][idx]
            .to(torch.float32)
            .cpu()
            .numpy()
        )

        if model_type == "experts":
            exp_lat = (
                rearrange(latent.latent_experts_mu, "b c h w -> (b h w) c")[~mask][idx]
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
                    [f"s1_lat_{i}" for i in range(s1_lat.shape[1])],
                    [f"s2_lat_{i}" for i in range(s2_lat.shape[1])],
                ),
                axis=0,
            )
            if model_type == "experts":
                header = np.concatenate(
                    (header, [f"exp_lat_{i}" for i in range(exp_lat.shape[1])]), axis=0
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
                    s1_lat,
                    s2_lat,
                    exp_lat,
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
                    s1_lat,
                    s2_lat,
                    gt,
                )
            )
        log.info("writing")
        with open(csv_name, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(data_to_write)
