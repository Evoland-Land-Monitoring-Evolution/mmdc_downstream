import argparse
import logging
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from mmdc_singledate.utils.filters import despeckle_batch

from mmdc_downstream_pastis.encode_series.encode import match_asc_desc_both_available
from mmdc_downstream_tcd.tile_processing.utils import (
    generate_coord_matrix,
    get_index,
    get_tcd_gt,
    rearrange_ts,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

log = logging.getLogger(__name__)


BANDS_S1 = [
    "VV_asc",
    "VH_asc",
    "ratio_asc",
    "VV_desc",
    "VH_desc",
    "ratio_desc",
    "ang_asc",
    "ang_desc",
]

SATELLITES = ["s1_asc", "s1_desc"]

WORK_FOLDER = (
    os.environ["WORK"] if "WORK" in os.environ else f"{os.environ['HOME']}/jeanzay"
)
SCRATCH_FOLDER = (
    os.environ["SCRATCH"]
    if "SCRATCH" in os.environ
    else f"{os.environ['HOME']}/scratch_jeanzay"
)


def get_parser() -> argparse.ArgumentParser:
    """
    Generate argument parser for CLI
    """
    arg_parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description="Get S1 pixel values for TCD ground truth points",
    )

    arg_parser.add_argument(
        "--folder_data",
        type=str,
        help="input folder",
        # default="/work/CESBIO/projects/DeepChange/Ekaterina/TCD/OpenEO_data/t32tnt/",
        default=f"{SCRATCH_FOLDER}/scratch_data/TCD/OpenEO_data/t32tnt/"
        # required=True
    )

    arg_parser.add_argument(
        "--output_folder",
        type=str,
        help="output folder",
        # default="/work/scratch/data/kalinie/TCD/t32tnt/pure_values/s1"
        default=f"{os.environ['WORK']}/results/TCD/t32tnt/pure_values/s1_desp"
        # required=True
    )

    arg_parser.add_argument(
        "--gt_path",
        type=str,
        help="path to geojson ground truth file",
        # default="/work/scratch/data/kalinie/TCD/OpenEO_data/SAMPLES_t32tnt_32632.geojson",  # noqa: E501
        default=f"{SCRATCH_FOLDER}/scratch_data/TCD/SAMPLES_t32tnt_32632.geojson",  # noqa: E501
        # required=True
    )

    arg_parser.add_argument(
        "--months_folders",
        type=str,
        help="list of available folders with months",
        default=["3", "4", "5", "6", "7", "8", "9", "10"]
        # required=False,
    )

    arg_parser.add_argument(
        "--prepared_data_path",
        type=str,
        help="Path to prepared data",
        default=f"{SCRATCH_FOLDER}/results/TCD/t32tnt/prepared_tiles_w_768_m_40"
        # required=False,
    )

    return arg_parser


def prepare_for_despeckle(img):
    nan_mask = torch.isnan(img)
    img[nan_mask] = 0
    desp_img = despeckle_batch(img)
    desp_img[nan_mask] = torch.nan
    return desp_img


def combine_s1_asc_desc(
    data: dict,
    asc_ind: np.ndarray,
    desc_ind: np.ndarray,
    days_s1: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine asc/desc orbits"""
    log.info(data["s1_asc"]["img"].shape)
    log.info(data["s1_asc"]["angles"].shape)
    log.info(data["s1_asc"]["mask"].shape)
    height, width = data["s1_asc"]["img"][0].shape[-2:]
    times = len(days_s1)

    img = torch.zeros(times, 6, height, width).to(torch.float32)
    angles = torch.zeros(times, 2, height, width).to(torch.float32)
    mask = torch.ones(times, 2, height, width).to(torch.int8)

    if len(asc_ind) > 0:
        img[asc_ind, :3, :, :] = prepare_for_despeckle(data["s1_asc"]["img"][0])
        angles[asc_ind, :1, :, :] = data["s1_asc"]["angles"][0]
        mask[asc_ind, :1, :, :] = data["s1_asc"]["mask"][0]
    if len(desc_ind) > 0:
        img[desc_ind, 3:, :, :] = prepare_for_despeckle(data["s1_desc"]["img"][0])
        angles[desc_ind, 1:, :, :] = data["s1_desc"]["angles"][0]
        mask[desc_ind, 1:, :, :] = data["s1_desc"]["mask"][0]

    return img, angles, mask


def get_s1_gt(
    folder_data: str | Path,
    months_folders: list[str | int],
    folder_prepared_data: str | Path,
    output_folder: str | Path,
    gt_file: str | Path = None,
    s1_join: bool = True,
):
    """
    Encode a tile with MMDC algorithm with a sliding window
    """

    margin = int(re.search(r"m_([0-9]*)", str(folder_prepared_data)).group(1))
    window = int(re.search(r"w_([0-9]*)", str(folder_prepared_data)).group(1))
    log.info(f"Margin={margin}")
    log.info(f"Window={window}")

    tcd_values = None
    if gt_file is not None:
        tcd_values = get_tcd_gt(gt_file)

    # for enum, sliced in enumerate(slices_list[::-1]):
    #     enum = len(slices_list) - enum - 1

    s1_dates_asc = np.load(os.path.join(folder_data, "dates_s1_asc_good.npy"))
    s1_dates_desc = np.load(os.path.join(folder_data, "dates_s1_desc_good.npy"))

    temp_slice = [
        f"2018-{months_folders[0]}-05",
        f"2018-{months_folders[-1]}-30",
    ]

    s1_dates_asc = s1_dates_asc[
        (s1_dates_asc >= pd.to_datetime(temp_slice[0]))
        & (s1_dates_asc <= pd.to_datetime(temp_slice[1]))
    ]
    s1_dates_desc = s1_dates_desc[
        (s1_dates_desc >= pd.to_datetime(temp_slice[0]))
        & (s1_dates_desc <= pd.to_datetime(temp_slice[1]))
    ]

    # TODO check
    Path(os.path.join(output_folder, "s1_desp")).mkdir(exist_ok=True, parents=True)

    available_tiles = [
        f
        for f in os.listdir(os.path.join(folder_prepared_data, SATELLITES[0]))
        if f.startswith(f"{SATELLITES[0]}_tile_") and f.endswith(".pt")
    ]
    # for enum in range(len(available_tiles))[::-1]:

    all_slices_gt = []

    for enum in range(len(available_tiles))[::-1]:
        if not Path(
            os.path.join(output_folder, f"gt_tcd_t32tnt_pure_s1_{enum}.csv")
        ).exists():
            log.info(f"Processing patch {enum}")
            data_files = {
                sat: torch.load(
                    os.path.join(folder_prepared_data, sat, f"{sat}_tile_{enum}.pt")
                )
                for sat in SATELLITES
            }

            data = {sat: data_files[sat]["data"] for sat in SATELLITES}

            xy_matrix = data_files[SATELLITES[0]]["matrix"]

            del data_files

            sliced = (
                xy_matrix["x_min"],
                xy_matrix["y_min"],
                xy_matrix["x_max"],
                xy_matrix["y_max"],
            )
            xy_matrix = generate_coord_matrix(sliced)[:, margin:-margin, margin:-margin]

            if tcd_values is not None:
                ind, tcd_values_sliced = get_index(tcd_values, xy_matrix)

                s1_dates_asc = s1_dates_asc[
                    (s1_dates_asc >= pd.to_datetime(temp_slice[0]))
                    & (s1_dates_asc <= pd.to_datetime(temp_slice[1]))
                ]
                s1_dates_desc = s1_dates_desc[
                    (s1_dates_desc >= pd.to_datetime(temp_slice[0]))
                    & (s1_dates_desc <= pd.to_datetime(temp_slice[1]))
                ]

                if s1_join:
                    # Get indices of asc/desc images in join days
                    asc_ind, desc_ind, days_s1 = match_asc_desc_both_available(
                        s1_dates_asc, s1_dates_desc
                    )
                    log.info(asc_ind, desc_ind)
                    img, angles, mask = combine_s1_asc_desc(
                        data, asc_ind, desc_ind, days_s1
                    )
                    log.info(img.shape, angles.shape, mask.shape)
                    img, angles, mask = (
                        img[:, :, margin:-margin, margin:-margin],
                        angles[:, :, margin:-margin, margin:-margin],
                        mask[:, :, margin:-margin, margin:-margin],
                    )

                else:
                    raise NotImplementedError("We are not using orbits separately yet!")

                log.info("Got tensors")

                masked_img = img.clone()
                masked_img[
                    torch.concat(
                        [
                            mask[:, :1, :, :]
                            .to(bool)
                            .expand_as(masked_img[:, :3, :, :]),
                            mask[:, 1:, :, :]
                            .to(bool)
                            .expand_as(masked_img[:, 3:, :, :]),
                        ],
                        dim=1,
                    )
                ] = torch.nan
                masked_angles = angles.clone()
                masked_angles[mask.to(bool)] = torch.nan

                masked_array = torch.concat([masked_img, masked_angles], dim=1)

                gt_ref_values = rearrange_ts(masked_array)[:, ind[:, 0], ind[:, 1]].T
                print(gt_ref_values.shape)
                values_df = pd.DataFrame(
                    data=gt_ref_values,
                    columns=[
                        b + "_d" + str(dd)
                        for dd in range(len(days_s1))
                        for b in BANDS_S1
                    ],
                )

                sliced_gt = pd.concat(
                    [tcd_values_sliced.reset_index(), values_df], axis=1
                )

                all_slices_gt.append(sliced_gt)

                sliced_gt.to_csv(
                    os.path.join(output_folder, f"gt_tcd_t32tnt_pure_s1_{enum}.csv")
                )
                if not Path(
                    os.path.join(output_folder, "gt_tcd_t32tnt_days_s1.npy")
                ).exists():
                    np.save(
                        os.path.join(output_folder, "gt_tcd_t32tnt_days_s1.npy"),
                        days_s1,
                    )

    return pd.concat(all_slices_gt, ignore_index=True)


if __name__ == "__main__":
    # Parser arguments
    parser = get_parser()
    args = parser.parse_args()

    gt_df = get_s1_gt(
        args.folder_data,
        args.months_folders,
        args.prepared_data_path,
        args.output_folder,
        args.gt_path,
    )
    gt_df.to_csv(os.path.join(args.output_folder, "gt_tcd_t32tnt_pure_s1_all.csv"))
