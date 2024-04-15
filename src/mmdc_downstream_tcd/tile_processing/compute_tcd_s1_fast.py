import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mmdc_downstream_pastis.encode_series.encode import match_asc_desc
from mmdc_downstream_tcd.tile_processing.utils import (
    generate_coord_matrix,
    get_index,
    get_s1,
    get_sliced_modality,
    get_slices,
    get_tcd_gt,
    rearrange_ts,
)

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
        default=f"{os.environ['SCRATCH']}/scratch_data/TCD/OpenEO_data/t32tnt/"
        # required=True
    )

    arg_parser.add_argument(
        "--output_folder",
        type=str,
        help="output folder",
        # default="/work/scratch/data/kalinie/TCD/t32tnt/pure_values/s1"
        default=f"{os.environ['WORK']}/results/TCD/t32tnt/pure_values/s1"
        # required=True
    )

    arg_parser.add_argument(
        "--gt_path",
        type=str,
        help="path to geojson ground truth file",
        # default="/work/scratch/data/kalinie/TCD/OpenEO_data/SAMPLES_t32tnt_32632.geojson",  # noqa: E501
        default=f"{os.environ['SCRATCH']}/scratch_data/TCD/SAMPLES_t32tnt_32632.geojson",  # noqa: E501
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
        "--window",
        type=int,
        help="size of sliding window",
        default=2000
        # required=False,
    )

    return arg_parser


def combine_s1_asc_desc(
    data: dict,
    asc_ind: np.ndarray,
    desc_ind: np.ndarray,
    days_s1: np.ndarray,
    window: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine asc/desc orbits"""

    height, width = data["s1_asc"]["img"].shape[-2:]
    times = len(days_s1)

    img = torch.zeros(times, 6, height, width)
    angles = torch.zeros(times, 2, height, width)
    mask = torch.ones(times, 2, height, width).to(torch.int8)

    if len(asc_ind) > 0:
        img[asc_ind, :3, :, :] = data["s1_asc"]["img"]
        angles[asc_ind, :1, :, :] = data["s1_asc"]["angles"]
        mask[asc_ind, :1, :, :] = data["s1_asc"]["mask"]
    if len(desc_ind) > 0:
        img[desc_ind, 3:, :, :] = data["s1_desc"]["img"]
        angles[desc_ind, 1:, :, :] = data["s1_desc"]["angles"]
        mask[desc_ind, 1:, :, :] = data["s1_desc"]["mask"]

    return img, angles, mask


def get_s1_gt(
    folder_data: str | Path,
    months_folders: list[str | int],
    window: int,
    output_folder: str | Path,
    gt_file: str | Path = None,
    s1_join: bool = True,
):
    """
    Get S1 pixel values for TCD
    """

    # We load dates that we are going to use
    s1_dates_asc = np.load(os.path.join(folder_data, "dates_s1_asc_good.npy"))
    s1_dates_desc = np.load(os.path.join(folder_data, "dates_s1_desc_good.npy"))

    # Get GT values and their coordinates
    tcd_values = get_tcd_gt(gt_file)
    all_slices_gt = []

    # Clip tile into slices with sliding window
    slices_list = get_slices(folder_data, months_folders, window)

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    temp_slice = [
        f"2018-{months_folders[0]}-05",
        f"2018-{months_folders[-1]}-30",
    ]

    for enum, sliced in enumerate(slices_list):
        # coord matrix to get gt pixel values
        xy_matrix = generate_coord_matrix(sliced)

        # their coords in the matrix
        ind, tcd_values_sliced = get_index(tcd_values, xy_matrix)

        if ind.size > 0:
            log.info(f"Slice {sliced}")
            images = {}
            data = {sat: {} for sat in SATELLITES}

            for sat in SATELLITES:
                modality = (
                    "Sentinel1_ASCENDING" if sat == "s1_asc" else "Sentinel1_DESCENDING"
                )
                log.info(f"Slicing {modality}")
                sliced_modality = get_sliced_modality(
                    months_folders,
                    folder_data,
                    sliced,
                    modality,
                    meteo=False,
                    dates=s1_dates_asc if sat == "s1_asc" else s1_dates_desc,
                )
                log.info(f"Sliced {modality}")
                log.info("Got modality")

                sliced_modality_selected = sliced_modality.sel(
                    t=slice(temp_slice[0], temp_slice[1])
                )

                images, data, dates = get_s1(
                    temp_slice, data, images, sliced_modality_selected, sat
                )
                del sliced_modality

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
                asc_ind, desc_ind, days_s1 = match_asc_desc(s1_dates_asc, s1_dates_desc)

                img, angles, mask = combine_s1_asc_desc(
                    data, asc_ind, desc_ind, days_s1, window
                )
            else:
                raise NotImplementedError("We are not using orbits separately yet!")

            log.info("Got tensors")

            masked_img = img.clone()
            masked_img[
                torch.concat(
                    [
                        mask[:, :1].to(bool).expand_as(masked_img[:, :3]),
                        mask[:, 1:].to(bool).expand_as(masked_img[:, 3:]),
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
                    b + "_" + str(dd) for dd in range(len(days_s1)) for b in BANDS_S1
                ],
            )

            sliced_gt = pd.concat([tcd_values_sliced.reset_index(), values_df], axis=1)

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
        args.window,
        args.output_folder,
        args.gt_path,
    )
    gt_df.to_csv(os.path.join(args.output_folder, "gt_tcd_t32tnt_all.csv"))
