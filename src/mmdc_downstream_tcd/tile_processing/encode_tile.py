import logging
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mmdc_downstream_pastis.datamodule.datatypes import METEO_BANDS
from mmdc_downstream_pastis.mmdc_model.model import PretrainedMMDCPastis
from mmdc_downstream_tcd.tile_processing.utils import (
    extract_gt_points,
    generate_coord_matrix,
    get_encoded_patch,
    get_index,
    get_slices,
    get_tcd_gt,
    prepare_patch,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

log = logging.getLogger(__name__)

meteo_modalities = METEO_BANDS.keys()


def encode_tile(
    folder_data: str | Path,
    months_folders: list[str | int],
    folder_prepared_data: str | Path | None,
    window: int,
    output_folder: str | Path,
    mmdc_model: PretrainedMMDCPastis,
    gt_file: str | Path = None,
    satellites: list[str] = ["s2", "s1_asc", "s1_desc"],
    s1_join: bool = True,
):
    """
    Encode a tile with MMDC algorithm with a sliding window
    """
    if folder_prepared_data is not None:
        margin = int(re.search(r"m_([0-9]*)", str(folder_prepared_data)).group(1))
    else:
        margin = mmdc_model.model_mmdc.nb_cropped_hw
    log.info(f"Margin= {margin}")

    slices_list = get_slices(
        folder_data, months_folders, window, margin=margin, drop_incomplete=True
    )

    dates_meteo: dict[str, pd.DataFrame] = {}

    s2_dates = np.load(os.path.join(folder_data, "dates_s2.npy"))
    s1_dates_asc = np.load(os.path.join(folder_data, "dates_s1_asc_good.npy"))
    s1_dates_desc = np.load(os.path.join(folder_data, "dates_s1_desc_good.npy"))

    # print("computing s1 asc")
    # s2_dates = check_s2_clouds(folder_data, months_folders)
    # s1_dates_asc = check_s1_nans(folder_data, months_folders,
    #                          orbit="ASCENDING")
    # np.save(os.path.join(folder_data, "dates_s1_asc.npy"), s1_dates_asc)
    # print("computing s1 desc")
    # s1_dates_desc = check_s1_nans(folder_data, months_folders,
    #                          orbit="DESCENDING")
    # #
    # np.save(os.path.join(folder_data, "dates_s1_desc.npy"), s1_dates_desc)
    tcd_values = None
    if gt_file is not None:
        tcd_values = get_tcd_gt(gt_file)

    # Whether we produce separate embeddings for s1_asc and s1_desc or not
    # Whether we produce separate embeddings for s1_asc and s1_desc or not
    if s1_join and "s1_asc" in satellites and "s1_desc" in satellites:
        satellites_to_write = ["s1"]
        if "s2" in satellites:
            satellites_to_write += ["s2"]
    else:
        satellites_to_write = satellites

    sliced_gt = {}
    # for enum, sliced in enumerate(slices_list[::-1]):
    #     enum = len(slices_list) - enum - 1

    for sat in satellites:
        print(folder_prepared_data, sat, f"{sat}_dates.pt")
        dates_path = os.path.join(folder_prepared_data, sat, f"{sat}_dates.pt")
        if Path(dates_path).exists():
            dates_meteo[sat] = torch.load(dates_path)

        Path(os.path.join(output_folder, sat)).mkdir(exist_ok=True, parents=True)

    for enum, sliced in enumerate(slices_list):
        if not np.all(
            [
                Path(
                    os.path.join(
                        output_folder, sat, f"gt_tcd_t32tnt_encoded_{sat}_{enum}.csv"
                    )
                ).exists()
                for sat in satellites_to_write
            ]
        ):
            xy_matrix = generate_coord_matrix(sliced)[:, margin:-margin, margin:-margin]

            ind = []
            tcd_values_sliced = None
            if tcd_values is not None:
                ind, tcd_values_sliced = get_index(tcd_values, xy_matrix)
            if tcd_values is None or (tcd_values is not None and ind.size > 0):
                if not np.all([sat in dates_meteo for sat in satellites]) or not np.all(
                    [
                        Path(
                            os.path.join(
                                folder_prepared_data, sat, f"{sat}_tile_{enum}.pt"
                            )
                        ).exists()
                        for sat in satellites
                    ]
                ):
                    data, dates_meteo = prepare_patch(
                        folder_data,
                        months_folders,
                        sliced,
                        s2_dates,
                        s1_dates_asc,
                        s1_dates_desc,
                        dates_meteo,
                        satellites,
                    )
                else:
                    data = {
                        sat: torch.load(
                            os.path.join(
                                folder_prepared_data, sat, f"{sat}_tile_{enum}.pt"
                            )
                        )["data"]
                        for sat in satellites
                    }

                encoded_patch = get_encoded_patch(
                    data, mmdc_model, dates_meteo, satellites, s1_join, margin
                )
                encoded_patch["matrix"] = {
                    "x_min": xy_matrix[0, 0].min(),
                    "x_max": xy_matrix[0, 0].max(),
                    "y_min": xy_matrix[1, :, 0].min(),
                    "y_max": xy_matrix[1, :, 0].max(),
                }

                if ind.size > 0 and tcd_values_sliced is not None:
                    for sat in satellites_to_write:
                        df_gt, days = extract_gt_points(
                            encoded_patch,
                            sat,
                            ind,
                            tcd_values_sliced,
                            sliced_gt,
                        )
                        df_gt.to_csv(
                            os.path.join(
                                output_folder,
                                sat,
                                f"gt_tcd_t32tnt_encoded_{sat}_{enum}.csv",
                            )
                        )
                        if not Path(
                            os.path.join(
                                output_folder, sat, f"gt_tcd_t32tnt_days_{sat}.npy"
                            )
                        ).exists():
                            np.save(
                                os.path.join(
                                    output_folder, sat, f"gt_tcd_t32tnt_days_{sat}.npy"
                                ),
                                days,
                            )

                # torch.save(encoded_patch,
                #            os.path.join(output_folder, f"Encoded_patch_{enum}.pt"))

            for sat in satellites_to_write:
                mask = encoded_patch[f"{sat}_mask"].expand_as(
                    encoded_patch[f"{sat}_lat_mu"]
                )

                encoded_patch[f"{sat}_lat_mu"][mask.bool()] = torch.nan
                encoded_patch[f"{sat}_lat_logvar"][mask.bool()] = torch.nan

                # days = encoded_patch[f"{sat}_doy"]

                torch.save(
                    {
                        "img": torch.concat(
                            [
                                encoded_patch[f"{sat}_lat_mu"],
                                encoded_patch[f"{sat}_lat_logvar"],
                            ],
                            1,
                        )
                        .cpu()
                        .numpy()
                        .astype(np.float32),
                        "matrix": {
                            "x_min": xy_matrix[0, 0].min(),
                            "x_max": xy_matrix[0, 0].max(),
                            "y_min": xy_matrix[1, :, 0].min(),
                            "y_max": xy_matrix[1, :, 0].max(),
                        },
                    },
                    os.path.join(output_folder, sat, f"{sat}_{enum}.pt"),
                )

            del encoded_patch
