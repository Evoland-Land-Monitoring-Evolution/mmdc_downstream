import logging
import os

# from mmdc_downstream_tcd.tile_processing.utils import (
#     create_netcdf_encoded,
#     create_netcdf_one_date_encoded,
#     create_netcdf_s2_one_date,
# )
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mmdc_downstream_pastis.datamodule.datatypes import METEO_BANDS
from mmdc_downstream_pastis.mmdc_model.model import PretrainedMMDCPastis
from mmdc_downstream_tcd.tile_processing.encode_tile import get_encoded_patch
from mmdc_downstream_tcd.tile_processing.utils import generate_coord_matrix, get_slices

warnings.filterwarnings("ignore", category=DeprecationWarning)


log = logging.getLogger(__name__)

satellites = ["s2", "s1_asc", "s1_desc"]
meteo_modalities = METEO_BANDS.keys()


def encode_tile(
    folder_data: str | Path,
    months_folders: list[str | int],
    window: int,
    output_folder: str | Path,
    mmdc_model: PretrainedMMDCPastis,
    gt_file: str | Path = None,
    s1_join: bool = True,
):
    """
    Encode a tile with MMDC algorithm with a sliding window
    """

    margin = mmdc_model.model_mmdc.nb_cropped_hw

    slices_list = get_slices(
        folder_data, months_folders, window, margin=margin, drop_incomplete=True
    )

    dates_meteo: dict[str, pd.DataFrame] = {}

    s2_dates = np.load(os.path.join(folder_data, "dates_s2.npy"))
    s1_dates_asc = np.load(os.path.join(folder_data, "dates_s1_asc_good.npy"))
    s1_dates_desc = np.load(os.path.join(folder_data, "dates_s1_desc_good.npy"))

    # Whether we produce separate embeddings for s1_asc and s1_desc or not
    satellites_to_write = ["s2", "s1"] if s1_join else satellites

    # for enum, sliced in enumerate(slices_list[::-1]):
    #     enum = len(slices_list) - enum - 1
    for enum, sliced in enumerate(slices_list):
        if not np.all(
            [
                Path(
                    os.path.join(os.path.join(output_folder, f"{sat}_{enum}.pt"))
                ).exists()
                for sat in satellites_to_write
            ]
        ):
            xy_matrix = generate_coord_matrix(sliced)[:, margin:-margin, margin:-margin]

            encoded_patch = get_encoded_patch(
                folder_data,
                months_folders,
                mmdc_model,
                xy_matrix,
                sliced,
                margin,
                s2_dates,
                s1_dates_asc,
                s1_dates_desc,
                dates_meteo,
                s1_join=True,
            )

            for sat in satellites_to_write:
                mask = encoded_patch[f"{sat}_mask"].expand_as(
                    encoded_patch[f"{sat}_lat_mu"]
                )

                encoded_patch[f"{sat}_lat_mu"][mask.bool()] = torch.nan
                encoded_patch[f"{sat}_lat_logvar"][mask.bool()] = torch.nan

                # days = encoded_patch[f"{sat}_doy"]

                print([xy_matrix[0, 0], xy_matrix[1, :, 0]])
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
                    os.path.join(output_folder, f"{sat}_{enum}.pt"),
                )

            del encoded_patch
    # for sat in satellites_to_write:
    #     final_df = pd.concat(sliced_gt[sat], ignore_index=True)
    #     final_df.to_csv(os.path.join(output_folder,
    #     f"gt_tcd_t32tnt_encoded_{sat}.csv"))
