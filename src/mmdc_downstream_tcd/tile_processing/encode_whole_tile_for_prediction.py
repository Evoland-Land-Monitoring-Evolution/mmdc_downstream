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
from mmdc_downstream_tcd.tile_processing.encode_tile import get_encoded_patch
from mmdc_downstream_tcd.tile_processing.utils import (
    generate_coord_matrix,
    get_slices,
    prepare_patch,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)


log = logging.getLogger(__name__)

meteo_modalities = METEO_BANDS.keys()


def encode_tile_for_pred(
    folder_data: str | Path,
    months_folders: list[str | int],
    folder_prepared_data: str | Path | None,
    window: int,
    output_folder: str | Path,
    mmdc_model: PretrainedMMDCPastis,
    satellites: list[str],
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

    # Whether we produce separate embeddings for s1_asc and s1_desc or not
    if s1_join and "s1_asc" in satellites and "s1_desc" in satellites:
        satellites_to_write = ["s1"]
        if "s2" in satellites:
            satellites_to_write += ["s2"]
    else:
        satellites_to_write = satellites

    slices_list = get_slices(
        folder_data, months_folders, window, margin=margin, drop_incomplete=True
    )

    dates_meteo: dict[str, pd.DataFrame] = {}

    s2_dates = np.load(os.path.join(folder_data, "dates_s2.npy"))
    s1_dates_asc = np.load(os.path.join(folder_data, "dates_s1_asc_good.npy"))
    s1_dates_desc = np.load(os.path.join(folder_data, "dates_s1_desc_good.npy"))

    for sat in satellites:
        print(folder_prepared_data, sat, f"{sat}_dates.pt")
        dates_path = os.path.join(folder_prepared_data, sat, f"{sat}_dates.pt")
        if Path(dates_path).exists():
            dates_meteo[sat] = torch.load(dates_path)

        Path(os.path.join(output_folder, sat)).mkdir(exist_ok=True, parents=True)

    # for enum, sliced in enumerate(slices_list[::-1]):
    #     enum = len(slices_list) - enum - 1
    for enum, sliced in enumerate(slices_list):
        if not np.all([sat in dates_meteo for sat in satellites]) or not np.all(
            [
                Path(
                    os.path.join(folder_prepared_data, sat, f"{sat}_tile_{enum}.pt")
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
                    os.path.join(folder_prepared_data, sat, f"{sat}_tile_{enum}.pt")
                )["data"]
                for sat in satellites
            }

        xy_matrix = generate_coord_matrix(sliced)[:, margin:-margin, margin:-margin]

        encoded_patch = get_encoded_patch(
            data, mmdc_model, dates_meteo, satellites, s1_join, margin
        )

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
    # for sat in satellites_to_write:
    #     final_df = pd.concat(sliced_gt[sat], ignore_index=True)
    #     final_df.to_csv(os.path.join(output_folder,
    #     f"gt_tcd_t32tnt_encoded_{sat}.csv"))
