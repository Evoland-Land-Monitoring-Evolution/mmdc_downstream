import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.mmdc_downstream_tcd.tile_processing.utils import (
    generate_coord_matrix,
    get_slices,
    prepare_patch,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

log = logging.getLogger(__name__)


def prepare_tiles(
    folder_data: str | Path,
    months_folders: list[str | int],
    window: int,
    output_folder: str | Path,
    margin: int = 50,
    satellites: list[str] = ("s2", "s1_asc", "s1_desc"),
):
    """
    Encode a tile with MMDC algorithm with a sliding window
    """

    slices_list = get_slices(
        folder_data, months_folders, window, margin=margin, drop_incomplete=True
    )

    dates_meteo: dict[str, pd.DataFrame] = {}
    s2_dates = np.load(os.path.join(folder_data, "dates_s2.npy"))
    s1_dates_asc = np.load(os.path.join(folder_data, "dates_s1_asc_good.npy"))
    s1_dates_desc = np.load(os.path.join(folder_data, "dates_s1_desc_good.npy"))

    # for enum, sliced in enumerate(slices_list[::-1]):
    #     enum = len(slices_list) - enum - 1
    for enum, sliced in enumerate(slices_list):
        if not np.all(
            [
                Path(os.path.join(output_folder, sat, f"{sat}_tile_{enum}.pt")).exists()
                for sat in satellites
            ]
        ):
            xy_matrix = generate_coord_matrix(sliced)

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
            data["matrix"] = {
                "x_min": xy_matrix[0, 0].min(),
                "x_max": xy_matrix[0, 0].max(),
                "y_min": xy_matrix[1, :, 0].min(),
                "y_max": xy_matrix[1, :, 0].max(),
            }

            for sat in satellites:
                torch.save(
                    {"data": data[sat], "matrix": data["matrix"]},
                    os.path.join(output_folder, sat, f"{sat}_tile_{enum}.pt"),
                )
                if not Path(
                    os.path.join(output_folder, sat, f"{sat}_dates.pt")
                ).exists():
                    torch.save(
                        dates_meteo[sat],
                        os.path.join(output_folder, sat, f"{sat}_dates.pt"),
                    )
