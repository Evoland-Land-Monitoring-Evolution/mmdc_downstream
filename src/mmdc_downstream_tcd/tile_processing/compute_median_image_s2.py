import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from encode_tile import generate_coord_matrix, get_s2, get_sliced_modality, get_slices

from mmdc_downstream_lai.utils import get_logger

log = get_logger(__name__)

SEL_BANDS = ["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"]
BANDS_S2 = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
bands_s2 = [0, 1, 2, 6, 8, 9]

# folder_data = \
#     "/home/kalinichevae/scratch_jeanzay/scratch_data/TCD/OpenEO_data/t32tnt"
# gt_path = \
#     "/home/kalinichevae/scratch_jeanzay/scratch_data/TCD/SAMPLES_t32tnt_32632.geojson"
# output_folder = \
#     "/home/kalinichevae/jeanzay/results/TCD/t32tnt/pure_values/S2_full_median"

folder_data = f"{os.environ['SCRATCH']}/scratch_data/TCD/OpenEO_data/t32tnt"
gt_path = f"{os.environ['SCRATCH']}/scratch_data/TCD/SAMPLES_t32tnt_32632.geojson"
output_folder = f"{os.environ['WORK']}/results/TCD/t32tnt/pure_values/S2_full_median"


months_folders = ["3", "4", "5", "6", "7", "8", "9", "10"]
window = 1000


tcd_values = pd.DataFrame(columns=["TCD", "x", "y", "x_round", "y_round"])
with open(gt_path) as f:
    data = json.load(f)


def get_s2_gt(
    folder_data: str | Path,
    output_folder: str | Path,
    months_folders: list[str | int],
    window: int,
):
    """
    Get a S2 month-median reflectance tile with a sliding window
    """

    slices_list = get_slices(folder_data, months_folders, window)

    log.info("Getting clouds")
    s2_dates = np.load(os.path.join(folder_data, "dates_s2.npy"))
    print(s2_dates)
    log.info("Done with clouds")
    log.info(f"Total number of slices {len(slices_list)}")

    for enum, sliced in enumerate(slices_list):
        log.info(sliced)
        images = {}
        data = {"s2": {}}
        modality = "Sentinel2"
        xy_matrix = generate_coord_matrix(sliced)

        sliced_modality = get_sliced_modality(
            months_folders,
            folder_data,
            sliced,
            modality,
            meteo=False,
            dates=s2_dates,
        )

        log.info("Got modality")

        temp_slice = [
            f"2018-{months_folders[0]}-05",
            f"2018-{months_folders[-1]}-30",
        ]

        sliced_modality = sliced_modality.sel(t=slice(temp_slice[0], temp_slice[1]))
        _, data, dates = get_s2(s2_dates, temp_slice, data, images, sliced_modality)
        log.info("Got tensors")

        del sliced_modality

        masked_array = data["s2"]["img"].squeeze(0)[:, bands_s2, :, :]
        masked_array[
            data["s2"]["mask"].squeeze(0).to(bool).expand_as(masked_array)
        ] = torch.nan
        del data

        all_months_med = []

        for month in months_folders:
            months_dates = (dates.date_s2.dt.month == int(month)).values
            months_masked_array = masked_array[months_dates]
            med_months = torch.nanmedian(months_masked_array, dim=0).values
            all_months_med.append(med_months)
        del masked_array, months_masked_array
        all_months_med = torch.cat(all_months_med)

        print(all_months_med.shape)
        torch.save(
            {"img": all_months_med, "matrix": xy_matrix},
            os.path.join(output_folder, f"s2_median_{enum}.pt"),
        )


get_s2_gt(folder_data, output_folder, months_folders, window)
