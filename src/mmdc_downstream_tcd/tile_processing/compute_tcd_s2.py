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

folder_data = "/work/scratch/data/kalinie/TCD/OpenEO_data/t32tnt/"
months_folders = ["3", "4", "5", "6", "7", "8", "9", "10"]
window = 500
gt_path = "/work/scratch/data/kalinie/TCD/OpenEO_data/SAMPLES_t32tnt_32632.geojson"

tcd_values = pd.DataFrame(columns=["TCD", "x", "y", "x_round", "y_round"])
with open(gt_path) as f:
    data = json.load(f)

for feature in data["features"]:
    x, y = feature["geometry"]["coordinates"][0]
    tcd = feature["properties"]["TCD"]

    x_r = round(x / 5) * 5
    y_r = round(y / 5) * 5
    if x_r % 10 == 0:
        x_r -= 5
    if y_r % 10 == 0:
        y_r -= 5

    if x - x_r > 5:
        x_r += 10

    if y - y_r > 5:
        y_r += 10
    tcd_values = pd.concat(
        [
            tcd_values,
            pd.DataFrame(
                [{"TCD": tcd, "x": x, "y": y, "x_round": x_r, "y_round": y_r}]
            ),
        ],
        ignore_index=True,
    )
log.info(tcd_values)


def get_s2_gt(folder_data: str | Path, months_folders: list[str | int], window: int):
    """
    Encode a tile with MMDC algorithm with a sliding window
    """

    slices_list = get_slices(folder_data, months_folders, window)

    log.info("Getting clouds")
    # s2_dates = check_s2_clouds(folder_data, months_folders)
    # np.save(os.path.join(folder_data, "dates.csv"), s2_dates)
    s2_dates = np.load(os.path.join(folder_data, "dates.npy"))
    print(s2_dates)
    log.info("Done with clouds")

    all_slices_gt = []

    for sliced in slices_list:
        log.info(sliced)
        images = {}
        data = {"s2": {}}
        modality = "Sentinel2"
        xy_matrix = generate_coord_matrix(sliced)
        tcd_values_sliced = tcd_values[
            (tcd_values["x_round"] >= xy_matrix[0].min())
            & (tcd_values["x_round"] <= xy_matrix[0].max())
            & (tcd_values["y_round"] >= xy_matrix[1].min())
            & (tcd_values["y_round"] <= xy_matrix[1].max())
        ]
        if len(tcd_values_sliced) > 0:
            ind = [
                np.array(
                    np.where(
                        np.all(
                            xy_matrix
                            == tcd_values_sliced.iloc[i][["x_round", "y_round"]]
                            .values.astype(float)
                            .reshape(-1, 1, 1),
                            axis=0,
                        )
                    )
                ).reshape(-1)
                for i in range(len(tcd_values_sliced))
            ]
            ind = np.array(ind)

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

            print(all_months_med.shape, ind.shape)

            gt_ref_values = all_months_med[:, ind[:, 0], ind[:, 1]].T
            values_df = pd.DataFrame(
                data=gt_ref_values,
                columns=[b + "_" + m for m in months_folders for b in SEL_BANDS],
            )

            sliced_gt = pd.concat([tcd_values_sliced.reset_index(), values_df], axis=1)

            all_slices_gt.append(sliced_gt)
    return pd.concat(all_slices_gt, ignore_index=True)


gt_df = get_s2_gt(folder_data, months_folders, window)
gt_df.to_csv(os.path.join(folder_data, "gt_tcd_t32tnt_not_fast.csv"))
