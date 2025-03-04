from __future__ import annotations

import json
import logging
import os
import re
import ssl
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mmdc_downstrteam_biomass.tile_processing.utils import extract_gt_points, get_index

ssl._create_default_https_context = ssl._create_stdlib_context

warnings.filterwarnings("ignore", category=DeprecationWarning)

log = logging.getLogger(__name__)


def get_tcd_gt(gt_path: str, variable: str = "TCD") -> pd.DataFrame:
    """
    Get coordinates of GT points and recompute them to the center of pixel
    """
    df = pd.DataFrame(columns=[variable, "x", "y", "x_round", "y_round"])
    with open(gt_path) as f:
        data = json.load(f)

    for feature in data["features"]:
        if feature["geometry"]["type"] == "MultiPoint":
            x, y = feature["geometry"]["coordinates"][0]
        else:  # Point
            x, y = feature["geometry"]["coordinates"]
        value = feature["properties"][variable]

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

        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [{variable: value, "x": x, "y": y, "x_round": x_r, "y_round": y_r}]
                ),
            ],
            ignore_index=True,
        )
    print(df.head(20))
    return df


model = torch.hub.load(
    "gastruc/anysat",
    "anysat",
    pretrained=True,
    force_reload=False,
    flash_attn=False,
    device="cuda",
)


def day_number_in_year(date_arr):
    day_number = []
    for date_string in date_arr:
        print(date_string)
        date_object = pd.to_datetime(date_string)
        day_number.append(date_object.timetuple().tm_yday)  # Get the day of the year
    return torch.tensor(day_number)


def generate_coord_matrix(sliced: tuple[float, float, float, float]) -> np.ndarray:
    """
    Generate a matrix with x,y coordinates for each pixel
    """
    slice_xmin, slice_ymin, slice_xmax, slice_ymax = sliced
    slice_ymin -= 10
    slice_xmax += 10

    x_row = np.arange(slice_xmin, slice_xmax, 10)
    y_col = np.arange(slice_ymax, slice_ymin, -10)

    x_coords = np.repeat([x_row], repeats=len(y_col), axis=0)
    y_coords = np.repeat(y_col.reshape(-1, 1), repeats=len(x_row), axis=1)
    return np.stack([x_coords, y_coords])


MEAN_S2 = torch.tensor(
    [
        1165.9398193359375,
        1375.6534423828125,
        1429.2191162109375,
        1764.798828125,
        2719.273193359375,
        3063.61181640625,
        3205.90185546875,
        3319.109619140625,
        2422.904296875,
        1639.370361328125,
    ]
).float()
STD_S2 = torch.tensor(
    [
        1942.6156005859375,
        1881.9234619140625,
        1959.3798828125,
        1867.2239990234375,
        1754.5850830078125,
        1769.4046630859375,
        1784.860595703125,
        1767.7100830078125,
        1458.963623046875,
        1299.2833251953125,
    ]
).float()


def encode_tile_anysat(
    folder_prepared_data: str | Path | None,
    output_folder: str | Path,
    gt_file: str | Path = None,
    satellites: list[str] = ["s2"],
):
    """
    Encode a tile with MMDC algorithm with a sliding window
    """
    s2_dates = torch.load(os.path.join(folder_prepared_data, "s2", "s2_dates.pt"))[
        "date_s2"
    ].values
    s2_dates = day_number_in_year(list(s2_dates))

    margin_global = int(re.search(r"m_([0-9]*)", str(folder_prepared_data)).group(1))
    log.info(f"Margin= {margin_global}")

    margin = 32

    tcd_values = None
    if gt_file is not None:
        tcd_values = get_tcd_gt(gt_file)

    sliced_gt = {}
    # for enum, sliced in enumerate(slices_list[::-1]):
    #     enum = len(slices_list) - enum - 1

    # TODO check
    for sat in satellites:
        Path(os.path.join(output_folder, sat)).mkdir(exist_ok=True, parents=True)

    available_tiles = [
        f
        for f in os.listdir(os.path.join(folder_prepared_data, satellites[0]))
        if f.startswith(f"{satellites[0]}_tile_") and f.endswith(".pt")
    ]

    for enum in range(len(available_tiles)):
        if not np.all(
            [
                Path(
                    os.path.join(
                        output_folder, f"gt_tcd_t32tnt_encoded_anysat_{sat}_{enum}.csv"
                    )
                ).exists()
                for sat in satellites
            ]
        ):
            xy_matrix = torch.load(
                os.path.join(
                    folder_prepared_data,
                    satellites[0],
                    f"{satellites[0]}_tile_{enum}.pt",
                )
            )["matrix"]
            sliced = (
                xy_matrix["x_min"],
                xy_matrix["y_min"],
                xy_matrix["x_max"],
                xy_matrix["y_max"],
            )
            xy_matrix = generate_coord_matrix(sliced)[
                :, margin_global:-margin_global, margin_global:-margin_global
            ]

            data = {
                sat: torch.load(
                    os.path.join(folder_prepared_data, sat, f"{sat}_tile_{enum}.pt")
                )["data"]
                for sat in satellites
            }

            ind = []
            tcd_values_sliced = None
            if tcd_values is not None:
                ind, tcd_values_sliced = get_index(tcd_values, xy_matrix)

            s2_ref = data["s2"]["img"][0, :, :, margin:-margin, margin:-margin]
            # s2_mask = data["s2"]["mask"][0, :, :, margin:-margin, margin:-margin]

            # aerial = (aerial - MEAN_AERIAL[:, None, None]) / STD_AERIAL[:, None, None]
            s2 = (s2_ref - MEAN_S2[:, None, None]) / STD_S2[:, None, None]

            final_encoded = None
            size = 32
            for xx in range(0, s2_ref.shape[-1], size):
                for yy in range(0, s2_ref.shape[-1], size):
                    data = {
                        # "spot": aerial.unsqueeze(0),
                        # #1 batch size, 4 channels, 300x300 pixels
                        "s2": s2[..., yy : yy + size, xx : xx + size]
                        .unsqueeze(0)
                        .to("cuda"),
                        # 1 batch size, 146 dates, 10 channels, 6x6 pixels
                        "s2_dates": s2_dates.unsqueeze(0).to("cuda"),
                    }

                    with torch.no_grad():
                        features = model(
                            data, patch_size=20, output="dense", output_modality="s2"
                        ).permute(0, 3, 1, 2)[:, 768:]

                    if final_encoded is None:
                        final_encoded = torch.zeros(
                            (*features.shape[:2], *s2.shape[-2:])
                        )

                    final_encoded[:, :, yy : yy + size, xx : xx + size] = features.cpu()
                    del features

            del data

            final_margin = 8

            encoded_patch = {}

            t, c, h, w = s2_ref.shape

            encoded_patch["s2_lat_mu"] = final_encoded[
                :, :, final_margin:-final_margin, final_margin:-final_margin
            ]

            encoded_patch["matrix"] = xy_matrix

            encoded_patch["s2_doy"] = np.arange(encoded_patch["s2_lat_mu"].shape[0])

            # encoded_patch["s2_lat_mu"][
            #     s2_mask.bool().expand_as(encoded_patch["s2_lat_mu"])
            # ] = torch.nan

            # torch.save(
            #     {
            #         "img": torch.Tensor(encoded_patch[f"s2_lat_mu"])
            #         .cpu()
            #         .numpy()
            #         .astype(np.float32),
            #         "matrix": {
            #             "x_min": xy_matrix[0, 0].min(),
            #             "x_max": xy_matrix[0, 0].max(),
            #             "y_min": xy_matrix[1, :, 0].min(),
            #             "y_max": xy_matrix[1, :, 0].max(),
            #         },
            #     },
            #     os.path.join(output_folder, f"s2_{enum}.pt"),
            # )

            if ind.size > 0 and tcd_values_sliced is not None:
                for sat in satellites:
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
                            f"gt_tcd_t32tnt_encoded_{sat}_{enum}.csv",
                        )
                    )
                    if not Path(
                        os.path.join(output_folder, f"gt_tcd_t32tnt_days_{sat}.npy")
                    ).exists():
                        np.save(
                            os.path.join(
                                output_folder, f"gt_tcd_t32tnt_days_{sat}.npy"
                            ),
                            days,
                        )

                # torch.save(encoded_patch,
                #            os.path.join(output_folder, f"Encoded_patch_{enum}.pt"))

                del encoded_patch
