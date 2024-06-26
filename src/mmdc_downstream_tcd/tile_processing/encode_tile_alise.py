import logging
import os
import re
import warnings
from pathlib import Path

import numpy as np
import onnxruntime
import pandas as pd
import torch
from einops import rearrange
from openeo_mmdc.dataset.to_tensor import load_transform_one_mod

from mmdc_downstream_tcd.tile_processing.utils import (
    extract_gt_points,
    generate_coord_matrix,
    get_index,
    get_tcd_gt,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

log = logging.getLogger(__name__)


def encode_tile_alise(
    folder_prepared_data: str | Path | None,
    path_alise_model: str | Path,
    path_csv: str | Path,
    output_folder: str | Path,
    gt_file: str | Path = None,
    satellites: list[str] = ["s2"],
):
    """
    Encode a tile with ALISE algorithm with a sliding window
    We divide the sliding window of 768 pixels into 4 overlapping patches
    to overcome memory problems
    """
    sess_opt = onnxruntime.SessionOptions()
    sess_opt.intra_op_num_threads = 16
    transform = load_transform_one_mod(path_csv, mod="s2").transform
    ort_session = onnxruntime.InferenceSession(
        path_alise_model,
        sess_opt,
        providers=["CUDAExecutionProvider"],
    )

    margin = int(re.search(r"m_([0-9]*)", str(folder_prepared_data)).group(1))
    window = int(re.search(r"w_([0-9]*)", str(folder_prepared_data)).group(1))
    log.info(f"Margin={margin}")
    log.info(f"Window={window}")

    s2_dates_final = np.load(
        os.path.join(folder_prepared_data, "s2", "gt_tcd_t32tnt_days_s2.npy")
    )
    reference_date = "2018-03-01"
    doy = torch.Tensor(
        (pd.to_datetime(s2_dates_final) - pd.to_datetime(reference_date)).days
    )

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
                        output_folder, sat, f"gt_tcd_t32tnt_encoded_{sat}_{enum}.csv"
                    )
                ).exists()
                for sat in satellites
            ]
        ):
            log.info(f"Processing patch {enum}")
            data_files = {
                sat: torch.load(
                    os.path.join(folder_prepared_data, sat, f"{sat}_tile_{enum}.pt")
                )
                for sat in satellites
            }

            data = {sat: data_files[sat]["data"] for sat in satellites}

            xy_matrix = data_files[satellites[0]]["matrix"]

            del data_files

            sliced = (
                xy_matrix["x_min"],
                xy_matrix["y_min"],
                xy_matrix["x_max"],
                xy_matrix["y_max"],
            )
            xy_matrix = generate_coord_matrix(sliced)[:, margin:-margin, margin:-margin]

            ind = []
            tcd_values_sliced = None
            if tcd_values is not None:
                ind, tcd_values_sliced = get_index(tcd_values, xy_matrix)

            # s2_ref = data["s2"]["img"][0, :, :, margin:-margin, margin:-margin]
            # s2_mask = data["s2"]["mask"][0, :, :, margin:-margin, margin:-margin]
            s2_ref = data["s2"]["img"][0]

            del data

            encoded_patch = {}

            encoded_patch["s2_lat_mu"] = np.zeros((10, 64, 768, 768))

            t, c, h, w = s2_ref.shape

            # Prepare patch
            s2_ref_norm = rearrange(
                transform(rearrange(s2_ref, "t c h w -> c t h w")),
                "c t h w -> 1 t h w c",
            )

            log.info(s2_ref_norm.shape)
            margin_local = 32
            for x in range(0, window, 352):
                if x + 416 > window:
                    continue
                for y in range(0, window, 352):
                    if y + 416 > window:
                        continue
                    log.info(f"Patch {x}:{x+416}, {y}:{y+416}")
                    sits = s2_ref_norm.numpy()[:, :, y : y + 416, x : x + 416, :]
                    log.info(sits.shape)
                    input = {
                        "sits": sits,
                        "tpe": doy[None, :].numpy(),
                        "padd_mask": torch.zeros(1, sits.shape[1]).bool().numpy(),
                    }

                    ort_out = ort_session.run(None, input)[0]  # (1, 10, 64, 64, 64)
                    encoded_patch["s2_lat_mu"][
                        :,
                        :,
                        y + margin_local : y + 416 - margin_local,
                        x + margin_local : x + 416 - margin_local,
                    ] = ort_out[
                        0, :, :, margin_local:-margin_local, margin_local:-margin_local
                    ]

            encoded_patch["matrix"] = xy_matrix
            encoded_patch["s2_lat_mu"] = encoded_patch["s2_lat_mu"][
                :, :, margin:-margin, margin:-margin
            ]

            encoded_patch["s2_doy"] = np.arange(encoded_patch["s2_lat_mu"].shape[0])

            # Save encoded patch
            torch.save(
                {
                    "img": encoded_patch["s2_lat_mu"].astype(np.float32),
                    "matrix": {
                        "x_min": xy_matrix[0, 0].min(),
                        "x_max": xy_matrix[0, 0].max(),
                        "y_min": xy_matrix[1, :, 0].min(),
                        "y_max": xy_matrix[1, :, 0].max(),
                    },
                },
                os.path.join(output_folder, sat, f"s2_{enum}.pt"),
            )

            # Save GT data
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

            del encoded_patch
