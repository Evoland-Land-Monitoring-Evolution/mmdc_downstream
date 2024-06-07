import logging
import os
import re
import warnings
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

from mmdc_downstream_pastis.datamodule.datatypes import METEO_BANDS
from mmdc_downstream_tcd.tile_processing.utils import (
    extract_gt_points,
    generate_coord_matrix,
    get_index,
    get_tcd_gt,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

log = logging.getLogger(__name__)

meteo_modalities = METEO_BANDS.keys()


def mmdc2pvae_batch(s2_ref, s2_angles) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO: use the bands selected for the model …
    # if (nb_nans := torch.isnan(s2_ref.view(-1)).sum()) > 0:
    #     log.info(f"{nb_nans=}")
    #     print(f"{nb_nans=}")
    # if (nb_nans_a := torch.isnan(s2_angles.view(-1)).sum()) > 0:
    #     log.info(f"{nb_nans_a=}")
    #     print(f"{nb_nans_a=}")
    s2_x = (
        s2_ref.nan_to_num() / 10000
    )  # MMDC S2 reflectances are *1000, while PVAE expects [0-1]
    s2_a = s2_angles.nan_to_num()
    return s2_x, torch.cat(
        (
            torch.rad2deg(torch.acos(s2_a[:, 0:1, ...])),  # sun_zen
            torch.rad2deg(torch.acos(s2_a[:, 3:4, ...])),  # view_zen
            torch.rad2deg(
                torch.acos(s2_a[:, 1:2, ...]) - torch.acos(s2_a[:, 4:5, ...])
            ),  # sun_az-view_az
        ),
        dim=1,
    )


def encode_tile_pvae(
    folder_data: str | Path,
    folder_prepared_data: str | Path | None,
    path_jit_model: str | Path,
    output_folder: str | Path,
    gt_file: str | Path = None,
    satellites: list[str] = ["s2"],
):
    """
    Encode a tile with MMDC algorithm with a sliding window
    """

    ts_net = torch.jit.load(path_jit_model)
    ts_net.eval()
    log.info(ts_net)

    margin = int(re.search(r"m_([0-9]*)", str(folder_prepared_data)).group(1))
    log.info(f"Margin= {margin}")

    s2_dates = torch.load(os.path.join(folder_prepared_data, "s2", "s2_dates.pt"))[
        "date_s2"
    ].values

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
                        output_folder, f"gt_tcd_t32tnt_encoded_pvae_{sat}_{enum}.csv"
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
            xy_matrix = generate_coord_matrix(sliced)[:, margin:-margin, margin:-margin]

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
            s2_angles = data["s2"]["angles"][0, :, :, margin:-margin, margin:-margin]
            s2_mask = data["s2"]["mask"][0, :, :, margin:-margin, margin:-margin]

            del data

            encoded_patch = {}
            encoded_patch["s2_mask"] = s2_mask

            t, c, h, w = s2_ref.shape

            # Prepare patch
            s2_ref, s2_angles = mmdc2pvae_batch(s2_ref, s2_angles)

            s2_ref_reshape = rearrange(s2_ref, "t c h w ->  (t h w) c")
            s2_angles_reshape = rearrange(s2_angles, "t c h w ->  (t h w) c")

            res = ts_net(s2_ref_reshape, s2_angles_reshape)
            res_shaped = rearrange(res, "(t h w) c ml  ->  t c ml h w", h=h, w=w)

            encoded_patch["s2_lat_mu"] = res_shaped[:, :, 0, :, :].detach()
            encoded_patch["s2_lat_logvar"] = res_shaped[:, :, 1, :, :].detach()

            del res, res_shaped, s2_ref_reshape, s2_angles_reshape, s2_ref, s2_angles

            encoded_patch["s2_doy"] = s2_dates

            encoded_patch["matrix"] = xy_matrix

            encoded_patch["s2_lat_mu"][
                s2_mask.bool().expand_as(encoded_patch["s2_lat_mu"])
            ] = torch.nan
            encoded_patch["s2_lat_logvar"][
                s2_mask.bool().expand_as(encoded_patch["s2_lat_logvar"])
            ] = torch.nan

            # days = encoded_patch[f"{sat}_doy"]

            torch.save(
                {
                    "img": torch.concat(
                        [
                            encoded_patch["s2_lat_mu"],
                            encoded_patch["s2_lat_logvar"],
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
                os.path.join(output_folder, f"s2_{enum}.pt"),
            )

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
