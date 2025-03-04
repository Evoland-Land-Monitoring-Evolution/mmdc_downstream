# flake8: noqa

from __future__ import annotations

import logging
import os
import ssl
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from einops import rearrange

from mmdc_downstrteam_biomass.tile_processing.utils import (
    extract_gt_points,
    generate_coord_matrix,
    get_gt,
    get_index,
    get_valid_dates,
    process_subpatches,
    xarray_to_tensor,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

ssl._create_default_https_context = ssl._create_stdlib_context

log = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def day_number_in_year(date_arr):
    day_number = []
    for date_string in date_arr:
        date_object = pd.to_datetime(date_string.t.values)
        day_number.append(date_object.timetuple().tm_yday)  # Get the day of the year
    return torch.tensor(day_number)


def build_s1_images_and_masks_anysat(
    images: dict[str, torch.Tensor],
    xarray_ds: xr.Dataset,
    s1_type: str = "s1_asc",
) -> dict[str, torch.Tensor]:
    """
    Generate backscatter and validity mask for a S1 data provided by OpenEO
    """

    # We open VH and VV bands from XArray dataset,
    # and we compute VH/VV as the third band
    available_images = xarray_to_tensor(xarray_ds, bands=["VV", "VH"])

    ratio = available_images[:, 0] / available_images[:, 1]

    available_images = torch.cat([available_images, ratio[:, None, :, :]], dim=1)

    # available_images[mask.expand_as(available_images).bool()] = 0

    images[s1_type] = available_images

    # generate nan mask
    nan_mask = torch.isnan(images[s1_type]).sum(dim=1).to(torch.int8) > 0
    images[f"{s1_type}_valmask"] = nan_mask

    return images


def encode_tile_biomass_anysat(
    folder_data: str | Path,
    output_folder: str | Path,
    gt_file: str | Path = None,
    satellites: list[str] = ["s1_asc"],
):
    """
    Encode a tile with ALISE algorithm with a sliding window
    We divide the sliding window of 768 pixels into 4 overlapping patches
    to overcome memory problems
    """

    biomass_values_all_years = get_gt(
        gt_file, variables=["AGB_t/ha", "MeanHeight", "Year", "class"]
    ).drop(columns=["x", "y"])
    log.info(f"Full dataset length {len(biomass_values_all_years)}")
    biomass_values_all_years = biomass_values_all_years.loc[
        biomass_values_all_years["class"].isin([2, 3, 4, 5])
    ]
    log.info(f"Forest dataset length {len(biomass_values_all_years)}")

    for year in range(2015, 2022):
        biomass_values = biomass_values_all_years[
            biomass_values_all_years["Year"] == year
        ]

        tile_path = os.path.join(folder_data, str(year))
        for sat in satellites:
            if Path(tile_path).exists() and os.listdir(tile_path):
                tiles = [f for f in os.listdir(tile_path) if f.startswith("3")]

                model = torch.hub.load(
                    "gastruc/anysat",
                    "anysat",
                    pretrained=True,
                    force_reload=False,
                    flash_attn=False,
                    device=DEVICE,
                )

                modality = "Sentinel1_ASCENDING" if "1" in sat else "Sentinel2"

                for tile in np.sort(tiles):
                    sliced_gt = {}

                    files = [
                        f
                        for f in os.listdir(os.path.join(folder_data, str(year), tile))
                        if f.endswith(".nc")
                    ]

                    Path(os.path.join(output_folder, sat, str(year), tile)).mkdir(
                        exist_ok=True, parents=True
                    )
                    patches_list = np.sort(np.array(files))
                    for enum, patch_name in enumerate(patches_list):
                        nc_file = os.path.join(folder_data, str(year), tile, patch_name)
                        if not Path(
                            os.path.join(
                                output_folder,
                                sat,
                                str(year),
                                tile,
                                f"gt_tcd_t32tnt_encoded_{sat}_{tile}_{enum}.csv",
                            )
                        ).exists():
                            log.info(f"Processing patch {nc_file}")

                            # Open netcdf as xarray dataset,
                            # get only valid dates without nodata
                            with xr.open_dataset(
                                nc_file,
                                decode_coords="all",
                                mask_and_scale=True,
                                decode_times=True,
                                engine="netcdf4",
                            ) as xarray_ds:
                                print(year, tile, nc_file)
                                # src = xarray_ds.crs
                                xarray_ds_valid = get_valid_dates(
                                    xarray_ds,
                                    percent_valid=0.5 if sat == "s1_asc" else 0.3,
                                )

                                if xarray_ds_valid.x.shape[0] % 16 != 0:
                                    to_distribute = xarray_ds_valid.x.shape[0] % 16
                                    xarray_ds_valid = xarray_ds_valid.isel(
                                        x=slice(
                                            int(to_distribute / 2),
                                            -(to_distribute - int(to_distribute / 2)),
                                        )
                                    )
                                if xarray_ds_valid.y.shape[0] % 16 != 0:
                                    to_distribute = xarray_ds_valid.y.shape[0] % 16
                                    xarray_ds_valid = xarray_ds_valid.isel(
                                        y=slice(
                                            int(to_distribute / 2),
                                            -(to_distribute - int(to_distribute / 2)),
                                        )
                                    )

                                dates = xarray_ds_valid.t
                                if len(dates) < 3:
                                    continue

                            doy = day_number_in_year(dates)

                            # data = {}
                            images = {}
                            # data[sat] = {}

                            if sat == "s1_asc":
                                images = build_s1_images_and_masks_anysat(
                                    images, xarray_ds_valid
                                )
                                to_keep = (
                                    torch.concat(
                                        (doy[1:] - doy[:-1], torch.Tensor([10000])), 0
                                    )
                                    > 2
                                )

                                doy = doy[to_keep]
                                image_satellite = images["s1_asc"][to_keep].nan_to_num()

                            else:
                                dates = pd.DatetimeIndex(dates).to_frame(name="date_s2")

                                # images = build_s2_images_and_masks(
                                #     images, xarray_ds_valid, dates
                                # )
                                image_satellite = images["s2_set"].nan_to_num()

                            log.info(f"Encoding patch {enum}")

                            encoded_patch = {}

                            xy_matrix = generate_coord_matrix(xarray_ds_valid)

                            t, c, h, w = image_satellite.shape
                            encoded_patch[f"{sat}_lat_mu"] = np.zeros((768, h, w))

                            MEAN_S1 = torch.tensor(
                                [
                                    3.2893013954162598,
                                    -3.682938814163208,
                                    0.6116273403167725,
                                ]
                            ).float()
                            STD_S1 = torch.tensor(
                                [
                                    40.11152267456055,
                                    40.535335540771484,
                                    1.0343183279037476,
                                ]
                            ).float()

                            # Prepare patch

                            s1_ref_norm = (
                                rearrange(image_satellite, "t c h w -> 1 t c h w")
                                - MEAN_S1[:, None, None]
                            ) / STD_S1[:, None, None]
                            del image_satellite
                            log.info(s1_ref_norm.shape)

                            local_patch = 16
                            margin_local = 0

                            if (
                                s1_ref_norm.shape[-1] > local_patch
                                or s1_ref_norm.shape[-2] > local_patch
                            ):
                                (
                                    start_margin_x,
                                    start_margin_y,
                                    nb_x,
                                    nb_y,
                                ) = process_subpatches(
                                    local_patch, margin_local, s1_ref_norm
                                )

                                for x in range(
                                    start_margin_x,
                                    nb_x * (local_patch - 2 * margin_local),
                                    local_patch - 2 * margin_local,
                                ):
                                    for y in range(
                                        start_margin_y,
                                        nb_y * (local_patch - 2 * margin_local),
                                        local_patch - 2 * margin_local,
                                    ):
                                        print(
                                            f"Patch {x}:{x + local_patch}, "
                                            f"{y}:{y + local_patch}"
                                        )
                                        sits = s1_ref_norm[
                                            :,
                                            :,
                                            :,
                                            y : y + local_patch,
                                            x : x + local_patch,
                                        ]
                                        print(sits.shape)

                                        data = {
                                            "s1": sits.to(DEVICE).nan_to_num(),
                                            # 1 bs, 60 dates, 3 channels, 6x6 pixels
                                            "s1_dates": doy.unsqueeze(0).to(DEVICE),
                                        }

                                        del sits

                                        ort_out = rearrange(
                                            model(
                                                data,
                                                patch_size=10,
                                                output="dense",
                                                output_modality="s1",
                                            ),
                                            "b h w f -> b f h w",
                                        )[:, 768:]
                                        print(ort_out.shape)
                                        encoded_patch[f"{sat}_lat_mu"][
                                            :,
                                            y : y + local_patch,
                                            x : x + local_patch,
                                        ] = (
                                            ort_out[0].cpu().detach()
                                        )

                                xy_matrix = xy_matrix[
                                    :,
                                    start_margin_y
                                    + margin_local : y
                                    + local_patch
                                    - margin_local,
                                    start_margin_x
                                    + margin_local : x
                                    + local_patch
                                    - margin_local,
                                ]

                                encoded_patch[f"{sat}_lat_mu"] = torch.Tensor(
                                    encoded_patch[f"{sat}_lat_mu"][
                                        :,
                                        start_margin_y
                                        + margin_local : y
                                        + local_patch
                                        - margin_local,
                                        start_margin_x
                                        + margin_local : x
                                        + local_patch
                                        - margin_local,
                                    ]
                                ).unsqueeze(0)
                                images["s1_asc_valmask"] = images["s1_asc_valmask"][
                                    ...,
                                    start_margin_y
                                    + margin_local : y
                                    + local_patch
                                    - margin_local,
                                    start_margin_x
                                    + margin_local : x
                                    + local_patch
                                    - margin_local,
                                ]
                            else:
                                data = {
                                    "s1": s1_ref_norm.to(DEVICE),
                                    "s1_dates": doy.unsqueeze(0).to(DEVICE),
                                }

                                ort_out = rearrange(
                                    model(
                                        data,
                                        patch_size=10,
                                        output="dense",
                                        output_modality="s1",
                                    ),
                                    "b h w f -> b f h w",
                                )[:, 768:]

                                encoded_patch[f"{sat}_lat_mu"] = ort_out.cpu().detach()

                            if "1" in modality:
                                mask_non_valid = (
                                    images["s1_asc_valmask"].sum(0)
                                    > images["s1_asc_valmask"].shape[0] * 0.66
                                )
                            else:
                                mask_non_valid = (
                                    images["s2_masks"].sum(0)
                                    == images["s2_masks"].shape[0]
                                )
                            encoded_patch[f"{sat}_lat_mu"][
                                mask_non_valid[None, ...].expand_as(
                                    encoded_patch[f"{sat}_lat_mu"]
                                )
                            ] = torch.nan

                            ind = []
                            biomass_values_sliced = None

                            if biomass_values is not None:
                                biomass_values_tile = get_gt(
                                    biomass_values,
                                    variables=[
                                        "AGB_t/ha",
                                        "MeanHeight",
                                        "Year",
                                        "class",
                                    ],
                                    crs=xarray_ds_valid.crs.crs_wkt,
                                )
                                ind, biomass_values_sliced = get_index(
                                    biomass_values_tile, xy_matrix
                                )

                            encoded_patch["matrix"] = xy_matrix

                            encoded_patch[f"{sat}_doy"] = np.arange(
                                encoded_patch[f"{sat}_lat_mu"].shape[0]
                            )
                            import matplotlib.pyplot as plt

                            plt.rcParams["backend"]
                            f, axarr = plt.subplots(5, 5, figsize=(80, 80))

                            for k in range(5):
                                for m in range(5):
                                    axarr[k, m].imshow(
                                        (
                                            encoded_patch[f"{sat}_lat_mu"][
                                                0,
                                                (k * 20 + m) * 3 : (k * 20 + m) * 3 + 3,
                                            ].permute(1, 2, 0)
                                            + 3
                                        )
                                        / 6
                                    )
                            plt.show()

                            # Save encoded patch
                            torch.save(
                                {
                                    "img": torch.Tensor(encoded_patch[f"{sat}_lat_mu"]),
                                    "matrix": {
                                        "x_min": encoded_patch["matrix"][0, 0].min(),
                                        "x_max": encoded_patch["matrix"][0, 0].max(),
                                        "y_min": encoded_patch["matrix"][1, :, 0].min(),
                                        "y_max": encoded_patch["matrix"][1, :, 0].max(),
                                    },
                                },
                                os.path.join(
                                    output_folder,
                                    sat,
                                    str(year),
                                    tile,
                                    f"{sat}_{tile}_{enum}.pt",
                                ),
                            )

                            # Save GT data
                            if ind.size > 0 and biomass_values_sliced is not None:
                                for sat in satellites:
                                    df_gt, days = extract_gt_points(
                                        encoded_patch,
                                        sat,
                                        ind,
                                        biomass_values_sliced,
                                        sliced_gt,
                                    )

                                    print(df_gt)
                                    df_gt.to_csv(
                                        os.path.join(
                                            output_folder,
                                            sat,
                                            str(year),
                                            tile,
                                            f"gt_tcd_t32tnt_encoded_{sat}_{tile}_{enum}.csv",
                                        )
                                    )
                                    if not Path(
                                        os.path.join(
                                            output_folder,
                                            sat,
                                            str(year),
                                            tile,
                                            f"gt_tcd_t32tnt_days_{sat}_{tile}.npy",
                                        )
                                    ).exists():
                                        np.save(
                                            os.path.join(
                                                output_folder,
                                                sat,
                                                str(year),
                                                tile,
                                                f"gt_tcd_t32tnt_days_{sat}_{tile}.npy",
                                            ),
                                            days,
                                        )

                            del encoded_patch
