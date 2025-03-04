import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from einops import rearrange
from mmdc_singledate.datamodules.components.datamodule_components import (
    build_s2_images_and_masks,
)
from mmmv_ssl.data.dataclass import BatchOneMod
from openeo_mmdc.dataset.to_tensor import load_transform_one_mod

from mmdc_downstrteam_biomass.tile_processing.utils import (
    build_s1_images_and_masks_biomass,
    extract_gt_points,
    generate_coord_matrix,
    get_gt,
    get_index,
    get_valid_dates,
    process_subpatches,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

log = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def encode_tile_biomass_malice(
    folder_data: str | Path,
    path_alise_model: str | Path,
    path_csv: str | Path,
    output_folder: str | Path,
    gt_file: str | Path = None,
    sat: str = "s1_asc",
):
    """
    Encode a tile with ALISE algorithm with a sliding window
    We divide the sliding window of 768 pixels into 4 overlapping patches
    to overcome memory problems
    """

    if path_alise_model.endswith("onnx"):
        import onnxruntime
    else:
        from mmdc_downstream_pastis.encode_series.encode_alise_s1_original_pt import (
            load_checkpoint,
        )

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
        transform = load_transform_one_mod(path_csv, mod=sat.lower()).transform
        if Path(tile_path).exists() and os.listdir(tile_path):
            tiles = [f for f in os.listdir(tile_path) if f.startswith("3")]

            if path_alise_model.endswith("onnx"):
                sess_opt = onnxruntime.SessionOptions()
                sess_opt.intra_op_num_threads = 16
                ort_session = onnxruntime.InferenceSession(
                    path_alise_model,
                    sess_opt,
                    providers=["CUDAExecutionProvider"],
                )
            else:
                hydra_conf = path_alise_model.split(".")[0] + "_config.yaml"

                repr_encoder = load_checkpoint(hydra_conf, path_alise_model, sat).to(
                    DEVICE
                )

            modality = "Sentinel1_ASCENDING" if "1" in sat else "Sentinel2"

            for tile in np.sort(tiles):
                reference_date = "2014-03-03"

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
                            # if xarray_ds_valid.x.shape[0] > 512:
                            #     to_distribute = xarray_ds_valid.x.shape[0] - 512
                            #     xarray_ds_valid = xarray_ds_valid.isel(
                            #         x=slice(
                            #             int(to_distribute / 2),
                            #             -(to_distribute - int(to_distribute / 2)),
                            #         )
                            #     )
                            # if xarray_ds_valid.y.shape[0] > 512:
                            #     to_distribute = xarray_ds_valid.y.shape[0] - 512
                            #     xarray_ds_valid = xarray_ds_valid.isel(
                            #         y=slice(
                            #             int(to_distribute / 2),
                            #             -(to_distribute - int(to_distribute / 2)),
                            #         )
                            #     )
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

                            # _, slice_ymin, slice_xmax, _ = (
                            #     xarray_ds_valid.x.min(),
                            #     xarray_ds_valid.y.min(),
                            #     xarray_ds_valid.x.max(),
                            #     xarray_ds_valid.y.max(),
                            # )
                            # slice_ymin -= 10
                            # slice_xmax += 10

                        doy = torch.Tensor(
                            (
                                pd.to_datetime(dates) - pd.to_datetime(reference_date)
                            ).days
                        )

                        # data = {}
                        images = {}
                        # data[sat] = {}

                        if sat == "s1_asc":
                            images = build_s1_images_and_masks_biomass(
                                images, xarray_ds_valid
                            )
                            to_keep = (
                                torch.concat(
                                    (doy[1:] - doy[:-1], torch.Tensor([10000])), 0
                                )
                                > 2
                            )

                            doy = doy[to_keep]
                            image_satellite = images["s1_asc"][to_keep]

                        else:
                            dates = pd.DatetimeIndex(dates).to_frame(name="date_s2")

                            images = build_s2_images_and_masks(
                                images, xarray_ds_valid, dates
                            )
                            image_satellite = images["s2_set"]

                        log.info(f"Encoding patch {enum}")

                        encoded_patch = {}

                        xy_matrix = generate_coord_matrix(xarray_ds_valid)

                        t, c, h, w = image_satellite.shape
                        encoded_patch[f"{sat}_lat_mu"] = np.zeros((10, 64, h, w))

                        # Prepare patch
                        s1_ref_norm = rearrange(
                            transform(rearrange(image_satellite, "t c h w -> c t h w")),
                            "c t h w -> 1 t c h w",
                        )
                        del image_satellite
                        log.info(s1_ref_norm.shape)

                        local_patch = 256 + 32
                        margin_local = 16

                        if (
                            s1_ref_norm.shape[-1] > local_patch + 32
                            or s1_ref_norm.shape[-2] > local_patch + 32
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
                                # if x + local_patch > w:
                                #     continue
                                for y in range(
                                    start_margin_y,
                                    nb_y * (local_patch - 2 * margin_local),
                                    local_patch - 2 * margin_local,
                                ):
                                    # if y + local_patch > h:
                                    #     continue
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

                                    if path_alise_model.endswith("onnx"):
                                        input = {
                                            "sits": sits.numpy(),
                                            "tpe": doy[None, :].numpy(),
                                            "padd_mask": torch.zeros(1, sits.shape[1])
                                            .bool()
                                            .numpy(),
                                        }

                                        ort_out = ort_session.run(None, input)[
                                            0
                                        ]  # (1, 10, 64, 64, 64)
                                    else:
                                        input = BatchOneMod(
                                            sits=sits.to(DEVICE),
                                            input_doy=doy[None, :].to(DEVICE),
                                            padd_index=torch.zeros(1, sits.shape[1])
                                            .bool()
                                            .to(DEVICE),
                                        )
                                        del sits
                                        with torch.no_grad():
                                            ort_out = (
                                                repr_encoder.forward_keep_input_dim(
                                                    input
                                                ).repr.cpu()
                                            )
                                        del input
                                    if margin_local > 0:
                                        encoded_patch[f"{sat}_lat_mu"][
                                            :,
                                            :,
                                            y
                                            + margin_local : y
                                            + local_patch
                                            - margin_local,
                                            x
                                            + margin_local : x
                                            + local_patch
                                            - margin_local,
                                        ] = ort_out[
                                            0,
                                            :,
                                            :,
                                            margin_local:-margin_local,
                                            margin_local:-margin_local,
                                        ]
                                    else:
                                        encoded_patch[f"{sat}_lat_mu"][
                                            :,
                                            :,
                                            y : y + local_patch,
                                            x : x + local_patch,
                                        ] = ort_out[0]

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
                            )
                            images["s2_masks"] = images["s2_masks"][
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
                            if path_alise_model.endswith("onnx"):
                                input = {
                                    "sits": s1_ref_norm.numpy(),
                                    "tpe": doy[None, :].numpy(),
                                    "padd_mask": torch.zeros(1, s1_ref_norm.shape[1])
                                    .bool()
                                    .numpy(),
                                }

                                ort_out = ort_session.run(None, input)[
                                    0
                                ]  # (1, 10, 64, 64, 64)
                            else:
                                input = BatchOneMod(
                                    sits=s1_ref_norm.to(DEVICE).nan_to_num(),
                                    input_doy=doy[None, :].to(DEVICE),
                                    padd_index=torch.zeros(1, s1_ref_norm.shape[1])
                                    .bool()
                                    .to(DEVICE),
                                )

                                with torch.no_grad():
                                    ort_out = repr_encoder.forward_keep_input_dim(
                                        input
                                    ).repr.cpu()
                                del input

                            encoded_patch[f"{sat}_lat_mu"] = ort_out[0]
                        if "1" in modality:
                            mask_non_valid = (
                                images["s1_asc_valmask"].sum(0)
                                > images["s1_asc_valmask"].shape[0] * 0.66
                            )
                        else:
                            mask_non_valid = (
                                images["s2_masks"].sum(0) == images["s2_masks"].shape[0]
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
