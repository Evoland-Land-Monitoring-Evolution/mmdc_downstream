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
    xarray_to_tensor,
)
from mmdc_singledate.datamodules.constants import ANGLES_S1
from openeo_mmdc.dataset.to_tensor import load_transform_one_mod

from mmdc_downstream_tcd.tile_processing.utils import (
    extract_gt_points,
    get_index,
    get_tcd_gt,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

log = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_valid_dates(x_array, percent_valid: float = 0.75):
    max_pix_valid = x_array.sizes["y"] * x_array.sizes["x"] * percent_valid

    # if "dataMask" in [x for x in x_array.data_vars]:
    #     valid = (x_array["dataMask"] == 1).sum(dim=["x", "y"]).compute()
    #     t_sel = valid.where(valid >= max_pix_valid, drop=True)["t"]
    # else:
    non_valid_vars = [(x_array[x].isnull()).values for x in x_array.data_vars]
    valid = (np.stack(non_valid_vars, 1).sum(1) == 0).sum((-1, -2))
    t_sel = x_array.t[valid > max_pix_valid]
    x_array = x_array.sel(t=t_sel)

    return x_array


def build_s1_images_and_masks_biomass(
    images: dict[str, torch.Tensor],
    xarray_ds: xr.Dataset,
    s1_type: str = "s1_asc",
) -> dict[str, torch.Tensor]:
    """
    Generate backscatter and validity mask for a S1 data provided by OpenEO
    """

    # We open VH and VV bands from XArray dataset,
    # and we compute VH/VV as the third band
    available_images = xarray_to_tensor(xarray_ds, bands=["VH", "VV"])

    ratio = available_images[:, 0] / available_images[:, 1]

    available_images = torch.cat([available_images, ratio[:, None, :, :]], dim=1)

    available_images = torch.log(available_images)

    # available_images[mask.expand_as(available_images).bool()] = 0

    images[s1_type] = available_images

    available_angles = torch.cos(
        torch.deg2rad(xarray_to_tensor(xarray_ds, bands=ANGLES_S1).squeeze(1))
    )

    images[f"{s1_type}_angles"] = available_angles

    # generate nan mask
    nan_mask = torch.isnan(images[s1_type]).sum(dim=1).to(torch.int8) > 0
    images[f"{s1_type}_valmask"] = nan_mask

    return images


def generate_coord_matrix(xarray_ds) -> np.ndarray:
    """
    Generate a matrix with x,y coordinates for each pixel
    """
    slice_xmin, slice_ymin, slice_xmax, slice_ymax = (
        xarray_ds.x.min(),
        xarray_ds.y.min(),
        xarray_ds.x.max(),
        xarray_ds.y.max(),
    )
    slice_ymin -= 10
    slice_xmax += 10

    x_row = np.arange(slice_xmin, slice_xmax, 10)
    y_col = np.arange(slice_ymax, slice_ymin, -10)

    x_coords = np.repeat([x_row], repeats=len(y_col), axis=0)
    y_coords = np.repeat(y_col.reshape(-1, 1), repeats=len(x_row), axis=1)
    return np.stack([x_coords, y_coords])


def encode_tile_biomass_alise_s1(
    folder_data: str | Path,
    path_alise_model: str | Path,
    path_csv: str | Path,
    output_folder: str | Path,
    gt_file: str | Path = None,
    satellites: list[str] = ["s1_asc"],
):
    """
    Encode a tile with ALISE algorithm with a sliding window
    We divide the sliding window of 768 pixels into 4 overlapping patches
    to overcome memory problems
    """

    if path_alise_model.endswith("onnx"):
        import onnxruntime
    else:
        from mt_ssl.data.classif_class import ClassifBInput

        from mmdc_downstream_pastis.encode_series.encode_malice_pt import (
            load_checkpoint,
        )

    reference_date = "2014-03-03"

    biomass_values = None
    if gt_file is not None:
        biomass_values = get_tcd_gt(gt_file, variable="AGB_T_HA").drop(
            columns=["x", "y"]
        )
        height_values = get_tcd_gt(
            gt_file.replace("AGB", "FCH"), variable="MeanHeight"
        ).drop(columns=["x", "y"])
        biomass_values = biomass_values.merge(
            height_values,
            how="outer",
            on=["x_round", "y_round"],
        )

    sliced_gt = {}

    for sat in satellites:
        transform = load_transform_one_mod(path_csv, mod=sat).transform

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

            repr_encoder = load_checkpoint(
                hydra_conf, path_alise_model, satellites[0]
            ).to(DEVICE)
        modality = "Sentinel1_ASCENDING" if sat == "s1_asc" else "Sentinel2"
        tiles = os.listdir(os.path.join(folder_data, modality))

        for tile in tiles:
            Path(os.path.join(output_folder, sat, tile)).mkdir(
                exist_ok=True, parents=True
            )
            patches_list = np.sort(
                np.array(
                    [
                        f
                        for f in os.listdir(os.path.join(folder_data, modality, tile))
                        if f.endswith(".nc")
                    ]
                )
            )
            for enum, patch_name in enumerate(patches_list):
                nc_file = os.path.join(folder_data, modality, tile, patch_name)
                if not Path(
                    os.path.join(
                        output_folder,
                        sat,
                        tile,
                        f"gt_tcd_t32tnt_encoded_{sat}_{tile}_{enum}.csv",
                    )
                ).exists():
                    log.info(f"Processing patch {enum}")

                    # Open netcdf as xarray dataset, get only valid dates without nodata
                    with xr.open_dataset(
                        nc_file,
                        decode_coords="all",
                        mask_and_scale=True,
                        decode_times=True,
                        engine="netcdf4",
                    ) as xarray_ds:
                        # src = xarray_ds.crs
                        xarray_ds_valid = get_valid_dates(
                            xarray_ds, percent_valid=0.9 if sat == "s1_asc" else 0.75
                        )
                        dates = xarray_ds_valid.t
                        _, slice_ymin, slice_xmax, _ = (
                            xarray_ds_valid.x.min(),
                            xarray_ds_valid.y.min(),
                            xarray_ds_valid.x.max(),
                            xarray_ds_valid.y.max(),
                        )
                        slice_ymin -= 10
                        slice_xmax += 10

                    doy = torch.Tensor(
                        (pd.to_datetime(dates) - pd.to_datetime(reference_date)).days
                    )
                    # data = {}
                    images = {}
                    # data[sat] = {}

                    if sat == "s1_asc":
                        images = build_s1_images_and_masks_biomass(
                            images, xarray_ds_valid
                        )
                        image_satellite = images["s1_asc"].nan_to_num()

                        # data[sat]["img"] = images[sat]
                        # data[sat]["mask"] = images[f"{sat}_valmask"].unsqueeze(-3)
                        # data[sat]["angles"] = images[f"{sat}_angles"].unsqueeze(-3)
                        # if "CRS" not in images:
                        #     images["CRS"] = src

                    else:
                        dates = pd.DatetimeIndex(dates).to_frame(name="date_s2")

                        images = build_s2_images_and_masks(
                            images, xarray_ds_valid, dates
                        )
                        image_satellite = images["s2_set"].nan_to_num()

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

                    local_patch = 256 + 64
                    margin_local = 32

                    not_used_pixels_y = (h - 2 * margin_local) % (
                        local_patch - 2 * margin_local
                    )
                    not_used_pixels_x = (w - 2 * margin_local) % (
                        local_patch - 2 * margin_local
                    )

                    start_margin_y = int(not_used_pixels_y / 2)
                    start_margin_x = int(not_used_pixels_x / 2)

                    nb_y = int(
                        np.floor(
                            (h - 2 * margin_local - not_used_pixels_y)
                            / (local_patch - 2 * margin_local)
                        )
                    )
                    nb_x = int(
                        np.floor(
                            (w - 2 * margin_local - not_used_pixels_x)
                            / (local_patch - 2 * margin_local)
                        )
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
                                input = ClassifBInput(
                                    sits=sits.to(DEVICE),
                                    input_doy=doy[None, :].to(DEVICE),
                                    padd_index=torch.zeros(1, sits.shape[1])
                                    .bool()
                                    .to(DEVICE),
                                    mask=None,
                                    labels=None,
                                )
                                del sits
                                with torch.no_grad():
                                    ort_out = repr_encoder.forward_keep_input_dim(
                                        input
                                    ).repr.cpu()
                                del input
                            if margin_local > 0:
                                encoded_patch[f"{sat}_lat_mu"][
                                    :,
                                    :,
                                    y + margin_local : y + local_patch - margin_local,
                                    x + margin_local : x + local_patch - margin_local,
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
                        start_margin_y + margin_local : y + local_patch - margin_local,
                        start_margin_x + margin_local : x + local_patch - margin_local,
                    ]

                    ind = []
                    biomass_values_sliced = None
                    if biomass_values is not None:
                        ind, biomass_values_sliced = get_index(
                            biomass_values, xy_matrix
                        )

                    encoded_patch["matrix"] = xy_matrix

                    encoded_patch[f"{sat}_lat_mu"] = encoded_patch[f"{sat}_lat_mu"][
                        :,
                        :,
                        start_margin_y + margin_local : y + local_patch - margin_local,
                        start_margin_x + margin_local : x + local_patch - margin_local,
                    ]

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
                            output_folder, sat, tile, f"{sat}_{tile}_{enum}.pt"
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
                                    tile,
                                    f"gt_tcd_t32tnt_encoded_{sat}_{tile}_{enum}.csv",
                                )
                            )
                            if not Path(
                                os.path.join(
                                    output_folder,
                                    sat,
                                    tile,
                                    f"gt_tcd_t32tnt_days_{sat}_{tile}.npy",
                                )
                            ).exists():
                                np.save(
                                    os.path.join(
                                        output_folder,
                                        sat,
                                        tile,
                                        f"gt_tcd_t32tnt_days_{sat}_{tile}.npy",
                                    ),
                                    days,
                                )

                    del encoded_patch
