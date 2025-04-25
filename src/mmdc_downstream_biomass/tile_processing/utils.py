from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import xarray as xr
from einops import rearrange


def rearrange_ts(ts: torch.Tensor) -> torch.Tensor:
    """Rearrange TS"""
    return rearrange(ts, "t c h w ->  (t c) h w")


def extract_gt_points(
    encoded_patch,
    sat,
    ind,
    tcd_values_sliced,
    sliced_gt,
):
    if f"{sat}_mask" in encoded_patch:
        mask = encoded_patch[f"{sat}_mask"].expand_as(encoded_patch[f"{sat}_lat_mu"])

        encoded_patch[f"{sat}_lat_mu"][mask.bool()] = torch.nan
        encoded_patch[f"{sat}_lat_logvar"][mask.bool()] = torch.nan

    gt_ref_values_mu = torch.Tensor(
        rearrange_ts(encoded_patch[f"{sat}_lat_mu"])[:, ind[:, 0], ind[:, 1]].T
    )

    days = encoded_patch[f"{sat}_doy"]

    values_df_mu = pd.DataFrame(
        data=gt_ref_values_mu.cpu().numpy(),
        columns=[
            f"mu_f{b}_d{m}"
            for m in range(len(days))
            for b in range(encoded_patch[f"{sat}_lat_mu"].shape[-3])
        ],
    )
    if f"{sat}_lat_logvar" in encoded_patch:
        gt_ref_values_logvar = rearrange_ts(encoded_patch[f"{sat}_lat_logvar"])[
            :, ind[:, 0], ind[:, 1]
        ].T
        values_df_logvar = pd.DataFrame(
            data=gt_ref_values_logvar.cpu().numpy(),
            columns=[
                f"logvar_f{b}_d{m}"
                for m in range(len(days))
                for b in range(encoded_patch[f"{sat}_lat_logvar"].shape[-3])
            ],
        )

        df_gt = pd.concat(
            [
                tcd_values_sliced.reset_index(),
                values_df_mu,
                values_df_logvar,
            ],
            axis=1,
        )
    else:
        df_gt = pd.concat(
            [
                tcd_values_sliced.reset_index(),
                values_df_mu,
            ],
            axis=1,
        )

    if sat not in sliced_gt:
        sliced_gt[sat] = [df_gt]
    else:
        sliced_gt[sat].append(df_gt)

    return df_gt, days


def xarray_to_tensor(arr: xr.Dataset, bands: list[str]) -> torch.Tensor:
    """Xarray dataset to tensor with band selection"""
    return torch.Tensor(arr[bands].to_array().values).permute(1, 0, 2, 3)


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
        torch.deg2rad(
            xarray_to_tensor(xarray_ds, bands=["local_incidence_angle"]).squeeze(1)
        )
    )

    images[f"{s1_type}_angles"] = available_angles

    # generate nan mask
    nan_mask = torch.isnan(images[s1_type]).sum(dim=1).to(torch.int8) > 0
    images[f"{s1_type}_valmask"] = nan_mask

    return images


def get_gt(
    gt: str | Path | gpd.GeoDataFrame, variables: list, crs: str | None = None
) -> pd.DataFrame:
    """
    Get coordinates of GT points and recompute them to the center of pixel
    """

    if type(gt) in (str, Path):
        data = gpd.read_file(gt)
    else:
        data = gt

    if crs is not None:
        data = data.to_crs(crs)
    crs = data.crs

    df = gpd.GeoDataFrame(
        columns=[*variables, "x", "y", "x_round", "y_round", "geometry"],
        geometry="geometry",
        crs=crs,
    )

    df[variables] = data[variables]

    x = data.geometry.x.values
    y = data.geometry.y.values

    df.geometry = data.geometry

    df.x = x
    df.y = y

    x_r = np.round(x / 5) * 5
    y_r = np.round(y / 5) * 5

    x_r = [xxr - 5 if xxr % 10 == 0 else xxr for xxr in x_r]
    y_r = [yyr - 5 if yyr % 10 == 0 else yyr for yyr in y_r]

    x_r = [xxr + 10 if xxx - xxr > 5 else xxr for xxr, xxx in zip(x_r, x)]
    y_r = [yyr + 10 if yyy - yyr > 5 else yyr for yyr, yyy in zip(y_r, y)]

    df["x_round"] = x_r
    df["y_round"] = y_r

    print(df.head(20))
    return df


def get_index(
    tcd_values: pd.DataFrame, xy_matrix: np.ndarray
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Get index of GT points, their coords and TCD values for the tile
    """
    tcd_values_sliced = tcd_values[
        (tcd_values["x_round"] >= xy_matrix[0].min())
        & (tcd_values["x_round"] <= xy_matrix[0].max())
        & (tcd_values["y_round"] >= xy_matrix[1].min())
        & (tcd_values["y_round"] <= xy_matrix[1].max())
    ]

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
    return np.array(ind), tcd_values_sliced


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


def process_subpatches(local_patch, margin_local, data_ref):
    h, w = data_ref.shape[-2:]
    not_used_pixels_y = (h - 2 * margin_local) % (local_patch - 2 * margin_local)
    not_used_pixels_x = (w - 2 * margin_local) % (local_patch - 2 * margin_local)

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

    return start_margin_x, start_margin_y, nb_x, nb_y
