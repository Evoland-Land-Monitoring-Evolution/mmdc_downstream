import json
import logging
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import xarray as xr
from affine import Affine
from einops import rearrange
from mmdc_singledate.datamodules.components.datamodule_components import (
    apply_log_to_s1,
    build_s1_images_and_masks,
    build_s2_images_and_masks,
    dem_height_aspect,
)
from xarray import DataArray, Dataset

from mmdc_downstream_pastis.datamodule.datatypes import (
    METEO_BANDS,
    MMDCDataStruct,
    PastisBatch,
)
from mmdc_downstream_pastis.encode_series.encode import (
    encode_one_batch_s1,
    encode_one_batch_s2,
)
from mmdc_downstream_pastis.mmdc_model.model import PretrainedMMDCPastis

log = logging.getLogger(__name__)

METEO_DAYS_BEFORE = 4
METEO_DAYS_AFTER = 1
BANDS_S2 = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
meteo_modalities = METEO_BANDS.keys()


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


def get_tcd_gt(gt_path: str) -> pd.DataFrame:
    """
    Get coordinates of GT points and recompute them to the center of pixel
    """
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
    print(tcd_values.head(20))
    return tcd_values


def rearrange_ts(ts: torch.Tensor) -> torch.Tensor:
    """Rearrange TS"""
    return rearrange(ts, "t c h w ->  (t c) h w")


def get_one_slice(
    nc_file: str | Path, slice: tuple[float, float, float, float]
) -> tuple[Any, Any]:
    """Get one slice for one month"""
    with xr.open_dataset(
        nc_file,
        decode_coords="all",
        mask_and_scale=True,
        decode_times=True,
        engine="netcdf4",
    ) as file:
        return file.rio.slice_xy(*slice), file.crs


def get_sliced_modality(
    months: list[str],
    folder: str | Path,
    slice: tuple[float, float, float, float],
    modality: str,
    meteo: bool = False,
    dates: np.ndarray = None,
    coord: list = None,
) -> tuple[DataArray | Dataset | Any, Any]:
    """Get a dataset slice for one modality for all available dates"""
    modality_slice_list = []
    for month in months:
        if meteo:
            nc_file = os.path.join(folder, month, "AGERA5", modality, "openEO.nc")
        else:
            nc_file = os.path.join(folder, month, modality, "openEO.nc")
        file, src = get_one_slice(nc_file, slice)
        print(len(file.x), len(file.y))
        if coord is not None:
            file = file.sel(x=coord[0], y=coord[1])
        if dates is not None:
            dates_months = pd.DatetimeIndex(dates).to_frame(name="dates")
            months_dates = (dates_months.dates.dt.month == int(month)).values
            file = file.sel(t=dates[months_dates])

        modality_slice_list.append(file)
        file.close()
    return xr.concat(modality_slice_list, dim="t").sortby("t"), src


def get_meteo_dates_modality(dates_df: pd.DataFrame, satellite: str) -> pd.DataFrame:
    meteo_before = pd.Timedelta(METEO_DAYS_BEFORE, "D")  # day_S2 - n_days
    meteo_after = pd.Timedelta(METEO_DAYS_AFTER, "D")  # day_S2 + n_days

    dates_df["meteo_start_date"] = dates_df["date_" + satellite] - meteo_before
    dates_df["meteo_end_date"] = dates_df["date_" + satellite] + meteo_after
    return dates_df


def check_s2_clouds(folder_data: str | Path, months_folders: list[str]) -> np.ndarray:
    """
    Check S2 images for invalid dates and choose dates for algorithm
    """
    all_dates = []
    for month in months_folders:
        nc_file = os.path.join(folder_data, month, "Sentinel2", "openEO.nc")
        with xr.open_dataset(
            nc_file,
            mask_and_scale=True,
            decode_times=True,
        ) as xarray_ds:
            xarray_ds = xarray_ds[["SCL", "CLM"]]

            cloud_coverage = []
            for t in xarray_ds.t:
                print("mask", t.values)
                cloud_coverage.append(
                    np.nanmean(
                        np.logical_or(
                            xarray_ds.sel(t=t.values)
                            .SCL.astype(np.int8)
                            .isin([0, 1, 3, 8, 9, 10, 11]),
                            xarray_ds.sel(t=t.values)
                            .CLM.astype(np.int8)
                            .isin([1, 255]),
                        ).values
                    )
                )
            cloud_coverage = np.asarray(cloud_coverage)
            cloudy = cloud_coverage > 0.6  # Where nodata > 60%
            clouds_order = cloud_coverage.argsort().argsort()
            ind_of_four_less_cloudy = np.arange(
                clouds_order[cloudy].min(), clouds_order[cloudy].min() + 4
            )
            _, ind, _ = np.intersect1d(
                clouds_order,
                ind_of_four_less_cloudy,
                assume_unique=True,
                return_indices=True,
            )

            dates = xarray_ds.t[~cloudy].sortby("t")
            four_cloudy_dates = xarray_ds.t[ind].sortby(
                "t"
            )  # We choose one cloudy date per month

            all_dates.extend(dates)
            all_dates.extend(four_cloudy_dates)
            del xarray_ds

    return xr.concat(all_dates, dim="t").sortby("t").values


def check_s1_nans(
    folder_data: str | Path, months_folders: list[str], orbit: str = "ASCENDING"
) -> np.ndarray:
    """
    Check S1 images for nans and choose dates for algorithm
    """
    all_dates = []
    for month in months_folders:
        nc_file = os.path.join(folder_data, month, f"Sentinel1_{orbit}", "openEO.nc")
        with xr.open_dataset(
            nc_file,
            mask_and_scale=True,
            decode_times=True,
        ) as xarray_ds:
            array = torch.Tensor(xarray_ds.VV.values)
            nans = torch.isnan(array).to(torch.float)
            nan_ratio = torch.mean(nans, dim=(1, 2))
            condition = nan_ratio < 0.05 if orbit == "ASCENDING" else nan_ratio > 0.06
            dates = xarray_ds.t[condition].sortby("t")
            all_dates.append(dates)
    return xr.concat(all_dates, dim="t").sortby("t").values


# def check_s1_nans_and_write(
#         folder_data: str | Path, months_folders: list[str], orbit="ASCENDING"
# ) -> np.ndarray:
#     all_dates = []
#     all_nans = []
#     all_dates_available = []
#     for month in months_folders:
#         nc_file = os.path.join(folder_data, month, f"Sentinel1_{orbit}", "openEO.nc")
#         with xr.open_dataset(
#                 nc_file,
#                 decode_coords="all",
#                 mask_and_scale=True,
#                 decode_times=True,
#                 engine="netcdf4",
#         ) as xarray_ds:
#             array = torch.Tensor(xarray_ds.VV.values)
#             nans = torch.isnan(array).to(torch.float)
#             all_nans.append(nans.to(torch.int))
#
#             with rasterio.open(
#                     os.path.join(folder_data, f"nans_{orbit}_{month}.TIFF"),
#                     'w',
#                     driver='GTiff',
#                     height=nans.shape[-2],
#                     width=nans.shape[-1],
#                     count=nans.shape[0],
#                     dtype=rasterio.int8,
#                     # crs=xarray_ds.crs,
#                     # transform=transform,
#             ) as dst:
#                 log.info(nans.to(torch.bool).shape)
#                 dst.write(nans.to(torch.bool).cpu().numpy())
#
#             nan_ratio = torch.mean(nans, dim=(1, 2))
#             dates = xarray_ds.t[nan_ratio < 0.05].sortby("t")
#             all_dates.append(dates)
#             log.info(orbit)
#             log.info(month)
#             log.info(nan_ratio)
#             log.info(xarray_ds.t.values)
#             all_dates_available.append(xarray_ds.t.values)
#     all_nans = torch.concat(all_nans)
#     with rasterio.open(
#             os.path.join(folder_data, f"nans_{orbit}"),
#             'w',
#             driver='GTiff',
#             height=all_nans.shape[-2],
#             width=all_nans.shape[-1],
#             count=all_nans.shape[0],
#             dtype=rasterio.int8,
#             # crs=xarray_ds.crs,
#             # transform=transform,
#     ) as dst:
#         log.info(all_nans.to(torch.bool).shape)
#         dst.write(all_nans.to(torch.bool).cpu().numpy())
#     print(orbit, np.concatenate(all_dates_available))
#     return xr.concat(all_dates, dim="t").sortby("t").values


def get_bounds(
    folder_data: str | Path, months_folders: list[str | int]
) -> tuple[float, float, float, float]:
    """
    Get tile bounds using a reference image,
    considering that geojson vector data does not match with rasters' boundaries
    """
    # Reference image
    nc_file = os.path.join(folder_data, months_folders[0], "Sentinel2", "openEO.nc")

    with xr.open_dataset(
        nc_file,
        decode_coords="all",
        mask_and_scale=False,
        decode_times=True,
        engine="netcdf4",
    ) as image_ref:
        image_ref = image_ref.sel(t=image_ref.t[0])
        xmax, ymax, xmin, ymin = (
            image_ref.x.values.max(),
            image_ref.y.values.max(),
            image_ref.x.values.min(),
            image_ref.y.values.min(),
        )

    del image_ref

    return xmax, ymax, xmin, ymin


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


def get_slices(
    folder_data: str | Path,
    months_folders: list[str | int],
    window: int,
    margin: int = 0,
    drop_incomplete: bool = False,
) -> np.ndarray:
    """
    Get x,y boundaries for each sliding window
    """
    # Get whole image/tile bounds
    xmax, ymax, xmin, ymin = get_bounds(folder_data, months_folders)
    print("xmax=", xmax, "ymax=", ymax, "xmin=", xmin, "ymin=", ymin)

    # We multiply by 10 because it is pixel size
    slices_list = []
    for slice_xmin in np.arange(xmin, xmax, window * 10 - margin * 10 * 2):
        for slice_ymax in np.arange(ymax, ymin, -window * 10 + margin * 10 * 2):
            slice_xmax, slice_ymin = slice_xmin + window * 10, slice_ymax - window * 10
            # If the patch has uneven size
            if drop_incomplete and (slice_xmax > xmax or slice_ymin < ymin):
                continue

            if slice_xmax > xmax:
                slice_xmax = xmax + 10
            if slice_ymin < ymin:
                slice_ymin = ymin - 10
            slices_list.append(
                [slice_xmin, slice_ymin + 10, slice_xmax - 10, slice_ymax]
            )
    return np.asarray(slices_list)


def get_s2(
    s2_dates: np.ndarray,
    t_slice: list[str],
    data: dict[str, dict],
    images: dict[str, Any],
    sliced_modality: xr.Dataset,
) -> tuple[dict[str, torch.Tensor], dict[str, dict], Any]:
    """
    Get S2 images
    """
    sliced_modality = sliced_modality.sel(t=slice(t_slice[0], t_slice[1]))

    dates = pd.DatetimeIndex(s2_dates).to_frame(name="date_s2")
    dates = dates[
        (dates["date_s2"] >= pd.to_datetime(t_slice[0]))
        & (dates["date_s2"] <= pd.to_datetime(t_slice[1]))
    ].dropna()

    images = build_s2_images_and_masks(
        images, sliced_modality.astype(np.float32), dates
    )
    data["s2"]["img"] = images["s2_set"].unsqueeze(0)
    data["s2"]["mask"] = images["s2_masks"].unsqueeze(0)
    data["s2"]["angles"] = images["s2_angles"].unsqueeze(0)
    images["CRS"] = sliced_modality.crs
    return images, data, dates


def get_s1(
    t_slice: list[str],
    data: dict[str, dict],
    images: dict[str, Any],
    sliced_modality: xr.Dataset,
    sat: Literal["s1_asc", "s1_desc"],
) -> tuple[dict[str, torch.Tensor], dict[str, dict], Any]:
    """
    Get S1 images
    """
    dates = pd.DatetimeIndex(sliced_modality.t.values)

    dates = dates.to_frame(name=f"date_{sat}")
    dates = dates[
        (dates[f"date_{sat}"] >= pd.to_datetime(t_slice[0]))
        & (dates[f"date_{sat}"] <= pd.to_datetime(t_slice[1]))
    ].dropna()
    images = build_s1_images_and_masks(images, sliced_modality, dates, sat)

    data[sat]["img"] = apply_log_to_s1(images[sat]).unsqueeze(0)
    data[sat]["mask"] = images[f"{sat}_valmask"].unsqueeze(0).unsqueeze(-3)
    data[sat]["angles"] = images[f"{sat}_angles"].unsqueeze(0).unsqueeze(-3)
    if "CRS" not in images:
        images["CRS"] = sliced_modality.crs

    return images, data, dates


def prepare_patch(
    folder_data: str | Path,
    months_folders: list[str | int],
    sliced: tuple[float, float, float, float],
    s2_dates: np.array,
    s1_dates_asc: np.array,
    s1_dates_desc: np.array,
    dates_meteo: dict[str, pd.DataFrame],
    satellites: list[str],
) -> tuple[dict[str, [str, torch.Tensor]], np.array]:
    log.info(f"Slice {sliced}")
    images = {}
    data = {sat: {} for sat in satellites}
    for sat in satellites:
        t_slice = [
            f"2018-{months_folders[0]}-05",
            f"2018-{months_folders[-1]}-30",
        ]

        if sat == "s2":
            modality = "Sentinel2"
            log.info(f"Slicing {modality}")
            sliced_modality, crs = get_sliced_modality(
                months_folders,
                folder_data,
                sliced,
                modality,
                meteo=False,
                dates=s2_dates,
            )
            log.info(f"Sliced {modality}")
            sliced_modality = sliced_modality.sel(t=slice(t_slice[0], t_slice[1]))

            # create_netcdf_s2_one_date(
            #     xy_matrix, sliced_modality, margin, enum, date=2)
            images, data, dates = get_s2(
                s2_dates, t_slice, data, images, sliced_modality
            )
            images["CRS"] = crs
        else:
            modality = (
                "Sentinel1_ASCENDING" if sat == "s1_asc" else "Sentinel1_DESCENDING"
            )
            log.info(f"Slicing {modality}")
            sliced_modality, _ = get_sliced_modality(
                months_folders,
                folder_data,
                sliced,
                modality,
                meteo=False,
                dates=s1_dates_asc if sat == "s1_asc" else s1_dates_desc,
            )
            log.info(f"Sliced {modality}")
            sliced_modality = sliced_modality.sel(t=slice(t_slice[0], t_slice[1]))

            images, data, dates = get_s1(t_slice, data, images, sliced_modality, sat)
            del sliced_modality

        if sat not in dates_meteo:
            dates_meteo[sat] = get_meteo_dates_modality(dates, sat)

    for modality in meteo_modalities:
        log.info(modality)
        sliced_meteo, _ = get_sliced_modality(
            months_folders,
            folder_data,
            sliced,
            modality.upper(),
            meteo=True,
        )
        for sat, dates_sat in dates_meteo.items():
            # print(modality)
            data[sat][f"meteo_{modality}"] = build_meteo(
                images, sliced_meteo.astype(np.float64), dates_sat, modality
            )[modality].unsqueeze(0)
    del images, sliced_meteo
    # DEM
    nc_file = os.path.join(folder_data, "DEM", "openEO.nc")
    dem_slice, _ = get_one_slice(nc_file, sliced)
    dem_dataset = torch.tensor(dem_slice.DEM.values)
    # We deal with the special case,
    # when dem band contains 2 dates and one of them is nan
    # So we choose the date with less nans than the other
    if len(dem_dataset) > 0:
        dem_dataset = dem_dataset[torch.argmin(torch.isnan(dem_dataset).sum((1, 2)))]
    dem = dem_height_aspect(dem_dataset.squeeze(0))
    for sat in data:
        data[sat]["dem"] = dem.unsqueeze(0)
    del dem
    return data, dates_meteo


def get_encoded_patch(
    data: dict[str, [str, torch.Tensor]],
    mmdc_model: PretrainedMMDCPastis,
    dates_meteo: np.array,
    satellites: list[str],
    s1_join: bool = True,
    margin: int = 40,
) -> dict[str, Any]:
    encoded_patch = {}

    batch_dict = {}
    for sat in data:
        sits = (
            MMDCDataStruct.init_empty().fill_empty_from_dict(data[sat]).concat_meteo()
        )
        setattr(
            sits,
            "meteo",
            torch.flatten(getattr(sits, "meteo"), start_dim=2, end_dim=3),
        )
        tcd_batch = PastisBatch(
            sits=sits,
            true_doy=dates_meteo[sat][f"date_{sat}"].values[None, :],
        )
        batch_dict[sat.upper()] = tcd_batch
        encoded_patch[f"{sat}_doy"] = dates_meteo[sat][f"date_{sat}"].values
        encoded_patch[f"{sat}_mask"] = sits.data.mask[
            0, :, :, margin:-margin, margin:-margin
        ]
    log.info("encoding")

    del data, tcd_batch

    if "s1_asc" in satellites or "s1_desc" in satellites:
        (
            latents1,
            latents1_asc,
            latents1_desc,
            mask_s1,
            days_s1,
        ) = encode_one_batch_s1(mmdc_model, batch_dict)
        if s1_join:
            encoded_patch["s1_lat_mu"] = latents1.mean[
                0, :, :, margin:-margin, margin:-margin
            ]
            encoded_patch["s1_lat_logvar"] = latents1.logvar[
                0, :, :, margin:-margin, margin:-margin
            ]
            encoded_patch["s1_mask"] = mask_s1[0, :, :, margin:-margin, margin:-margin]
            encoded_patch["s1_doy"] = days_s1
        else:
            encoded_patch["s1_asc_lat_mu"] = latents1_asc.mean[
                0, :, :, margin:-margin, margin:-margin
            ]
            encoded_patch["s1_asc_lat_logvar"] = latents1_asc.logvar[
                0, :, :, margin:-margin, margin:-margin
            ]
            encoded_patch["s1_desc_lat_mu"] = latents1_desc.mean[
                0, :, :, margin:-margin, margin:-margin
            ]
            encoded_patch["s1_desc_lat_logvar"] = latents1_desc.logvar[
                0, :, :, margin:-margin, margin:-margin
            ]

        del (
            latents1,
            latents1_asc,
            latents1_desc,
        )

    if "s2" in satellites:
        latents2 = encode_one_batch_s2(mmdc_model, batch_dict)

        encoded_patch["s2_lat_mu"] = latents2.mean[
            0, :, :, margin:-margin, margin:-margin
        ]
        encoded_patch["s2_lat_logvar"] = latents2.logvar[
            0, :, :, margin:-margin, margin:-margin
        ]
        del latents2
    del batch_dict
    return encoded_patch


def extract_gt_points(
    encoded_patch,
    sat,
    ind,
    tcd_values_sliced,
    sliced_gt,
):
    mask = encoded_patch[f"{sat}_mask"].expand_as(encoded_patch[f"{sat}_lat_mu"])

    encoded_patch[f"{sat}_lat_mu"][mask.bool()] = torch.nan
    encoded_patch[f"{sat}_lat_logvar"][mask.bool()] = torch.nan

    gt_ref_values_mu = rearrange_ts(encoded_patch[f"{sat}_lat_mu"])[
        :, ind[:, 0], ind[:, 1]
    ].T

    gt_ref_values_logvar = rearrange_ts(encoded_patch[f"{sat}_lat_logvar"])[
        :, ind[:, 0], ind[:, 1]
    ].T
    days = encoded_patch[f"{sat}_doy"]

    values_df_mu = pd.DataFrame(
        data=gt_ref_values_mu.cpu().numpy(),
        columns=[
            f"mu_f{b}_d{m}"
            for m in range(len(days))
            for b in range(encoded_patch[f"{sat}_lat_logvar"].shape[-3])
        ],
    )
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

    if sat not in sliced_gt:
        sliced_gt[sat] = [df_gt]
    else:
        sliced_gt[sat].append(df_gt)

    return df_gt, days

    # torch.save(encoded_patch,
    #            os.path.join(output_folder, f"Encoded_patch_{enum}.pt"))


def set_system_and_save(
    ds: xr.Dataset, xx: torch.Tensor, yy: torch.Tensor, name: str
) -> None:
    ds.rio.write_transform(
        Affine(10.0, 0.0, xx.min() - 5, 0.0, -10.0, yy.max() + 5),
        grid_mapping_name="crs",
        inplace=True,
    )

    ds.rio.write_crs(32632, inplace=True)
    ds.to_netcdf(name, engine="h5netcdf")


def create_netcdf_s2_all_dates(
    xy_matrix: torch.Tensor, sliced_modality: xr.Dataset, margin: int, enum: int
) -> None:
    xx = xy_matrix[0]
    yy = xy_matrix[1]
    ds = xr.Dataset(
        data_vars=dict(
            B02=(
                ["t", "x", "y"],
                sliced_modality.B02.values[:, margin:-margin, margin:-margin],
            ),
            B03=(
                ["t", "x", "y"],
                sliced_modality.B03.values[:, margin:-margin, margin:-margin],
            ),
            B04=(
                ["t", "x", "y"],
                sliced_modality.B04.values[:, margin:-margin, margin:-margin],
            ),
            B08=(
                ["t", "x", "y"],
                sliced_modality.B08.values[:, margin:-margin, margin:-margin],
            ),
        ),
        coords={
            "x": ("x", xx[:, 0]),
            "y": ("y", yy[0]),
            "t": ("t", sliced_modality.t.values),
        },
        attrs=dict(description="Original image"),
    )

    set_system_and_save(ds, xx, yy, name=f"s2_patch_{enum}.nc")


def create_netcdf_s1_all_dates(
    xy_matrix: torch.Tensor,
    sliced_modality: xr.Dataset,
    margin: int,
    enum: int,
    orbit: str = "asc",
) -> None:
    xx = xy_matrix[0]
    yy = xy_matrix[1]
    ds = xr.Dataset(
        data_vars=dict(
            VV=(
                ["t", "x", "y"],
                sliced_modality.VV.values[:, margin:-margin, margin:-margin],
            ),
            VH=(
                ["t", "x", "y"],
                sliced_modality.VH.values[:, margin:-margin, margin:-margin],
            ),
        ),
        coords={
            "x": ("x", xx[:, 0]),
            "y": ("y", yy[0]),
            "t": ("t", sliced_modality.t.values),
        },
        attrs=dict(description="Original image"),
    )

    set_system_and_save(ds, xx, yy, name=f"s1_{orbit}_patch_{enum}.nc")


def create_netcdf_s2_one_date(
    xy_matrix: torch.Tensor,
    sliced_modality: xr.Dataset,
    margin: int,
    enum: int,
    date: int,
) -> None:
    xx = xy_matrix[0]
    yy = xy_matrix[1]
    ds = xr.Dataset(
        data_vars=dict(
            B02=(
                ["x", "y"],
                sliced_modality.isel(t=date).B02.values[margin:-margin, margin:-margin],
            ),
            B03=(
                ["x", "y"],
                sliced_modality.isel(t=date).B03.values[margin:-margin, margin:-margin],
            ),
            B04=(
                ["x", "y"],
                sliced_modality.isel(t=date).B04.values[margin:-margin, margin:-margin],
            ),
            B08=(
                ["x", "y"],
                sliced_modality.isel(t=date).B08.values[margin:-margin, margin:-margin],
            ),
        ),
        coords={"x": ("x", xx[:, 0]), "y": ("y", yy[0])},
        attrs=dict(description="Original image"),
    )
    set_system_and_save(ds, xx, yy, name=f"s2_patch_{enum}_t{date}.nc")


def create_netcdf_s1_one_date(
    xy_matrix: torch.Tensor,
    sliced_modality: xr.Dataset,
    margin: int,
    enum: int,
    date: int,
):
    xx = xy_matrix[0]
    yy = xy_matrix[1]
    ds = xr.Dataset(
        data_vars=dict(
            B02=(
                ["x", "y"],
                sliced_modality.isel(t=date).B02.values[margin:-margin, margin:-margin],
            ),
            B03=(
                ["x", "y"],
                sliced_modality.isel(t=date).B03.values[margin:-margin, margin:-margin],
            ),
            B04=(
                ["x", "y"],
                sliced_modality.isel(t=date).B04.values[margin:-margin, margin:-margin],
            ),
            B08=(
                ["x", "y"],
                sliced_modality.isel(t=date).B08.values[margin:-margin, margin:-margin],
            ),
        ),
        coords={"x": ("x", xx[:, 0]), "y": ("y", yy[0])},
        attrs=dict(description="Original image"),
    )
    set_system_and_save(ds, xx, yy, name=f"s2_patch_{enum}_t{date}.nc")


def create_netcdf_encoded(
    xy_matrix: torch.Tensor,
    s2_mu: np.ndarray,
    enum: int,
    dates: np.ndarray,
    s1: bool = False,
):
    xx = xy_matrix[0]
    yy = xy_matrix[1]
    ds = xr.Dataset(
        data_vars=dict(
            f0=(["t", "x", "y"], s2_mu[:, 0]),
            f1=(["t", "x", "y"], s2_mu[:, 1]),
            f2=(["t", "x", "y"], s2_mu[:, 2]),
            f3=(["t", "x", "y"], s2_mu[:, 3]),
            f4=(["t", "x", "y"], s2_mu[:, 4]),
            f5=(["t", "x", "y"], s2_mu[:, 5]),
        ),
        coords={"x": ("x", xx[:, 0]), "y": ("y", yy[0]), "t": ("t", dates)},
        attrs=dict(description="Encoded image"),
    )
    if not s1:
        set_system_and_save(ds, xx, yy, name=f"s2_patch_encoded_{enum}.nc")
    else:
        set_system_and_save(ds, xx, yy, name=f"s1_patch_encoded_{enum}.nc")


def create_netcdf_one_date_encoded(
    xy_matrix: torch.Tensor, s2_mu: np.ndarray, enum: int, date: int, s1: bool = False
):
    xx = xy_matrix[0]
    yy = xy_matrix[1]
    ds = xr.Dataset(
        data_vars=dict(
            f0=(["x", "y"], s2_mu[date, 0]),
            f1=(["x", "y"], s2_mu[date, 1]),
            f2=(["x", "y"], s2_mu[date, 2]),
            f3=(["x", "y"], s2_mu[date, 3]),
            f4=(["x", "y"], s2_mu[date, 4]),
            f5=(["x", "y"], s2_mu[date, 5]),
        ),
        coords={"x": ("x", xx[:, 0]), "y": ("y", yy[0])},
        attrs=dict(description="Encoded image one date"),
    )
    if not s1:
        set_system_and_save(ds, xx, yy, name=f"s2_patch_encoded_{enum}_t{date}.nc")
    else:
        set_system_and_save(ds, xx, yy, name=f"s1_patch_encoded_{enum}_t{date}.nc")


def build_meteo(
    images: dict[str, Any],
    xarray_ds: xr.Dataset,
    dates_df: pd.DataFrame,
    meteo_type: str,
    light: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Generate meteo data provided by OpenEO,
    and interpolate in case there are NaNs
    """
    if not light:
        dataset_meteo = []
        dates = dates_df[["meteo_start_date", "meteo_end_date"]]
        for _, row in dates.iterrows():
            # We convert netcdf directly to tensors
            # (contrary to previous xr.concat) to avoid memory problems
            meteo_array = xarray_ds.sel(
                t=slice(row.meteo_start_date, row.meteo_end_date)
            )
            meteo_array[METEO_BANDS[meteo_type]].rio.write_nodata(np.nan, inplace=True)
            # Interpolate NaNs
            null_values = meteo_array[METEO_BANDS[meteo_type]].isnull()
            null_values_per_date = null_values.sum(dim=["x", "y"])
            ratio_null_values_per_date = null_values.mean(dim=["x", "y"])
            if null_values_per_date.sum() > 0:
                meteo_array = meteo_array.rio.write_crs(images["CRS"].crs_wkt)
                # pylint: disable=C0103
                t = null_values_per_date.where(
                    (null_values_per_date > 0) & (ratio_null_values_per_date < 0.1)
                ).t
                if len(t) > 0:
                    interpolated_dates = meteo_array.sel(t=t).rio.interpolate_na()
                    # pylint: disable=use-dict-literal
                    meteo_array.loc[dict(t=t)] = interpolated_dates

            dataset_meteo.append(
                torch.Tensor(meteo_array[METEO_BANDS[meteo_type]].values).unsqueeze(0)
            )
        images[meteo_type] = torch.cat(dataset_meteo, 0)
    else:  # We choose only one image and not the series
        dates = dates_df["date_s2"]
        images[meteo_type] = torch.Tensor(
            xarray_ds.sel(t=pd.DatetimeIndex(dates))
            .sortby("t")[METEO_BANDS[meteo_type]]
            .values
        ).unsqueeze(1)
    return images
