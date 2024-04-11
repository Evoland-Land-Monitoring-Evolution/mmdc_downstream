import json
import logging
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import xarray as xr
from einops import rearrange
from mmdc_singledate.datamodules.components.datamodule_components import (
    apply_log_to_s1,
    build_meteo,
    build_s1_images_and_masks,
    build_s2_images_and_masks,
    dem_height_aspect,
)

from mmdc_downstream_pastis.datamodule.datatypes import PastisBatch
from mmdc_downstream_pastis.datamodule.utils import MMDCDataStruct
from mmdc_downstream_pastis.encode_series.encode import encode_one_batch
from mmdc_downstream_pastis.mmdc_model.model import PretrainedMMDCPastis

# from mmdc_downstream_tcd.tile_processing.utils import (
#     create_netcdf_encoded,
#     create_netcdf_one_date_encoded,
#     create_netcdf_s2_one_date,
# )


METEO_DAYS_BEFORE = 4
METEO_DAYS_AFTER = 1
BANDS_S2 = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

log = logging.getLogger(__name__)

satellites = ["s2", "s1_asc", "s1_desc"]


def get_tcd_gt(gt_path: str) -> pd.DataFrame:
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
    return rearrange(ts, "t c h w ->  (t c) h w")


def get_one_slice(nc_file: str | Path, slice: list[float]) -> xr.Dataset:
    """Get one slice for one month"""
    with xr.open_dataset(
        nc_file,
        decode_coords="all",
        mask_and_scale=False,
        decode_times=True,
        engine="h5netcdf",
    ) as file:
        return file.rio.slice_xy(*slice)


def get_sliced_modality(
    months: list[str],
    folder: str | Path,
    slice: list[float],
    modality: str,
    meteo: bool = False,
    dates: np.ndarray = None,
    coord=None,
) -> xr.Dataset:
    """Get a dataset slice for one modality for all available dates"""
    modality_slice_list = []
    for month in months:
        if meteo:
            nc_file = os.path.join(folder, month, "AGERA5", modality, "openEO.nc")
        else:
            nc_file = os.path.join(folder, month, modality, "openEO.nc")
        file = get_one_slice(nc_file, slice)
        print(len(file.x), len(file.y))
        if coord is not None:
            file = file.sel(x=coord[0], y=coord[1])
        if dates is not None:
            dates_months = pd.DatetimeIndex(dates).to_frame(name="dates")
            months_dates = (dates_months.dates.dt.month == int(month)).values
            file = file.sel(t=dates[months_dates])

        modality_slice_list.append(file)
        file.close()
    return xr.concat(modality_slice_list, dim="t").sortby("t")


def get_meteo_dates_modality(dates_df, satellite):
    meteo_before = pd.Timedelta(METEO_DAYS_BEFORE, "D")  # day_S2 - n_days
    meteo_after = pd.Timedelta(METEO_DAYS_AFTER, "D")  # day_S2 + n_days

    dates_df["meteo_start_date"] = dates_df["date_" + satellite] - meteo_before
    dates_df["meteo_end_date"] = dates_df["date_" + satellite] + meteo_after
    return dates_df


METEO_BANDS = {
    "dew_temp": "dewpoint-temperature",
    "prec": "precipitation-flux",
    "sol_rad": "solar-radiation-flux",
    "temp_max": "temperature-max",
    "temp_mean": "temperature-mean",
    "temp_min": "temperature-min",
    "vap_press": "vapour-pressure",
    "wind_speed": "wind-speed",
}
meteo_modalities = METEO_BANDS.keys()


def check_s2_clouds(folder_data: str | Path, months_folders: list[str]) -> np.ndarray:
    all_dates = []
    for month in months_folders:
        nc_file = os.path.join(folder_data, month, "Sentinel2", "openEO.nc")
        with xr.open_dataset(
            nc_file,
            decode_coords="all",
            mask_and_scale=True,
            decode_times=True,
            engine="netcdf4",
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
        # del masks
    return xr.concat(all_dates, dim="t").sortby("t").values


def check_s1_nans(
    folder_data: str | Path, months_folders: list[str], orbit="ASCENDING"
) -> np.ndarray:
    all_dates = []
    for month in months_folders:
        nc_file = os.path.join(folder_data, month, f"Sentinel1_{orbit}", "openEO.nc")
        with xr.open_dataset(
            nc_file,
            decode_coords="all",
            mask_and_scale=True,
            decode_times=True,
            engine="netcdf4",
        ) as xarray_ds:
            array = torch.Tensor(xarray_ds.VV.values)
            nans = torch.isnan(array).to(torch.float)
            nan_ratio = torch.mean(nans, dim=(1, 2))
            condition = nan_ratio < 0.05 if orbit == "ASCENDING" else nan_ratio > 0.06
            dates = xarray_ds.t[condition].sortby("t")
            all_dates.append(dates)
    return xr.concat(all_dates, dim="t").sortby("t").values


# def check_s1_nans(
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


def get_bounds(folder_data: str | Path, months_folders: list[str | int]):
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

    print(len(image_ref.x), len(image_ref.y))

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
):
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
    return slices_list


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
    data["s2"]["img"] = images["s2_set"].unsqueeze(0) / 10000
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
    return images, data, dates


def encode_tile(
    folder_data: str | Path,
    months_folders: list[str | int],
    window: int,
    output_folder: str | Path,
    mmdc_model: PretrainedMMDCPastis,
    gt_file: str | Path = None,
    s1_join: bool = True,
):
    """
    Encode a tile with MMDC algorithm with a sliding window
    """

    margin = mmdc_model.model_mmdc.nb_cropped_hw

    slices_list = get_slices(
        folder_data, months_folders, window, margin=margin, drop_incomplete=True
    )

    dates_meteo: dict[str, pd.DataFrame] = {}

    # s2_dates = check_s2_clouds(folder_data, months_folders)
    s2_dates = np.load(os.path.join(folder_data, "dates_s2.npy"))
    s1_dates_asc = np.load(os.path.join(folder_data, "dates_s1_asc_good.npy"))
    s1_dates_desc = np.load(os.path.join(folder_data, "dates_s1_desc_good.npy"))

    # print("computing s1 asc")
    # s1_dates_asc = check_s1_nans(folder_data, months_folders,
    #                          orbit="ASCENDING")
    # np.save(os.path.join(folder_data, "dates_s1_asc.npy"), s1_dates_asc)
    # print("computing s1 desc")
    # s1_dates_desc = check_s1_nans(folder_data, months_folders,
    #                          orbit="DESCENDING")
    # #
    # np.save(os.path.join(folder_data, "dates_s1_desc.npy"), s1_dates_desc)
    tcd_values = None
    if gt_file is not None:
        tcd_values = get_tcd_gt(gt_file)

    # Whether we produce separate embeddings for s1_asc and s1_desc or not
    satellites_to_write = ["s2", "s1"] if s1_join else satellites

    print(tcd_values)

    sliced_gt = {}
    # for enum, sliced in enumerate(slices_list[::-1]):
    #     enum = len(slices_list) - enum - 1
    for enum, sliced in enumerate(slices_list):
        xy_matrix = generate_coord_matrix(sliced)[:, margin:-margin, margin:-margin]

        ind = []
        tcd_values_sliced = None
        if tcd_values is not None:
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
            ind = np.array(ind)

        if tcd_values is None or (tcd_values is not None and ind.size > 0):
            encoded_patch = {}

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
                    sliced_modality = get_sliced_modality(
                        months_folders,
                        folder_data,
                        sliced,
                        modality,
                        meteo=False,
                        dates=s2_dates,
                    )
                    log.info(f"Sliced {modality}")
                    sliced_modality = sliced_modality.sel(
                        t=slice(t_slice[0], t_slice[1])
                    )

                    # create_netcdf_s2_one_date(
                    #     xy_matrix, sliced_modality, margin, enum, date=2)

                    images, data, dates = get_s2(
                        s2_dates, t_slice, data, images, sliced_modality
                    )
                else:
                    modality = (
                        "Sentinel1_ASCENDING"
                        if sat == "s1_asc"
                        else "Sentinel1_DESCENDING"
                    )
                    log.info(f"Slicing {modality}")
                    sliced_modality = get_sliced_modality(
                        months_folders,
                        folder_data,
                        sliced,
                        modality,
                        meteo=False,
                        dates=s1_dates_asc if sat == "s1_asc" else s1_dates_desc,
                    )
                    log.info(f"Sliced {modality}")
                    sliced_modality = sliced_modality.sel(
                        t=slice(t_slice[0], t_slice[1])
                    )

                    images, data, dates = get_s1(
                        t_slice, data, images, sliced_modality, sat
                    )
                    del sliced_modality

                if sat not in dates_meteo:
                    dates_meteo[sat] = get_meteo_dates_modality(dates, sat)

            for modality in meteo_modalities:
                log.info(modality)
                sliced_meteo = get_sliced_modality(
                    months_folders, folder_data, sliced, modality.upper(), meteo=True
                )
                for sat, dates_sat in dates_meteo.items():
                    # print(modality)
                    data[sat][f"meteo_{modality}"] = build_meteo(
                        images, sliced_meteo, dates_sat, modality
                    )[modality].unsqueeze(0)
            del images, sliced_meteo
            # DEM
            nc_file = os.path.join(folder_data, "DEM", "openEO.nc")
            dem_slice = get_one_slice(nc_file, sliced)
            dem_dataset = torch.tensor(dem_slice.DEM.values)
            # We deal with the special case,
            # when dem band contains 2 dates and one of them is nan
            # So we choose the date with less nans than the other
            if len(dem_dataset) > 0:
                dem_dataset = dem_dataset[
                    torch.argmin(torch.isnan(dem_dataset).sum((1, 2)))
                ]
            dem = dem_height_aspect(dem_dataset.squeeze(0))
            for sat in data:
                data[sat]["dem"] = dem.unsqueeze(0)
            del dem
            batch_dict = {}
            for sat in data:
                sits = (
                    MMDCDataStruct.init_empty()
                    .fill_empty_from_dict(data[sat])
                    .concat_meteo()
                )
                setattr(
                    sits,
                    "meteo",
                    torch.flatten(getattr(sits, "meteo"), start_dim=2, end_dim=3),
                )
                tcd_batch = PastisBatch(
                    sits=sits,
                    input_doy=None,
                    true_doy=dates_meteo[sat][f"date_{sat}"].values[None, :],
                    padd_val=torch.Tensor([]),
                    padd_index=torch.Tensor([]),
                )
                batch_dict[sat.upper()] = tcd_batch
                encoded_patch[f"{sat}_doy"] = dates_meteo[sat][f"date_{sat}"].values
                encoded_patch[f"{sat}_mask"] = sits.data.mask[
                    0, :, :, margin:-margin, margin:-margin
                ]
            log.info("encoding")

            del data, tcd_batch

            (
                latents1,
                latents1_asc,
                latents1_desc,
                latents2,
                mask_s1,
                days_s1,
            ) = encode_one_batch(mmdc_model, batch_dict)
            encoded_patch["matrix"] = xy_matrix
            encoded_patch["s2_lat_mu"] = latents2.mean[
                0, :, :, margin:-margin, margin:-margin
            ]
            encoded_patch["s2_lat_logvar"] = latents2.logvar[
                0, :, :, margin:-margin, margin:-margin
            ]

            if s1_join:
                encoded_patch["s1_lat_mu"] = latents1.mean[
                    0, :, :, margin:-margin, margin:-margin
                ]
                encoded_patch["s1_lat_logvar"] = latents1.logvar[
                    0, :, :, margin:-margin, margin:-margin
                ]
                encoded_patch["s1_mask"] = mask_s1[
                    0, :, :, margin:-margin, margin:-margin
                ]
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

            # create_netcdf_encoded(xy_matrix,
            #                       encoded_patch["s2_lat_mu"].cpu().numpy(),
            #                       enum,
            #                       dates=dates_meteo["s2"][f"date_s2"].values)
            # create_netcdf_one_date_encoded(xy_matrix,
            #                                encoded_patch["s2_lat_mu"].cpu().numpy(),
            #                                enum,
            #                                date=2)
            # create_netcdf_encoded(xy_matrix,
            #                       encoded_patch["s1_lat_mu"].cpu().numpy(),
            #                       enum,
            #                       dates=dates_meteo["s1"][f"date_s1"].values,
            #                       s1=True)
            # create_netcdf_one_date_encoded(xy_matrix,
            #                                encoded_patch["s2_lat_mu"].cpu().numpy(),
            #                                enum,
            #                                date=2,
            #                                s1=True)

            if ind.size > 0 and tcd_values_sliced is not None:
                for sat in satellites_to_write:
                    mask = encoded_patch[f"{sat}_mask"].expand_as(
                        encoded_patch[f"{sat}_lat_mu"]
                    )

                    encoded_patch[f"{sat}_lat_mu"][mask.bool()] = torch.nan
                    encoded_patch[f"{sat}_lat_logvar"][mask.bool()] = torch.nan

                    gt_ref_values_mu = rearrange_ts(encoded_patch[f"{sat}_lat_mu"])[
                        :, ind[:, 0], ind[:, 1]
                    ].T

                    gt_ref_values_logvar = rearrange_ts(
                        encoded_patch[f"{sat}_lat_logvar"]
                    )[:, ind[:, 0], ind[:, 1]].T
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

                    df_gt.to_csv(
                        os.path.join(
                            output_folder, f"gt_tcd_t32tnt_encoded_{sat}_{enum}.csv"
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

            del (
                latents1,
                latents1_asc,
                latents1_desc,
                latents2,
                encoded_patch,
                batch_dict,
            )

    for sat in satellites_to_write:
        final_df = pd.concat(sliced_gt[sat], ignore_index=True)
        final_df.to_csv(os.path.join(output_folder, f"gt_tcd_t32tnt_encoded_{sat}.csv"))
