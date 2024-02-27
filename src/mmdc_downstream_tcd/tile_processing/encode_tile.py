import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from mmdc_singledate.datamodules.components.datamodule_components import (
    apply_log_to_s1,
    build_meteo,
    build_s1_images_and_masks,
    build_s2_images_and_masks,
    dem_height_aspect,
)

# from mmdc_downstream_pastis.utils.utils import MMDCPartialBatch

METEO_DAYS_BEFORE = 4
METEO_DAYS_AFTER = 1


def get_one_slice(nc_file: str | Path, slice: list[float]) -> xr.Dataset:
    """Get one slice for one month"""
    with xr.open_dataset(
        nc_file,
        decode_coords="all",
        mask_and_scale=False,
        decode_times=True,
        engine="netcdf4",
    ) as file:
        return file.rio.slice_xy(*slice)


def get_sliced_modality(
    months: list[str],
    folder: str | Path,
    slice: list[float],
    modality: str,
    meteo: bool = False,
) -> xr.Dataset:
    """Get a dataset slice for one modality for all available dates"""
    modality_slice_list = []
    for month in months:
        if meteo:
            nc_file = os.path.join(folder, month, "AGERA5", modality, "openEO.nc")
        else:
            nc_file = os.path.join(folder, month, modality, "openEO.nc")
        file = get_one_slice(nc_file, slice)
        modality_slice_list.append(file)
        file.close()
    return xr.concat(modality_slice_list, dim="t").sortby("t")


def get_meteo_dates_modality(dates_df, satellite):
    meteo_before = pd.Timedelta(METEO_DAYS_BEFORE, "D")  # day_S2 - n_days
    meteo_after = pd.Timedelta(METEO_DAYS_AFTER, "D")  # day_S2 + n_days

    dates_df["meteo_start_date"] = dates_df["date_" + satellite] - meteo_before
    dates_df["meteo_end_date"] = dates_df["date_" + satellite] + meteo_after
    return dates_df


# def filter_clouds():
#


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
meteo_modalities = [k.upper() for k in METEO_BANDS]


# json_data = gpd.read_file(geojson_path)
# utm_crs = json_data.estimate_utm_crs()
# json_data = json_data.to_crs(utm_crs)
#
# tile_geom = json_data.iloc[0]["geometry"]
# tile_coords = json_data.get_coordinates(include_z=False)


def get_bounds(folder_data: str | Path, months_folders: list[str | int]):
    """
    Get tile bounds using a reference image,
    considering that geojson vector data does not match with rasters' boundaries
    """
    # Reference image
    nc_file = os.path.join(folder_data, months_folders[0], "Sentinel2", "openEO.nc")
    # nc_file = os.path.join(folder_data, "3", "AGERA5", "DEW_TEMP", "openEO.nc")

    with xr.open_dataset(nc_file) as image_ref:
        image_ref = image_ref.sel(t=image_ref.t[0])
        xmax, ymax, xmin, ymin = (
            image_ref.x.values.max(),
            image_ref.y.values.max(),
            image_ref.x.values.min(),
            image_ref.y.values.min(),
        )
        return xmax, ymax, xmin, ymin


def encode_tile(
    folder_data: str | Path,
    months_folders: list[str | int],
    geojson_path: str | Path,
    window: int,
):
    xmax, ymax, xmin, ymin = get_bounds(folder_data, months_folders)

    # TODO add margins in the future
    slices_list = []
    for slice_xmin in np.arange(xmin, xmax, window * 10):
        for slice_ymax in np.arange(ymax, ymin, -window * 10):
            slice_xmax, slice_ymin = slice_xmin + window * 10, slice_ymax - window * 10
            if slice_xmax > xmax:
                slice_xmax = xmax
            if slice_ymin < ymin:
                slice_ymin = ymin
            slices_list.append(
                [slice_xmin, slice_ymin + 10, slice_xmax - 10, slice_ymax]
            )

    dates = {}
    for slice in slices_list:
        images = {}
        data = {"s2": {}, "s1_asc": {}, "s1_desc": {}}
        for modality in ("Sentinel2", "Sentinel1_ASCENDING", "Sentinel1_DESCENDING"):
            if modality == "Sentinel2":
                sliced_modality = get_sliced_modality(
                    months_folders, folder_data, slice, modality, meteo=False
                )
                sliced_modality = sliced_modality.sel(
                    t=slice("2018-03-05", "2019-10-30")
                )
                dates = pd.DatetimeIndex(sliced_modality.t.values)
                if modality == "Sentinel2":
                    dates = dates.to_frame(name="date_s2")
                    images = build_s2_images_and_masks(
                        images, sliced_modality.astype(np.float32), dates
                    )
                    data["s2"]["img"] = images["s2_set"]
                    data["s2"]["mask"] = images["s2_masks"]
                    data["s2"]["angles"] = images["s2_angles"]
                    images["CRS"] = sliced_modality.crs
                    sat = "s2"
                else:
                    sat = "s1_asc" if modality == "Sentinel1_ASCENDING" else "s1_desc"
                    dates = dates.to_frame(name=f"date_{sat}")
                    images = build_s1_images_and_masks(
                        images, sliced_modality, dates, sat
                    )

                    data[sat]["img"] = apply_log_to_s1(images[sat])
                    data[sat]["mask"] = images[f"{sat}_valmask"]
                    data[sat]["angles"] = images[f"{sat}_angles"]

                if sat not in dates:
                    dates[sat] = get_meteo_dates_modality(dates, sat)

        for modality in meteo_modalities:
            sliced_meteo = get_sliced_modality(
                months_folders, folder_data, slice, modality, meteo=True
            )
            for sat, dates_sat in dates.items():
                data[sat][modality] = build_meteo(
                    images, sliced_meteo, dates_sat, modality
                )[modality]

        # DEM
        nc_file = os.path.join(folder_data, "DEM", "openEO.nc")
        dem_slice = get_one_slice(nc_file, slice)
        dem_dataset = torch.tensor(dem_slice.DEM.values)
        # We deal with the special case,
        # when dem band contains 2 dates and one of them is nan
        # So we choose the date with less nans than the other
        if len(dem_dataset) > 0:
            dem_dataset = dem_dataset[
                torch.argmin(torch.isnan(dem_dataset).sum((1, 2)))
            ]
        dem = dem_height_aspect(dem_dataset.squeeze(0))
        for key in data:
            data[key]["dem"] = dem

        # class PastisBatch:
        #     sits: MMDCDataStruct
        #     input_doy: torch.Tensor
        #     padd_val: torch.Tensor
        #     padd_index: torch.Tensor
        #     true_doy: torch.Tensor = None
        #
        # class MMDCDataStruct:
        #     """Dataclass holding the tensors for image and auxiliary data in MMDC"""
        #
        #     data: MMDCData
        #     meteo: MMDCMeteoData | torch.Tensor
        #     dem: torch.Tensor
        #
        # class MMDCData:
        #     """Dataclass holding the tensors for image and auxiliary data in MMDC"""
        #
        #     img: torch.Tensor | None
        #     mask: torch.Tensor | None
        #     angles: torch.Tensor | None
