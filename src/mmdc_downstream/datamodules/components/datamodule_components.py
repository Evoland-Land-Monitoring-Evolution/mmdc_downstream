#!/usr/bin/env python3
# copyright: (c) 2022 cesbio / centre national d'Etudes Spatiales
"""
Collection of functions and utilities for manage the dataset of MMDC
"""

import logging
import os
import warnings
from dataclasses import fields
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import xarray as xr
from rasterio import logging as rio_logger
from shapely import geometry
from torchutils import patches
from tqdm import tqdm

from ..constants import (
    ANGLES_S1,
    ANGLES_S2,
    BANDS_S1,
    BANDS_S2,
    METEO_BANDS,
    MODALITIES,
    S1_NOISE_LEVEL,
)
from ..datatypes import MMDCDataStruct, MMDCMeteoStats, MMDCTensorStats
from .datamodule_utils import apply_log_to_s1, compute_stats, s2_angles_processing
from .datatypes import KeepCounts, PatchConfig, PatchesToFilter, PatchFilterConditions

# Configure the logger
NUMERIC_LEVEL = getattr(logging, "INFO", None)
logging.basicConfig(
    level=NUMERIC_LEVEL, format="%(asctime)-15s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

logger_rio = rio_logger.getLogger("rasterio._env")
logger_rio.setLevel(logging.ERROR)

warnings.simplefilter("always")


# Auxiliar Functions
def xarray_to_tensor(arr: xr.Dataset, bands: list[str]) -> torch.Tensor:
    """Xarray dataset to tensor with band selection"""
    return torch.Tensor(arr[bands].to_array().values).permute(1, 0, 2, 3)


def create_s1_vh_vv_ratio(radar_img: torch.Tensor) -> torch.Tensor:
    """Compute VH/VV ratio"""
    ratio = radar_img[:, 1, :, :] / radar_img[:, 0, :, :] + S1_NOISE_LEVEL
    return torch.cat((radar_img, ratio.unsqueeze(1)), 1)


def build_s1_images_and_masks(
    images: dict[str, torch.Tensor],
    xarray_ds: xr.Dataset,
    dates_df: pd.DataFrame,
    s1_type: Literal["s1_asc", "s1_desc"],
) -> dict[str, torch.Tensor]:
    """
    Generate backscatter and validity mask for a S1 data provided by OpenEO
    """
    dates = dates_df[f"date_{s1_type}"]
    xarray_ds = xarray_ds.sel(t=pd.DatetimeIndex(dates.dropna())).sortby("t")
    masks = dates.isnull().values  # non available days for validity mask

    # We open VH and VV bands from XArray dataset,
    # and we compute VH/VV as the third band
    available_images = create_s1_vh_vv_ratio(
        xarray_to_tensor(xarray_ds, bands=BANDS_S1)
    )
    images[s1_type] = torch.full((len(dates), *available_images.shape[-3:]), torch.nan)

    images[s1_type][~masks] = available_images

    available_angles = torch.cos(
        torch.deg2rad(xarray_to_tensor(xarray_ds, bands=ANGLES_S1).squeeze(1))
    )
    images[f"{s1_type}_angles"] = torch.full(
        (len(dates), *available_images.shape[-2:]), torch.nan, dtype=torch.float32
    )
    images[f"{s1_type}_angles"][~masks] = available_angles

    images[f"{s1_type}_valmask"] = torch.zeros(
        len(dates), *available_images.shape[-2:]
    ).to(torch.int8)
    images[f"{s1_type}_valmask"][masks] = 1

    # generate nan mask
    nan_mask = torch.isnan(images[s1_type]).sum(dim=1).to(torch.int8) > 0
    images[f"{s1_type}_valmask"][nan_mask] = 1

    return images


def build_s2_images_and_masks(
    images: dict[str, torch.Tensor],
    xarray_ds: xr.Dataset,
    dates_df: pd.DataFrame,
) -> dict[str, torch.Tensor]:
    """
    Generate S2 reflectance images and masks provided by OpenEO
    """
    dates = dates_df["date_s2"]
    images["CRS"] = xarray_ds.crs
    xarray_ds = xarray_ds.sel(t=pd.DatetimeIndex(dates)).sortby("t")
    images["s2_set"] = xarray_to_tensor(xarray_ds, bands=BANDS_S2).clip(
        min=0, max=20000
    )
    images["s2_masks"] = (
        torch.Tensor(
            np.logical_or(
                xarray_ds.SCL.astype(np.uint8).isin([0, 1, 3, 8, 9, 10, 11]),
                xarray_ds.CLM.astype(np.uint8).isin([1, 255]),
            ).values
        )
        .to(torch.int8)
        .unsqueeze(1)
    )
    images["s2_angles"] = s2_angles_processing(
        xarray_to_tensor(xarray_ds, bands=ANGLES_S2)
    )

    # generate nan mask
    nan_mask = torch.isnan(torch.sum(images["s2_set"], dim=1, keepdim=True))
    nan_mask_angles = torch.isnan(torch.sum(images["s2_angles"], dim=1, keepdim=True))
    images["s2_masks"] = (
        torch.cat(
            (
                images["s2_masks"][:, :1],
                nan_mask.to(torch.int8),
                nan_mask_angles.to(torch.int8),
            ),
            1,
        ).sum(dim=1, keepdim=True)
        >= 1
    )
    images["s2_masks"] = images["s2_masks"].to(torch.int8)
    return images


def build_meteo(
    images: dict[str, torch.Tensor],
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


def build_dem(
    images: dict[str, torch.Tensor],
    path_df: pd.DataFrame,
) -> dict[str, torch.Tensor]:
    """Build DEM from data provided by OpenEO"""
    dem_dataset = torch.tensor(xr.open_dataset(path_df["dem"].iloc[0][0]).DEM.values)
    # We deal with the special case,
    # when dem band contains 2 dates and one of them is nan
    # So we choose the date with less nans than the other
    if len(dem_dataset) > 0:
        dem_dataset = dem_dataset[torch.argmin(torch.isnan(dem_dataset).sum((1, 2)))]
    images["dem"] = (
        dem_height_aspect(dem_dataset.squeeze(0))
        .unsqueeze(0)
        .repeat_interleave(dim=0, repeats=images["s2_set"].shape[0])
    )
    return images


def dem_height_aspect(dem_data: torch.Tensor) -> torch.Tensor:
    """
    compute the dem gradient, and then slope and aspect
    param : dem_data
    return : stack with height, sin slope, cos aspect, sin aspect
    """
    x, y = torch.gradient(dem_data.to(torch.float32))  # pylint: disable=invalid-name
    slope = torch.rad2deg(
        torch.arctan(torch.sqrt(x * x + y * y) / 10)
    )  # 10m = resolution of SRTM
    # Aspect unfolding rules from
    # pylint: disable=C0301
    # https://github.com/r-barnes/richdem/blob/603cd9d16164393e49ba8e37322fe82653ed5046/include/richdem/methods/terrain_attributes.hpp#L236  # noqa: E501
    aspect = torch.rad2deg(torch.arctan2(x, -y))
    lt_0 = aspect < 0
    gt_90 = aspect > 90
    remaining = torch.logical_and(aspect >= 0, aspect <= 90)
    aspect[lt_0] = 90 - aspect[lt_0]
    aspect[gt_90] = 360 - aspect[gt_90] + 90
    aspect[remaining] = 90 - aspect[remaining]

    # calculate trigonometric representation
    sin_slope = torch.sin(torch.deg2rad(slope))
    cos_aspect = torch.cos(torch.deg2rad(aspect))
    sin_aspect = torch.sin(torch.deg2rad(aspect))

    return torch.stack([dem_data, sin_slope, cos_aspect, sin_aspect])


def concat_s1_asc_desc(image: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Concatenate ascending and descending orbit:
    images, angles and validity masks
    """
    image["s1_set"] = torch.cat((image["s1_asc"], image["s1_desc"]), -3)
    image["s1_valmasks"] = torch.cat(
        (image["s1_asc_valmask"].unsqueeze(-3), image["s1_desc_valmask"].unsqueeze(-3)),
        -3,
    )
    image["s1_angles"] = torch.cat(
        (image["s1_asc_angles"].unsqueeze(-3), image["s1_desc_angles"].unsqueeze(-3)),
        -3,
    )
    del image["s1_asc"], image["s1_desc"]
    del image["s1_asc_valmask"], image["s1_desc_valmask"]
    del image["s1_asc_angles"], image["s1_desc_angles"]
    return image


def create_patchified_data_struct(image: dict[str, torch.Tensor]) -> MMDCDataStruct:
    """Wrap patchified data into datastruct"""
    image = {
        mod: torch.flatten(img, start_dim=0, end_dim=1) for mod, img in image.items()
    }
    return MMDCDataStruct.init_empty().fill_empty_from_dict(image)


def patchify_data(
    image: dict[str, torch.Tensor], patch_size: int = 256, margin: int = 0
) -> dict[str, torch.Tensor]:
    """Divide ROI into patches"""
    if patch_size is not None or margin is not None:
        patchified_tensors = {
            mod: patches.flatten2d(
                patches.patchify(
                    img, patch_size, margin=margin, spatial_dim1=-2, spatial_dim2=-1
                )
            )
            for mod, img in image.items()
        }
    else:
        patchified_tensors = image.copy()
    return patchified_tensors


def create_tensor_patches(
    tiled_roi_date_df: pd.DataFrame,
    tiled_roi_path_df: pd.DataFrame,
    patch_size: int | None = None,
    patch_margin: int | None = None,
) -> dict[str, torch.Tensor]:
    """
    Create list of patches and masks to apply filter on
    :param tiled_roi_date_df: Dataframe with selected dates for each time roi
    :param tiled_roi_path_df: Dataframe with paths to .nc files for the selected roi
    :param patch_size: Size of the square patch
    :param patch_margin: Overlap of patches on each side
    """

    images: dict[str, torch.Tensor] = {}

    assert (
        len(tiled_roi_path_df) == 1
    ), "Multiple lines in path dataframe instead of only one, please, check your data"

    for modality in MODALITIES:  # iteration over different data modalities
        logger.info("Creating dataset for modality %s...", modality)

        if modality != "dem":
            modality_pathes = tiled_roi_path_df[modality].iloc[0]
            xarray_datasets = [
                xr.open_dataset(modality_pathes[y]) for y in range(len(modality_pathes))
            ]
            # different path for each year acquisitions
            xarray_datasets = xr.concat(xarray_datasets, dim="t")
            if modality == "s2":
                images = build_s2_images_and_masks(
                    images, xarray_datasets, tiled_roi_date_df
                )
            elif modality in ("s1_asc", "s1_desc"):
                images = build_s1_images_and_masks(
                    images, xarray_datasets, tiled_roi_date_df, s1_type=modality
                )
            else:
                images = build_meteo(
                    images, xarray_datasets, tiled_roi_date_df, meteo_type=modality
                )
        else:  # Only one dem image exists for all dates
            images = build_dem(images, tiled_roi_path_df)

    images.pop("CRS")

    images = concat_s1_asc_desc(images)

    logger.info("Creating patches with size %s", patch_size)

    patchified_tensors = patchify_data(images, patch_size, margin=patch_margin)

    del images

    return patchified_tensors


def keep_patch(mask: torch.Tensor, threshold: float) -> bool:
    """Predicate to decide whether a patch is kept"""
    assert mask.shape[0] == 1
    return torch.mean(1.0 - mask).item() > threshold


def patch_filter_conditions(
    in_patches: PatchesToFilter,
    threshold: float,
    counts: KeepCounts,
) -> PatchFilterConditions:
    """Compute the different conditions to filter a patch"""
    keep_s2 = keep_patch(in_patches.s2m_p, threshold)
    keep_s1asc = keep_patch(in_patches.s1m_p[:1, ...], threshold)
    keep_s1desc = keep_patch(in_patches.s1m_p[1:, ...], threshold)
    keep_dem = (in_patches.dem_p.isnan().sum() == 0).item()
    keep_meteo = (in_patches.meteo_p.isnan().sum() == 0).item()

    def update_count(count: int, keep: bool) -> int:
        return (count + 1) if keep else count

    s2_keep_count = update_count(counts.s2_keep_count, keep_s2)
    s1asc_keep_count = update_count(counts.s1asc_keep_count, keep_s1asc)
    s1desc_keep_count = update_count(counts.s1desc_keep_count, keep_s1desc)
    dem_keep_count = update_count(counts.dem_keep_count, keep_dem)
    meteo_keep_count = update_count(counts.meteo_keep_count, keep_meteo)

    return PatchFilterConditions(
        keep_s2,
        keep_s1asc,
        keep_s1desc,
        keep_dem,
        keep_meteo,
        s2_keep_count,
        s1asc_keep_count,
        s1desc_keep_count,
        dem_keep_count,
        meteo_keep_count,
    )


def filter_patches(
    data: MMDCDataStruct,
    threshold: float,
) -> list[int]:
    """
    Compute percentage of good pixels in a patch
    for the dataset and keep only patches and their indexes above threshold

    :param data: Tensor of images to go through
    :param threshold: Threshold to decide on good pixels
    """
    # filter count
    counts = KeepCounts(0, 0, 0, 0, 0)
    # filter list
    idx_filtered = []
    for i, (
        _,
        s2m_p,
        _,
        s1m_p,
        dem_p,
        meteo_p,
    ) in enumerate(
        zip(
            data.s2_data.s2_set,
            data.s2_data.s2_masks,
            data.s1_data.s1_set,
            data.s1_data.s1_valmasks,
            data.dem,
            data.meteo.concat_data(),
        )
    ):
        conditions = patch_filter_conditions(
            PatchesToFilter(s2m_p, s1m_p, dem_p, meteo_p),
            threshold,
            counts,
        )
        counts = KeepCounts(
            conditions.s2_keep_count,
            conditions.s1asc_keep_count,
            conditions.s1desc_keep_count,
            conditions.dem_keep_count,
            conditions.meteo_keep_count,
        )
        if (
            conditions.keep_s2
            and (conditions.keep_s1asc or conditions.keep_s1desc)
            and conditions.keep_dem
            and conditions.keep_meteo
        ):
            idx_filtered.append(i)

    logger.info("Kept s2: %s", counts.s2_keep_count)
    logger.info("Kept s1 asc: %s", counts.s1asc_keep_count)
    logger.info("Kept s1 desc: %s", counts.s1desc_keep_count)
    logger.info("Kept dem: %s", counts.dem_keep_count)
    logger.info("Kept meteo: %s", counts.meteo_keep_count)

    if not idx_filtered:
        logger.info("Zero patches selected!")
    else:
        logger.info(
            "Total number patches %s \n"
            "Number of patches that fulfill the conditions :"
            " %s \n"
            "Number of discarded patches : "
            " %s \n",
            data.s2_data.s2_set.shape[0],
            len(idx_filtered),
            data.s2_data.s2_set.shape[0] - len(idx_filtered),
        )

    return idx_filtered


def do_export(
    tiled_roi_date_df: pd.DataFrame,
    export_path: str,
    patch_info: tuple[str, int, str],
    tensor_list: list[torch.Tensor | MMDCTensorStats],
    tensor_label: list[str],
) -> None:
    """Write the files to disk."""
    (tile, patch_size, roi) = patch_info
    # export dataframe
    tiled_roi_date_df.to_csv(
        os.path.join(
            export_path,
            f"{tile}_dataset_{patch_size}x{patch_size}" f"_{roi}.csv",
        ),
        sep="\t",
    )

    # add patch size to the name
    tensor_label = [i + f"_{patch_size}x{patch_size}" for i in tensor_label]

    # export loop
    for tensor, label in zip(tensor_list, tensor_label):
        # # We convert nans to zeros
        # tensor = torch.nan_to_num(tensor, nan=0) if "stats" not in label else tensor
        torch.save(
            tensor,
            os.path.join(export_path, f"{tile}_{label}_{roi}.pth"),
        )


def save_roi_bounds(
    dataframe: pd.DataFrame, export_path: str, patch_info: tuple[str, int, int]
) -> None:
    """Save GPKG file with the bounds of the ROI"""
    (tile, roi, _) = patch_info
    # apply func
    with xr.open_dataset(dataframe.s2.iloc[0][0]) as raster_sample:
        roi_bounds = gpd.GeoSeries(geometry.box(*raster_sample.rio.bounds()))

        # export geoseries
        roi_bounds.to_file(
            Path(
                os.path.join(
                    export_path,
                    f"{tile}_bounds_{roi}.gpkg",
                )
            ),
            driver="GPKG",
            crs=raster_sample.crs,
        )

    del raster_sample, roi_bounds


def process_roi(
    tiled_roi_date_df: pd.DataFrame,
    tiled_roi_path_df: pd.DataFrame,
    export_path: str,
    patch_config: PatchConfig,
) -> None:
    """Patchify, filter and export data for a ROI"""
    save_roi_bounds(
        tiled_roi_path_df,
        export_path,
        (patch_config.tile, patch_config.roi, patch_config.patch_size),
    )

    # Open netcdf, select data, convert to tensor and clip to patches
    patchified_tensors = create_tensor_patches(
        tiled_roi_date_df,
        tiled_roi_path_df,
        patch_config.patch_size,
        patch_config.patch_margin,
    )

    # Keep number of patches in a ROI to use when replicating s1
    # angles asc and desc data
    nb_rep, nb_dates = patchified_tensors["s2_set"].shape[:2]

    # flatten the tensors and create data structure dataclass
    roi_data = create_patchified_data_struct(patchified_tensors)

    # divide the dataframe into clipped patches
    tiled_roi_date_df = tiled_roi_date_df.loc[tiled_roi_date_df.index.repeat(nb_rep)]
    tiled_roi_date_df["patch_id"] = list(range(nb_rep)) * nb_dates
    tiled_roi_date_df = tiled_roi_date_df.reset_index()

    logger.info("Filtering patches")
    idx_filtered = filter_patches(
        roi_data,
        patch_config.threshold,
    )

    if idx_filtered:
        # filter the datasets
        roi_data = roi_data[idx_filtered]
        # roi_data.filter_data(idx_filtered)

        # Apply log to S1 patches
        roi_data.s1_data.s1_set = apply_log_to_s1(roi_data.s1_data.s1_set)

        # filter the datafrmaes
        tiled_roi_date_df = tiled_roi_date_df.loc[idx_filtered, :]

        # Compute stats
        logger.info("Computing stats for S2 images")
        stats_s2 = compute_stats(
            roi_data.s2_data.s2_set.to(torch.float32),
            mask=roi_data.s2_data.s2_masks.expand_as(roi_data.s2_data.s2_set),
        )

        logger.info("Computing stats for S1 images")
        stats_s1 = compute_stats(
            roi_data.s1_data.s1_set,
            mask=roi_data.s1_data.s1_valmasks.repeat_interleave(3, -3),
            quant=(0.05, 0.5, 0.99),
        )

        logger.info("Computing stats for meteo")
        stats_meteo = MMDCMeteoStats(
            *[
                compute_stats(
                    torch.flatten(
                        getattr(roi_data.meteo, m.name).to(torch.float32),
                        start_dim=0,
                        end_dim=1,
                    ).unsqueeze(1)
                )
                for m in fields(roi_data.meteo)
            ]
        )

        logger.info("Computing stats for dem")
        stats_dem = compute_stats(roi_data.dem)

        logger.info("Exporting tensors")
        # export to torchfiles
        tensor_list: list[torch.Tensor | MMDCTensorStats] = [
            roi_data.s2_data.s2_set,
            roi_data.s2_data.s2_masks,
            roi_data.s2_data.s2_angles,
            roi_data.s1_data.s1_set,
            roi_data.s1_data.s1_valmasks,
            roi_data.s1_data.s1_angles,
            roi_data.meteo.dew_temp,
            roi_data.meteo.prec,
            roi_data.meteo.sol_rad,
            roi_data.meteo.temp_max,
            roi_data.meteo.temp_mean,
            roi_data.meteo.temp_min,
            roi_data.meteo.vap_press,
            roi_data.meteo.wind_speed,
            roi_data.dem,
            stats_s2,
            stats_s1,
            stats_meteo.dew_temp,
            stats_meteo.prec,
            stats_meteo.sol_rad,
            stats_meteo.temp_max,
            stats_meteo.temp_mean,
            stats_meteo.temp_min,
            stats_meteo.vap_press,
            stats_meteo.wind_speed,
            stats_dem,
        ]

        # create label

        tensor_label: list[str] = [
            "s2_set",
            "s2_masks",
            "s2_angles",
            "s1_set",
            "s1_valmasks",
            "s1_angles",
            "meteo_dew_temp",
            "meteo_prec",
            "meteo_sol_rad",
            "meteo_temp_max",
            "meteo_temp_mean",
            "meteo_temp_min",
            "meteo_vap_press",
            "meteo_wind_speed",
            "dem",
            "stats_s2",
            "stats_s1",
            "stats_meteo_dew_temp",
            "stats_meteo_prec",
            "stats_meteo_sol_rad",
            "stats_meteo_temp_max",
            "stats_meteo_temp_mean",
            "stats_meteo_temp_min",
            "stats_meteo_vap_press",
            "stats_meteo_wind_speed",
            "stats_dem",
        ]
        logger.info(
            "s2 tensor of shape N,C,H,W is %s. N is the number of patches",
            roi_data.s2_data.s2_set.shape,
        )
        logger.info(
            "s1 tensor of shape N,C,H,W is %s. N is the number of patches ",
            roi_data.s1_data.s1_set.shape,
        )

        do_export(
            tiled_roi_date_df,
            export_path,
            (patch_config.tile, patch_config.patch_size, patch_config.roi),
            tensor_list,
            tensor_label,
        )

        # del unnecesary data
        del roi_data
        del tensor_label, tensor_list


def store_data(
    path_df: pd.DataFrame,
    date_df: pd.DataFrame,
    export_dir: Path,
    patch_config: tuple[int, int],
    threshold: float,
) -> None:
    """
    Preprocess the data and save in torch files
    """
    (patch_size, patch_margin) = patch_config
    tiles = list(date_df.tile.unique())
    rois = list(date_df.roi.unique())

    logger.info("%s", tiles)
    for tile in tqdm(tiles):
        logger.info("Processing tile : %s", tile)
        # export path
        export_path = Path(export_dir) / tile
        # check the folder existence
        if not export_path.exists():
            export_path.mkdir()

        for roi in rois:
            logger.info("Processing tile : %s roi : %s", tile, roi)

            # slice dataframe
            tiled_roi_date_df = date_df[(date_df.roi == roi) & (date_df.tile == tile)]
            tiled_roi_path_df = path_df[(path_df.roi == roi) & (path_df.tile == tile)]

            if not len(tiled_roi_date_df.roi) > 0:
                logger.info("No available data for ROI %s tile %s", roi, tile)
                logger.info("%s", tiled_roi_date_df.head())
                continue
            process_roi(
                tiled_roi_date_df,
                tiled_roi_path_df,
                export_path,
                PatchConfig(roi, tile, patch_size, patch_margin, threshold),
            )
