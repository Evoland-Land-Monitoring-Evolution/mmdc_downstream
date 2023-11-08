#!/usr/bin/env python3
# copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales

"""
This module opens .nc files to find coocurrences between S2 and S2 asc_desc images
and also creates a tile descriptor file with paths to every data type
"""

import logging
import os
import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from mmdc_singledate.datamodules.components.datatypes import SeriesDays

from ..constants import D_MODALITY, MODALITIES
from .datamodule_utils import find_path_by_year

my_logger = logging.getLogger(__name__)


def extract_paths(
    input_path: Path, output_path: Path, tiles: list[str]
) -> pd.DataFrame:
    """
    Find paths to dataset files and save them to csv file
    """

    Path(output_path).mkdir(exist_ok=True, parents=True)

    mmdc_df = build_dataset_info(path_dir=input_path, l_tile_s2=tiles)

    for tile in list(mmdc_df.tile.unique())[::-1]:
        Path(os.path.join(output_path, tile)).mkdir(exist_ok=True)
        mmdc_df[mmdc_df.tile == tile].to_csv(
            Path(os.path.join(output_path, tile)).joinpath(
                f"{tile}_tile_description.csv"
            ),
            sep="\t",
        )

    mmdc_df.to_csv(Path(output_path).joinpath("tiles_descriptions.csv"), sep="\t")
    return mmdc_df


def extract_dates(
    mmdc_df: pd.DataFrame,
    output: Path,
    days_selection: SeriesDays = (0, 2, 2, None, None),
    asc_des_strict: bool = False,
    prefilter_clouds: float = 0.9,
) -> pd.DataFrame:
    """
    We create separate dfs per tile with S2, S1 asc, S1 desc date acquisitions,
    meteo TS start and end dates,
    S2 cloud coverage, roi index and Tile
    """

    sel_dates_all = []
    for _, row in mmdc_df.iterrows():
        tile = row.tile
        roi = row.roi
        sel_dates = match_s1_s2(
            row,
            days_selection=days_selection,
            asc_des_strict=asc_des_strict,
            prefilter_clouds=prefilter_clouds,
        )
        sel_dates["roi"] = roi
        sel_dates["tile"] = tile
        Path(os.path.join(output, tile)).mkdir(exist_ok=True, parents=True)
        sel_dates.to_csv(
            os.path.join(
                output,
                tile,
                f"{tile}_{roi}.csv",
            ),
            sep="\t",
        )
        sel_dates_all.append(sel_dates)
        my_logger.info(
            "%i image dates were preselected for ROI %s Tile %s",
            len(sel_dates),
            roi,
            tile,
        )
    sel_dates_all = pd.concat(sel_dates_all)
    return sel_dates_all


def compute_cloud_coverage(
    s2_arr: xr.Dataset, prefilter_clouds: float
) -> (np.array, xr.Dataset):
    """
    Compute cloud coverage per patch to filter out patches
    with cloud coverage greater than prefilter_clouds.
    """
    cldb = s2_arr["CLM"] == 1
    ccp = cldb.sum(dim=["x", "y"])
    ccp = ccp.compute()
    if prefilter_clouds is not None:
        max_pix_cc = s2_arr.sizes["y"] * s2_arr.sizes["x"] * prefilter_clouds
        t_sel = ccp.where(ccp < max_pix_cc, drop=True)["t"]
        s2_arr = s2_arr.sel(t=t_sel)
        ccp = ccp.sel(t=t_sel)
    return (ccp / (s2_arr.sizes["y"] * s2_arr.sizes["x"])).values, s2_arr


def append_cloud_coverage(
    final_dates: pd.DataFrame, s2_dates_df: pd.DataFrame, cloud_coverage: np.asarray
) -> pd.DataFrame:
    """
    We append cloud coverage to the dataframe
    """

    s2_dates_clouds_df = s2_dates_df.copy()
    s2_dates_clouds_df["clouds"] = cloud_coverage.round(2)
    final_dates = pd.merge(final_dates, s2_dates_clouds_df, on="date_s2", how="inner")
    return final_dates


def match_pairs(
    s2_dates_df: pd.DataFrame,
    s1_asc_dates_df: pd.DataFrame,
    s1_desc_dates_df: pd.DataFrame,
    tolerance: pd.Timedelta,
    asc_des_strict: bool,
):
    """Matches dates dataframes"""

    s2_s1_asc = pd.merge_asof(
        s2_dates_df,
        s1_asc_dates_df,
        left_on="date_s2",
        right_on="date_s1_asc",
        tolerance=tolerance,
    ).dropna()
    s2_s1_desc = pd.merge_asof(
        s2_dates_df,
        s1_desc_dates_df,
        left_on="date_s2",
        right_on="date_s1_desc",
        tolerance=tolerance,
    ).dropna()
    if not asc_des_strict:
        final_dates = pd.merge(s2_s1_asc, s2_s1_desc, on="date_s2", how="outer")
    else:
        final_dates = pd.merge(s2_s1_asc, s2_s1_desc, on="date_s2", how="inner")

    return final_dates


def match_one_year(
    paths: dict[str, str | None],
    prefilter_clouds: float,
    tolerance: pd.Timedelta,
    asc_des_strict,
) -> pd.DataFrame:
    """Matches S1 and S2 images from one year"""
    s2_arr = xr.open_dataset(paths["s2"])
    # We deal with the case where only one S1 orbit is available for whole year
    s1_asc_arr = (
        xr.open_dataset(paths["s1_asc"]) if paths["s1_asc"] is not None else None
    )
    s1_desc_arr = (
        xr.open_dataset(paths["s1_desc"]) if paths["s1_desc"] is not None else None
    )

    cloud_coverage, s2_arr = compute_cloud_coverage(s2_arr, prefilter_clouds)

    s2_dates = pd.DatetimeIndex(s2_arr.t)
    # If data for one of the orbits is not available,
    # we set random very far away date to avoid bugs
    s1_asc_dates = (
        pd.DatetimeIndex(s1_asc_arr.t)
        if s1_asc_arr is not None
        else pd.DatetimeIndex(["1990-11-01"])
    )
    s1_desc_dates = (
        pd.DatetimeIndex(s1_desc_arr.t)
        if s1_desc_arr is not None
        else pd.DatetimeIndex(["1990-11-01"])
    )

    s2_dates_df = s2_dates.to_frame(name="date_s2")
    s1_asc_dates_df = s1_asc_dates.to_frame(name="date_s1_asc")
    s1_desc_dates_df = s1_desc_dates.to_frame(name="date_s1_desc")

    final_dates = match_pairs(
        s2_dates_df, s1_asc_dates_df, s1_desc_dates_df, tolerance, asc_des_strict
    )

    final_dates = append_cloud_coverage(final_dates, s2_dates_df, cloud_coverage)

    return final_dates


def update_meteo_min_max(
    paths: dict[str, str | None],
    meteo_dates_min: pd.DatetimeIndex,
    meteo_dates_max: pd.DatetimeIndex,
) -> (pd.DatetimeIndex, pd.DatetimeIndex):
    """Compute meteo min and max dates for the tile"""
    meteo_arr = xr.open_dataset(paths["dew_temp"])
    meteo_dates = pd.DatetimeIndex(meteo_arr.t)
    meteo_dates_max = max(meteo_dates.max(), meteo_dates_max)
    meteo_dates_min = min(meteo_dates.min(), meteo_dates_min)
    return meteo_dates_min, meteo_dates_max


def match_s1_s2(
    roi_df: pd.Series,
    days_selection: SeriesDays = (0, 2, 2, None, None),
    asc_des_strict: bool = False,
    prefilter_clouds: float = 0.9,
) -> pd.DataFrame:
    """
    Find S1 and S2 images that correspond to the same day,
    and corresponding beginning and end of meteo series data.
    """
    tolerance = pd.Timedelta(days_selection.days_gap, "D")
    meteo_before = pd.Timedelta(days_selection.meteo_days_before, "D")
    meteo_after = pd.Timedelta(days_selection.meteo_days_after, "D")
    start_date = (
        pd.to_datetime(days_selection.start_date_series)
        if days_selection.start_date_series is not None
        else None
    )
    end_date = (
        pd.to_datetime(days_selection.end_date_series)
        if days_selection.end_date_series is not None
        else None
    )

    # We initialize min/max meteo dates for further filtering of S2 image dates
    # that do not have all dates for meteo series
    meteo_dates_max = pd.to_datetime("1900-01-01")
    meteo_dates_min = pd.to_datetime("2100-12-31")

    sel_dates = []

    # We iterate through all years, using S2 img as reference
    for s2_path in roi_df.s2:
        # Sometimes we don't have all the years available for each product
        # We need to have S2, at least one S1 orbit,
        # DEM and all meteo data for a given year
        paths = find_path_by_year(s2_path, roi_df)
        if paths["s2"] is not None:
            final_dates = match_one_year(
                paths, prefilter_clouds, tolerance, asc_des_strict
            )

            sel_dates.append(final_dates)

            meteo_dates_min, meteo_dates_max = update_meteo_min_max(
                paths, meteo_dates_min, meteo_dates_max
            )

    sel_dates = (
        pd.concat(sel_dates, ignore_index=True)
        .sort_values(by="date_s2")
        .reset_index(drop=True)
    )

    # Filter by user-defined start and end dates
    if start_date is not None:
        sel_dates = sel_dates[sel_dates["date_s2"] >= start_date]
    if end_date is not None:
        sel_dates = sel_dates[sel_dates["date_s2"] <= end_date]

    sel_dates["meteo_start_date"] = sel_dates["date_s2"] - meteo_before
    sel_dates["meteo_end_date"] = sel_dates["date_s2"] + meteo_after

    # Filter so each S2 image has complete meteo series
    sel_dates = sel_dates[
        (sel_dates["meteo_start_date"] >= meteo_dates_min)
        & (sel_dates["meteo_end_date"] <= meteo_dates_max)
    ]
    sel_dates = sel_dates.reset_index(drop=True)
    return sel_dates


def build_dataset_info(path_dir: Path, l_tile_s2: list[str]) -> pd.DataFrame:
    """Build dataframe with paths to dataset data"""
    d_mod = None
    for mod in MODALITIES:
        df_mod = load_mmdc_path(path_dir=path_dir, modality=mod, s2_tile=l_tile_s2)
        if d_mod is None:
            d_mod = df_mod
        else:
            d_mod = pd.merge(d_mod, df_mod, on=["roi", "tile"], how="inner")

    return d_mod


def load_mmdc_path(
    path_dir: Path,
    s2_tile: list[str],
    modality: Literal[
        "s2",
        "S1_asc",
        "s1_desc",
        "dem",
        "dew_temp",
        "prec",
        "sol_rad",
        "temp_max",
        "temp_mean",
        "temp_min",
        "vap_press",
        "wind_speed",
    ],
) -> pd.DataFrame:
    """
    Find paths to individual modalities of dataset
    """
    assert Path(path_dir).exists(), f"{path_dir} not found"

    format_sits = "clip.nc"

    list_available_sits = list(
        Path(path_dir).rglob(f"*/*{D_MODALITY[modality]}*/**/*_{format_sits}")
    )
    my_logger.debug(list_available_sits[0])
    available_patch = list({elem.name for elem in list_available_sits})
    my_logger.debug("available patch %s", available_patch)
    l_df = []
    for tile in s2_tile:
        for patch_id in sorted(available_patch):
            pattern = f"{tile}/**/*{D_MODALITY[modality]}*/**/{patch_id}"
            my_logger.debug("path dir %s patch id %s", path_dir, patch_id)
            l_s2 = [str(p) for p in sorted(Path(path_dir).rglob(pattern))]
            id_series = {
                "roi": re.findall("[0-9]+", patch_id)[0],
                "tile": tile,
                modality: l_s2,
            }
            assert l_s2, ("No image found at %s at %s", pattern, path_dir)
            l_df += [pd.DataFrame([id_series])]
    my_logger.debug(l_df)
    final_df = pd.concat(l_df, ignore_index=True)
    return final_df
