#!/usr/bin/env python3
# Copyright: (c) 2022 CESBIO / Centre National d'Etudes Spatiales
"""
Collection of functions and utilities for process some data
related with MMDC project


"""

# imports

import glob
import logging
import re
from collections.abc import Iterable, Iterator
from itertools import chain
from pathlib import Path
from typing import AnyStr, Literal, TypeVar, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import torch
from shapely import geometry

from ..constants import METEO_BANDS, MMDC_DATA, MODALITIES
from ..datatypes import MMDCTensorStats

# Configure the logger
NUMERIC_LEVEL = getattr(logging, "INFO", None)
logging.basicConfig(
    level=NUMERIC_LEVEL, format="%(asctime)-15s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


T = TypeVar("T", torch.Tensor, np.ndarray)


def apply_log_to_s1(
    data: T,
    clip_min: float = 1e-4,
    clip_max: float = 2.0,
) -> T:
    """
    Apply log to S1 data

    :param data: S1 data
    :param clip_min: Min value to clip S1 data and avoid having zeros
    :param clip_max: Max value for S1 data
    """
    data[~torch.isnan(data)] = torch.log10(
        torch.clip(data[~torch.isnan(data)], clip_min, clip_max)
    )
    if isinstance(data, torch.Tensor):
        return data
    return cast(np.ndarray, data)


def compute_roi_envelope(raster_bounds: rio.DatasetReader) -> gpd.GeoSeries:
    """
    Compute the ROI envelope and return a geoseries
    """
    bbox = geometry.box(
        raster_bounds.bounds.left,
        raster_bounds.bounds.bottom,
        raster_bounds.bounds.right,
        raster_bounds.bounds.top,
    )

    return gpd.GeoSeries(bbox)


def s2_angles_processing(s2_angles_data: torch.Tensor) -> torch.Tensor:
    """
    Transform S2 angles from degrees to sin/cos
    """

    (sun_az, sun_zen, view_az, view_zen) = (
        s2_angles_data[:, 0, ...],
        s2_angles_data[:, 1, ...],
        s2_angles_data[:, 2, ...],
        s2_angles_data[:, 3, ...],
    )

    return torch.stack(
        [
            torch.cos(torch.deg2rad(sun_zen)),
            torch.cos(torch.deg2rad(sun_az)),
            torch.sin(torch.deg2rad(sun_az)),
            torch.cos(torch.deg2rad(view_zen)),
            torch.cos(torch.deg2rad(view_az)),
            torch.sin(torch.deg2rad(view_az)),
        ],
        dim=1,
    )


MAX_SAMPLES_QUANTILE = 16_000_000


def compute_stats_1d(
    data: torch.Tensor, quant: tuple[float, float, float] = (0.05, 0.5, 0.95)
) -> MMDCTensorStats:
    """Compute stats (MMDCDataStats) for a 1D vector and limit to
    MAX_SAMPLES_QUANTILE points."""
    assert len(data.shape) == 1

    nbpixels = data.shape[0]
    mean = data.mean()
    std = data.std()
    # Compute quantiles using a percentage of pixels
    # the quantile function not work with more that 16 millions of points

    # computing quantiles
    if nbpixels > MAX_SAMPLES_QUANTILE:
        quantiles = data[torch.randperm(16_000_000)].quantile(
            torch.tensor(quant, dtype=torch.float), dim=0
        )
    else:
        quantiles = data.quantile(torch.tensor(quant, dtype=torch.float), dim=0)
    qmin, median, qmax = quantiles[0], quantiles[1], quantiles[2]

    logger.info(
        "Stats:\n mean: %s\n std: %s\n qmin: %s\n median: %s\n qmax: %s",
        mean,
        std,
        qmin,
        median,
        qmax,
    )

    return MMDCTensorStats(
        torch.nan_to_num(mean),
        torch.nan_to_num(std),
        torch.nan_to_num(qmin),
        torch.nan_to_num(median),
        torch.nan_to_num(qmax),
    )


def mask_and_compute_stats_1d(
    flattened_data_1d: torch.Tensor,
    quant: tuple[float, float, float],
    mask: torch.Tensor,
    dim: int,
) -> MMDCTensorStats:
    """Compute stats for a 1D vector using a validity mask. The
    validity mask has more than 1D and the corresponding dimension has
    to be given with the dim parameter

    """
    flattened_masks = mask < 1
    logger.info("Flattened masks: %s", flattened_masks.shape)
    flattened_masks_1d = flattened_masks

    logger.info("Flattened masks 1d: %s", flattened_masks_1d.shape)
    flattened_data_1d = flattened_data_1d[flattened_masks_1d]
    if len(flattened_data_1d) == 0:
        logger.info("No valid data for dimension %s", dim)
        return MMDCTensorStats(
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
        )
    return compute_stats_1d(flattened_data_1d, quant)


def compute_stats(
    data: torch.Tensor,
    quant: tuple[float, float, float] = (0.05, 0.5, 0.95),
    mask: torch.Tensor | None = None,
) -> MMDCTensorStats:
    """
    Compute stats from data

    :param data: Tensor of data with shape [N,C,H,W] or [D,C] (D=N*H*W)
    :param q: Q-th quantiles to compute
    :param mask: Optional tensor with validity masks for stats
    """
    if torch.isnan(data).any().item():
        logger.info("NaN detected before stats computation")
    # Merge all pixels to get D,C shape (D=N*H*W)
    flattened_data = data.transpose(1, -1).flatten(0, -2)
    flattened_mask = mask.transpose(1, -1).flatten(0, -2) if mask is not None else None
    mean: torch.Tensor | None = None
    std: torch.Tensor | None = None
    qmin: torch.Tensor | None = None
    median: torch.Tensor | None = None
    qmax: torch.Tensor | None = None

    logger.info("Flattened data : %s", flattened_data.shape)
    # Valid pixels can be different in each band (ex. S1 asc/desc)
    for dim in range(flattened_data.shape[1]):
        flattened_data_1d = flattened_data[:, dim]
        logger.info("Flattened data 1d: %s", flattened_data_1d.shape)
        stats_1d: MMDCTensorStats
        if mask is None:
            stats_1d = compute_stats_1d(flattened_data_1d, quant)
        else:
            flattened_mask_1d = flattened_mask[:, dim]
            stats_1d = mask_and_compute_stats_1d(
                flattened_data_1d, quant, flattened_mask_1d, dim
            )
        if (
            mean is not None
            and std is not None
            and qmin is not None
            and median is not None
            and qmax is not None
        ):
            mean = torch.cat(
                [mean, stats_1d.mean.reshape(1, 1)],
                dim=1,
            )
            std = torch.cat(
                [std, stats_1d.std.reshape(1, 1)],
                dim=1,
            )
            qmin = torch.cat(
                [qmin, stats_1d.qmin.reshape(1, 1)],
                dim=1,
            )
            median = torch.cat(
                [median, stats_1d.median.reshape(1, 1)],
                dim=1,
            )
            qmax = torch.cat(
                [qmax, stats_1d.qmax.reshape(1, 1)],
                dim=1,
            )
        else:
            mean = stats_1d.mean.reshape(1, 1)
            std = stats_1d.std.reshape(1, 1)
            qmin = stats_1d.qmin.reshape(1, 1)
            median = stats_1d.median.reshape(1, 1)
            qmax = stats_1d.qmax.reshape(1, 1)
    if (
        mean is not None
        and std is not None
        and qmin is not None
        and median is not None
        and qmax is not None
    ):
        return MMDCTensorStats(
            torch.nan_to_num(mean),
            torch.nan_to_num(std),
            torch.nan_to_num(qmin),
            torch.nan_to_num(median),
            torch.nan_to_num(qmax),
        )
    return MMDCTensorStats(
        stats_1d.mean.reshape(1, 1),
        stats_1d.std.reshape(1, 1),
        stats_1d.qmin.reshape(1, 1),
        stats_1d.median.reshape(1, 1),
        stats_1d.qmax.reshape(1, 1),
    )


def average_stats(
    samples: Iterable[MMDCTensorStats], sanity_check: bool = False
) -> MMDCTensorStats:
    """Compute the average for each of the stats given a list of
    MMDCDataStats

    """
    local_samples = samples
    if sanity_check:
        local_samples = [
            s
            for s in samples
            if not (torch.any(s.qmin < -1e35) or torch.any(torch.isnan(s.qmin)))
        ]  # This is rather ad-hoc for some WC ROIS with corrupted data
        if not local_samples:
            raise RuntimeError("No valid stats")
    means = torch.stack([s.mean for s in local_samples]).mean(0)
    stds = torch.stack([s.std for s in local_samples]).mean(0)
    qmins = torch.stack([s.qmin for s in local_samples]).mean(0)
    medians = torch.stack([s.median for s in local_samples]).mean(0)
    qmaxs = torch.stack([s.qmax for s in local_samples]).mean(0)
    return MMDCTensorStats(means, stds, qmins, medians, qmaxs)


def generate_path_for_tensor(
    path_to_exported_files: Path,
    set_name: str,
    data_source: Literal[tuple(MMDC_DATA)],
    suffix: str,
    tile_list: list[str],
) -> list[AnyStr]:
    """Generate a list for the tensor files containing a data set"""
    return sorted(
        list(
            chain(
                *[  # type: ignore
                    [
                        fn
                        for fn in glob.glob(
                            f"{path_to_exported_files}/{i}/*{set_name}_"
                            f"{data_source}*{suffix}"
                        )
                        if "stats" not in fn
                    ]
                    for i in tile_list
                ]
            )
        )
    )


def create_tensors_path_set(
    set_name: str,
    path_to_exported_files: Path,
    tile_list: list[str],
    roi_list: list[int] | None = None,
) -> Iterator[tuple]:
    """Generate the paths for the tensor files containing a data set"""
    suffix = ""
    if roi_list is not None:
        assert all(i < 10 for i in roi_list), "Only ROI indices < 10 are accepted."
        roi_regex = "".join([str(i) for i in roi_list])
        suffix = f"_[{roi_regex}].pth"

    list_of_paths = [
        generate_path_for_tensor(
            path_to_exported_files, set_name, modality, suffix, tile_list
        )
        for modality in MMDC_DATA
    ]

    return zip(*list_of_paths)


def find_path_by_year(s2_path: str, roi_df: pd.Series) -> dict[str, str or None]:
    """
    We check if we have all the products available for a given year.
    We need to have S2, at least one S1 orbit, DEM and all meteo data,
    otherwise code returns None paths for all the products.
    """

    year = re.search(r"(\d{4})", s2_path).group(0)
    if (
        any(year in s for s in roi_df.s1_asc) or any(year in s for s in roi_df.s1_desc)
    ) and any(year in s for k in METEO_BANDS for s in roi_df[k]):
        all_meteo_paths = np.asarray([y for k in METEO_BANDS for y in roi_df[k]])
        meteo_paths = all_meteo_paths[[year in s for s in all_meteo_paths]]
        if len(meteo_paths) < len(METEO_BANDS.keys()):
            return {k: None for k in MODALITIES}

        s1_asc_path = (
            np.asarray(roi_df.s1_asc)[[year in s for s in roi_df.s1_asc]][0]
            if any(year in s for s in roi_df.s1_asc)
            else None
        )
        s1_desc_path = (
            np.asarray(roi_df.s1_desc)[[year in s for s in roi_df.s1_desc]][0]
            if any(year in s for s in roi_df.s1_desc)
            else None
        )
        dem_path = roi_df.dem[0]
        all_paths = [s2_path, s1_asc_path, s1_desc_path, dem_path, *meteo_paths]
        return {k: all_paths[i] for i, k in enumerate(MODALITIES)}
    return {k: None for k in MODALITIES}
