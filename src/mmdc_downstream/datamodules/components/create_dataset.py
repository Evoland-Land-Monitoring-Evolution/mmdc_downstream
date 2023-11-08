"""Function for dataset creation from existing ROIS. The geotiffs are
patchified, the patches are filtered and everything is exported to torch
tensors."""

import logging
import sys
import warnings
from pathlib import Path

from rasterio import logging as rio_logger

from mmdc_singledate.datamodules.components.compute_paths_and_dates_df import (
    extract_dates,
    extract_paths,
)

from .datamodule_components import store_data
from .datatypes import DatasetPaths, MatchParams, PatchCreationParams, SeriesDays

# Configure the logger
NUMERIC_LEVEL = getattr(logging, "INFO", None)
logging.basicConfig(
    level=NUMERIC_LEVEL, format="%(asctime)-15s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

logger_rio = rio_logger.getLogger("rasterio._env")
logger_rio.setLevel(logging.ERROR)

warnings.simplefilter("always")


def create_dataset(
    dataset_paths: DatasetPaths,
    input_tile_list: list | None,
    days_selection: SeriesDays(0, 2, 2, None, None),
    patch_params: PatchCreationParams = PatchCreationParams(32, 4, 0.95),
    match_params: MatchParams = MatchParams(False, 0.95, False),
) -> None:
    """
    Create Dataset :
    1: Generate paths to all data
    2: Create a csv file with matching dates
    3: Open data from netcdf and write it to tensors
    """

    (patch_size, patch_margin, threshold) = patch_params
    (s1_asc_desc_strict, prefilter_clouds, only_csv) = match_params
    # Depending if the data is already exported or not read
    export_path = Path(dataset_paths.export_path)
    samples_dir = Path(dataset_paths.samples_dir)

    logger.info("Exporting...")
    # Read the metadata and create one dataframe with the image paths,
    # and the second one with selected dates
    mmdc_df = extract_paths(samples_dir, export_path, input_tile_list)
    sel_dates_all = extract_dates(
        mmdc_df, export_path, days_selection, s1_asc_desc_strict, prefilter_clouds
    )

    logger.info("df : %s", len(sel_dates_all))
    logger.info("df.cols %s", sel_dates_all.columns)

    if only_csv:  # We only produce csv files with dates and paths
        sys.exit()

    # create tile list if is necessary
    input_tile_list: list[str] | None = None
    if input_tile_list is not None:
        input_tile_list = [f"{samples_dir}/{i}" for i in input_tile_list]
        logging.info(
            "Limit the serialized files to the following list: %s",
            input_tile_list,
        )
    store_data(
        path_df=mmdc_df,
        date_df=sel_dates_all,
        export_dir=export_path,
        patch_config=(patch_size, patch_margin),
        threshold=threshold,
    )

    logger.info("Export finished")
