#!/usr/bin/env python3
# copyright: (c) 2024 cesbio / centre national d'Etudes Spatiales
"""
This module computes different bio-physical variables for MMDC dataset
"""

import argparse
import logging
import os
from pathlib import Path

from mmdc_downstream_tcd.tile_processing.prepare_tile import prepare_tiles

my_logger = logging.getLogger(__name__)


if not Path("/work/scratch").exists():
    SCRATCH_FOLDER = (
        f"{os.environ['SCRATCH']}/scratch_data/results"
        if "SCRATCH" in os.environ
        else f"{os.environ['HOME']}/scratch_jeanzay/scratch_data/results"
    )
    WORK_FOLDER = (
        f'{os.environ["WORK"]}/results'
        if "WORK" in os.environ
        else f"{os.environ['HOME']}/jeanzay/results"
    )
else:
    SCRATCH_FOLDER = "/work/CESBIO/projects/DeepChange/Ekaterina"
    WORK_FOLDER = "/work/scratch/data/kalinie"


def get_parser() -> argparse.ArgumentParser:
    """
    Generate argument parser for CLI
    """
    arg_parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description="Serialize as torch tensors the data generated for the"
        " MMDC-DataCollection project",
    )

    arg_parser.add_argument(
        "--loglevel",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logger level (default: INFO. Should be one of "
        "(DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    arg_parser.add_argument(
        "--folder_data",
        type=str,
        help="input folder",
        # default="/work/CESBIO/projects/DeepChange/Ekaterina/TCD/OpenEO_data/t32tnt/",
        default=f"{SCRATCH_FOLDER}/TCD/OpenEO_data/t32tnt/"
        # required=True
    )

    arg_parser.add_argument(
        "--output_path",
        type=str,
        help="output folder",
        default=f"{SCRATCH_FOLDER}/TCD/t32tnt"
        # required=True
    )

    arg_parser.add_argument(
        "--months_folders",
        type=str,
        help="list of available folders with months",
        default=["3", "4", "5", "6", "7", "8", "9", "10"]
        # required=False,
    )

    arg_parser.add_argument(
        "--window",
        type=int,
        help="size of sliding window",
        default=768
        # required=False,
    )

    arg_parser.add_argument(
        "--margin",
        type=int,
        help="size of margin",
        default=40
        # required=False,
    )

    arg_parser.add_argument(
        "--satellites",
        type=int,
        help="satellites to prepare",
        default=["s2"]
        # required=False,
    )

    return arg_parser


if __name__ == "__main__":
    # Parser arguments
    parser = get_parser()
    args = parser.parse_args()

    # Configure logging
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {args.loglevel}")

    logging.basicConfig(
        level=numeric_level,
        datefmt="%y-%m-%d %H:%M:%S",
        format="%(asctime)s :: %(levelname)s :: %(message)s",
    )

    output_path = os.path.join(
        args.output_path, f"prepared_tiles_w_{args.window}_m_{args.margin}"
    )

    for sat in args.satellites:
        Path(os.path.join(output_path, sat)).mkdir(parents=True, exist_ok=True)

    prepare_tiles(
        args.folder_data,
        args.months_folders,
        args.window,
        output_path,
        args.margin,
        args.satellites,
    )
