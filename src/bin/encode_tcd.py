#!/usr/bin/env python3
# copyright: (c) 2023 cesbio / centre national d'Etudes Spatiales
"""
This module computes different bio-physical variables for MMDC dataset
"""

import argparse
import logging
import os

from mmdc_downstream_tcd.tile_processing.encode_tile import encode_tile

my_logger = logging.getLogger(__name__)


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
        default="/work/scratch/data/kalinie/TCD/OpenEO_data/t32tnt"
        # required=True
    )

    arg_parser.add_argument(
        "--output_path",
        type=str,
        help="output folder",
        default=""
        # required=True
    )

    arg_parser.add_argument(
        "--geojson_path",
        type=str,
        help="path to geojson file",
        default="/home/kalinichevae/Documents/TCD/tile_TCD_t32tnt.geojson"
        # required=True
    )

    arg_parser.add_argument(
        "--months_folders",
        type=str,
        nargs="+",
        help="list of available folders with months",
        default=["3", "45", "6", "7", "9", "10"]
        # default=["9", "10"]
        # required=False,
    )

    arg_parser.add_argument(
        "--window",
        type=int,
        help="size of sliding window",
        default=100
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

    encode_tile(args.folder_data, args.months_folders, args.geojson_path, args.window)
