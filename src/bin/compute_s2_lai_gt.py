#!/usr/bin/env python3
# copyright: (c) 2023 cesbio / centre national d'Etudes Spatiales
"""
This module computes different bio-physical variables for MMDC dataset
"""

import argparse
import logging
import os
from collections import namedtuple
from pathlib import Path

from src.mmdc_downstream.snap.components.compute_bio_var import compute_variables

my_logger = logging.getLogger(__name__)

DatasetPaths = namedtuple("DatasetPaths", ["input_path", "output_path"])


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
        "--input_path",
        type=str,
        help="input folder",
        default="/work/CESBIO/projects/DeepChange/Ekaterina/MMDC_OE"
        # required=True
    )

    arg_parser.add_argument(
        "--output_path",
        type=str,
        help="output folder",
        default="/work/CESBIO/projects/DeepChange/Ekaterina/MMDC_OE"
        # required=True
    )

    arg_parser.add_argument(
        "--input_tile_list",
        type=str,
        nargs="+",
        help="list of available tiles",
        default=["30TXT"]
        # default=["30TXT", "31TEN", "32UMB", "33UXR", "34UEC", "34TFR"]
        # required=False,
    )

    arg_parser.add_argument("--variables",
                            type=str,
                            nargs="+",
                            help="list of variables to compute",
                            default=["lai"]
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

    if not os.path.isdir(args.output_path):
        logging.info("The directory is not present. Creating a new one.. %s",
                     args.output_path)
        Path(os.path.join(args.output_path)).mkdir()

    logging.info("Starting the computation")

    # creating dataset
    compute_variables(
        DatasetPaths(input_path=args.input_path, output_path=args.output_path),
        args.input_tile_list, args.variables)
