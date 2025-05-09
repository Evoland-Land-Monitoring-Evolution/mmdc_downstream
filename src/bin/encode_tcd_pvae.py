#!/usr/bin/env python3
# copyright: (c) 2024 cesbio / centre national d'Etudes Spatiales
"""
This module computes different bio-physical variables for MMDC dataset
"""

import argparse
import logging
import os
from pathlib import Path

from mmdc_downstream_tcd.tile_processing.encode_tile_pvae import encode_tile_pvae

my_logger = logging.getLogger(__name__)


WORK_FOLDER = (
    os.environ["WORK"] if "WORK" in os.environ else f"{os.environ['HOME']}/jeanzay"
)
SCRATCH_FOLDER = (
    os.environ["SCRATCH"]
    if "SCRATCH" in os.environ
    else f"{os.environ['HOME']}/scratch_jeanzay"
)


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
        default=f"{SCRATCH_FOLDER}/scratch_data/TCD/OpenEO_data/t32tnt/"
        # required=True
    )

    arg_parser.add_argument(
        "--output_path",
        type=str,
        help="output folder",
        default=f"{WORK_FOLDER}/results/TCD/t32tnt/encoded"
        # required=True
    )

    arg_parser.add_argument(
        "--gt_path",
        type=str,
        help="path to geojson ground truth file",
        default=f"{SCRATCH_FOLDER}/scratch_data/TCD/SAMPLES_t32tnt_32632.geojson",  # noqa: E501
        # required=True
    )

    arg_parser.add_argument(
        "--path_jit_model",
        type=str,
        help="path to pvae model",
        default="./../../pretrained_models/pvae/pvae.torchscript",
    )

    arg_parser.add_argument(
        "--prepared_data_path",
        type=str,
        help="Path to prepared data",
        default=f"{SCRATCH_FOLDER}/results/TCD/t32tnt/prepared_tiles_w_768_m_40"
        # required=False,
    )

    arg_parser.add_argument(
        "--satellites",
        type=int,
        help="modalities to encode",
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

    output_path = Path(os.path.join(args.output_path, "pvae"))

    Path(output_path).mkdir(parents=True, exist_ok=True)
    print(output_path)

    encode_tile_pvae(
        args.folder_data,
        args.prepared_data_path,
        args.path_jit_model,
        output_path,
        args.gt_path,
        args.satellites,
    )
