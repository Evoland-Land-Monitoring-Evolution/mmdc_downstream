#!/usr/bin/env python3
# copyright: (c) 2024 cesbio / centre national d'Etudes Spatiales
"""
This module computes different bio-physical variables for MMDC dataset
"""

import argparse
import logging
import os
from pathlib import Path

from mmdc_downstream_biomass.tile_processing.encode_tile_biomass_malice import (
    encode_tile_biomass_malice,
)

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
        "--output_path",
        type=str,
        help="output folder",
        default=f"{WORK_FOLDER}/results/BioMass/encoded/Sweden"
        # required=True
    )

    arg_parser.add_argument(
        "--gt_path",
        type=str,
        help="path to geojson ground truth file",
        default=f"{SCRATCH_FOLDER}/BioMass/Sweden_NFI/SWE_NFIplotsAGB_2014_2021_class.gpkg",  # noqa: E501
        # required=True
    )

    arg_parser.add_argument(
        "--path_alise_model",
        type=str,
        help="path to pre-entrained model",
        default="./../../pretrained_models/malice/"
        "malice-wr1-winv1-wcr0_f64_seed0_same_mod_epoch=142.ckpt",
    )

    arg_parser.add_argument(
        "--path_csv_norm",
        type=str,
        help="path to pre-entrained model",
        default="./../../pretrained_models/malice",
    )

    arg_parser.add_argument(
        "--folder_data",
        type=str,
        help="input folder",
        default=f"{SCRATCH_FOLDER}/BioMass/images/Sweden_all_Sentinel1_ASCENDING"
        # required=True
    )

    arg_parser.add_argument(
        "--satellites",
        type=int,
        help="modalities to encode",
        default=["s1_asc"]
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

    output_path = Path(
        os.path.join(
            args.output_path,
            "malice_" + args.path_alise_model.split("/")[-1].split(".")[0][7:],
        )
    )
    Path(output_path).mkdir(parents=True, exist_ok=True)
    print(output_path)

    for sat in args.satellites:
        encode_tile_biomass_malice(
            args.folder_data,
            args.path_alise_model,
            args.path_csv_norm,
            output_path,
            args.gt_path,
            sat,
        )
