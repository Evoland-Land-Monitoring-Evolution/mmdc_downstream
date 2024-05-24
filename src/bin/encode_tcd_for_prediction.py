#!/usr/bin/env python3
# copyright: (c) 2024 cesbio / centre national d'Etudes Spatiales
"""
This module computes different bio-physical variables for MMDC dataset
"""

import argparse
import logging
import os
from pathlib import Path

from mmdc_downstream_pastis.mmdc_model.model import PretrainedMMDCPastis
from mmdc_downstream_tcd.tile_processing.encode_whole_tile_for_prediction import (
    encode_tile,
)

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
        # default="/work/CESBIO/projects/DeepChange/Ekaterina/TCD/OpenEO_data/t32tnt/",
        default=f"{os.environ['SCRATCH']}/scratch_data/TCD/OpenEO_data/t32tnt/"
        # required=True
    )

    arg_parser.add_argument(
        "--output_path",
        type=str,
        help="output folder",
        # default="/work/scratch/data/kalinie/TCD/OpenEO_data/t32tnt/encoded"
        default=f"{os.environ['WORK']}/results/TCD/t32tnt/encoded"
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
        "--pretrained_path",
        type=str,
        help="list of available tiles",
        default=f"{os.environ['WORK']}/results/MMDC/checkpoint_best",
    )

    arg_parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the pretrained model (last or epoch_XXX)",
        default="epoch_141"
        # required=False,
    )

    arg_parser.add_argument(
        "--model_type",
        type=str,
        help="Type of the model (baseline or experts)",
        default="baseline"
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

    mmdc_model = PretrainedMMDCPastis(
        args.pretrained_path, args.model_name, args.model_type
    )

    output_path = os.path.join(
        args.output_path, "res_" + args.pretrained_path.split("/")[-1]
    )
    Path(output_path).mkdir(parents=True, exist_ok=True)

    encode_tile(
        args.folder_data,
        args.months_folders,
        args.window,
        output_path,
        mmdc_model,
        gt_file=None,
        s1_join=True,
    )
