#!/usr/bin/env python3
# copyright: (c) 2024 cesbio / centre national d'Etudes Spatiales
"""
This module computes different bio-physical variables for MMDC dataset
"""

import argparse
import logging
import os
from pathlib import Path

from mmdc_downstream_tcd.tile_processing.encode_tile_alise_s1 import (
    encode_tile_alise_s1,
)
from mmdc_downstream_tcd.tile_processing.encode_tile_malice_aux import (
    encode_tile_malice_aux,
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
        default=f"{WORK_FOLDER}/results/TCD/t32tnt/encoded"
        # required=True
    )

    arg_parser.add_argument(
        "--gt_path",
        type=str,
        help="path to geojson ground truth file",
        default=f"{SCRATCH_FOLDER}/TCD/SAMPLES_t32tnt_32632.geojson",  # noqa: E501
        # required=True
    )

    arg_parser.add_argument(
        "--path_alise_model",
        type=str,
        help="path to pre-entrained model",
        # default=f"{WORK_FOLDER}/results/alise_preentrained/
        # malice_for_katya/malice_flexible_s1_wrec1_winv1_wcr0_seed3.onnx",
        default=f"{WORK_FOLDER}/results/alise_preentrained/my_checkpoints_malice_aux/"
        f"malice-aux-wr1-winv1-wcr0_f64_seed0_same_mod_epoch=121.ckpt",
    )

    arg_parser.add_argument(
        "--path_csv_norm",
        type=str,
        help="path to pre-entrained model",
        # default=f"{WORK_FOLDER}/results/alise_preentrained",
        default=f"{SCRATCH_FOLDER}/MMDC_MALICE",
    )

    arg_parser.add_argument(
        "--prepared_data_path",
        type=str,
        help="Path to prepared data",
        # default=f"{SCRATCH_FOLDER}/TCD/t32tnt/prepared_tiles_malice_w_768_m_40"
        default=f"{SCRATCH_FOLDER}/TCD/t32tnt/prepared_tiles_w_768_m_40"
        # required=False,
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
        "--satellites",
        type=int,
        help="modalities to encode",
        default=["s2"]
        # required=False,
    )

    arg_parser.add_argument(
        "--aux",
        type=bool,
        help="auxiliary data used",
        default=True,
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

    if args.aux:
        encode_tile_malice_aux(
            args.prepared_data_path,
            args.folder_data,
            args.path_alise_model,
            args.path_csv_norm,
            output_path,
            args.gt_path,
            args.satellites,
        )
    else:
        encode_tile_alise_s1(
            args.prepared_data_path,
            args.folder_data,
            args.path_alise_model,
            args.path_csv_norm,
            output_path,
            args.gt_path,
            args.satellites,
        )

    # if args.satellites[0] == "s2":
    #     output_path = Path(
    #         os.path.join(
    #             args.output_path,
    #             "malice_s2_" + args.path_alise_model.split("/")[-1].split(".")[0][7:],
    #         )
    #     )
    #     Path(output_path).mkdir(parents=True, exist_ok=True)
    #     print(output_path)
    #     from src.mmdc_downstream_tcd.tile_processing.encode_tile_alise import (
    #         encode_tile_alise,
    #     )
    #
    #     encode_tile_alise(
    #         args.prepared_data_path,
    #         args.path_alise_model,
    #         args.path_csv_norm,
    #         output_path,
    #         args.gt_path,
    #         args.satellites,
    #     )
    # else:
    #     output_path = Path(
    #         os.path.join(
    #             args.output_path,
    #             "malice_s1_" + args.path_alise_model.split("/")[-1].split(".")[0][7:],
    #         )
    #     )
    #     Path(output_path).mkdir(parents=True, exist_ok=True)
    #     print(output_path)
    #     from mmdc_downstream_tcd.tile_processing.encode_tile_alise_s1 import (
    #         encode_tile_alise_s1,
    #     )
    #
    #     encode_tile_alise_s1(
    #         args.prepared_data_path,
    #         args.folder_data,
    #         args.path_alise_model,
    #         args.path_csv_norm,
    #         output_path,
    #         args.gt_path,
    #         args.satellites,
    #     )
