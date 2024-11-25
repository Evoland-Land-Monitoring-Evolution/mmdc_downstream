#!/usr/bin/env python3
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
import argparse
import logging
import os
from pathlib import Path

from encode_alise_s1_original_pl import encode_series_alise_s1

# from src.mmdc_downstream_pastis.encode_series.encode_alise_s1_original_onnx \
#     import encode_series_alise_onnx


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
    Encode Pastis with pVAE
    """
    arg_parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description="Encode Pastis with pVAE",
    )

    arg_parser.add_argument(
        "--loglevel",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logger level (default: INFO. Should be one of "
        "(DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    arg_parser.add_argument(
        "--dataset_path_oe",
        type=str,
        help="OE data folder",
        default=f"{SCRATCH_FOLDER}/Pastis_OE_Malice"
        # required=True
    )

    arg_parser.add_argument(
        "--dataset_path_pastis",
        type=str,
        help="Original Pastis data folder",
        default=f"{SCRATCH_FOLDER}/Pastis"
        # required=True
    )

    arg_parser.add_argument(
        "--sat",
        type=str,
        help="Chosen modalities",
        default="S1_ASC"
        # required=True
    )

    arg_parser.add_argument(
        "--output_path",
        type=str,
        help="output folder",
        default=f"{WORK_FOLDER}/results/Pastis_encoded"
        # required=True
    )

    arg_parser.add_argument(
        "--path_alise_model",
        type=str,
        help="path to alise model",
        # default=f"{WORK_FOLDER}/results/alise_preentrained/
        # malice_for_katya/malice_flexible_s1_wrec1_winv1_wcr0_seed3.onnx",
        default=f"{WORK_FOLDER}/results/alise_preentrained/all_checkpoints_malice/"
        f"malice-wr1-winv1-wcr0_seed4.ckpt",
    )

    arg_parser.add_argument(
        "--path_to_csv",
        type=str,
        help="path to csv",
        default=f"{WORK_FOLDER}/results/alise_preentrained",
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

    output_path = os.path.join(args.output_path, "alise")

    postfix = args.path_alise_model.split("/")[-1][19:-5]
    print(postfix)
    postfix = "good_log_ratio_bands"

    Path(os.path.join(output_path, args.sat + "_" + postfix)).mkdir(
        exist_ok=True, parents=True
    )

    encode_series_alise_s1(
        args.path_alise_model,
        args.path_to_csv,
        args.dataset_path_oe,
        args.dataset_path_pastis,
        args.sat,
        output_path,
        postfix,
    )
