#!/usr/bin/env python3
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
import argparse
import logging
import os
from pathlib import Path

from mmdc_downstream_pastis.encode_series.encode_pvae import encode_series_pvae

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
        default=f"{SCRATCH_FOLDER}/scratch_data/Pastis_OE_corr"
        # required=True
    )

    arg_parser.add_argument(
        "--dataset_path_pastis",
        type=str,
        help="Original Pastis data folder",
        default=f"{SCRATCH_FOLDER}/scratch_data/Pastis"
        # required=True
    )

    arg_parser.add_argument(
        "--sats",
        type=list[str],
        help="Chosen modalities",
        default=["S2"]
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
        "--path_jit_model",
        type=str,
        help="path to pvae model",
        default=f"{WORK_FOLDER}/results/MMDC/pvae.torchscript",
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

    output_path = os.path.join(args.output_path, "pvae")

    Path(os.path.join(output_path, "S2")).mkdir(exist_ok=True, parents=True)

    encode_series_pvae(
        args.path_jit_model,
        args.dataset_path_oe,
        args.dataset_path_pastis,
        args.sats,
        output_path,
    )
