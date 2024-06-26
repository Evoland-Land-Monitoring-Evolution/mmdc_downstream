#!/usr/bin/env python3
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
import argparse
import logging
import os
from pathlib import Path

from mmdc_downstream_pastis.encode_series.encode import encode_series
from mmdc_downstream_pastis.mmdc_model.model import PretrainedMMDCPastis

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
        default=["S1_ASC", "S1_DESC"]
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
        "--pretrained_path",
        type=str,
        help="list of available tiles",
        default=f"{WORK_FOLDER}/results/MMDC/checkpoint_best"
        # required=False,
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

    output_path = os.path.join(args.output_path, args.pretrained_path.split("/")[-1])
    if "S1_ASC" in args.sats and "S1_DESC" in args.sats:
        Path(os.path.join(output_path, "S1")).mkdir(exist_ok=True, parents=True)
    else:
        if "S1_ASC" in args.sats:
            Path(os.path.join(output_path, "S1_ASC")).mkdir(exist_ok=True, parents=True)
        elif "S1_DESC" in args.sats:
            Path(os.path.join(output_path, "S1_DESC")).mkdir(
                exist_ok=True, parents=True
            )
    if "S2" in args.sats:
        Path(os.path.join(output_path, "S2")).mkdir(exist_ok=True, parents=True)

    encode_series(
        mmdc_model,
        args.dataset_path_oe,
        args.dataset_path_pastis,
        args.sats,
        output_path,
    )
