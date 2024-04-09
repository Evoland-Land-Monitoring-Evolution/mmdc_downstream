#!/usr/bin/env python3
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
import argparse
import logging
import os

from mmdc_downstream_pastis.encode_series.encode import encode_series
from mmdc_downstream_pastis.mmdc_model.model import PretrainedMMDCPastis

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
        "--dataset_path_oe",
        type=str,
        help="OE data folder",
        default=f"{os.environ['WORK']}/scratch_data/PASTIS_OE"
        # required=True
    )

    arg_parser.add_argument(
        "--dataset_path_pastis",
        type=str,
        help="Original Pastis data folder",
        default=f"{os.environ['WORK']}/scratch_data/PASTIS"
        # required=True
    )

    arg_parser.add_argument(
        "--sats",
        type=list[str],
        help="Chosen modalities",
        default=["S2", "S1_ASC", "S1_DESC"]
        # required=True
    )

    # arg_parser.add_argument(
    #     "--output_path",
    #     type=str,
    #     help="output folder",
    #     default="/work/CESBIO/projects/DeepChange/Ekaterina/MMDC_OE"
    #     # required=True
    # )

    arg_parser.add_argument(
        "--pretrained_path",
        type=str,
        help="list of available tiles",
        default=f"{os.environ['WORK']}/results/MMDC/results/latent/checkpoints/"
        "mmdc_full_tiny/2024-04-05_14-58-22"
        # required=False,
    )

    arg_parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the pretrained model (last or epoch_XXX)",
        default="epoch_129"
        # required=False,
    )

    arg_parser.add_argument(
        "--model_type",
        type=str,
        help="Type of the model (baseline or experts)",
        default="baseline"
        # required=False,
    )

    arg_parser.add_argument(
        "--variables",
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

    mmdc_model = PretrainedMMDCPastis(
        args.pretrained_path, args.model_name, args.model_type
    )

    encode_series(mmdc_model, args.dataset_path_oe, args.dataset_path_pastis, args.sats)
