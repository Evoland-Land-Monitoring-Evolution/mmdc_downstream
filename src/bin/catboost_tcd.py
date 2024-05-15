import argparse
import logging
import os
from collections import namedtuple

from mmdc_downstream_tcd.catboost.processing import CatBoostTCD

CB_params = namedtuple("CB_params", "n_iter leaf depth")

log = logging.getLogger(__name__)


def get_parser() -> argparse.ArgumentParser:
    """
    Generate argument parser for Catboost
    """
    arg_parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description="TCD Catboost",
    )

    arg_parser.add_argument(
        "--folder_data",
        type=str,
        help="input folder",
        # default="/home/kalinichevae/jeanzay/results/TCD/t32tnt/pure_values/s1"
        # default=f"{os.environ['WORK']}/results/TCD/t32tnt/pure_values/s2"
        # "/home/kalinichevae/jeanzay/results/TCD/t32tnt/pure_values/s2"
        # default="/home/kalinichevae/jeanzay/results/TCD/t32tnt/encoded"
        default=f"{os.environ['WORK']}/results/TCD/t32tnt/encoded"
        # default=f"{os.environ['WORK']}/results/TCD/t32tnt/pure_values/s1"
        # required=True
    )

    arg_parser.add_argument(
        "--model",
        type=str,
        help="Name of encoded model, only if we work with encoded data",
        default="res_checkpoint_best",
    )

    arg_parser.add_argument(
        "--n_iter",
        type=int,
        help="number of iterations for bootstrap",
        default=150
        # required=False,
    )

    arg_parser.add_argument(
        "--leaf",
        type=int,
        help="leaf parameter for bootstrap",
        default=17
        # required=False,
    )

    arg_parser.add_argument(
        "--depth",
        type=int,
        help="number of features per timestamp",
        default=7
        # required=False,
    )

    arg_parser.add_argument(
        "--months_median",
        type=bool,
        help="if we compute median per months or not",
        default=False
        # required=False,
    )

    arg_parser.add_argument(
        "--encoded_data",
        type=bool,
        help="if we work with encoded data or raw one",
        default=True
        # required=False,
    )

    arg_parser.add_argument(
        "--satellites",
        type=list[str],
        help="satellites we deal with",
        default=["s1", "s2"]
        # required=False,
    )

    arg_parser.add_argument(
        "--path_predict_new_image",
        type=str,
        help="Path to image tiles we are going to predict, if None no prediction",
        # default=None
        # default=f"{os.environ['WORK']}/results/TCD/t32tnt/pure_values/S2_full_median/"
        # default=
        # f"/home/kalinichevae/jeanzay/results/TCD/t32tnt/encoded/res_checkpoint_best"
        default=f"{os.environ['WORK']}/results/TCD/t32tnt/encoded/res_checkpoint_best"
        # required=False,
    )

    return arg_parser


if __name__ == "__main__":
    # Parser arguments
    parser = get_parser()
    args = parser.parse_args()

    csv_folder = (
        os.path.join(args.folder_data, args.model)
        if (args.model is not None and args.encoded_data)
        else args.folder_data
    )

    final_folder = (
        f"encoded_d_{args.depth}_l_{args.leaf}_iter_{args.n_iter}_{args.model}"
        if args.encoded_data
        else f"pure_d_{args.depth}_leaf_{args.leaf}_iter_{args.n_iter}"
    )
    folder = os.path.join(args.folder_data, "results", final_folder)

    cb_params = CB_params(args.n_iter, args.leaf, args.depth)

    for sat in args.satellites:
        cat = CatBoostTCD(
            sat, csv_folder, folder, cb_params, args.encoded_data, args.months_median
        )
        model_cat, month_median_feat = cat.process_satellites()

        if args.path_predict_new_image is not None:
            cat.predict_new_image(
                path_image_tiles=args.path_predict_new_image,
                model=model_cat,
                model_name=f"{args.n_iter}_{args.leaf}_{args.depth}",
                month_median_feat=month_median_feat,
            )
