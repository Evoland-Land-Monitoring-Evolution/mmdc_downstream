import argparse
import logging
import os
from collections import namedtuple
from pathlib import Path

from mmdc_downstream_tcd.catboost.processing import CatBoostTCD
from src.mmdc_downstream_biomass.tile_processing.catboost_biomass import CatBoostBioMass

CB_params = namedtuple("CB_params", "n_iter leaf depth")

log = logging.getLogger(__name__)

if not Path("/work/scratch").exists():
    SCRATCH_FOLDER = (
        f"{os.environ['SCRATCH']}/scratch_data"
        if "SCRATCH" in os.environ
        else f"{os.environ['HOME']}/scratch_jeanzay/scratch_data"
    )
    WORK_FOLDER = (
        f'{os.environ["WORK"]}/results'
        if "WORK" in os.environ
        else f"{os.environ['HOME']}/jeanzay/results"
    )
else:
    SCRATCH_FOLDER = "/work/CESBIO/projects/DeepChange/Ekaterina/"
    WORK_FOLDER = "/work/scratch/data/kalinie"


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
        # default=f"{WORK_FOLDER}/results/TCD/t32tnt/pure_values/s2"
        # default=f"{WORK_FOLDER}/results/BioMass/encoded/Sweden"
        default=f"{WORK_FOLDER}/results/TCD/t32tnt/encoded/"
        # required=True
    )

    arg_parser.add_argument(
        "--model",
        type=str,
        help="Name of encoded model, only if we work with encoded data",
        # default="malice_aux-wr1-winv1-wcr0_f64_seed0_same_mod_epoch=121",
        default="malice_wr1-winv1-wcr0_f64_seed0_same_mod_epoch=142"
        # default="anysat"
    )

    arg_parser.add_argument(
        "--n_iter",
        type=int,
        help="number of iterations for bootstrap",
        default=1000
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
        default=["s1_asc"]
        # required=False,
    )

    arg_parser.add_argument(
        "--use_logvar",
        type=list[str],
        help="satellites we deal with",
        default=False
        # required=False,
    )

    arg_parser.add_argument(
        "--path_predict_new_image",
        type=str,
        help="Path to image tiles we are going to predict, if None no prediction",
        # default=None
        # default=f"{WORK_FOLDER}/results/TCD/t32tnt/pure_values/S1/"
        default=f"{WORK_FOLDER}/" f"/results/TCD/t32tnt/encoded/"
        # malice_aux-wr1-winv1-wcr0_f64_seed0_same_mod_epoch=121"
        f"malice_wr1-winv1-wcr0_f64_seed0_same_mod_epoch=142",
    )

    arg_parser.add_argument("--type", type=str, help="TCD or BioMass", default="TCD")

    return arg_parser


if __name__ == "__main__":
    # Parser arguments
    parser = get_parser()
    args = parser.parse_args()

    log.info(args.satellites)

    final_folder = (
        f"encoded_d_{args.depth}_l_{args.leaf}_iter_{args.n_iter}_{args.model}"
        if args.encoded_data
        else f"pure_d_{args.depth}_leaf_{args.leaf}_iter_{args.n_iter}"
    )
    folder = os.path.join(args.folder_data, "results", final_folder)

    cb_params = CB_params(args.n_iter, args.leaf, args.depth)
    print(args.model)
    for sat in args.satellites:
        csv_folder = (
            os.path.join(args.folder_data, args.model, sat)
            if (args.model is not None and args.encoded_data)
            else args.folder_data
        )
        if args.type == "TCD":
            cat = CatBoostTCD(
                sat,
                csv_folder,
                folder,
                cb_params,
                args.encoded_data,
                args.months_median,
                args.use_logvar,
            )
            model_cat, month_median_feat = cat.process_satellites()
            if args.path_predict_new_image is not None:
                cat.predict_new_image(
                    path_image_tiles=os.path.join(args.path_predict_new_image, sat),
                    model=model_cat,
                    model_name=f"{args.n_iter}_{args.leaf}_{args.depth}",
                    month_median_feat=month_median_feat,
                )
        else:
            cat = CatBoostBioMass(
                sat,
                csv_folder,
                folder,
                cb_params,
                args.encoded_data,
                args.months_median,
                args.use_logvar,
            )
            model_cat, month_median_feat = cat.process_satellites(variable="Height")
            # if args.path_predict_new_image is not None:
            #     cat.predict_new_image(
            #         path_image_tiles=os.path.join(args.path_predict_new_image, sat),
            #         model=model_cat,
            #         model_name=f"{args.n_iter}_{args.leaf}_{args.depth}",
            #         month_median_feat=month_median_feat,
            #         variable="MeanHeight"
            #     )
            model_cat, month_median_feat = cat.process_satellites(variable="AGB")
            # if args.path_predict_new_image is not None:
            #     cat.predict_new_image(
            #         path_image_tiles=os.path.join(args.path_predict_new_image, sat),
            #         model=model_cat,
            #         model_name=f"{args.n_iter}_{args.leaf}_{args.depth}",
            #         month_median_feat=month_median_feat,
            #         variable="BioMass"
            #     )
