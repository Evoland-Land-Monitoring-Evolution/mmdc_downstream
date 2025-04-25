import argparse
import logging
import os
from collections import namedtuple
from pathlib import Path

from mmdc_downstream_biomass.tile_processing.mlp_processing import MLPBiomass

MLP_params = namedtuple("MLP_params", "n_iter bs lr")

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
        # default=f"{WORK_FOLDER}/results/TCD/t32tnt/encoded/"
        default=f"{WORK_FOLDER}/results/BioMass/encoded/Sweden"
        # default=f"{WORK_FOLDER}/results/TCD/t32tnt/encoded/"
        # required=True
    )

    arg_parser.add_argument(
        "--model",
        type=str,
        help="Name of encoded model, only if we work with encoded data",
        # default="malice_aux-wr1-winv1-wcr0_f64_seed0_same_mod_epoch=121",
        # default="malice_wr1-winv1-wcr0_f64_seed0_bn_transp_no_same_mod"
        default="anysat",
    )

    arg_parser.add_argument(
        "--n_iter",
        type=int,
        help="number of iterations for bootstrap",
        default=300
        # required=False,
    )

    arg_parser.add_argument(
        "--bs",
        type=int,
        help="number of iterations for bootstrap",
        default=50
        # required=False,
    )

    arg_parser.add_argument(
        "--lr",
        type=int,
        help="number of iterations for bootstrap",
        default=0.0001
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
        "--weights",
        type=bool,
        help="apply weights during training",
        default=True
        # required=False,
    )

    arg_parser.add_argument(
        "--path_predict_new_image",
        type=str,
        help="Path to image tiles we are going to predict, if None no prediction",
        # default=None
        # default=f"{WORK_FOLDER}/results/TCD/t32tnt/pure_values/S1/"
        default=f"{WORK_FOLDER}"
        # f"/results/TCD/t32tnt/encoded/
        # malice_aux-wr1-winv1-wcr0_f64_seed0_same_mod_epoch=121"
        f"/results/BioMass/encoded/Sweden/anysat"
        # required=False,
    )

    arg_parser.add_argument(
        "--type", type=str, help="TCD or BioMass", default="BioMass"
    )

    return arg_parser


if __name__ == "__main__":
    # Parser arguments
    parser = get_parser()
    args = parser.parse_args()

    log.info(args.satellites)

    final_folder = (
        f"encoded_ep_{args.n_iter}_bs_{args.bs}_lr_{args.lr}_{args.model}"
        if args.encoded_data
        else f"pure_iter_{args.n_iter}"
    )
    if args.weights:
        final_folder += "_weights"
    folder = os.path.join(args.folder_data, "results", final_folder)

    mlp_params = MLP_params(args.n_iter, args.bs, args.lr)
    print(args.model)
    for sat in args.satellites:
        csv_folder = (
            os.path.join(args.folder_data, args.model, sat)
            if (args.model is not None and args.encoded_data)
            else args.folder_data
        )
        mlp = MLPBiomass(
            sat,
            csv_folder,
            folder,
            mlp_params,
            args.encoded_data,
            args.months_median,
            args.use_logvar,
            args.weights,
        )
        model_mlp, month_median_feat = mlp.process_satellites(variable="Height")
        if args.path_predict_new_image is not None:
            mlp.predict_new_image(
                path_image_tiles=os.path.join(args.path_predict_new_image, sat),
                model=model_mlp,
                model_name=f"ep_{args.n_iter}_bs_{args.bs}_lr_{args.lr}_{args.model}",
                variable="MeanHeight",
            )
        cat = MLPBiomass(
            sat,
            csv_folder,
            folder,
            mlp_params,
            args.encoded_data,
            args.months_median,
            args.use_logvar,
            args.weights,
        )
        model_cat, month_median_feat = cat.process_satellites(variable="AGB")
        # if args.path_predict_new_image is not None:
        #     cat.predict_new_image(
        #         path_image_tiles=os.path.join(args.path_predict_new_image, sat),
        #         model=model_cat,
        #         model_name=f"{args.n_iter}_{args.leaf}_{args.depth}",
        #         month_median_feat=month_median_feat,
        #         variable="BioMass"
        #     )
