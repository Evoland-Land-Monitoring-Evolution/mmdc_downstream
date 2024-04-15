import argparse
import logging
import os
from collections import namedtuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

MAIN_COLS = ["index", "TCD", "x", "y", "x_round", "y_round"]

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
        default="/home/kalinichevae/jeanzay/results/TCD/t32tnt/pure_values/s1"
        # default="/home/kalinichevae/jeanzay/results/TCD/t32tnt/encoded"
        # default=f"{os.environ['WORK']}/results/TCD/t32tnt/pure_values/s1"
        # required=True
    )

    arg_parser.add_argument(
        "--model",
        type=str,
        help="Name of encoded model, only if we work with encoded data",
        # default="res_2024-04-05_14-58-22"
        default=None,
    )

    arg_parser.add_argument(
        "--n_iter",
        type=int,
        help="number of iterations for bootstrap",
        default=1500
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
        default=12
        # required=False,
    )

    arg_parser.add_argument(
        "--months_median",
        type=bool,
        help="if we compute median per months or not",
        default=True
        # required=False,
    )

    arg_parser.add_argument(
        "--encoded_data",
        type=bool,
        help="if we work with encoded data or raw one",
        default=False
        # required=False,
    )

    arg_parser.add_argument(
        "--satellites",
        type=list[str],
        help="satellites we deal with",
        default=["s1"]
        # required=False,
    )

    return arg_parser


def open_csv_files_and_combine(
    csv_folder: Path | str, sat: str
) -> tuple[pd.DataFrame, np.array]:
    """
    Open csv files saved per patch and combine them in one dataframe.
    """
    all_df = []

    if (
        len(
            [
                file
                for file in np.asarray(os.listdir(csv_folder))
                if file.endswith("_all.csv")
            ]
        )
        > 0
    ):
        file = [
            file
            for file in np.asarray(os.listdir(csv_folder))
            if file.endswith("_all.csv")
        ][0]
        data = pd.read_csv(os.path.join(csv_folder, file))

    else:
        csv_files = [
            file
            for file in np.asarray(os.listdir(csv_folder))
            if file.endswith("csv") and sat in file and not file.endswith("_all.csv")
        ]
        for csv_name in np.sort(csv_files):
            data = pd.read_csv(os.path.join(csv_folder, csv_name))
            all_df.append(data)

        data = pd.concat(all_df, ignore_index=True)
    log.info(f"Shape of input data {data.shape}")

    days = np.load(os.path.join(csv_folder, f"gt_tcd_t32tnt_days_{sat}.npy"))

    return data, days


def get_month_median(
    data: pd.DataFrame,
    days: np.ndarray,
    encoded: bool = True,
    bands: list | None = None,
) -> tuple[pd.DataFrame, np.array]:
    """Compute a months median for each feature and create a new df"""

    median_dict = {}

    months_dates = pd.DatetimeIndex(days).to_frame(name="dates").dates.dt.month.values
    dates_ind = np.arange(len(months_dates))

    for month in np.unique(months_dates):
        days = dates_ind[months_dates == month]

        if encoded:
            data_cols_month = [
                c for day in days for c in data.columns if c.endswith(f"d{day}")
            ]
        else:
            data_cols_month = [
                c for day in days for c in data.columns if c.endswith(f"_{day}")
            ]

        if encoded:
            for value in ("mu", "logvar"):
                feat_nb = int(
                    len([c for c in data_cols_month if value in c]) / len(days)
                )
                for f in range(feat_nb):
                    data_cols_feat = [
                        c for c in data_cols_month if f"{value}_f{f}" in c
                    ]
                    median = np.nanmedian(data[data_cols_feat].values, axis=1)
                    median_dict[f"{value}_f{f}_m{month}"] = median
        else:
            for value in bands:
                data_cols_feat = [c for c in data_cols_month if value in c]
                median = np.nanmedian(data[data_cols_feat].values, axis=1)
                median_dict[f"{value}_m{month}"] = median

    data = pd.concat([data[MAIN_COLS], pd.DataFrame.from_dict(median_dict)], axis=1)
    data_cols = np.asarray(median_dict.keys())
    # if encoded:
    #     assert data_cols.size * 2 * feat_nb
    # else:

    log.info(f"Shape of meadian input data {data.shape}")
    print(f"Shape of meadian input data {data.shape}")
    return data, data_cols


def catboost_algorothm(
    cb_params: CB_params, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
) -> np.ndarray:
    """Create model and make a prediction"""

    log.info("Getting model")

    model_cat = CatBoostRegressor(
        iterations=cb_params.n_iter,
        l2_leaf_reg=cb_params.leaf,
        depth=cb_params.depth,
    )

    log.info("Fitting model")
    model_cat.fit(X_train, y_train, verbose=True)

    log.info("Predicting")
    pred = model_cat.predict(X_test).clip(0, 100)

    log.info("Predicted")
    log.info(np.round(pred))
    log.info("GT")
    log.info(y_test.flatten())
    return pred


def results_scatterplot(
    pred: np.array, y_test: np.array, rmse: float, folder: str, name: str
) -> None:
    """Make a scatterplot prediction vs reality"""

    plt.scatter(pred, y_test, s=1, c="red", alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])
    plt.title(f"RMSE={rmse}%")
    plt.xlabel("pred")
    plt.ylabel("gt")
    Path(folder).mkdir(exist_ok=True, parents=True)
    plt.savefig(os.path.join(folder, name))
    plt.close()


if __name__ == "__main__":
    # Parser arguments
    parser = get_parser()
    args = parser.parse_args()

    csv_folder = (
        os.path.join(args.folder_data, args.model)
        if (args.model is not None and args.encoded_data)
        else args.folder_data
    )

    for sat in args.satellites:
        data, days = open_csv_files_and_combine(csv_folder, sat)

        if args.encoded_data:
            data_cols = [c for c in data.columns if ("mu" in c) or ("logvar" in c)]
            if args.months_median:
                data, data_cols = get_month_median(data, days)
        else:
            if sat == "s2":
                bands = ["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"]
                data_cols = [c for c in data.columns if any(b in c for b in bands)]
            else:  # sat s1
                bands = [
                    "VV_asc",
                    "VH_asc",
                    "ratio_asc",
                    "VV_desc",
                    "VH_desc",
                    "ratio_desc",
                    "ang_asc",
                    "ang_desc",
                ]
                # BANDS_S1 = ["VV_asc", "VH_asc", "ratio_asc",
                #             "VV_desc", "VH_desc", "ratio_desc"]
                data_cols = [c for c in data.columns if any(b in c for b in bands)]
            if args.months_median:
                data, data_cols = get_month_median(
                    data, days, encoded=False, bands=bands
                )

        gt_cols = ["TCD"]

        log.info("splitting data")
        X_train, X_test, y_train, y_test = train_test_split(
            data[data_cols].values, data[gt_cols].values, test_size=0.3, random_state=42
        )

        # Create model, fit and predict
        pred = catboost_algorothm(
            CB_params(args.n_iter, args.leaf, args.depth),
            X_train,
            y_train,
            X_test,
        )

        # Compute error
        rmse = round(
            mean_squared_error(y_test.flatten(), np.round(pred), squared=False), 2
        )
        log.info(f"RMSE={rmse}")
        print(f"RMSE={rmse}")
        print(np.round(pred))
        print(y_test.flatten())

        # Get scatterplot of the result
        final_folder = (
            f"encoded_d_{args.depth}_l_{args.leaf}_iter_{args.n_iter}_{args.model}"
            if args.encoded_data
            else f"pure_d_{args.depth}_leaf_{args.leaf}_iter_{args.n_iter}"
        )
        if args.months_median:
            final_folder += "_median"
        img_name = f"{final_folder}_{sat}.png"
        folder = os.path.join(args.folder_data, "results", final_folder)
        Path(folder).mkdir(exist_ok=True)
        results_scatterplot(np.round(pred), y_test.flatten(), rmse, folder, img_name)
