import logging
import os
import warnings
from collections import namedtuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from catboost import CatBoostRegressor
from einops import rearrange
from rasterio.transform import Affine
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from mmdc_downstream_tcd.tile_processing.utils import rearrange_ts

warnings.filterwarnings("ignore")

MAIN_COLS = ["index", "TCD", "x", "y", "x_round", "y_round"]

CB_params = namedtuple("CB_params", "n_iter leaf depth")

log = logging.getLogger(__name__)


class CatBoostTCD:
    def __init__(
        self,
        sat: str,
        csv_folder: str | Path,
        folder: str | Path,
        cb_params: CB_params,
        encoded: bool,
        months_median: bool,
        use_logvar: bool,
    ):
        self.sat = sat
        self.csv_folder = csv_folder
        self.folder = folder
        self.cb_params = cb_params
        self.encoded = encoded
        self.months_median = months_median

        self.data_cols = None

        self.clip_x = 519000

        self.use_logvar = use_logvar

    def predict_new_image(
        self,
        path_image_tiles: str | Path,
        model: CatBoostRegressor,
        model_name: str,
        month_median_feat: list[list] | None,
    ) -> None:
        tile_names = np.sort(
            [
                f
                for f in os.listdir(path_image_tiles)
                if (f.endswith(".pt") and f.__contains__(self.sat))
            ]
        )
        if self.sat == "s1":
            tile_names = np.sort(
                [
                    f
                    for f in tile_names
                    if not f.__contains__("s1_asc") or not f.__contains__("s1_desc")
                ]
            )
        pred_coords = []
        pred_tiles = []
        all_img = []
        x_min_glob, x_max_glob, y_min_glob, y_max_glob = (
            torch.inf,
            -torch.inf,
            torch.inf,
            -torch.inf,
        )

        for tt in range(len(tile_names)):
            tile = torch.load(os.path.join(path_image_tiles, tile_names[tt]))
            log.info(f"Predicting tile {tile_names[tt]}")
            matrix = tile["matrix"]
            if type(matrix) is dict:
                x_min, x_max, y_min, y_max = (
                    matrix["x_min"] - 5,
                    matrix["x_max"] + 5,
                    matrix["y_min"] - 5,
                    matrix["y_max"] + 5,
                )
            else:
                x_min, x_max, y_min, y_max = (
                    matrix[0].min() - 5,
                    matrix[0].max() + 5,
                    matrix[1].min() - 5,
                    matrix[1].max() + 5,
                )
            pred_coords.append([x_min, x_max, y_min, y_max])
            x_min_glob, x_max_glob, y_min_glob, y_max_glob = (
                min(x_min_glob, x_min),
                max(x_max_glob, x_max),
                min(y_min_glob, y_min),
                max(y_max_glob, y_max),
            )

            img = tile["img"]
            t, c, h, w = img.shape

            if not ("alise" in path_image_tiles or "malice" in path_image_tiles):
                mu = img[:, : int(c / 2)]
                if not self.use_logvar:
                    img = mu
                else:
                    logvar = img[:, int(c / 2) :]
                    img = np.concatenate([mu, logvar], 0)

            img = rearrange_ts(img)
            all_img.append(img)

            img_flat = rearrange(img, "c h w -> (h w) c")

            if self.months_median:
                median_data = [
                    torch.nanmedian(torch.Tensor(img_flat)[:, feat], dim=1)[0]
                    for month in month_median_feat
                    for feat in month
                ]
                img_flat = torch.stack(median_data, dim=-1)

            if type(img_flat) is torch.Tensor:
                img_flat = img_flat.cpu().numpy()
            pred_flat = model.predict(img_flat)

            pred_tiles.append(
                rearrange(pred_flat, "(h w) -> h w", h=h, w=w).astype(int)
            )

        pred_img = np.zeros(
            (int((y_max_glob - y_min_glob) / 10), int((x_max_glob - x_min_glob) / 10)),
            dtype=int,
        )
        # gt_img = np.zeros(
        #     (24, int((y_max_glob - y_min_glob) / 10),
        #     int((x_max_glob - x_min_glob) / 10)),
        #     dtype=float,
        # )

        x_range = np.arange(x_min_glob, x_max_glob, 10)
        y_range = np.arange(y_max_glob, y_min_glob, -10)

        for tt in range(len(pred_coords)):
            x_min, x_max, y_min, y_max = pred_coords[tt]
            idx_x_min, idx_x_max, idx_y_min, idx_y_max = (
                np.where(x_range == x_min)[0][0],
                np.where(x_range == (x_max - 10))[0][0],
                np.where(y_range == y_max)[0][0],
                np.where(y_range == (y_min + 10))[0][0],
            )

            pred_img[idx_y_min : idx_y_max + 1, idx_x_min : idx_x_max + 1] = pred_tiles[
                tt
            ]

            # gt_img[:, idx_y_min : idx_y_max + 1, idx_x_min : idx_x_max + 1]
            # = all_img[tt][:24, :, :]

        profile = {
            "driver": "GTiff",
            "width": len(x_range),
            "height": len(y_range),
            "count": 1,
            "dtype": "uint8",
            "crs": "epsg:32632",
            "transform": Affine(10.0, 0.0, x_min_glob, 0.0, -10.0, y_max_glob),
        }

        pred_image_name = f"tdc_pred_from_{self.sat}_{model_name}"
        if self.months_median:
            pred_image_name += "_median"
        if self.encoded and self.use_logvar:
            pred_image_name += "_uselogvar"
        with rasterio.open(
            os.path.join(path_image_tiles, f"{pred_image_name}.tif"), "w", **profile
        ) as dst:
            dst.write(pred_img, 1)

        del pred_img

        # profile = {
        #     "driver": "GTiff",
        #     "width": len(x_range),
        #     "height": len(y_range),
        #     "count": 24,
        #     "dtype": "float32",
        #     "crs": "epsg:32632",
        #     "transform": Affine(10.0, 0.0, x_min_glob, 0.0, -10.0, y_max_glob),
        # }
        #
        # pred_image_name = f"tdc_gt_from_{self.sat}_{model_name}"
        #
        # with rasterio.open(
        #     os.path.join(path_image_tiles, f"{pred_image_name}.tif"), "w", **profile
        # ) as dst:
        #     dst.write(gt_img)

    def open_csv_files_and_combine(self) -> tuple[pd.DataFrame, np.array]:
        """
        Open csv files saved per patch and combine them in one dataframe.
        """
        all_df = []

        if (
            len(
                [
                    file
                    for file in np.asarray(os.listdir(self.csv_folder))
                    if file.endswith("_all.csv")
                ]
            )
            > 0
        ):
            file = [
                file
                for file in np.asarray(os.listdir(self.csv_folder))
                if file.endswith("_all.csv")
            ][0]
            data = pd.read_csv(os.path.join(self.csv_folder, file))

        else:
            csv_files = [
                file
                for file in np.asarray(os.listdir(self.csv_folder))
                if file.endswith("csv")
                and self.sat in file
                and not file.endswith("_all.csv")
            ]
            for csv_name in np.sort(csv_files):
                data = pd.read_csv(os.path.join(self.csv_folder, csv_name))
                all_df.append(data)

            data = pd.concat(all_df, ignore_index=True)
        log.info(f"Shape of input data {data.shape}")

        days = (
            np.load(os.path.join(self.csv_folder, f"gt_tcd_t32tnt_days_{self.sat}.npy"))
            if Path(
                os.path.join(self.csv_folder, f"gt_tcd_t32tnt_days_{self.sat}.npy")
            ).exists()
            else None
        )

        return data, days

    def get_month_median(
        self,
        data: pd.DataFrame,
        days: np.ndarray,
        bands: list | None = None,
    ) -> tuple[pd.DataFrame, np.array, list[list]]:
        """Compute a months median for each feature and create a new df"""

        median_dict = {}
        print(days)
        months_dates = (
            pd.DatetimeIndex(days).to_frame(name="dates").dates.dt.month.values
        )
        dates_ind = np.arange(len(months_dates))

        months_feat_ind = []

        for month in np.unique(months_dates):
            days = dates_ind[months_dates == month]
            feat_ind = []

            if self.encoded:
                data_cols_month = [
                    c for day in days for c in data.columns if c.endswith(f"d{day}")
                ]
                values = ["mu", "logvar"] if self.use_logvar else ["mu"]
                for value in values:
                    feat_nb = int(
                        len([c for c in data_cols_month if value in c]) / len(days)
                    )
                    for f in range(feat_nb):
                        data_cols_feat = [
                            c for c in data_cols_month if f"{value}_f{f}" in c
                        ]
                        median = np.nanmedian(data[data_cols_feat].values, axis=1)
                        median_dict[f"{value}_f{f}_m{month}"] = median
                        feat_ind.append(
                            [
                                c
                                for c in range(len(self.data_cols))
                                if self.data_cols[c] in data_cols_feat
                            ]
                        )
                months_feat_ind.append(feat_ind)
            else:
                data_cols_month = [
                    c
                    for day in days
                    for c in data.columns
                    if (c.endswith(f"_d{day}") or c.endswith(f"_{day}"))
                ]
                for value in bands:
                    data_cols_feat = [c for c in data_cols_month if value in c]
                    median = np.nanmedian(data[data_cols_feat].values, axis=1)
                    median_dict[f"{value}_m{month}"] = median

                    feat_ind.append(
                        [
                            c
                            for c in range(len(self.data_cols))
                            if self.data_cols[c] in data_cols_feat
                        ]
                    )
                months_feat_ind.append(feat_ind)

        data = pd.concat([data[MAIN_COLS], pd.DataFrame.from_dict(median_dict)], axis=1)
        # if self.sat == "s1":
        #     data = data[data["x"] > self.clip_x]
        data_cols = np.asarray(median_dict.keys())
        # if encoded:
        #     assert data_cols.size * 2 * feat_nb
        # else:

        log.info(f"Shape of median input data {data.shape}")
        print(f"Shape of median input data {data.shape}")
        return data, data_cols, months_feat_ind

    def catboost_algorothm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> tuple[np.ndarray, CatBoostRegressor]:
        """Create model and make a prediction"""

        log.info("Getting model")

        model_cat = CatBoostRegressor(
            iterations=self.cb_params.n_iter,
            l2_leaf_reg=self.cb_params.leaf,
            depth=self.cb_params.depth,
            # loss_function='MAE'
        )

        log.info("Fitting model")
        model_cat.fit(X_train, y_train, verbose=True)

        log.info("Predicting")
        pred = model_cat.predict(X_test).clip(0, 100)

        log.info("Predicted")
        log.info(np.round(pred))
        log.info("GT")
        log.info(y_test.flatten())
        return pred, model_cat

    @staticmethod
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

    def prepare_data(
        self, data: pd.DataFrame, days: np.array
    ) -> tuple[pd.DataFrame, np.array, list[list] | None]:
        if self.encoded:
            data_cols_mu = [c for c in data.columns if "mu" in c]
            if self.use_logvar:
                data_cols_logvar = [c for c in data.columns if "logvar" in c]
                self.data_cols = data_cols_mu + data_cols_logvar
            else:
                self.data_cols = data_cols_mu
            if self.months_median:
                data, data_cols_final, median_months_feat = self.get_month_median(
                    data, days
                )
                return data, data_cols_final, median_months_feat
        else:
            if self.sat == "s2":
                bands = ["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"]
                self.data_cols = [c for c in data.columns if any(b in c for b in bands)]
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
                self.data_cols = [c for c in data.columns if any(b in c for b in bands)]
            if self.months_median:
                data, data_cols_final, median_months_feat = self.get_month_median(
                    data, days, bands=bands
                )
                return data, data_cols_final, median_months_feat
        return data, self.data_cols, None

    def process_satellites(
        self, variable: str = "TCD"
    ) -> tuple[CatBoostRegressor, list[list] | None]:
        data, days = self.open_csv_files_and_combine()

        data, data_cols_final, months_median_feat = self.prepare_data(data, days)

        gt_cols = [variable]

        log.info("splitting data")
        X_train, X_test, y_train, y_test = train_test_split(
            data[data_cols_final].values,
            data[gt_cols].values,
            test_size=0.3,
            random_state=42,
        )

        # Create model, fit and predict
        pred, model_cat = self.catboost_algorothm(X_train, y_train, X_test, y_test)

        # Compute error
        rmse = round(
            mean_squared_error(
                y_test.flatten()[~np.isnan(y_test.flatten())],
                np.round(pred)[~np.isnan(y_test.flatten())],
                squared=False,
            ),
            2,
        )
        log.info(f"RMSE={rmse}")
        print(f"RMSE={rmse}")
        print(np.round(pred))
        print(y_test.flatten())

        # Get scatterplot of the result
        if self.months_median:
            self.folder += "_median"

        if self.encoded and self.use_logvar:
            self.folder += "_logvar"

        img_name = f"{self.folder.split('/')[-1]}_{self.sat}.png"
        Path(self.folder).mkdir(exist_ok=True, parents=True)
        self.results_scatterplot(
            np.round(pred), y_test.flatten(), rmse, self.folder, img_name
        )

        return model_cat, months_median_feat
