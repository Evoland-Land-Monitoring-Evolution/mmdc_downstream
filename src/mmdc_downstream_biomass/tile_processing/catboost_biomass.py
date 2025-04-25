import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from affine import Affine
from catboost import CatBoostRegressor
from einops import rearrange
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from mmdc_downstream_tcd.catboost.processing import CatBoostTCD, CB_params
from src.mmdc_downstream_tcd.tile_processing.utils import rearrange_ts

log = logging.getLogger(__name__)


class CatBoostBioMass(CatBoostTCD):
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
        super().__init__(
            sat, csv_folder, folder, cb_params, encoded, months_median, use_logvar
        )

    def predict_new_image(
        self,
        path_image_tiles: str | Path,
        model: CatBoostRegressor,
        model_name: str,
        month_median_feat: list[list] | None,
        variable: str = "TCD",
    ) -> None:
        for year in os.listdir(path_image_tiles):
            print(year)
            for s2_tile in [
                f
                for f in os.listdir(os.path.join(path_image_tiles, year))
                if Path(os.path.join(path_image_tiles, year, f)).is_dir()
            ]:
                print(s2_tile)
                for tt in [
                    t
                    for t in os.listdir(os.path.join(path_image_tiles, year, s2_tile))
                    if t.endswith(".pt")
                ]:
                    idx = int(re.search(r"_([0-9]).pt", tt).group(1))
                    tile_data = torch.load(
                        os.path.join(path_image_tiles, year, s2_tile, tt)
                    )
                    log.info(f"Predicting tile {tt}")
                    matrix = tile_data["matrix"]
                    if type(matrix) is dict:
                        x_min, _, _, y_max = (
                            matrix["x_min"] - 5,
                            matrix["x_max"] + 5,
                            matrix["y_min"] - 5,
                            matrix["y_max"] + 5,
                        )
                    else:
                        x_min, _, _, y_max = (
                            matrix[0].min() - 5,
                            matrix[0].max() + 5,
                            matrix[1].min() - 5,
                            matrix[1].max() + 5,
                        )

                    img = tile_data["img"]
                    t, c, h, w = img.shape

                    if not (
                        "alise" in path_image_tiles or "malice" in path_image_tiles
                    ):
                        mu = img[:, : int(c / 2)]
                        if not self.use_logvar:
                            img = mu
                        else:
                            logvar = img[:, int(c / 2) :]
                            img = np.concatenate([mu, logvar], 0)

                    img = rearrange_ts(img)

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

                    pred_tile = np.round(
                        rearrange(pred_flat, "(h w) -> h w", h=h, w=w), 2
                    )

                    profile = {
                        "driver": "GTiff",
                        "width": pred_tile.shape[-1],
                        "height": pred_tile.shape[-2],
                        "count": 1,
                        "dtype": "float32",
                        "crs": "epsg:32633",
                        "transform": Affine(10.0, 0.0, x_min, 0.0, -10.0, y_max),
                    }
                    output_dir = os.path.join(
                        path_image_tiles.replace("encoded", "results")
                    )
                    Path(output_dir).mkdir(exist_ok=True, parents=True)
                    pred_image_name = (
                        f"{variable}_{s2_tile}_{idx}_pred_from_{self.sat}_{model_name}"
                    )
                    if self.months_median:
                        pred_image_name += "_median"
                    if self.encoded and self.use_logvar:
                        pred_image_name += "_uselogvar"
                    with rasterio.open(
                        os.path.join(output_dir, f"{pred_image_name}.tif"),
                        "w",
                        **profile,
                    ) as dst:
                        dst.write(pred_tile, 1)

                    del pred_tile

    def open_csv_files_and_combine(self) -> tuple[pd.DataFrame, np.array]:
        """
        Open csv files saved per patch and combine them in one dataframe.
        """
        all_df = []
        for year in os.listdir(self.csv_folder):
            for tile in os.listdir(os.path.join(self.csv_folder, year)):
                csv_files = [
                    file
                    for file in np.asarray(
                        os.listdir(os.path.join(self.csv_folder, year, tile))
                    )
                    if file.endswith("csv")
                    and self.sat in file
                    and not file.endswith("_all.csv")
                ]
                for csv_name in np.sort(csv_files):
                    data = pd.read_csv(
                        os.path.join(self.csv_folder, year, tile, csv_name)
                    )
                    data["tile"] = tile
                    all_df.append(data)

        data = pd.concat(all_df, ignore_index=True)
        log.info(f"Shape of input data {data.shape}")

        days = (
            np.load(
                os.path.join(
                    self.csv_folder, year, tile, f"gt_tcd_t32tnt_days_{self.sat}.npy"
                )
            )
            if Path(
                os.path.join(
                    self.csv_folder, year, tile, f"gt_tcd_t32tnt_days_{self.sat}.npy"
                )
            ).exists()
            else None
        )

        return data, days

    def catboost_algorithm(
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
        pred = model_cat.predict(X_test)

        log.info("Predicted")
        log.info(np.round(pred))
        log.info("GT")
        log.info(y_test.flatten())
        return pred, model_cat

    def mlp_algorithm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> tuple[np.ndarray, MLPRegressor]:
        model_mlp = MLPRegressor(
            hidden_layer_sizes=[512, 128, 64],
            batch_size=25,
            learning_rate="adaptive",
            learning_rate_init=0.0001,
            early_stopping=False,
            verbose=True,
            tol=0.0000001,
            n_iter_no_change=50,
            max_iter=250,
        )
        scaler_x = StandardScaler()
        scaler_x.fit(X_train)
        scaler_y = StandardScaler()
        scaler_y.fit(y_train)

        log.info("Fitting model")
        model_mlp.fit(scaler_x.transform(X_train), scaler_y.transform(y_train))

        log.info("Predicting")
        pred = (
            scaler_y.inverse_transform(
                model_mlp.predict(scaler_x.transform(X_test)).reshape(-1, 1)
            )
            .flatten()
            .clip(0, 1000)
        )

        log.info("Predicted")
        log.info(pred)
        log.info("GT")
        log.info(y_test.flatten())
        return pred, model_mlp

    def process_satellites(
        self, variable: str = "TCD"
    ) -> tuple[CatBoostRegressor, list[list] | None]:
        data, days = self.open_csv_files_and_combine()

        data, data_cols_final, months_median_feat = self.prepare_data(data, days)
        gt_cols = [v for v in data.columns if variable in v]
        data = data.loc[~(data[data_cols_final] == 0).all(axis=1)]
        data = data.loc[~(data[gt_cols] == 0).any(axis="columns")]
        data = data.loc[~(data[gt_cols].isnull()).any(axis="columns")]
        data = data.loc[pd.notna(data[gt_cols]).any(axis="columns")]
        data = data.loc[pd.notna(data[data_cols_final]).any(axis="columns")]
        if variable == "AGB":
            data = data.loc[data[gt_cols[0]] < 500]
        log.info(f"Shape of input data after filtering {data.shape}")
        log.info(data_cols_final)

        log.info("splitting data")
        # X_train, X_test, y_train, y_test = train_test_split(
        #     data[data_cols_final].values,
        #     data[gt_cols].values,
        #     test_size=0.3,
        #     random_state=42,
        # )
        site_tiles = [
            "3VVH",
            "33VVJ",
            "33VVK",
            "33VVL",
            "33VWH",
            "33VWJ",
            "33VWK",
            "33VWL",
            "33VXH",
            "33VXJ",
            "33VXK",
            "33VXL",
            "33WVM",
            "33WWM",
            "33WXM",
            "34VCN",
            "34VCP",
            "34VCQ",
            "34VCR",
        ]
        val_tiles = ["33VVK", "33VWK"]
        data = data[data.tile.isin(site_tiles)]
        xy_train = data[~data.tile.isin(val_tiles)]
        xy_test = data[data.tile.isin(val_tiles)]

        log.info(f"training {xy_train.shape}")
        log.info(f"testing {xy_test.shape}")

        X_train, y_train = xy_train[data_cols_final].values, xy_train[gt_cols].values
        X_test, y_test = xy_test[data_cols_final].values, xy_test[gt_cols].values

        # Create model, fit and predict
        pred, model_cat = self.catboost_algorithm(X_train, y_train, X_test, y_test)

        pred = pred.clip(0, 10000)

        # Compute error
        rmse = round(
            mean_squared_error(
                y_test.flatten()[~np.isnan(y_test.flatten())],
                np.round(pred, 1)[~np.isnan(y_test.flatten())],
                squared=False,
            ),
            2,
        )
        r2 = r2_score(
            y_test.flatten()[~np.isnan(y_test.flatten())],
            np.round(pred, 1)[~np.isnan(y_test.flatten())],
        )

        mae = round(
            mean_absolute_error(
                y_test.flatten()[~np.isnan(y_test.flatten())],
                np.round(pred, 1)[~np.isnan(y_test.flatten())],
            ),
            2,
        )

        log.info(f"RMSE={rmse}")
        print(f"RMSE={rmse}")

        log.info(f"R2={r2}")
        print(f"R2={r2}")

        log.info(f"MAE={mae}")
        print(f"MAE={mae}")

        print(np.round(pred))
        print(y_test.flatten())

        # Get scatterplot of the result
        if self.months_median:
            self.folder += "_median"

        if self.encoded and self.use_logvar:
            self.folder += "_logvar"

        img_name = f"{self.folder.split('/')[-1]}_{self.sat}_{variable}.png"
        Path(self.folder).mkdir(exist_ok=True, parents=True)
        self.results_scatterplot(
            np.round(pred, 1), y_test.flatten(), rmse, self.folder, img_name
        )

        return model_cat, months_median_feat
