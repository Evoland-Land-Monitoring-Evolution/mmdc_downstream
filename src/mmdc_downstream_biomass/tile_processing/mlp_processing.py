from __future__ import annotations

import copy
import logging
import os
import re
import warnings
from collections import namedtuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
import tqdm
from affine import Affine
from einops import rearrange
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
from torch import nn, optim
from torchmetrics import MeanSquaredError

# from torchmetrics.regression import R2Score

warnings.filterwarnings("ignore")

MAIN_COLS = ["index", "TCD", "x", "y", "x_round", "y_round"]

MLP_params = namedtuple("MLP_params", "n_iter bs lr")

log = logging.getLogger(__name__)


def rearrange_ts(ts: torch.Tensor) -> torch.Tensor:
    """Rearrange TS"""
    return rearrange(ts, "t c h w ->  (t c) h w")


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, weights, norm=False):
        if norm:
            torch.sqrt(
                ((inputs - targets) ** 2 * (weights + 10e-6)).sum() / weights.sum()
            )
        return torch.sqrt(((inputs - targets) ** 2 * weights).sum())


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class MLP(nn.Module):
    def __init__(self, input_size: int = 640, sizes=(256, 64)):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        log.info(self.model)
        # self.model = nn.Sequential(
        #     nn.Linear(640, 512),
        #     nn.LeakyReLU(),
        #     nn.Linear(512, 256),
        #     nn.LeakyReLU(),
        #     nn.Linear(256, 64),
        #     nn.LeakyReLU(),
        #     nn.Linear(64, 32),
        #     nn.LeakyReLU(),
        #     nn.Linear(32, 1),
        # )

        self.model.apply(init_weights)

    def forward(self, x):
        return self.model(x)


def get_performance(y, pred, stage="test"):
    # Compute error
    rmse = round(
        mean_squared_error(
            y.flatten()[~np.isnan(y.flatten())],
            np.round(pred, 1)[~np.isnan(y.flatten())],
            squared=False,
        ),
        2,
    )
    r2 = r2_score(
        y.flatten()[~np.isnan(y.flatten())],
        np.round(pred, 1)[~np.isnan(y.flatten())],
    )

    mae = round(
        mean_absolute_error(
            y.flatten()[~np.isnan(y.flatten())],
            np.round(pred, 1)[~np.isnan(y.flatten())],
        ),
        2,
    )

    log.info(f"RMSE {stage}={rmse}")
    print(f"RMSE {stage}={rmse}")

    log.info(f"R2 {stage}={r2}")
    print(f"R2 {stage}={r2}")

    log.info(f"MAE {stage}={mae}")
    print(f"MAE {stage}={mae}")

    print(np.round(pred).min(), np.round(pred).max())
    print(y.flatten().min(), y.flatten().max())

    return rmse


class MLPBiomass:
    def __init__(
        self,
        sat: str,
        csv_folder: str | Path,
        folder: str | Path,
        cb_params: MLP_params,
        encoded: bool,
        months_median: bool,
        use_logvar: bool,
        weights: bool,
    ):
        self.sat = sat
        self.csv_folder = csv_folder
        self.folder = folder
        self.cb_params = cb_params
        self.encoded = encoded
        self.months_median = months_median
        self.weights = weights

        self.data_cols = None

        self.clip_x = 519000

        self.use_logvar = use_logvar

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.site_tiles = [
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
        self.val_tiles = ["33VVK", "33VWK"]

    def predict_new_image(
        self,
        path_image_tiles: str | Path,
        model: MLP,
        model_name: str,
        variable: str = "TCD",
    ) -> None:
        for year in [
            f
            for f in os.listdir(path_image_tiles)
            if Path(os.path.join(path_image_tiles, f)).is_dir()
        ]:
            log.info(year)
            for s2_tile in [
                f
                for f in os.listdir(os.path.join(path_image_tiles, year))
                if Path(os.path.join(path_image_tiles, year, f)).is_dir()
            ]:
                log.info(s2_tile)
                for tt in [
                    t
                    for t in os.listdir(os.path.join(path_image_tiles, year, s2_tile))
                    if t.endswith(".pt")
                ]:
                    idx = int(re.search(r"_([0-9]).pt", tt).group(1))
                    log.info(os.path.join(path_image_tiles, year, s2_tile, tt))
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

                    img = rearrange_ts(img)

                    img_flat = rearrange(img, "c h w -> (h w) c")
                    log.info(img_flat.shape)

                    if type(img_flat) is not torch.Tensor:
                        img_flat = torch.Tensor(img_flat)
                    with torch.no_grad():
                        pred_flat = (
                            model(img_flat.to(self.device)).cpu().numpy().flatten()
                        )

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
                    prefix = f"{variable}_{s2_tile}_{year}_{idx}"
                    pred_image_name = f"{prefix}_pred_from_{self.sat}_{model_name}"
                    log.info(os.path.join(output_dir, f"{pred_image_name}.tif"))
                    with rasterio.open(
                        os.path.join(output_dir, f"{pred_image_name}.tif"),
                        "w",
                        **profile,
                    ) as dst:
                        dst.write(pred_tile, 1)

                    del pred_tile

                    profile = {
                        "driver": "GTiff",
                        "width": w,
                        "height": h,
                        "count": 24,
                        "dtype": "float32",
                        "crs": "epsg:32632",
                        "transform": Affine(10.0, 0.0, x_min, 0.0, -10.0, y_max),
                    }

                    pred_image_name = f"{prefix}_gt_{self.sat}_{model_name}"

                    with rasterio.open(
                        os.path.join(output_dir, f"{pred_image_name}.tif"),
                        "w",
                        **profile,
                    ) as dst:
                        dst.write(img[-24:, :, :])

    def open_csv_files_and_combine(self) -> tuple[pd.DataFrame, np.array]:
        """
        Open csv files saved per patch and combine them in one dataframe.
        """
        all_df = []
        for year in [
            t
            for t in os.listdir(self.csv_folder)
            if Path(os.path.join(self.csv_folder, t)).is_dir()
        ]:
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

    @staticmethod
    def results_scatterplot(
        pred: np.array, y_test: np.array, rmse: float, folder: str, name: str
    ) -> None:
        """Make a scatterplot prediction vs reality"""
        plt.close()
        plt.scatter(pred, y_test, s=1, c="red", alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])
        plt.title(f"RMSE={rmse}%")
        plt.xlabel("pred")
        plt.ylabel("gt")
        Path(folder).mkdir(exist_ok=True, parents=True)
        plt.savefig(os.path.join(folder, name))
        plt.close()

    def prepare_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, np.array]:
        if self.encoded:
            data_cols_mu = [c for c in data.columns if "mu" in c]
            if self.use_logvar:
                data_cols_logvar = [c for c in data.columns if "logvar" in c]
                self.data_cols = data_cols_mu + data_cols_logvar
            else:
                self.data_cols = data_cols_mu

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

        return data, self.data_cols

    def process_satellites(
        self, variable: str = "TCD"
    ) -> tuple[MLP, list[list] | None]:
        data, days = self.open_csv_files_and_combine()
        # data.to_csv("data_biomass_fch.csv", sep=',')
        # data = pd.read_csv("data_biomass_fch.csv", sep=",")

        data, data_cols_final = self.prepare_data(data)
        gt_cols = [v for v in data.columns if variable in v]
        data = data.loc[~(data[data_cols_final] == 0).all(axis=1)]
        data = data.loc[~(data[gt_cols] == 0).any(axis="columns")]
        data = data.loc[~(data[gt_cols].isnull()).any(axis="columns")]
        data = data.loc[pd.notna(data[gt_cols]).any(axis="columns")]
        data = data.loc[pd.notna(data[data_cols_final]).any(axis="columns")]

        if variable == "AGB":
            data = data.loc[data[gt_cols[0]] < 500]
        log.info(f"Shape of input data after filtering {data.shape}")
        # log.info(data_cols_final)

        log.info("splitting data")
        # X_train, X_test, y_train, y_test = train_test_split(
        #     data[data_cols_final].values,
        #     data[gt_cols].values,
        #     test_size=0.3,
        #     random_state=42,
        # )

        # data = data[data.tile.isin(self.site_tiles)]
        all_training_data = data[~data.tile.isin(self.val_tiles)]
        training_tiles = all_training_data.tile.unique()[
            : int(len(all_training_data.tile.unique()) * 0.66)
        ]
        xy_train = all_training_data[all_training_data.tile.isin(training_tiles)]
        xy_val = all_training_data[~all_training_data.tile.isin(training_tiles)]
        xy_test = data[data.tile.isin(self.val_tiles)]
        log.info(f"training {xy_train.shape}")
        log.info(f"validation {xy_val.shape}")
        log.info(f"testing {xy_test.shape}")

        X_train, y_train = xy_train[data_cols_final].values, xy_train[gt_cols].values
        X_val, y_val = xy_val[data_cols_final].values, xy_val[gt_cols].values
        X_test, y_test = xy_test[data_cols_final].values, xy_test[gt_cols].values

        X_train, y_train = (
            X_train[~np.isnan(X_train).any(axis=1)],
            y_train[~np.isnan(X_train).any(axis=1)],
        )
        X_val, y_val = (
            X_val[~np.isnan(X_val).any(axis=1)],
            y_val[~np.isnan(X_val).any(axis=1)],
        )
        X_test, y_test = (
            X_test[~np.isnan(X_test).any(axis=1)],
            y_test[~np.isnan(X_test).any(axis=1)],
        )

        # scaler_x = StandardScaler()
        # scaler_x.fit(X_train)
        scaler_y = RobustScaler(quantile_range=(0.5, 99.5))  # StandardScaler()
        scaler_y.fit(y_train)

        log.info("Fitting model")

        count, value = np.histogram(scaler_y.transform(y_train), bins=20)
        weights = count / count.sum()
        a = -1
        weights = np.exp((weights**0.5) * a)
        weights = weights / weights.sum()
        plt.plot(weights)
        plt.show()

        def get_weights(y_value):
            weight_position = np.digitize(y_value, value, right=True).flatten() - 1
            weight_position[weight_position < 0] = 0
            weight_position[weight_position >= len(value) - 1] = len(value) - 2
            return weights[weight_position]

        # Create model, fit and predict
        pred, model_mlp = self.mlp_algorithm(
            torch.Tensor(X_train),
            torch.Tensor(scaler_y.transform(y_train)),
            torch.Tensor(X_val),
            torch.Tensor(scaler_y.transform(y_val)),
            weights_train=torch.Tensor(get_weights(scaler_y.transform(y_train)))
            if self.weights
            else None,
            weights_test=torch.Tensor(get_weights(scaler_y.transform(y_val)))
            if self.weights
            else None,
        )

        pred_train = (
            scaler_y.inverse_transform(
                model_mlp(torch.Tensor(X_train).to(pred.device)).cpu().detach().numpy()
            )
            .flatten()
            .clip(0, 1000)
        )
        _ = get_performance(y_train, pred_train, stage="train")

        pred_val = (
            scaler_y.inverse_transform(pred.cpu().detach().numpy())
            .flatten()
            .clip(0, 1000)
        )
        _ = get_performance(y_val, pred_val, stage="val")

        pred_test = (
            scaler_y.inverse_transform(
                model_mlp(torch.Tensor(X_test).to(pred.device)).cpu().detach().numpy()
            )
            .flatten()
            .clip(0, 1000)
        )
        rmse = get_performance(y_test, pred_test, stage="test")

        # Get scatterplot of the result
        if self.months_median:
            self.folder += "_median"

        if self.encoded and self.use_logvar:
            self.folder += "_logvar"

        img_name = f"{self.folder.split('/')[-1]}_{self.sat}_{variable}.png"
        Path(self.folder).mkdir(exist_ok=True, parents=True)
        self.results_scatterplot(
            np.round(pred_test, 1), y_test.flatten(), rmse, self.folder, img_name
        )

        return model_mlp, None

    def mlp_algorithm(
        self, X_train, y_train, X_test, y_test, weights_train=None, weights_test=None
    ):
        model = MLP(X_train.shape[1]).to(self.device)

        # training parameters
        n_epochs = self.cb_params.n_iter  # number of epochs to run
        batch_size = self.cb_params.bs  # size of each batch
        lr = self.cb_params.lr  # size of each batch
        batch_start = torch.arange(0, len(X_train), batch_size)

        best_model = None

        # Hold the best model
        best_rmse = np.inf  # init to infinity
        history = []

        # loss function and optimizer
        if not self.weights:
            loss_fn = MeanSquaredError(squared=False).to(
                self.device
            )  # mean square error
        else:
            loss_fn = WeightedMSELoss()  # mean square error
        # metric = R2Score().to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)

        count = 0

        # training loop
        for epoch in range(n_epochs):
            if epoch == 25:
                optimizer.lr = lr * 0.1
            #     batch_size = int(batch_size / 2)
            model.train()
            ind = torch.randperm(y_train.size()[0])
            X_train = X_train[ind]
            y_train = y_train[ind]
            if weights_train is not None:
                weights_train_ = weights_train[ind]
            with tqdm.tqdm(
                batch_start, unit="batch", mininterval=0, disable=True
            ) as bar:
                bar.set_description(f"Epoch {epoch}")
                log.info(f"Epoch {epoch}")
                for start in bar:
                    # take a batch
                    X_batch = X_train[start : start + batch_size].to(self.device)
                    y_batch = y_train[start : start + batch_size].to(self.device)
                    if weights_train is not None:
                        weights_train_batch = weights_train_[
                            start : start + batch_size
                        ].to(self.device)
                    # forward pass
                    y_pred = model(X_batch)
                    if not self.weights:
                        loss = loss_fn(y_pred, y_batch)
                    else:
                        loss = loss_fn(y_pred, y_batch, weights=weights_train_batch)
                    # backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    # update weights
                    optimizer.step()
                    # print progress
                    bar.set_postfix(rmse=float(loss))
            # evaluate accuracy at end of each epoch
            model.eval()
            y_pred = model(X_test.to(self.device))
            if not self.weights:
                rmse = loss_fn(y_pred, y_test.to(self.device))
            else:
                rmse = loss_fn(
                    y_pred, y_test.to(self.device), weights=weights_test.to(self.device)
                )
            # r2_score = metric(y_pred, y_test.to(device))
            rmse = float(rmse)
            history.append(rmse)
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = copy.deepcopy(model)
                count = 0
            else:
                count += 1
                if count > 50:
                    break

        # restore model and return best accuracy
        # best_model = model.load_state_dict(best_weights)
        print(best_model)

        with torch.no_grad():
            pred = best_model(X_test.to(self.device))

        print("MSE: %.2f" % best_rmse**2)
        print("RMSE: %.2f" % best_rmse)
        plt.plot(history)
        # plt.show()
        plt.savefig("model.png")

        return pred, best_model
