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
        for s2_tile in [
            f
            for f in os.listdir(path_image_tiles)
            if Path(os.path.join(path_image_tiles, f)).is_dir()
        ]:
            for tt in [
                t
                for t in os.listdir(os.path.join(path_image_tiles, s2_tile))
                if t.endswith(".pt")
            ]:
                idx = int(re.search(r"_([0-9]).pt", tt).group(1))
                tile_data = torch.load(os.path.join(path_image_tiles, s2_tile, tt))
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

                if not ("alise" in path_image_tiles or "malice" in path_image_tiles):
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

                pred_tile = np.round(rearrange(pred_flat, "(h w) -> h w", h=h, w=w), 2)

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
                    os.path.join(output_dir, f"{pred_image_name}.tif"), "w", **profile
                ) as dst:
                    dst.write(pred_tile, 1)

                del pred_tile

    def open_csv_files_and_combine(self) -> tuple[pd.DataFrame, np.array]:
        """
        Open csv files saved per patch and combine them in one dataframe.
        """
        all_df = []
        for tile in os.listdir(self.csv_folder):
            csv_files = [
                file
                for file in np.asarray(os.listdir(os.path.join(self.csv_folder, tile)))
                if file.endswith("csv")
                and self.sat in file
                and not file.endswith("_all.csv")
            ]
            for csv_name in np.sort(csv_files):
                data = pd.read_csv(os.path.join(self.csv_folder, tile, csv_name))
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
