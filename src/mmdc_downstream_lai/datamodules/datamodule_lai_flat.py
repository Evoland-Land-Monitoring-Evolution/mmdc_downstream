import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset

logging.getLogger().setLevel(logging.INFO)
np.random.seed(42)


def rename_s1_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "s1_0": "s1_asc_0",
            "s1_1": "s1_asc_1",
            "s1_2": "s1_asc_2",
            "s1_3": "s1_desc_3",
            "s1_4": "s1_desc_4",
            "s1_5": "s1_desc_5",
            "s1_mask_0": "s1_mask_asc",
            "s1_mask_1": "s1_mask_desc",
        }
    )


def get_mean_std(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Get mean, std"""
    mean = df.mean()
    std = df.std()
    return mean, std


def standardize(df: pd.DataFrame, mean: pd.Series, std: pd.Series) -> pd.DataFrame:
    """Standardize training data"""
    return (df - mean) / std


@dataclass
class DataPath:
    train: str | Path
    val: str | Path
    test: str | Path | None = None


@dataclass
class BatchSize:
    train: int
    val: int
    test: int | None = None


@dataclass
class LAIDataSets:
    train: tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]
    val: tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]
    test: tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor] | None = None


class LAIDataModule(LightningDataModule):
    def __init__(
        self,
        path_data: DataPath,
        selected_data: str = "s2_lat",
        batch_size: BatchSize = BatchSize(4096, 4096 * 100, 4096 * 100),
        use_logvar: bool = True,
        norm: bool = False,
    ):
        super().__init__()
        self.path_data = path_data
        self.batch_size = batch_size
        self.selected_data = selected_data
        self.use_logvar = use_logvar
        self.norm = norm

        self.mean, self.std = None, None

        self.selected_col = self.get_column_info()

        self.data: LAIDataSets = None

    def get_column_info(self):
        test_data = pd.read_csv(self.path_data.test, index_col=0, nrows=0)
        column_names = test_data.columns.tolist()
        logging.info("All columns")
        logging.info(column_names)
        x_col = [i for i in column_names if i.startswith(self.selected_data)]
        if not self.use_logvar:
            x_col = [i for i in x_col if "logvar" not in i]

        return [i for i in x_col if i.startswith(self.selected_data)]

    def prepare_xy_data(self, x, y, pos):
        if self.norm:
            x = standardize(x, self.mean, self.std)
        x = torch.tensor(x[self.selected_col].values, dtype=torch.float32)
        y = torch.tensor(y.values, dtype=torch.float32).reshape(-1, 1)
        img, pix = pos
        return x, y, img, pix

    def get_value_label(self, path: str | Path):
        use_cols = self.selected_col + ["gt"] + ["img"] + ["pix"]
        logging.info("Columns to be used")
        logging.info(use_cols)
        df = pd.read_csv(path, usecols=use_cols)
        if len([col for col in self.selected_col if col.startswith("s1")]) > 0:
            logging.info([col for col in self.selected_col if col.startswith("s1")])
            df = rename_s1_cols(df)
        y = df["gt"]  # centered
        x = df[self.selected_col]
        img, pix = torch.Tensor(df["img"]), torch.Tensor(df["pix"])
        return x, y, (img, pix)

    def setup(self, stage: str):
        if stage == "fit":
            logging.info("Opening train df")
            # 102805129
            x_train_all, y_train, pos_train = self.get_value_label(self.path_data.train)
            logging.info("Opening val df")
            # 25720720
            x_val_all, y_val, pos_val = self.get_value_label(self.path_data.val)

            if self.norm:
                self.mean, self.std = get_mean_std(pd.concat([x_train_all, x_val_all]))
                logging.info(self.mean)
                logging.info(self.std)

            self.data = LAIDataSets(
                self.prepare_xy_data(x_train_all, y_train, pos_train),
                self.prepare_xy_data(x_val_all, y_val, pos_val),
            )
        if stage == "test":
            logging.info("Opening test df")
            # 12900234
            x_test_all, y_test, pos_test = self.get_value_label(self.path_data.test)

            self.data = LAIDataSets(
                self.data.train,
                self.data.val,
                self.prepare_xy_data(x_test_all, y_test, pos_test),
            )

    def train_dataloader(self):
        assert self.data is not None
        assert self.data.train is not None
        return DataLoader(
            TensorDataset(*self.data.train),
            batch_size=self.batch_size.train,
            shuffle=True,
            num_workers=4,
        )

    def val_dataloader(self):
        assert self.data is not None
        assert self.data.val is not None
        return DataLoader(
            TensorDataset(*self.data.val),
            batch_size=self.batch_size.val,
            shuffle=False,
            num_workers=4,
        )

    def test_dataloader(self):
        assert self.data is not None
        assert self.data.test is not None
        DataLoader(
            TensorDataset(*self.data.test),
            batch_size=self.batch_size.test,
            shuffle=False,
            num_workers=4,
        )
