import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.regression import MeanSquaredError

from mmdc_downstream.snap.lai_snap import denormalize

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


def get_value_label(path, selected_data, column_names, use_logvar):
    x_col = [
        i
        for i in column_names
        for data_type in selected_data
        if i.startswith(data_type)
    ]
    if not use_logvar:
        x_col = [i for i in x_col if "logvar" not in i]
    use_cols = x_col + ["gt"]
    logging.info("Columns to be used")
    logging.info(use_cols)
    df = pd.read_csv(path, usecols=use_cols)
    if len([col for col in x_col if col.startswith("s1")]) > 0:
        logging.info([col for col in x_col if col.startswith("s1")])
        df = rename_s1_cols(df)
    y = df["gt"]  # centered
    x = df[x_col]
    return x, y, x_col


def get_mean_std(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Get mean, std"""
    mean = df.mean()
    std = df.std()
    return mean, std


def standardize(df: pd.DataFrame, mean: pd.Series, std: pd.Series) -> pd.DataFrame:
    """Standardize training data"""
    return (df - mean) / std


def select_s1(x_train, x_test, x_val, y_train, y_test, y_val, orbit):
    mask_train = x_train[f"s1_mask_{orbit}"].values.astype(bool)
    mask_test = x_test[f"s1_mask_{orbit}"].values.astype(bool)
    mask_val = x_val[f"s1_mask_{orbit}"].values.astype(bool)
    return (
        x_train[~mask_train],
        x_test[~mask_test],
        x_val[~mask_val],
        y_train[~mask_train],
        y_test[~mask_test],
        y_val[~mask_val],
    )


class MLP(nn.Module):
    """Multilayer Perceptron for regression."""

    def __init__(self, in_feat=11):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_feat, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        """Forward pass"""
        return self.layers(x)


logging.info("Starting everything")

lai_min = torch.tensor(0.000319182538301)
lai_max = torch.tensor(14.4675094548)

folder_data = "/work/scratch/data/kalinie/MMDC/jobs"
folder_data = f"{os.environ['WORK']}/results/LAI/"


path_train = os.path.join(
    folder_data, "data_values_baseline_2024-02-27_14-47-18_train.csv"
)
path_val = os.path.join(folder_data, "data_values_baseline_2024-02-27_14-47-18_val.csv")
path_test = os.path.join(
    folder_data, "data_values_baseline_2024-02-27_14-47-18_test.csv"
)

column_names = pd.read_csv(path_test, index_col=0, nrows=0).columns.tolist()
logging.info("All columns")
logging.info(column_names)

selected_data = ["exp_lat", "s1_lat", "s2_lat", "s1_asc", "s1_desc", "s1_mask"]
# train_data = \
#     ["s2_input", "exp_lat", "s1_lat", "s2_lat", 's1_asc', "s1_desc", "s1_mask"]

selected_data = ["s1_lat"]

USE_LOGVAR = True

logging.info("Opening train df")
# 102805129
x_train_all, y_train, x_col = get_value_label(
    path_train, selected_data, column_names, USE_LOGVAR
)
logging.info("Opening val df")
# 25720720
x_val_all, y_val, _ = get_value_label(path_val, selected_data, column_names, USE_LOGVAR)
logging.info("Opening test df")
# 12900234
x_test_all, y_test, _ = get_value_label(
    path_test, selected_data, column_names, USE_LOGVAR
)

mean, std = get_mean_std(x_train_all)
logging.info(mean)
logging.info(std)

x_train_all = standardize(x_train_all, mean, std)
x_val_all = standardize(x_val_all, mean, std)
x_test_all = standardize(x_test_all, mean, std)


results = {}
y_test_denorm = {}

for data_type in selected_data:
    if data_type != "s1_mask":
        train_col = [i for i in x_col if i.startswith(data_type)]
        logging.info(train_col)
        if train_col[0].startswith("s1_asc"):
            x_train_all, x_test_all, x_val_all, y_train, y_test, y_val = select_s1(
                x_train_all, x_test_all, x_val_all, y_train, y_test, y_val, orbit="asc"
            )

        if train_col[0].startswith("s1_desc"):
            x_train_all, x_test_all, x_val_all, y_train, y_test, y_val = select_s1(
                x_train_all, x_test_all, x_val_all, y_train, y_test, y_val, orbit="desc"
            )

        X_train = x_train_all[train_col]
        X_val = x_val_all[train_col]
        X_test = x_test_all[train_col]

        X_train = torch.tensor(X_train.values, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
        X_val = torch.tensor(X_val.values, dtype=torch.float32)
        y_val = torch.tensor(y_val.values, dtype=torch.float32).reshape(-1, 1)
        X_test = torch.tensor(X_test.values, dtype=torch.float32)
        y_test = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)

        model = MLP(in_feat=X_train.shape[1])
        print(
            denormalize(y_train, lai_min, lai_max).min(),
            denormalize(y_train, lai_min, lai_max).max(),
        )
        train_dl = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=64 * 64,
            shuffle=True,
            num_workers=4,
        )
        print(
            denormalize(y_val, lai_min, lai_max).min(),
            denormalize(y_train, lai_min, lai_max).max(),
        )
        val_dl = DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=400 * 64 * 64,
            shuffle=False,
            num_workers=4,
        )
        test_dl = DataLoader(
            TensorDataset(X_test, y_test),
            batch_size=400 * 64 * 64,
            shuffle=False,
            num_workers=4,
        )

        # loss function and optimizer
        loss_fn = MeanSquaredError(squared=False)  # root mean square error
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        # training parameters
        n_epochs = 50  # number of epochs to run

        # Hold the best model
        best_mse = np.inf  # init to infinity
        best_weights = None
        history = []

        loss_epochs = []

        # path = f"/work/scratch/data/kalinie/MMDC/results/simple_deep/{data_type}"
        path = f"{os.environ['WORK']}/results/MMDC/results/simple_deep/{data_type}"

        folder = (
            f"bs_{train_dl.batch_size}_lr_"
            f"{optimizer.param_groups[0]['lr']}_hidden_16_16_norm"
        )
        if USE_LOGVAR:
            folder += "_logvar"
        output_folder = os.path.join(path, folder)
        logging.info(output_folder)
        Path(output_folder).mkdir(exist_ok=True, parents=True)

        # Run the training loop
        for epoch in range(n_epochs):  # 5 epochs at maximum
            if epoch == 5:
                optimizer.param_groups[0]["lr"] = 0.00001

            # Print epoch
            logging.info(f"Starting epoch {epoch + 1}")

            # Set current loss value
            current_loss_train = 0.0
            current_loss_val = 0.0

            # Iterate over the DataLoader for training data
            for i, data in enumerate(tqdm.tqdm(train_dl)):
                model.train()
                # Get and prepare inputs
                inputs, targets = data
                inputs, targets = inputs.float(), targets.float()
                targets = targets.reshape((targets.shape[0], 1))

                # Zero the gradients
                optimizer.zero_grad()

                # Perform forward pass
                outputs = model(inputs)

                # Compute loss
                loss = loss_fn(outputs, targets)

                # Perform backward pass
                loss.backward()

                # Perform optimization
                optimizer.step()

                # Print statistics
                current_loss_train += loss.item()
            logging.info(
                "Train Loss after epoch %5d: %.3f"
                % (
                    epoch + 1,
                    current_loss_train / train_dl.__len__(),
                )
            )

            preds = []
            for i, data in enumerate(tqdm.tqdm(val_dl)):
                model.eval()
                # Get and prepare inputs
                inputs, targets = data
                inputs, targets = inputs.float(), targets.float()
                targets = targets.reshape((targets.shape[0], 1))

                # Zero the gradients
                optimizer.zero_grad()

                # Perform forward pass
                outputs = model.forward(inputs)
                preds.append(outputs.detach().cpu().numpy().reshape(-1))

                # Compute loss
                loss = loss_fn(outputs, targets)

                # Print statistics
                current_loss_val += loss.item()
            logging.info(
                "Validation Loss after epoch %5d: %.3f"
                % (
                    epoch + 1,
                    current_loss_val / val_dl.__len__(),
                )
            )
            preds = np.concatenate(preds, 0)

            idx = np.random.choice(len(preds), int(len(preds) * 0.05), replace=False)
            preds = denormalize(torch.Tensor(preds), lai_min, lai_max)
            y_val_denorm = denormalize(y_val, lai_min, lai_max).flatten()
            print(y_val_denorm)

            logging.info("pred max " + str(preds.max()))
            logging.info("pred min " + str(preds.min()))

            plt.close()
            plt.scatter(preds[idx], y_val_denorm[idx], s=0.1)
            plt.plot(y_val_denorm[idx], y_val_denorm[idx])

            plt.title(
                "loss="
                + str(np.round(loss_fn(torch.Tensor(preds), y_val_denorm).item(), 4))
            )
            plt.xlabel("pred")
            plt.ylabel("gt")

            plt.savefig(output_folder + f"/mlp_{epoch}.png")

            # Process is complete.
        logging.info("Training process has finished.")
