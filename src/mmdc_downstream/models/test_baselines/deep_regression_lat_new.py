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

from mmdc_downstream.snap.lai_snap import denormalize
from mmdc_downstream.utils import get_logger

log = get_logger(__name__)


class MLP(nn.Module):
    """Multilayer Perceptron for regression."""

    def __init__(self, in_feat=11):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(in_feat, 5), nn.Tanh(), nn.Linear(5, 1))

    def forward(self, x):
        """Forward pass"""
        return self.layers(x)


lai_min = 0.000319182538301
lai_max = 14.4675094548

folder_data = "/work/scratch/data/kalinie/MMDC/jobs"

path_train = os.path.join(
    folder_data, "data_values_baseline_2024-02-27_14-47-18_train.csv"
)
path_val = os.path.join(folder_data, "data_values_baseline_2024-02-27_14-47-18_val.csv")
path_test = os.path.join(
    folder_data, "data_values_baseline_2024-02-27_14-47-18_test.csv"
)

column_names = pd.read_csv(path_test, index_col=0, nrows=0).columns.tolist()
log.info("Columns to be used")
log.info(column_names)

df_train = pd.read_csv(path_train)
df_test = pd.read_csv(path_test)
df_val = pd.read_csv(path_val)


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


def get_value_label(path, selected_data, column_names):
    x_col = [
        i
        for i in column_names
        for data_type in selected_data
        if i.startswith(data_type)
    ]
    df = pd.read_csv(path, usecols=x_col.extend("gt"))
    if len([col for col in x_col if col.startswith("s1")]) > 0:
        df = rename_s1_cols(df)
    y = df["gt"]  # centered
    x = df[x_col]
    return x, y


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


selected_data = ["exp_lat", "s1_lat", "s2_lat", "s1_asc", "s1_desc", "s1_mask"]
# train_data = \
#     ["s2_input", "exp_lat", "s1_lat", "s2_lat", 's1_asc', "s1_desc", "s1_mask"]

selected_data = ["s2_lat"]

log.info("Opening train df")
x_train_all, y_train = get_value_label(path_train, selected_data, column_names)
log.info("Opening val df")
x_val_all, y_val = get_value_label(path_val, selected_data, column_names)
log.info("Opening test df")
x_test_all, y_test = get_value_label(path_test, selected_data, column_names)

results = {}
y_test_denorm = {}

for data_type in selected_data:
    if data_type != "s1_mask":
        train_col = [i for i in df_train.columns if i.startswith(data_type)]
        log.info(train_col)
        if train_col[0].startswith("s1_asc"):
            x_train_all, x_test_all, x_val_all, y_train, y_test, y_val = select_s1(
                x_train_all, x_test_all, x_val_all, y_train, y_test, y_val, orbit="asc"
            )

        if train_col[0].startswith("s1_desc"):
            x_train, x_test, x_val, y_train, y_test, y_val = select_s1(
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
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)

        train_dl = DataLoader(
            train_dataset, batch_size=50 * 64 * 64, shuffle=True, num_workers=2
        )
        val_dl = DataLoader(
            val_dataset, batch_size=200 * 64 * 64, shuffle=False, num_workers=2
        )
        test_dl = DataLoader(
            test_dataset, batch_size=200 * 64 * 64, shuffle=False, num_workers=2
        )

        # loss function and optimizer
        loss_fn = nn.MSELoss()  # mean square error
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # training parameters
        n_epochs = 100  # number of epochs to run

        # Hold the best model
        best_mse = np.inf  # init to infinity
        best_weights = None
        history = []

        loss_epochs = []

        path = f"/work/scratch/data/kalinie/MMDC/results/simple_deep/{data_type}"
        folder = (
            f"bs_{train_dl.batch_size}_lr_{optimizer.param_groups[0]['lr']}_hidden_5"
        )
        output_folder = os.path.join(path, folder)
        log.info(output_folder)
        Path(output_folder).mkdir(exist_ok=True, parents=True)

        # Run the training loop
        for epoch in range(n_epochs):  # 5 epochs at maximum
            if epoch == 5:
                optimizer.param_groups[0]["lr"] = 0.0001

            # Print epoch
            log.info(f"Starting epoch {epoch + 1}")

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
            log.info(
                "Train Loss after epoch %5d: %.3f"
                % (
                    epoch + 1,
                    current_loss_train / int(len(train_dataset) / train_dl.batch_size),
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
            log.info(
                "Validation Loss after epoch %5d: %.3f"
                % (
                    epoch + 1,
                    current_loss_val / int(len(val_dataset) / val_dl.batch_size),
                )
            )
            preds = np.concatenate(preds, 0)

            pred = denormalize(preds, lai_min, lai_max)
            y_val_denorm = (
                denormalize(y_val, lai_min, lai_max).flatten().to(torch.float32)
            )

            log.info("max", pred.max())
            log.info("min", pred.min())

            plt.close()
            plt.scatter(y_val_denorm, pred, s=0.1)
            plt.plot(y_val_denorm, y_val_denorm)
            plt.xlabel(
                "loss="
                + str(np.round(loss_fn(torch.Tensor(pred), y_val_denorm).item(), 4))
            )
            plt.savefig(output_folder + f"/mlp_{epoch}.png")

            # Process is complete.
        log.info("Training process has finished.")
