import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

from mmdc_downstream_lai.snap.lai_snap import denormalize

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
    print(len(df))
    if len([col for col in x_col if col.startswith("s1")]) > 0:
        logging.info([col for col in x_col if col.startswith("s1")])
        df = rename_s1_cols(df)
    y = df["gt"]  # centered
    x = df[x_col]
    return x, y, x_col


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


logging.info("Starting everything")

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
logging.info("All columns")
logging.info(column_names)

selected_data = ["exp_lat", "s1_lat", "s2_lat", "s1_asc", "s1_desc", "s1_mask"]
# train_data = \
#     ["s2_input", "exp_lat", "s1_lat", "s2_lat", 's1_asc', "s1_desc", "s1_mask"]

selected_data = ["s2_lat"]

USE_LOGVAR = True

depth = 15

logging.info("Opening train df")
x_train_all, y_train, x_col = get_value_label(
    path_train, selected_data, column_names, USE_LOGVAR
)
logging.info("Opening val df")
x_val_all, y_val, _ = get_value_label(path_val, selected_data, column_names, USE_LOGVAR)
logging.info("Opening test df")
x_test_all, y_test, _ = get_value_label(
    path_test, selected_data, column_names, USE_LOGVAR
)

results = {}
y_test_denorm = {}

for data_type in selected_data:
    path = f"/work/scratch/data/kalinie/MMDC/results/random_forest/{data_type}"
    folder = f"forest_{depth}"
    if USE_LOGVAR:
        folder += "_logvar"
    output_folder = os.path.join(path, folder)
    Path(output_folder).mkdir(exist_ok=True, parents=True)

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

        idx = np.random.choice(len(X_train), int(len(X_train) * 0.1), replace=False)

        tree_15 = DecisionTreeRegressor(max_depth=depth)
        tree_15.fit(X_train.values[idx], y_train.values[idx])

        # Predict
        preds = tree_15.predict(X_val)

        idx = np.random.choice(len(preds), int(len(preds) * 0.05), replace=False)
        preds = denormalize(preds, lai_min, lai_max)
        y_val_denorm = denormalize(y_val.values, lai_min, lai_max).flatten()

        logging.info("pred max " + str(preds.max()))
        logging.info("pred min " + str(preds.min()))

        plt.close()
        plt.scatter(preds[idx], y_val_denorm[idx], s=0.1)
        plt.plot(y_val_denorm[idx], y_val_denorm[idx])
        plt.title(
            "loss="
            + str(np.round(mean_squared_error(preds, y_val_denorm, squared=False), 4))
        )
        plt.xlabel("pred")
        plt.ylabel("gt")

        plt.savefig(output_folder + "/nn_forest.png")

        # Process is complete.
        logging.info("Training process has finished.")
