import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

main_cols = ["index", "TCD", "x", "y", "x_round", "y_round"]

folder_data = "/work/scratch/data/kalinie/TCD/OpenEO_data/t32tnt/encoded/"

model = "res_2024-03-08_11-09-43"
model = "res_2024-02-27_14-47-18"
# model = "res_2024-03-06_08-51-27"

feat_nb = 6

n_iterations = 1000
leaf = 17
depth = 11


months_median = True

csv_folder = os.path.join(folder_data, model)

satellites = ["s2", "s1_asc", "s1_desc"]

for sat in satellites:
    all_df = []

    csv_files = [
        file for file in os.listdir(csv_folder) if file.endswith("csv") and sat in file
    ]

    months_dates = None
    for csv_name in np.sort(csv_files):
        data = pd.read_csv(os.path.join(csv_folder, csv_name))
        all_df.append(data)

        if months_dates is None:
            days_name = csv_name.replace("encoded", "days").replace("csv", "npy")
            days = np.load(os.path.join(csv_folder, days_name))
            months_dates = (
                pd.DatetimeIndex(days).to_frame(name="dates").dates.dt.month.values
            )
            dates_ind = np.arange(len(months_dates))

    data = pd.concat(all_df, ignore_index=True)
    print(data.shape)

    data_cols = [c for c in data.columns if ("mu" in c)]  # or ('logvar' in c)]
    gt_cols = ["TCD"]

    median_dict = {}

    if months_median:
        for month in np.unique(months_dates):
            days = dates_ind[months_dates == month]

            data_cols_month = [
                c for day in days for c in data.columns if c.endswith(f"d{day}")
            ]
            for value in ("mu", "logvar"):
                for f in range(feat_nb):
                    data_cols_feat = [
                        c for c in data_cols_month if f"{value}_f{f}" in c
                    ]
                    median = np.nanmedian(data[data_cols_feat].values, axis=1)
                    median_dict[f"{value}_f{f}_m{month}"] = median

        data = pd.concat([data[main_cols], pd.DataFrame.from_dict(median_dict)], axis=1)
        data_cols = median_dict.keys()

        data = data.fillna(-1)

    print("splitting data")

    X_train, X_test, y_train, y_test = train_test_split(
        data[data_cols].values, data[gt_cols].values, test_size=0.3, random_state=42
    )

    print("getting model")

    model_cat = CatBoostRegressor(
        iterations=n_iterations,
        learning_rate=0.01,
        l2_leaf_reg=leaf,
        depth=depth,
    )

    print("Fitting model")
    # Fit model
    model_cat.fit(X_train, y_train, verbose=True)

    print("Predicting")
    # Get predictions
    pred = model_cat.predict(X_test).clip(0, 100)

    print(np.round(pred))
    print(y_test.flatten())

    rmse = mean_squared_error(y_test.flatten(), np.round(pred), squared=False)

    plt.scatter(pred, y_test, s=1, c="red", alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])
    plt.title(f"RMSE={rmse}%")
    plt.xlabel("pred")
    plt.ylabel("gt")
    print(rmse)
    plt.savefig(
        os.path.join(
            folder_data,
            f"encoded_d_{depth}_leaf_{leaf}_iter_{n_iterations}_{model}_{sat}.png",
        )
    )
    plt.close()
