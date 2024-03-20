import os

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

main_cols = ["index", "TCD", "x", "y", "x_round", "y_round"]

folder_data = "/work/scratch/data/kalinie/TCD/OpenEO_data/t32tnt/encoded/"

model = "2024-03-08_11-09-43"
# model = "test"

feat_nb = 6

months_median = True

csv_folder = os.path.join(folder_data, model)


satellites = ["s2", "s1_asc", "s1_desc"]

sat = "s2"

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

data_cols = [c for c in data.columns if ("mu" in c)]  # or 'logvar' in c)]
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
                data_cols_feat = [c for c in data_cols_month if f"{value}_f{f}" in c]
                median = np.nanmedian(data[data_cols_feat].values, axis=1)
                median_dict[f"{value}_f{f}_m{month}"] = median

    data = pd.concat([data[main_cols], pd.DataFrame.from_dict(median_dict)], axis=1)
    data_cols = median_dict.keys()

print("splitting data")

X_train, X_test, y_train, y_test = train_test_split(
    data[data_cols].values, data[gt_cols].values, test_size=0.33, random_state=42
)


print("getting model")
model = CatBoostRegressor(iterations=150, l2_leaf_reg=17, depth=8)
#
# loaded_model = CatBoostRegressor()
# name = os.path.join(folder_data, "catboost_model")
# loaded_model.load_model(fname=name, format='cbm')

print("Fitting model")
# Fit model
model.fit(X_train, y_train, verbose=True)


print("Predicting")
# Get predictions
pred = model.predict(X_test)

# print(pred)
# print(y_test.flatten())

# loss_fn = nn.MSELoss()
#
# loss = loss_fn(torch.Tensor(pred), torch.Tensor(y_test).reshape(-1)).item()
#

mse = mean_squared_error(y_test.flatten(), pred, squared=False)

print(mse)
model.save_model(os.path.join(folder_data, "catboost_model_depth_8"), format="cbm")
