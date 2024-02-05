import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from torch import nn

from mmdc_downstream.snap.lai_snap import denormalize

lai_min = 0.000319182538301
lai_max = 14.4675094548

input_min = np.array([
    0.0000, 0.0000, 0.0000, 0.00663797254225, 0.0139727270189, 0.0266901380821,
    0.0163880741923, 0.0000, 0.918595400582, 0.342022871159, -1.0000
])

input_max = np.array([
    0.253061520472, 0.290393577911, 0.305398915249, 0.608900395798,
    0.753827384323, 0.782011770669, 0.493761397883, 0.49302598446, 1.0000,
    0.936206429175, 1.0000
])

path = "/work/scratch/data/kalinie/MMDC/jobs/data_values_exp.csv"

df = pd.read_csv(path)

y = df["gt"]  # centered

df = df.rename(
    columns={
        's1_0': 's1_asc_0',
        's1_1': 's1_asc_1',
        's1_2': 's1_asc_2',
        's1_3': 's1_desc_3',
        's1_4': 's1_desc_4',
        's1_5': 's1_desc_5',
        's1_mask_0': 's1_mask_asc',
        's1_mask_1': 's1_mask_desc'
    })

# train_data = ["s2_input", "exp_lat", "s1_lat", "s2_lat", 's1_asc', "s1_desc", "s1_mask"]
train_data = ['s1_asc', "s1_desc", "s1_mask"]

train_col_all = [
    i for i in df.columns for data_type in train_data
    if i.startswith(data_type)
]

print(train_col_all)

X_all = df[train_col_all]

X_train_all, X_test_all, y_train, y_test = train_test_split(X_all,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=42)

results = {}
y_test_denorm = {}

for data_type in train_data:
    if data_type != "s1_mask":
        train_col = [i for i in df.columns if i.startswith(data_type)]
        print(train_col)
        if train_col[0].startswith('s1_asc'):
            mask_train = X_train_all['s1_mask_asc'].values.astype(bool)
            mask_test = X_test_all['s1_mask_asc'].values.astype(bool)
            X_train_all, X_test_all, y_train, y_test = X_train_all[
                ~mask_train], X_test_all[~mask_test], y_train[
                    ~mask_train], y_test[~mask_test]

        if train_col[0].startswith('s1_desc'):
            mask_train = X_train_all['s1_mask_desc'].values.astype(bool)
            mask_test = X_test_all['s1_mask_desc'].values.astype(bool)
            X_train_all, X_test_all, y_train, y_test = X_train_all[
                ~mask_train], X_test_all[~mask_test], y_train[
                    ~mask_train], y_test[~mask_test]

        X_train = X_train_all[train_col]
        X_test = X_test_all[train_col]

        # Normalized data
        tree_15 = DecisionTreeRegressor(max_depth=15)
        tree_15.fit(X_train, y_train)

        # Predict
        pred = tree_15.predict(X_test)
        results[data_type] = pred
        y_test_denorm[data_type] = denormalize(y_test.values, lai_min, lai_max)

# y_test_denorm = denormalize(y_test, lai_min, lai_max)

nrows = 1
ncols = 2
fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    sharex=False,
    sharey=False,
    figsize=(20, nrows * 3.5 + 2),
)

loss_fn = nn.MSELoss()  # mean square error

for i in range(nrows):
    for j in range(ncols):
        key = list(results.keys())[i * ncols + j]
        pred = denormalize(results[key], lai_min, lai_max)
        axis = axes[j]
        axis.set_title(key)
        axis.scatter(y_test_denorm[key], pred, s=0.1)
        axis.set_xlabel("loss=" + str(
            np.round(
                loss_fn(torch.Tensor(pred), torch.Tensor(
                    y_test_denorm[key])).item(), 3)))
        axis.plot(y_test_denorm[key], y_test_denorm[key])
plt.savefig("random_forest.png")
plt.show()
