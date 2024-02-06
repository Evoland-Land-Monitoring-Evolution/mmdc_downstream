import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from torch import nn

lai_min = 0.000319182538301
lai_max = 14.4675094548

input_min = np.array(
    [
        0.0000,
        0.0000,
        0.0000,
        0.00663797254225,
        0.0139727270189,
        0.0266901380821,
        0.0163880741923,
        0.0000,
        0.918595400582,
        0.342022871159,
        -1.0000,
    ]
)

input_max = np.array(
    [
        0.253061520472,
        0.290393577911,
        0.305398915249,
        0.608900395798,
        0.753827384323,
        0.782011770669,
        0.493761397883,
        0.49302598446,
        1.0000,
        0.936206429175,
        1.0000,
    ]
)

path = "/work/scratch/data/kalinie/MMDC/jobs/data_values.csv"

df = pd.read_csv(path)

# df["gt_denorm"] = denormalize(df["gt"], lai_min, lai_max)

# train_col = [i for i in df.columns if i.startswith("s2_input")]

train_col = [i for i in df.columns if i.startswith("lat")]
# X = df[train_col]

y = df["gt"]
X = df[train_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Non-normalized data
regr_1 = DecisionTreeRegressor(max_depth=5)
# regr_2 = DecisionTreeRegressor(max_depth=10)
regr_3 = DecisionTreeRegressor(max_depth=15)

regr_1.fit(X_train, y_train)
# regr_2.fit(X_train, y_train)
regr_3.fit(X_train, y_train)

# Predict
y_1 = regr_1.predict(X_test)
# y_2 = regr_2.predict(X_test)
y_3 = regr_3.predict(X_test)
# print("max", y_1.max(), y_2.max(), y_3.max())
# print("min", y_1.min(), y_2.min(), y_3.min())

# results = {"tree_5": y_1, "tree_10": y_2, "tree_15": y_3}
results = {"tree_5": y_1, "tree_15": y_3}

nrows = 1
ncols = len(results)
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
        key = list(results.keys())[j]
        pred = results[key]
        axis = axes[j]
        axis.set_title(key)
        axis.scatter(y_test, pred, s=0.1)
        axis.set_xlabel(
            "loss="
            + str(np.round(loss_fn(torch.Tensor(pred), torch.Tensor(y_test)).item(), 4))
        )
        axis.plot(y_test, y_test)
plt.savefig("random_forest.png")
plt.show()

# # Normalized data
# regr_1_n = DecisionTreeRegressor(max_depth=5)
# regr_2_n = DecisionTreeRegressor(max_depth=10)
# reg_bay_n = linear_model.BayesianRidge()
# regr_1_n.fit(normalize(X_train, input_min, input_max),
#              normalize(y_train, lai_min, lai_max))
# regr_2_n.fit(normalize(X_train, input_min, input_max),
#              normalize(y_train, lai_min, lai_max))
# reg_bay.fit(normalize(X_train, input_min, input_max),
#             normalize(y_train, lai_min, lai_max))
#
# # Predict
# y_1_n = denormalize(regr_1_n.predict(normalize(X_test, input_min, input_max)),
#                     lai_min, lai_max)
# y_2_n = denormalize(regr_2_n.predict(normalize(X_test, input_min, input_max)),
#                     lai_min, lai_max)
# print("max", y_1_n.max(), y_2_n.max())
# print("min", y_1_n.min(), y_2_n.min())
