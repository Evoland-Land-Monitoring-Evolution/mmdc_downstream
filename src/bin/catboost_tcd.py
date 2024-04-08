import os

import matplotlib.pyplot as plt
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

SEL_BANDS = ["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"]
months_folders = ["3", "4", "5", "6", "7", "8", "9", "10"]

folder_data = "/work/scratch/data/kalinie/TCD/OpenEO_data/t32tnt/"
csv_name = "gt_tcd_t32tnt.csv"

data = pd.read_csv(os.path.join(folder_data, csv_name))

data_cols = [b + "_" + m for m in months_folders for b in SEL_BANDS]
gt_cols = ["TCD"]


print("splitting data")

X_train, X_test, y_train, y_test = train_test_split(
    data[data_cols].values, data[gt_cols].values, test_size=0.33, random_state=42
)


print("getting model")
model = CatBoostRegressor(iterations=150, l2_leaf_reg=17, depth=8)

print("Fitting model")
# Fit model
model.fit(X_train, y_train, verbose=True)


print("Predicting")
# Get predictions
pred = model.predict(X_test)

print(pred)
print(y_test.flatten())

plt.scatter(pred, y_test, s=1, c="red", alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])
plt.xlabel("pred")
plt.ylabel("gt")
plt.savefig("our_plot_s2.png")


rmse = mean_squared_error(y_test.flatten(), pred, squared=False)

print(rmse)
