import os

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

print(pred)
print(y_test.flatten())

# loss_fn = nn.MSELoss()
#
# loss = loss_fn(torch.Tensor(pred), torch.Tensor(y_test).reshape(-1)).item()
#

mse = mean_squared_error(y_test.flatten(), pred, squared=False)

print(mse)
model.save_model(os.path.join(folder_data, "catboost_model_depth_8"), format="cbm")
