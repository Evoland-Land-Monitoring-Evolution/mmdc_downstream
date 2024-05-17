import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn import neural_network
from sklearn.model_selection import train_test_split
from torch import nn

from mmdc_downstream_lai.snap.lai_snap import denormalize, normalize

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

df["gt_denorm"] = denormalize(df["gt"], lai_min, lai_max)

train_col = [i for i in df.columns if i.startswith("s2_input")]
X = df[train_col]

# train_col = [i for i in df.columns if i.startswith("lat")]
# X = df[train_col]

y = df["gt_denorm"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Non-normalized data
mlp = neural_network.MLPRegressor(
    hidden_layer_sizes=(5),
    activation="tanh",
    solver="adam",
    alpha=0.01,
    batch_size=1 * 64 * 64,
    learning_rate="constant",
    learning_rate_init=0.0001,
    max_iter=2000,
    shuffle=True,
    tol=0.001,
    verbose=True,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
)

mlp.fit(normalize(X_train, input_min, input_max), normalize(y_train, lai_min, lai_max))

# mlp.fit(X_train,
#         normalize(y_train, lai_min, lai_max))

print("loss", mlp.loss_)

# Predict
# pred = denormalize(mlp.predict(normalize(X_test, input_min, input_max)),
#                    lai_min, lai_max)
pred = denormalize(mlp.predict(X_test), lai_min, lai_max)

print("max", pred.max())
print("min", pred.min())

loss_fn = nn.MSELoss()

plt.scatter(y_test, pred, s=0.1)
plt.plot(y_test, y_test)
plt.xlabel(
    "loss=" + str(np.round(loss_fn(torch.Tensor(pred), torch.Tensor(y_test)).item(), 4))
)
plt.savefig("mlp.png")
plt.show()
