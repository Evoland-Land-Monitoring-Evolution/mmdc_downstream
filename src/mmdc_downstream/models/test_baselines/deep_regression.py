import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from mmdc_downstream.snap.lai_snap import denormalize, normalize


class MLP(nn.Module):
    """Multilayer Perceptron for regression."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(11, 5), nn.Tanh(), nn.Linear(5, 1))

    def forward(self, x):
        """Forward pass"""
        return self.layers(x)


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

# train_col = [i for i in df.columns if i.startswith("s2_input")]
# X = df[train_col]

train_col = [i for i in df.columns if i.startswith("s2_input")]
X = df[train_col]

y = df["gt"]
X = normalize(X, input_min, input_max)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train-test split of the dataset
X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)

model = MLP()
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_dl = DataLoader(train_dataset, batch_size=64 * 64, shuffle=True, num_workers=2)
test_dl = DataLoader(
    test_dataset, batch_size=100 * 64 * 64, shuffle=False, num_workers=2
)

# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training parameters
n_epochs = 200  # number of epochs to run

# Hold the best model
best_mse = np.inf  # init to infinity
best_weights = None
history = []

loss_epochs = []

path = "/work/scratch/data/kalinie/MMDC/results/simple_deep"
folder = f"bs_{train_dl.batch_size}_lr_{optimizer.param_groups[0]['lr']}_hidden_5_huber"
output_folder = os.path.join(path, folder)
print(output_folder)
Path(output_folder).mkdir(exist_ok=True, parents=True)

# Run the training loop
for epoch in range(n_epochs):  # 5 epochs at maximum
    if epoch == 10:
        optimizer.param_groups[0]["lr"] = 0.0001

    # Print epoch
    print(f"Starting epoch {epoch + 1}")

    # Set current loss value
    current_loss_train = 0.0
    current_loss_test = 0.0

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
    print(
        "Train Loss after epoch %5d: %.3f"
        % (
            epoch + 1,
            current_loss_train / int(len(train_dataset) / train_dl.batch_size),
        )
    )

    preds = []
    for i, data in enumerate(tqdm.tqdm(test_dl)):
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
        current_loss_test += loss.item()
    print(
        "Test Loss after epoch %5d: %.3f"
        % (
            epoch + 1,
            current_loss_test / int(len(test_dataset) / test_dl.batch_size),
        )
    )
    preds = np.concatenate(preds, 0)

    pred = denormalize(preds, lai_min, lai_max)
    y_test_denorm = denormalize(y_test, lai_min, lai_max).flatten().to(torch.float32)

    print("max", pred.max())
    print("min", pred.min())

    plt.close()
    plt.scatter(y_test_denorm, pred, s=0.1)
    plt.plot(y_test_denorm, y_test_denorm)
    plt.xlabel(
        "loss=" + str(np.round(loss_fn(torch.Tensor(pred), y_test_denorm).item(), 4))
    )
    plt.savefig(output_folder + f"/mlp_{epoch}.png")

    # Process is complete.
print("Training process has finished.")
