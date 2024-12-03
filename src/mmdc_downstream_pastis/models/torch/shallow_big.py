# ruff: noqa
# flake8: noqa
"""
Shallow classifier for linear probing
"""
import torch
import torch.nn as nn
from einops import rearrange


class ShallowBig(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 20):
        """
        Shallow classifier for encoded alise
        """
        super().__init__()

        self.shallow_layer1 = nn.Conv2d(input_dim, 256, kernel_size=1)
        self.shallow_layer2 = nn.Conv2d(256, 128, kernel_size=1)
        self.shallow_layer3 = nn.Conv2d(128, num_classes, kernel_size=1)

        self.num_classes = num_classes
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        b, q, f, h, w = x.shape
        x = rearrange(x, "b q f h w -> b (q f) h w")
        x = self.shallow_layer3(
            self.relu(self.shallow_layer2(self.relu(self.shallow_layer1(x))))
        )
        return x
