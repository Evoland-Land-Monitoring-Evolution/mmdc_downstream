# ruff: noqa
# flake8: noqa
"""
Shallow classifier for linear probing
"""
import torch
import torch.nn as nn
from einops import rearrange


class Shallow(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 20):
        """
        Shallow classifier for encoded alise
        """
        super().__init__()

        self.shallow_layer = nn.Linear(input_dim, num_classes)

        self.num_classes = num_classes
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        b, q, f, h, w = x.shape
        x = rearrange(x, "b q f h w -> (b h w) (q f)")
        # x = self.shallow_layer2(self.relu(self.shallow_layer1(x)))
        x = self.shallow_layer(x)
        return rearrange(x, "(b h w) o -> b o h w", b=b, h=h, w=w)
