from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class InputInterp:
    sits: Tensor
    doy: Tensor
    padd_index: torch.Tensor | None
    mask_valid: Tensor | None = None
    true_doy: Tensor | None = None


@dataclass
class InterporlItem:
    sits_in: InputInterp
    sits_pred: InputInterp


@dataclass
class ModuleInput:
    sits: Tensor
    input_doy: Tensor
    padd_index: Tensor | None
    cld_mask: Tensor | None = None
    true_doy: Tensor | None = None

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.sits = self.sits.pin_memory()
        self.input_doy = self.input_doy.pin_memory()
        self.padd_index = self.padd_index.pin_memory()
        if self.cld_mask is not None:
            self.cld_mask = self.cld_mask.pin_memory()
        if self.true_doy is not None:
            self.true_doy.pin_memory()
        return self
