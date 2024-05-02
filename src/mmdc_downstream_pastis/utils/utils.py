#!/usr/bin/env python3
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

"""Project utils"""

import dataclasses
from copy import deepcopy
from dataclasses import dataclass, fields

import torch
from typing_extensions import Self

from mmdc_downstream_pastis.datamodule.datatypes import PastisBatch


@dataclass
class MMDCPartialBatch:
    """Partial MMDC batch that is fed to MMDC model"""

    img: torch.Tensor
    angles: torch.Tensor
    mask: torch.Tensor
    dem: torch.Tensor
    meteo: torch.Tensor
    type: str = "S2"

    @staticmethod
    def fill_from(batch_sat: PastisBatch, satellite: str) -> Self:
        """Fill from PastisBatch"""
        assert batch_sat is not None
        return MMDCPartialBatch(
            img=batch_sat.sits.data.img,
            angles=batch_sat.sits.data.angles,
            mask=batch_sat.sits.data.mask,
            meteo=batch_sat.sits.meteo,
            dem=batch_sat.sits.dem,
            type="S2" if satellite == "S2" else "S1",
        )

    @staticmethod
    def create_empty_s1_with_shape(
        batch_size: int,
        times: int,
        height: int,
        width: int,
        nb_channels: int = 6,
        nb_angles: int = 2,
        nb_meteo: int = 48,
        nb_mask: int = 2,
    ) -> Self:
        """
        Create zero S1 instance with defined shape.
        Mask values are 1, because it is nodata
        """
        return MMDCPartialBatch(
            img=torch.zeros(batch_size, times, nb_channels, height, width),
            angles=torch.zeros(batch_size, times, nb_angles, height, width),
            mask=torch.ones(batch_size, times, nb_mask, height, width).to(torch.int8),
            meteo=torch.zeros(batch_size, times, nb_meteo, height, width),
            dem=torch.zeros(batch_size, 4, height, width),
            type="S1",
        )

    def to_device(self, device: str | torch.device) -> Self:
        """Send data to device"""
        for field in fields(self):
            if field.type is torch.Tensor:
                setattr(self, field.name, getattr(self, field.name).to(device))
        return self

    def __getitem__(self, item: int | list[int]) -> Self:
        """Get the same slice (batch elements) for each field (tensor)
        of the data class"""
        new = deepcopy(self)
        for key, value in dataclasses.asdict(self).items():
            if key != "type" and key != "dem":
                setattr(new, key, value[item])
            else:
                setattr(new, key, value)
        return new
