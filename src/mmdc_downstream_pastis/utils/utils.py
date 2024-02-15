#!/usr/bin/env python3
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

"""Project utils"""

from dataclasses import dataclass, fields

import torch
from typing_extensions import Self

from mmdc_downstream_pastis.datamodule.datatypes import PastisBatch


@dataclass
class MMDCPartialBatch:
    """Partial MMDC batch that is fed to MMDC model"""

    img: torch.Tensor | None
    angles: torch.Tensor | None
    mask: torch.Tensor | None
    dem: torch.Tensor | None
    meteo: torch.Tensor | None
    type: str = "S2"

    @staticmethod
    def fill_from(batch_sat: PastisBatch, satellite: str) -> Self:
        """Fill from PastisBatch"""
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
                setattr(self, field.name, getattr(self, field.name).to_device(device))
        return self
